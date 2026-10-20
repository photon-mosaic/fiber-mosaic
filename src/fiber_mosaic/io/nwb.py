"""Write fiber photometry recordings to NWB through NeuroConv.

SpikeInterface cannot write NWB -- ``BaseRecording.save(format="nwb")``
raises ``NotImplementedError`` -- so export goes through NeuroConv, whose
``BaseFiberPhotometryInterface`` emits ``ndx-fiber-photometry`` types.
NeuroConv's own fiber photometry interfaces all read files, so this module
supplies the missing seam: an interface backed by an in-memory recording.

One interface writes one ``FiberPhotometryResponseSeries``. Following the
extension's recommended layout, that is one series per color, with ``data``
shaped ``(n_samples, n_fibers)`` and a ``fiber_photometry_table_region``
naming one table row per fiber, in fiber order.

A series carries one 1-D time base, which is exactly what a recording holds,
so its timestamps are written as they are. One limit of the format is
enforced here rather than discovered at write time: a series covers one
contiguous block, so one segment is written per interface.

Write the finished file with :func:`write_nwbfile`, which applies
NeuroConv's chunking and compression.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from spikeinterface.core import BaseRecording

from fiber_mosaic.core.base import FiberPhotometryRecordingGroup

if TYPE_CHECKING:
    from pynwb import NWBFile

_INSTALL_HINT = (
    "Writing NWB needs neuroconv and the fiber photometry extensions. "
    "Install them with: uv pip install 'fiber-mosaic[nwb]'"
)

try:
    from neuroconv.datainterfaces.fiber_photometry import (
        basefiberphotometryinterface as _base_module,
    )
    from neuroconv.tools.nwb_helpers import configure_and_write_nwbfile
except ImportError as error:  # pragma: no cover - import guard
    raise ImportError(_INSTALL_HINT) from error

BaseFiberPhotometryInterface = _base_module.BaseFiberPhotometryInterface


def _deep_merge(base: dict, other: Mapping) -> dict:
    """Recursively merge ``other`` into ``base`` and return ``base``.

    NeuroConv's own ``DeepDict.deep_update`` recurses only into values that
    are themselves ``DeepDict``, and its metadata mixes plain dicts in, so
    merging two templates through it fails. This works on any mapping.
    """
    for key, value in other.items():
        if isinstance(base.get(key), dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _camel(name: str) -> str:
    """Return ``name`` as CamelCase, so ``green_corr`` reads GreenCorr."""
    return "".join(part.capitalize() for part in str(name).split("_") if part)


def _resolve_color(recording: BaseRecording, color: str | None) -> str:
    """Return the explicit color, or read it off the recording."""
    if color is not None:
        return str(color)
    found = recording.get_annotation("color")
    if found is None:
        raise ValueError(
            "No color found on the recording. Pass color= explicitly, or "
            "use a BaseFiberPhotometryExtractor, which annotates it."
        )
    return str(found)


def _select_fiber(recording: BaseRecording, fiber_id: Any) -> BaseRecording:
    """Return ``recording`` restricted to ``fiber_id``, which must exist."""
    available = [str(f) for f in recording.get_channel_ids()]
    if str(fiber_id) not in available:
        raise ValueError(
            f"Fiber {fiber_id!r} is not in this recording. Available: "
            f"{available}."
        )
    return recording.select_channels([fiber_id])


def _resolve_segment_index(
    recording: BaseRecording, segment_index: int | None
) -> int:
    """Return the segment to write, refusing to guess for multi-segment."""
    num_segments = recording.get_num_segments()
    if segment_index is not None:
        return int(segment_index)
    if num_segments > 1:
        raise ValueError(
            f"Recording has {num_segments} segments and one response "
            "series covers one contiguous block. Pass segment_index= to "
            "choose one, build one interface per segment, or concatenate "
            "the segments before exporting."
        )
    return 0


class FiberMosaicPhotometryInterface(BaseFiberPhotometryInterface):
    """NeuroConv interface backed by an in-memory fiber-mosaic recording.

    Parameters
    ----------
    recording : BaseRecording
        One color's recording, channels being fibers. Traces are read via
        ``get_traces``, so any spikeinterface preprocessor output works. A
        recording holding a single fiber is named after it -- one trace, one
        table row, 1-D ``data`` -- so several can share one file.
    color : str or None, default: None
        Color label. Read from the recording's ``color`` annotation when
        None.
    segment_index : int or None, default: None
        Segment to write. Required when the recording has several.
    metadata_key : str or None, default: None
        Key under ``metadata["FiberPhotometry"]``. Defaults to
        ``f"fiber_photometry_{color}"``, with ``_{fiber_id}`` appended when
        the recording holds a single fiber.
    verbose : bool, default: False
        Whether to print status messages.
    """

    display_name = "Fiber Mosaic"
    keywords = ("fiber photometry",)

    def __init__(
        self,
        recording: BaseRecording,
        *,
        color: str | None = None,
        segment_index: int | None = None,
        metadata_key: str | None = None,
        verbose: bool = False,
    ) -> None:
        """Bind one color's recording to a single response series."""
        self._recording = recording
        self._color = _resolve_color(recording, color)
        fiber_ids = recording.get_channel_ids().tolist()
        self._fiber_id = fiber_ids[0] if len(fiber_ids) == 1 else None
        self._segment_index = _resolve_segment_index(recording, segment_index)
        if metadata_key is None:
            suffix = "" if self._fiber_id is None else f"_{self._fiber_id}"
            metadata_key = f"fiber_photometry_{self._color}{suffix}"
        # the base registers timestamps as a lazy callback, so everything it
        # will later read must be set before this call.
        super().__init__(
            stream_names=self._color,
            metadata_key=metadata_key,
            verbose=verbose,
        )

    @property
    def color(self) -> str:
        """Color label of the recording this interface writes."""
        return self._color

    @property
    def fiber_ids(self) -> np.ndarray:
        """Fiber IDs, in the column order of the written series."""
        return np.asarray(self._recording.get_channel_ids())

    @property
    def fiber_id(self) -> Any:
        """The single fiber this interface writes.

        Raises
        ------
        ValueError
            When the interface covers several fibers, since then no one
            fiber identifies the series. Use :attr:`fiber_ids` instead.
        """
        if self._fiber_id is None:
            raise ValueError(
                f"This interface writes {len(self.fiber_ids)} fibers as one "
                "series, so it has no single fiber_id. Select one fiber "
                "first, e.g. with add_photometry_trace_to_nwb, or read "
                "fiber_ids."
            )
        return self._fiber_id

    def _check_stream(self, stream_name: str) -> None:
        """Raise when asked for a stream this interface does not hold."""
        if stream_name != self._color:
            raise KeyError(
                f"Unknown stream {stream_name!r}; this interface holds "
                f"only {self._color!r}."
            )

    def _get_stream_data(self, *, stream_name: str) -> np.ndarray:
        """Return traces for this color, shaped ``(n_samples, n_fibers)``."""
        self._check_stream(stream_name)
        return self._recording.get_traces(segment_index=self._segment_index)

    def _get_stream_timestamps(self, *, stream_name: str) -> np.ndarray:
        """Return the recording's 1-D time vector for this segment."""
        self._check_stream(stream_name)
        return self._recording.get_times(segment_index=self._segment_index)

    def _get_number_of_traces(self) -> int:
        """Return the trace count without reading the traces."""
        return len(self.fiber_ids)

    def get_metadata(self) -> Any:
        """Return base metadata with a name identifying what is written."""
        metadata = super().get_metadata()
        series = metadata["FiberPhotometry"][self.metadata_key]
        suffix = "" if self._fiber_id is None else _camel(self._fiber_id)
        series["name"] = (
            f"FiberPhotometryResponseSeries{_camel(self._color)}{suffix}"
        )
        if self._fiber_id is None:
            series["description"] = (
                f"{self._color} fiber photometry, one column per fiber, in "
                f"fiber order: {[str(f) for f in self.fiber_ids]}."
            )
        else:
            series["description"] = (
                f"{self._color} fiber photometry for fiber {self._fiber_id}."
            )
        return metadata

    def get_metadata_template(self) -> Any:
        """Return the provenance template, keyed by color and fiber ID.

        The inherited template keys rows ``trace_0..trace_n``, which would
        collide when several colors share one ``FiberPhotometryTable``.
        Rows become ``{color}_{fiber_id}`` and per-color hardware is
        prefixed, while the optical fibers stay un-prefixed: the same
        physical fiber carries every color, so those entries are meant to
        merge across interfaces.
        """
        template = super().get_metadata_template()
        fiber_ids = [str(fiber_id) for fiber_id in self.fiber_ids]

        renames = {
            f"trace_{index}": f"{self._color}_{fiber_id}"
            for index, fiber_id in enumerate(fiber_ids)
        }
        renames.update(
            {
                f"optical_fiber_{index}": f"optical_fiber_{fiber_id}"
                for index, fiber_id in enumerate(fiber_ids)
            }
        )
        for key in ("excitation_source", "photodetector", "indicator"):
            renames[key] = f"{self._color}_{key}"
        for key in ("dichroic_mirror", "excitation_filter", "emission_filter"):
            renames[key] = f"{self._color}_{key}"
        for key in ("excitation_source_model", "photodetector_model"):
            renames[key] = f"{self._color}_{key}"

        return _rekey_template(template, renames, series_key=self.metadata_key)


def _rekey(mapping: dict, renames: dict[str, str]) -> dict:
    """Return ``mapping`` with its top-level keys renamed.

    Rebuilt through ``type(mapping)`` so a NeuroConv ``DeepDict`` stays one:
    its ``deep_update`` recurses only into values that are themselves
    ``DeepDict``, so a plain dict here would break template merging.
    """
    renamed = {renames.get(key, key): value for key, value in mapping.items()}
    return type(mapping)(renamed)


def _rekey_template(
    template: Any, renames: dict[str, str], series_key: str
) -> Any:
    """Apply ``renames`` to every place a template key is referenced."""
    template["Devices"] = _rekey(template["Devices"], renames)
    template["DeviceModels"] = _rekey(template["DeviceModels"], renames)

    photometry = template["FiberPhotometry"]
    photometry["FiberPhotometryIndicators"] = _rekey(
        photometry["FiberPhotometryIndicators"], renames
    )
    table = photometry["FiberPhotometryTable"]
    table["rows"] = _rekey(table["rows"], renames)

    # cross-references are stored as key strings, so they rename too.
    for row in table["rows"].values():
        for field, value in list(row.items()):
            if field.endswith("_metadata_key") and value in renames:
                row[field] = renames[value]

    for device in template["Devices"].values():
        model_key = device.get("device_model_metadata_key")
        if model_key in renames:
            device["device_model_metadata_key"] = renames[model_key]

    photometry[series_key]["fiber_photometry_table_region"] = [
        renames.get(key, key)
        for key in photometry[series_key]["fiber_photometry_table_region"]
    ]
    return template


def _bare_metadata(
    interfaces: dict[str, FiberMosaicPhotometryInterface],
) -> dict:
    """Merge the interfaces' default metadata: series names, no table."""
    merged: dict = {}
    for interface in interfaces.values():
        _deep_merge(merged, interface.get_metadata())
    return merged


def _add_interfaces(
    interfaces: dict[str, FiberMosaicPhotometryInterface],
    nwbfile: NWBFile,
    metadata: dict | None,
    parent_container: str,
    conversion_options: dict,
) -> None:
    """Write every interface's series into ``nwbfile``.

    The shared containers (devices, indicators, ``FiberPhotometryTable``)
    are built by whichever interface runs first and reused by the rest, so
    all of them are handed the same ``metadata``.
    """
    if metadata is None:
        metadata = _bare_metadata(interfaces)
    for interface in interfaces.values():
        interface.add_to_nwbfile(
            nwbfile,
            metadata=metadata,
            parent_container=parent_container,
            **conversion_options,
        )


def add_photometry_recording_to_nwb(
    signal: BaseRecording,
    nwbfile: NWBFile,
    *,
    metadata: dict | None = None,
    color: str | None = None,
    segment_index: int | None = None,
    parent_container: Literal[
        "acquisition", "processing/ophys"
    ] = "processing/ophys",
    **conversion_options: Any,
) -> FiberMosaicPhotometryInterface:
    """Add one recording to an NWBFile as one response series.

    One recording is one color, so it becomes exactly one series with
    ``data`` shaped ``(n_samples, n_fibers)`` -- a series carries one shared
    clock, which is what lets each color keep its own time base. A
    recording holding a single fiber, as the Doric, NPM and TDT readers
    return, is named after that fiber, so several can share one file.

    ``metadata`` is optional. Without it a bare series is written -- no
    devices, no ``FiberPhotometryTable`` -- which is a valid NWB file the
    data round-trips through. Pass :func:`metadata_template_for_recording`
    output to add the provenance chain.

    Parameters
    ----------
    signal : BaseRecording
        One color's recording, channels being fibers. Only ``get_traces``,
        ``get_times`` and the channel accessors are used, so spikeinterface
        preprocessor output is exportable too.
    nwbfile : NWBFile
        In-memory file to add the series to.
    metadata : dict or None, default: None
        Metadata for this series. A bare series when None.
    color : str or None, default: None
        Color label. Read off the recording when None.
    segment_index : int or None, default: None
        Segment to write. Required when the recording has several.
    parent_container : {"acquisition", "processing/ophys"}
        Where the series lands. Processed data by default; use
        ``"acquisition"`` for raw data off the rig.
    **conversion_options
        Forwarded to the interface's ``add_to_nwbfile``.

    Returns
    -------
    FiberMosaicPhotometryInterface
        The interface used, for inspection or reuse.
    """
    interface = FiberMosaicPhotometryInterface(
        signal, color=color, segment_index=segment_index
    )
    _add_interfaces(
        {interface.color: interface},
        nwbfile,
        metadata,
        parent_container,
        conversion_options,
    )
    return interface


def add_photometry_recording_group_to_nwb(
    group: FiberPhotometryRecordingGroup,
    nwbfile: NWBFile,
    *,
    metadata: dict | None = None,
    colors: list[str] | None = None,
    segment_index: int | None = None,
    parent_container: Literal[
        "acquisition", "processing/ophys"
    ] = "processing/ophys",
    **conversion_options: Any,
) -> dict[str, FiberMosaicPhotometryInterface]:
    """Add every color of a group to an NWBFile, one series per color.

    All colors share one ``FiberPhotometryTable``, so they are written
    together rather than by repeated :func:`add_photometry_trace_to_nwb`
    calls: the first interface builds the shared containers and the rest
    reuse them, which only works if they see the same ``metadata``.

    Parameters
    ----------
    group : FiberPhotometryRecordingGroup
        Group whose colors become response series.
    nwbfile : NWBFile
        In-memory file to add the series to.
    metadata : dict or None, default: None
        Merged metadata covering every color. Bare series when None.
    colors : list of str or None, default: None
        Subset of colors to write. All of them when None.
    segment_index : int or None, default: None
        Segment to write, applied to every color.
    parent_container : {"acquisition", "processing/ophys"}
        Where the series land. Processed data by default; use
        ``"acquisition"`` for raw data off the rig.
    **conversion_options
        Forwarded to each interface's ``add_to_nwbfile``.

    Returns
    -------
    dict
        Mapping from color to the interface used.
    """
    interfaces = _interfaces_for_group(group, colors, segment_index)
    _add_interfaces(
        interfaces,
        nwbfile,
        metadata,
        parent_container,
        conversion_options,
    )
    return interfaces


def _interfaces_for_group(
    group: FiberPhotometryRecordingGroup,
    colors: list[str] | None,
    segment_index: int | None,
) -> dict[str, FiberMosaicPhotometryInterface]:
    """Build one interface per selected color of ``group``."""
    selected = group.colors if colors is None else colors
    return {
        color: FiberMosaicPhotometryInterface(
            group[color], color=color, segment_index=segment_index
        )
        for color in selected
    }


def metadata_template_for_recording(
    signal: BaseRecording,
    *,
    color: str | None = None,
    segment_index: int | None = None,
) -> dict:
    """Return the metadata skeleton for one recording, all fibers.

    Fill the ``None`` blanks and pass the result to
    :func:`add_photometry_recording_to_nwb`. A blank left in place is an
    error at write time rather than a value.
    """
    interface = FiberMosaicPhotometryInterface(
        signal, color=color, segment_index=segment_index
    )
    return interface.get_metadata_template()


def metadata_template_for_group(
    group: FiberPhotometryRecordingGroup,
    *,
    colors: list[str] | None = None,
    segment_index: int | None = None,
) -> dict:
    """Return one metadata skeleton covering every color of ``group``.

    Each color contributes its own response series and table rows; shared
    entries (the optical fibers and their model) collapse onto one.
    """
    interfaces = _interfaces_for_group(group, colors, segment_index)
    templates = [
        interface.get_metadata_template() for interface in interfaces.values()
    ]
    merged = templates[0]
    for template in templates[1:]:
        _deep_merge(merged, template)
    return merged


def add_photometry_trace_to_nwb(
    signal: BaseRecording,
    nwbfile: NWBFile,
    fiber_id: Any,
    *,
    metadata: dict | None = None,
    color: str | None = None,
    segment_index: int | None = None,
    parent_container: Literal[
        "acquisition", "processing/ophys"
    ] = "processing/ophys",
    **conversion_options: Any,
) -> FiberMosaicPhotometryInterface:
    """Add one fiber's trace to an NWBFile as one response series.

    A trace is one fiber of one color, which is exactly one
    ``FiberPhotometryTable`` row, and is written with 1-D ``data``. Use
    this to write a single fiber, such as one processed trace; for a whole
    color, :func:`add_photometry_recording_to_nwb` writes one series with a
    column per fiber, which is the layout ``ndx-fiber-photometry``
    recommends.

    Parameters
    ----------
    signal : BaseRecording
        One color's recording, channels being fibers.
    nwbfile : NWBFile
        In-memory file to add the series to.
    fiber_id : object
        Which fiber to write. Must be one of the recording's fiber IDs.
    metadata : dict or None, default: None
        Metadata for this series. A bare series when None.
    color : str or None, default: None
        Color label. Read off the recording when None.
    segment_index : int or None, default: None
        Segment to write. Required when the recording has several.
    parent_container : {"acquisition", "processing/ophys"}
        Where the series lands. Processed data by default; use
        ``"acquisition"`` for raw data off the rig.
    **conversion_options
        Forwarded to the interface's ``add_to_nwbfile``.

    Returns
    -------
    FiberMosaicPhotometryInterface
        The interface used, whose ``fiber_id`` is ``fiber_id``.
    """
    interface = FiberMosaicPhotometryInterface(
        _select_fiber(signal, fiber_id),
        color=color,
        segment_index=segment_index,
    )
    _add_interfaces(
        {interface.metadata_key: interface},
        nwbfile,
        metadata,
        parent_container,
        conversion_options,
    )
    return interface


def metadata_template_for_trace(
    signal: BaseRecording,
    fiber_id: Any,
    *,
    color: str | None = None,
    segment_index: int | None = None,
) -> dict:
    """Return the metadata skeleton for one fiber's trace.

    One table row, since a trace is one fiber of one color. Fill the
    ``None`` blanks and pass the result to
    :func:`add_photometry_trace_to_nwb`.
    """
    interface = FiberMosaicPhotometryInterface(
        _select_fiber(signal, fiber_id),
        color=color,
        segment_index=segment_index,
    )
    return interface.get_metadata_template()


def write_nwbfile(
    nwbfile: NWBFile,
    nwbfile_path: str | Path,
    *,
    backend: Literal["hdf5", "zarr"] = "hdf5",
) -> None:
    """Write an NWBFile to disk with chunking and compression.

    Uses NeuroConv's default backend configuration, which chooses chunk
    shapes and compression for each dataset from its data, so the file
    streams well from DANDI. Writing with pynwb's ``NWBHDF5IO`` directly
    stores every dataset unchunked and uncompressed.

    Parameters
    ----------
    nwbfile : NWBFile
        In-memory file, e.g. filled by :func:`add_photometry_recording_to_nwb`.
    nwbfile_path : str or Path
        Where to write the file.
    backend : {"hdf5", "zarr"}, default: "hdf5"
        Storage backend.
    """
    configure_and_write_nwbfile(
        nwbfile=nwbfile, nwbfile_path=Path(nwbfile_path), backend=backend
    )


def dump_recording(recording: BaseRecording, file_path: str | Path) -> Path:
    """Write ``recording`` to a SpikeInterface JSON file.

    The file holds no samples, only how to rebuild the recording from its
    source files, which must stay where they are.

    Parameters
    ----------
    recording : BaseRecording
        A recording read from files on disk, with any processing steps.
    file_path : str or Path
        Where to write the JSON file.

    Returns
    -------
    Path
        ``file_path``, ready for ``FiberMosaicPhotometryInterface``.
    """
    recording.dump_to_json(file_path)
    return Path(file_path)
