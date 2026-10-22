"""
Core classes for fiber photometry data handling.

This module provides the base classes that form the foundation of fiber-mosaic:

- FiberPhotometryMixin: The fiber-native API, mixable into any SI recording
- BaseFiberPhotometryExtractor: A single named recording (wraps SI
  BaseRecording)
- FiberPhotometryRecordingGroup: Named recordings sharing fibers
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
from spikeinterface.core import BaseRecording
from spikeinterface.core.numpyextractors import NumpyRecordingSegment


class FiberPhotometryMixin:
    """
    Fiber-native API for fiber photometry recordings.

    Provides fiber-photometry vocabulary on top of a SpikeInterface
    recording:

    - Channels are called "fibers"
    - `get_fluorescence()` wraps `get_traces()` with fiber-native naming

    This is deliberately separate from BaseFiberPhotometryExtractor, which
    adds the source-side concerns (construction, stream discovery, segment
    loading). Keeping the API in a mixin lets recording-to-recording classes
    -- notably SpikeInterface preprocessors, which cannot inherit from an
    extractor -- expose the same fiber-native surface.

    Notes
    -----
    Must be mixed into a SpikeInterface ``BaseRecording``, on whose
    ``get_channel_ids``, ``get_num_channels``, ``get_num_segments``,
    ``get_traces``, ``get_sampling_frequency``, ``get_dtype`` and
    ``get_annotation`` it relies. It defines no ``__init__``, so it never
    interferes with the host class's construction.

    The recording's label is SpikeInterface's own ``name``, stored as the
    ``"name"`` annotation rather than an attribute, because ``copy_metadata``
    propagates annotations but not attributes -- so the name survives
    SpikeInterface preprocessing steps.
    """

    @property
    def fiber_ids(self) -> np.ndarray:
        """Return the fiber identifiers (alias for channel_ids)."""
        return self.get_channel_ids()

    def get_fiber_ids(self) -> np.ndarray:
        """Return the fiber identifiers (alias for get_channel_ids)."""
        return self.get_channel_ids()

    def get_num_fibers(self) -> int:
        """Return the number of fibers (alias for get_num_channels)."""
        return self.get_num_channels()

    def get_fluorescence(
        self,
        segment_index: int | None = None,
        start_frame: int | None = None,
        end_frame: int | None = None,
        fiber_ids: Sequence | None = None,
    ) -> np.ndarray:
        """
        Get fluorescence traces for specified fibers and time range.

        This is a fiber-native wrapper around `get_traces()` that uses
        fiber-photometry vocabulary.

        Parameters
        ----------
        segment_index : int, optional
            The segment index. If None and only one segment exists,
            defaults to 0.
        start_frame : int, optional
            The start frame. If None, starts from the beginning.
        end_frame : int, optional
            The end frame (exclusive). If None, reads to the end.
        fiber_ids : list or array-like, optional
            The fiber IDs to retrieve. If None, returns all fibers.

        Returns
        -------
        traces : np.ndarray
            Array of shape (n_samples, n_fibers) containing the
            fluorescence data.
        """
        return self.get_traces(
            segment_index=segment_index,
            start_frame=start_frame,
            end_frame=end_frame,
            channel_ids=fiber_ids,
        )

    def __repr__(self) -> str:
        """Return a one-line summary: class, name, fiber/segment count."""
        n_seg = self.get_num_segments()
        n_fib = self.get_num_fibers()
        sf = self.get_sampling_frequency()
        dtype = self.get_dtype()
        return (
            f"{self.__class__.__name__} | name={self.name} | "
            f"{n_fib} fiber(s) | {n_seg} segment(s) | "
            f"{sf:.1f} Hz | dtype: {dtype}"
        )


class BaseFiberPhotometryExtractor(FiberPhotometryMixin, BaseRecording):
    """
    Base class for fiber photometry recordings.

    This class represents a single recording from a fiber photometry
    experiment, one excitation/emission channel. It wraps SpikeInterface's
    BaseRecording, takes its fiber-native API from FiberPhotometryMixin,
    and adds the source-side concerns: construction, `get_streams()` for
    format-specific discovery of available data streams, and segment
    loading.

    Each channel in an experiment (e.g. signal and isosbestic) should be its
    own BaseFiberPhotometryExtractor instance, grouped together via
    FiberPhotometryRecordingGroup.

    Parameters
    ----------
    sampling_frequency : float
        The nominal sampling frequency in Hz.
    fiber_ids : list or array-like
        Identifiers for each fiber (will be used as channel_ids internally).
    name : str
        Free-text label for this recording (e.g. "470nm", "GCaMP6f",
        "iso"), stored as SpikeInterface's ``name``.
    dtype : dtype, optional
        The data type of the traces. Default is float64.

    Notes
    -----
    Metadata for the NWB fiber table lives in per-fiber SpikeInterface
    properties, one value per fiber, set with ``rec.set_property(name,
    values)``:

    - ``fiber_location``: brain area, e.g. ``["DMS", "NAc"]``. The name
      ``location`` is reserved by SpikeInterface for probe coordinates.
    - ``indicator``: indicator label, e.g. ``["GCaMP6f", "dLight1.3b"]``.
    - ``excitation_nm`` and ``emission_nm``: wavelengths in nanometres.

    All of them survive SpikeInterface processing steps and the
    ``dump_to_json`` round trip.
    """

    # SpikeInterface's default ``dump_to_json`` keeps only these properties.
    _main_properties = [
        *BaseRecording._main_properties,
        "fiber_location",
        "indicator",
        "excitation_nm",
        "emission_nm",
    ]

    def __init__(
        self,
        sampling_frequency: float,
        fiber_ids: Sequence,
        name: str,
        dtype: np.dtype = np.float64,
    ):
        # Pass fiber_ids as channel_ids to SI's BaseRecording
        BaseRecording.__init__(
            self,
            sampling_frequency=sampling_frequency,
            channel_ids=list(fiber_ids),
            dtype=dtype,
        )
        # SpikeInterface stores the name as an annotation, not an attribute,
        # so copy_metadata carries it across preprocessing steps.
        self.name = name
        # Store kwargs for serialization
        self._kwargs = {
            "sampling_frequency": sampling_frequency,
            "fiber_ids": list(fiber_ids),
            "name": name,
            "dtype": str(dtype),
        }

    @classmethod
    def get_streams(
        cls, file_path: str, **kwargs
    ) -> tuple[list[str], list[str]]:
        """
        Discover available streams in a file.

        This is a discovery hook that file-format-specific readers should
        override. Returns the available stream names and IDs so users can
        choose which to load.

        Parameters
        ----------
        file_path : str
            Path to the file or folder to inspect.
        **kwargs
            Format-specific discovery options. Most readers ignore these;
            some accept keywords that mirror their constructor (e.g. the CSV
            reader accepts ``time_column`` so discovery excludes the same
            column the loader will). Overrides should declare the options
            they support explicitly.

        Returns
        -------
        stream_names : list of str
            Human-readable names of available streams.
        stream_ids : list of str
            Identifiers to pass to the reader's constructor.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.
        """
        raise NotImplementedError(
            f"{cls.__name__}.get_streams() must be implemented by file readers"
        )

    def _add_numpy_segment(
        self,
        traces: np.ndarray,
        timestamps: np.ndarray,
    ) -> None:
        """
        Add an in-memory segment and set its timestamps.

        Convenience method used by all file-format extractors that load
        data into NumPy arrays. Calls :meth:`add_segment` and
        :meth:`set_times` so subclasses do not repeat the boilerplate.

        Parameters
        ----------
        traces : np.ndarray
            2-D array of shape ``(n_samples, n_fibers)``. Cast to float64
            before storing.
        timestamps : np.ndarray
            1-D array of timestamps in seconds, length ``n_samples``.
        """
        t_start = float(timestamps[0]) if len(timestamps) > 0 else 0.0
        segment = NumpyRecordingSegment(
            traces=traces.astype(np.float64),
            sampling_frequency=self.get_sampling_frequency(),
            t_start=t_start,
        )
        self.add_segment(segment)
        self.set_times(timestamps, with_warning=False)


class FiberPhotometryRecordingGroup:
    """
    A container for multiple fiber photometry recordings sharing fibers.

    This class groups several named recordings
    (BaseFiberPhotometryExtractor instances) that share the same fiber
    layout but may have different timebases or sample counts.

    This is NOT a BaseRecording because the recordings can have different
    timebases, so there's no single trace matrix to return.

    Parameters
    ----------
    recordings : dict
        A dictionary mapping names to
        BaseFiberPhotometryExtractor instances. All recordings must have
        identical fiber_ids in the same order.

    Examples
    --------
    >>> green = CsvFiberPhotometryExtractor("green.csv", name="green")
    >>> iso = CsvFiberPhotometryExtractor("iso.csv", name="iso")
    >>> group = FiberPhotometryRecordingGroup({"green": green, "iso": iso})
    >>> group.names
    ['green', 'iso']
    >>> group["green"].get_fluorescence()
    """

    def __init__(self, recordings: dict[str, BaseRecording]):
        if not recordings:
            raise ValueError("recordings dict cannot be empty")

        self._recordings = dict(recordings)
        self._validate_fiber_ids()

    def _validate_fiber_ids(self) -> None:
        """Validate that all recordings have identical fiber_ids."""
        fiber_ids_list = [
            tuple(rec.get_channel_ids()) for rec in self._recordings.values()
        ]
        reference = fiber_ids_list[0]
        for name, fiber_ids in zip(
            self._recordings.keys(), fiber_ids_list, strict=True
        ):
            if fiber_ids != reference:
                raise ValueError(
                    f"Recording '{name}' has different fiber_ids "
                    f"than the first recording. All recordings must "
                    f"share identical fiber_ids in the same order."
                )

    @property
    def names(self) -> list[str]:
        """Return the list of names in this group."""
        return list(self._recordings.keys())

    @property
    def fiber_ids(self) -> np.ndarray:
        """Return the shared fiber identifiers."""
        first_rec = next(iter(self._recordings.values()))
        return first_rec.get_channel_ids()

    def get_num_fibers(self) -> int:
        """Return the number of fibers shared by all recordings."""
        return len(self.fiber_ids)

    def get_recording(self, name: str) -> BaseRecording:
        """
        Get the recording for a specific name.

        Parameters
        ----------
        name : str
            The recording's key in the group.

        Returns
        -------
        BaseRecording
            The recording for the specified name.
        """
        return self._recordings[name]

    # Dict-like access
    def __getitem__(self, name: str) -> BaseRecording:
        """Return the recording for a specific name."""
        return self._recordings[name]

    def __contains__(self, name: str) -> bool:
        """Check if a name exists in this group."""
        return name in self._recordings

    def __len__(self) -> int:
        """Return the number of recordings in this group."""
        return len(self._recordings)

    def __iter__(self) -> Iterable[str]:
        """Iterate over the names in this group."""
        return iter(self._recordings)

    def keys(self):
        """Return the names in this group."""
        return self._recordings.keys()

    def values(self):
        """Return the recordings in this group."""
        return self._recordings.values()

    def items(self):
        """Return (name, recording) pairs in this group."""
        return self._recordings.items()

    def __repr__(self) -> str:
        """Return a multi-line summary listing each named recording."""
        lines = [
            f"{self.__class__.__name__} | "
            f"{self.get_num_fibers()} fiber(s), {len(self)} recording(s)"
        ]
        for name, rec in self._recordings.items():
            lines.append(f"  [{name}] {rec!r}")
        return "\n".join(lines)
