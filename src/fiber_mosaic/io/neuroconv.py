"""NeuroConv interface writing a fiber-mosaic recording to NWB.

Developed here first, to be moved into NeuroConv later. Covers one color,
one fiber and at most one single-input processing step so far.
"""

import numpy as np
import spikeinterface
from neuroconv.basedatainterface import BaseDataInterface
from neuroconv.datainterfaces.fiber_photometry import (
    basefiberphotometryinterface,
)
from neuroconv.utils import dict_deep_update
from pydantic import FilePath
from pynwb import NWBFile
from spikeinterface.core import BaseRecording


def _camel(name: str) -> str:
    """Return ``name`` as CamelCase, so ``green_corr`` reads GreenCorr."""
    return "".join(part.capitalize() for part in str(name).split("_") if part)


class _RecordingPhotometryInterface(
    basefiberphotometryinterface.BaseFiberPhotometryInterface
):
    """Write one live recording as one ``FiberPhotometryResponseSeries``."""

    def __init__(
        self, recording: BaseRecording, metadata_key: str, name: str
    ) -> None:
        self._recording = recording
        self._name = name
        super().__init__(
            stream_names=recording.get_annotation("color"),
            metadata_key=metadata_key,
        )

    def _get_stream_data(self, *, stream_name: str) -> np.ndarray:
        return self._recording.get_traces()

    def _get_stream_timestamps(self, *, stream_name: str) -> np.ndarray:
        return self._recording.get_times()

    def get_metadata(self) -> dict:
        """Return the base metadata with this series' name filled in."""
        metadata = super().get_metadata()
        metadata["FiberPhotometry"][self.metadata_key]["name"] = self._name
        return metadata


def _raw_reader(recording: BaseRecording) -> BaseRecording:
    """Return the reader at the root of ``recording``'s processing chain."""
    while recording.get_parent() is not None:
        recording = recording.get_parent()
    return recording


def _writer(
    recording: BaseRecording, role: str, name_suffix: str = ""
) -> _RecordingPhotometryInterface:
    """Build the single-series writer for ``recording``.

    One series per color, one column per fiber, so name and key carry the
    color only. ``role`` ("raw" or "processed") ends the metadata key, so the
    raw and processed series of one color read separate metadata entries.
    """
    color = recording.get_annotation("color")
    return _RecordingPhotometryInterface(
        recording,
        metadata_key=f"fiber_photometry_{color}_{role}",
        name=f"FiberPhotometryResponseSeries{_camel(color)}{name_suffix}",
    )


class FiberMosaicPhotometryInterface(BaseDataInterface):
    """Write a recording saved as SpikeInterface JSON to NWB.

    The raw reader at the root of the recording's processing chain goes to
    ``acquisition``; the processed recording, if the chain has a step, goes
    to ``processing/ophys``.

    Parameters
    ----------
    file_path : FilePath
        JSON file written by ``BaseRecording.dump_to_json``, e.g. through
        ``fiber_mosaic.io.nwb.dump_recording``.
    verbose : bool, default: False
        Whether to print status messages.
    """

    def __init__(self, file_path: FilePath, verbose: bool = False) -> None:
        super().__init__(verbose=verbose, file_path=file_path)
        processed = spikeinterface.load(file_path)
        raw = _raw_reader(processed)
        self._writers = {"acquisition": _writer(raw, "raw")}
        if processed is not raw:
            self._writers["processing/ophys"] = _writer(
                processed, "processed", name_suffix="Processed"
            )

    def get_metadata(self) -> dict:
        """Return NWBFile metadata plus one entry per written series."""
        metadata = super().get_metadata()
        for writer in self._writers.values():
            metadata = dict_deep_update(metadata, writer.get_metadata())
        return metadata

    def add_to_nwbfile(
        self,
        nwbfile: NWBFile,
        metadata: dict | None = None,
        **conversion_options,
    ) -> None:
        """Add the raw and processed series to ``nwbfile``."""
        metadata = self.get_metadata() if metadata is None else metadata
        for container, writer in self._writers.items():
            writer.add_to_nwbfile(
                nwbfile,
                metadata=metadata,
                parent_container=container,
                **conversion_options,
            )
