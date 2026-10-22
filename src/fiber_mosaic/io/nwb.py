"""Turn fiber-mosaic objects into inputs for the NWB conversion.

The conversion itself, ``fiber_mosaic.io.neuroconv``, takes file paths only.
The helpers here write live objects to files it can read.
"""

from pathlib import Path

from spikeinterface.core import BaseRecording

from fiber_mosaic.core.base import FiberPhotometryRecordingGroup
from fiber_mosaic.io.neuroconv import _raw_reader

# reader arguments that name the source: file readers, TDT, DANDI
_SOURCE_KEYS = ("file_path", "folder_path", "dandi_uri")


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


def dump_group(
    group: FiberPhotometryRecordingGroup,
    folder: str | Path,
    prefix: str | None = None,
) -> dict[str, Path]:
    """Write each recording of ``group`` to its own SpikeInterface JSON file.

    One file per recording, each for its own
    ``FiberMosaicPhotometryInterface``; combine those in NeuroConv's
    ``ConverterPipe`` to write one NWB file.

    Parameters
    ----------
    group : FiberPhotometryRecordingGroup
        Recordings read from files on disk, keyed by name.
    folder : str or Path
        Existing folder to write the files into.
    prefix : str or None, default: None
        Files are named ``{prefix}_{name}.json``. When None, each recording
        uses the name of its own raw source file (without extension).

    Returns
    -------
    dict
        Mapping from name to the JSON file written for it.
    """
    return {
        name: dump_recording(
            recording,
            Path(folder) / f"{prefix or _source_stem(recording)}_{name}.json",
        )
        for name, recording in group.items()
    }


def _source_stem(recording: BaseRecording) -> str:
    """Return the name of the file or folder the recording was read from."""
    kwargs = _raw_reader(recording)._kwargs
    source = next(kwargs[key] for key in _SOURCE_KEYS if key in kwargs)
    return Path(source).stem
