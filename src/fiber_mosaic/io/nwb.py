"""Turn fiber-mosaic objects into inputs for the NWB conversion.

The conversion itself, ``fiber_mosaic.io.neuroconv``, takes file paths only.
The helpers here write live objects to files it can read.
"""

from pathlib import Path

from spikeinterface.core import BaseRecording


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
