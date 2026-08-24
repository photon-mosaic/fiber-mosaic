"""
Doric file extractor for fiber photometry data.

Reads Doric Neuroscience .doric files and CSV exports.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from fiber_mosaic.core.base import BaseFiberPhotometryExtractor

logger = logging.getLogger(__name__)

# Check for h5py availability
try:
    import h5py

    HAVE_H5PY = True
except ImportError:  # pragma: no cover
    HAVE_H5PY = False


def _check_h5py():
    """Raise ImportError if h5py is not available."""
    if not HAVE_H5PY:
        raise ImportError(
            "h5py is required to read Doric .doric files. "
            "Install it with: pip install h5py"
        )


def _find_doric_files(folder_path: Path) -> tuple[list[Path], list[Path]]:
    """Find .doric and Doric CSV files in a folder."""
    doric_files = list(folder_path.glob("*.doric"))
    csv_files = list(folder_path.glob("*.csv"))

    # Filter CSV files to only Doric CSV format
    doric_csv_files = []
    for csv_path in csv_files:
        try:
            df = pd.read_csv(
                csv_path, header=None, nrows=2, index_col=False, dtype=str
            )
            df = df.dropna(axis=1, how="all")
            df_arr = np.array(df).flatten()
            # Doric CSVs have all-string first rows
            non_numeric = [el for el in df_arr if not _is_float(el)]
            if len(non_numeric) == len(df_arr):
                doric_csv_files.append(csv_path)
        except Exception:
            pass

    return doric_files, doric_csv_files


def _is_float(value) -> bool:
    """Return True if value can be interpreted as a float."""
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _read_doric_v1_keys(h5file) -> list[str]:
    """Read signal keys from Doric V1 format."""
    if "Traces" not in h5file or "Console" not in h5file["Traces"]:
        return []
    keys = list(h5file["Traces"]["Console"].keys())
    if "Time(s)" in keys:
        keys.remove("Time(s)")
    return keys


def _read_doric_v6_keys(h5file) -> list[str]:
    """Read signal keys from Doric V6 format."""
    if "DataAcquisition" not in h5file:
        return []

    keys = []
    data = [h5file["DataAcquisition"]]
    results = []

    while data:
        current = data.pop()
        if isinstance(current, h5py.Dataset):
            if not current.name.endswith("/Time"):
                results.append(current.name)
        elif isinstance(current, h5py.Group):
            for k in current.keys():
                data.append(current[k])

    for element in results:
        sep_values = element.split("/")
        if sep_values[-1] == "Values":
            keys.append(f"{sep_values[-3]}/{sep_values[-2]}")
        else:
            keys.append(f"{sep_values[-2]}/{sep_values[-1]}")

    return keys


def _discover_doric_file_streams(file_path: Path) -> list[str]:
    """Discover available streams in a .doric file."""
    _check_h5py()

    with h5py.File(str(file_path), "r") as f:
        top_keys = list(f.keys())

        if "Traces" in top_keys:
            # V1 format
            return _read_doric_v1_keys(f)
        elif set(top_keys) == {"Configurations", "DataAcquisition"}:
            # V6 format
            return _read_doric_v6_keys(f)
        else:
            logger.warning("Unknown Doric file format: %s", file_path)
            return []


def _discover_doric_csv_streams(file_path: Path) -> list[str]:
    """Discover available streams in a Doric CSV file."""
    df = pd.read_csv(file_path, header=1, index_col=False, nrows=0)
    columns = list(df.columns)
    # Remove time column
    return [c for c in columns if c != "Time(s)"]


def _get_v1_dataset(node, name: str):
    """Resolve a Doric V1 entry that may be a dataset or a group.

    Real exports nest 'Time(s)' and stream names as groups wrapping a
    single dataset (e.g. 'Time(s)/Console_time(s)').
    """
    item = node[name]
    if isinstance(item, h5py.Group):
        keys = list(item.keys())
        if name in keys:
            return item[name]
        if len(keys) == 1:
            return item[keys[0]]
        raise ValueError(
            f"Cannot resolve dataset for '{name}': multiple candidates {keys}"
        )
    return item


def _read_doric_v1_data(
    h5file, stream_name: str
) -> tuple[np.ndarray, np.ndarray, float]:
    """Read data from Doric V1 format."""
    console = h5file["Traces"]["Console"]
    timestamps = np.array(_get_v1_dataset(console, "Time(s)")[:])
    data = np.array(_get_v1_dataset(console, stream_name)[:])

    if len(timestamps) > 1:
        sampling_rate = 1.0 / np.median(np.diff(timestamps))
    else:
        sampling_rate = 1.0

    return data, timestamps, sampling_rate


def _find_v6_group(group, target_parts):
    """Recursively find the group containing data for target path parts."""
    for key in group.keys():
        item = group[key]
        if isinstance(item, h5py.Group):
            if len(target_parts) > 0 and key == target_parts[0]:
                if len(target_parts) == 1:
                    return item
                return _find_v6_group(item, target_parts[1:])
            result = _find_v6_group(item, target_parts)
            if result is not None:
                return result
    return None


def _extract_v6_data(data_group, stream_name: str) -> np.ndarray:
    """Extract data array from a Doric V6 data group."""
    if "Values" in data_group:
        return np.array(data_group["Values"][:])

    for key in data_group.keys():
        if isinstance(data_group[key], h5py.Dataset) and key != "Time":
            return np.array(data_group[key][:])

    raise ValueError(f"No data found in stream: {stream_name}")


def _extract_v6_timestamps(data_group, data_len: int) -> np.ndarray:
    """Extract timestamps from a Doric V6 data group."""
    if "Time" in data_group:
        return np.array(data_group["Time"][:])

    parent = data_group.parent
    if "Time" in parent:
        return np.array(parent["Time"][:])

    return np.arange(data_len)


def _find_v6_parent_dataset(group, target_parts):
    """Find a group directly containing a dataset named target_parts[-1].

    target_parts[:-1] gives its nesting path. Handles Doric V6 exports
    where the leaf value is stored as a dataset alongside a sibling
    "Time" dataset, rather than inside its own "Values"-bearing group
    (e.g. multi-ROI groups like CAM1_EXC1/{ROI01,ROI02,ROI03,Time}).
    """
    if len(target_parts) == 1:
        name = target_parts[0]
        if name in group and isinstance(group[name], h5py.Dataset):
            return group, name
        return None

    for key in group.keys():
        item = group[key]
        if isinstance(item, h5py.Group):
            if key == target_parts[0]:
                result = _find_v6_parent_dataset(item, target_parts[1:])
                if result is not None:
                    return result
            result = _find_v6_parent_dataset(item, target_parts)
            if result is not None:
                return result
    return None


def _read_doric_v6_data(
    h5file, stream_name: str
) -> tuple[np.ndarray, np.ndarray, float]:
    """Read data from Doric V6 format."""
    parts = stream_name.split("/")
    data_group = _find_v6_group(h5file["DataAcquisition"], parts)

    if data_group is not None:
        data = _extract_v6_data(data_group, stream_name)
        timestamps = _extract_v6_timestamps(data_group, len(data))
    else:
        found = _find_v6_parent_dataset(h5file["DataAcquisition"], parts)
        if found is None:
            raise ValueError(f"Could not find data for stream: {stream_name}")
        parent_group, dataset_name = found
        data = np.array(parent_group[dataset_name][:])
        timestamps = _extract_v6_timestamps(parent_group, len(data))

    if len(timestamps) > 1:
        sampling_rate = 1.0 / np.median(np.diff(timestamps))
    else:
        sampling_rate = 1.0

    return data, timestamps, sampling_rate


def _read_doric_csv_data(
    file_path: Path, stream_name: str
) -> tuple[np.ndarray, np.ndarray, float]:
    """Read data from Doric CSV file."""
    df = pd.read_csv(file_path, header=1, index_col=False)
    df = df.dropna(axis=1, how="all")
    df = df.dropna(axis=0, how="any")

    if stream_name not in df.columns:
        raise ValueError(
            f"Stream '{stream_name}' not found in CSV. "
            f"Available: {list(df.columns)}"
        )

    timestamps = df["Time(s)"].to_numpy()
    # Normalize timestamps to start at 0
    timestamps = timestamps - timestamps[0]
    data = df[stream_name].to_numpy()

    if len(timestamps) > 1:
        sampling_rate = 1.0 / np.median(np.diff(timestamps))
    else:
        sampling_rate = 1.0

    return data, timestamps, sampling_rate


def _resolve_doric_path(file_path: Path) -> Path:
    """Resolve a directory path to a specific Doric file."""
    if not file_path.is_dir():
        return file_path

    doric_files, csv_files = _find_doric_files(file_path)
    if doric_files:
        return doric_files[0]
    if csv_files:
        return csv_files[0]
    raise FileNotFoundError(f"No Doric files found in {file_path}")


def _select_doric_stream(
    stream_name: str | None,
    available_streams: list[str],
) -> str:
    """Select a stream from available streams."""
    if stream_name is None:
        if len(available_streams) == 1:
            return available_streams[0]
        raise ValueError(
            f"Multiple streams found: {available_streams}. "
            "Please specify stream_name."
        )

    if stream_name not in available_streams:
        raise ValueError(
            f"Stream '{stream_name}' not found. Available: {available_streams}"
        )
    return stream_name


def _read_doric_file_data(
    file_path: Path,
    stream_name: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Read data from a .doric HDF5 file."""
    with h5py.File(str(file_path), "r") as f:
        top_keys = list(f.keys())
        if "Traces" in top_keys:
            return _read_doric_v1_data(f, stream_name)
        return _read_doric_v6_data(f, stream_name)


class DoricFiberPhotometryExtractor(BaseFiberPhotometryExtractor):
    """
    Extractor for Doric Neuroscience fiber photometry files.

    Supports both .doric (HDF5-based) files and Doric CSV exports.

    Parameters
    ----------
    file_path : str or Path
        Path to the .doric file, Doric CSV file, or folder containing
        such files.
    stream_name : str, optional
        Name of the stream to read. If None and only one stream exists,
        it will be used automatically.
    color : str
        Color/wavelength identifier (e.g. ``'470nm'``, ``'415nm'``).
        Must be provided explicitly; Doric stream names do not encode
        illumination color.

    Examples
    --------
    >>> # List available streams
    >>> streams = DoricFiberPhotometryExtractor.get_streams(
    ...     "/path/to/data.doric"
    ... )
    >>> names, ids = streams

    >>> # Read a specific stream
    >>> recording = DoricFiberPhotometryExtractor(
    ...     "/path/to/data.doric",
    ...     stream_name="AIn-1",
    ...     color="green"
    ... )
    """

    def __init__(
        self,
        file_path: str | Path,
        stream_name: str | None = None,
        color: str | None = None,
    ):
        file_path = _resolve_doric_path(Path(file_path))
        is_doric_file = file_path.suffix == ".doric"

        # Discover streams
        if is_doric_file:
            _check_h5py()
            available_streams = _discover_doric_file_streams(file_path)
        else:
            available_streams = _discover_doric_csv_streams(file_path)

        if not available_streams:
            raise ValueError(f"No streams found in {file_path}")

        stream_name = _select_doric_stream(stream_name, available_streams)

        # Read data
        if is_doric_file:
            data, timestamps, sampling_rate = _read_doric_file_data(
                file_path, stream_name
            )
        else:
            data, timestamps, sampling_rate = _read_doric_csv_data(
                file_path, stream_name
            )

        # Ensure data is 2D
        if data.ndim == 1:
            data = data[:, np.newaxis]

        # Create fiber IDs
        fiber_ids = [stream_name]

        if color is None:
            raise ValueError(
                "color must be provided explicitly (e.g. color='470nm'). "
                "Doric stream names encode demodulation type, "
                "not illumination color."
            )

        # Initialize base class
        super().__init__(
            sampling_frequency=sampling_rate,
            fiber_ids=fiber_ids,
            color=color,
            dtype=data.dtype,
        )

        self._add_numpy_segment(data, timestamps)

        # Store kwargs
        self._kwargs = {
            "file_path": str(file_path),
            "stream_name": stream_name,
            "color": color,
        }

        logger.info(
            "Loaded Doric file %s, stream '%s': %d samples, %.2f Hz",
            file_path.name,
            stream_name,
            len(data),
            sampling_rate,
        )

    @classmethod
    def get_streams(cls, file_path: str | Path) -> tuple[list[str], list[str]]:
        """
        Discover available streams in a Doric file.

        Parameters
        ----------
        file_path : str or Path
            Path to a Doric file (.doric or .csv).

        Returns
        -------
        stream_names : list of str
            Names of available streams.
        stream_ids : list of str
            Same as stream_names for Doric.
        """
        file_path = Path(file_path)

        if file_path.suffix == ".doric":
            streams = _discover_doric_file_streams(file_path)
        else:
            streams = _discover_doric_csv_streams(file_path)

        return streams, streams


def read_doric_fiber_photometry(
    file_path: str | Path,
    stream_name: str | None = None,
    color: str | None = None,
) -> DoricFiberPhotometryExtractor:
    """
    Read fiber photometry data from a Doric file.

    Convenience function that creates a DoricFiberPhotometryExtractor.

    Parameters
    ----------
    file_path : str or Path
        Path to the Doric file or folder.
    stream_name : str, optional
        Name of the stream to read.
    color : str, optional
        Color/wavelength identifier.

    Returns
    -------
    DoricFiberPhotometryExtractor
        The loaded recording.
    """
    return DoricFiberPhotometryExtractor(
        file_path=file_path,
        stream_name=stream_name,
        color=color,
    )
