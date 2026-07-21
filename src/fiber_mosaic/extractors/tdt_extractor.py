"""
TDT (Tucker-Davis Technologies) file extractor for fiber photometry data.

Reads TDT binary format files (.tsq and .tev files).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from numpy import float32, float64, int32, int64, uint16
from spikeinterface.core.numpyextractors import NumpyRecordingSegment

from fiber_mosaic.core.base import BaseFiberPhotometryExtractor

logger = logging.getLogger(__name__)


def _read_tsq(folder_path: str | Path) -> tuple[pd.DataFrame | None, bool]:
    """
    Read TDT header file (.tsq).

    Parameters
    ----------
    folder_path : str or Path
        Path to folder containing TDT files.

    Returns
    -------
    header_df : pd.DataFrame or None
        DataFrame containing TSQ header information.
    found : bool
        True if TSQ file was found and read successfully.
    """
    names = (
        "size", "type", "name", "chan", "sort_code",
        "timestamp", "fp_loc", "strobe", "format", "frequency"
    )
    formats = (
        int32, int32, "S4", uint16, uint16,
        float64, int64, float64, int32, float32
    )
    offsets = (0, 4, 8, 12, 14, 16, 24, 24, 32, 36)
    tsq_dtype = np.dtype(
        {"names": names, "formats": formats, "offsets": offsets},
        align=True
    )

    folder_path = Path(folder_path)
    tsq_files = list(folder_path.glob("*.tsq"))

    if len(tsq_files) > 1:
        raise ValueError(f"Multiple TSQ files found in {folder_path}")
    if len(tsq_files) == 0:
        return None, False

    tsq_path = tsq_files[0]
    tsq = np.fromfile(str(tsq_path), dtype=tsq_dtype)
    df = pd.DataFrame(tsq)

    logger.debug("Read TSQ file: %s", tsq_path)
    return df, True


def _get_store_names(header_df: pd.DataFrame) -> list[str]:
    """Extract unique store names from TSQ header."""
    header_df = header_df.copy()
    header_df["name"] = np.asarray(header_df["name"], dtype=str)
    all_names = np.unique(header_df["name"])

    # Filter out names shorter than 4 characters (invalid)
    valid_names = [n for n in all_names if len(str(n)) >= 4]
    return valid_names


def _read_tev_store(
    folder_path: Path,
    header_df: pd.DataFrame,
    store_name: str
) -> dict[str, np.ndarray]:
    """
    Read data for a specific store from TDT TEV file.

    Returns a dict with keys: storename, sampling_rate, timestamps, data,
    channels, npoints.
    """
    tev_files = list(folder_path.glob("*.tev"))
    if len(tev_files) > 1:
        raise ValueError(f"Multiple TEV files found in {folder_path}")
    if len(tev_files) == 0:
        raise FileNotFoundError(f"No TEV file found in {folder_path}")

    tev_path = tev_files[0]
    data = header_df.copy()
    data["name"] = np.asarray(data["name"], dtype=str)

    # Find rows for this store
    row_mask = data["name"] == store_name
    if row_mask.sum() == 0:
        available = _get_store_names(header_df)
        raise ValueError(
            f"Store '{store_name}' not found. Available stores: {available}"
        )

    indices = np.where(row_mask)[0]
    first_row = indices[0]

    format_code = data["format"][first_row] + 1

    # Format lookup table: [_, dtype_name, multiplier, numpy_dtype]
    format_table = [
        [0, None, 0, None],
        [0, "float", 1, np.float32],
        [0, "long", 1, np.int32],
        [0, "short", 2, np.int16],
        [0, "byte", 4, np.int8],
    ]

    result = {
        "storename": store_name,
        "sampling_rate": float(data["frequency"][first_row]),
        "timestamps": np.asarray(data["timestamp"][indices]),
        "channels": np.asarray(data["chan"][indices]),
    }

    fp_loc = np.asarray(data["fp_loc"][indices])
    data_size = np.asarray(data["size"])

    if format_code != 5:
        # Continuous data
        multiplier = int(format_table[format_code][2])
        nsample = (data_size[first_row] - 10) * multiplier
        store_data = np.zeros((len(fp_loc), nsample))

        with open(tev_path, "rb") as fp:
            for i, loc in enumerate(fp_loc):
                fp.seek(int(loc), os.SEEK_SET)
                store_data[i, :] = np.fromfile(
                    fp,
                    dtype=format_table[format_code][3],
                    count=nsample
                ).reshape(1, nsample, order="F")

        result["data"] = store_data.T.reshape(-1, order="F")
        result["npoints"] = nsample
    else:
        # Strobe/event data
        result["data"] = np.asarray(data["strobe"][indices])
        result["npoints"] = 1
        result["channels"] = np.ones(result["data"].shape[0], dtype=int)

    return result


class TdtFiberPhotometryExtractor(BaseFiberPhotometryExtractor):
    """
    Extractor for TDT (Tucker-Davis Technologies) fiber photometry files.

    Reads TDT binary format consisting of .tsq (header) and .tev (data) files.

    Parameters
    ----------
    folder_path : str or Path
        Path to folder containing TDT files (.tsq and .tev).
    store_name : str, optional
        Name of the store to read. If None and only one photometry store
        exists, it will be used automatically.
    color : str, optional
        Color/wavelength identifier. If None, uses the store name.

    Examples
    --------
    >>> # List available stores
    >>> names, ids = TdtFiberPhotometryExtractor.get_streams("/path/to/tdt")

    >>> # Read a specific store
    >>> recording = TdtFiberPhotometryExtractor(
    ...     "/path/to/tdt",
    ...     store_name="Dv1A",
    ...     color="green"
    ... )
    """

    def __init__(
        self,
        folder_path: str | Path,
        store_name: str | None = None,
        color: str | None = None,
    ):
        folder_path = Path(folder_path)

        # Read TSQ header
        header_df, found = _read_tsq(folder_path)
        if not found:
            raise FileNotFoundError(f"No TDT files found in {folder_path}")

        # Get available stores
        store_names = _get_store_names(header_df)

        if not store_names:
            msg = f"No valid stores found in TDT files at {folder_path}"
            raise ValueError(msg)

        # Select store
        if store_name is None:
            if len(store_names) == 1:
                store_name = store_names[0]
            else:
                raise ValueError(
                    f"Multiple stores found: {store_names}. "
                    "Please specify store_name."
                )

        if store_name not in store_names:
            raise ValueError(
                f"Store '{store_name}' not found. Available: {store_names}"
            )

        # Read store data
        store_data = _read_tev_store(folder_path, header_df, store_name)

        # Check if this is event/strobe data (no continuous samples)
        if store_data["npoints"] == 1:
            raise ValueError(
                f"Store '{store_name}' is an event/strobe store, "
                "not continuous data. Use a different extractor for "
                "event data."
            )

        # Extract timing and data
        sampling_rate = store_data["sampling_rate"]
        timestamps = store_data["timestamps"]
        raw_data = store_data["data"]
        npoints = store_data["npoints"]

        # Reshape data: TDT stores interleaved samples from all channels
        n_chunks = len(timestamps)
        n_samples = n_chunks * npoints
        data = raw_data.reshape(n_samples, 1)

        # Build continuous timestamp array
        # Each chunk has npoints samples at the given sampling rate
        full_timestamps = np.zeros(n_samples)
        for i, ts in enumerate(timestamps):
            start_idx = i * npoints
            end_idx = start_idx + npoints
            chunk_times = ts + np.arange(npoints) / sampling_rate
            full_timestamps[start_idx:end_idx] = chunk_times

        # Create fiber IDs
        fiber_ids = [store_name]

        # Use store name as color if not provided
        if color is None:
            color = store_name

        # Initialize base class
        super().__init__(
            sampling_frequency=sampling_rate,
            fiber_ids=fiber_ids,
            color=color,
            dtype=data.dtype,
        )

        # Add segment
        segment = NumpyRecordingSegment(
            traces=data.astype(np.float64),
            sampling_frequency=sampling_rate,
            t_start=full_timestamps[0] if len(full_timestamps) > 0 else 0.0,
        )
        self.add_segment(segment)

        # Store timestamps
        self.set_times(full_timestamps)

        # Store kwargs
        self._kwargs = {
            "folder_path": str(folder_path),
            "store_name": store_name,
            "color": color,
        }

        logger.info(
            "Loaded TDT folder %s, store '%s': %d samples, %.2f Hz",
            folder_path.name,
            store_name,
            n_samples,
            sampling_rate,
        )

    @classmethod
    def get_streams(cls, folder_path: str) -> tuple[list[str], list[str]]:
        """
        Discover available stores in a TDT folder.

        Parameters
        ----------
        folder_path : str
            Path to folder containing TDT files.

        Returns
        -------
        stream_names : list of str
            Names of available stores.
        stream_ids : list of str
            Same as stream_names for TDT.
        """
        header_df, found = _read_tsq(folder_path)
        if not found:
            return [], []

        store_names = _get_store_names(header_df)
        return store_names, store_names


def read_tdt_fiber_photometry(
    folder_path: str | Path,
    store_name: str | None = None,
    color: str | None = None,
) -> TdtFiberPhotometryExtractor:
    """
    Read fiber photometry data from TDT files.

    Convenience function that creates a TdtFiberPhotometryExtractor.

    Parameters
    ----------
    folder_path : str or Path
        Path to folder containing TDT files.
    store_name : str, optional
        Name of the store to read.
    color : str, optional
        Color/wavelength identifier.

    Returns
    -------
    TdtFiberPhotometryExtractor
        The loaded recording.
    """
    return TdtFiberPhotometryExtractor(
        folder_path=folder_path,
        store_name=store_name,
        color=color,
    )
