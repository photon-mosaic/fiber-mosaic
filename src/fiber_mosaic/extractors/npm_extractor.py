"""
Neurophotometrics (NPM) file extractor for fiber photometry data.

Reads Neurophotometrics multi-channel CSV files with interleaved LED states.
"""

from __future__ import annotations

import glob
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from spikeinterface.core.numpyextractors import NumpyRecordingSegment

from fiber_mosaic.core.base import BaseFiberPhotometryExtractor

logger = logging.getLogger(__name__)


def _find_npm_csv(folder_path: Path) -> Optional[Path]:
    """Find NPM CSV file in a folder."""
    csv_files = list(folder_path.glob("*.csv"))

    for csv_path in csv_files:
        # Skip event files and files with specific prefixes
        basename = csv_path.name.lower()
        if any(basename.startswith(prefix) for prefix in ["event", "chev", "chod", "chpr"]):
            continue

        try:
            df = pd.read_csv(csv_path, index_col=False, nrows=5)
            columns = list(df.columns)

            # NPM files typically have more than 3 columns and contain
            # LED state or flags column
            if len(columns) > 3:
                cols_lower = [c.lower() for c in columns]
                if any(x in cols_lower for x in ["ledstate", "flags", "led"]):
                    return csv_path
                # Also check for Region columns typical of NPM format
                if any("region" in c.lower() for c in columns):
                    return csv_path
        except Exception:
            continue

    return None


def _discover_npm_streams(file_path: Path) -> Tuple[List[str], Dict[str, int]]:
    """
    Discover available channels and LED states in NPM file.

    Returns stream names and mapping to LED states.
    """
    df = pd.read_csv(file_path, index_col=False)
    columns = list(df.columns)

    # Find LED state column
    led_col = None
    for col in columns:
        if col.lower() in ["ledstate", "flags", "led"]:
            led_col = col
            break

    # Find data columns (not timestamp, not LED state)
    timestamp_cols = {"timestamp", "timestamps", "time", "framecounter"}
    data_columns = [
        c for c in columns
        if c.lower() not in timestamp_cols and c != led_col
    ]

    # Find unique LED states
    if led_col:
        led_states = sorted(df[led_col].unique())
    else:
        led_states = [0]

    # Generate stream names: channel_ledstate
    streams = []
    led_mapping = {}
    for col in data_columns:
        for led in led_states:
            stream_name = f"{col}_led{led}"
            streams.append(stream_name)
            led_mapping[stream_name] = (col, led)

    return streams, led_mapping


def _read_npm_data(
    file_path: Path,
    column_name: str,
    led_state: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Read data for a specific channel and LED state from NPM file.

    Parameters
    ----------
    file_path : Path
        Path to NPM CSV file.
    column_name : str
        Name of the data column.
    led_state : int
        LED state to filter by.

    Returns
    -------
    data : np.ndarray
        Fluorescence data.
    timestamps : np.ndarray
        Timestamps in seconds.
    sampling_rate : float
        Sampling rate in Hz.
    """
    df = pd.read_csv(file_path, index_col=False)
    columns = list(df.columns)

    # Find LED state column
    led_col = None
    for col in columns:
        if col.lower() in ["ledstate", "flags", "led"]:
            led_col = col
            break

    # Find timestamp column
    ts_col = None
    for col in columns:
        if col.lower() in ["timestamp", "timestamps", "time"]:
            ts_col = col
            break

    if ts_col is None:
        # Use frame counter if available
        for col in columns:
            if col.lower() == "framecounter":
                ts_col = col
                break

    # Filter by LED state
    if led_col:
        mask = df[led_col] == led_state
        df_filtered = df[mask].copy()
    else:
        df_filtered = df.copy()

    # Extract data
    data = df_filtered[column_name].to_numpy()

    # Extract timestamps
    if ts_col:
        timestamps = df_filtered[ts_col].to_numpy()
        # Check if timestamps are in milliseconds
        if ts_col.lower() == "timestamp" and timestamps.max() > 1e6:
            timestamps = timestamps / 1000.0  # Convert to seconds
    else:
        timestamps = np.arange(len(data))

    # Calculate sampling rate
    if len(timestamps) > 1:
        sampling_rate = (len(timestamps) - 1) / (timestamps[-1] - timestamps[0])
    else:
        sampling_rate = 1.0

    return data.astype(np.float64), timestamps.astype(np.float64), sampling_rate


class NpmFiberPhotometryExtractor(BaseFiberPhotometryExtractor):
    """
    Extractor for Neurophotometrics (NPM) fiber photometry files.

    Reads multi-channel NPM CSV files with interleaved LED states.
    Each LED state (signal, control/isosbestic) is treated as a separate stream.

    Parameters
    ----------
    file_path : str or Path
        Path to the NPM CSV file or folder containing NPM files.
    stream_name : str, optional
        Name of the stream to read (format: "ColumnName_ledN").
        If None and only one stream exists, it will be used automatically.
    color : str, optional
        Color/wavelength identifier. If None, uses the stream name.

    Examples
    --------
    >>> # List available streams
    >>> names, ids = NpmFiberPhotometryExtractor.get_streams("/path/to/npm.csv")

    >>> # Read signal channel (LED state 2 is typically 470nm signal)
    >>> recording = NpmFiberPhotometryExtractor(
    ...     "/path/to/npm.csv",
    ...     stream_name="Region0G_led2",
    ...     color="green"
    ... )

    >>> # Read isosbestic/control channel (LED state 1 is typically 415nm)
    >>> iso_recording = NpmFiberPhotometryExtractor(
    ...     "/path/to/npm.csv",
    ...     stream_name="Region0G_led1",
    ...     color="iso"
    ... )
    """

    def __init__(
        self,
        file_path: Union[str, Path],
        stream_name: Optional[str] = None,
        color: Optional[str] = None,
    ):
        file_path = Path(file_path)

        # Find NPM file
        if file_path.is_dir():
            npm_file = _find_npm_csv(file_path)
            if npm_file is None:
                raise FileNotFoundError(f"No NPM CSV file found in {file_path}")
            file_path = npm_file
        elif not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Discover streams
        available_streams, led_mapping = _discover_npm_streams(file_path)

        if not available_streams:
            raise ValueError(f"No streams found in {file_path}")

        # Select stream
        if stream_name is None:
            if len(available_streams) == 1:
                stream_name = available_streams[0]
            else:
                raise ValueError(
                    f"Multiple streams found: {available_streams}. "
                    "Please specify stream_name."
                )

        if stream_name not in available_streams:
            raise ValueError(
                f"Stream '{stream_name}' not found. Available: {available_streams}"
            )

        # Get column and LED state
        column_name, led_state = led_mapping[stream_name]

        # Read data
        data, timestamps, sampling_rate = _read_npm_data(
            file_path, column_name, led_state
        )

        # Ensure data is 2D
        if data.ndim == 1:
            data = data[:, np.newaxis]

        # Create fiber IDs
        fiber_ids = [stream_name]

        # Use stream name as color if not provided
        if color is None:
            color = stream_name

        # Initialize base class
        super().__init__(
            sampling_frequency=sampling_rate,
            fiber_ids=fiber_ids,
            color=color,
            dtype=data.dtype,
        )

        # Add segment
        segment = NumpyRecordingSegment(
            traces=data,
            sampling_frequency=sampling_rate,
            t_start=timestamps[0] if len(timestamps) > 0 else 0.0,
        )
        self.add_segment(segment)

        # Store timestamps
        self.set_times(timestamps)

        # Store kwargs
        self._kwargs = {
            "file_path": str(file_path),
            "stream_name": stream_name,
            "color": color,
        }

        logger.info(
            "Loaded NPM file %s, stream '%s': %d samples, %.2f Hz",
            file_path.name,
            stream_name,
            len(data),
            sampling_rate,
        )

    @classmethod
    def get_streams(cls, file_path: str) -> Tuple[List[str], List[str]]:
        """
        Discover available streams in an NPM file.

        Parameters
        ----------
        file_path : str
            Path to the NPM CSV file or folder.

        Returns
        -------
        stream_names : list of str
            Names of available streams (format: "ColumnName_ledN").
        stream_ids : list of str
            Same as stream_names for NPM.
        """
        file_path = Path(file_path)

        if file_path.is_dir():
            npm_file = _find_npm_csv(file_path)
            if npm_file is None:
                return [], []
            file_path = npm_file

        streams, _ = _discover_npm_streams(file_path)
        return streams, streams


def read_npm_fiber_photometry(
    file_path: Union[str, Path],
    stream_name: Optional[str] = None,
    color: Optional[str] = None,
) -> NpmFiberPhotometryExtractor:
    """
    Read fiber photometry data from an NPM (Neurophotometrics) file.

    Convenience function that creates an NpmFiberPhotometryExtractor.

    Parameters
    ----------
    file_path : str or Path
        Path to the NPM CSV file or folder.
    stream_name : str, optional
        Name of the stream to read.
    color : str, optional
        Color/wavelength identifier.

    Returns
    -------
    NpmFiberPhotometryExtractor
        The loaded recording.
    """
    return NpmFiberPhotometryExtractor(
        file_path=file_path,
        stream_name=stream_name,
        color=color,
    )
