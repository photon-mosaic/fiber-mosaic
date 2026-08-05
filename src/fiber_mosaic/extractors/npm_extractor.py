"""
Neurophotometrics (NPM) file extractor for fiber photometry data.

Reads Neurophotometrics multi-channel CSV files with interleaved LED states.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from fiber_mosaic.core.base import BaseFiberPhotometryExtractor

logger = logging.getLogger(__name__)


def _discover_npm_streams(file_path: Path) -> tuple[list[str], dict[str, int]]:
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
        c for c in columns if c.lower() not in timestamp_cols and c != led_col
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


def _resolve_npm_stream(
    available_streams: list[str],
    stream_name: str | None,
) -> str:
    """Resolve a stream name, auto-selecting when only one stream exists."""
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


def _find_npm_column(columns: list[str], candidates: list[str]) -> str | None:
    """Find a column matching one of the candidate names (case-insensitive)."""
    for col in columns:
        if col.lower() in candidates:
            return col
    return None


def _find_npm_timestamp_columns(columns: list[str]) -> list[str]:
    """Find all timestamp-like columns (e.g. Timestamp, SystemTimestamp)."""
    result = []
    for col in columns:
        lower = col.lower()
        if lower.endswith("timestamp") or lower in ("timestamps", "time"):
            result.append(col)
    return result


def _resolve_npm_timestamp_column(
    columns: list[str],
    timestamp_column: str | None,
) -> str | None:
    """Resolve which timestamp column to use.

    Returns the column name, or None to fall back to frame-index timestamps.
    """
    if timestamp_column is not None:
        if timestamp_column not in columns:
            raise ValueError(
                f"timestamp_column '{timestamp_column}' not found. "
                f"Available columns: {columns}"
            )
        return timestamp_column

    ts_cols = _find_npm_timestamp_columns(columns)
    if len(ts_cols) == 1:
        return ts_cols[0]
    if len(ts_cols) > 1:
        raise ValueError(
            f"Multiple timestamp columns found: {ts_cols}. "
            "Pass timestamp_column= to select one "
            "(e.g. timestamp_column='SystemTimestamp')."
        )
    return _find_npm_column(columns, ["framecounter"])


def _timestamps_in_milliseconds(timestamps: np.ndarray) -> bool:
    """Return True if timestamps appear to be in milliseconds.

    Checks the median inter-sample interval: at any physiological sampling
    rate (1–200 Hz) a millisecond timebase produces intervals well above 1,
    while a second timebase produces intervals well below 1.
    """
    if len(timestamps) < 2:
        return False
    return float(np.median(np.diff(timestamps))) > 1.0


def _extract_npm_timestamps(
    df_filtered: pd.DataFrame,
    ts_col: str | None,
    timestamp_unit: str | None = None,
) -> np.ndarray:
    """Extract timestamps from filtered NPM dataframe."""
    if ts_col is None:
        return np.arange(len(df_filtered))

    timestamps = df_filtered[ts_col].to_numpy()
    if ts_col.lower().endswith("timestamp"):
        in_ms = timestamp_unit == "ms" or (
            timestamp_unit is None and _timestamps_in_milliseconds(timestamps)
        )
        if in_ms:
            timestamps = timestamps / 1000.0
    return timestamps


def _read_npm_data(
    file_path: Path,
    column_name: str,
    led_state: int,
    timestamp_unit: str | None = None,
    timestamp_column: str | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
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
    timestamp_unit : {"s", "ms"}, optional
        Unit override for the timestamp column.
    timestamp_column : str, optional
        Name of the timestamp column to use.

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

    # Find LED state and timestamp columns
    led_col = _find_npm_column(columns, ["ledstate", "flags", "led"])
    ts_col = _resolve_npm_timestamp_column(columns, timestamp_column)

    # Filter by LED state
    if led_col:
        mask = df[led_col] == led_state
        df_filtered = df[mask].copy()
    else:
        df_filtered = df.copy()

    # Extract data and timestamps
    data = df_filtered[column_name].to_numpy()
    timestamps = _extract_npm_timestamps(df_filtered, ts_col, timestamp_unit)

    # Calculate sampling rate
    if len(timestamps) > 1:
        dt = timestamps[-1] - timestamps[0]
        sampling_rate = (len(timestamps) - 1) / dt
    else:
        sampling_rate = 1.0

    data_out = data.astype(np.float64)
    ts_out = timestamps.astype(np.float64)
    return data_out, ts_out, sampling_rate


class NpmFiberPhotometryExtractor(BaseFiberPhotometryExtractor):
    """
    Extractor for Neurophotometrics (NPM) fiber photometry files.

    Reads multi-channel NPM CSV files with interleaved LED states.
    Each LED state (signal, control/isosbestic) is treated as a separate
    stream.

    Parameters
    ----------
    file_path : str or Path
        Path to the NPM CSV file.
    stream_name : str, optional
        Name of the stream to read (format: "ColumnName_ledN").
        If None and only one stream exists, it will be used automatically.
    color : str
        Color/wavelength identifier (e.g. ``"green"``, ``"415nm"``).
    timestamp_unit : {"s", "ms"}, optional
        Unit of the timestamp column. If None (default), the unit is
        inferred from the inter-sample interval. Pass ``"ms"`` or ``"s"``
        to override the heuristic when the file's unit is known.
    timestamp_column : str, optional
        Name of the timestamp column to use (e.g. ``"SystemTimestamp"``,
        ``"ComputerTimestamp"``). If None and exactly one timestamp column
        is found it is used automatically; if multiple are found a
        ``ValueError`` is raised listing the options.

    Examples
    --------
    >>> # List available streams
    >>> streams = NpmFiberPhotometryExtractor.get_streams("/path/to/npm.csv")
    >>> names, ids = streams

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
        file_path: str | Path,
        stream_name: str | None = None,
        color: str | None = None,
        timestamp_unit: str | None = None,
        timestamp_column: str | None = None,
    ):
        if timestamp_unit not in (None, "s", "ms"):
            raise ValueError(
                f"timestamp_unit must be 's', 'ms', or None, "
                f"got {timestamp_unit!r}"
            )

        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Discover streams
        available_streams, led_mapping = _discover_npm_streams(file_path)

        if not available_streams:
            raise ValueError(f"No streams found in {file_path}")

        stream_name = _resolve_npm_stream(available_streams, stream_name)

        # Get column and LED state
        column_name, led_state = led_mapping[stream_name]

        # Read data
        data, timestamps, sampling_rate = _read_npm_data(
            file_path, column_name, led_state, timestamp_unit, timestamp_column
        )

        # Ensure data is 2D
        if data.ndim == 1:
            data = data[:, np.newaxis]

        # Create fiber IDs
        fiber_ids = [stream_name]

        if color is None:
            raise ValueError(
                "color must be provided explicitly (e.g. color='green'). "
                "NPM stream names encode channel and LED state "
                "(e.g. 'G0_led2'), not illumination color."
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
            "timestamp_unit": timestamp_unit,
            "timestamp_column": timestamp_column,
        }

        logger.info(
            "Loaded NPM file %s, stream '%s': %d samples, %.2f Hz",
            file_path.name,
            stream_name,
            len(data),
            sampling_rate,
        )

    @classmethod
    def get_streams(cls, file_path: str | Path) -> tuple[list[str], list[str]]:
        """
        Discover available streams in an NPM file.

        Parameters
        ----------
        file_path : str or Path
            Path to the NPM CSV file.

        Returns
        -------
        stream_names : list of str
            Names of available streams (format: "ColumnName_ledN").
        stream_ids : list of str
            Same as stream_names for NPM.
        """
        streams, _ = _discover_npm_streams(Path(file_path))
        return streams, streams


def read_npm_fiber_photometry(
    file_path: str | Path,
    stream_name: str | None = None,
    color: str | None = None,
    timestamp_unit: str | None = None,
    timestamp_column: str | None = None,
) -> NpmFiberPhotometryExtractor:
    """
    Read fiber photometry data from an NPM (Neurophotometrics) file.

    Convenience function that creates an NpmFiberPhotometryExtractor.

    Parameters
    ----------
    file_path : str or Path
        Path to the NPM CSV file.
    stream_name : str, optional
        Name of the stream to read.
    color : str, optional
        Color/wavelength identifier.
    timestamp_unit : {"s", "ms"}, optional
        Unit of the timestamp column. If None, inferred automatically.
    timestamp_column : str, optional
        Name of the timestamp column to use. If None, auto-detected.

    Returns
    -------
    NpmFiberPhotometryExtractor
        The loaded recording.
    """
    return NpmFiberPhotometryExtractor(
        file_path=file_path,
        stream_name=stream_name,
        color=color,
        timestamp_unit=timestamp_unit,
        timestamp_column=timestamp_column,
    )
