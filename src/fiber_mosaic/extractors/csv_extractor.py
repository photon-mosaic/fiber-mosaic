"""
CSV file extractor for fiber photometry data.

Supports standard CSV files with timestamps and fluorescence data columns.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from spikeinterface.core.numpyextractors import NumpyRecordingSegment

from fiber_mosaic.core.base import BaseFiberPhotometryExtractor

logger = logging.getLogger(__name__)


class CsvFiberPhotometryExtractor(BaseFiberPhotometryExtractor):
    """
    Extractor for CSV fiber photometry files.

    Reads CSV files containing timestamps and fluorescence data. The CSV
    should have:
    - A time column (default: "time" or "timestamps")
    - One or more data columns (fiber channels)

    Parameters
    ----------
    file_path : str or Path
        Path to the CSV file.
    color : str
        Color/wavelength identifier for this recording (e.g., "green", "iso").
    time_column : str, optional
        Name of the timestamp column. Default is "time".
        Will also try "timestamps" if "time" is not found.
    fiber_columns : list of str, optional
        Names of the fiber data columns to read. If None, reads all columns
        except the time column.
    sampling_frequency : float, optional
        Override the sampling frequency. If None, it's computed from
        timestamps.

    Examples
    --------
    >>> recording = CsvFiberPhotometryExtractor("data.csv", color="green")
    >>> recording.get_fluorescence()
    """

    def __init__(
        self,
        file_path: str | Path,
        color: str,
        time_column: str = "time",
        fiber_columns: list[str] | None = None,
        sampling_frequency: float | None = None,
    ):
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")

        # Read CSV file
        df = pd.read_csv(file_path, index_col=False)
        columns = list(df.columns)

        # Find time column
        time_col = None
        time_candidates = [
            time_column,
            "time",
            "timestamps",
            "Time",
            "Timestamps",
            "Time(s)",
        ]
        for candidate in time_candidates:
            if candidate in columns:
                time_col = candidate
                break
        if time_col is None:
            raise ValueError(
                f"Could not find time column. "
                f"Tried: {time_column}, time, timestamps. "
                f"Available columns: {columns}"
            )

        # Determine fiber columns
        if fiber_columns is None:
            fiber_columns = [c for c in columns if c != time_col]
        if not fiber_columns:
            raise ValueError(
                f"No fiber columns found. Available columns: {columns}"
            )

        # Validate fiber columns exist
        missing = [c for c in fiber_columns if c not in columns]
        if missing:
            raise ValueError(f"Fiber columns not found in CSV: {missing}")

        # Extract data
        times = df[time_col].to_numpy().astype(np.float64)
        fiber_data = [df[c].to_numpy() for c in fiber_columns]
        traces = np.column_stack(fiber_data).astype(np.float64)

        # Compute or use provided sampling frequency
        if sampling_frequency is None:
            if len(times) > 1:
                sampling_frequency = (len(times) - 1) / (times[-1] - times[0])
            else:
                raise ValueError(
                    "Cannot compute sampling frequency from single sample. "
                    "Please provide sampling_frequency parameter."
                )

        # Initialize base class
        super().__init__(
            sampling_frequency=sampling_frequency,
            fiber_ids=fiber_columns,
            color=color,
            dtype=traces.dtype,
        )

        # Add segment with data
        segment = NumpyRecordingSegment(
            traces=traces,
            sampling_frequency=sampling_frequency,
            t_start=times[0] if len(times) > 0 else 0.0,
        )
        self.add_segment(segment)

        # Store actual timestamps
        self.set_times(times)

        # Store kwargs for serialization
        self._kwargs = {
            "file_path": str(file_path),
            "color": color,
            "time_column": time_col,
            "fiber_columns": list(fiber_columns),
            "sampling_frequency": sampling_frequency,
        }

        logger.info(
            "Loaded CSV file %s: %d fibers, %d samples, %.2f Hz",
            file_path.name,
            len(fiber_columns),
            len(times),
            sampling_frequency,
        )

    @classmethod
    def get_streams(cls, file_path: str) -> tuple[list[str], list[str]]:
        """
        Discover available streams (columns) in a CSV file.

        Parameters
        ----------
        file_path : str
            Path to the CSV file.

        Returns
        -------
        stream_names : list of str
            Names of data columns (excluding time column).
        stream_ids : list of str
            Same as stream_names for CSV files.
        """
        file_path = Path(file_path)
        df = pd.read_csv(file_path, nrows=0, index_col=False)
        columns = list(df.columns)

        # Identify time column
        time_columns = {"time", "timestamps", "Time", "Timestamps", "Time(s)"}
        data_columns = [c for c in columns if c not in time_columns]

        return data_columns, data_columns


def read_csv_fiber_photometry(
    file_path: str | Path,
    color: str,
    time_column: str = "time",
    fiber_columns: list[str] | None = None,
    sampling_frequency: float | None = None,
) -> CsvFiberPhotometryExtractor:
    """
    Read a CSV fiber photometry file.

    Convenience function that creates a CsvFiberPhotometryExtractor.

    Parameters
    ----------
    file_path : str or Path
        Path to the CSV file.
    color : str
        Color/wavelength identifier for this recording.
    time_column : str, optional
        Name of the timestamp column. Default is "time".
    fiber_columns : list of str, optional
        Names of the fiber data columns to read.
    sampling_frequency : float, optional
        Override the sampling frequency.

    Returns
    -------
    CsvFiberPhotometryExtractor
        The loaded recording.

    Examples
    --------
    >>> rec = read_csv_fiber_photometry("green_channel.csv", color="green")
    >>> rec.get_fluorescence()
    """
    return CsvFiberPhotometryExtractor(
        file_path=file_path,
        color=color,
        time_column=time_column,
        fiber_columns=fiber_columns,
        sampling_frequency=sampling_frequency,
    )
