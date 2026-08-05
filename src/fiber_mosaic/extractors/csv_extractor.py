"""
CSV file extractor for fiber photometry data.

Supports standard CSV files with timestamps and fluorescence data columns.
"""

from __future__ import annotations

import fnmatch
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from spikeinterface.core.numpyextractors import NumpyRecordingSegment

from fiber_mosaic.core.base import BaseFiberPhotometryExtractor

logger = logging.getLogger(__name__)


def _select_fiber_columns(
    columns: list[str],
    time_col: str,
    fiber_columns: list[str] | None,
    fiber_pattern: str | None,
) -> list[str]:
    """Select the fluorescence (fiber) data columns.

    Selection is positive: unknown columns are not assumed to be data,
    because files can carry non-fluorescence columns (Background,
    CameraFrameNumber, ...) next to the Fiber_* channels.

    - ``fiber_columns`` is authoritative when given.
    - ``fiber_pattern`` positively selects data columns by glob.
    - With neither, all non-time columns are used, but a warning is logged
      when that absorbs more than one column, since it may sweep in metadata.
    """
    if fiber_columns is not None:
        return list(fiber_columns)

    if fiber_pattern is not None:
        matched = [
            c
            for c in columns
            if c != time_col and fnmatch.fnmatchcase(c, fiber_pattern)
        ]
        if not matched:
            raise ValueError(
                f"fiber_pattern={fiber_pattern!r} matched no columns besides "
                f"the time column. Available columns: {columns}"
            )
        return matched

    non_time = [c for c in columns if c != time_col]
    if len(non_time) > 1:
        logger.warning(
            "No fiber_columns or fiber_pattern given; treating all %d "
            "non-time columns as fluorescence data: %s. If this file has "
            "metadata columns (e.g. Background / CameraFrameNumber), "
            "pass fiber_columns=[...] or fiber_pattern=... to select only the "
            "data columns.",
            len(non_time),
            non_time,
        )
    return non_time


class CsvFiberPhotometryExtractor(BaseFiberPhotometryExtractor):
    """
    Extractor for CSV fiber photometry files.

    Reads CSV files containing timestamps and fluorescence data. The CSV
    should have:
    - A time column (default name "time"; pass time_column for others)
    - One or more data columns (fiber channels)

    Parameters
    ----------
    file_path : str or Path
        Path to the CSV file.
    color : str
        Color/wavelength identifier for this recording (e.g., "green", "iso").
    time_column : str
        Name of the timestamp column. Matched exactly against the CSV
        headers (e.g. ``"time"``, ``"timestamps"``, ``"Time(s)"``).
        Required; timestamps synchronize signals with behavior events.
    fiber_columns : list of str, optional
        Explicit names of the fluorescence (fiber) data columns to read.
        Authoritative when given; every name must exist in the file.
    fiber_pattern : str, optional
        Glob pattern (e.g. "Fiber_*") used to positively select the data
        columns by name, ignoring everything else except the time column.
        Use this for files that carry non-fluorescence columns alongside the
        data, such as the AIND FIP standard (Background, CameraFrameNumber,
        CameraFrameTime). Ignored if fiber_columns is given.
    sampling_frequency : float, optional
        Override the sampling frequency. If None, it's computed from
        timestamps.

    Notes
    -----
    If neither fiber_columns nor fiber_pattern is given, all columns except
    the time column are treated as data. This is convenient for clean files
    but unsafe for files with metadata columns, so a warning is logged when
    it falls back to absorbing more than one column. Prefer fiber_columns or
    fiber_pattern for anything other than a plain time + data layout.

    Examples
    --------
    >>> recording = CsvFiberPhotometryExtractor(
    ...     "data.csv", color="green", time_column="time"
    ... )
    >>> recording.get_fluorescence()
    >>> # AIND FIP file: select only the fiber channels
    >>> fip = CsvFiberPhotometryExtractor(
    ...     "green.csv", color="green",
    ...     time_column="ReferenceTime", fiber_pattern="Fiber_*",
    ... )
    """

    def __init__(
        self,
        file_path: str | Path,
        color: str,
        time_column: str,
        fiber_columns: list[str] | None = None,
        fiber_pattern: str | None = None,
        sampling_frequency: float | None = None,
    ):
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")

        # Read CSV file
        df = pd.read_csv(file_path, index_col=False)
        columns = list(df.columns)

        if time_column not in columns:
            raise ValueError(
                f"Time column {time_column!r} not found. "
                f"Available columns: {columns}"
            )
        time_col = time_column

        fiber_columns = _select_fiber_columns(
            columns, time_col, fiber_columns, fiber_pattern
        )
        if not fiber_columns:
            raise ValueError(
                f"No fiber columns found. Available columns: {columns}"
            )

        # Validate fiber columns exist (an explicit list may name absent ones)
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
    def get_streams(
        cls,
        file_path: str,
        time_column: str | None = None,
        fiber_pattern: str | None = None,
        **kwargs,
    ) -> tuple[list[str], list[str]]:
        """
        Discover available streams (columns) in a CSV file.

        Parameters
        ----------
        file_path : str
            Path to the CSV file.
        time_column : str, optional
            Name of the timestamp column to exclude from reported streams.
            Pass the same name you will give the constructor. If None
            (default), all columns are returned including the time column.
        fiber_pattern : str, optional
            Glob pattern (e.g. "Fiber_*") to positively select data columns,
            matching the constructor. If given, only matching columns are
            reported; otherwise every non-time column is reported. Pass this
            for files with metadata columns so discovery does not list them
            as streams.

        Returns
        -------
        stream_names : list of str
            Names of data columns (excluding the time column).
        stream_ids : list of str
            Same as stream_names for CSV files.
        """
        file_path = Path(file_path)
        df = pd.read_csv(file_path, nrows=0, index_col=False)
        columns = list(df.columns)

        # Exclude the named time column, matching the loader. When a pattern
        # is given, positively select data columns instead of reporting all.
        data_columns = [c for c in columns if c != time_column]
        if fiber_pattern is not None:
            data_columns = [
                c
                for c in data_columns
                if fnmatch.fnmatchcase(c, fiber_pattern)
            ]

        return data_columns, data_columns


def read_csv_fiber_photometry(
    file_path: str | Path,
    color: str,
    time_column: str,
    fiber_columns: list[str] | None = None,
    fiber_pattern: str | None = None,
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
    time_column : str
        Name of the timestamp column.
    fiber_columns : list of str, optional
        Explicit names of the fiber data columns to read.
    fiber_pattern : str, optional
        Glob pattern (e.g. "Fiber_*") to positively select data columns,
        ignoring metadata columns. Ignored if fiber_columns is given.
    sampling_frequency : float, optional
        Override the sampling frequency.

    Returns
    -------
    CsvFiberPhotometryExtractor
        The loaded recording.

    Examples
    --------
    >>> rec = read_csv_fiber_photometry(
    ...     "green_channel.csv", color="green", time_column="time"
    ... )
    >>> rec.get_fluorescence()
    """
    return CsvFiberPhotometryExtractor(
        file_path=file_path,
        color=color,
        time_column=time_column,
        fiber_columns=fiber_columns,
        fiber_pattern=fiber_pattern,
        sampling_frequency=sampling_frequency,
    )
