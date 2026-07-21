"""
NWB file extractor for fiber photometry data.

Reads NWB files containing FiberPhotometryResponseSeries objects.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from spikeinterface.core.numpyextractors import NumpyRecordingSegment

from fiber_mosaic.core.base import BaseFiberPhotometryExtractor

logger = logging.getLogger(__name__)

# Check for pynwb availability
try:
    from pynwb import NWBHDF5IO

    HAVE_PYNWB = True
except ImportError:  # pragma: no cover
    HAVE_PYNWB = False


def _check_pynwb():
    """Raise ImportError if pynwb is not available."""
    if not HAVE_PYNWB:
        raise ImportError(
            "pynwb is required to read NWB files. "
            "Install it with: pip install pynwb"
        )


def _find_nwb_file(folder_path: str | Path) -> Path:
    """
    Find the single NWB file in a folder.

    Parameters
    ----------
    folder_path : str or Path
        Path to folder containing NWB file(s).

    Returns
    -------
    Path
        Path to the NWB file.

    Raises
    ------
    FileNotFoundError
        If no NWB file is found.
    ValueError
        If multiple NWB files are found.
    """
    folder_path = Path(folder_path)

    # If it's a file, return it directly
    if folder_path.is_file() and folder_path.suffix == ".nwb":
        return folder_path

    # Search for NWB files in folder
    nwb_files = list(folder_path.glob("*.nwb"))
    if len(nwb_files) == 0:
        raise FileNotFoundError(f"No NWB file found in {folder_path}")
    if len(nwb_files) > 1:
        names = [f.name for f in nwb_files]
        raise ValueError(
            f"Multiple NWB files found in {folder_path}: {names}. "
            "Please specify the exact file path."
        )
    return nwb_files[0]


def _get_ndx_events_version(io) -> str | None:  # pragma: no cover
    """Get the ndx-events namespace version from an open NWBHDF5IO."""
    try:
        namespaces = io._file.attrs.get("namespace_versions", {})
        if isinstance(namespaces, dict):
            return namespaces.get("ndx-events")
        return None
    except Exception:
        return None


def _discover_fiber_photometry_series(nwbfile) -> dict[str, Any]:
    """
    Discover all FiberPhotometryResponseSeries in an NWB file.

    Returns a dict mapping series name to the series object.
    """
    series_dict = {}
    for obj in nwbfile.objects.values():
        if obj.neurodata_type == "FiberPhotometryResponseSeries":
            series_dict[obj.name] = obj
    return series_dict


def _resolve_timing(series, n_samples: int) -> tuple[float, np.ndarray]:
    """
    Resolve timestamps and sampling rate from an NWB series.

    Returns (sampling_rate, timestamps).
    """
    if series.timestamps is not None:
        timestamps = np.array(series.timestamps[:])
        if len(timestamps) > 1:
            dt = timestamps[-1] - timestamps[0]
            sampling_rate = (len(timestamps) - 1) / dt
        else:
            has_rate = hasattr(series, "rate") and series.rate
            sampling_rate = series.rate if has_rate else 1.0
    elif hasattr(series, "rate") and series.rate:
        sampling_rate = float(series.rate)
        has_t_start = hasattr(series, "starting_time")
        t_start = series.starting_time if has_t_start else 0.0
        timestamps = np.arange(n_samples) / sampling_rate + (t_start or 0.0)
    else:
        raise ValueError(
            f"Series {series.name} has neither timestamps nor rate. "
            "Cannot determine timing."
        )
    return sampling_rate, timestamps


class NwbFiberPhotometryExtractor(BaseFiberPhotometryExtractor):
    """
    Extractor for NWB fiber photometry files.

    Reads FiberPhotometryResponseSeries objects from NWB files. Multi-channel
    series (2D data arrays) produce one fiber per column.

    Parameters
    ----------
    file_path : str or Path
        Path to the NWB file or folder containing a single NWB file.
    series_name : str, optional
        Name of the FiberPhotometryResponseSeries to read. If None and only
        one series exists, it will be used automatically.
    color : str, optional
        Color/wavelength identifier. If None, uses the series name.

    Examples
    --------
    >>> # Read from a single-series NWB file
    >>> recording = NwbFiberPhotometryExtractor("session.nwb")

    >>> # Read a specific series
    >>> recording = NwbFiberPhotometryExtractor(
    ...     "session.nwb",
    ...     series_name="GCaMP_signal",
    ...     color="green"
    ... )
    """

    def __init__(
        self,
        file_path: str | Path,
        series_name: str | None = None,
        color: str | None = None,
    ):
        _check_pynwb()

        nwb_path = _find_nwb_file(file_path)

        with NWBHDF5IO(str(nwb_path), "r") as io:
            nwbfile = io.read()

            # Discover available series
            series_dict = _discover_fiber_photometry_series(nwbfile)

            if not series_dict:
                raise ValueError(
                    f"No FiberPhotometryResponseSeries found in {nwb_path}"
                )

            # Select series
            if series_name is None:
                if len(series_dict) == 1:
                    series_name = list(series_dict.keys())[0]
                else:
                    raise ValueError(
                        f"Multiple series found: {list(series_dict.keys())}. "
                        "Please specify series_name."
                    )

            if series_name not in series_dict:
                raise ValueError(
                    f"Series '{series_name}' not found. "
                    f"Available: {list(series_dict.keys())}"
                )

            series = series_dict[series_name]

            # Read data
            data = np.array(series.data[:])
            n_samples = data.shape[0]

            # Handle multi-channel data
            if data.ndim == 1:
                data = data[:, np.newaxis]
                fiber_ids = [series_name]
            else:
                n_channels = data.shape[1]
                fiber_ids = [f"{series_name}_{i}" for i in range(n_channels)]

            # Get timing
            sampling_rate, timestamps = _resolve_timing(series, n_samples)

            # Use series name as color if not provided
            if color is None:
                color = series_name

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
            t_start=timestamps[0] if len(timestamps) > 0 else 0.0,
        )
        self.add_segment(segment)

        # Store timestamps
        self.set_times(timestamps)

        # Store kwargs for serialization
        self._kwargs = {
            "file_path": str(nwb_path),
            "series_name": series_name,
            "color": color,
        }

        logger.info(
            "Loaded NWB file %s, series '%s': %d fibers, %d samples, %.2f Hz",
            nwb_path.name,
            series_name,
            len(fiber_ids),
            n_samples,
            sampling_rate,
        )

    @classmethod
    def get_streams(cls, file_path: str) -> tuple[list[str], list[str]]:
        """
        Discover available FiberPhotometryResponseSeries in an NWB file.

        Parameters
        ----------
        file_path : str
            Path to the NWB file or folder.

        Returns
        -------
        stream_names : list of str
            Names of available series.
        stream_ids : list of str
            Same as stream_names for NWB files.
        """
        _check_pynwb()

        nwb_path = _find_nwb_file(file_path)

        with NWBHDF5IO(str(nwb_path), "r") as io:
            nwbfile = io.read()
            series_dict = _discover_fiber_photometry_series(nwbfile)

        series_names = list(series_dict.keys())
        return series_names, series_names


def read_nwb_fiber_photometry(
    file_path: str | Path,
    series_name: str | None = None,
    color: str | None = None,
) -> NwbFiberPhotometryExtractor:
    """
    Read fiber photometry data from an NWB file.

    Convenience function that creates an NwbFiberPhotometryExtractor.

    Parameters
    ----------
    file_path : str or Path
        Path to the NWB file or folder.
    series_name : str, optional
        Name of the FiberPhotometryResponseSeries to read.
    color : str, optional
        Color/wavelength identifier.

    Returns
    -------
    NwbFiberPhotometryExtractor
        The loaded recording.
    """
    return NwbFiberPhotometryExtractor(
        file_path=file_path,
        series_name=series_name,
        color=color,
    )
