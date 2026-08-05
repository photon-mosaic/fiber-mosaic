"""
NWB file extractor for fiber photometry data.

Reads NWB files containing FiberPhotometryResponseSeries objects.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

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


def _discover_fiber_photometry_series(nwbfile) -> dict[str, Any]:
    """
    Discover all FiberPhotometryResponseSeries in an NWB file.

    Returns a dict mapping series name to the series object.
    """
    series_dict = {}
    for obj in nwbfile.objects.values():
        if (
            getattr(obj, "neurodata_type", None)
            == "FiberPhotometryResponseSeries"
        ):
            series_dict[obj.name] = obj
    return series_dict


def _resolve_nwb_series(
    series_dict: dict[str, Any],
    series_name: str | None,
) -> str:
    """Resolve a series name, auto-selecting when only one series exists."""
    if series_name is None:
        if len(series_dict) == 1:
            return list(series_dict.keys())[0]
        raise ValueError(
            f"Multiple series found: {list(series_dict.keys())}. "
            "Please specify series_name."
        )
    if series_name not in series_dict:
        raise ValueError(
            f"Series '{series_name}' not found. "
            f"Available: {list(series_dict.keys())}"
        )
    return series_name


class NwbFiberPhotometryExtractor(BaseFiberPhotometryExtractor):
    """
    Extractor for NWB fiber photometry files.

    Reads FiberPhotometryResponseSeries objects from NWB files. Multi-channel
    series (2D data arrays) produce one fiber per column.

    Parameters
    ----------
    file_path : str or Path
        Path to the NWB file.
    series_name : str, optional
        Name of the FiberPhotometryResponseSeries to read. If None and only
        one series exists, it will be used automatically.
    color : str
        Color/wavelength identifier (e.g. ``"green"``, ``"415nm"``).
        Must be provided explicitly; NWB series names encode the
        experimental signal, not illumination color.

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

        nwb_path = Path(file_path)
        if not nwb_path.exists():
            raise FileNotFoundError(f"File not found: {nwb_path}")

        with NWBHDF5IO(str(nwb_path), "r") as io:
            nwbfile = io.read()

            # Discover available series
            series_dict = _discover_fiber_photometry_series(nwbfile)

            if not series_dict:
                raise ValueError(
                    f"No FiberPhotometryResponseSeries found in {nwb_path}"
                )

            series_name = _resolve_nwb_series(series_dict, series_name)
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

            # Get timing via pynwb convenience method (handles both
            # explicit timestamps and rate+starting_time cases)
            timestamps = series.get_timestamps()
            if timestamps is None:
                raise ValueError(
                    f"Series '{series_name}' has neither timestamps "
                    "nor rate. Cannot determine timing."
                )
            timestamps = np.asarray(timestamps)
            if series.rate is not None:
                sampling_rate = float(series.rate)
            elif len(timestamps) > 1:
                sampling_rate = (len(timestamps) - 1) / (
                    timestamps[-1] - timestamps[0]
                )
            else:
                sampling_rate = 1.0

        if color is None:
            raise ValueError(
                "color must be provided explicitly (e.g. color='green'). "
                "NWB series names encode the experimental signal "
                "(e.g. 'GCaMP_signal'), not illumination color."
            )

        # Initialize base class
        super().__init__(
            sampling_frequency=sampling_rate,
            fiber_ids=fiber_ids,
            color=color,
            dtype=data.dtype,
        )

        self._add_numpy_segment(data, timestamps)

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
    def get_streams(cls, file_path: str | Path) -> tuple[list[str], list[str]]:
        """
        Discover available FiberPhotometryResponseSeries in an NWB file.

        Parameters
        ----------
        file_path : str or Path
            Path to the NWB file.

        Returns
        -------
        stream_names : list of str
            Names of available series.
        stream_ids : list of str
            Same as stream_names for NWB files.
        """
        _check_pynwb()

        nwb_path = Path(file_path)

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
