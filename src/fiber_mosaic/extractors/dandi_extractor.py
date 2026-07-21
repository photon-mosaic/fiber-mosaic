"""
DANDI Archive streaming extractor for fiber photometry data.

Streams NWB fiber photometry data directly from the DANDI Archive without
downloading the entire file.
"""

from __future__ import annotations

import logging

import numpy as np
from spikeinterface.core.numpyextractors import NumpyRecordingSegment

from fiber_mosaic.core.base import BaseFiberPhotometryExtractor

logger = logging.getLogger(__name__)

DANDI_URI_PREFIX = "dandi://"


def is_dandi_uri(path: str) -> bool:
    """
    Check whether a path is a DANDI URI.

    Parameters
    ----------
    path : str
        Path to check.

    Returns
    -------
    bool
        True if the path starts with ``dandi://``.

    Examples
    --------
    >>> is_dandi_uri("dandi://000971/sub-112/file.nwb")
    True
    >>> is_dandi_uri("/local/path/file.nwb")
    False
    """
    return isinstance(path, str) and path.startswith(DANDI_URI_PREFIX)


def parse_dandi_uri(uri: str) -> tuple[str, str]:
    """
    Parse a DANDI URI into its dandiset ID and asset path.

    Parameters
    ----------
    uri : str
        A DANDI URI of the form ``dandi://DANDISET_ID/asset/path.nwb``.

    Returns
    -------
    dandiset_id : str
        The dandiset identifier (e.g. ``"000971"``).
    asset_path : str
        The asset path within the dandiset (e.g. ``"sub-112/file.nwb"``).

    Raises
    ------
    ValueError
        If the URI is not a valid DANDI URI.

    Examples
    --------
    >>> dandiset_id, asset_path = parse_dandi_uri("dandi://000971/sub-112/file.nwb")
    >>> dandiset_id
    '000971'
    >>> asset_path
    'sub-112/file.nwb'
    """
    if not is_dandi_uri(uri):
        raise ValueError(f"Not a valid DANDI URI: {uri}")

    stripped = uri[len(DANDI_URI_PREFIX):]
    parts = stripped.split("/", 1)

    if len(parts) < 2:
        raise ValueError(
            f"Invalid DANDI URI format: {uri}. "
            "Expected: dandi://DANDISET_ID/asset/path.nwb"
        )

    dandiset_id = parts[0]
    asset_path = parts[1]
    return dandiset_id, asset_path


def _check_dandi_dependencies():
    """Check for required DANDI streaming dependencies."""
    missing = []

    try:
        import dandi  # noqa: F401
    except ImportError:
        missing.append("dandi")

    try:
        import remfile  # noqa: F401
    except ImportError:
        missing.append("remfile")

    try:
        import h5py  # noqa: F401
    except ImportError:
        missing.append("h5py")

    try:
        from pynwb import NWBHDF5IO  # noqa: F401
    except ImportError:
        missing.append("pynwb")

    if missing:
        raise ImportError(
            f"Missing dependencies for DANDI streaming: {missing}. "
            f"Install with: pip install {' '.join(missing)}"
        )


def _stream_nwb(dandiset_id: str, asset_path: str):
    """
    Open a streaming connection to an NWB file on the DANDI Archive.

    Parameters
    ----------
    dandiset_id : str
        Dandiset ID.
    asset_path : str
        Path to the NWB file within the dandiset.

    Returns
    -------
    nwbfile : pynwb.NWBFile
        The opened NWB file.
    io : pynwb.NWBHDF5IO
        The IO object (must be closed when done).
    """
    import h5py
    import remfile
    from dandi.dandiapi import DandiAPIClient
    from pynwb import NWBHDF5IO

    with DandiAPIClient() as client:
        # Try to authenticate but don't fail if not possible
        try:
            client.dandi_authenticate()
        except Exception:
            logger.debug(
                "DANDI authentication failed, proceeding without auth"
            )

        dandiset = client.get_dandiset(dandiset_id, "draft")
        asset = dandiset.get_asset_by_path(asset_path)
        s3_url = asset.get_content_url(follow_redirects=1, strip_query=False)

    # Open remote file
    file_system = remfile.File(s3_url)
    h5_file = h5py.File(file_system, mode="r")
    io = NWBHDF5IO(file=h5_file, load_namespaces=True)
    nwbfile = io.read()

    return nwbfile, io


def _discover_fiber_photometry_series(nwbfile) -> dict:
    """Discover FiberPhotometryResponseSeries in an NWB file."""
    series_dict = {}
    for obj in nwbfile.objects.values():
        if obj.neurodata_type == "FiberPhotometryResponseSeries":
            series_dict[obj.name] = obj
    return series_dict


def _resolve_timing(series, n_samples: int) -> tuple[float, np.ndarray]:
    """Resolve timestamps and sampling rate from an NWB series."""
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
        msg = f"Series {series.name} has neither timestamps nor rate"
        raise ValueError(msg)

    return sampling_rate, timestamps


class DandiFiberPhotometryExtractor(BaseFiberPhotometryExtractor):
    """
    Extractor that streams fiber photometry data from the DANDI Archive.

    Accepts a DANDI URI and streams NWB data directly without downloading
    the entire file.

    Parameters
    ----------
    dandi_uri : str
        A DANDI URI of the form ``dandi://DANDISET_ID/asset/path.nwb``.
    series_name : str, optional
        Name of the FiberPhotometryResponseSeries to read.
    color : str, optional
        Color/wavelength identifier. If None, uses the series name.

    Examples
    --------
    >>> # List available series
    >>> names, ids = DandiFiberPhotometryExtractor.get_streams(
    ...     "dandi://000971/sub-112/session.nwb"
    ... )

    >>> # Read data
    >>> recording = DandiFiberPhotometryExtractor(
    ...     "dandi://000971/sub-112/session.nwb",
    ...     series_name="GCaMP_signal",
    ...     color="green"
    ... )
    """

    def __init__(
        self,
        dandi_uri: str,
        series_name: str | None = None,
        color: str | None = None,
    ):
        _check_dandi_dependencies()

        if not is_dandi_uri(dandi_uri):
            raise ValueError(f"Not a valid DANDI URI: {dandi_uri}")

        dandiset_id, asset_path = parse_dandi_uri(dandi_uri)

        logger.info("Streaming from DANDI: %s/%s", dandiset_id, asset_path)

        # Open streaming connection
        nwbfile, io = _stream_nwb(dandiset_id, asset_path)

        try:
            # Discover available series
            series_dict = _discover_fiber_photometry_series(nwbfile)

            if not series_dict:
                raise ValueError(
                    f"No FiberPhotometryResponseSeries found in {dandi_uri}"
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

            # Read data (this streams from DANDI)
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

        finally:
            # Close the streaming connection
            io.close()

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

        # Store kwargs
        self._kwargs = {
            "dandi_uri": dandi_uri,
            "series_name": series_name,
            "color": color,
        }

        logger.info(
            "Loaded DANDI stream %s, series '%s': "
            "%d fibers, %d samples, %.2f Hz",
            dandi_uri,
            series_name,
            len(fiber_ids),
            n_samples,
            sampling_rate,
        )

    @classmethod
    def get_streams(cls, dandi_uri: str) -> tuple[list[str], list[str]]:
        """
        Discover available FiberPhotometryResponseSeries in a DANDI NWB file.

        Parameters
        ----------
        dandi_uri : str
            A DANDI URI of the form ``dandi://DANDISET_ID/asset/path.nwb``.

        Returns
        -------
        stream_names : list of str
            Names of available series.
        stream_ids : list of str
            Same as stream_names.
        """
        _check_dandi_dependencies()

        if not is_dandi_uri(dandi_uri):
            raise ValueError(f"Not a valid DANDI URI: {dandi_uri}")

        dandiset_id, asset_path = parse_dandi_uri(dandi_uri)
        nwbfile, io = _stream_nwb(dandiset_id, asset_path)

        try:
            series_dict = _discover_fiber_photometry_series(nwbfile)
            series_names = list(series_dict.keys())
        finally:
            io.close()

        return series_names, series_names


def read_dandi_fiber_photometry(
    dandi_uri: str,
    series_name: str | None = None,
    color: str | None = None,
) -> DandiFiberPhotometryExtractor:
    """
    Read fiber photometry data from the DANDI Archive.

    Convenience function that creates a DandiFiberPhotometryExtractor.

    Parameters
    ----------
    dandi_uri : str
        A DANDI URI of the form ``dandi://DANDISET_ID/asset/path.nwb``.
    series_name : str, optional
        Name of the FiberPhotometryResponseSeries to read.
    color : str, optional
        Color/wavelength identifier.

    Returns
    -------
    DandiFiberPhotometryExtractor
        The loaded recording.

    Examples
    --------
    >>> rec = read_dandi_fiber_photometry(
    ...     "dandi://000971/sub-112/session.nwb",
    ...     series_name="GCaMP_signal",
    ...     color="green"
    ... )
    """
    return DandiFiberPhotometryExtractor(
        dandi_uri=dandi_uri,
        series_name=series_name,
        color=color,
    )
