"""
Core classes for fiber photometry data handling.

This module provides the base classes that form the foundation of fiber-mosaic:

- BaseFiberPhotometryExtractor: A per-color recording (wraps SI BaseRecording)
- FiberPhotometryRecordingGroup: A container for multiple colors sharing fibers
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
from numpy.typing import ArrayLike
from spikeinterface.core import BaseRecording


class BaseFiberPhotometryExtractor(BaseRecording):
    """
    Base class for fiber photometry recordings.

    This class represents a single-color recording from a fiber photometry
    experiment. It wraps SpikeInterface's BaseRecording and provides
    fiber-photometry-specific vocabulary and methods:

    - Channels are called "fibers"
    - `get_fluorescence()` wraps `get_traces()` with fiber-native naming
    - Per-fiber timestamps via `set_times()` and `get_fiber_times()`
    - `get_streams()` for format-specific discovery of available data streams

    Each color in an experiment should be its own BaseFiberPhotometryExtractor
    instance, grouped together via FiberPhotometryRecordingGroup.

    Parameters
    ----------
    sampling_frequency : float
        The nominal sampling frequency in Hz.
    fiber_ids : list or array-like
        Identifiers for each fiber (will be used as channel_ids internally).
    color : str
        The color/wavelength identifier for this recording
        (e.g., "green", "red", "iso").
    dtype : dtype, optional
        The data type of the traces. Default is float64.
    """

    def __init__(
        self,
        sampling_frequency: float,
        fiber_ids: Sequence,
        color: str,
        dtype: np.dtype = np.float64,
    ):
        # Pass fiber_ids as channel_ids to SI's BaseRecording
        BaseRecording.__init__(
            self,
            sampling_frequency=sampling_frequency,
            channel_ids=list(fiber_ids),
            dtype=dtype,
        )
        self._color = color
        # Store kwargs for serialization
        self._kwargs = {
            "sampling_frequency": sampling_frequency,
            "fiber_ids": list(fiber_ids),
            "color": color,
            "dtype": str(dtype),
        }

    @property
    def color(self) -> str:
        """Return the color/wavelength identifier for this recording."""
        return self._color

    @property
    def fiber_ids(self) -> np.ndarray:
        """Return the fiber identifiers (alias for channel_ids)."""
        return self.get_channel_ids()

    def get_fiber_ids(self) -> np.ndarray:
        """Return the fiber identifiers (alias for get_channel_ids)."""
        return self.get_channel_ids()

    def get_num_fibers(self) -> int:
        """Return the number of fibers (alias for get_num_channels)."""
        return self.get_num_channels()

    def get_fluorescence(
        self,
        segment_index: int | None = None,
        start_frame: int | None = None,
        end_frame: int | None = None,
        fiber_ids: Sequence | None = None,
    ) -> np.ndarray:
        """
        Get fluorescence traces for specified fibers and time range.

        This is a fiber-native wrapper around `get_traces()` that uses
        fiber-photometry vocabulary.

        Parameters
        ----------
        segment_index : int, optional
            The segment index. If None and only one segment exists,
            defaults to 0.
        start_frame : int, optional
            The start frame. If None, starts from the beginning.
        end_frame : int, optional
            The end frame (exclusive). If None, reads to the end.
        fiber_ids : list or array-like, optional
            The fiber IDs to retrieve. If None, returns all fibers.

        Returns
        -------
        traces : np.ndarray
            Array of shape (n_samples, n_fibers) containing the
            fluorescence data.
        """
        return self.get_traces(
            segment_index=segment_index,
            start_frame=start_frame,
            end_frame=end_frame,
            channel_ids=fiber_ids,
        )

    def set_times(
        self,
        times: ArrayLike,
        segment_index: int | None = None,
    ) -> None:
        """
        Set per-fiber timestamps for this recording.

        Unlike SpikeInterface's set_times (which sets a single shared time
        vector), this method supports per-fiber timestamps stored as a 2D
        array.

        Parameters
        ----------
        times : array-like
            Time values in seconds. Can be:
            - 1D array of shape (n_samples,): broadcast to all fibers
            - 2D array of shape (n_samples, n_fibers): one column per fiber
        segment_index : int, optional
            The segment index. If None and only one segment exists,
            defaults to 0.
        """
        segment_index = self._check_segment_index(segment_index)
        rs = self._recording_segments[segment_index]

        times = np.asarray(times)
        n_samples = self.get_num_samples(segment_index)
        n_fibers = self.get_num_fibers()

        # Validate and reshape times
        if times.ndim == 1:
            if len(times) != n_samples:
                raise ValueError(
                    f"1D times array length ({len(times)}) must match "
                    f"number of samples ({n_samples})"
                )
            # Broadcast 1D to 2D (n_samples, n_fibers)
            broadcast_shape = (n_samples, n_fibers)
            times = np.broadcast_to(
                times[:, np.newaxis], broadcast_shape
            ).copy()
        elif times.ndim == 2:
            if times.shape != (n_samples, n_fibers):
                raise ValueError(
                    f"2D times array shape {times.shape} must match "
                    f"(n_samples={n_samples}, n_fibers={n_fibers})"
                )
        else:
            msg = f"times must be 1D or 2D array, got {times.ndim}D"
            raise ValueError(msg)

        # Store per-fiber times on the segment
        rs.fiber_time_vectors = times

    def has_fiber_times(self, segment_index: int | None = None) -> bool:
        """
        Check if per-fiber timestamps have been set.

        Parameters
        ----------
        segment_index : int, optional
            The segment index. If None and only one segment exists,
            defaults to 0.

        Returns
        -------
        bool
            True if per-fiber timestamps are available.
        """
        segment_index = self._check_segment_index(segment_index)
        rs = self._recording_segments[segment_index]
        has_attr = hasattr(rs, "fiber_time_vectors")
        return has_attr and rs.fiber_time_vectors is not None

    def get_fiber_times(
        self,
        segment_index: int | None = None,
        start_frame: int | None = None,
        end_frame: int | None = None,
        fiber_ids: Sequence | None = None,
    ) -> np.ndarray:
        """
        Get per-fiber timestamps.

        If per-fiber times were set via set_times(), returns those. Otherwise,
        falls back to the nominal SI timestamps and broadcasts them to all
        fibers.

        Parameters
        ----------
        segment_index : int, optional
            The segment index. If None and only one segment exists,
            defaults to 0.
        start_frame : int, optional
            The start frame. If None, starts from the beginning.
        end_frame : int, optional
            The end frame (exclusive). If None, reads to the end.
        fiber_ids : list or array-like, optional
            The fiber IDs to retrieve. If None, returns all fibers.

        Returns
        -------
        times : np.ndarray
            Array of shape (n_samples, n_fibers) containing timestamps.
        """
        segment_index = self._check_segment_index(segment_index)
        rs = self._recording_segments[segment_index]

        # Resolve frame range
        n_samples = self.get_num_samples(segment_index)
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = n_samples

        # Resolve fiber indices
        if fiber_ids is None:
            fiber_indices = slice(None)
        else:
            fiber_indices = self.ids_to_indices(fiber_ids)

        # Get per-fiber times if available, else fall back to nominal
        if self.has_fiber_times(segment_index):
            return rs.fiber_time_vectors[start_frame:end_frame, fiber_indices]
        else:
            # Fall back to SI's nominal timestamps, broadcast to all fibers
            times_1d = self.get_times(segment_index=segment_index)
            times_1d = times_1d[start_frame:end_frame]
            if fiber_ids is not None:
                n_fibers = len(fiber_ids)
            else:
                n_fibers = self.get_num_fibers()
            return np.broadcast_to(
                times_1d[:, np.newaxis], (len(times_1d), n_fibers)
            ).copy()

    @classmethod
    def get_streams(cls, file_path: str) -> tuple[list[str], list[str]]:
        """
        Discover available streams in a file.

        This is a discovery hook that file-format-specific readers should
        override. Returns the available stream names and IDs so users can
        choose which to load.

        Parameters
        ----------
        file_path : str
            Path to the file or folder to inspect.

        Returns
        -------
        stream_names : list of str
            Human-readable names of available streams.
        stream_ids : list of str
            Identifiers to pass to the reader's constructor.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.
        """
        raise NotImplementedError(
            f"{cls.__name__}.get_streams() must be implemented by file readers"
        )

    def add_segment(self, segment) -> None:
        """
        Add a segment to this recording.

        This is a thin wrapper around BaseExtractor.add_segment for API
        consistency.

        Parameters
        ----------
        segment : BaseRecordingSegment
            The segment to add.
        """
        self._recording_segments.append(segment)
        segment.set_parent_extractor(self)


class FiberPhotometryRecordingGroup:
    """
    A container for multiple fiber photometry recordings sharing fibers.

    This class groups several per-color recordings
    (BaseFiberPhotometryExtractor instances) that share the same fiber
    layout but may have different timebases or sample counts.

    This is NOT a BaseRecording because the colors can have different
    timebases, so there's no single trace matrix to return.

    Parameters
    ----------
    recordings : dict
        A dictionary mapping color names to
        BaseFiberPhotometryExtractor instances. All recordings must have
        identical fiber_ids in the same order.

    Examples
    --------
    >>> green = CsvFiberPhotometryExtractor("green.csv", color="green")
    >>> iso = CsvFiberPhotometryExtractor("iso.csv", color="iso")
    >>> group = FiberPhotometryRecordingGroup({"green": green, "iso": iso})
    >>> group.colors
    ['green', 'iso']
    >>> group["green"].get_fluorescence()
    """

    def __init__(self, recordings: dict[str, BaseRecording]):
        if not recordings:
            raise ValueError("recordings dict cannot be empty")

        self._recordings = dict(recordings)
        self._validate_fiber_ids()

    def _validate_fiber_ids(self) -> None:
        """Validate that all recordings have identical fiber_ids."""
        fiber_ids_list = [
            tuple(rec.get_channel_ids()) for rec in self._recordings.values()
        ]
        reference = fiber_ids_list[0]
        for color, fiber_ids in zip(
            self._recordings.keys(), fiber_ids_list, strict=True
        ):
            if fiber_ids != reference:
                raise ValueError(
                    f"Recording '{color}' has different fiber_ids "
                    f"than the first recording. All recordings must "
                    f"share identical fiber_ids in the same order."
                )

    @property
    def colors(self) -> list[str]:
        """Return the list of color names in this group."""
        return list(self._recordings.keys())

    @property
    def fiber_ids(self) -> np.ndarray:
        """Return the shared fiber identifiers."""
        first_rec = next(iter(self._recordings.values()))
        return first_rec.get_channel_ids()

    def get_num_fibers(self) -> int:
        """Return the number of fibers shared by all recordings."""
        return len(self.fiber_ids)

    def get_recording(self, color: str) -> BaseRecording:
        """
        Get the recording for a specific color.

        Parameters
        ----------
        color : str
            The color name.

        Returns
        -------
        BaseRecording
            The recording for the specified color.
        """
        return self._recordings[color]

    # Dict-like access
    def __getitem__(self, color: str) -> BaseRecording:
        """Return the recording for a specific color."""
        return self._recordings[color]

    def __contains__(self, color: str) -> bool:
        """Check if a color exists in this group."""
        return color in self._recordings

    def __len__(self) -> int:
        """Return the number of colors in this group."""
        return len(self._recordings)

    def __iter__(self) -> Iterable[str]:
        """Iterate over color names in this group."""
        return iter(self._recordings)

    def keys(self):
        """Return the color names in this group."""
        return self._recordings.keys()

    def values(self):
        """Return the recordings in this group."""
        return self._recordings.values()

    def items(self):
        """Return (color, recording) pairs in this group."""
        return self._recordings.items()

    def __repr__(self) -> str:
        """Return a string representation of this group."""
        n_colors = len(self._recordings)
        n_fibers = self.get_num_fibers()
        colors = ", ".join(self.colors)
        return (
            f"FiberPhotometryRecordingGroup: "
            f"{n_colors} colors ({colors}) - "
            f"{n_fibers} fibers"
        )
