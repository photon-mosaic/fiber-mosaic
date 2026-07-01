"""Single-color and multi-color fiber photometry recordings."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np
from spikeinterface.core.baserecording import BaseRecording


class BaseFiberPhotometryExtractor(BaseRecording):
    """
    Single-color fiber photometry recording.

    Each channel corresponds to one fiber, so ``channel_ids`` are the fiber
    identifiers. The ``color`` attribute records which emission/excitation
    color this recording represents (e.g. ``"green"``, ``"red"``, ``"iso"``).

    Parameters
    ----------
    sampling_frequency : float
        Nominal sampling rate in Hz. Actual per-sample times can be attached
        via ``recording.set_times(...)`` after adding a segment.
    fiber_ids : sequence
        Fiber IDs, one per channel.
    color : str
        Emission/excitation color for this recording.
    dtype : np.dtype or str
        Data type of the traces.
    """

    def __init__(
        self,
        sampling_frequency: float,
        fiber_ids: Sequence,
        color: str,
        dtype,
    ) -> None:
        """Initialize the recording with fiber IDs, color, and dtype."""
        channel_ids = list(fiber_ids)
        BaseRecording.__init__(
            self,
            sampling_frequency=sampling_frequency,
            channel_ids=channel_ids,
            dtype=dtype,
        )
        self.color = color
        self._kwargs["color"] = color
        self._kwargs["fiber_ids"] = channel_ids

    @property
    def fiber_ids(self) -> np.ndarray:
        """Fiber IDs for this recording, one per channel."""
        return np.asarray(self.get_channel_ids())

    def get_fiber_ids(self) -> np.ndarray:
        """Return :attr:`fiber_ids` (method-style accessor)."""
        return self.fiber_ids

    def get_num_fibers(self) -> int:
        """Return the number of fibers (channels) in this recording."""
        return self.get_num_channels()

    def has_fiber_times(self, segment_index: int | None = None) -> bool:
        """Return True when explicit per-fiber times are stored for segment."""
        segment_index = self._check_segment_index(segment_index)
        rs = self.segments[segment_index]
        return hasattr(rs, "_fiber_times") and rs._fiber_times is not None

    def set_times(
        self,
        times,
        segment_index: int | None = None,
        with_warning: bool = True,
    ) -> None:
        """Set per-fiber times for one segment.

        Accepts either a 1-D vector of shape ``(n_samples,)`` and broadcasts
        to all fibers, or a 2-D matrix of shape ``(n_samples, n_fibers)``.
        """
        segment_index = self._check_segment_index(segment_index)
        rs = self.segments[segment_index]

        times_arr = np.asarray(times, dtype="float64")
        n_samples = rs.get_num_samples()
        n_fibers = self.get_num_fibers()

        if times_arr.ndim == 1:
            if times_arr.shape[0] != n_samples:
                raise ValueError(
                    "1-D times must have shape (n_samples,) matching the "
                    "selected segment."
                )
            fiber_times = np.broadcast_to(
                times_arr[:, np.newaxis], (n_samples, n_fibers)
            ).copy()
        elif times_arr.ndim == 2:
            if times_arr.shape != (n_samples, n_fibers):
                raise ValueError(
                    "2-D times must have shape (n_samples, n_fibers)."
                )
            fiber_times = times_arr
        else:
            raise ValueError("times must be 1-D or 2-D.")

        rs._fiber_times = fiber_times

    def get_fiber_times(
        self,
        segment_index: int | None = None,
        start_frame: int | None = None,
        end_frame: int | None = None,
        fiber_ids: list | np.ndarray | tuple | None = None,
    ) -> np.ndarray:
        """Return per-fiber times with optional frame/fiber subsetting."""
        segment_index = self._check_segment_index(segment_index)
        rs = self.segments[segment_index]

        if self.has_fiber_times(segment_index=segment_index):
            times = rs._fiber_times
        else:
            n_samples = rs.get_num_samples()
            dt = 1.0 / self.get_sampling_frequency()
            base_times = np.arange(n_samples, dtype="float64") * dt
            times = np.broadcast_to(
                base_times[:, np.newaxis], (n_samples, self.get_num_fibers())
            )

        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = rs.get_num_samples()

        times = times[start_frame:end_frame]

        if fiber_ids is not None:
            channel_ids = np.asarray(self.get_channel_ids())
            fiber_ids = np.asarray(fiber_ids)
            channel_inds = [int(np.where(channel_ids == fid)[0][0]) for fid in fiber_ids]
            times = times[:, channel_inds]

        return times

    @classmethod
    def get_streams(cls, *args, **kwargs) -> tuple[list, list]:
        """
        List the data streams available in a source before instantiation.

        Mirrors spikeinterface's extractor discovery step: a raw fiber
        photometry file (e.g. TDT, Doric, Neurophotometrics) usually bundles
        several streams -- different excitation/LED colors, demodulated vs.
        raw signals, analog inputs, TTLs. Call this on a concrete subclass to
        see what is available, then pass the chosen ``stream_id`` /
        ``stream_name`` to that subclass's constructor.

        Parameters
        ----------
        *args, **kwargs
            Source locator (e.g. a file path) and format-specific options,
            defined by the concrete subclass.

        Returns
        -------
        stream_names : list of str
            Human-readable name for each stream.
        stream_ids : list
            Stable identifier for each stream, aligned with ``stream_names``.

        Notes
        -----
        The base extractor reads no file format, so it has no streams. Concrete
        file-reading subclasses must override this method.
        """
        raise NotImplementedError(
            f"{cls.__name__} does not read a file format and has no streams. "
            "Concrete file-reading subclasses must override get_streams() to "
            "return (stream_names, stream_ids)."
        )

    def get_fluorescence(
        self,
        segment_index: int | None = None,
        start_frame: int | None = None,
        end_frame: int | None = None,
        fiber_ids: list | np.ndarray | tuple | None = None,
    ) -> np.ndarray:
        """
        Return fluorescence traces, shape ``(n_samples, n_fibers)``.

        Fiber-photometry-native wrapper around
        :meth:`~spikeinterface.core.BaseRecording.get_traces`. Drops
        ephys-only options (``order``, ``return_scaled``, ``return_in_uV``)
        and renames ``channel_ids`` to ``fiber_ids``.

        Parameters
        ----------
        segment_index : int or None, default: None
            Segment to read from. Required for multi-segment recordings.
        start_frame : int or None, default: None
            Start sample, or 0 if None.
        end_frame : int or None, default: None
            End sample, or num_samples if None.
        fiber_ids : sequence or None, default: None
            Subset of fibers to return. If None, all fibers are returned.
        """
        return self.get_traces(
            segment_index=segment_index,
            start_frame=start_frame,
            end_frame=end_frame,
            channel_ids=fiber_ids,
        )

    def __repr__(self) -> str:
        """Return a one-line summary: class, color, fiber/segment count."""
        n_seg = self.get_num_segments()
        n_fib = self.get_num_fibers()
        sf = self.get_sampling_frequency()
        dtype = self.get_dtype()
        return (
            f"{self.__class__.__name__} | color={self.color} | "
            f"{n_fib} fiber(s) | {n_seg} segment(s) | "
            f"{sf:.1f} Hz | dtype: {dtype}"
        )


class FiberPhotometryRecordingGroup:
    """
    Container of per-color fiber photometry recordings sharing a fiber set.

    Each inner recording is a :class:`BaseFiberPhotometryExtractor` (or any
    :class:`~spikeinterface.core.BaseRecording`) where channels are fibers.
    The group itself is *not* a ``BaseRecording`` — colors can have different
    time bases, so there is no single set of traces to return.

    Parameters
    ----------
    recordings : dict[str, BaseFiberPhotometryExtractor]
        Mapping from color to per-color recording. All recordings must share
        the same ``channel_ids`` (fiber IDs) in the same order.
    """

    def __init__(self, recordings: dict[str, BaseRecording]) -> None:
        """Store per-color recordings; validate they share the same fibers."""
        if not recordings:
            raise ValueError("recordings must be non-empty")

        self._recordings: dict[str, BaseRecording] = dict(recordings)

        fiber_sets = {
            color: tuple(rec.get_fiber_ids())
            for color, rec in self._recordings.items()
        }
        unique = set(fiber_sets.values())
        if len(unique) > 1:
            raise ValueError(
                f"Fiber IDs differ across colors: {fiber_sets}. "
                "All per-color recordings must share the same fibers."
            )

    @property
    def colors(self) -> list[str]:
        """List of color labels in insertion order."""
        return list(self._recordings.keys())

    @property
    def fiber_ids(self) -> np.ndarray:
        """Shared fiber IDs across all colors."""
        first = next(iter(self._recordings.values()))
        return np.asarray(first.get_channel_ids())

    def get_num_fibers(self) -> int:
        """Return the number of fibers shared across all colors."""
        return len(self.fiber_ids)

    def get_recording(self, color: str) -> BaseRecording:
        """Return the per-color recording for ``color``."""
        return self._recordings[color]

    def __getitem__(self, color: str) -> BaseRecording:
        """Return the per-color recording via ``group[color]``."""
        return self._recordings[color]

    def __contains__(self, color: str) -> bool:
        """Return True if ``color`` is one of the recordings in this group."""
        return color in self._recordings

    def __iter__(self) -> Iterator[str]:
        """Iterate over color labels."""
        return iter(self._recordings)

    def __len__(self) -> int:
        """Return the number of colors in this group."""
        return len(self._recordings)

    def items(self):
        """Return ``(color, recording)`` pairs (dict-style view)."""
        return self._recordings.items()

    def values(self):
        """Return the per-color recordings (dict-style view)."""
        return self._recordings.values()

    def keys(self):
        """Return the color labels (dict-style view)."""
        return self._recordings.keys()

    def __repr__(self) -> str:
        """Return a multi-line summary listing each per-color recording."""
        lines = [
            f"{self.__class__.__name__} | "
            f"{self.get_num_fibers()} fiber(s), {len(self)} color(s)"
        ]
        for color, rec in self._recordings.items():
            lines.append(f"  [{color}] {rec!r}")
        return "\n".join(lines)
