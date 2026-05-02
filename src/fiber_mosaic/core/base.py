"""Single-color and multi-color fiber photometry recordings."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np
from spikeinterface.core.baserecording import (
    BaseRecording,
    BaseRecordingSegment,
)


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

    def add_segment(self, segment: BaseRecordingSegment) -> None:
        """Attach a contiguous block of fluorescence data to this recording."""
        self.add_recording_segment(segment)

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


class BaseFiberPhotometrySegment(BaseRecordingSegment):
    """
    One contiguous block of single-color fiber photometry data.

    Subclasses implement ``get_traces`` and ``get_num_samples`` per the
    ``BaseRecordingSegment`` contract (spikeinterface internals call
    ``get_traces`` directly). Traces are 2D ``(n_samples, n_fibers)``.

    User-facing code should prefer :meth:`get_fluorescence`, which uses
    fiber-photometry-native parameter names.
    """

    def get_fluorescence(
        self,
        start_frame: int | None = None,
        end_frame: int | None = None,
        fiber_indices: list | np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Return fluorescence for this segment, shape ``(n_samples, n_fibers)``.

        Thin wrapper around :meth:`get_traces` that renames
        ``channel_indices`` to ``fiber_indices``.
        """
        return self.get_traces(
            start_frame=start_frame,
            end_frame=end_frame,
            channel_indices=fiber_indices,
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
    recordings : dict[str, BaseRecording]
        Mapping from color to per-color recording. All recordings must share
        the same ``channel_ids`` (fiber IDs) in the same order.
    """

    def __init__(self, recordings: dict[str, BaseRecording]) -> None:
        """Store per-color recordings; validate they share the same fibers."""
        if not recordings:
            raise ValueError("recordings must be non-empty")

        self._recordings: dict[str, BaseRecording] = dict(recordings)

        fiber_sets = {
            color: tuple(rec.get_channel_ids())
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
