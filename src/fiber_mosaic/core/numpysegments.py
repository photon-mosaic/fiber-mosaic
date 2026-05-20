"""In-memory, numpy-backed fiber photometry segments."""

from __future__ import annotations

import numpy as np

from .base import BaseFiberPhotometrySegment


class NumpyFiberPhotometrySegment(BaseFiberPhotometrySegment):
    """
    In-memory fiber photometry segment backed by a numpy array.

    Wraps a 2-D ``(n_samples, n_fibers)`` array so it can be attached to a
    :class:`~fiber_mosaic.core.base.BaseFiberPhotometryExtractor`. Handy for
    synthetic data, tests, and arrays already loaded into memory; file-format
    readers should provide their own segment that streams from disk.

    Parameters
    ----------
    traces : np.ndarray
        2-D fluorescence array, shape ``(n_samples, n_fibers)``.
    sampling_frequency : float
        Sampling rate in Hz, shared with the parent recording.
    t_start : float or None, default: None
        Start time of the segment in seconds, if known.
    """

    def __init__(
        self,
        traces: np.ndarray,
        sampling_frequency: float,
        t_start: float | None = None,
    ) -> None:
        """Wrap a 2-D ``(n_samples, n_fibers)`` array as a segment."""
        super().__init__(
            sampling_frequency=sampling_frequency, t_start=t_start
        )
        traces = np.asarray(traces)
        if traces.ndim != 2:
            raise ValueError(
                "traces must be 2-D (n_samples, n_fibers), got "
                f"{traces.ndim}-D"
            )
        self._traces = traces

    def get_traces(
        self,
        start_frame: int | None = None,
        end_frame: int | None = None,
        channel_indices: list | np.ndarray | slice | None = None,
    ) -> np.ndarray:
        """Return an optionally sliced view of the backing array."""
        traces = self._traces[start_frame:end_frame]
        if channel_indices is not None:
            traces = traces[:, channel_indices]
        return traces

    def get_num_samples(self) -> int:
        """Return the number of time samples in the backing array."""
        return self._traces.shape[0]
