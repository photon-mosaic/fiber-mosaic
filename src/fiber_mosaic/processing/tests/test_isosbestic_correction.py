"""Pytest tests for ``fiber_mosaic.processing.isosbestic_correction``."""

from __future__ import annotations

import numpy as np
from spikeinterface.core.numpyextractors import NumpyRecordingSegment

from fiber_mosaic.core.base import BaseFiberPhotometryExtractor
from fiber_mosaic.processing import isosbestic_correction


def _make_recording(
    color: str, levels: list[float], n_segments: int = 2
) -> BaseFiberPhotometryExtractor:
    """Build a recording holding a constant level per fiber."""
    recording = BaseFiberPhotometryExtractor(
        sampling_frequency=100.0,
        fiber_ids=[f"f{index}" for index in range(len(levels))],
        color=color,
    )
    for _ in range(n_segments):
        recording.add_segment(
            NumpyRecordingSegment(
                traces=np.tile(levels, (50, 1)),
                sampling_frequency=100.0,
                t_start=None,
            )
        )
    return recording


def test_divides_signal_by_reference() -> None:
    """Each fiber's signal is divided by its reference, in every segment."""
    signal = _make_recording("green", [4.0, 9.0])
    reference = _make_recording("iso", [2.0, 3.0])
    corrected = isosbestic_correction(signal, reference)
    assert corrected.get_num_segments() == 2
    for segment_index in range(2):
        traces = corrected.get_traces(segment_index=segment_index)
        np.testing.assert_array_equal(traces, np.tile([2.0, 3.0], (50, 1)))
