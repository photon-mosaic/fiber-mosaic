"""Tests for the step registry and contributor helpers."""

from __future__ import annotations

import numpy as np
import pytest
from spikeinterface.core import NumpyRecording
from spikeinterface.core.numpyextractors import NumpyRecordingSegment

from fiber_mosaic.core.base import BaseFiberPhotometryExtractor
from fiber_mosaic.processing.helpers import map_segments, recording_like
from fiber_mosaic.processing.registry import (
    get_step,
    pipeline_step,
    register_step,
    registered_steps,
)

_SF = 100.0
_N = 10


def _make_recording(color="green", n_segments=1):
    """Build a fiber recording of ones with the given segment count."""
    rec = BaseFiberPhotometryExtractor(
        sampling_frequency=_SF,
        fiber_ids=["f0", "f1"],
        color=color,
        dtype="float64",
    )
    for _ in range(n_segments):
        rec.add_segment(
            NumpyRecordingSegment(
                traces=np.ones((_N, 2), dtype="float64"),
                sampling_frequency=_SF,
                t_start=None,
            )
        )
    return rec


# ---------------- registry ----------------


def test_register_and_get_step():
    def step(recording):
        return recording

    register_step("reg_plain", step)
    assert get_step("reg_plain") is step
    assert "reg_plain" in registered_steps()


def test_pipeline_step_decorator_registers_and_returns():
    @pipeline_step("reg_decorated", by_segment=False)
    def step(recording):
        return recording

    assert get_step("reg_decorated") is step
    assert step(_make_recording()) is not None


def test_register_duplicate_raises():
    def step(recording):
        return recording

    register_step("reg_dup", step)
    with pytest.raises(ValueError):
        register_step("reg_dup", step)


def test_register_overwrite_replaces():
    def first(recording):
        return recording

    def second(recording):
        return recording

    register_step("reg_ovr", first)
    register_step("reg_ovr", second, overwrite=True)
    assert get_step("reg_ovr") is second


def test_get_unknown_step_raises():
    with pytest.raises(KeyError):
        get_step("reg_definitely_missing")


def test_registered_steps_sorted():
    steps = registered_steps()
    assert steps == sorted(steps)


# ---------------- helpers ----------------


def test_recording_like_single_array():
    ref = _make_recording()
    new = recording_like(ref, np.zeros((_N, 2)))
    assert new.get_num_segments() == 1
    assert new.color == "green"
    np.testing.assert_array_equal(new.get_fiber_ids(), ref.get_fiber_ids())
    np.testing.assert_array_equal(new.get_traces(), np.zeros((_N, 2)))


def test_recording_like_multi_segment_list():
    ref = _make_recording(n_segments=2)
    new = recording_like(ref, [np.zeros((_N, 2)), np.ones((_N, 2))])
    assert new.get_num_segments() == 2
    np.testing.assert_array_equal(
        new.get_traces(segment_index=1), np.ones((_N, 2))
    )


def test_recording_like_explicit_color():
    ref = _make_recording()
    new = recording_like(ref, np.zeros((_N, 2)), color="custom")
    assert new.color == "custom"


def test_recording_like_color_from_annotation():
    ref = NumpyRecording([np.zeros((_N, 2))], sampling_frequency=_SF)
    ref.annotate(color="teal")
    new = recording_like(ref, np.zeros((_N, 2)))
    assert new.color == "teal"


def test_recording_like_carries_fiber_times():
    ref = _make_recording()
    times = np.linspace(0.0, 1.0, _N)
    ref.set_times(times)
    new = recording_like(ref, np.zeros((_N, 2)))
    assert new.has_fiber_times()
    np.testing.assert_array_equal(new.get_fiber_times()[:, 0], times)


def test_map_segments_applies_per_segment():
    ref = _make_recording(n_segments=2)
    out = map_segments(ref, lambda traces: traces * 2.0)
    np.testing.assert_array_equal(
        out.get_traces(segment_index=0), np.ones((_N, 2)) * 2.0
    )
    assert out.get_num_segments() == 2


def test_map_segments_forwards_params():
    ref = _make_recording()
    out = map_segments(ref, lambda traces, k: traces * k, k=3.0)
    np.testing.assert_array_equal(out.get_traces(), np.ones((_N, 2)) * 3.0)
