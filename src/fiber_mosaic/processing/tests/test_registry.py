"""Tests for step registration, introspection and the runner."""

from __future__ import annotations

import numpy as np
import pytest
from spikeinterface.core import NumpyRecording
from spikeinterface.core.numpyextractors import NumpyRecordingSegment

from fiber_mosaic.core.base import BaseFiberPhotometryExtractor
from fiber_mosaic.processing.registry import (
    _introspect_signature,
    concatenate_segments,
    get_step,
    processing_step,
    register_step,
    registered_steps,
    validate_timeline,
)

_SF = 100.0


def _make_recording(lengths=(10,), color="green", gap=10.0, with_times=True):
    """Build a fiber recording whose segments are separated in time."""
    recording = BaseFiberPhotometryExtractor(
        sampling_frequency=_SF,
        fiber_ids=["f0", "f1"],
        color=color,
        dtype="float64",
    )
    start = 0.0
    for length in lengths:
        recording.add_segment(
            NumpyRecordingSegment(
                traces=np.ones((length, 2), dtype="float64"),
                sampling_frequency=_SF,
                t_start=None,
            )
        )
        if with_times:
            times = start + np.arange(length) / _SF
            recording.set_times(
                np.stack([times, times + 0.002], axis=1),
                segment_index=recording.get_num_segments() - 1,
                with_warning=False,
            )
            start = times[-1] + gap
    return recording


# ---------------- signature introspection ----------------


def test_introspect_derives_primary_and_extras():
    def step(signal, times, reference, *, method="x"):
        return signal

    names, params = _introspect_signature(step)
    assert names == ("signal", "reference")
    assert params == {"method"}
    assert step(1, 2, 3) == 1


def test_introspect_rejects_var_args():
    def step(traces, times, *extra, **params):
        return traces

    with pytest.raises(TypeError):
        _introspect_signature(step)
    assert step(1, 2) == 1


def test_introspect_requires_primary_and_times():
    def step(traces):
        return traces

    with pytest.raises(TypeError):
        _introspect_signature(step)
    assert step(1) == 1


# ---------------- registry ----------------


def test_register_step_and_get_step():
    def step(recording):
        return recording

    register_step("reg_plain", step)
    spec = get_step("reg_plain")
    assert spec.function is step
    assert spec.input_names == ("recording",)
    assert "reg_plain" in registered_steps()
    recording = _make_recording()
    assert spec.run({"recording": recording}, {})["result"] is recording


def test_register_duplicate_raises():
    register_step("reg_dup", lambda recording: recording)
    with pytest.raises(ValueError):
        register_step("reg_dup", lambda recording: recording)


def test_register_overwrite_replaces():
    def second(recording):
        return recording

    register_step("reg_ovr", lambda recording: recording)
    register_step("reg_ovr", second, overwrite=True)
    spec = get_step("reg_ovr")
    assert spec.function is second
    recording = _make_recording()
    assert spec.run({"recording": recording}, {})["result"] is recording


def test_get_unknown_step_raises():
    with pytest.raises(KeyError):
        get_step("reg_definitely_missing")


def test_registered_steps_sorted():
    names = registered_steps()
    assert names == sorted(names)


def test_recording_step_runs_callable():
    register_step("reg_passthrough", lambda recording: recording)
    spec = get_step("reg_passthrough")
    recording = _make_recording()
    assert spec.run({"recording": recording}, {})["result"] is recording


# ---------------- gap-tolerant vs contiguous ----------------


def test_gap_tolerant_sees_all_segments_at_once():
    seen = {}

    @processing_step("run_all")
    def step(traces, times):
        seen["samples"] = traces.shape[0]
        seen["times"] = times.shape
        return traces * 2.0

    recording = _make_recording((5, 3))
    result = step(recording)

    assert seen["samples"] == 8
    assert seen["times"] == (8, 2)
    assert result.get_num_segments() == 2
    assert [result.get_num_samples(i) for i in range(2)] == [5, 3]
    np.testing.assert_array_equal(
        result.get_traces(segment_index=1), np.ones((3, 2)) * 2.0
    )


def test_contiguous_runs_once_per_segment():
    calls = []

    @processing_step("run_seg", contiguous=True)
    def step(traces, times):
        calls.append(traces.shape[0])
        return traces

    step(_make_recording((5, 3)))
    assert calls == [5, 3]


def test_gap_tolerant_preserves_per_fiber_times():
    @processing_step("run_times")
    def step(traces, times):
        return traces

    recording = _make_recording((5, 3))
    result = step(recording)
    for index in range(2):
        np.testing.assert_allclose(
            result.get_fiber_times(segment_index=index),
            recording.get_fiber_times(segment_index=index),
        )


def test_gap_tolerant_times_span_the_gap():
    spans = {}

    @processing_step("run_span")
    def step(traces, times):
        spans["value"] = float(times[-1, 0] - times[0, 0])
        return traces

    step(_make_recording((5, 3), gap=10.0))
    # 4 samples of the first segment, a 10 s gap, then 2 more samples
    assert spans["value"] == pytest.approx(0.04 + 10.0 + 0.02)


# ---------------- padding ----------------


def test_margin_pads_then_trims():
    widths = []

    @processing_step("run_pad", contiguous=True, margin=2)
    def step(traces, times):
        widths.append((traces.shape[0], times.shape[0]))
        return traces

    result = step(_make_recording((6, 4)))
    assert widths == [(10, 10), (8, 8)]
    assert [result.get_num_samples(i) for i in range(2)] == [6, 4]


def test_margin_times_are_extrapolated():
    seen = {}

    @processing_step("run_pad_times", contiguous=True, margin=2)
    def step(traces, times):
        seen["times"] = times[:, 0].copy()
        return traces

    step(_make_recording((6,)))
    times = seen["times"]
    # the leading samples continue the 1 / _SF spacing backwards
    assert times[0] == pytest.approx(-0.02)
    assert times[1] == pytest.approx(-0.01)
    assert times[-1] == pytest.approx(0.07)


def test_margin_larger_than_segment_raises():
    @processing_step("run_pad_big", contiguous=True, margin=20)
    def step(traces, times):
        return traces  # pragma: no cover - the margin check raises first

    with pytest.raises(ValueError, match="margin"):
        step(_make_recording((5,)))


# ---------------- outputs ----------------


def test_multi_output_dict_becomes_named_recordings():
    @processing_step("run_multi")
    def step(traces, times):
        return {"doubled": traces * 2.0, "halved": traces / 2.0}

    results = step(_make_recording((4,)))
    assert sorted(results) == ["doubled", "halved"]
    np.testing.assert_array_equal(
        results["halved"].get_traces(), np.ones((4, 2)) / 2.0
    )


def test_length_changing_output_raises():
    @processing_step("run_shrink")
    def step(traces, times):
        return traces[:-1]

    with pytest.raises(ValueError, match="preserve length"):
        step(_make_recording((5,)))


def test_length_changing_output_per_segment_raises():
    @processing_step("run_shrink_seg", contiguous=True)
    def step(traces, times):
        return traces[:-1]

    with pytest.raises(ValueError, match="preserve length"):
        step(_make_recording((5,)))


def test_bad_return_type_raises():
    @processing_step("run_bad_return")
    def step(traces, times):
        return 42

    with pytest.raises(TypeError):
        step(_make_recording((4,)))


# ---------------- inputs ----------------


@processing_step("run_pair")
def _pair(signal, times, reference):
    """A two-input step, shared by the success and error cases below."""
    return signal + reference


def test_extra_input_is_passed_positionally():
    signal = _make_recording((4,))
    reference = _make_recording((4,), color="iso")
    result = _pair(signal, reference)
    np.testing.assert_array_equal(result.get_traces(), np.ones((4, 2)) * 2.0)
    # metadata comes from the primary input
    assert result.color == "green"


def test_direct_call_with_wrong_arity_raises():
    with pytest.raises(TypeError, match="recording"):
        _pair(_make_recording((4,)))


def test_missing_input_raises():
    spec = get_step("run_pair")
    with pytest.raises(ValueError, match="expects inputs"):
        spec.run({"signal": _make_recording((4,))}, {})


def test_incompatible_segment_count_raises():
    with pytest.raises(ValueError, match="segment count"):
        _pair(_make_recording((4, 4)), _make_recording((4,), color="iso"))


def test_incompatible_sample_count_raises():
    with pytest.raises(ValueError, match="sample counts"):
        _pair(_make_recording((4,)), _make_recording((5,), color="iso"))


def test_incompatible_fiber_count_raises():
    other = BaseFiberPhotometryExtractor(_SF, ["a"], "iso", "float64")
    other.add_segment(
        NumpyRecordingSegment(
            np.ones((4, 1)), sampling_frequency=_SF, t_start=None
        )
    )
    with pytest.raises(ValueError, match="fiber count"):
        _pair(_make_recording((4,)), other)


# ---------------- timeline validation ----------------


def test_validate_timeline_accepts_ordered_segments():
    validate_timeline(_make_recording((5, 3)))


def test_validate_timeline_accepts_single_segment():
    validate_timeline(_make_recording((5,), with_times=False))


def test_validate_timeline_rejects_missing_clock():
    with pytest.raises(ValueError, match="disjoint"):
        validate_timeline(_make_recording((5, 3), with_times=False))


def test_gap_tolerant_step_validates_the_timeline():
    @processing_step("run_needs_clock")
    def step(traces, times):
        return traces  # pragma: no cover - the timeline check raises first

    with pytest.raises(ValueError, match="disjoint"):
        step(_make_recording((5, 3), with_times=False))


def test_contiguous_step_does_not_need_a_clock():
    @processing_step("run_no_clock", contiguous=True)
    def step(traces, times):
        return traces

    result = step(_make_recording((5, 3), with_times=False))
    assert result.get_num_segments() == 2


# ---------------- concatenate_segments ----------------


def test_concatenate_segments_carries_metadata():
    reference = _make_recording((4,))
    new = concatenate_segments(reference, [np.zeros((4, 2))])
    assert new.color == "green"
    np.testing.assert_array_equal(
        new.get_fiber_ids(), reference.get_fiber_ids()
    )
    np.testing.assert_allclose(
        new.get_fiber_times(), reference.get_fiber_times()
    )


def test_concatenate_segments_color_from_annotation():
    reference = NumpyRecording([np.zeros((4, 2))], sampling_frequency=_SF)
    reference.annotate(color="teal")
    assert concatenate_segments(reference, [np.zeros((4, 2))]).color == "teal"


def test_plain_recording_times_are_broadcast():
    @processing_step("run_plain_times")
    def step(traces, times):
        assert times.shape == traces.shape
        return traces

    plain = NumpyRecording([np.ones((4, 2))], sampling_frequency=_SF)
    plain.annotate(color="teal")
    assert step(plain).get_num_samples() == 4
