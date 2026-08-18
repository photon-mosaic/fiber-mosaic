"""Tests for the built-in processing functions."""

from __future__ import annotations

import numpy as np
import pytest
from spikeinterface.core.numpyextractors import NumpyRecordingSegment

from fiber_mosaic.core.base import BaseFiberPhotometryExtractor
from fiber_mosaic.processing.functions import (
    apply_correction,
    bleach_correction,
    isosbestic_correction,
)

_SF = 100.0


def _recording(segments, color="green", gap=5.0):
    """Build a fiber recording from per-segment arrays, spaced in time."""
    recording = BaseFiberPhotometryExtractor(
        sampling_frequency=_SF,
        fiber_ids=["f0", "f1"],
        color=color,
        dtype="float64",
    )
    start = 0.0
    for traces in segments:
        traces = np.asarray(traces, dtype="float64")
        recording.add_segment(
            NumpyRecordingSegment(
                traces=traces, sampling_frequency=_SF, t_start=None
            )
        )
        times = start + np.arange(traces.shape[0]) / _SF
        recording.set_times(
            np.stack([times, times], axis=1),
            segment_index=recording.get_num_segments() - 1,
            with_warning=False,
        )
        start = times[-1] + gap
    return recording


# ---------------- apply_correction ----------------


def test_subtract():
    traces = np.full((3, 2), 5.0)
    baseline = np.full((3, 2), 2.0)
    np.testing.assert_allclose(
        apply_correction(traces, baseline, "subtract"), 3.0
    )


def test_divide():
    traces = np.full((3, 2), 6.0)
    baseline = np.full((3, 2), 2.0)
    np.testing.assert_allclose(
        apply_correction(traces, baseline, "divide"), 3.0
    )


def test_subtract_divide():
    traces = np.full((3, 2), 6.0)
    baseline = np.full((3, 2), 2.0)
    np.testing.assert_allclose(
        apply_correction(traces, baseline, "subtract-divide"), 2.0
    )


def test_zero_baseline_yields_nan():
    traces = np.full((3, 2), 1.0)
    baseline = np.zeros((3, 2))
    assert np.all(np.isnan(apply_correction(traces, baseline, "divide")))


def test_unknown_method_raises():
    traces = np.ones((3, 2))
    with pytest.raises(ValueError, match="unknown correction method"):
        apply_correction(traces, traces, "nope")


# ---------------- bleach_correction ----------------


def test_bleach_removes_a_polynomial_trend():
    times = np.arange(200) / _SF
    trend = 100.0 + 30.0 * np.exp(-times / times[-1] * 2.0)
    traces = np.stack([trend, trend * 1.5], axis=1)

    corrected = bleach_correction(
        _recording([traces]), degree=5, method="subtract"
    )

    # a smooth decay is captured almost entirely by a degree-5 polynomial
    assert np.abs(corrected.get_traces()).max() < 0.05


def test_bleach_fits_across_segments_with_a_gap():
    # one continuous trend in real time, sampled either side of a 5 s gap
    first_times = np.arange(120) / _SF
    second_times = first_times[-1] + 5.0 + np.arange(80) / _SF
    trend = lambda t: 50.0 + 10.0 * t  # noqa: E731
    first = np.stack([trend(first_times)] * 2, axis=1)
    second = np.stack([trend(second_times)] * 2, axis=1)

    corrected = bleach_correction(
        _recording([first, second], gap=5.0), degree=1, method="subtract"
    )

    assert corrected.get_num_segments() == 2
    for index in range(2):
        assert np.abs(corrected.get_traces(segment_index=index)).max() < 1e-6


def test_bleach_uses_per_fiber_times():
    # fiber 1 is sampled on a stretched clock, so its trend differs in time
    length = 150
    recording = BaseFiberPhotometryExtractor(
        _SF, ["f0", "f1"], "green", "float64"
    )
    slow = np.arange(length) / _SF
    fast = slow * 2.0
    traces = np.stack([1.0 + 2.0 * slow, 1.0 + 2.0 * fast], axis=1)
    recording.add_segment(
        NumpyRecordingSegment(
            traces=traces, sampling_frequency=_SF, t_start=None
        )
    )
    recording.set_times(np.stack([slow, fast], axis=1), with_warning=False)

    corrected = bleach_correction(recording, degree=1, method="subtract")
    assert np.abs(corrected.get_traces()).max() < 1e-9


def test_bleach_is_registered_and_gap_tolerant():
    assert bleach_correction.spec.contiguous is False
    assert bleach_correction.spec.input_names == ("traces",)


# ---------------- isosbestic_correction ----------------


def test_isosbestic_removes_a_scaled_reference():
    ramp = np.linspace(0.0, 1.0, 50)
    reference = np.stack([1.0 + ramp, 2.0 - ramp], axis=1)
    signal = 3.0 * reference + 7.0

    corrected = isosbestic_correction(
        _recording([signal]),
        _recording([reference], color="iso"),
        method="subtract",
    )

    assert np.abs(corrected.get_traces()).max() < 1e-9


def test_isosbestic_recovers_the_extra_component():
    ramp = np.linspace(0.0, 1.0, 400)
    reference = np.stack([1.0 + ramp, 2.0 - ramp], axis=1)
    extra = np.zeros_like(reference)
    extra[100:110] = 5.0
    signal = 3.0 * reference + 7.0 + extra

    corrected = isosbestic_correction(
        _recording([signal]),
        _recording([reference], color="iso"),
        method="subtract",
    ).get_traces()

    # the injected transient dominates the residual
    assert corrected[100:110].mean() > 4.0
    assert np.abs(np.delete(corrected, slice(100, 110), axis=0)).max() < 1.0


def test_isosbestic_keeps_the_signal_metadata():
    ramp = np.linspace(0.0, 1.0, 30)
    reference = np.stack([1.0 + ramp, 2.0 - ramp], axis=1)
    corrected = isosbestic_correction(
        _recording([2.0 * reference]),
        _recording([reference], color="iso"),
    )
    assert corrected.color == "green"


def test_isosbestic_inputs_and_primary():
    spec = isosbestic_correction.spec
    assert spec.input_names == ("signal", "reference")
    assert spec.primary_input == "signal"
