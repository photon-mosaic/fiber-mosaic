"""Pytest tests for ``fiber_mosaic.processing.motion_correction``."""

from __future__ import annotations

import sys

import numpy as np
import pytest
from spikeinterface.core.base import BaseExtractor

pytest.importorskip("aind_ophys_utils")

from fiber_mosaic.core.base import (  # noqa: E402
    FiberPhotometryRecordingGroup,
    recording_from_traces,
)
from fiber_mosaic.generation import calcium_transients  # noqa: E402
from fiber_mosaic.processing import center, motion_correction  # noqa: E402

motion_module = sys.modules["fiber_mosaic.processing.motion_correction"]

_FS = 20.0
_N = 24000
#: True motion coefficients of the two fibers; a negative one is clipped to 0.
_COEFFICIENTS = np.array([0.6, -0.5])
_NOISE = 0.01


def _recordings(seed=0):
    """Return signal and reference dF/F recordings and the true calcium."""
    rng = np.random.default_rng(seed)
    times = np.arange(_N) / _FS
    motion = np.stack(
        [
            sum(
                0.03 * np.sin(2 * np.pi * f * times + rng.uniform(0, 6.3))
                for f in (0.004, 0.011, 0.023)
            )
            for _ in _COEFFICIENTS
        ],
        axis=1,
    )
    calcium = np.stack(
        [
            calcium_transients(
                times, rng, num_events=_N // 60, decay=0.25, amplitude=0.4
            )
            for _ in _COEFFICIENTS
        ],
        axis=1,
    )
    noise = rng.normal(0.0, _NOISE, (2, _N, len(_COEFFICIENTS)))
    signal = calcium + _COEFFICIENTS * motion + noise[0]
    reference = motion + noise[1]
    return (
        recording_from_traces(
            signal.astype("float32"), name="green", sampling_frequency=_FS
        ),
        recording_from_traces(
            reference.astype("float32"), name="iso", sampling_frequency=_FS
        ),
        calcium,
    )


@pytest.fixture(scope="module")
def simulated():
    """Signal, reference and the corrected signal, with the true calcium."""
    signal, reference, calcium = _recordings()
    return signal, reference, motion_correction(signal, reference), calcium


def test_removes_motion_keeps_calcium(simulated):
    """The coefficient is found and the residual drops to the noise floor.

    An anti-correlated fiber gets coefficient 0 and is left alone. Slicing
    frames and fibers agrees with slicing the full output.
    """
    signal, _, corrected, calcium = simulated
    coefficients = corrected._kwargs["coefficients"][0]
    assert coefficients[0] == pytest.approx(0.6, abs=0.05)
    assert coefficients[1] == 0.0
    before = signal.get_traces()[:, 0] - calcium[:, 0]
    after = corrected.get_traces()[:, 0] - calcium[:, 0]
    floor = _NOISE * np.hypot(1, _COEFFICIENTS[0])  # signal + scaled iso noise
    assert before.std() > 2 * floor
    assert after.std() < 1.1 * floor
    np.testing.assert_array_equal(  # clipped fiber is left alone
        corrected.get_traces()[:, 1], signal.get_traces()[:, 1]
    )
    np.testing.assert_allclose(  # frame and fiber subsets
        corrected.get_traces(
            start_frame=100, end_frame=500, channel_ids=["fiber_0"]
        ),
        corrected.get_traces()[100:500, [0]],
    )


@pytest.mark.parametrize(
    "container, reference_name",
    [(dict, "iso"), (FiberPhotometryRecordingGroup, "uv")],
)
def test_container_is_corrected_with_its_reference(
    simulated, container, reference_name
):
    """Everything but the named reference is corrected, in its container."""
    signal, reference, single, _ = simulated

    def renamed(recording, name):
        return recording_from_traces(
            recording.get_traces(), name=name, sampling_frequency=_FS
        )

    recordings = container(
        {
            "green": renamed(signal, "green"),
            "red": renamed(signal, "red"),
            reference_name: renamed(reference, reference_name),
        }
    )
    kwargs = {} if reference_name == "iso" else {"reference": reference_name}

    corrected = motion_correction(recordings, **kwargs)

    assert isinstance(corrected, container)
    assert sorted(corrected) == ["green", "red"]
    for name in corrected:
        np.testing.assert_array_equal(
            corrected[name].get_traces(), single.get_traces()
        )


def test_invalid_arguments_raise(simulated):
    """A missing reference, a wrong combination or a mismatch is an error."""
    signal, reference, _, _ = simulated
    traces = reference.get_traces()
    with pytest.raises(TypeError, match="reference_recording"):
        motion_correction(signal)
    with pytest.raises(TypeError, match="accepts"):
        motion_correction([signal], reference)
    with pytest.raises(TypeError, match="`reference`"):
        motion_correction({"green": signal, "iso": reference}, reference)
    with pytest.raises(ValueError, match="reference 'iso'"):
        motion_correction({"green": signal})
    with pytest.raises(ValueError, match="both coefficients"):
        motion_correction(signal, reference, coefficients=[[0.1, 0.1]])
    with pytest.raises(ValueError, match="one value per fiber"):
        motion_correction(
            signal, reference, coefficients=[[0.1]], reference_means=[[0.0]]
        )
    with pytest.raises(ValueError, match="same fibers in the same order"):
        motion_correction(
            signal, reference.select_channels(["fiber_1", "fiber_0"])
        )
    with pytest.raises(ValueError, match="sampling frequency"):
        motion_correction(
            signal,
            recording_from_traces(
                traces, name="iso", sampling_frequency=2 * _FS
            ),
        )
    with pytest.raises(ValueError, match="a frame or more"):
        motion_correction(
            signal,
            recording_from_traces(
                traces,
                name="iso",
                sampling_frequency=_FS,
                timestamps=np.arange(_N) / _FS + 1.0,
            ),
        )
    with pytest.raises(ValueError, match="number of samples"):
        motion_correction(
            signal,
            recording_from_traces(
                traces[:-1], name="iso", sampling_frequency=_FS
            ),
        )


def test_stages_walk_back_to_the_raw_recordings(simulated):
    """``stages()`` lists the inputs, the reference and repeated steps."""
    signal, reference, corrected, _ = simulated
    once = center(corrected)
    twice = center(once)

    assert list(corrected.stages()) == [
        "raw",
        "motion_correction.reference_recording",
        "motion_correction",
    ]
    stages = twice.stages()
    assert list(stages)[-3:] == ["motion_correction", "center", "center_2"]
    assert stages["raw"] is signal
    assert stages["motion_correction.reference_recording"] is reference
    assert stages["motion_correction"] is corrected
    assert stages["center"] is once
    assert stages["center_2"] is twice

    other = recording_from_traces(
        2 * reference.get_traces(), name="iso", sampling_frequency=_FS
    )
    again = motion_correction(corrected, other)
    stages = again.stages()
    assert stages["motion_correction.reference_recording"] is reference
    assert stages["motion_correction_2.reference_recording"] is other


def _multiplexed(delay=1 / 60):
    """Fast motion seen by the signal at t and by the reference at t + delay.

    Returns the signal, the reference with its true timestamps and the same
    reference with nominal ones.
    """
    rng = np.random.default_rng(0)
    times = np.arange(_N) / _FS

    def motion(t):
        return 0.3 * np.sin(2 * np.pi * t) + 0.03 * np.sin(0.05 * t)

    signal = 0.6 * motion(times) + rng.normal(0, _NOISE, _N)
    reference = motion(times + delay) + rng.normal(0, _NOISE, _N)
    make = recording_from_traces
    return (
        make(signal[:, None], "green", _FS),
        make(reference[:, None], "iso", _FS, timestamps=times + delay),
        make(reference[:, None], "iso", _FS),
    )


def test_offset_reference_is_interpolated_to_the_signal_times():
    """A reference sampled a fraction of a frame later is aligned first.

    The signal is pure motion plus noise, so the corrected trace should be at
    the noise floor; ignoring the offset leaves some of the fast motion.
    """
    signal, reference, nominal = _multiplexed()

    aligned = motion_correction(signal, reference)
    ignored = motion_correction(signal, nominal)

    floor = _NOISE * np.hypot(1, 0.6)
    assert aligned.get_traces().std() < 1.05 * floor
    assert ignored.get_traces().std() > 1.3 * floor
    measured = recording_from_traces(  # clocks differ in the 7th digit
        reference.get_traces(),
        name="iso",
        sampling_frequency=_FS * (1 + 1e-6),
        timestamps=reference.get_times(),
    )
    np.testing.assert_array_equal(
        motion_correction(signal, measured).get_traces(),
        aligned.get_traces(),
    )
    np.testing.assert_allclose(  # frame subsets, ends included
        aligned.get_traces(start_frame=0, end_frame=300),
        aligned.get_traces()[:300],
    )
    np.testing.assert_allclose(
        aligned.get_traces(start_frame=_N - 300), aligned.get_traces()[-300:]
    )


def test_segments_are_fitted_and_reloaded_separately(simulated, monkeypatch):
    """Each segment has its own coefficients, kept when reloading.

    The second segment's signal is halved, so its coefficient is half too. An
    empty third segment has nothing to fit and stays empty.
    """
    signal, reference, _, _ = simulated
    traces = signal.get_traces()
    two = recording_from_traces(
        [traces, traces / 2, traces[:0]], name="green", sampling_frequency=_FS
    )
    ref_traces = reference.get_traces()
    two_reference = recording_from_traces(
        [ref_traces, ref_traces, ref_traces[:0]],
        name="iso",
        sampling_frequency=_FS,
    )
    with pytest.warns(UserWarning, match="no finite sample"):
        corrected = motion_correction(two, two_reference)
    first, second, empty = corrected._kwargs["coefficients"]
    assert first[0] == pytest.approx(0.6, abs=0.05)
    assert second[0] == pytest.approx(0.3, abs=0.03)
    assert empty == [None, None]
    monkeypatch.setattr(motion_module, "fit_coefficient", pytest.fail)

    reloaded = BaseExtractor.from_dict(corrected.to_dict())

    for index in (0, 1, 2):
        np.testing.assert_array_equal(
            reloaded.get_traces(segment_index=index),
            corrected.get_traces(segment_index=index),
        )


def test_non_finite_samples_are_skipped():
    """Non-finite samples warn and are left out of the fit and the mean."""
    signal, reference, _ = _recordings()
    traces = signal.get_traces().copy()
    traces[[10, 500], 0] = np.nan
    ref_traces = reference.get_traces().copy()
    ref_traces[20, 0] = np.inf
    ref_traces[30, 1] = np.nan
    signal = recording_from_traces(
        traces, name="green", sampling_frequency=_FS
    )
    reference = recording_from_traces(
        ref_traces, name="iso", sampling_frequency=_FS
    )

    with pytest.warns(UserWarning, match="leaving 3 non-finite"):
        corrected = motion_correction(signal, reference)

    result = corrected.get_traces()
    assert np.flatnonzero(~np.isfinite(result[:, 0])).tolist() == [10, 20, 500]
    np.testing.assert_array_equal(result[:, 1], traces[:, 1])
    assert np.isfinite(corrected._kwargs["reference_means"][0]).all()
    assert corrected._kwargs["coefficients"][0][0] == pytest.approx(
        0.6, abs=0.05
    )


def test_unfittable_fibers_warn_and_the_others_are_corrected(simulated):
    """A dead reference gives NaN (null in JSON), a dead signal 0."""
    signal, reference, corrected, _ = simulated

    def with_dead_fiber(recording, name):
        traces = recording.get_traces().copy()
        traces[:, 1] = np.nan
        return recording_from_traces(traces, name=name, sampling_frequency=_FS)

    with pytest.warns(UserWarning, match="no finite sample"):
        dead_reference = motion_correction(
            signal, with_dead_fiber(reference, "iso")
        )
    with pytest.warns(UserWarning, match="nothing to fit"):
        dead_signal = motion_correction(
            with_dead_fiber(signal, "green"), reference
        )

    for result, coefficient in ((dead_reference, None), (dead_signal, 0.0)):
        assert result._kwargs["coefficients"][0][1] == coefficient
        assert np.isnan(result.get_traces()[:, 1]).all()
        np.testing.assert_array_equal(
            result.get_traces()[:, 0], corrected.get_traces()[:, 0]
        )
    reloaded = BaseExtractor.from_dict(dead_reference.to_dict())
    assert np.isnan(reloaded.get_traces()[:, 1]).all()


def test_fit_coefficient_edge_cases(monkeypatch):
    """Unfittable traces get coefficient 0; bad input or setup is an error."""
    short = np.arange(9.0)
    for signal, reference in (
        (np.arange(10.0), np.ones(10)),  # constant reference
        (short, short**2),  # too short to filter
    ):
        with pytest.warns(UserWarning, match="nothing to fit"):
            assert motion_module.fit_coefficient(signal, reference, _FS) == 0.0
    with pytest.raises(ValueError, match="shape"):
        motion_module.fit_coefficient(np.ones(10), np.ones(11), _FS)
    monkeypatch.setitem(sys.modules, "aind_ophys_utils.baseline_fitting", None)
    with pytest.raises(ImportError, match="aind-ophys-utils"):
        motion_module.fit_coefficient(np.ones(10), np.ones(10), _FS)
