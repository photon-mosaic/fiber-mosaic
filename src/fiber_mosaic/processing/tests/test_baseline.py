"""Pytest tests for ``fiber_mosaic.processing.baseline``."""

from __future__ import annotations

import json
import sys
from importlib.util import find_spec

import numpy as np
import pytest
from spikeinterface.core.base import BaseExtractor

from fiber_mosaic.core.base import (
    FiberPhotometryRecordingGroup,
    recording_from_traces,
)
from fiber_mosaic.generation import calcium_transients
from fiber_mosaic.processing import (
    baseline_correction,
    multiexp_baseline,
)
from fiber_mosaic.processing import dff as dff_function

baseline_module = sys.modules["fiber_mosaic.processing.baseline"]

needs_aind = pytest.mark.skipif(
    find_spec("aind_ophys_utils") is None, reason="needs aind-ophys-utils"
)


def _recording(
    level, name="green", lengths=(50, 30), fibers=2, dtype="float32"
):
    """A constant-level recording with one segment per length."""
    return recording_from_traces(
        [np.full((n, fibers), level, dtype=dtype) for n in lengths],
        name=name,
        sampling_frequency=10.0,
    )


@pytest.mark.parametrize(
    "mode, dtype, signal, baseline, expected",
    [
        ("df", "float32", 4.0, 2.0, 2.0),
        ("dff", "float32", 4.0, 2.0, 1.0),
        # a baseline above an unsigned-integer signal must not wrap around
        ("df", "uint16", 90, 100, -10.0),
        ("dff", "uint16", 90, 100, -0.1),
    ],
)
def test_modes_in_every_segment(mode, dtype, signal, baseline, expected):
    """``df`` subtracts the baseline, ``dff`` also divides by it."""
    corrected = baseline_correction(
        _recording(signal, dtype=dtype),
        _recording(baseline, dtype=dtype),
        mode=mode,
    )

    for segment_index, n in enumerate((50, 30)):
        np.testing.assert_allclose(
            corrected.get_traces(segment_index=segment_index),
            np.full((n, 2), expected),
            rtol=1e-6,
        )


def test_dict_and_group_are_paired_by_name():
    """Containers pair signal and baseline by name and keep their type."""
    signals = {"green": _recording(4.0), "iso": _recording(8.0, name="iso")}
    baselines = {"iso": _recording(2.0, name="iso"), "green": _recording(2.0)}

    as_dict = baseline_correction(signals, baselines)
    as_group = baseline_correction(
        FiberPhotometryRecordingGroup(signals),
        FiberPhotometryRecordingGroup(baselines),
    )

    assert isinstance(as_group, FiberPhotometryRecordingGroup)
    for name, expected in (("green", 1.0), ("iso", 3.0)):
        np.testing.assert_array_equal(
            as_dict[name].get_traces(segment_index=0)[:, 0], expected
        )
        np.testing.assert_array_equal(
            as_group[name].get_traces(segment_index=0)[:, 0], expected
        )


def test_invalid_input_raises():
    """A baseline that doesn't fit, a bad mode, names or type is an error."""
    for baseline in (
        _recording(2.0, lengths=(50, 31)),
        _recording(2.0, fibers=3),
        _recording(2.0, lengths=(50,)),
    ):
        with pytest.raises(ValueError, match="differ in fibers"):
            baseline_correction(_recording(4.0), baseline)
    with pytest.raises(ValueError, match="mode"):
        baseline_correction(_recording(4.0), _recording(2.0), mode="divide")
    with pytest.raises(ValueError, match="names"):
        baseline_correction(
            {"green": _recording(4.0)}, {"red": _recording(2.0)}
        )
    with pytest.raises(TypeError, match="accepts"):
        baseline_correction([_recording(4.0)], [_recording(2.0)])


_FS = 20.0


def _simulate(
    num_samples,
    b_inf=100.0,
    terms=((40.0, 900.0), (20.0, 100.0), (10.0, 10.0)),
    amplitude=0.4,
    noise_sd=0.013,
    seed=0,
):
    """Return ``(trace, f0, calcium)``: baseline with transients and noise.

    ``terms`` are the bleaching ``(amplitude, tau in s)`` pairs of ``f0``;
    ``amplitude`` and ``noise_sd`` are relative to ``f0``.
    """
    times = np.arange(num_samples) / _FS
    rng = np.random.default_rng(seed)
    f0 = b_inf + sum(a * np.exp(-times / tau) for a, tau in terms)
    calcium = calcium_transients(
        times,
        rng,
        num_events=num_samples // 60,
        decay=0.25,
        amplitude=amplitude,
    )
    trace = f0 * (1 + calcium + rng.normal(0.0, noise_sd, num_samples))
    return trace.astype("float32"), f0, calcium


# Bleaching curves are fits to the two dLight fibers of a real 85-minute
# session; the slow taus are as long as the recording or longer, and a
# negative amplitude is a brightening term. Green events (~40 % dF/F, ~20/min,
# tau 0.25 s) and white-noise SDs are measured across sessions; iso and red
# have no events.
# (b_inf, terms, event amplitude, relative noise sd, F0 error tolerance)
_REAL_LIKE = [
    pytest.param(
        1421.0,
        [(616.0, 7696.0), (132.0, 1573.0), (95.8, 103.0)],
        0.4,
        0.014,
        0.01,
        id="green-three-exponentials",
    ),
    pytest.param(
        134.0,
        [(1401.0, 33579.0), (1351.0, 533.0), (-1313.0, 582.0)],
        0.0,
        0.0081,
        0.01,
        id="iso-cancelling-terms",
    ),
    pytest.param(
        293.0,
        [(1005.0, 49879.0), (1005.0, 4475.0), (-851.0, 3059.0)],
        0.0,
        0.010,
        0.01,
        id="red-brightening",
    ),
]


@pytest.mark.parametrize(
    "b_inf, terms, amplitude, noise_sd, tolerance", _REAL_LIKE
)
@needs_aind
def test_recovers_real_like_baselines(
    b_inf, terms, amplitude, noise_sd, tolerance
):
    """F0 is recovered on bleaching curves like those of a real session."""
    trace, f0, calcium = _simulate(102_000, b_inf, terms, amplitude, noise_sd)
    recording = recording_from_traces(
        trace[:, None], name="green", sampling_frequency=_FS
    )

    result = dff_function(recording).get_traces()[:, 0]

    assert np.max(np.abs(trace / (result + 1.0) / f0 - 1.0)) < tolerance
    if amplitude:
        assert np.corrcoef(result, calcium)[0, 1] > 0.85


@needs_aind
def test_saved_json_is_standard_and_reuses_fit(tmp_path, monkeypatch):
    """Padded taus (inf) are saved as null: standard JSON, no refit on load."""
    recording = recording_from_traces(
        _simulate(3000)[0][:, None], name="green", sampling_frequency=_FS
    )
    # recordings made from arrays can't be dumped, so write the traces first
    saved = recording.save(folder=tmp_path / "traces", progress_bar=False)
    dff = dff_function(saved)
    path = tmp_path / "dff.json"
    dff.dump_to_json(path)

    # Infinity and NaN reach parse_constant; a standard document has neither
    json.loads(path.read_text(), parse_constant=pytest.fail)
    monkeypatch.setattr(baseline_module, "fit_baseline", pytest.fail)
    reloaded = BaseExtractor.load(path)

    np.testing.assert_array_equal(reloaded.get_traces(), dff.get_traces())

    # fitted parameters alone are enough: the origins default to the start
    baseline = dff._kwargs["baseline_recording"]
    reused = multiexp_baseline(
        saved, baseline_params=baseline._kwargs["baseline_params"]
    )
    np.testing.assert_array_equal(reused.get_traces(), baseline.get_traces())


@needs_aind
def test_dff_is_the_baseline_correction_of_the_multiexp_baseline():
    """``dff`` is the shortcut for its two parts, on a group too.

    Slicing frames and fibers agrees with slicing the full output.
    """
    traces = np.stack([_simulate(3000, seed=seed)[0] for seed in (1, 2)], 1)
    group = FiberPhotometryRecordingGroup(
        {
            color: recording_from_traces(
                traces, name=color, sampling_frequency=_FS
            )
            for color in ("green", "iso")
        }
    )

    shortcut = dff_function(group)
    parts = baseline_correction(group, multiexp_baseline(group))

    assert shortcut.names == parts.names == ["green", "iso"]
    for color in group:
        np.testing.assert_array_equal(
            shortcut[color].get_traces(), parts[color].get_traces()
        )
    np.testing.assert_allclose(
        shortcut["green"].get_traces(
            start_frame=100, end_frame=500, channel_ids=["fiber_1"]
        ),
        shortcut["green"].get_traces()[100:500, [1]],
    )


@needs_aind
def test_segments_use_their_own_times():
    """Each segment is fitted on its own, and real timestamps enter the fit.

    Offsetting a segment's timestamps changes nothing: the fit's ``t = 0``
    sits at the segment's first sample.
    """
    traces = [
        _simulate(n, seed=seed)[0][:, None]
        for n, seed in ((3000, 3), (2400, 4))
    ]
    times = [np.arange(len(trace)) / _FS for trace in traces]

    plain = recording_from_traces(
        traces, name="green", sampling_frequency=_FS, timestamps=times
    )
    shifted = recording_from_traces(
        traces,
        name="green",
        sampling_frequency=_FS,
        timestamps=[times[0] + 1000.0, times[1] + 5000.0],
    )

    shifted_dff, plain_dff = dff_function(shifted), dff_function(plain)
    for segment_index in range(2):
        np.testing.assert_allclose(
            shifted_dff.get_traces(segment_index=segment_index),
            plain_dff.get_traces(segment_index=segment_index),
            atol=1e-3,
        )


@needs_aind
def test_non_finite_samples_are_skipped():
    """NaN samples warn, stay NaN in the output, and leave the rest fitted.

    A trace with no finite sample is an error, not a NaN baseline.
    """
    trace = _simulate(3000)[0]
    trace[[10, 500, 501]] = np.nan
    recording = recording_from_traces(
        trace[:, None], name="green", sampling_frequency=_FS
    )

    with pytest.warns(UserWarning, match="dropping 3 non-finite"):
        result = dff_function(recording).get_traces()[:, 0]

    assert np.isnan(result[[10, 500, 501]]).all()
    assert np.isfinite(np.delete(result, [10, 500, 501])).all()
    with pytest.raises(ValueError, match="non-finite"):  # nothing to fit
        baseline_module.fit_baseline(np.full(100, np.nan), np.arange(100.0))


def test_missing_ophys_utils_message(monkeypatch):
    """Without aind-ophys-utils the fit explains what to install."""
    monkeypatch.setitem(sys.modules, "aind_ophys_utils.signal_utils", None)

    with pytest.raises(ImportError, match="aind-ophys-utils"):
        baseline_module.fit_baseline(np.ones(100), np.arange(100.0))
