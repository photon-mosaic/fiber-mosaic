"""Tests for the DAG pipeline executor."""

from __future__ import annotations

import numpy as np
import pytest
from spikeinterface.core import NumpyRecording
from spikeinterface.core.numpyextractors import NumpyRecordingSegment

from fiber_mosaic.core.base import (
    BaseFiberPhotometryExtractor,
    FiberPhotometryRecordingGroup,
)
from fiber_mosaic.processing.helpers import map_segments
from fiber_mosaic.processing.pipeline import apply_pipeline, validate_pipeline
from fiber_mosaic.processing.registry import pipeline_step

_SF = 100.0
_N = 20


def _make_recording(color="green", traces=None):
    """Build a single-segment fiber recording."""
    rec = BaseFiberPhotometryExtractor(
        sampling_frequency=_SF,
        fiber_ids=["f0", "f1"],
        color=color,
        dtype="float64",
    )
    if traces is None:
        traces = np.ones((_N, 2), dtype="float64")
    rec.add_segment(
        NumpyRecordingSegment(
            traces=np.asarray(traces, dtype="float64"),
            sampling_frequency=_SF,
            t_start=None,
        )
    )
    return rec


def _group():
    """A green+iso group with distinct, non-constant traces."""
    t = np.linspace(0.0, 1.0, _N)
    green = _make_recording("green", np.stack([1.0 + t, 2.0 + t], axis=1))
    iso = _make_recording(
        "iso", np.stack([0.5 + 0.5 * t, 1.0 + 0.5 * t], axis=1)
    )
    return FiberPhotometryRecordingGroup({"green": green, "iso": iso})


@pipeline_step("pipe_double", by_segment=False)
def _double(recording):
    return map_segments(recording, lambda traces: traces * 2.0)


@pipeline_step("pipe_split", by_segment=False)
def _split(recording):
    return FiberPhotometryRecordingGroup(
        {
            "plus": map_segments(recording, lambda traces: traces + 1.0),
            "minus": map_segments(recording, lambda traces: traces - 1.0),
        }
    )


@pipeline_step("pipe_bad", by_segment=False)
def _bad(recording):
    return 42


# ---------------- apply_pipeline ----------------


def test_pipeline_linear_recording_to_recording():
    group = _group()
    green_traces = group["green"].get_traces()
    result = apply_pipeline(
        group,
        [{"function": "pipe_double", "input": "green", "output": "green2"}],
    )
    assert "green2" in result
    np.testing.assert_array_equal(
        result["green2"].get_traces(), green_traces * 2.0
    )
    assert "green" in result and "iso" in result


def test_pipeline_group_input_step():
    pipeline = [
        {
            "function": "isosbestic_correction",
            "output": "corr",
            "params": {
                "signal": "green",
                "reference": "iso",
                "method": "subtract",
            },
        }
    ]
    result = apply_pipeline(_group(), pipeline)
    assert "corr" in result
    assert result["corr"].get_num_samples() == _N


def test_pipeline_recording_to_group_merges():
    result = apply_pipeline(
        _group(), [{"function": "pipe_split", "input": "green"}]
    )
    assert "plus" in result and "minus" in result


def test_pipeline_single_recording_source_seeds_by_color():
    rec = _make_recording("green")
    result = apply_pipeline(
        rec, [{"function": "pipe_double", "input": "green", "output": "g2"}]
    )
    assert "g2" in result


def test_pipeline_recording_without_output_raises():
    with pytest.raises(ValueError):
        apply_pipeline(
            _group(), [{"function": "pipe_double", "input": "green"}]
        )


def test_pipeline_bad_return_type_raises():
    with pytest.raises(TypeError):
        apply_pipeline(_group(), [{"function": "pipe_bad", "input": "green"}])


def test_pipeline_unknown_input_band_raises():
    with pytest.raises(ValueError):
        apply_pipeline(
            _group(),
            [{"function": "pipe_double", "input": "nope", "output": "x"}],
        )


# ---------------- validate_pipeline & seeding ----------------


def test_validate_pipeline_not_a_list():
    with pytest.raises(ValueError):
        validate_pipeline({"function": "pipe_double"})


def test_validate_pipeline_step_not_dict():
    with pytest.raises(ValueError):
        validate_pipeline([42])


def test_validate_pipeline_missing_function():
    with pytest.raises(ValueError):
        validate_pipeline([{"input": "green"}])


def test_validate_pipeline_unknown_function():
    with pytest.raises(ValueError):
        validate_pipeline([{"function": "pipe_no_such"}])


def test_validate_pipeline_input_not_string():
    with pytest.raises(ValueError):
        validate_pipeline([{"function": "pipe_double", "input": 123}])


def test_validate_pipeline_output_not_string():
    with pytest.raises(ValueError):
        validate_pipeline([{"function": "pipe_double", "output": 123}])


def test_validate_pipeline_params_not_dict():
    with pytest.raises(ValueError):
        validate_pipeline([{"function": "pipe_double", "params": []}])


def test_validate_pipeline_accepts_valid():
    validate_pipeline(
        [
            {
                "function": "pipe_double",
                "input": "green",
                "output": "g2",
                "params": {},
            }
        ]
    )


def test_seed_namespace_rejects_bad_source():
    with pytest.raises(TypeError):
        apply_pipeline(42, [])


def test_seed_namespace_recording_without_color_named_input():
    plain = NumpyRecording([np.ones((_N, 2))], sampling_frequency=_SF)
    result = apply_pipeline(
        plain,
        [{"function": "pipe_double", "input": "input", "output": "out"}],
    )
    assert "out" in result
