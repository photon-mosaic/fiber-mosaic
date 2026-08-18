"""Tests for the DAG pipeline executor and its schema."""

from __future__ import annotations

import json

import numpy as np
import pytest
from spikeinterface.core import NumpyRecording
from spikeinterface.core.numpyextractors import NumpyRecordingSegment

from fiber_mosaic.core.base import (
    BaseFiberPhotometryExtractor,
    FiberPhotometryRecordingGroup,
)
from fiber_mosaic.processing.pipeline import apply_pipeline, validate_pipeline
from fiber_mosaic.processing.registry import processing_step

_SF = 100.0
_N = 20


def _make_recording(color="green", traces=None):
    """Build a single-segment fiber recording."""
    recording = BaseFiberPhotometryExtractor(
        sampling_frequency=_SF,
        fiber_ids=["f0", "f1"],
        color=color,
        dtype="float64",
    )
    if traces is None:
        traces = np.ones((_N, 2), dtype="float64")
    recording.add_segment(
        NumpyRecordingSegment(
            traces=np.asarray(traces, dtype="float64"),
            sampling_frequency=_SF,
            t_start=None,
        )
    )
    return recording


def _group():
    """A green + iso group with distinct, non-constant traces."""
    ramp = np.linspace(0.0, 1.0, _N)
    green = _make_recording("green", np.stack([1.0 + ramp, 2.0 + ramp], 1))
    iso = _make_recording(
        "iso", np.stack([0.5 + 0.5 * ramp, 1.0 + 0.5 * ramp], 1)
    )
    return FiberPhotometryRecordingGroup({"green": green, "iso": iso})


@processing_step("pipe_double")
def _double(traces, times, *, factor=2.0):
    return traces * factor


@processing_step("pipe_combine")
def _combine(signal, times, reference):
    return signal - reference


@processing_step("pipe_split")
def _split(traces, times):
    return {"plus": traces + 1.0, "minus": traces - 1.0}


# ---------------- execution ----------------


def test_singular_input_and_output():
    result = apply_pipeline(
        _group(),
        [{"function": "pipe_double", "input": "green", "output": "green2"}],
    )
    np.testing.assert_allclose(
        result["green2"].get_traces(),
        _group()["green"].get_traces() * 2.0,
    )
    # the seeded entries survive alongside the derived one
    assert set(result) == {"green", "iso", "green2"}


def test_params_are_forwarded():
    result = apply_pipeline(
        _group(),
        [
            {
                "function": "pipe_double",
                "input": "green",
                "output": "green3",
                "params": {"factor": 3.0},
            }
        ],
    )
    np.testing.assert_allclose(
        result["green3"].get_traces(),
        _group()["green"].get_traces() * 3.0,
    )


def test_plural_inputs_wire_by_name():
    result = apply_pipeline(
        _group(),
        [
            {
                "function": "pipe_combine",
                "inputs": {"signal": "green", "reference": "iso"},
                "output": "difference",
            }
        ],
    )
    group = _group()
    np.testing.assert_allclose(
        result["difference"].get_traces(),
        group["green"].get_traces() - group["iso"].get_traces(),
    )


def test_plural_outputs_bind_by_name():
    result = apply_pipeline(
        _group(),
        [
            {
                "function": "pipe_split",
                "input": "green",
                "outputs": {"plus": "up", "minus": "down"},
            }
        ],
    )
    green = _group()["green"].get_traces()
    np.testing.assert_allclose(result["up"].get_traces(), green + 1.0)
    np.testing.assert_allclose(result["down"].get_traces(), green - 1.0)


def test_steps_chain_through_the_workspace():
    result = apply_pipeline(
        _group(),
        [
            {"function": "pipe_double", "input": "green", "output": "a"},
            {"function": "pipe_double", "input": "a", "output": "b"},
        ],
    )
    np.testing.assert_allclose(
        result["b"].get_traces(), _group()["green"].get_traces() * 4.0
    )


def test_one_to_many_as_sibling_steps():
    # two steps reading the same entry, without recomputing it
    result = apply_pipeline(
        _group(),
        [
            {
                "function": "pipe_double",
                "input": "green",
                "output": "twice",
                "params": {"factor": 2.0},
            },
            {
                "function": "pipe_double",
                "input": "green",
                "output": "thrice",
                "params": {"factor": 3.0},
            },
        ],
    )
    assert {"twice", "thrice"} <= set(result)


def test_registered_upstream_preprocessor_runs():
    result = apply_pipeline(
        _group(),
        [{"function": "center", "input": "green", "output": "centered"}],
    )
    assert result["centered"].get_num_samples() == _N


def test_explicit_seeding_binds_colors_to_roles():
    group = _group()
    result = apply_pipeline(
        {"raw_signal": group["green"], "raw_reference": group["iso"]},
        [
            {
                "function": "pipe_combine",
                "inputs": {
                    "signal": "raw_signal",
                    "reference": "raw_reference",
                },
                "output": "difference",
            }
        ],
    )
    np.testing.assert_allclose(
        result["difference"].get_traces(),
        group["green"].get_traces() - group["iso"].get_traces(),
    )


def test_explicit_seeding_makes_a_pipeline_color_agnostic():
    # the same pipeline runs on a group whose colors are named differently
    pipeline = [
        {
            "function": "pipe_combine",
            "inputs": {"signal": "raw_signal", "reference": "raw_reference"},
            "output": "difference",
        }
    ]
    odd = FiberPhotometryRecordingGroup(
        {
            "470": _make_recording("470", np.full((_N, 2), 3.0)),
            "415": _make_recording("415", np.full((_N, 2), 1.0)),
        }
    )
    result = apply_pipeline(
        {"raw_signal": odd["470"], "raw_reference": odd["415"]}, pipeline
    )
    np.testing.assert_allclose(result["difference"].get_traces(), 2.0)


def test_seed_mapping_rejects_non_recording_values():
    with pytest.raises(TypeError, match="must be a recording"):
        apply_pipeline({"raw_signal": 42}, [])


def test_seed_mapping_rejects_non_string_names():
    with pytest.raises(TypeError, match="names must be strings"):
        apply_pipeline({1: _make_recording()}, [])


def test_single_recording_source_seeds_by_color():
    result = apply_pipeline(
        _make_recording("green"),
        [{"function": "pipe_double", "input": "green", "output": "g2"}],
    )
    assert "g2" in result


def test_recording_without_color_seeds_as_input():
    plain = NumpyRecording([np.ones((_N, 2))], sampling_frequency=_SF)
    result = apply_pipeline(
        plain,
        [{"function": "pipe_double", "input": "input", "output": "out"}],
    )
    assert "out" in result


def test_pipeline_is_json_round_trippable():
    pipeline = [
        {
            "function": "pipe_combine",
            "inputs": {"signal": "green", "reference": "iso"},
            "output": "difference",
        }
    ]
    restored = json.loads(json.dumps(pipeline))
    assert "difference" in apply_pipeline(_group(), restored)


# ---------------- validation ----------------


def test_pipeline_must_be_a_list():
    with pytest.raises(ValueError, match="must be a list"):
        validate_pipeline({"function": "pipe_double"})


def test_step_must_be_a_dict():
    with pytest.raises(ValueError, match="must be a dict"):
        validate_pipeline([42])


def test_missing_function_key_raises():
    with pytest.raises(ValueError, match="'function'"):
        validate_pipeline([{"input": "green", "output": "x"}])


def test_unknown_function_raises():
    with pytest.raises(ValueError, match="unknown function"):
        validate_pipeline([{"function": "pipe_nope", "output": "x"}])


def test_params_must_be_a_dict():
    with pytest.raises(ValueError, match="'params'"):
        validate_pipeline(
            [
                {
                    "function": "pipe_double",
                    "input": "green",
                    "output": "x",
                    "params": [],
                }
            ]
        )


def test_unknown_parameter_raises():
    with pytest.raises(ValueError, match="unknown parameter"):
        validate_pipeline(
            [
                {
                    "function": "pipe_double",
                    "input": "green",
                    "output": "x",
                    "params": {"factr": 2.0},
                }
            ]
        )


def test_both_input_forms_raises():
    with pytest.raises(ValueError, match="'input' or 'inputs'"):
        validate_pipeline(
            [
                {
                    "function": "pipe_double",
                    "input": "green",
                    "inputs": {"traces": "green"},
                    "output": "x",
                }
            ]
        )


def test_neither_input_form_raises():
    with pytest.raises(ValueError, match="'input' or 'inputs'"):
        validate_pipeline([{"function": "pipe_double", "output": "x"}])


def test_neither_output_form_raises():
    with pytest.raises(ValueError, match="'output' or 'outputs'"):
        validate_pipeline([{"function": "pipe_double", "input": "green"}])


def test_singular_input_on_multi_input_step_raises():
    with pytest.raises(ValueError, match="use 'inputs'"):
        validate_pipeline(
            [
                {
                    "function": "pipe_combine",
                    "input": "green",
                    "output": "x",
                }
            ]
        )


def test_unknown_input_name_raises():
    with pytest.raises(ValueError, match="inputs are"):
        validate_pipeline(
            [
                {
                    "function": "pipe_combine",
                    "inputs": {"signl": "green", "reference": "iso"},
                    "output": "x",
                }
            ]
        )


def test_input_must_be_a_string():
    with pytest.raises(ValueError, match="'input' must be a string"):
        validate_pipeline(
            [{"function": "pipe_double", "input": 1, "output": "x"}]
        )


def test_output_must_be_a_string():
    with pytest.raises(ValueError, match="'output' must be a string"):
        validate_pipeline(
            [{"function": "pipe_double", "input": "green", "output": 1}]
        )


def test_inputs_mapping_must_be_strings():
    with pytest.raises(ValueError, match="names to entry names"):
        validate_pipeline(
            [
                {
                    "function": "pipe_combine",
                    "inputs": {"signal": 1, "reference": "iso"},
                    "output": "x",
                }
            ]
        )


def test_outputs_must_be_a_dict():
    with pytest.raises(ValueError, match="'outputs' must be a dict"):
        validate_pipeline(
            [
                {
                    "function": "pipe_double",
                    "input": "green",
                    "outputs": ["a"],
                }
            ]
        )


def test_unknown_entry_raises():
    with pytest.raises(ValueError, match="unknown entry"):
        apply_pipeline(
            _group(),
            [{"function": "pipe_double", "input": "nope", "output": "x"}],
        )


def test_known_entries_are_tracked_across_steps():
    # 'a' only exists because the first step produces it
    validate_pipeline(
        [
            {"function": "pipe_double", "input": "green", "output": "a"},
            {"function": "pipe_double", "input": "a", "output": "b"},
        ],
        known_entries={"green"},
    )


def test_singular_output_on_multi_output_step_raises():
    with pytest.raises(ValueError, match="single 'output'"):
        apply_pipeline(
            _group(),
            [{"function": "pipe_split", "input": "green", "output": "x"}],
        )


def test_mismatched_output_names_raise():
    with pytest.raises(ValueError, match="declared outputs"):
        apply_pipeline(
            _group(),
            [
                {
                    "function": "pipe_split",
                    "input": "green",
                    "outputs": {"plus": "up", "nope": "down"},
                }
            ],
        )


def test_bad_source_type_raises():
    with pytest.raises(TypeError, match="must be a"):
        apply_pipeline(42, [])
