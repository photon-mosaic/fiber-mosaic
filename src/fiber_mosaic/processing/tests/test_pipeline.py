"""Pytest tests for ``fiber_mosaic.processing.pipeline``."""

from __future__ import annotations

import numpy as np
import pytest
from spikeinterface.core.numpyextractors import NumpyRecording

from fiber_mosaic.core import FiberPhotometryRecordingGroup
from fiber_mosaic.processing import (
    bandpass_filter,
    isosbestic_correction,
    zscore,
)
from fiber_mosaic.processing.from_spikeinterface import FiberZScoreRecording
from fiber_mosaic.processing.pipeline import (
    FiberProcessingPipeline,
    apply_processing_pipeline,
)


def _rec(seed):
    traces = np.random.default_rng(seed).standard_normal((500, 3))
    return NumpyRecording([traces.astype("float32")], 100.0)


def test_pipeline_matches_applying_the_steps_by_hand():
    """The pipeline resolves names to fiber steps and chains them in order."""
    rec = _rec(0)
    pipeline = FiberProcessingPipeline(
        [
            {"name": "bandpass_filter", "params": {"freq_min": 0.2}},
            {"name": "zscore", "params": {"seed": 0}},
        ]
    )

    processed = apply_processing_pipeline(rec, pipeline)

    assert isinstance(processed, FiberZScoreRecording)
    expected = zscore(bandpass_filter(rec, freq_min=0.2), seed=0)
    np.testing.assert_allclose(processed.get_traces(), expected.get_traces())


def test_group_pipeline_resolves_cross_recording_references():
    """Steps run in dependency order: iso, then red, then green."""
    group = FiberPhotometryRecordingGroup(
        {"green": _rec(0), "red": _rec(1), "iso": _rec(2)}
    )
    bp = {"name": "bandpass_filter", "params": {"freq_min": 0.2}}
    pipeline = {
        "green": [
            bp,
            {
                "name": "isosbestic_correction",
                "params": {"reference_recording": "pipeline[red_corrected]"},
            },
        ],
        "red": [
            bp,
            {
                "name": "isosbestic_correction",
                "params": {"reference_recording": "pipeline[iso_bandpass]"},
                "output_name": "red_corrected",
            },
        ],
        "iso": [{**bp, "output_name": "iso_bandpass"}],
    }

    processed = apply_processing_pipeline(group, pipeline)

    def filt(name):
        return bandpass_filter(group[name], freq_min=0.2)

    red = isosbestic_correction(filt("red"), filt("iso"))
    green = isosbestic_correction(filt("green"), red)
    np.testing.assert_allclose(
        processed["green"].get_traces(), green.get_traces()
    )
    np.testing.assert_allclose(processed["red"].get_traces(), red.get_traces())
    np.testing.assert_allclose(
        processed["iso"].get_traces(), filt("iso").get_traces()
    )


def test_group_pipeline_rejects_cycles_and_unknown_names():
    group = FiberPhotometryRecordingGroup({"green": _rec(0), "iso": _rec(1)})

    def corr(ref, out):
        return {
            "name": "isosbestic_correction",
            "params": {"reference_recording": f"pipeline[{ref}]"},
            "output_name": out,
        }

    with pytest.raises(ValueError, match="cycle"):
        apply_processing_pipeline(
            group, {"green": [corr("b", "a")], "iso": [corr("a", "b")]}
        )
    with pytest.raises(ValueError, match="not recordings"):
        apply_processing_pipeline(group, {"red": []})
    with pytest.raises(TypeError, match="single recording"):
        apply_processing_pipeline(group["green"], {"green": []})
