"""Pytest tests for ``fiber_mosaic.processing.pipeline``."""

from __future__ import annotations

import numpy as np
from spikeinterface.core.numpyextractors import NumpyRecording
from spikeinterface.preprocessing import apply_preprocessing_pipeline

from fiber_mosaic.processing import bandpass_filter, zscore
from fiber_mosaic.processing.from_spikeinterface import FiberZScoreRecording
from fiber_mosaic.processing.pipeline import FiberPreprocessingPipeline


def test_pipeline_matches_applying_the_steps_by_hand():
    """The pipeline resolves names to fiber steps and chains them in order."""
    traces = np.random.default_rng(0).standard_normal((500, 3))
    rec = NumpyRecording([traces.astype("float32")], 100.0)
    pipeline = FiberPreprocessingPipeline(
        [
            {"name": "bandpass_filter", "params": {"freq_min": 0.2}},
            {"name": "zscore", "params": {"seed": 0}},
        ]
    )

    processed = apply_preprocessing_pipeline(rec, pipeline)

    assert isinstance(processed, FiberZScoreRecording)
    expected = zscore(bandpass_filter(rec, freq_min=0.2), seed=0)
    np.testing.assert_allclose(processed.get_traces(), expected.get_traces())
