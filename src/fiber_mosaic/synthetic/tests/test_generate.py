"""Pytest tests for ``fiber_mosaic.synthetic.generate``."""

from __future__ import annotations

import numpy as np

from fiber_mosaic.core.base import (
    BaseFiberPhotometryExtractor,
    FiberPhotometryRecordingGroup,
)
from fiber_mosaic.synthetic import (
    recording_from_traces,
    simulate_bands,
    simulate_group,
)

_NUM_SAMPLES = 500
_NUM_FIBERS = 2


def _mean_correlation(first, second):
    """Return the per-fiber Pearson correlation, averaged over fibers."""
    return np.mean(
        [
            np.corrcoef(first[:, fiber], second[:, fiber])[0, 1]
            for fiber in range(first.shape[1])
        ]
    )


def test_simulate_bands():
    signal, reference, calcium = simulate_bands(
        num_samples=_NUM_SAMPLES, num_fibers=_NUM_FIBERS
    )

    expected_shape = (_NUM_SAMPLES, _NUM_FIBERS)
    assert signal.shape == reference.shape == calcium.shape == expected_shape

    # fluorescence sits on a positive baseline so "divide" methods are safe
    assert signal.min() > 0.0
    assert reference.min() > 0.0

    # the signal model: both bands share the bleach and motion artifacts,
    # while the calcium transients live in the signal band only
    assert _mean_correlation(signal, reference) > 0.8
    assert abs(_mean_correlation(calcium, reference)) < 0.2
    assert calcium.max() > 0.0

    # the seed is a documented reproducibility guarantee
    repeated, _, _ = simulate_bands(
        num_samples=_NUM_SAMPLES, num_fibers=_NUM_FIBERS
    )
    np.testing.assert_array_equal(signal, repeated)


def test_simulate_group_and_recording_from_traces():
    group, calcium = simulate_group(
        num_samples=_NUM_SAMPLES, num_fibers=_NUM_FIBERS
    )

    assert isinstance(group, FiberPhotometryRecordingGroup)
    assert group.colors == ["green", "iso"]
    assert group.get_num_fibers() == _NUM_FIBERS
    assert calcium.shape == (_NUM_SAMPLES, _NUM_FIBERS)
    green = group["green"]
    assert isinstance(green, BaseFiberPhotometryExtractor)
    assert green.color == "green"
    assert green.get_num_samples() == _NUM_SAMPLES

    # a sequence of arrays becomes one segment each, with explicit fiber ids
    multi = recording_from_traces(
        [np.ones((10, 2)), np.zeros((4, 2))],
        color="red",
        fiber_ids=["a", "b"],
    )
    assert multi.get_num_segments() == 2
    assert multi.get_num_samples(1) == 4
    np.testing.assert_array_equal(multi.get_fiber_ids(), ["a", "b"])
