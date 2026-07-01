"""Pytest tests for ``fiber_mosaic.core.base``."""

from __future__ import annotations

import numpy as np
import pytest
from spikeinterface.core.numpyextractors import NumpyRecordingSegment

from fiber_mosaic.core.base import (
    BaseFiberPhotometryExtractor,
    FiberPhotometryRecordingGroup,
)


def _make_recording(
    fiber_ids=("f0", "f1", "f2"),
    color="green",
    n_samples=10,
    sampling_frequency=100.0,
    dtype="float32",
):
    """Build a single-segment recording with deterministic ramp traces."""
    rec = BaseFiberPhotometryExtractor(
        sampling_frequency=sampling_frequency,
        fiber_ids=list(fiber_ids),
        color=color,
        dtype=dtype,
    )
    n_fibers = len(fiber_ids)
    traces = np.arange(n_samples * n_fibers, dtype=dtype).reshape(
        n_samples, n_fibers
    )
    rec.add_segment(
        NumpyRecordingSegment(
            traces=traces,
            sampling_frequency=sampling_frequency,
            t_start=None,
        )
    )
    return rec, traces


@pytest.fixture
def recording():
    """Default 10x3 recording reused across extractor tests."""
    return _make_recording()


@pytest.fixture
def group():
    """green+red recording group sharing the default fiber set."""
    green, _ = _make_recording(color="green")
    red, _ = _make_recording(color="red")
    grp = FiberPhotometryRecordingGroup({"green": green, "red": red})
    return grp, green, red


# ---------------- BaseFiberPhotometryExtractor ----------------


def test_construction_sets_color_and_fibers(recording):
    """Color, fiber count, and fiber_ids are wired through __init__."""
    rec, _ = recording
    assert rec.color == "green"
    assert rec.get_num_fibers() == 3
    np.testing.assert_array_equal(rec.fiber_ids, np.array(["f0", "f1", "f2"]))
    np.testing.assert_array_equal(rec.get_fiber_ids(), rec.fiber_ids)


def test_kwargs_round_trip_metadata(recording):
    """``_kwargs`` carries color and fiber_ids for serialization."""
    rec, _ = recording
    assert rec._kwargs["color"] == "green"
    assert rec._kwargs["fiber_ids"] == ["f0", "f1", "f2"]


def test_get_fluorescence_full(recording):
    """No-arg call returns the entire (n_samples, n_fibers) array."""
    rec, traces = recording
    np.testing.assert_array_equal(rec.get_fluorescence(), traces)


def test_get_fluorescence_frame_range(recording):
    """``start_frame``/``end_frame`` slice along the time axis."""
    rec, traces = recording
    np.testing.assert_array_equal(
        rec.get_fluorescence(start_frame=2, end_frame=5),
        traces[2:5],
    )


def test_get_fluorescence_fiber_subset(recording):
    """``fiber_ids`` selects columns by fiber label."""
    rec, traces = recording
    np.testing.assert_array_equal(
        rec.get_fluorescence(fiber_ids=["f0", "f2"]),
        traces[:, [0, 2]],
    )


def test_get_fluorescence_matches_get_traces(recording):
    """``get_fluorescence`` and inherited ``get_traces`` agree."""
    rec, _ = recording
    np.testing.assert_array_equal(rec.get_fluorescence(), rec.get_traces())


def test_repr_includes_color_and_fiber_count(recording):
    """``repr`` surfaces color and fiber count."""
    rec, _ = recording
    r = repr(rec)
    assert "color=green" in r
    assert "3 fiber" in r


def test_multi_segment():
    """Multiple segments are addressable via ``segment_index``."""
    rec = BaseFiberPhotometryExtractor(
        sampling_frequency=100.0,
        fiber_ids=["f0", "f1"],
        color="red",
        dtype="float32",
    )
    rec.add_segment(
        NumpyRecordingSegment(
            traces=np.zeros((4, 2), dtype="float32"),
            sampling_frequency=100.0,
            t_start=None,
        )
    )
    rec.add_segment(
        NumpyRecordingSegment(
            traces=np.ones((6, 2), dtype="float32"),
            sampling_frequency=100.0,
            t_start=None,
        )
    )
    assert rec.get_num_segments() == 2
    assert rec.get_fluorescence(segment_index=0).shape == (4, 2)
    assert rec.get_fluorescence(segment_index=1).shape == (6, 2)


# ---------------- per-fiber times & streams ----------------


def test_get_streams_not_implemented():
    """Base extractor reads no file format, so get_streams is abstract."""
    with pytest.raises(NotImplementedError):
        BaseFiberPhotometryExtractor.get_streams("any-source")


def test_set_times_per_fiber_2d(recording):
    """A 2-D (n_samples, n_fibers) array is stored as per-fiber times."""
    rec, traces = recording
    n_samples, n_fibers = traces.shape
    times = np.arange(
        n_samples * n_fibers, dtype="float64"
    ).reshape(n_samples, n_fibers)
    times = times / 1000.0

    assert not rec.has_fiber_times()
    rec.set_times(times)
    assert rec.has_fiber_times()
    np.testing.assert_array_equal(rec.get_fiber_times(), times)


def test_set_times_1d_broadcasts_to_all_fibers(recording):
    """A 1-D vector is applied identically to every fiber."""
    rec, traces = recording
    n_samples, n_fibers = traces.shape
    times = np.linspace(0.0, 1.0, n_samples)

    rec.set_times(times)
    fiber_times = rec.get_fiber_times()

    assert fiber_times.shape == (n_samples, n_fibers)
    for fiber in range(n_fibers):
        np.testing.assert_array_equal(fiber_times[:, fiber], times)


def test_get_fiber_times_default_uses_sampling_frequency(recording):
    """Without set_times, times are synthesized from the sample rate."""
    rec, traces = recording
    n_samples, n_fibers = traces.shape
    expected = np.arange(n_samples) / rec.get_sampling_frequency()

    fiber_times = rec.get_fiber_times()

    assert not rec.has_fiber_times()
    assert fiber_times.shape == (n_samples, n_fibers)
    for fiber in range(n_fibers):
        np.testing.assert_allclose(fiber_times[:, fiber], expected)


def test_get_fiber_times_frame_and_fiber_subset(recording):
    """start/end frame slice rows; fiber_ids selects columns."""
    rec, _ = recording
    n_samples = rec.get_num_samples()
    n_fibers = rec.get_num_fibers()
    times = np.arange(
        n_samples * n_fibers, dtype="float64"
    ).reshape(n_samples, n_fibers)
    rec.set_times(times)

    np.testing.assert_array_equal(
        rec.get_fiber_times(
            start_frame=2, end_frame=5, fiber_ids=["f0", "f2"]
        ),
        times[2:5][:, [0, 2]],
    )


def test_set_times_wrong_1d_length_raises(recording):
    """A 1-D vector whose length != n_samples is rejected."""
    rec, _ = recording
    with pytest.raises(ValueError):
        rec.set_times(np.zeros(rec.get_num_samples() + 1))


def test_set_times_wrong_2d_shape_raises(recording):
    """A 2-D array not matching (n_samples, n_fibers) is rejected."""
    rec, _ = recording
    bad = np.zeros((rec.get_num_samples(), rec.get_num_fibers() + 1))
    with pytest.raises(ValueError):
        rec.set_times(bad)


def test_set_times_wrong_ndim_raises(recording):
    """Times with more than two dimensions are rejected."""
    rec, _ = recording
    bad = np.zeros((rec.get_num_samples(), rec.get_num_fibers(), 1))
    with pytest.raises(ValueError):
        rec.set_times(bad)


def test_set_times_multi_segment():
    """Per-fiber times are stored independently per segment."""
    rec = BaseFiberPhotometryExtractor(
        sampling_frequency=100.0,
        fiber_ids=["f0", "f1"],
        color="red",
        dtype="float32",
    )
    rec.add_segment(
        NumpyRecordingSegment(
            traces=np.zeros((4, 2), dtype="float32"),
            sampling_frequency=100.0,
            t_start=None,
        )
    )
    rec.add_segment(
        NumpyRecordingSegment(
            traces=np.ones((6, 2), dtype="float32"),
            sampling_frequency=100.0,
            t_start=None,
        )
    )
    t0 = np.zeros((4, 2), dtype="float64")
    t1 = np.ones((6, 2), dtype="float64")
    rec.set_times(t0, segment_index=0)
    rec.set_times(t1, segment_index=1)

    assert rec.has_fiber_times(segment_index=0)
    np.testing.assert_array_equal(rec.get_fiber_times(segment_index=0), t0)
    np.testing.assert_array_equal(rec.get_fiber_times(segment_index=1), t1)


# ---------------- FiberPhotometryRecordingGroup ----------------


def test_empty_recordings_raises():
    """Passing an empty mapping is rejected."""
    with pytest.raises(ValueError):
        FiberPhotometryRecordingGroup({})


def test_mismatched_fiber_sets_raises():
    """Colors with differing fiber IDs are rejected at construction."""
    g, _ = _make_recording(fiber_ids=["a", "b"])
    r, _ = _make_recording(fiber_ids=["a", "c"])
    with pytest.raises(ValueError):
        FiberPhotometryRecordingGroup({"green": g, "red": r})


def test_group_colors_and_fiber_ids(group):
    """``colors``, ``fiber_ids``, ``get_num_fibers`` reflect inputs."""
    grp, *_ = group
    assert grp.colors == ["green", "red"]
    np.testing.assert_array_equal(grp.fiber_ids, np.array(["f0", "f1", "f2"]))
    assert grp.get_num_fibers() == 3


def test_group_indexing_and_lookup(group):
    """``__getitem__``, ``get_recording``, ``in``, ``len``, ``iter``."""
    grp, green, red = group
    assert grp["green"] is green
    assert grp.get_recording("red") is red
    assert "green" in grp
    assert "blue" not in grp
    assert len(grp) == 2
    assert list(iter(grp)) == ["green", "red"]


def test_group_dict_like_methods(group):
    """``keys``, ``values``, ``items`` mirror the underlying dict."""
    grp, green, red = group
    assert list(grp.keys()) == ["green", "red"]
    assert list(grp.values()) == [green, red]
    assert list(grp.items()) == [("green", green), ("red", red)]


def test_group_repr(group):
    """``repr`` lists the fiber count and each per-color recording."""
    grp, *_ = group
    r = repr(grp)
    assert "3 fiber(s)" in r
    assert "[green]" in r
    assert "[red]" in r
