"""Pytest tests for ``fiber_mosaic.core.base``."""

from __future__ import annotations

import numpy as np
import pytest

from fiber_mosaic.core.base import (BaseFiberPhotometryExtractor,
                                    BaseFiberPhotometrySegment,
                                    FiberPhotometryRecordingGroup)


class _ArraySegment(BaseFiberPhotometrySegment):
    """Minimal in-memory segment used to drive tests."""

    def __init__(self, traces: np.ndarray, sampling_frequency: float) -> None:
        """Wrap a 2D ``(n_samples, n_fibers)`` numpy array."""
        super().__init__(sampling_frequency=sampling_frequency)
        self._traces = traces

    def get_traces(
        self,
        start_frame: int | None = None,
        end_frame: int | None = None,
        channel_indices: list | np.ndarray | None = None,
    ) -> np.ndarray:
        """Return an optionally sliced view of the backing array."""
        traces = self._traces[start_frame:end_frame]
        if channel_indices is not None:
            traces = traces[:, channel_indices]
        return traces

    def get_num_samples(self) -> int:
        """Return the number of time samples in the backing array."""
        return self._traces.shape[0]


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
        _ArraySegment(traces=traces, sampling_frequency=sampling_frequency)
    )
    return rec, traces


@pytest.fixture
def recording():
    """Default 10x3 recording reused across extractor tests."""
    return _make_recording()


@pytest.fixture
def segment():
    """10x2 ramp segment reused across segment tests."""
    traces = np.arange(20, dtype="float32").reshape(10, 2)
    return _ArraySegment(traces=traces, sampling_frequency=100.0), traces


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
        _ArraySegment(
            traces=np.zeros((4, 2), dtype="float32"),
            sampling_frequency=100.0,
        )
    )
    rec.add_segment(
        _ArraySegment(
            traces=np.ones((6, 2), dtype="float32"),
            sampling_frequency=100.0,
        )
    )
    assert rec.get_num_segments() == 2
    assert rec.get_fluorescence(segment_index=0).shape == (4, 2)
    assert rec.get_fluorescence(segment_index=1).shape == (6, 2)


# ---------------- BaseFiberPhotometrySegment ----------------


def test_segment_get_fluorescence_renames_fiber_indices(segment):
    """``fiber_indices`` is forwarded to ``get_traces`` correctly."""
    seg, traces = segment
    np.testing.assert_array_equal(
        seg.get_fluorescence(fiber_indices=[1]),
        traces[:, [1]],
    )


def test_segment_get_fluorescence_frame_slice(segment):
    """Frame range on the segment slices along the time axis."""
    seg, traces = segment
    np.testing.assert_array_equal(
        seg.get_fluorescence(start_frame=3, end_frame=7),
        traces[3:7],
    )


def test_segment_get_fluorescence_full_matches_get_traces(segment):
    """No-arg fluorescence equals no-arg get_traces on a segment."""
    seg, _ = segment
    np.testing.assert_array_equal(seg.get_fluorescence(), seg.get_traces())


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
