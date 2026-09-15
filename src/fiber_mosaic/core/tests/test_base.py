"""Pytest tests for ``fiber_mosaic.core.base``."""

from __future__ import annotations

import numpy as np
import pytest
from spikeinterface.core import BaseRecording
from spikeinterface.core.numpyextractors import NumpyRecordingSegment

from fiber_mosaic.core.base import (
    BaseFiberPhotometryExtractor,
    FiberPhotometryMixin,
    FiberPhotometryRecordingGroup,
    recording_from_traces,
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
    times = np.arange(n_samples * n_fibers, dtype="float64").reshape(
        n_samples, n_fibers
    )
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
    times = np.arange(n_samples * n_fibers, dtype="float64").reshape(
        n_samples, n_fibers
    )
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


# ---------------- FiberPhotometryMixin ----------------


class _PlainFiberRecording(FiberPhotometryMixin, BaseRecording):
    """A non-extractor host for the mixin, standing in for a preprocessor.

    Mirrors what a SpikeInterface preprocessor looks like from the mixin's
    point of view: a plain ``BaseRecording`` that never runs
    ``BaseFiberPhotometryExtractor.__init__``, so it has no ``color``
    attribute and none of the extractor's source-side machinery.
    """

    def __init__(self, sampling_frequency, fiber_ids, color, dtype):
        BaseRecording.__init__(
            self,
            sampling_frequency=sampling_frequency,
            channel_ids=list(fiber_ids),
            dtype=dtype,
        )
        self.annotate(color=color)


def _make_plain_recording(n_samples=10, fiber_ids=("f0", "f1", "f2")):
    """Build a mixin-only recording with deterministic ramp traces."""
    sampling_frequency = 100.0
    rec = _PlainFiberRecording(
        sampling_frequency=sampling_frequency,
        fiber_ids=fiber_ids,
        color="green",
        dtype="float32",
    )
    traces = np.arange(n_samples * len(fiber_ids), dtype="float32").reshape(
        n_samples, len(fiber_ids)
    )
    segment = NumpyRecordingSegment(
        traces=traces,
        sampling_frequency=sampling_frequency,
        t_start=None,
    )
    rec._recording_segments.append(segment)
    segment.set_parent_extractor(rec)
    return rec, traces


def test_mixin_provides_fiber_api_off_a_plain_recording():
    """The fiber API works on a host that is not an extractor.

    This is the point of the split: a preprocessor cannot inherit from
    BaseFiberPhotometryExtractor, so the fiber-native surface has to work
    when mixed into a plain BaseRecording.
    """
    rec, traces = _make_plain_recording()

    assert rec.color == "green"
    assert rec.get_num_fibers() == 3
    np.testing.assert_array_equal(rec.fiber_ids, np.array(["f0", "f1", "f2"]))
    np.testing.assert_array_equal(rec.get_fiber_ids(), rec.fiber_ids)
    np.testing.assert_array_equal(rec.get_fluorescence(), traces)
    assert "color=green" in repr(rec)


def test_mixin_per_fiber_times_round_trip():
    """set_times/get_fiber_times work off a non-extractor host too."""
    rec, _ = _make_plain_recording()
    times = np.linspace(0.0, 1.0, rec.get_num_samples())[:, None] + np.array(
        [0.0, 0.001, 0.002]
    )

    assert not rec.has_fiber_times()
    rec.set_times(times)
    assert rec.has_fiber_times()
    np.testing.assert_allclose(rec.get_fiber_times(), times)


def test_color_survives_copy_metadata(recording):
    """Color rides on an annotation, so copy_metadata carries it.

    SpikeInterface's BasePreprocessor propagates metadata with
    ``copy_metadata(only_main=False)``, which copies annotations but not
    attributes -- so the color has to be an annotation to reach a
    preprocessor at all.
    """
    rec, _ = recording
    other = BaseFiberPhotometryExtractor(
        sampling_frequency=rec.get_sampling_frequency(),
        fiber_ids=list(rec.get_channel_ids()),
        color="placeholder",
        dtype=rec.get_dtype(),
    )

    rec.copy_metadata(other, only_main=False)

    assert other.get_annotation("color") == "green"
    assert other.color == "green"


# ---------------- recording_from_traces ----------------


def test_recording_from_traces_without_timestamps():
    """No ``timestamps`` -> nominal times, from ``sampling_frequency``."""
    traces = np.ones((10, 2), dtype="float32")
    rec = recording_from_traces(
        traces, color="green", sampling_frequency=100.0
    )

    assert isinstance(rec, FiberPhotometryMixin)
    assert rec.color == "green"
    assert not rec.has_fiber_times()


def test_recording_from_traces_with_1d_timestamps():
    """A bare 1-D ``timestamps`` array sets real times, single segment."""
    traces = np.ones((10, 2), dtype="float32")
    times = np.linspace(0.0, 0.09, 10) + 0.001  # jittery, not nominal

    rec = recording_from_traces(traces, color="green", timestamps=times)

    assert rec.has_fiber_times()
    np.testing.assert_allclose(rec.get_fiber_times()[:, 0], times)
    np.testing.assert_allclose(rec.get_fiber_times()[:, 1], times)


def test_recording_from_traces_with_per_segment_timestamps():
    """A list of ``timestamps`` sets real times per segment."""
    traces = [np.ones((5, 2)), np.zeros((4, 2))]
    times = [np.linspace(0.0, 0.04, 5), np.linspace(1.0, 1.03, 4)]

    rec = recording_from_traces(traces, color="red", timestamps=times)

    assert rec.has_fiber_times(segment_index=0)
    assert rec.has_fiber_times(segment_index=1)
    np.testing.assert_allclose(
        rec.get_fiber_times(segment_index=1)[:, 0], times[1]
    )


def test_recording_from_traces_sets_si_t_start():
    """Real ``timestamps`` also set SI's own per-segment ``t_start``.

    So ``get_times()`` (the plain, nominal SI API) at least starts at the
    right time instead of at zero.
    """
    traces = [np.ones((5, 2)), np.zeros((4, 2))]
    times = [np.linspace(1.0, 1.04, 5), np.linspace(2.0, 2.03, 4)]

    rec = recording_from_traces(traces, color="red", timestamps=times)

    np.testing.assert_allclose(rec.get_times(segment_index=0)[0], times[0][0])
    np.testing.assert_allclose(rec.get_times(segment_index=1)[0], times[1][0])


def test_recording_from_traces_with_tuple_traces():
    """A tuple of per-segment arrays works, not just a list."""
    traces = (np.ones((5, 2)), np.zeros((4, 2)))

    rec = recording_from_traces(traces, color="red")

    assert rec.get_num_segments() == 2
    assert rec.get_num_samples(1) == 4


def test_recording_from_traces_with_plain_list_timestamps():
    """A plain list (not ``np.ndarray``) is one segment's timestamps."""
    traces = np.ones((5, 2), dtype="float32")
    times = [0.0, 0.01, 0.02, 0.03, 0.04]

    rec = recording_from_traces(traces, color="green", timestamps=times)

    assert rec.has_fiber_times()
    np.testing.assert_allclose(rec.get_fiber_times()[:, 0], times)


def test_recording_from_traces_timestamps_segment_count_mismatch():
    """Too few/many ``timestamps`` arrays raises rather than silently
    leaving segments on nominal times."""
    traces = [np.ones((5, 2)), np.zeros((4, 2))]
    times = [np.linspace(0.0, 0.04, 5)]  # only 1 array for 2 segments

    with pytest.raises(ValueError, match="one array per segment"):
        recording_from_traces(traces, color="red", timestamps=times)


def test_recording_from_traces_with_nested_list_2d_timestamps():
    """A single 2-D segment as nested lists isn't split into per-row
    segments -- ``num_segments`` (from ``traces``), not ``timestamps``'
    own shape, decides how it's read."""
    traces = np.ones((3, 2), dtype="float32")
    times = [[0.0, 0.001], [0.01, 0.011], [0.02, 0.021]]

    rec = recording_from_traces(traces, color="green", timestamps=times)

    assert rec.get_num_segments() == 1
    np.testing.assert_allclose(rec.get_fiber_times(), times)


def test_recording_from_traces_empty_timestamps_raises_cleanly():
    """Empty ``timestamps`` for a non-empty segment raises a clear
    length-mismatch error from ``set_times``, not a raw ``IndexError``
    from the ``t_start`` computation."""
    traces = np.ones((5, 2), dtype="float32")

    with pytest.raises(ValueError, match="must match"):
        recording_from_traces(traces, color="green", timestamps=np.array([]))


def test_recording_from_traces_3d_timestamps_raises_cleanly():
    """A malformed (3-D) ``timestamps`` raises ``set_times``'s own
    dimension error, not an ``IndexError``/``TypeError`` from computing
    ``t_start``."""
    traces = np.ones((5, 2), dtype="float32")
    times = np.zeros((5, 2, 1))

    with pytest.raises(ValueError, match="1D or 2D"):
        recording_from_traces(traces, color="green", timestamps=times)


def test_recording_from_traces_2d_zero_columns_timestamps_raises_cleanly():
    """A 2-D ``timestamps`` with zero columns raises ``set_times``'s own
    shape-mismatch error, not an ``IndexError`` from computing
    ``t_start``."""
    traces = np.ones((5, 2), dtype="float32")
    times = np.zeros((5, 0))

    with pytest.raises(ValueError, match="must match"):
        recording_from_traces(traces, color="green", timestamps=times)


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
