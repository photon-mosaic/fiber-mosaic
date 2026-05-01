"""Unit tests for ``fiber_mosaic.base``."""

from __future__ import annotations

import unittest

import numpy as np

from fiber_mosaic.base import (
    BaseFiberPhotometryExtractor,
    FiberPhotometryRecordingGroup,
)
from fiber_mosaic.baseextractor import NumpyFiberPhotometrySegment


def _make_recording(
    fiber_ids=("f0", "f1", "f2"),
    color="green",
    n_samples=10,
    sampling_frequency=100.0,
    dtype="float32",
):
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
    seg = NumpyFiberPhotometrySegment(
        traces=traces, sampling_frequency=sampling_frequency
    )
    rec.add_segment(seg)
    return rec, traces


class TestBaseFiberPhotometryExtractor(unittest.TestCase):
    def test_construction_sets_color_and_fibers(self):
        rec, _ = _make_recording()
        self.assertEqual(rec.color, "green")
        self.assertEqual(rec.get_num_fibers(), 3)
        np.testing.assert_array_equal(
            rec.fiber_ids, np.array(["f0", "f1", "f2"])
        )
        np.testing.assert_array_equal(rec.get_fiber_ids(), rec.fiber_ids)

    def test_kwargs_round_trip_metadata(self):
        rec, _ = _make_recording()
        self.assertEqual(rec._kwargs["color"], "green")
        self.assertEqual(rec._kwargs["fiber_ids"], ["f0", "f1", "f2"])

    def test_get_fluorescence_full(self):
        rec, traces = _make_recording()
        np.testing.assert_array_equal(rec.get_fluorescence(), traces)

    def test_get_fluorescence_frame_range(self):
        rec, traces = _make_recording()
        np.testing.assert_array_equal(
            rec.get_fluorescence(start_frame=2, end_frame=5),
            traces[2:5],
        )

    def test_get_fluorescence_fiber_subset(self):
        rec, traces = _make_recording()
        np.testing.assert_array_equal(
            rec.get_fluorescence(fiber_ids=["f0", "f2"]),
            traces[:, [0, 2]],
        )

    def test_get_fluorescence_matches_get_traces(self):
        rec, _ = _make_recording()
        np.testing.assert_array_equal(
            rec.get_fluorescence(), rec.get_traces()
        )

    def test_repr_includes_color_and_fiber_count(self):
        rec, _ = _make_recording()
        r = repr(rec)
        self.assertIn("color=green", r)
        self.assertIn("3 fiber", r)

    def test_multi_segment(self):
        rec = BaseFiberPhotometryExtractor(
            sampling_frequency=100.0,
            fiber_ids=["f0", "f1"],
            color="red",
            dtype="float32",
        )
        seg0 = NumpyFiberPhotometrySegment(
            traces=np.zeros((4, 2), dtype="float32"),
            sampling_frequency=100.0,
        )
        seg1 = NumpyFiberPhotometrySegment(
            traces=np.ones((6, 2), dtype="float32"),
            sampling_frequency=100.0,
        )
        rec.add_segment(seg0)
        rec.add_segment(seg1)
        self.assertEqual(rec.get_num_segments(), 2)
        self.assertEqual(
            rec.get_fluorescence(segment_index=0).shape, (4, 2)
        )
        self.assertEqual(
            rec.get_fluorescence(segment_index=1).shape, (6, 2)
        )


class TestBaseFiberPhotometrySegment(unittest.TestCase):
    """Exercised via the concrete ``NumpyFiberPhotometrySegment`` subclass."""

    def setUp(self):
        self.traces = np.arange(20, dtype="float32").reshape(10, 2)
        self.seg = NumpyFiberPhotometrySegment(
            traces=self.traces, sampling_frequency=100.0
        )

    def test_get_fluorescence_renames_fiber_indices(self):
        np.testing.assert_array_equal(
            self.seg.get_fluorescence(fiber_indices=[1]),
            self.traces[:, [1]],
        )

    def test_get_fluorescence_frame_slice(self):
        np.testing.assert_array_equal(
            self.seg.get_fluorescence(start_frame=3, end_frame=7),
            self.traces[3:7],
        )

    def test_get_fluorescence_full_matches_get_traces(self):
        np.testing.assert_array_equal(
            self.seg.get_fluorescence(), self.seg.get_traces()
        )


class TestFiberPhotometryRecordingGroup(unittest.TestCase):
    def _build_group(self):
        green, _ = _make_recording(color="green")
        red, _ = _make_recording(color="red")
        group = FiberPhotometryRecordingGroup(
            {"green": green, "red": red}
        )
        return group, green, red

    def test_empty_recordings_raises(self):
        with self.assertRaises(ValueError):
            FiberPhotometryRecordingGroup({})

    def test_mismatched_fiber_sets_raises(self):
        g, _ = _make_recording(fiber_ids=["a", "b"])
        r, _ = _make_recording(fiber_ids=["a", "c"])
        with self.assertRaises(ValueError):
            FiberPhotometryRecordingGroup({"green": g, "red": r})

    def test_colors_and_fiber_ids(self):
        group, *_ = self._build_group()
        self.assertEqual(group.colors, ["green", "red"])
        np.testing.assert_array_equal(
            group.fiber_ids, np.array(["f0", "f1", "f2"])
        )
        self.assertEqual(group.get_num_fibers(), 3)

    def test_indexing_and_lookup(self):
        group, green, red = self._build_group()
        self.assertIs(group["green"], green)
        self.assertIs(group.get_recording("red"), red)
        self.assertIn("green", group)
        self.assertNotIn("blue", group)
        self.assertEqual(len(group), 2)
        self.assertEqual(list(iter(group)), ["green", "red"])

    def test_dict_like_methods(self):
        group, green, red = self._build_group()
        self.assertEqual(list(group.keys()), ["green", "red"])
        self.assertEqual(list(group.values()), [green, red])
        self.assertEqual(
            list(group.items()), [("green", green), ("red", red)]
        )

    def test_repr(self):
        group, *_ = self._build_group()
        r = repr(group)
        self.assertIn("3 fiber(s)", r)
        self.assertIn("[green]", r)
        self.assertIn("[red]", r)


if __name__ == "__main__":
    unittest.main()
