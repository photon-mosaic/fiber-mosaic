"""
Tests for fiber-mosaic extractors.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest


class TestCsvExtractor:
    """Tests for CSV extractor."""

    def test_csv_extractor_basic(self):
        """Test basic CSV file loading."""
        from fiber_mosaic.extractors import CsvFiberPhotometryExtractor

        # Create test CSV file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            f.write("time,fiber1,fiber2\n")
            for i in range(100):
                r1, r2 = np.random.randn(), np.random.randn()
                f.write(f"{i * 0.01},{r1},{r2}\n")
            csv_path = f.name

        try:
            # Load with extractor
            rec = CsvFiberPhotometryExtractor(csv_path, color="green")

            # Verify basic properties
            assert rec.get_num_fibers() == 2
            assert rec.get_num_samples() == 100
            assert rec.color == "green"
            assert list(rec.fiber_ids) == ["fiber1", "fiber2"]

            # Verify data access
            traces = rec.get_fluorescence()
            assert traces.shape == (100, 2)

            # Verify timestamps
            times = rec.get_fiber_times()
            assert times.shape == (100, 2)
            assert np.allclose(times[0, 0], 0.0)
        finally:
            Path(csv_path).unlink()

    def test_csv_extractor_get_streams(self):
        """Test stream discovery."""
        from fiber_mosaic.extractors import CsvFiberPhotometryExtractor

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            f.write("time,signal,control\n")
            f.write("0,1.0,0.5\n")
            csv_path = f.name

        try:
            names, ids = CsvFiberPhotometryExtractor.get_streams(csv_path)
            assert "signal" in names
            assert "control" in names
        finally:
            Path(csv_path).unlink()


class TestBaseFiberPhotometryExtractor:
    """Tests for base extractor functionality."""

    def test_per_fiber_times(self):
        """Test per-fiber timestamp handling."""
        from spikeinterface.core.numpyextractors import NumpyRecordingSegment

        from fiber_mosaic import BaseFiberPhotometryExtractor

        # Create simple recording
        n_samples = 100
        n_fibers = 3
        sampling_rate = 100.0

        traces = np.random.randn(n_samples, n_fibers)

        class TestExtractor(BaseFiberPhotometryExtractor):
            pass

        rec = TestExtractor(
            sampling_frequency=sampling_rate,
            fiber_ids=["f1", "f2", "f3"],
            color="green",
        )
        segment = NumpyRecordingSegment(traces, sampling_rate)
        rec.add_segment(segment)

        # Test 1D times (broadcast to all fibers)
        times_1d = np.arange(n_samples) / sampling_rate
        rec.set_times(times_1d)
        assert rec.has_fiber_times()

        retrieved = rec.get_fiber_times()
        assert retrieved.shape == (n_samples, n_fibers)
        np.testing.assert_allclose(retrieved[:, 0], times_1d)

        # Test 2D times (per-fiber)
        times_2d = np.column_stack([
            times_1d,
            times_1d + 0.001,  # Slightly offset
            times_1d + 0.002,
        ])
        rec.set_times(times_2d)

        retrieved = rec.get_fiber_times()
        np.testing.assert_allclose(retrieved, times_2d)


class TestFiberPhotometryRecordingGroup:
    """Tests for multi-color recording group."""

    def test_group_creation(self):
        """Test creating a recording group."""
        from spikeinterface.core.numpyextractors import NumpyRecordingSegment

        from fiber_mosaic import (
            BaseFiberPhotometryExtractor,
            FiberPhotometryRecordingGroup,
        )

        # Create two recordings with same fibers
        n_samples = 100
        sampling_rate = 100.0
        fiber_ids = ["NAc", "DMS"]

        class TestExtractor(BaseFiberPhotometryExtractor):
            pass

        green = TestExtractor(
            sampling_frequency=sampling_rate,
            fiber_ids=fiber_ids,
            color="green",
        )
        green.add_segment(NumpyRecordingSegment(
            np.random.randn(n_samples, 2), sampling_rate
        ))

        iso = TestExtractor(
            sampling_frequency=sampling_rate,
            fiber_ids=fiber_ids,
            color="iso",
        )
        iso.add_segment(NumpyRecordingSegment(
            np.random.randn(n_samples, 2), sampling_rate
        ))

        # Create group
        group = FiberPhotometryRecordingGroup({"green": green, "iso": iso})

        # Verify properties
        assert group.colors == ["green", "iso"]
        assert list(group.fiber_ids) == fiber_ids
        assert group.get_num_fibers() == 2
        assert len(group) == 2

        # Verify access
        assert group["green"] is green
        assert group.get_recording("iso") is iso
        assert "green" in group

    def test_group_fiber_validation(self):
        """Test that mismatched fiber IDs raise error."""
        from spikeinterface.core.numpyextractors import NumpyRecordingSegment

        from fiber_mosaic import (
            BaseFiberPhotometryExtractor,
            FiberPhotometryRecordingGroup,
        )

        class TestExtractor(BaseFiberPhotometryExtractor):
            pass

        rec1 = TestExtractor(
            sampling_frequency=100.0,
            fiber_ids=["f1", "f2"],
            color="green",
        )
        rec1.add_segment(NumpyRecordingSegment(np.zeros((10, 2)), 100.0))

        rec2 = TestExtractor(
            sampling_frequency=100.0,
            fiber_ids=["f3", "f4"],  # Different fibers!
            color="iso",
        )
        rec2.add_segment(NumpyRecordingSegment(np.zeros((10, 2)), 100.0))

        with pytest.raises(ValueError, match="different fiber_ids"):
            FiberPhotometryRecordingGroup({"green": rec1, "iso": rec2})


class TestFormatDetection:
    """Tests for format detection."""

    def test_detect_csv_format(self):
        """Test CSV format detection."""
        from fiber_mosaic.extractors import detect_formats

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a simple CSV
            csv_path = Path(tmpdir) / "data.csv"
            csv_path.write_text("timestamps,data,sampling_rate\n0,1.0,100\n")

            formats = detect_formats(tmpdir)
            assert "csv" in formats

    def test_detect_nwb_format(self):
        """Test NWB format detection."""
        from fiber_mosaic.extractors import detect_formats

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create empty NWB file (just the extension for detection)
            nwb_path = Path(tmpdir) / "session.nwb"
            nwb_path.touch()

            formats = detect_formats(tmpdir)
            assert "nwb" in formats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
