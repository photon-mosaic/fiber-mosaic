"""
Tests for fiber-mosaic extractors.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# =============================================================================
# CSV Extractor Tests
# =============================================================================


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
            rec = CsvFiberPhotometryExtractor(
                csv_path, color="green", time_column="time"
            )

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
            names, ids = CsvFiberPhotometryExtractor.get_streams(
                csv_path, time_column="time"
            )
            assert "signal" in names
            assert "control" in names
        finally:
            Path(csv_path).unlink()

    def test_csv_get_streams_custom_time_column(self):
        """A custom-named time column is excluded when passed to discovery."""
        from fiber_mosaic.extractors import CsvFiberPhotometryExtractor

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            f.write("t_sec,signal,control\n")
            f.write("0,1.0,0.5\n")
            csv_path = f.name

        try:
            # Without the hint, the time column shows up as a data stream.
            names, _ = CsvFiberPhotometryExtractor.get_streams(csv_path)
            assert "t_sec" in names

            # Passing time_column excludes it, matching what the loader uses.
            names, _ = CsvFiberPhotometryExtractor.get_streams(
                csv_path, time_column="t_sec"
            )
            assert "t_sec" not in names
            assert names == ["signal", "control"]
        finally:
            Path(csv_path).unlink()

    def test_csv_fiber_pattern_selects_only_data_columns(self):
        """AIND FIP layout: fiber_pattern selects Fiber_* and drops
        metadata."""
        from fiber_mosaic.extractors import CsvFiberPhotometryExtractor

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            # FIP-style columns: a time column, metadata, and fiber channels.
            f.write(
                "ReferenceTime,CameraFrameNumber,Background,Fiber_0,Fiber_1\n"
            )
            f.write("0.00,0,10.0,1.0,2.0\n")
            f.write("0.01,1,10.1,1.1,2.1\n")
            csv_path = f.name

        try:
            rec = CsvFiberPhotometryExtractor(
                csv_path,
                color="green",
                time_column="ReferenceTime",
                fiber_pattern="Fiber_*",
            )
            # Only the two fiber channels are loaded; metadata is ignored.
            assert list(rec.fiber_ids) == ["Fiber_0", "Fiber_1"]
            assert rec.get_num_fibers() == 2
            assert "Background" not in rec.fiber_ids
            assert "CameraFrameNumber" not in rec.fiber_ids
        finally:
            Path(csv_path).unlink()

    def test_csv_fiber_pattern_no_match_raises(self):
        """A pattern that matches nothing is a loud error, not empty data."""
        from fiber_mosaic.extractors import CsvFiberPhotometryExtractor

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            f.write("time,signal\n0.0,1.0\n0.1,2.0\n")
            csv_path = f.name

        try:
            with pytest.raises(ValueError, match="matched no columns"):
                CsvFiberPhotometryExtractor(
                    csv_path,
                    color="green",
                    time_column="time",
                    fiber_pattern="Fiber_*",
                )
        finally:
            Path(csv_path).unlink()

    def test_csv_get_streams_fiber_pattern(self):
        """Discovery can positively filter streams by fiber_pattern."""
        from fiber_mosaic.extractors import CsvFiberPhotometryExtractor

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            f.write(
                "ReferenceTime,CameraFrameNumber,Background,Fiber_0,Fiber_1\n"
            )
            f.write("0.00,0,10.0,1.0,2.0\n")
            csv_path = f.name

        try:
            names, _ = CsvFiberPhotometryExtractor.get_streams(
                csv_path, time_column="ReferenceTime", fiber_pattern="Fiber_*"
            )
            assert names == ["Fiber_0", "Fiber_1"]
        finally:
            Path(csv_path).unlink()

    def test_csv_extractor_file_not_found(self):
        """Test that FileNotFoundError is raised for missing file."""
        from fiber_mosaic.extractors import CsvFiberPhotometryExtractor

        with pytest.raises(FileNotFoundError, match="not found"):
            CsvFiberPhotometryExtractor(
                "/nonexistent/file.csv", color="green", time_column="time"
            )

    def test_csv_extractor_no_time_column(self):
        """Test error when the specified time column is not found."""
        from fiber_mosaic.extractors import CsvFiberPhotometryExtractor

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            f.write("data1,data2\n")
            f.write("1.0,2.0\n")
            csv_path = f.name

        try:
            with pytest.raises(
                ValueError, match="Time column 'time' not found"
            ):
                CsvFiberPhotometryExtractor(
                    csv_path, color="green", time_column="time"
                )
        finally:
            Path(csv_path).unlink()

    def test_csv_extractor_no_fiber_columns(self):
        """Test error when no fiber columns are found."""
        from fiber_mosaic.extractors import CsvFiberPhotometryExtractor

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            f.write("time\n")
            f.write("0.0\n")
            csv_path = f.name

        try:
            with pytest.raises(ValueError, match="No fiber columns found"):
                CsvFiberPhotometryExtractor(
                    csv_path, color="green", time_column="time"
                )
        finally:
            Path(csv_path).unlink()

    def test_csv_extractor_missing_fiber_columns(self):
        """Test error when specified fiber columns don't exist."""
        from fiber_mosaic.extractors import CsvFiberPhotometryExtractor

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            f.write("time,data\n")
            f.write("0.0,1.0\n")
            csv_path = f.name

        try:
            with pytest.raises(ValueError, match="not found in CSV"):
                CsvFiberPhotometryExtractor(
                    csv_path,
                    color="green",
                    time_column="time",
                    fiber_columns=["nonexistent"],
                )
        finally:
            Path(csv_path).unlink()

    def test_csv_extractor_single_sample(self):
        """Test error with single sample and no sampling frequency."""
        from fiber_mosaic.extractors import CsvFiberPhotometryExtractor

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            f.write("time,data\n")
            f.write("0.0,1.0\n")
            csv_path = f.name

        try:
            with pytest.raises(ValueError, match="Cannot compute"):
                CsvFiberPhotometryExtractor(
                    csv_path, color="green", time_column="time"
                )
        finally:
            Path(csv_path).unlink()

    def test_csv_extractor_with_sampling_frequency(self):
        """Test CSV extractor with explicit sampling frequency."""
        from fiber_mosaic.extractors import CsvFiberPhotometryExtractor

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            f.write("time,data\n")
            f.write("0.0,1.0\n")
            csv_path = f.name

        try:
            rec = CsvFiberPhotometryExtractor(
                csv_path,
                color="green",
                time_column="time",
                sampling_frequency=100.0,
            )
            assert rec.get_sampling_frequency() == 100.0
        finally:
            Path(csv_path).unlink()

    def test_csv_extractor_alternate_time_columns(self):
        """The time column must be named explicitly to match the file."""
        from fiber_mosaic.extractors import CsvFiberPhotometryExtractor

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            f.write("timestamps,data\n")
            f.write("0.0,1.0\n")
            f.write("0.1,2.0\n")
            csv_path = f.name

        try:
            # Wrong column name raises a clear error.
            with pytest.raises(
                ValueError, match="Time column 'time' not found"
            ):
                CsvFiberPhotometryExtractor(
                    csv_path, color="green", time_column="time"
                )

            # Correct column name works.
            rec = CsvFiberPhotometryExtractor(
                csv_path, color="green", time_column="timestamps"
            )
            assert rec.get_num_samples() == 2
        finally:
            Path(csv_path).unlink()

    def test_read_csv_convenience_function(self):
        """Test the convenience function."""
        from fiber_mosaic.extractors.csv_extractor import (
            read_csv_fiber_photometry,
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            f.write("time,data\n")
            f.write("0.0,1.0\n")
            f.write("0.1,2.0\n")
            csv_path = f.name

        try:
            rec = read_csv_fiber_photometry(
                csv_path, color="green", time_column="time"
            )
            assert rec.get_num_samples() == 2
        finally:
            Path(csv_path).unlink()


# =============================================================================
# Base Extractor Tests
# =============================================================================


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
        segment = NumpyRecordingSegment(traces, sampling_rate, t_start=0.0)
        rec.add_segment(segment)

        # Test 1D times (broadcast to all fibers)
        times_1d = np.arange(n_samples) / sampling_rate
        rec.set_times(times_1d)
        assert rec.has_fiber_times()

        retrieved = rec.get_fiber_times()
        assert retrieved.shape == (n_samples, n_fibers)
        np.testing.assert_allclose(retrieved[:, 0], times_1d)

        # Test 2D times (per-fiber)
        times_2d = np.column_stack(
            [
                times_1d,
                times_1d + 0.001,  # Slightly offset
                times_1d + 0.002,
            ]
        )
        rec.set_times(times_2d)

        retrieved = rec.get_fiber_times()
        np.testing.assert_allclose(retrieved, times_2d)

    def test_set_times_invalid_1d_length(self):
        """Test error with wrong length 1D times."""
        from spikeinterface.core.numpyextractors import NumpyRecordingSegment

        from fiber_mosaic import BaseFiberPhotometryExtractor

        class TestExtractor(BaseFiberPhotometryExtractor):
            pass

        rec = TestExtractor(
            sampling_frequency=100.0,
            fiber_ids=["f1", "f2"],
            color="green",
        )
        segment = NumpyRecordingSegment(np.zeros((100, 2)), 100.0, t_start=0.0)
        rec.add_segment(segment)

        with pytest.raises(ValueError, match="1D times array length"):
            rec.set_times(np.arange(50))  # Wrong length

    def test_set_times_invalid_2d_shape(self):
        """Test error with wrong shape 2D times."""
        from spikeinterface.core.numpyextractors import NumpyRecordingSegment

        from fiber_mosaic import BaseFiberPhotometryExtractor

        class TestExtractor(BaseFiberPhotometryExtractor):
            pass

        rec = TestExtractor(
            sampling_frequency=100.0,
            fiber_ids=["f1", "f2"],
            color="green",
        )
        segment = NumpyRecordingSegment(np.zeros((100, 2)), 100.0, t_start=0.0)
        rec.add_segment(segment)

        with pytest.raises(ValueError, match="2D times array shape"):
            rec.set_times(np.zeros((100, 3)))  # Wrong shape

    def test_set_times_invalid_ndim(self):
        """Test error with 3D times array."""
        from spikeinterface.core.numpyextractors import NumpyRecordingSegment

        from fiber_mosaic import BaseFiberPhotometryExtractor

        class TestExtractor(BaseFiberPhotometryExtractor):
            pass

        rec = TestExtractor(
            sampling_frequency=100.0,
            fiber_ids=["f1", "f2"],
            color="green",
        )
        segment = NumpyRecordingSegment(np.zeros((100, 2)), 100.0, t_start=0.0)
        rec.add_segment(segment)

        with pytest.raises(ValueError, match="must be 1D or 2D"):
            rec.set_times(np.zeros((100, 2, 1)))  # 3D array

    def test_get_fiber_times_no_times_set(self):
        """Test get_fiber_times falls back to nominal timestamps."""
        from spikeinterface.core.numpyextractors import NumpyRecordingSegment

        from fiber_mosaic import BaseFiberPhotometryExtractor

        class TestExtractor(BaseFiberPhotometryExtractor):
            pass

        rec = TestExtractor(
            sampling_frequency=100.0,
            fiber_ids=["f1", "f2"],
            color="green",
        )
        segment = NumpyRecordingSegment(np.zeros((100, 2)), 100.0, t_start=0.0)
        rec.add_segment(segment)

        # Don't set times - should fall back to nominal
        assert not rec.has_fiber_times()
        times = rec.get_fiber_times()
        assert times.shape == (100, 2)

    def test_get_fiber_times_with_fiber_ids(self):
        """Test get_fiber_times with specific fiber IDs."""
        from spikeinterface.core.numpyextractors import NumpyRecordingSegment

        from fiber_mosaic import BaseFiberPhotometryExtractor

        class TestExtractor(BaseFiberPhotometryExtractor):
            pass

        rec = TestExtractor(
            sampling_frequency=100.0,
            fiber_ids=["f1", "f2", "f3"],
            color="green",
        )
        segment = NumpyRecordingSegment(np.zeros((100, 3)), 100.0, t_start=0.0)
        rec.add_segment(segment)

        # Get times for specific fibers
        times = rec.get_fiber_times(fiber_ids=["f1", "f2"])
        assert times.shape == (100, 2)

    def test_get_fiber_times_with_range(self):
        """Test get_fiber_times with frame range."""
        from spikeinterface.core.numpyextractors import NumpyRecordingSegment

        from fiber_mosaic import BaseFiberPhotometryExtractor

        class TestExtractor(BaseFiberPhotometryExtractor):
            pass

        rec = TestExtractor(
            sampling_frequency=100.0,
            fiber_ids=["f1", "f2"],
            color="green",
        )
        segment = NumpyRecordingSegment(np.zeros((100, 2)), 100.0, t_start=0.0)
        rec.add_segment(segment)
        rec.set_times(np.arange(100) / 100.0)

        times = rec.get_fiber_times(start_frame=10, end_frame=20)
        assert times.shape == (10, 2)

    def test_get_streams_not_implemented(self):
        """Test get_streams raises NotImplementedError on base class."""
        from fiber_mosaic import BaseFiberPhotometryExtractor

        with pytest.raises(NotImplementedError):
            BaseFiberPhotometryExtractor.get_streams("/path")

    def test_get_fiber_ids(self):
        """Test get_fiber_ids method."""
        from fiber_mosaic import BaseFiberPhotometryExtractor

        class TestExtractor(BaseFiberPhotometryExtractor):
            pass

        rec = TestExtractor(
            sampling_frequency=100.0,
            fiber_ids=["f1", "f2"],
            color="green",
        )
        assert list(rec.get_fiber_ids()) == ["f1", "f2"]


# =============================================================================
# Recording Group Tests
# =============================================================================


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
        green.add_segment(
            NumpyRecordingSegment(
                np.random.randn(n_samples, 2), sampling_rate, t_start=0.0
            )
        )

        iso = TestExtractor(
            sampling_frequency=sampling_rate,
            fiber_ids=fiber_ids,
            color="iso",
        )
        iso.add_segment(
            NumpyRecordingSegment(
                np.random.randn(n_samples, 2), sampling_rate, t_start=0.0
            )
        )

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
        rec1.add_segment(
            NumpyRecordingSegment(np.zeros((10, 2)), 100.0, t_start=0.0)
        )

        rec2 = TestExtractor(
            sampling_frequency=100.0,
            fiber_ids=["f3", "f4"],  # Different fibers!
            color="iso",
        )
        rec2.add_segment(
            NumpyRecordingSegment(np.zeros((10, 2)), 100.0, t_start=0.0)
        )

        with pytest.raises(ValueError, match="different fiber_ids"):
            FiberPhotometryRecordingGroup({"green": rec1, "iso": rec2})

    def test_group_empty_recordings(self):
        """Test error with empty recordings dict."""
        from fiber_mosaic import FiberPhotometryRecordingGroup

        with pytest.raises(ValueError, match="cannot be empty"):
            FiberPhotometryRecordingGroup({})

    def test_group_iteration(self):
        """Test iterating over group colors."""
        from spikeinterface.core.numpyextractors import NumpyRecordingSegment

        from fiber_mosaic import (
            BaseFiberPhotometryExtractor,
            FiberPhotometryRecordingGroup,
        )

        class TestExtractor(BaseFiberPhotometryExtractor):
            pass

        rec = TestExtractor(
            sampling_frequency=100.0,
            fiber_ids=["f1"],
            color="green",
        )
        rec.add_segment(
            NumpyRecordingSegment(np.zeros((10, 1)), 100.0, t_start=0.0)
        )

        group = FiberPhotometryRecordingGroup({"green": rec})

        # Test iteration
        colors = list(group)
        assert colors == ["green"]

        # Test keys/values/items
        assert list(group.keys()) == ["green"]
        assert list(group.values()) == [rec]
        assert list(group.items()) == [("green", rec)]

    def test_group_repr(self):
        """Test string representation of group."""
        from spikeinterface.core.numpyextractors import NumpyRecordingSegment

        from fiber_mosaic import (
            BaseFiberPhotometryExtractor,
            FiberPhotometryRecordingGroup,
        )

        class TestExtractor(BaseFiberPhotometryExtractor):
            pass

        rec = TestExtractor(
            sampling_frequency=100.0,
            fiber_ids=["f1", "f2"],
            color="green",
        )
        rec.add_segment(
            NumpyRecordingSegment(np.zeros((10, 2)), 100.0, t_start=0.0)
        )

        group = FiberPhotometryRecordingGroup({"green": rec})
        repr_str = repr(group)

        assert "FiberPhotometryRecordingGroup" in repr_str
        assert "1 colors" in repr_str
        assert "2 fibers" in repr_str


# =============================================================================
# Format Detection Tests
# =============================================================================


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

    def test_detect_tdt_format(self):
        """Test TDT format detection."""
        from fiber_mosaic.extractors import detect_formats

        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "data.tsq").touch()
            formats = detect_formats(tmpdir)
            assert "tdt" in formats

    def test_detect_doric_format(self):
        """Test Doric format detection."""
        from fiber_mosaic.extractors import detect_formats

        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "data.doric").touch()
            formats = detect_formats(tmpdir)
            assert "doric" in formats

    def test_detect_npm_format(self):
        """Test NPM format detection."""
        from fiber_mosaic.extractors import detect_formats

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "data.csv"
            csv_path.write_text(
                "Timestamp,LedState,Region0G,Region1G\n"
                "0,1,100.0,200.0\n"
                "1,2,101.0,201.0\n"
            )
            formats = detect_formats(tmpdir)
            assert "npm" in formats

    def test_detect_doric_csv_format(self):
        """Test Doric CSV format detection."""
        from fiber_mosaic.extractors import detect_formats

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "doric_data.csv"
            # Doric CSV has all-string header rows
            csv_path.write_text(
                "Console,I/O,AIn-1\nTime(s),Voltage,Signal\n0.0,1.0,100.0\n"
            )
            formats = detect_formats(tmpdir)
            assert "doric" in formats

    def test_classify_csv_file_event_csv(self):
        """Test classification of event CSV files."""
        from fiber_mosaic.extractors.detect_formats import _is_event_csv

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "event.csv"
            csv_path.write_text("timestamps\n1.0\n2.0\n")
            assert _is_event_csv(str(csv_path))

    def test_classify_csv_file_not_event(self):
        """Test classification of non-event CSV files."""
        from fiber_mosaic.extractors.detect_formats import _is_event_csv

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "data.csv"
            csv_path.write_text("time,data\n0,1\n")
            assert not _is_event_csv(str(csv_path))

    def test_classify_csv_single_column_invalid(self):
        """Test error for single-column CSV without 'timestamps' header."""
        from fiber_mosaic.extractors.detect_formats import classify_csv_file

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "data.csv"
            csv_path.write_text("other\n1.0\n")

            with pytest.raises(ValueError, match="timestamps"):
                classify_csv_file(str(csv_path))

    def test_classify_csv_three_column_invalid(self):
        """Test error for 3-column CSV without required headers."""
        from fiber_mosaic.extractors.detect_formats import classify_csv_file

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "data.csv"
            csv_path.write_text("a,b,c\n1,2,3\n")

            with pytest.raises(ValueError, match="3-column CSV"):
                classify_csv_file(str(csv_path))

    def test_is_float_helper(self):
        """Test _is_float helper function."""
        from fiber_mosaic.extractors.detect_formats import _is_float

        assert _is_float("1.0")
        assert _is_float(1.0)
        assert _is_float("123")
        assert not _is_float("abc")
        assert not _is_float(None)

    def test_detect_event_csv_with_npm(self):
        """Test that NPM event files are handled correctly."""
        from fiber_mosaic.extractors import detect_formats

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create NPM data file
            npm_path = Path(tmpdir) / "npm_data.csv"
            npm_path.write_text(
                "Timestamp,LedState,Region0G,Region1G\n0,1,100.0,200.0\n"
            )
            # Create NPM event file (should be suppressed)
            event_path = Path(tmpdir) / "event_data.csv"
            event_path.write_text("timestamps\n1.0\n")

            formats = detect_formats(tmpdir)
            assert "npm" in formats


# =============================================================================
# Doric Extractor Tests
# =============================================================================


class TestDoricExtractor:
    """Tests for Doric extractor."""

    def test_doric_csv_extractor(self):
        """Test reading Doric CSV file."""
        from fiber_mosaic.extractors import DoricFiberPhotometryExtractor

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "doric_data.csv"
            # Doric CSV: row 0 is metadata, row 1 is column headers
            csv_path.write_text(
                "Console,I/O,Signal\n"
                "Time(s),Voltage,Signal\n"
                "0.0,1.0,100.0\n"
                "0.1,1.0,101.0\n"
                "0.2,1.0,102.0\n"
            )

            rec = DoricFiberPhotometryExtractor(
                csv_path, stream_name="Signal", color="green"
            )
            assert rec.get_num_samples() == 3
            assert rec.color == "green"

    def test_doric_csv_auto_stream(self):
        """Test auto stream selection with single stream."""
        from fiber_mosaic.extractors import DoricFiberPhotometryExtractor

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "doric_data.csv"
            # Single data column (Signal) plus Time(s)
            csv_path.write_text(
                "Console,Signal\nTime(s),Signal\n0.0,100.0\n0.1,101.0\n"
            )

            rec = DoricFiberPhotometryExtractor(csv_path, color="green")
            assert rec.get_num_samples() == 2

    def test_doric_no_color_raises(self):
        """Test that omitting color raises a ValueError."""
        from fiber_mosaic.extractors import DoricFiberPhotometryExtractor

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "doric_data.csv"
            csv_path.write_text(
                "Console,Signal\nTime(s),Signal\n0.0,100.0\n0.1,101.0\n"
            )

            with pytest.raises(ValueError, match="color must be provided"):
                DoricFiberPhotometryExtractor(csv_path)

    def test_doric_csv_no_streams(self):
        """Test error when no streams found."""
        from fiber_mosaic.extractors import DoricFiberPhotometryExtractor

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "doric_data.csv"
            # Only Time(s) column, no data streams
            csv_path.write_text("Console\nTime(s)\n0.0\n")

            with pytest.raises(ValueError, match="No streams"):
                DoricFiberPhotometryExtractor(csv_path)

    def test_doric_csv_multiple_streams_no_name(self):
        """Test error when multiple streams and no name specified."""
        from fiber_mosaic.extractors import DoricFiberPhotometryExtractor

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "doric_data.csv"
            csv_path.write_text(
                "Console,I/O,Signal,Control\n"
                "Time(s),Voltage,Signal,Control\n"
                "0.0,1.0,100.0,50.0\n"
            )

            with pytest.raises(ValueError, match="Multiple streams"):
                DoricFiberPhotometryExtractor(csv_path)

    def test_doric_csv_stream_not_found(self):
        """Test error when specified stream not found."""
        from fiber_mosaic.extractors import DoricFiberPhotometryExtractor

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "doric_data.csv"
            csv_path.write_text("Console,Signal\nTime(s),Signal\n0.0,100.0\n")

            with pytest.raises(ValueError, match="not found"):
                DoricFiberPhotometryExtractor(
                    csv_path, stream_name="nonexistent"
                )

    def test_doric_get_streams(self):
        """Test stream discovery for Doric CSV."""
        from fiber_mosaic.extractors import DoricFiberPhotometryExtractor

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "doric_data.csv"
            csv_path.write_text(
                "Console,I/O,Signal,Control\n"
                "Time(s),Voltage,Signal,Control\n"
                "0.0,1.0,100.0,50.0\n"
            )

            names, ids = DoricFiberPhotometryExtractor.get_streams(
                str(csv_path)
            )
            assert "Signal" in names
            assert "Control" in names

    def test_doric_folder_resolution(self):
        """Test Doric file resolution from folder."""
        from fiber_mosaic.extractors import DoricFiberPhotometryExtractor

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "doric_data.csv"
            # Single stream so auto-selection works
            csv_path.write_text("Console,Signal\nTime(s),Signal\n0.0,100.0\n")

            rec = DoricFiberPhotometryExtractor(tmpdir, color="green")
            assert rec.get_num_samples() == 1

    def test_doric_folder_no_files(self):
        """Test error when no Doric files in folder."""
        from fiber_mosaic.extractors import DoricFiberPhotometryExtractor

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError, match="No Doric files"):
                DoricFiberPhotometryExtractor(tmpdir)

    def test_read_doric_convenience_function(self):
        """Test the convenience function."""
        from fiber_mosaic.extractors.doric_extractor import (
            read_doric_fiber_photometry,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "doric_data.csv"
            # Single stream so auto-selection works
            csv_path.write_text("Console,Signal\nTime(s),Signal\n0.0,100.0\n")

            rec = read_doric_fiber_photometry(csv_path, color="green")
            assert rec.get_num_samples() == 1

    @patch("fiber_mosaic.extractors.doric_extractor.HAVE_H5PY", False)
    def test_doric_h5py_not_available(self):
        """Test error when h5py not available for .doric files."""
        from fiber_mosaic.extractors.doric_extractor import _check_h5py

        with pytest.raises(ImportError, match="h5py is required"):
            _check_h5py()


# =============================================================================
# NPM Extractor Tests
# =============================================================================


class TestNpmExtractor:
    """Tests for NPM extractor."""

    def test_npm_extractor_basic(self):
        """Test basic NPM file loading."""
        from fiber_mosaic.extractors import NpmFiberPhotometryExtractor

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "npm_data.csv"
            csv_path.write_text(
                "Timestamp,LedState,Region0G,Region1G\n"
                "0,1,100.0,200.0\n"
                "10,2,101.0,201.0\n"
                "20,1,102.0,202.0\n"
                "30,2,103.0,203.0\n"
            )

            rec = NpmFiberPhotometryExtractor(
                csv_path, stream_name="Region0G_led1", color="iso"
            )
            assert rec.get_num_samples() == 2
            assert rec.color == "iso"

    def test_npm_extractor_auto_stream(self):
        """Test auto stream selection fails with multiple streams."""
        from fiber_mosaic.extractors import NpmFiberPhotometryExtractor

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "npm_data.csv"
            # Multiple streams: Region0G_led1, Region0G_led2, Region1G_led1...
            csv_path.write_text(
                "Timestamp,LedState,Region0G,Region1G\n"
                "0,1,100.0,200.0\n"
                "10,2,101.0,201.0\n"
            )

            # Multiple streams exist (2 regions x 2 led states = 4)
            with pytest.raises(ValueError, match="Multiple streams"):
                NpmFiberPhotometryExtractor(csv_path)

    def test_npm_extractor_stream_not_found(self):
        """Test error when stream not found."""
        from fiber_mosaic.extractors import NpmFiberPhotometryExtractor

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "npm_data.csv"
            csv_path.write_text(
                "Timestamp,LedState,Region0G,Region1G\n0,1,100.0,200.0\n"
            )

            with pytest.raises(ValueError, match="not found"):
                NpmFiberPhotometryExtractor(
                    csv_path, stream_name="nonexistent"
                )

    def test_npm_extractor_file_not_found(self):
        """Test error when file not found."""
        from fiber_mosaic.extractors import NpmFiberPhotometryExtractor

        with pytest.raises(FileNotFoundError):
            NpmFiberPhotometryExtractor("/nonexistent/file.csv")

    def test_npm_get_streams(self):
        """Test stream discovery."""
        from fiber_mosaic.extractors import NpmFiberPhotometryExtractor

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "npm_data.csv"
            csv_path.write_text(
                "Timestamp,LedState,Region0G,Region1G\n"
                "0,1,100.0,200.0\n"
                "10,2,101.0,201.0\n"
            )

            names, ids = NpmFiberPhotometryExtractor.get_streams(str(csv_path))
            assert "Region0G_led1" in names
            assert "Region0G_led2" in names
            assert "Region1G_led1" in names

    def test_npm_timestamp_ms_conversion(self):
        """Test NPM timestamps in milliseconds are converted to seconds.

        Uses a short recording (max < 1_000_000 ms) to verify the heuristic
        works on interval rather than magnitude.
        """
        from fiber_mosaic.extractors import NpmFiberPhotometryExtractor

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "npm_data.csv"
            # 40 Hz in milliseconds: inter-sample interval = 25 ms
            csv_path.write_text(
                "Timestamp,LedState,Region0G\n"
                "0,1,100.0\n"
                "25,1,101.0\n"
                "50,1,102.0\n"
                "75,1,103.0\n"
            )

            rec = NpmFiberPhotometryExtractor(
                csv_path, stream_name="Region0G_led1", color="green"
            )
            times = rec.get_fiber_times()
            # Converted to seconds: 0, 0.025, 0.05, 0.075
            np.testing.assert_allclose(times[:, 0], [0.0, 0.025, 0.05, 0.075])

    def test_npm_timestamp_seconds_not_converted(self):
        """Test NPM timestamps already in seconds are not converted."""
        from fiber_mosaic.extractors import NpmFiberPhotometryExtractor

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "npm_data.csv"
            # 40 Hz in seconds: inter-sample interval = 0.025 s
            csv_path.write_text(
                "Timestamp,LedState,Region0G\n"
                "0.000,1,100.0\n"
                "0.025,1,101.0\n"
                "0.050,1,102.0\n"
                "0.075,1,103.0\n"
            )

            rec = NpmFiberPhotometryExtractor(
                csv_path, stream_name="Region0G_led1", color="green"
            )
            times = rec.get_fiber_times()
            np.testing.assert_allclose(times[:, 0], [0.0, 0.025, 0.05, 0.075])

    def test_npm_timestamp_unit_ms_explicit(self):
        """timestamp_unit='ms' forces conversion regardless of interval."""
        from fiber_mosaic.extractors import NpmFiberPhotometryExtractor

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "npm_data.csv"
            # Inter-sample interval is 0.025 — heuristic would leave as-is,
            # but explicit "ms" must still convert.
            csv_path.write_text(
                "Timestamp,LedState,Region0G\n"
                "0.000,1,100.0\n"
                "0.025,1,101.0\n"
                "0.050,1,102.0\n"
                "0.075,1,103.0\n"
            )

            rec = NpmFiberPhotometryExtractor(
                csv_path,
                stream_name="Region0G_led1",
                color="green",
                timestamp_unit="ms",
            )
            times = rec.get_fiber_times()
            np.testing.assert_allclose(
                times[:, 0], [0.0, 0.000025, 0.00005, 0.000075]
            )

    def test_npm_timestamp_unit_s_explicit(self):
        """timestamp_unit='s' suppresses conversion regardless of interval."""
        from fiber_mosaic.extractors import NpmFiberPhotometryExtractor

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "npm_data.csv"
            # Inter-sample interval is 25 — heuristic would convert,
            # but explicit "s" must leave the values unchanged.
            csv_path.write_text(
                "Timestamp,LedState,Region0G\n"
                "0,1,100.0\n"
                "25,1,101.0\n"
                "50,1,102.0\n"
                "75,1,103.0\n"
            )

            rec = NpmFiberPhotometryExtractor(
                csv_path,
                stream_name="Region0G_led1",
                color="green",
                timestamp_unit="s",
            )
            times = rec.get_fiber_times()
            np.testing.assert_allclose(times[:, 0], [0.0, 25.0, 50.0, 75.0])

    def test_npm_timestamp_unit_invalid(self):
        """timestamp_unit rejects values other than 's', 'ms', or None."""
        from fiber_mosaic.extractors import NpmFiberPhotometryExtractor

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "npm_data.csv"
            csv_path.write_text(
                "Timestamp,LedState,Region0G\n0,1,100.0\n25,1,101.0\n"
            )

            with pytest.raises(ValueError, match="timestamp_unit"):
                NpmFiberPhotometryExtractor(
                    csv_path,
                    stream_name="Region0G_led1",
                    timestamp_unit="milliseconds",
                )

    def test_npm_no_timestamp_column(self):
        """Test NPM file without timestamp column."""
        from fiber_mosaic.extractors import NpmFiberPhotometryExtractor

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "npm_data.csv"
            csv_path.write_text("LedState,Region0G\n1,100.0\n1,101.0\n")

            rec = NpmFiberPhotometryExtractor(
                csv_path, stream_name="Region0G_led1", color="green"
            )
            assert rec.get_num_samples() == 2

    def test_npm_color_required(self):
        """Test that omitting color raises ValueError."""
        from fiber_mosaic.extractors import NpmFiberPhotometryExtractor

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "npm_data.csv"
            csv_path.write_text("Timestamp,LedState,Region0G\n0,1,100.0\n")

            with pytest.raises(ValueError, match="color must be provided"):
                NpmFiberPhotometryExtractor(
                    csv_path, stream_name="Region0G_led1"
                )

    def test_read_npm_convenience_function(self):
        """Test the convenience function."""
        from fiber_mosaic.extractors.npm_extractor import (
            read_npm_fiber_photometry,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "npm_data.csv"
            csv_path.write_text("Timestamp,LedState,Region0G\n0,1,100.0\n")

            rec = read_npm_fiber_photometry(
                csv_path, stream_name="Region0G_led1", color="green"
            )
            assert rec.get_num_samples() == 1

    def test_npm_no_led_column(self):
        """Test NPM file without LED state column."""
        from fiber_mosaic.extractors import NpmFiberPhotometryExtractor

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "npm_data.csv"
            csv_path.write_text("Timestamp,Region0G\n0,100.0\n10,101.0\n")

            rec = NpmFiberPhotometryExtractor(
                csv_path, stream_name="Region0G_led0", color="green"
            )
            assert rec.get_num_samples() == 2


# =============================================================================
# NWB Extractor Tests
# =============================================================================


class TestNwbExtractor:
    """Tests for NWB extractor (mocked)."""

    @patch("fiber_mosaic.extractors.nwb_extractor.HAVE_PYNWB", False)
    def test_nwb_pynwb_not_available(self):
        """Test error when pynwb not available."""
        from fiber_mosaic.extractors.nwb_extractor import _check_pynwb

        with pytest.raises(ImportError, match="pynwb is required"):
            _check_pynwb()


# =============================================================================
# TDT Extractor Tests
# =============================================================================


class TestTdtExtractor:
    """Tests for TDT extractor (mocked)."""

    def test_tdt_read_tsq_not_found(self):
        """Test when no TSQ file found."""
        from fiber_mosaic.extractors.tdt_extractor import _read_tsq

        with tempfile.TemporaryDirectory() as tmpdir:
            df, found = _read_tsq(tmpdir)
            assert df is None
            assert found is False

    def test_tdt_read_tsq_multiple_files(self):
        """Test error when multiple TSQ files found."""
        from fiber_mosaic.extractors.tdt_extractor import _read_tsq

        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "file1.tsq").touch()
            (Path(tmpdir) / "file2.tsq").touch()

            with pytest.raises(ValueError, match="Multiple TSQ files"):
                _read_tsq(tmpdir)

    def test_tdt_get_streams_empty_folder(self):
        """Test stream discovery from empty folder."""
        from fiber_mosaic.extractors import TdtFiberPhotometryExtractor

        with tempfile.TemporaryDirectory() as tmpdir:
            names, ids = TdtFiberPhotometryExtractor.get_streams(tmpdir)
            assert names == []
            assert ids == []


# =============================================================================
# DANDI Extractor Tests
# =============================================================================


class TestDandiExtractor:
    """Tests for DANDI extractor."""

    def test_is_dandi_uri(self):
        """Test DANDI URI detection."""
        from fiber_mosaic.extractors.dandi_extractor import is_dandi_uri

        assert is_dandi_uri("dandi://000971/sub-112/file.nwb")
        assert not is_dandi_uri("/local/path/file.nwb")
        assert not is_dandi_uri("file.nwb")
        assert not is_dandi_uri(None)

    def test_parse_dandi_uri_valid(self):
        """Test parsing valid DANDI URI."""
        from fiber_mosaic.extractors.dandi_extractor import parse_dandi_uri

        dandiset_id, asset_path = parse_dandi_uri(
            "dandi://000971/sub-112/file.nwb"
        )
        assert dandiset_id == "000971"
        assert asset_path == "sub-112/file.nwb"

    def test_parse_dandi_uri_invalid(self):
        """Test parsing invalid DANDI URI."""
        from fiber_mosaic.extractors.dandi_extractor import parse_dandi_uri

        with pytest.raises(ValueError, match="Not a valid DANDI URI"):
            parse_dandi_uri("/local/path")

    def test_parse_dandi_uri_no_asset_path(self):
        """Test parsing DANDI URI without asset path."""
        from fiber_mosaic.extractors.dandi_extractor import parse_dandi_uri

        with pytest.raises(ValueError, match="Invalid DANDI URI format"):
            parse_dandi_uri("dandi://000971")

    def test_dandi_check_dependencies_exists(self):
        """Test that _check_dandi_dependencies function exists."""
        from fiber_mosaic.extractors.dandi_extractor import (
            _check_dandi_dependencies,
        )

        # Just verify the function is importable and callable
        assert callable(_check_dandi_dependencies)

    def test_dandi_extractor_invalid_uri(self):
        """Test error with invalid URI."""
        from fiber_mosaic.extractors import DandiFiberPhotometryExtractor

        # Mock the dependency check to pass
        with patch(
            "fiber_mosaic.extractors.dandi_extractor._check_dandi_dependencies"
        ):
            with pytest.raises(ValueError, match="Not a valid DANDI URI"):
                DandiFiberPhotometryExtractor("/local/path")

    def test_dandi_get_streams_invalid_uri(self):
        """Test get_streams with invalid URI."""
        from fiber_mosaic.extractors import DandiFiberPhotometryExtractor

        with patch(
            "fiber_mosaic.extractors.dandi_extractor._check_dandi_dependencies"
        ):
            with pytest.raises(ValueError, match="Not a valid DANDI URI"):
                DandiFiberPhotometryExtractor.get_streams("/local/path")


# =============================================================================
# Additional Coverage Tests
# =============================================================================


class TestAdditionalCoverage:
    """Additional tests to increase coverage."""

    def test_csv_time_column_variations(self):
        """Any time column name works when passed explicitly."""
        from fiber_mosaic.extractors import CsvFiberPhotometryExtractor

        for time_col in ["time", "Time", "Timestamps", "Time(s)"]:
            with tempfile.TemporaryDirectory() as tmpdir:
                csv_path = Path(tmpdir) / "data.csv"
                csv_path.write_text(f"{time_col},data\n0.0,1.0\n0.1,2.0\n")

                rec = CsvFiberPhotometryExtractor(
                    csv_path, color="green", time_column=time_col
                )
                assert rec.get_num_samples() == 2

    def test_csv_extractor_with_fiber_columns(self):
        """Test CSV extractor with specific fiber columns."""
        from fiber_mosaic.extractors import CsvFiberPhotometryExtractor

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "data.csv"
            csv_path.write_text(
                "time,fiber1,fiber2,fiber3\n0.0,1.0,2.0,3.0\n0.1,1.1,2.1,3.1\n"
            )

            rec = CsvFiberPhotometryExtractor(
                csv_path,
                color="green",
                time_column="time",
                fiber_columns=["fiber1", "fiber2"],
            )
            assert rec.get_num_fibers() == 2

    def test_classify_csv_numeric_headers(self):
        """Test CSV with numeric column headers."""
        from fiber_mosaic.extractors.detect_formats import classify_csv_file

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "data.csv"
            # File with numeric headers (headerless)
            csv_path.write_text("0.0,1.0,2.0,3.0\n0.1,1.1,2.1,3.1\n")

            result = classify_csv_file(str(csv_path))
            assert result == "npm"

    def test_classify_csv_with_region(self):
        """Test CSV classification with Region column."""
        from fiber_mosaic.extractors.detect_formats import classify_csv_file

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "data.csv"
            # Need >3 columns for NPM detection via Region columns
            csv_path.write_text(
                "Timestamp,LedState,Region0G,Region1G\n0,1,100.0,200.0\n"
            )

            result = classify_csv_file(str(csv_path))
            assert result == "npm"

    def test_detect_formats_exception_handling(self):
        """Test that CSV classification errors are handled."""
        from fiber_mosaic.extractors import detect_formats

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create an invalid CSV that will cause classification error
            csv_path = Path(tmpdir) / "invalid.csv"
            csv_path.write_text("")  # Empty file

            formats = detect_formats(tmpdir)
            # Should not crash, just skip the file
            assert isinstance(formats, set)

    def test_is_event_csv_exception(self):
        """Test _is_event_csv with invalid file."""
        from fiber_mosaic.extractors.detect_formats import _is_event_csv

        # Non-existent file
        result = _is_event_csv("/nonexistent/file.csv")
        assert result is False

    def test_npm_framecounter_timestamp(self):
        """Test NPM with FrameCounter as timestamp."""
        from fiber_mosaic.extractors import NpmFiberPhotometryExtractor

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "npm_data.csv"
            csv_path.write_text(
                "FrameCounter,LedState,Region0G\n0,1,100.0\n1,1,101.0\n"
            )

            rec = NpmFiberPhotometryExtractor(
                csv_path, stream_name="Region0G_led1", color="green"
            )
            assert rec.get_num_samples() == 2

    def test_npm_flags_column(self):
        """Test NPM with Flags column instead of LedState."""
        from fiber_mosaic.extractors import NpmFiberPhotometryExtractor

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "npm_data.csv"
            csv_path.write_text(
                "Timestamp,Flags,Region0G\n0,1,100.0\n10,1,101.0\n"
            )

            rec = NpmFiberPhotometryExtractor(
                csv_path, stream_name="Region0G_led1", color="green"
            )
            assert rec.get_num_samples() == 2

    def test_npm_single_sample(self):
        """Test NPM with single sample."""
        from fiber_mosaic.extractors import NpmFiberPhotometryExtractor

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "npm_data.csv"
            csv_path.write_text("Timestamp,LedState,Region0G\n0,1,100.0\n")

            rec = NpmFiberPhotometryExtractor(
                csv_path, stream_name="Region0G_led1", color="green"
            )
            assert rec.get_num_samples() == 1

    def test_doric_csv_single_sample(self):
        """Test Doric CSV with single sample."""
        from fiber_mosaic.extractors import DoricFiberPhotometryExtractor

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "doric_data.csv"
            # Single stream (Signal) for auto-selection
            csv_path.write_text("Console,Signal\nTime(s),Signal\n0.0,100.0\n")

            rec = DoricFiberPhotometryExtractor(csv_path, color="green")
            assert rec.get_num_samples() == 1

    def test_doric_find_files_in_folder(self):
        """Test Doric file finding helpers."""
        from fiber_mosaic.extractors.doric_extractor import _find_doric_files

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create Doric CSV
            csv_path = Path(tmpdir) / "doric_data.csv"
            csv_path.write_text(
                "Console,I/O,AIn-1\nTime(s),Voltage,Signal\n0.0,1.0,100.0\n"
            )

            doric_files, csv_files = _find_doric_files(Path(tmpdir))
            assert len(doric_files) == 0
            assert len(csv_files) == 1

    def test_doric_find_files_non_doric_csv(self):
        """Test that non-Doric CSVs are filtered out."""
        from fiber_mosaic.extractors.doric_extractor import _find_doric_files

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create non-Doric CSV
            csv_path = Path(tmpdir) / "other_data.csv"
            csv_path.write_text("time,data\n0,1\n")

            doric_files, csv_files = _find_doric_files(Path(tmpdir))
            assert len(doric_files) == 0
            assert len(csv_files) == 0

    def test_external_event_csv_without_npm(self):
        """Test event CSV detection without NPM files."""
        from fiber_mosaic.extractors import detect_formats

        with tempfile.TemporaryDirectory() as tmpdir:
            event_path = Path(tmpdir) / "external_events.csv"
            event_path.write_text("timestamps\n1.0\n2.0\n")

            formats = detect_formats(tmpdir)
            assert "csv" in formats


class TestDoricHDF5Mock:
    """Tests for Doric HDF5 functionality with mocks."""

    def test_doric_v1_key_reading(self):
        """Test reading V1 Doric keys."""
        from fiber_mosaic.extractors.doric_extractor import _read_doric_v1_keys

        # Create a mock h5file with proper structure
        # _read_doric_v1_keys checks: "Traces" in h5file, "Console" in
        # h5file["Traces"]
        mock_console = MagicMock()
        mock_console.keys.return_value = ["Time(s)", "AIn-1", "AIn-2"]

        mock_traces = MagicMock()
        mock_traces.__contains__ = lambda self, x: x == "Console"
        mock_traces.__getitem__ = lambda self, x: mock_console

        mock_h5file = MagicMock()
        mock_h5file.__contains__ = lambda self, x: x == "Traces"
        mock_h5file.__getitem__ = lambda self, x: mock_traces

        keys = _read_doric_v1_keys(mock_h5file)
        assert "AIn-1" in keys
        assert "AIn-2" in keys
        assert "Time(s)" not in keys

    def test_doric_v1_keys_no_traces(self):
        """Test V1 keys when Traces not present."""
        from fiber_mosaic.extractors.doric_extractor import _read_doric_v1_keys

        mock_h5file = MagicMock()
        mock_h5file.__contains__ = lambda self, x: False

        keys = _read_doric_v1_keys(mock_h5file)
        assert keys == []

    def test_doric_v1_keys_no_console(self):
        """Test V1 keys when Console not present in Traces."""
        from fiber_mosaic.extractors.doric_extractor import _read_doric_v1_keys

        mock_traces = MagicMock()
        mock_traces.__contains__ = lambda self, x: False

        mock_h5file = MagicMock()
        mock_h5file.__contains__ = lambda self, x: x == "Traces"
        mock_h5file.__getitem__ = lambda self, x: mock_traces

        keys = _read_doric_v1_keys(mock_h5file)
        assert keys == []

    def test_doric_v6_keys_no_data_acquisition(self):
        """Test V6 keys when DataAcquisition not present."""
        from fiber_mosaic.extractors.doric_extractor import _read_doric_v6_keys

        mock_h5file = MagicMock()
        mock_h5file.__contains__ = lambda self, x: False

        keys = _read_doric_v6_keys(mock_h5file)
        assert keys == []


class TestNWBMock:
    """Tests for NWB functionality with mocks."""

    def test_nwb_no_timing_info(self):
        """Test error when series has neither timestamps nor rate."""
        from fiber_mosaic.extractors import NwbFiberPhotometryExtractor

        with tempfile.TemporaryDirectory() as tmpdir:
            nwb_path = Path(tmpdir) / "session.nwb"
            nwb_path.touch()

            mock_series = MagicMock()
            mock_series.name = "FPSeries"
            mock_series.neurodata_type = "FiberPhotometryResponseSeries"
            mock_series.data.__getitem__ = lambda self, x: np.zeros((10, 1))
            mock_series.data.shape = (10, 1)
            mock_series.get_timestamps.return_value = None
            mock_series.rate = None

            mock_nwbfile = MagicMock()
            mock_nwbfile.objects.values.return_value = [mock_series]

            mock_io = MagicMock()
            mock_io.read.return_value = mock_nwbfile
            mock_io.__enter__ = MagicMock(return_value=mock_io)
            mock_io.__exit__ = MagicMock(return_value=False)

            with patch("fiber_mosaic.extractors.nwb_extractor._check_pynwb"):
                with patch(
                    "fiber_mosaic.extractors.nwb_extractor.NWBHDF5IO",
                    return_value=mock_io,
                ):
                    with pytest.raises(
                        ValueError, match="neither timestamps nor rate"
                    ):
                        NwbFiberPhotometryExtractor(
                            str(nwb_path),
                            series_name="FPSeries",
                            color="green",
                        )


class TestDANDIMock:
    """Tests for DANDI functionality with mocks."""

    def test_dandi_resolve_timing_with_timestamps(self):
        """Test DANDI timing resolution with timestamps."""
        from fiber_mosaic.extractors.dandi_extractor import _resolve_timing

        class MockSeries:
            def __init__(self):
                self.timestamps = [0.0, 0.1, 0.2]
                self.rate = None

        mock_series = MockSeries()
        sampling_rate, timestamps = _resolve_timing(mock_series, 3)
        assert sampling_rate == pytest.approx(10.0, rel=0.1)

    def test_dandi_resolve_timing_with_rate(self):
        """Test DANDI timing resolution with rate."""
        from fiber_mosaic.extractors.dandi_extractor import _resolve_timing

        class MockSeries:
            def __init__(self):
                self.timestamps = None
                self.rate = 100.0
                self.starting_time = 0.0

        mock_series = MockSeries()
        sampling_rate, timestamps = _resolve_timing(mock_series, 100)
        assert sampling_rate == 100.0

    def test_dandi_resolve_timing_no_info(self):
        """Test DANDI timing resolution without timing info."""
        from fiber_mosaic.extractors.dandi_extractor import _resolve_timing

        class MockSeries:
            def __init__(self):
                self.timestamps = None
                self.rate = None
                self.name = "test"

        mock_series = MockSeries()
        with pytest.raises(ValueError, match="neither timestamps nor rate"):
            _resolve_timing(mock_series, 100)

    def test_dandi_discover_series(self):
        """Test DANDI series discovery."""
        from fiber_mosaic.extractors.dandi_extractor import (
            _discover_fiber_photometry_series,
        )

        mock_nwbfile = MagicMock()
        mock_series = MagicMock()
        mock_series.neurodata_type = "FiberPhotometryResponseSeries"
        mock_series.name = "GCaMP"

        mock_nwbfile.objects.values.return_value = [mock_series]

        result = _discover_fiber_photometry_series(mock_nwbfile)
        assert "GCaMP" in result


# =============================================================================
# DANDI Extractor Full Coverage Tests
# =============================================================================


class TestDandiExtractorFullCoverage:
    """Full coverage tests for DANDI extractor."""

    def test_dandi_extractor_full_init(self):
        """Test full DandiFiberPhotometryExtractor initialization with
        mocks."""
        from fiber_mosaic.extractors import DandiFiberPhotometryExtractor

        mock_data = np.random.randn(100, 2)
        mock_timestamps = np.linspace(0, 10, 100)

        mock_series = MagicMock()
        mock_series.name = "GCaMP"
        mock_series.neurodata_type = "FiberPhotometryResponseSeries"
        mock_series.data = MagicMock()
        mock_series.data.__getitem__ = lambda self, x: mock_data
        mock_series.data.shape = mock_data.shape
        mock_series.timestamps = MagicMock()
        mock_series.timestamps.__getitem__ = lambda self, x: mock_timestamps
        mock_series.timestamps.__len__ = lambda self: len(mock_timestamps)
        mock_series.timestamps.__bool__ = lambda self: True

        mock_nwbfile = MagicMock()
        mock_nwbfile.objects.values.return_value = [mock_series]

        mock_io = MagicMock()

        with patch(
            "fiber_mosaic.extractors.dandi_extractor._check_dandi_dependencies"
        ):
            with patch(
                "fiber_mosaic.extractors.dandi_extractor._stream_nwb",
                return_value=(mock_nwbfile, mock_io),
            ):
                rec = DandiFiberPhotometryExtractor(
                    "dandi://000123/sub-1/file.nwb",
                    series_name="GCaMP",
                    color="green",
                )
                assert rec.get_num_fibers() == 2
                assert rec.color == "green"

    def test_dandi_extractor_auto_series_selection(self):
        """Test auto series selection when only one series exists."""
        from fiber_mosaic.extractors import DandiFiberPhotometryExtractor

        mock_data = np.random.randn(100)
        mock_timestamps = np.linspace(0, 10, 100)

        mock_series = MagicMock()
        mock_series.name = "OnlySeries"
        mock_series.neurodata_type = "FiberPhotometryResponseSeries"
        mock_series.data = MagicMock()
        mock_series.data.__getitem__ = lambda self, x: mock_data
        mock_series.data.shape = mock_data.shape
        mock_series.timestamps = MagicMock()
        mock_series.timestamps.__getitem__ = lambda self, x: mock_timestamps
        mock_series.timestamps.__len__ = lambda self: len(mock_timestamps)
        mock_series.timestamps.__bool__ = lambda self: True

        mock_nwbfile = MagicMock()
        mock_nwbfile.objects.values.return_value = [mock_series]

        mock_io = MagicMock()

        with patch(
            "fiber_mosaic.extractors.dandi_extractor._check_dandi_dependencies"
        ):
            with patch(
                "fiber_mosaic.extractors.dandi_extractor._stream_nwb",
                return_value=(mock_nwbfile, mock_io),
            ):
                # No series_name specified - should auto-select
                rec = DandiFiberPhotometryExtractor(
                    "dandi://000123/sub-1/file.nwb", color="green"
                )
                assert rec.get_num_fibers() == 1

    def test_dandi_extractor_no_series_found(self):
        """Test error when no FiberPhotometryResponseSeries found."""
        from fiber_mosaic.extractors import DandiFiberPhotometryExtractor

        mock_nwbfile = MagicMock()
        mock_nwbfile.objects.values.return_value = []

        mock_io = MagicMock()

        with patch(
            "fiber_mosaic.extractors.dandi_extractor._check_dandi_dependencies"
        ):
            with patch(
                "fiber_mosaic.extractors.dandi_extractor._stream_nwb",
                return_value=(mock_nwbfile, mock_io),
            ):
                with pytest.raises(
                    ValueError, match="No FiberPhotometryResponseSeries"
                ):
                    DandiFiberPhotometryExtractor(
                        "dandi://000123/sub-1/file.nwb"
                    )

    def test_dandi_extractor_multiple_series_no_name(self):
        """Test error when multiple series and no name specified."""
        from fiber_mosaic.extractors import DandiFiberPhotometryExtractor

        mock_series1 = MagicMock()
        mock_series1.name = "Series1"
        mock_series1.neurodata_type = "FiberPhotometryResponseSeries"

        mock_series2 = MagicMock()
        mock_series2.name = "Series2"
        mock_series2.neurodata_type = "FiberPhotometryResponseSeries"

        mock_nwbfile = MagicMock()
        mock_nwbfile.objects.values.return_value = [mock_series1, mock_series2]

        mock_io = MagicMock()

        with patch(
            "fiber_mosaic.extractors.dandi_extractor._check_dandi_dependencies"
        ):
            with patch(
                "fiber_mosaic.extractors.dandi_extractor._stream_nwb",
                return_value=(mock_nwbfile, mock_io),
            ):
                with pytest.raises(ValueError, match="Multiple series found"):
                    DandiFiberPhotometryExtractor(
                        "dandi://000123/sub-1/file.nwb"
                    )

    def test_dandi_extractor_series_not_found(self):
        """Test error when specified series not found."""
        from fiber_mosaic.extractors import DandiFiberPhotometryExtractor

        mock_series = MagicMock()
        mock_series.name = "ExistingSeries"
        mock_series.neurodata_type = "FiberPhotometryResponseSeries"

        mock_nwbfile = MagicMock()
        mock_nwbfile.objects.values.return_value = [mock_series]

        mock_io = MagicMock()

        with patch(
            "fiber_mosaic.extractors.dandi_extractor._check_dandi_dependencies"
        ):
            with patch(
                "fiber_mosaic.extractors.dandi_extractor._stream_nwb",
                return_value=(mock_nwbfile, mock_io),
            ):
                with pytest.raises(ValueError, match="not found"):
                    DandiFiberPhotometryExtractor(
                        "dandi://000123/sub-1/file.nwb",
                        series_name="NonexistentSeries",
                    )

    def test_dandi_get_streams_full(self):
        """Test DandiFiberPhotometryExtractor.get_streams with mocks."""
        from fiber_mosaic.extractors import DandiFiberPhotometryExtractor

        mock_series = MagicMock()
        mock_series.name = "TestSeries"
        mock_series.neurodata_type = "FiberPhotometryResponseSeries"

        mock_nwbfile = MagicMock()
        mock_nwbfile.objects.values.return_value = [mock_series]

        mock_io = MagicMock()

        with patch(
            "fiber_mosaic.extractors.dandi_extractor._check_dandi_dependencies"
        ):
            with patch(
                "fiber_mosaic.extractors.dandi_extractor._stream_nwb",
                return_value=(mock_nwbfile, mock_io),
            ):
                names, ids = DandiFiberPhotometryExtractor.get_streams(
                    "dandi://000123/sub-1/file.nwb"
                )
                assert "TestSeries" in names
                assert "TestSeries" in ids

    def test_read_dandi_convenience_function(self):
        """Test read_dandi_fiber_photometry convenience function."""
        from fiber_mosaic.extractors.dandi_extractor import (
            read_dandi_fiber_photometry,
        )

        mock_data = np.random.randn(100)
        mock_timestamps = np.linspace(0, 10, 100)

        mock_series = MagicMock()
        mock_series.name = "GCaMP"
        mock_series.neurodata_type = "FiberPhotometryResponseSeries"
        mock_series.data = MagicMock()
        mock_series.data.__getitem__ = lambda self, x: mock_data
        mock_series.data.shape = mock_data.shape
        mock_series.timestamps = MagicMock()
        mock_series.timestamps.__getitem__ = lambda self, x: mock_timestamps
        mock_series.timestamps.__len__ = lambda self: len(mock_timestamps)
        mock_series.timestamps.__bool__ = lambda self: True

        mock_nwbfile = MagicMock()
        mock_nwbfile.objects.values.return_value = [mock_series]

        mock_io = MagicMock()

        with patch(
            "fiber_mosaic.extractors.dandi_extractor._check_dandi_dependencies"
        ):
            with patch(
                "fiber_mosaic.extractors.dandi_extractor._stream_nwb",
                return_value=(mock_nwbfile, mock_io),
            ):
                rec = read_dandi_fiber_photometry(
                    "dandi://000123/sub-1/file.nwb",
                    series_name="GCaMP",
                    color="green",
                )
                assert rec.get_num_fibers() == 1

    def test_dandi_resolve_timing_single_timestamp_with_rate(self):
        """Test timing resolution with single timestamp and rate."""
        from fiber_mosaic.extractors.dandi_extractor import _resolve_timing

        class MockSeries:
            def __init__(self):
                self.timestamps = [0.0]
                self.rate = 50.0

        mock_series = MockSeries()
        sampling_rate, timestamps = _resolve_timing(mock_series, 1)
        assert sampling_rate == 50.0

    def test_dandi_resolve_timing_single_timestamp_no_rate(self):
        """Test timing resolution with single timestamp and no rate
        (defaults to 1.0)."""
        from fiber_mosaic.extractors.dandi_extractor import _resolve_timing

        class MockSeries:
            def __init__(self):
                self.timestamps = [0.0]
                self.rate = None

        mock_series = MockSeries()
        sampling_rate, timestamps = _resolve_timing(mock_series, 1)
        assert sampling_rate == 1.0

    def test_dandi_resolve_timing_rate_no_starting_time(self):
        """Test timing resolution with rate but no starting_time attribute."""
        from fiber_mosaic.extractors.dandi_extractor import _resolve_timing

        class MockSeries:
            def __init__(self):
                self.timestamps = None
                self.rate = 100.0
                # No starting_time attribute

        mock_series = MockSeries()
        sampling_rate, timestamps = _resolve_timing(mock_series, 100)
        assert sampling_rate == 100.0
        assert timestamps[0] == 0.0


# =============================================================================
# Detect Formats Full Coverage Tests
# =============================================================================


class TestDetectFormatsFullCoverage:
    """Full coverage tests for detect_formats."""

    def test_classify_csv_doric_format(self):
        """Test classify_csv_file correctly identifies Doric format."""
        from fiber_mosaic.extractors.detect_formats import classify_csv_file

        with tempfile.TemporaryDirectory() as tmpdir:
            # Doric CSV (all strings in first rows)
            csv_path = Path(tmpdir) / "doric.csv"
            csv_path.write_text(
                "Console,I/O,Signal\nTime(s),Voltage,Value\n0.0,1.0,100.0\n"
            )
            result = classify_csv_file(str(csv_path))
            assert result == "doric"

    def test_classify_csv_npm_format(self):
        """Test classify_csv_file correctly identifies NPM format."""
        from fiber_mosaic.extractors.detect_formats import classify_csv_file

        with tempfile.TemporaryDirectory() as tmpdir:
            # NPM-like CSV (2+ columns, numeric data)
            csv_path = Path(tmpdir) / "npm.csv"
            csv_path.write_text("time,data\n0.0,1.0\n")
            result = classify_csv_file(str(csv_path))
            assert result == "npm"


# =============================================================================
# Doric Extractor Full Coverage Tests
# =============================================================================


class TestDoricExtractorFullCoverage:
    """Full coverage tests for Doric extractor with HDF5 mocks."""

    def test_doric_v6_key_reading_with_values(self):
        """Test reading V6 Doric keys from a real HDF5 file.

        Exercises both the ``Values`` dataset branch and the fallback
        branch for datasets not named ``Values``.
        """
        import h5py

        from fiber_mosaic.extractors.doric_extractor import (
            _read_doric_v6_keys,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            doric_path = Path(tmpdir) / "test.doric"
            with h5py.File(str(doric_path), "w") as f:
                grp = f.create_group(
                    "DataAcquisition/FPConsole/Signals/Series0001/AnalogIn"
                )
                # Dataset named "Values" -> uses the -3/-2 naming branch
                ain1 = grp.create_group("AIn-1")
                ain1.create_dataset("Values", data=np.arange(5.0))
                ain1.create_dataset("Time", data=np.arange(5.0))
                # Dataset NOT named "Values" -> uses the -2/-1 branch
                grp.create_dataset("AIn-2", data=np.arange(5.0))

            with h5py.File(str(doric_path), "r") as f:
                keys = _read_doric_v6_keys(f)

        assert "AnalogIn/AIn-1" in keys
        assert "AnalogIn/AIn-2" in keys

    def test_doric_discover_file_streams(self):
        """Test _discover_doric_file_streams function."""
        from fiber_mosaic.extractors.doric_extractor import (
            _discover_doric_file_streams,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock .doric file
            doric_path = Path(tmpdir) / "test.doric"

            mock_console = MagicMock()
            mock_console.keys.return_value = ["Time(s)", "AIn-1", "AIn-2"]

            mock_traces = MagicMock()
            mock_traces.__contains__ = lambda self, x: x == "Console"
            mock_traces.__getitem__ = lambda self, x: mock_console

            mock_h5file = MagicMock()
            mock_h5file.keys.return_value = ["Traces"]
            mock_h5file.__contains__ = lambda self, x: x == "Traces"
            mock_h5file.__getitem__ = lambda self, x: mock_traces

            with patch(
                "fiber_mosaic.extractors.doric_extractor.h5py.File",
                return_value=mock_h5file,
            ):
                with patch(
                    "fiber_mosaic.extractors.doric_extractor._check_h5py"
                ):
                    mock_h5file.__enter__ = MagicMock(return_value=mock_h5file)
                    mock_h5file.__exit__ = MagicMock(return_value=False)
                    streams = _discover_doric_file_streams(doric_path)
                    assert "AIn-1" in streams
                    assert "AIn-2" in streams

    def test_doric_discover_file_streams_v6(self):
        """Test _discover_doric_file_streams with V6 format."""
        from fiber_mosaic.extractors.doric_extractor import (
            _discover_doric_file_streams,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            doric_path = Path(tmpdir) / "test.doric"

            mock_h5file = MagicMock()
            mock_h5file.keys.return_value = [
                "Configurations",
                "DataAcquisition",
            ]

            # V6 has Configurations and DataAcquisition
            mock_h5file.__contains__ = lambda self, x: (
                x
                in [
                    "Configurations",
                    "DataAcquisition",
                ]
            )

            with patch(
                "fiber_mosaic.extractors.doric_extractor.h5py.File",
                return_value=mock_h5file,
            ):
                with patch(
                    "fiber_mosaic.extractors.doric_extractor._check_h5py"
                ):
                    with patch(
                        "fiber_mosaic.extractors.doric_extractor."
                        "_read_doric_v6_keys",
                        return_value=["Signal1"],
                    ):
                        mock_h5file.__enter__ = MagicMock(
                            return_value=mock_h5file
                        )
                        mock_h5file.__exit__ = MagicMock(return_value=False)
                        streams = _discover_doric_file_streams(doric_path)
                        assert "Signal1" in streams

    def test_doric_discover_file_streams_unknown_format(self):
        """Test _discover_doric_file_streams with unknown format."""
        from fiber_mosaic.extractors.doric_extractor import (
            _discover_doric_file_streams,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            doric_path = Path(tmpdir) / "test.doric"

            mock_h5file = MagicMock()
            mock_h5file.keys.return_value = ["UnknownKey"]
            mock_h5file.__contains__ = lambda self, x: False

            with patch(
                "fiber_mosaic.extractors.doric_extractor.h5py.File",
                return_value=mock_h5file,
            ):
                with patch(
                    "fiber_mosaic.extractors.doric_extractor._check_h5py"
                ):
                    mock_h5file.__enter__ = MagicMock(return_value=mock_h5file)
                    mock_h5file.__exit__ = MagicMock(return_value=False)
                    streams = _discover_doric_file_streams(doric_path)
                    assert streams == []

    def test_doric_read_v1_data(self):
        """Test _read_doric_v1_data function."""
        from fiber_mosaic.extractors.doric_extractor import _read_doric_v1_data

        mock_timestamps = np.array([0.0, 0.1, 0.2])
        mock_data = np.array([100.0, 101.0, 102.0])

        mock_console = MagicMock()
        mock_console.__getitem__ = lambda self, k: (
            mock_timestamps if k == "Time(s)" else mock_data
        )

        mock_traces = MagicMock()
        mock_traces.__getitem__ = lambda self, k: mock_console

        mock_h5file = MagicMock()
        mock_h5file.__getitem__ = lambda self, k: mock_traces

        data, timestamps, sampling_rate = _read_doric_v1_data(
            mock_h5file, "Signal"
        )
        np.testing.assert_array_equal(data, mock_data)
        np.testing.assert_array_equal(timestamps, mock_timestamps)

    def test_doric_read_v6_data(self):
        """Test _read_doric_v6_data function."""
        from fiber_mosaic.extractors.doric_extractor import _read_doric_v6_data

        mock_data = np.array([100.0, 101.0, 102.0])
        mock_timestamps = np.array([0.0, 0.1, 0.2])

        mock_data_group = MagicMock()
        mock_data_group.__contains__ = lambda self, k: k in ["Values", "Time"]
        mock_data_group.__getitem__ = lambda self, k: (
            mock_data if k == "Values" else mock_timestamps
        )
        mock_data_group.keys.return_value = ["Values", "Time"]

        mock_h5file = MagicMock()

        with patch(
            "fiber_mosaic.extractors.doric_extractor._find_v6_group",
            return_value=mock_data_group,
        ):
            data, timestamps, sr = _read_doric_v6_data(
                mock_h5file, "Signal/AIn-1"
            )
            np.testing.assert_array_equal(data, mock_data)

    def test_doric_read_v6_data_not_found(self):
        """Test _read_doric_v6_data when stream not found."""
        from fiber_mosaic.extractors.doric_extractor import _read_doric_v6_data

        mock_h5file = MagicMock()

        with patch(
            "fiber_mosaic.extractors.doric_extractor._find_v6_group",
            return_value=None,
        ):
            with pytest.raises(ValueError, match="Could not find data"):
                _read_doric_v6_data(mock_h5file, "Nonexistent/Stream")

    def test_doric_extract_v6_data_fallback(self):
        """Test _extract_v6_data fallback to non-Values dataset."""
        import h5py

        from fiber_mosaic.extractors.doric_extractor import _extract_v6_data

        mock_data = np.array([1.0, 2.0, 3.0])
        mock_dataset = MagicMock(spec=h5py.Dataset)
        mock_dataset.__getitem__ = lambda self, x: mock_data

        mock_group = MagicMock()
        mock_group.__contains__ = lambda self, k: k != "Values"
        mock_group.keys.return_value = ["AIn-1"]
        mock_group.__getitem__ = lambda self, k: mock_dataset

        data = _extract_v6_data(mock_group, "test")
        np.testing.assert_array_equal(data, mock_data)

    def test_doric_extract_v6_data_no_data(self):
        """Test _extract_v6_data when no data found."""
        import h5py

        from fiber_mosaic.extractors.doric_extractor import _extract_v6_data

        mock_group = MagicMock()
        mock_group.__contains__ = lambda self, k: False
        mock_group.keys.return_value = ["Time"]
        # Return a non-dataset for "Time" key
        mock_group.__getitem__ = lambda self, k: MagicMock(spec=h5py.Group)

        with pytest.raises(ValueError, match="No data found"):
            _extract_v6_data(mock_group, "test")

    def test_doric_extract_v6_timestamps_from_parent(self):
        """Test _extract_v6_timestamps fallback to parent group."""
        from fiber_mosaic.extractors.doric_extractor import (
            _extract_v6_timestamps,
        )

        mock_timestamps = np.array([0.0, 0.1, 0.2])

        mock_parent = MagicMock()
        mock_parent.__contains__ = lambda self, k: k == "Time"
        mock_parent.__getitem__ = lambda self, k: mock_timestamps

        mock_group = MagicMock()
        mock_group.__contains__ = lambda self, k: False
        mock_group.parent = mock_parent

        timestamps = _extract_v6_timestamps(mock_group, 3)
        np.testing.assert_array_equal(timestamps, mock_timestamps)

    def test_doric_extract_v6_timestamps_fallback_arange(self):
        """Test _extract_v6_timestamps fallback to arange."""
        from fiber_mosaic.extractors.doric_extractor import (
            _extract_v6_timestamps,
        )

        mock_parent = MagicMock()
        mock_parent.__contains__ = lambda self, k: False

        mock_group = MagicMock()
        mock_group.__contains__ = lambda self, k: False
        mock_group.parent = mock_parent

        timestamps = _extract_v6_timestamps(mock_group, 3)
        np.testing.assert_array_equal(timestamps, np.arange(3))

    def test_doric_hdf5_extractor_full(self):
        """Test DoricFiberPhotometryExtractor with .doric file."""
        from fiber_mosaic.extractors import DoricFiberPhotometryExtractor

        with tempfile.TemporaryDirectory() as tmpdir:
            doric_path = Path(tmpdir) / "test.doric"
            doric_path.touch()

            mock_data = np.array([100.0, 101.0, 102.0])
            mock_timestamps = np.array([0.0, 0.1, 0.2])

            with patch("fiber_mosaic.extractors.doric_extractor._check_h5py"):
                with patch(
                    "fiber_mosaic.extractors.doric_extractor."
                    "_discover_doric_file_streams",
                    return_value=["Signal"],
                ):
                    with patch(
                        "fiber_mosaic.extractors.doric_extractor."
                        "_read_doric_file_data",
                        return_value=(mock_data, mock_timestamps, 10.0),
                    ):
                        rec = DoricFiberPhotometryExtractor(
                            str(doric_path),
                            stream_name="Signal",
                            color="green",
                        )
                        assert rec.get_num_samples() == 3

    def test_doric_read_file_data_v1(self):
        """Test _read_doric_file_data with V1 format."""
        from fiber_mosaic.extractors.doric_extractor import (
            _read_doric_file_data,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            doric_path = Path(tmpdir) / "test.doric"

            mock_h5file = MagicMock()
            mock_h5file.keys.return_value = ["Traces"]
            mock_h5file.__contains__ = lambda self, x: x == "Traces"

            mock_data = (np.array([1.0, 2.0]), np.array([0.0, 0.1]), 10.0)

            with patch(
                "fiber_mosaic.extractors.doric_extractor.h5py.File",
                return_value=mock_h5file,
            ):
                with patch(
                    "fiber_mosaic.extractors.doric_extractor."
                    "_read_doric_v1_data",
                    return_value=mock_data,
                ):
                    mock_h5file.__enter__ = MagicMock(return_value=mock_h5file)
                    mock_h5file.__exit__ = MagicMock(return_value=False)
                    data, timestamps, sr = _read_doric_file_data(
                        doric_path, "Signal"
                    )
                    np.testing.assert_array_equal(data, mock_data[0])


# =============================================================================
# NPM Extractor Full Coverage Tests
# =============================================================================


class TestNpmExtractorFullCoverage:
    """Full coverage tests for NPM extractor."""

    def test_npm_timestamp_column_not_found(self):
        """Test error when an explicit timestamp_column does not exist."""
        from fiber_mosaic.extractors.npm_extractor import (
            _resolve_npm_timestamp_column,
        )

        with pytest.raises(ValueError, match="timestamp_column 'Missing'"):
            _resolve_npm_timestamp_column(
                ["Timestamp", "LedState", "Region0G"], "Missing"
            )

    def test_npm_explicit_timestamp_column(self):
        """Test loading with an explicit timestamp_column selection."""
        from fiber_mosaic.extractors import NpmFiberPhotometryExtractor

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "npm_data.csv"
            csv_path.write_text(
                "SystemTimestamp,ComputerTimestamp,LedState,Region0G\n"
                "1000,1001,1,100.0\n"
                "1020,1021,1,101.0\n"
            )

            rec = NpmFiberPhotometryExtractor(
                csv_path,
                stream_name="Region0G_led1",
                color="green",
                timestamp_column="SystemTimestamp",
            )
            assert rec.get_num_samples() == 2

    def test_npm_multiple_timestamp_columns_no_selection(self):
        """Test error when multiple timestamp columns and none selected."""
        from fiber_mosaic.extractors import NpmFiberPhotometryExtractor

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "npm_data.csv"
            csv_path.write_text(
                "SystemTimestamp,ComputerTimestamp,LedState,Region0G\n"
                "1000,1001,1,100.0\n"
                "1020,1021,1,101.0\n"
            )

            with pytest.raises(ValueError, match="Multiple timestamp columns"):
                NpmFiberPhotometryExtractor(
                    csv_path, stream_name="Region0G_led1", color="green"
                )


# =============================================================================
# NWB Extractor Full Coverage Tests
# =============================================================================


class TestNwbExtractorFullCoverage:
    """Full coverage tests for NWB extractor."""

    def test_nwb_extractor_full_init(self):
        """Test full NwbFiberPhotometryExtractor initialization."""
        from fiber_mosaic.extractors import NwbFiberPhotometryExtractor

        with tempfile.TemporaryDirectory() as tmpdir:
            nwb_path = Path(tmpdir) / "session.nwb"
            nwb_path.touch()

            mock_data = np.random.randn(100, 2)
            mock_timestamps = np.linspace(0, 10, 100)

            mock_series = MagicMock()
            mock_series.name = "FPSeries"
            mock_series.neurodata_type = "FiberPhotometryResponseSeries"
            mock_series.data = MagicMock()
            mock_series.data.__getitem__ = lambda self, x: mock_data
            mock_series.data.shape = mock_data.shape
            mock_series.get_timestamps.return_value = mock_timestamps
            mock_series.rate = None

            mock_nwbfile = MagicMock()
            mock_nwbfile.objects.values.return_value = [mock_series]

            mock_io = MagicMock()
            mock_io.read.return_value = mock_nwbfile
            mock_io.__enter__ = MagicMock(return_value=mock_io)
            mock_io.__exit__ = MagicMock(return_value=False)

            with patch("fiber_mosaic.extractors.nwb_extractor._check_pynwb"):
                with patch(
                    "fiber_mosaic.extractors.nwb_extractor.NWBHDF5IO",
                    return_value=mock_io,
                ):
                    rec = NwbFiberPhotometryExtractor(
                        str(nwb_path), series_name="FPSeries", color="green"
                    )
                    assert rec.get_num_fibers() == 2
                    assert rec.color == "green"

    def test_nwb_extractor_no_series(self):
        """Test error when no FiberPhotometryResponseSeries found."""
        from fiber_mosaic.extractors import NwbFiberPhotometryExtractor

        with tempfile.TemporaryDirectory() as tmpdir:
            nwb_path = Path(tmpdir) / "session.nwb"
            nwb_path.touch()

            mock_nwbfile = MagicMock()
            mock_nwbfile.objects.values.return_value = []

            mock_io = MagicMock()
            mock_io.read.return_value = mock_nwbfile
            mock_io.__enter__ = MagicMock(return_value=mock_io)
            mock_io.__exit__ = MagicMock(return_value=False)

            with patch("fiber_mosaic.extractors.nwb_extractor._check_pynwb"):
                with patch(
                    "fiber_mosaic.extractors.nwb_extractor.NWBHDF5IO",
                    return_value=mock_io,
                ):
                    with pytest.raises(
                        ValueError,
                        match="No FiberPhotometryResponseSeries",
                    ):
                        NwbFiberPhotometryExtractor(str(nwb_path))

    def test_nwb_extractor_multiple_series_no_name(self):
        """Test error when multiple series and no name specified."""
        from fiber_mosaic.extractors import NwbFiberPhotometryExtractor

        with tempfile.TemporaryDirectory() as tmpdir:
            nwb_path = Path(tmpdir) / "session.nwb"
            nwb_path.touch()

            mock_series1 = MagicMock()
            mock_series1.name = "Series1"
            mock_series1.neurodata_type = "FiberPhotometryResponseSeries"

            mock_series2 = MagicMock()
            mock_series2.name = "Series2"
            mock_series2.neurodata_type = "FiberPhotometryResponseSeries"

            mock_nwbfile = MagicMock()
            mock_nwbfile.objects.values.return_value = [
                mock_series1,
                mock_series2,
            ]

            mock_io = MagicMock()
            mock_io.read.return_value = mock_nwbfile
            mock_io.__enter__ = MagicMock(return_value=mock_io)
            mock_io.__exit__ = MagicMock(return_value=False)

            with patch("fiber_mosaic.extractors.nwb_extractor._check_pynwb"):
                with patch(
                    "fiber_mosaic.extractors.nwb_extractor.NWBHDF5IO",
                    return_value=mock_io,
                ):
                    with pytest.raises(
                        ValueError, match="Multiple series found"
                    ):
                        NwbFiberPhotometryExtractor(str(nwb_path))

    def test_nwb_extractor_series_not_found(self):
        """Test error when specified series not found."""
        from fiber_mosaic.extractors import NwbFiberPhotometryExtractor

        with tempfile.TemporaryDirectory() as tmpdir:
            nwb_path = Path(tmpdir) / "session.nwb"
            nwb_path.touch()

            mock_series = MagicMock()
            mock_series.name = "ExistingSeries"
            mock_series.neurodata_type = "FiberPhotometryResponseSeries"

            mock_nwbfile = MagicMock()
            mock_nwbfile.objects.values.return_value = [mock_series]

            mock_io = MagicMock()
            mock_io.read.return_value = mock_nwbfile
            mock_io.__enter__ = MagicMock(return_value=mock_io)
            mock_io.__exit__ = MagicMock(return_value=False)

            with patch("fiber_mosaic.extractors.nwb_extractor._check_pynwb"):
                with patch(
                    "fiber_mosaic.extractors.nwb_extractor.NWBHDF5IO",
                    return_value=mock_io,
                ):
                    with pytest.raises(ValueError, match="not found"):
                        NwbFiberPhotometryExtractor(
                            str(nwb_path), series_name="NonexistentSeries"
                        )

    def test_nwb_get_streams_full(self):
        """Test NwbFiberPhotometryExtractor.get_streams."""
        from fiber_mosaic.extractors import NwbFiberPhotometryExtractor

        with tempfile.TemporaryDirectory() as tmpdir:
            nwb_path = Path(tmpdir) / "session.nwb"
            nwb_path.touch()

            mock_series = MagicMock()
            mock_series.name = "TestSeries"
            mock_series.neurodata_type = "FiberPhotometryResponseSeries"

            mock_nwbfile = MagicMock()
            mock_nwbfile.objects.values.return_value = [mock_series]

            mock_io = MagicMock()
            mock_io.read.return_value = mock_nwbfile
            mock_io.__enter__ = MagicMock(return_value=mock_io)
            mock_io.__exit__ = MagicMock(return_value=False)

            with patch("fiber_mosaic.extractors.nwb_extractor._check_pynwb"):
                with patch(
                    "fiber_mosaic.extractors.nwb_extractor.NWBHDF5IO",
                    return_value=mock_io,
                ):
                    names, ids = NwbFiberPhotometryExtractor.get_streams(
                        str(nwb_path)
                    )
                    assert "TestSeries" in names

    def test_read_nwb_convenience_function(self):
        """Test read_nwb_fiber_photometry convenience function."""
        from fiber_mosaic.extractors.nwb_extractor import (
            read_nwb_fiber_photometry,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            nwb_path = Path(tmpdir) / "session.nwb"
            nwb_path.touch()

            mock_data = np.random.randn(100)
            mock_timestamps = np.linspace(0, 10, 100)

            mock_series = MagicMock()
            mock_series.name = "FPSeries"
            mock_series.neurodata_type = "FiberPhotometryResponseSeries"
            mock_series.data = MagicMock()
            mock_series.data.__getitem__ = lambda self, x: mock_data
            mock_series.data.shape = mock_data.shape
            mock_series.get_timestamps.return_value = mock_timestamps
            mock_series.rate = None

            mock_nwbfile = MagicMock()
            mock_nwbfile.objects.values.return_value = [mock_series]

            mock_io = MagicMock()
            mock_io.read.return_value = mock_nwbfile
            mock_io.__enter__ = MagicMock(return_value=mock_io)
            mock_io.__exit__ = MagicMock(return_value=False)

            with patch("fiber_mosaic.extractors.nwb_extractor._check_pynwb"):
                with patch(
                    "fiber_mosaic.extractors.nwb_extractor.NWBHDF5IO",
                    return_value=mock_io,
                ):
                    rec = read_nwb_fiber_photometry(
                        str(nwb_path), color="green"
                    )
                    assert rec.get_num_fibers() == 1

    def test_nwb_file_not_found(self):
        """Test FileNotFoundError when file does not exist."""
        from fiber_mosaic.extractors import NwbFiberPhotometryExtractor

        with patch("fiber_mosaic.extractors.nwb_extractor._check_pynwb"):
            with pytest.raises(FileNotFoundError, match="File not found"):
                NwbFiberPhotometryExtractor("/nonexistent/path/session.nwb")

    def test_nwb_extractor_no_color(self):
        """Test error when color is not provided."""
        from fiber_mosaic.extractors import NwbFiberPhotometryExtractor

        with tempfile.TemporaryDirectory() as tmpdir:
            nwb_path = Path(tmpdir) / "session.nwb"
            nwb_path.touch()

            mock_data = np.random.randn(100)
            mock_timestamps = np.linspace(0, 10, 100)

            mock_series = MagicMock()
            mock_series.name = "FPSeries"
            mock_series.neurodata_type = "FiberPhotometryResponseSeries"
            mock_series.data = MagicMock()
            mock_series.data.__getitem__ = lambda self, x: mock_data
            mock_series.data.shape = mock_data.shape
            mock_series.get_timestamps.return_value = mock_timestamps
            mock_series.rate = None

            mock_nwbfile = MagicMock()
            mock_nwbfile.objects.values.return_value = [mock_series]

            mock_io = MagicMock()
            mock_io.read.return_value = mock_nwbfile
            mock_io.__enter__ = MagicMock(return_value=mock_io)
            mock_io.__exit__ = MagicMock(return_value=False)

            with patch("fiber_mosaic.extractors.nwb_extractor._check_pynwb"):
                with patch(
                    "fiber_mosaic.extractors.nwb_extractor.NWBHDF5IO",
                    return_value=mock_io,
                ):
                    with pytest.raises(
                        ValueError, match="color must be provided"
                    ):
                        NwbFiberPhotometryExtractor(
                            str(nwb_path), series_name="FPSeries"
                        )

    def test_nwb_extractor_with_rate(self):
        """Test that series.rate is used as sampling_rate when set."""
        from fiber_mosaic.extractors import NwbFiberPhotometryExtractor

        with tempfile.TemporaryDirectory() as tmpdir:
            nwb_path = Path(tmpdir) / "session.nwb"
            nwb_path.touch()

            mock_data = np.random.randn(100)
            mock_timestamps = np.linspace(0, 10, 100)

            mock_series = MagicMock()
            mock_series.name = "FPSeries"
            mock_series.neurodata_type = "FiberPhotometryResponseSeries"
            mock_series.data = MagicMock()
            mock_series.data.__getitem__ = lambda self, x: mock_data
            mock_series.data.shape = mock_data.shape
            mock_series.get_timestamps.return_value = mock_timestamps
            mock_series.rate = 20.0

            mock_nwbfile = MagicMock()
            mock_nwbfile.objects.values.return_value = [mock_series]

            mock_io = MagicMock()
            mock_io.read.return_value = mock_nwbfile
            mock_io.__enter__ = MagicMock(return_value=mock_io)
            mock_io.__exit__ = MagicMock(return_value=False)

            with patch("fiber_mosaic.extractors.nwb_extractor._check_pynwb"):
                with patch(
                    "fiber_mosaic.extractors.nwb_extractor.NWBHDF5IO",
                    return_value=mock_io,
                ):
                    rec = NwbFiberPhotometryExtractor(
                        str(nwb_path), series_name="FPSeries", color="green"
                    )
                    assert rec.get_sampling_frequency() == 20.0

    def test_nwb_extractor_single_sample_no_rate(self):
        """Test sampling_rate fallback to 1.0 for single-sample series."""
        from fiber_mosaic.extractors import NwbFiberPhotometryExtractor

        with tempfile.TemporaryDirectory() as tmpdir:
            nwb_path = Path(tmpdir) / "session.nwb"
            nwb_path.touch()

            mock_data = np.random.randn(1)
            mock_timestamps = np.array([0.0])

            mock_series = MagicMock()
            mock_series.name = "FPSeries"
            mock_series.neurodata_type = "FiberPhotometryResponseSeries"
            mock_series.data = MagicMock()
            mock_series.data.__getitem__ = lambda self, x: mock_data
            mock_series.data.shape = mock_data.shape
            mock_series.get_timestamps.return_value = mock_timestamps
            mock_series.rate = None

            mock_nwbfile = MagicMock()
            mock_nwbfile.objects.values.return_value = [mock_series]

            mock_io = MagicMock()
            mock_io.read.return_value = mock_nwbfile
            mock_io.__enter__ = MagicMock(return_value=mock_io)
            mock_io.__exit__ = MagicMock(return_value=False)

            with patch("fiber_mosaic.extractors.nwb_extractor._check_pynwb"):
                with patch(
                    "fiber_mosaic.extractors.nwb_extractor.NWBHDF5IO",
                    return_value=mock_io,
                ):
                    rec = NwbFiberPhotometryExtractor(
                        str(nwb_path), series_name="FPSeries", color="green"
                    )
                    assert rec.get_sampling_frequency() == 1.0

    def test_dandi_no_color_raises(self):
        """Test error when color is not provided."""
        from fiber_mosaic.extractors import DandiFiberPhotometryExtractor

        mock_data = np.random.randn(100)
        mock_timestamps = np.linspace(0, 10, 100)

        mock_series = MagicMock()
        mock_series.name = "OnlySeries"
        mock_series.neurodata_type = "FiberPhotometryResponseSeries"
        mock_series.data = MagicMock()
        mock_series.data.__getitem__ = lambda self, x: mock_data
        mock_series.data.shape = mock_data.shape
        mock_series.timestamps = MagicMock()
        mock_series.timestamps.__getitem__ = lambda self, x: mock_timestamps
        mock_series.timestamps.__len__ = lambda self: len(mock_timestamps)
        mock_series.timestamps.__bool__ = lambda self: True

        mock_nwbfile = MagicMock()
        mock_nwbfile.objects.values.return_value = [mock_series]

        mock_io = MagicMock()

        with patch(
            "fiber_mosaic.extractors.dandi_extractor._check_dandi_dependencies"
        ):
            with patch(
                "fiber_mosaic.extractors.dandi_extractor._stream_nwb",
                return_value=(mock_nwbfile, mock_io),
            ):
                with pytest.raises(ValueError, match="color must be provided"):
                    DandiFiberPhotometryExtractor(
                        "dandi://000123/sub-1/file.nwb"
                    )


# =============================================================================
# TDT Extractor Full Coverage Tests
# =============================================================================


class TestTdtExtractorFullCoverage:
    """Full coverage tests for TDT extractor."""

    def test_tdt_read_tsq_success(self):
        """Test successful TSQ file reading."""
        from fiber_mosaic.extractors.tdt_extractor import _read_tsq

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a minimal valid TSQ file
            tsq_path = Path(tmpdir) / "test.tsq"

            # TSQ format: size(4), type(4), name(4), chan(2), sort_code(2),
            #             timestamp(8), fp_loc(8), format(4), frequency(4)
            dtype = np.dtype(
                [
                    ("size", np.int32),
                    ("type", np.int32),
                    ("name", "S4"),
                    ("chan", np.uint16),
                    ("sort_code", np.uint16),
                    ("timestamp", np.float64),
                    ("fp_loc", np.int64),
                    ("strobe", np.float64),
                    ("format", np.int32),
                    ("frequency", np.float32),
                ]
            )

            # Create a minimal TSQ record
            record = np.array(
                [
                    (40, 2, b"Dv1A", 1, 0, 0.0, 40, 0.0, 0, 1000.0),
                ],
                dtype=dtype,
            )
            record.tofile(str(tsq_path))

            df, found = _read_tsq(tmpdir)
            assert found is True
            assert df is not None

    def test_tdt_get_store_names(self):
        """Test _get_store_names function."""
        from fiber_mosaic.extractors.tdt_extractor import _get_store_names

        # Create a mock DataFrame
        df = pd.DataFrame(
            {
                "name": [b"Dv1A", b"Dv2A", b"xxx", b"Dv1A"],  # Some duplicates
            }
        )

        names = _get_store_names(df)
        assert "Dv1A" in names
        assert "Dv2A" in names
        assert "xxx" not in names  # Too short

    def test_tdt_read_tev_store_not_found(self):
        """Test error when store not found in TEV."""
        from fiber_mosaic.extractors.tdt_extractor import _read_tev_store

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create minimal TEV file
            tev_path = Path(tmpdir) / "test.tev"
            tev_path.write_bytes(b"\x00" * 100)

            header_df = pd.DataFrame(
                {
                    "name": [b"Dv1A"],
                    "format": [0],
                    "frequency": [1000.0],
                    "timestamp": [0.0],
                    "chan": [1],
                    "fp_loc": [0],
                    "size": [50],
                }
            )

            with pytest.raises(ValueError, match="not found"):
                _read_tev_store(Path(tmpdir), header_df, "NonexistentStore")

    def test_tdt_read_tev_store_no_tev_file(self):
        """Test error when no TEV file found."""
        from fiber_mosaic.extractors.tdt_extractor import _read_tev_store

        with tempfile.TemporaryDirectory() as tmpdir:
            header_df = pd.DataFrame(
                {
                    "name": [b"Dv1A"],
                }
            )

            with pytest.raises(FileNotFoundError, match="No TEV file"):
                _read_tev_store(Path(tmpdir), header_df, "Dv1A")

    def test_tdt_read_tev_store_multiple_tev(self):
        """Test error when multiple TEV files found."""
        from fiber_mosaic.extractors.tdt_extractor import _read_tev_store

        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "file1.tev").touch()
            (Path(tmpdir) / "file2.tev").touch()

            header_df = pd.DataFrame(
                {
                    "name": [b"Dv1A"],
                }
            )

            with pytest.raises(ValueError, match="Multiple TEV files"):
                _read_tev_store(Path(tmpdir), header_df, "Dv1A")

    def test_tdt_extractor_no_tdt_files(self):
        """Test error when no TDT files found."""
        from fiber_mosaic.extractors import TdtFiberPhotometryExtractor

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError, match="No TDT files"):
                TdtFiberPhotometryExtractor(tmpdir)

    def test_tdt_extractor_no_stores(self):
        """Test error when no valid stores found."""
        from fiber_mosaic.extractors import TdtFiberPhotometryExtractor

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create TSQ with no valid stores (names too short)
            tsq_path = Path(tmpdir) / "test.tsq"

            dtype = np.dtype(
                [
                    ("size", np.int32),
                    ("type", np.int32),
                    ("name", "S4"),
                    ("chan", np.uint16),
                    ("sort_code", np.uint16),
                    ("timestamp", np.float64),
                    ("fp_loc", np.int64),
                    ("strobe", np.float64),
                    ("format", np.int32),
                    ("frequency", np.float32),
                ]
            )

            # Name too short - will be filtered
            record = np.array(
                [
                    (40, 2, b"x", 1, 0, 0.0, 40, 0.0, 0, 1000.0),
                ],
                dtype=dtype,
            )
            record.tofile(str(tsq_path))

            with pytest.raises(ValueError, match="No valid stores"):
                TdtFiberPhotometryExtractor(tmpdir)

    def test_tdt_extractor_multiple_stores_no_name(self):
        """Test error when multiple stores and no name specified."""
        from fiber_mosaic.extractors import TdtFiberPhotometryExtractor

        with tempfile.TemporaryDirectory() as tmpdir:
            tsq_path = Path(tmpdir) / "test.tsq"

            # Match the real TDT layout used by the source reader: fp_loc
            # and strobe form an 8-byte union at offset 24 (40-byte records).
            dtype = np.dtype(
                {
                    "names": (
                        "size",
                        "type",
                        "name",
                        "chan",
                        "sort_code",
                        "timestamp",
                        "fp_loc",
                        "format",
                        "frequency",
                    ),
                    "formats": (
                        np.int32,
                        np.int32,
                        "S4",
                        np.uint16,
                        np.uint16,
                        np.float64,
                        np.int64,
                        np.int32,
                        np.float32,
                    ),
                    "offsets": (0, 4, 8, 12, 14, 16, 24, 32, 36),
                },
                align=True,
            )

            record = np.array(
                [
                    (40, 2, b"Dv1A", 1, 0, 0.0, 40, 0, 1000.0),
                    (40, 2, b"Dv2A", 1, 0, 0.0, 80, 0, 1000.0),
                ],
                dtype=dtype,
            )
            record.tofile(str(tsq_path))

            with pytest.raises(ValueError, match="Multiple stores"):
                TdtFiberPhotometryExtractor(tmpdir)

    def test_tdt_extractor_store_not_found(self):
        """Test error when specified store not found."""
        from fiber_mosaic.extractors import TdtFiberPhotometryExtractor

        with tempfile.TemporaryDirectory() as tmpdir:
            tsq_path = Path(tmpdir) / "test.tsq"

            dtype = np.dtype(
                [
                    ("size", np.int32),
                    ("type", np.int32),
                    ("name", "S4"),
                    ("chan", np.uint16),
                    ("sort_code", np.uint16),
                    ("timestamp", np.float64),
                    ("fp_loc", np.int64),
                    ("strobe", np.float64),
                    ("format", np.int32),
                    ("frequency", np.float32),
                ]
            )

            record = np.array(
                [
                    (40, 2, b"Dv1A", 1, 0, 0.0, 40, 0.0, 0, 1000.0),
                ],
                dtype=dtype,
            )
            record.tofile(str(tsq_path))

            with pytest.raises(ValueError, match="not found"):
                TdtFiberPhotometryExtractor(
                    tmpdir, store_name="NonexistentStore"
                )

    def test_tdt_extractor_event_store(self):
        """Test error when store is event/strobe data."""
        from fiber_mosaic.extractors import TdtFiberPhotometryExtractor

        with tempfile.TemporaryDirectory() as tmpdir:
            tsq_path = Path(tmpdir) / "test.tsq"
            tev_path = Path(tmpdir) / "test.tev"

            # Event/strobe records store the strobe value in the offset-24
            # union (where continuous records keep fp_loc).
            dtype = np.dtype(
                {
                    "names": (
                        "size",
                        "type",
                        "name",
                        "chan",
                        "sort_code",
                        "timestamp",
                        "strobe",
                        "format",
                        "frequency",
                    ),
                    "formats": (
                        np.int32,
                        np.int32,
                        "S4",
                        np.uint16,
                        np.uint16,
                        np.float64,
                        np.float64,
                        np.int32,
                        np.float32,
                    ),
                    "offsets": (0, 4, 8, 12, 14, 16, 24, 32, 36),
                },
                align=True,
            )

            # format=4 (code 5 after +1) means strobe/event data
            record = np.array(
                [
                    (10, 2, b"Evnt", 1, 0, 0.0, 123.0, 4, 0.0),
                ],
                dtype=dtype,
            )
            record.tofile(str(tsq_path))

            # Create minimal TEV file
            tev_path.write_bytes(b"\x00" * 100)

            with pytest.raises(ValueError, match="event/strobe store"):
                TdtFiberPhotometryExtractor(tmpdir, store_name="Evnt")

    def test_tdt_extractor_full_success(self):
        """Test successful TDT extractor initialization."""
        from fiber_mosaic.extractors import TdtFiberPhotometryExtractor

        with tempfile.TemporaryDirectory() as tmpdir:
            tsq_path = Path(tmpdir) / "test.tsq"
            tev_path = Path(tmpdir) / "test.tev"

            # 40-byte records with fp_loc in the offset-24 union.
            dtype = np.dtype(
                {
                    "names": (
                        "size",
                        "type",
                        "name",
                        "chan",
                        "sort_code",
                        "timestamp",
                        "fp_loc",
                        "format",
                        "frequency",
                    ),
                    "formats": (
                        np.int32,
                        np.int32,
                        "S4",
                        np.uint16,
                        np.uint16,
                        np.float64,
                        np.int64,
                        np.int32,
                        np.float32,
                    ),
                    "offsets": (0, 4, 8, 12, 14, 16, 24, 32, 36),
                },
                align=True,
            )

            # format=0 means float32 data
            # size=50 means (50-10)*1 = 40 float32 samples per chunk
            record = np.array(
                [
                    (50, 2, b"Dv1A", 1, 0, 0.0, 0, 0, 1000.0),
                    (50, 2, b"Dv1A", 1, 0, 0.04, 160, 0, 1000.0),
                ],
                dtype=dtype,
            )
            record.tofile(str(tsq_path))

            # Create TEV file with float32 data
            # Each chunk has 40 float32 values
            data = np.arange(80, dtype=np.float32)
            data.tofile(str(tev_path))

            rec = TdtFiberPhotometryExtractor(
                tmpdir, store_name="Dv1A", color="green"
            )
            assert rec.get_num_samples() == 80
            assert rec.color == "green"

    def test_tdt_get_streams(self):
        """Test TdtFiberPhotometryExtractor.get_streams."""
        from fiber_mosaic.extractors import TdtFiberPhotometryExtractor

        with tempfile.TemporaryDirectory() as tmpdir:
            tsq_path = Path(tmpdir) / "test.tsq"

            dtype = np.dtype(
                [
                    ("size", np.int32),
                    ("type", np.int32),
                    ("name", "S4"),
                    ("chan", np.uint16),
                    ("sort_code", np.uint16),
                    ("timestamp", np.float64),
                    ("fp_loc", np.int64),
                    ("strobe", np.float64),
                    ("format", np.int32),
                    ("frequency", np.float32),
                ]
            )

            record = np.array(
                [
                    (40, 2, b"Dv1A", 1, 0, 0.0, 40, 0.0, 0, 1000.0),
                ],
                dtype=dtype,
            )
            record.tofile(str(tsq_path))

            names, ids = TdtFiberPhotometryExtractor.get_streams(tmpdir)
            assert "Dv1A" in names

    def test_read_tdt_convenience_function(self):
        """Test read_tdt_fiber_photometry convenience function."""
        from fiber_mosaic.extractors.tdt_extractor import (
            read_tdt_fiber_photometry,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tsq_path = Path(tmpdir) / "test.tsq"
            tev_path = Path(tmpdir) / "test.tev"

            dtype = np.dtype(
                {
                    "names": (
                        "size",
                        "type",
                        "name",
                        "chan",
                        "sort_code",
                        "timestamp",
                        "fp_loc",
                        "format",
                        "frequency",
                    ),
                    "formats": (
                        np.int32,
                        np.int32,
                        "S4",
                        np.uint16,
                        np.uint16,
                        np.float64,
                        np.int64,
                        np.int32,
                        np.float32,
                    ),
                    "offsets": (0, 4, 8, 12, 14, 16, 24, 32, 36),
                },
                align=True,
            )

            record = np.array(
                [
                    (50, 2, b"Dv1A", 1, 0, 0.0, 0, 0, 1000.0),
                ],
                dtype=dtype,
            )
            record.tofile(str(tsq_path))

            data = np.arange(40, dtype=np.float32)
            data.tofile(str(tev_path))

            rec = read_tdt_fiber_photometry(
                tmpdir, store_name="Dv1A", color="green"
            )
            assert rec.get_num_samples() == 40


# =============================================================================
# Remaining branch coverage
# =============================================================================


class TestDetectFormatsSingleColumn:
    """Cover the single-column CSV classification branch."""

    def test_classify_csv_single_column_timestamps(self):
        """Single-column CSV named 'timestamps' is classified as csv."""
        from fiber_mosaic.extractors.detect_formats import classify_csv_file

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "single.csv"
            csv_path.write_text("timestamps\n0.0\n1.0\n")
            assert classify_csv_file(str(csv_path)) == "csv"


class TestDoricRealFiles:
    """Cover Doric branches using real HDF5 and CSV files."""

    def test_doric_find_files_bad_csv_ignored(self):
        """A CSV that fails to parse is skipped by _find_doric_files."""
        from fiber_mosaic.extractors.doric_extractor import _find_doric_files

        with tempfile.TemporaryDirectory() as tmpdir:
            # Empty file makes pandas raise -> caught and skipped.
            (Path(tmpdir) / "bad.csv").write_text("")
            doric_files, csv_files = _find_doric_files(Path(tmpdir))
            assert doric_files == []
            assert csv_files == []

    def test_doric_v1_dataset_ambiguous_group(self):
        """_get_v1_dataset raises when a wrapping group has multiple
        children and none matches the requested name."""
        import h5py

        from fiber_mosaic.extractors.doric_extractor import (
            _get_v1_dataset,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            doric_path = Path(tmpdir) / "test.doric"
            with h5py.File(str(doric_path), "w") as f:
                group = f.create_group("Ambiguous")
                group.create_dataset("A", data=np.array([1.0]))
                group.create_dataset("B", data=np.array([2.0]))

            with h5py.File(str(doric_path), "r") as f:
                with pytest.raises(ValueError, match="multiple candidates"):
                    _get_v1_dataset(f, "Ambiguous")

    def test_find_v6_parent_dataset_leaf_not_found(self):
        """_find_v6_parent_dataset returns None when the final path part
        does not resolve to a dataset in the matched group."""
        import h5py

        from fiber_mosaic.extractors.doric_extractor import (
            _find_v6_parent_dataset,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            doric_path = Path(tmpdir) / "test.doric"
            with h5py.File(str(doric_path), "w") as f:
                group = f.create_group("Group")
                group.create_dataset("Other", data=np.array([1.0]))

            with h5py.File(str(doric_path), "r") as f:
                result = _find_v6_parent_dataset(f["Group"], ["Missing"])
                assert result is None

    def test_doric_read_v1_data_single_sample(self):
        """Single-sample V1 data falls back to sampling_rate=1.0."""
        import h5py

        from fiber_mosaic.extractors.doric_extractor import (
            _read_doric_v1_data,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            doric_path = Path(tmpdir) / "test.doric"
            with h5py.File(str(doric_path), "w") as f:
                console = f.create_group("Traces/Console")
                console.create_dataset("Time(s)", data=np.array([0.0]))
                console.create_dataset("Signal", data=np.array([1.0]))

            with h5py.File(str(doric_path), "r") as f:
                data, timestamps, sr = _read_doric_v1_data(f, "Signal")

        assert sr == 1.0

    def test_doric_v6_end_to_end(self):
        """Full V6 read exercises _find_v6_group and _read_doric_file_data."""
        from fiber_mosaic.extractors import DoricFiberPhotometryExtractor

        with tempfile.TemporaryDirectory() as tmpdir:
            doric_path = Path(tmpdir) / "test.doric"
            self._write_v6_file(doric_path, n_samples=10)

            # Pass the folder so _resolve_doric_path locates the .doric file.
            rec = DoricFiberPhotometryExtractor(
                tmpdir, stream_name="AnalogIn/AIn-1", color="green"
            )
            assert rec.get_num_samples() == 10

    def test_doric_read_v6_data_single_sample(self):
        """Single-sample V6 data falls back to sampling_rate=1.0."""
        import h5py

        from fiber_mosaic.extractors.doric_extractor import (
            _read_doric_v6_data,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            doric_path = Path(tmpdir) / "test.doric"
            self._write_v6_file(doric_path, n_samples=1)

            with h5py.File(str(doric_path), "r") as f:
                data, timestamps, sr = _read_doric_v6_data(f, "AnalogIn/AIn-1")

        assert sr == 1.0

    def test_doric_read_csv_data_stream_not_found(self):
        """Requesting a missing CSV stream raises ValueError."""
        from fiber_mosaic.extractors.doric_extractor import (
            _read_doric_csv_data,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "data.csv"
            csv_path.write_text(
                "Console,Console\nTime(s),Signal\n0.0,1.0\n0.1,2.0\n"
            )
            with pytest.raises(ValueError, match="not found"):
                _read_doric_csv_data(csv_path, "Nonexistent")

    def test_doric_resolve_path_falls_back_to_csv(self):
        """_resolve_doric_path returns a Doric CSV when no .doric exists."""
        from fiber_mosaic.extractors.doric_extractor import (
            _resolve_doric_path,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "data.csv"
            csv_path.write_text("Console,Console\nTime(s),Signal\n0.0,1.0\n")
            result = _resolve_doric_path(Path(tmpdir))
            assert result == csv_path

    @staticmethod
    def _write_v6_file(doric_path: Path, n_samples: int) -> None:
        """Write a minimal Doric V6 HDF5 file."""
        import h5py

        with h5py.File(str(doric_path), "w") as f:
            f.create_group("Configurations")
            # A sibling subtree (sorted before FPConsole) that does not
            # contain the target group, so _find_v6_group must exhaust it
            # and return None before recursing into the real branch.
            extra = f.create_group("DataAcquisition/AAExtra")
            extra.create_dataset("Ignored", data=np.arange(3.0))
            ain = f.create_group(
                "DataAcquisition/FPConsole/Signals/Series0001/AnalogIn/AIn-1"
            )
            ain.create_dataset("Values", data=np.arange(float(n_samples)))
            ain.create_dataset("Time", data=np.linspace(0.0, 1.0, n_samples))


class TestNpmBranchCoverage:
    """Cover remaining NPM extractor branches."""

    def test_npm_no_streams_raises(self):
        """A file with no data columns raises ValueError."""
        from fiber_mosaic.extractors import NpmFiberPhotometryExtractor

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "npm.csv"
            csv_path.write_text("timestamp,time,ledstate\n0,0,1\n")
            with pytest.raises(ValueError, match="No streams found"):
                NpmFiberPhotometryExtractor(str(csv_path))

    def test_npm_single_stream_auto_selected(self):
        """A single available stream is selected automatically."""
        from fiber_mosaic.extractors import NpmFiberPhotometryExtractor

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "npm.csv"
            csv_path.write_text(
                "Timestamp,LedState,Region0G\n0,1,100.0\n1000,1,101.0\n"
            )
            rec = NpmFiberPhotometryExtractor(str(csv_path), color="green")
            assert rec.get_num_samples() == 2


class TestTdtBranchCoverage:
    """Cover the single-store auto-selection branch."""

    def test_tdt_single_store_auto_selected(self):
        """A single valid store is selected automatically."""
        from fiber_mosaic.extractors import TdtFiberPhotometryExtractor

        with tempfile.TemporaryDirectory() as tmpdir:
            tsq_path = Path(tmpdir) / "test.tsq"
            tev_path = Path(tmpdir) / "test.tev"

            dtype = np.dtype(
                {
                    "names": (
                        "size",
                        "type",
                        "name",
                        "chan",
                        "sort_code",
                        "timestamp",
                        "fp_loc",
                        "format",
                        "frequency",
                    ),
                    "formats": (
                        np.int32,
                        np.int32,
                        "S4",
                        np.uint16,
                        np.uint16,
                        np.float64,
                        np.int64,
                        np.int32,
                        np.float32,
                    ),
                    "offsets": (0, 4, 8, 12, 14, 16, 24, 32, 36),
                },
                align=True,
            )

            record = np.array(
                [
                    (50, 2, b"Dv1A", 1, 0, 0.0, 0, 0, 1000.0),
                ],
                dtype=dtype,
            )
            record.tofile(str(tsq_path))

            np.arange(40, dtype=np.float32).tofile(str(tev_path))

            rec = TdtFiberPhotometryExtractor(tmpdir, color="green")
            assert rec.get_num_samples() == 40

    def test_tdt_no_color_raises(self):
        """Test error when color is not provided."""
        from fiber_mosaic.extractors import TdtFiberPhotometryExtractor

        with tempfile.TemporaryDirectory() as tmpdir:
            tsq_path = Path(tmpdir) / "test.tsq"
            tev_path = Path(tmpdir) / "test.tev"

            dtype = np.dtype(
                {
                    "names": (
                        "size",
                        "type",
                        "name",
                        "chan",
                        "sort_code",
                        "timestamp",
                        "fp_loc",
                        "format",
                        "frequency",
                    ),
                    "formats": (
                        np.int32,
                        np.int32,
                        "S4",
                        np.uint16,
                        np.uint16,
                        np.float64,
                        np.int64,
                        np.int32,
                        np.float32,
                    ),
                    "offsets": (0, 4, 8, 12, 14, 16, 24, 32, 36),
                },
                align=True,
            )
            record = np.array(
                [(50, 2, b"Dv1A", 1, 0, 0.0, 0, 0, 1000.0)], dtype=dtype
            )
            record.tofile(str(tsq_path))
            np.arange(40, dtype=np.float32).tofile(str(tev_path))

            with pytest.raises(ValueError, match="color must be provided"):
                TdtFiberPhotometryExtractor(tmpdir, store_name="Dv1A")


RESOURCES_DIR = Path(__file__).resolve().parents[4] / "resources"


@pytest.mark.skipif(
    not RESOURCES_DIR.exists(), reason="resources/ directory not available"
)
class TestDoricGuPPySampleSessions:
    """Doric extractor tests against real sessions ported from GuPPy's
    stubbed_testing_data. These are truncated real recordings, not
    synthetic mocks, so they exercise the actual on-disk HDF5 layouts
    produced by Doric hardware (as opposed to the flatter layouts the
    rest of this test module assumes).
    """

    def test_sample_doric_1_v1_console_group(self):
        """V1 format where 'Time(s)' and each stream are groups wrapping
        a nested dataset, rather than datasets directly (regression
        test: previously raised TypeError from indexing a Group with a
        slice)."""
        from fiber_mosaic.extractors import DoricFiberPhotometryExtractor

        session_dir = RESOURCES_DIR / "doric" / "sample_doric_1"
        rec = DoricFiberPhotometryExtractor(
            session_dir, stream_name="AIn-1 - Raw", color="green"
        )
        assert rec.get_num_samples() > 0
        assert rec.color == "green"

    def test_sample_doric_1_get_streams(self):
        """Stream discovery lists the V1 console channels."""
        from fiber_mosaic.extractors import DoricFiberPhotometryExtractor

        doric_file = (
            RESOURCES_DIR
            / "doric"
            / "sample_doric_1"
            / "D2-EPConsole_0039.doric"
        )
        names, _ = DoricFiberPhotometryExtractor.get_streams(doric_file)
        assert "AIn-1 - Raw" in names
        assert "AIn-2 - Raw" in names
        assert "DI--O-1" in names

    def test_sample_doric_3_v6_nested_roi_datasets(self):
        """V6 format where sibling ROI datasets (ROI01/ROI02/ROI03) live
        directly inside a shared group alongside 'Time', rather than
        each having its own 'Values'-bearing group (regression test:
        previously raised ValueError since _find_v6_group only matches
        nested Groups, not sibling Datasets)."""
        from fiber_mosaic.extractors import DoricFiberPhotometryExtractor

        session_dir = RESOURCES_DIR / "doric" / "sample_doric_3"
        rec1 = DoricFiberPhotometryExtractor(
            session_dir, stream_name="CAM1_EXC1/ROI01", color="470nm"
        )
        rec2 = DoricFiberPhotometryExtractor(
            session_dir, stream_name="CAM1_EXC1/ROI02", color="415nm"
        )
        assert rec1.get_num_samples() > 0
        assert rec1.get_num_samples() == rec2.get_num_samples()
        # Distinct ROIs must not silently resolve to the same trace.
        assert not np.array_equal(rec1.get_traces(), rec2.get_traces())

    def test_sample_doric_3_v6_nested_ttl_dataset(self):
        """V6 sibling-dataset resolution also covers the DigitalIO/CAM1
        TTL channel (not just ROI signal channels)."""
        from fiber_mosaic.extractors import DoricFiberPhotometryExtractor

        session_dir = RESOURCES_DIR / "doric" / "sample_doric_3"
        rec = DoricFiberPhotometryExtractor(
            session_dir, stream_name="DigitalIO/CAM1", color="ttl"
        )
        assert rec.get_num_samples() > 0

    def test_sample_doric_4_v6_lockin_group_with_values(self):
        """V6 format where the leaf group has its own 'Values'/'Time'
        pair (the pre-existing, already-working V6 shape) still works
        after the sibling-dataset fallback was added."""
        from fiber_mosaic.extractors import DoricFiberPhotometryExtractor

        session_dir = RESOURCES_DIR / "doric" / "sample_doric_4"
        rec = DoricFiberPhotometryExtractor(
            session_dir,
            stream_name="Series0001/AIN01xAOUT01-LockIn",
            color="green",
        )
        assert rec.get_num_samples() > 0

    def test_sample_doric_5_v6_lockin_group_with_values(self):
        """Second independent V6 lock-in example (no TTL events)."""
        from fiber_mosaic.extractors import DoricFiberPhotometryExtractor

        session_dir = RESOURCES_DIR / "doric" / "sample_doric_5"
        rec = DoricFiberPhotometryExtractor(
            session_dir,
            stream_name="Series0001/AIN01xAOUT01-LockIn",
            color="green",
        )
        assert rec.get_num_samples() > 0

    def test_sample_doric_2_csv_export(self):
        """Doric CSV export format (distinct from GuPPy's generic CSV
        layout) still resolves via the CSV code path."""
        from fiber_mosaic.extractors import DoricFiberPhotometryExtractor

        session_dir = RESOURCES_DIR / "doric" / "sample_doric_2"
        rec = DoricFiberPhotometryExtractor(
            session_dir, stream_name="AIn-1 - Dem (da)", color="470nm"
        )
        assert rec.get_num_samples() > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
