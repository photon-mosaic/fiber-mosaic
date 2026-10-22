"""Shared fixtures for ``fiber_mosaic.io`` tests."""

from __future__ import annotations

import numpy as np
import pytest

from fiber_mosaic.core.base import FiberPhotometryRecordingGroup
from fiber_mosaic.extractors import CsvFiberPhotometryExtractor


def _csv_recording(folder, name, fiber_ids=("fiber_a",)):
    """Write ``{name}.csv`` with one sine column per fiber and read it."""
    times = np.arange(200) / 20.0
    traces = np.sin(np.outer(times, np.arange(1, 1 + len(fiber_ids))))
    csv_path = folder / f"{name}.csv"
    np.savetxt(
        csv_path,
        np.column_stack([times, traces]),
        delimiter=",",
        header=",".join(["time", *fiber_ids]),
        comments="",
    )
    return CsvFiberPhotometryExtractor(csv_path, name=name, time_column="time")


@pytest.fixture
def recording(tmp_path):
    """One-fiber green recording read from a CSV file in ``tmp_path``."""
    return _csv_recording(tmp_path, "green")


@pytest.fixture
def group(tmp_path):
    """Green and iso recordings, each read from its own CSV file."""
    return FiberPhotometryRecordingGroup(
        {name: _csv_recording(tmp_path, name) for name in ("green", "iso")}
    )


@pytest.fixture
def two_fiber_recording(tmp_path):
    """Two-fiber green recording read from a CSV file in ``tmp_path``."""
    return _csv_recording(tmp_path, "green", ("fiber_a", "fiber_b"))
