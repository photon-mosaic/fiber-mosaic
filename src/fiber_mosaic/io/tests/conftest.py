"""Shared fixtures for ``fiber_mosaic.io`` tests."""

from __future__ import annotations

import numpy as np
import pytest

from fiber_mosaic.extractors import CsvFiberPhotometryExtractor


@pytest.fixture
def recording(tmp_path):
    """One-fiber green recording read from a CSV file in ``tmp_path``."""
    times = np.arange(200) / 20.0
    rows = [f"{t},{np.sin(t)}" for t in times]
    csv_path = tmp_path / "green.csv"
    csv_path.write_text("\n".join(["time,fiber_a", *rows]))
    return CsvFiberPhotometryExtractor(
        csv_path, color="green", time_column="time"
    )
