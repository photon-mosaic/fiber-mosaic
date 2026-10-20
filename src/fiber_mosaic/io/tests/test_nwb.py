"""Pytest tests for ``fiber_mosaic.io.nwb``."""

from __future__ import annotations

import numpy as np
import spikeinterface as si

from fiber_mosaic.extractors import CsvFiberPhotometryExtractor
from fiber_mosaic.io.nwb import dump_recording


def _make_recording(folder):
    """Write a one-fiber CSV into ``folder`` and read it back."""
    times = np.arange(200) / 20.0
    rows = [f"{t},{np.sin(t)}" for t in times]
    csv_path = folder / "green.csv"
    csv_path.write_text("\n".join(["time,fiber_a", *rows]))
    return CsvFiberPhotometryExtractor(
        csv_path, color="green", time_column="time"
    )


def test_dump_recording_round_trips(tmp_path):
    recording = _make_recording(tmp_path)

    path = dump_recording(recording, tmp_path / "recording.json")

    assert path == tmp_path / "recording.json"
    np.testing.assert_array_equal(
        si.load(path).get_traces(), recording.get_traces()
    )
