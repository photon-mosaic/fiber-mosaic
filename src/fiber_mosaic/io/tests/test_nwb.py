"""Pytest tests for ``fiber_mosaic.io.nwb``."""

from __future__ import annotations

import numpy as np
import spikeinterface as si

from fiber_mosaic.io.nwb import dump_recording


def test_dump_recording_round_trips(recording, tmp_path):
    path = dump_recording(recording, tmp_path / "recording.json")

    assert path == tmp_path / "recording.json"
    np.testing.assert_array_equal(
        si.load(path).get_traces(), recording.get_traces()
    )
