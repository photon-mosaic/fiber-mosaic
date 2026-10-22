"""Pytest tests for ``fiber_mosaic.io.nwb``."""

from __future__ import annotations

import numpy as np
import spikeinterface as si

from fiber_mosaic.io.nwb import dump_group, dump_recording


def test_dump_recording_round_trips(recording, tmp_path):
    path = dump_recording(recording, tmp_path / "recording.json")

    assert path == tmp_path / "recording.json"
    np.testing.assert_array_equal(
        si.load(path).get_traces(), recording.get_traces()
    )


def test_dump_group_names_files_after_each_raw_source(group, tmp_path):
    paths = dump_group(group, tmp_path)

    assert paths == {
        "green": tmp_path / "green_green.json",
        "iso": tmp_path / "iso_iso.json",
    }
    for name, path in paths.items():
        np.testing.assert_array_equal(
            si.load(path).get_traces(), group[name].get_traces()
        )


def test_dump_group_uses_a_given_prefix(group, tmp_path):
    paths = dump_group(group, tmp_path, prefix="session")

    assert paths == {
        "green": tmp_path / "session_green.json",
        "iso": tmp_path / "session_iso.json",
    }
