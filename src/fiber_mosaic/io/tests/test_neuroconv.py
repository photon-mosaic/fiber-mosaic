"""Pytest tests for ``fiber_mosaic.io.neuroconv``."""

from __future__ import annotations

import numpy as np
from pynwb.testing.mock.file import mock_NWBFile
from spikeinterface.preprocessing import bandpass_filter

from fiber_mosaic.io.neuroconv import FiberMosaicPhotometryInterface
from fiber_mosaic.io.nwb import dump_recording


def _write(recording, tmp_path):
    """Dump ``recording`` and write it through the interface."""
    interface = FiberMosaicPhotometryInterface(
        dump_recording(recording, tmp_path / "recording.json")
    )
    nwbfile = mock_NWBFile()
    interface.add_to_nwbfile(nwbfile)
    return nwbfile


def test_source_schema_takes_a_path():
    schema = FiberMosaicPhotometryInterface.get_source_schema()

    assert set(schema["properties"]) == {"file_path", "verbose"}


def test_raw_recording_goes_to_acquisition_only(recording, tmp_path):
    nwbfile = _write(recording, tmp_path)

    assert list(nwbfile.acquisition) == ["FiberPhotometryResponseSeriesGreen"]
    assert not nwbfile.processing


def test_processed_two_fiber_recording_adds_its_raw_reader(
    two_fiber_recording, tmp_path
):
    processed = bandpass_filter(
        two_fiber_recording,
        freq_min=0.1,
        freq_max=5.0,
        ignore_low_freq_error=True,
    )

    nwbfile = _write(processed, tmp_path)

    raw_series = nwbfile.acquisition["FiberPhotometryResponseSeriesGreen"]
    processed_series = nwbfile.processing["ophys"][
        "FiberPhotometryResponseSeriesGreenProcessed"
    ]
    np.testing.assert_allclose(
        raw_series.data, two_fiber_recording.get_traces()
    )
    np.testing.assert_allclose(processed_series.data, processed.get_traces())
