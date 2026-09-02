"""Pytest tests for ``fiber_mosaic.core.core_tools``."""

from __future__ import annotations

import numpy as np
import pytest
from spikeinterface.core.numpyextractors import NumpyRecordingSegment

from fiber_mosaic.core.base import (
    BaseFiberPhotometryExtractor,
    FiberPhotometryRecordingGroup,
)
from fiber_mosaic.core.core_tools import (
    define_function_handling_group_from_class,
)


class _Doubler:
    """Minimal stand-in for a preprocessor class.

    Records what it was called with, so dispatch can be asserted without
    depending on any real preprocessing behaviour. Forwards the fiber IDs so
    that a group built from its results still validates.
    """

    def __init__(self, recording, factor=2):
        self.recording = recording
        self.factor = factor

    def get_channel_ids(self):
        """Return the wrapped recording's fiber IDs."""
        return self.recording.get_channel_ids()


double = define_function_handling_group_from_class(_Doubler, "double")


def _make_recording(color="green", fiber_ids=("f0", "f1")):
    """Build a small single-segment recording."""
    rec = BaseFiberPhotometryExtractor(
        sampling_frequency=100.0,
        fiber_ids=list(fiber_ids),
        color=color,
        dtype="float32",
    )
    rec.add_segment(
        NumpyRecordingSegment(
            traces=np.zeros((10, len(fiber_ids)), dtype="float32"),
            sampling_frequency=100.0,
            t_start=None,
        )
    )
    return rec


def test_dispatch_on_single_recording():
    """A recording in, a single preprocessed object out."""
    rec = _make_recording()
    result = double(rec)
    assert isinstance(result, _Doubler)
    assert result.recording is rec
    assert result.factor == 2


def test_dispatch_forwards_params():
    """Extra keywords reach the wrapped class."""
    rec = _make_recording()
    assert double(rec, factor=5).factor == 5


def test_dispatch_accepts_recording_as_keyword():
    """The input can be passed as ``recording=`` rather than positionally."""
    rec = _make_recording()
    result = double(recording=rec, factor=3)
    assert result.recording is rec
    assert result.factor == 3


def test_dispatch_on_dict():
    """A dict in, a dict with the same keys out."""
    recordings = {
        "green": _make_recording("green"),
        "iso": _make_recording("iso"),
    }
    result = double(recordings)
    assert isinstance(result, dict)
    assert set(result) == {"green", "iso"}
    assert result["iso"].recording is recordings["iso"]


def test_dispatch_on_empty_dict():
    """An empty dict dispatches as a dict, not as a missing argument.

    SpikeInterface's dict-only version uses a truthiness test here, so an
    empty dict falls through to the positional branch and raises IndexError.
    """
    assert double({}) == {}


def test_dispatch_on_group_returns_group():
    """A group in, a new group out, with colors preserved in order."""
    green = _make_recording("green")
    iso = _make_recording("iso")
    group = FiberPhotometryRecordingGroup({"green": green, "iso": iso})

    result = double(group, factor=4)

    assert isinstance(result, FiberPhotometryRecordingGroup)
    assert result.colors == ["green", "iso"]
    assert result["green"].recording is green
    assert result["iso"].factor == 4


def test_dispatch_leaves_input_group_unchanged():
    """Dispatch builds a new group rather than mutating the input."""
    green = _make_recording("green")
    group = FiberPhotometryRecordingGroup({"green": green})

    double(group)

    assert group["green"] is green


def test_dispatch_rejects_unsupported_type():
    """Anything that is not a recording, dict, or group is an error."""
    with pytest.raises(TypeError, match="accepts a recording"):
        double("not-a-recording")


def test_dispatch_requires_an_argument():
    """Calling with no input at all is an error, not an IndexError."""
    with pytest.raises(TypeError, match="requires a recording"):
        double()


def test_wrapper_copies_introspection_metadata():
    """Name, docstring and signature come from the wrapped class."""
    assert double.__name__ == "double"
    assert double.__doc__ == _Doubler.__doc__
    assert list(double.__signature__.parameters) == ["recording", "factor"]
