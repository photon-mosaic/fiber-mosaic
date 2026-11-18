"""
Core module for fiber-mosaic.

Contains base classes for fiber photometry data handling:
- FiberPhotometryMixin: Fiber-native API, mixable into any SI recording
- BaseFiberPhotometryRecording: Single named recording
- FiberPhotometryRecordingGroup: Container for named recordings
- recording_from_traces: General-purpose constructor from in-memory arrays
"""

from .base import (
    BaseFiberPhotometryRecording,
    FiberPhotometryMixin,
    FiberPhotometryRecordingGroup,
    recording_from_traces,
)

__all__ = [
    "BaseFiberPhotometryRecording",
    "FiberPhotometryMixin",
    "FiberPhotometryRecordingGroup",
    "recording_from_traces",
]
