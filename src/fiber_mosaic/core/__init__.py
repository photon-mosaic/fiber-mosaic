"""
Core module for fiber-mosaic.

Contains base classes for fiber photometry data handling:
- FiberPhotometryMixin: Fiber-native API, mixable into any SI recording
- BaseFiberPhotometryExtractor: Single-color recording
- FiberPhotometryRecordingGroup: Multi-color container
- recording_from_traces: General-purpose constructor from in-memory arrays
"""

from .base import (
    BaseFiberPhotometryExtractor,
    FiberPhotometryMixin,
    FiberPhotometryRecordingGroup,
    recording_from_traces,
)

__all__ = [
    "BaseFiberPhotometryExtractor",
    "FiberPhotometryMixin",
    "FiberPhotometryRecordingGroup",
    "recording_from_traces",
]
