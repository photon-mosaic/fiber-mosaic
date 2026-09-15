"""
Core module for fiber-mosaic.

Contains base classes for fiber photometry data handling:
- FiberPhotometryMixin: Fiber-native API, mixable into any SI recording
- BaseFiberPhotometryExtractor: Single-color recording
- FiberPhotometryRecordingGroup: Multi-color container
"""

from .base import (
    BaseFiberPhotometryExtractor,
    FiberPhotometryMixin,
    FiberPhotometryRecordingGroup,
)

__all__ = [
    "BaseFiberPhotometryExtractor",
    "FiberPhotometryMixin",
    "FiberPhotometryRecordingGroup",
]
