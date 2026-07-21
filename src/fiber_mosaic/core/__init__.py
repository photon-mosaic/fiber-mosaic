"""
Core module for fiber-mosaic.

Contains base classes for fiber photometry data handling:
- BaseFiberPhotometryExtractor: Single-color recording
- FiberPhotometryRecordingGroup: Multi-color container
"""

from .base import BaseFiberPhotometryExtractor, FiberPhotometryRecordingGroup

__all__ = [
    "BaseFiberPhotometryExtractor",
    "FiberPhotometryRecordingGroup",
]
