"""initialize fiber mosaic."""

__version__ = "0.1.0"

from .core.base import (
    BaseFiberPhotometryExtractor,
    BaseFiberPhotometrySegment,
    FiberPhotometryRecordingGroup,
)

__all__ = [
    "BaseFiberPhotometryExtractor",
    "BaseFiberPhotometrySegment",
    "FiberPhotometryRecordingGroup",
]
