"""initialize fiber mosaic."""

from importlib.metadata import PackageNotFoundError, version

from .core.base import (
    BaseFiberPhotometryExtractor,
    FiberPhotometryRecordingGroup,
)
from .processing import (
    apply_pipeline,
    bleach_correction,
    concatenate_segments,
    isosbestic_correction,
    processing_step,
    register_step,
    registered_steps,
    validate_pipeline,
    validate_timeline,
)

try:
    __version__ = version("fiber-mosaic")
except PackageNotFoundError:  # running from a source tree without install
    __version__ = "0.0.0+unknown"

__all__ = [
    "BaseFiberPhotometryExtractor",
    "FiberPhotometryRecordingGroup",
    "apply_pipeline",
    "bleach_correction",
    "concatenate_segments",
    "isosbestic_correction",
    "processing_step",
    "register_step",
    "registered_steps",
    "validate_pipeline",
    "validate_timeline",
]
