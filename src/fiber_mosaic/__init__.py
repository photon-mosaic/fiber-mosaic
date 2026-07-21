"""initialize fiber mosaic."""

from importlib.metadata import PackageNotFoundError, version

from .core.base import (
    BaseFiberPhotometryExtractor,
    FiberPhotometryRecordingGroup,
)
from .processing import (
    apply_pipeline,
    apply_to_group,
    bleach_correction,
    isosbestic_correction,
    map_segments,
    pipeline_step,
    recording_like,
    register_step,
    registered_steps,
)

try:
    __version__ = version("fiber-mosaic")
except PackageNotFoundError:  # running from a source tree without install
    __version__ = "0.0.0+unknown"

__all__ = [
    "BaseFiberPhotometryExtractor",
    "FiberPhotometryRecordingGroup",
    "apply_pipeline",
    "apply_to_group",
    "bleach_correction",
    "isosbestic_correction",
    "map_segments",
    "pipeline_step",
    "recording_like",
    "register_step",
    "registered_steps",
]
