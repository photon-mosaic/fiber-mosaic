"""Fiber-photometry processing layer: steps, registry, and pipeline.

The counterpart to :mod:`spikeinterface.preprocessing`, sitting beside
:mod:`fiber_mosaic.core` (the recording layer). Importing this package
registers the built-in steps defined in :mod:`.steps`.
"""

from fiber_mosaic.processing.helpers import map_segments, recording_like
from fiber_mosaic.processing.pipeline import apply_pipeline, validate_pipeline
from fiber_mosaic.processing.registry import (
    get_step,
    pipeline_step,
    register_step,
    registered_steps,
)
from fiber_mosaic.processing.steps import (
    apply_to_group,
    bandpass_filter,
    bleach_correction,
    center,
    common_reference,
    decimate,
    gaussian_filter,
    highpass_filter,
    isosbestic_correction,
    normalize_by_quantile,
    notch_filter,
    resample,
    scale,
    zscore,
)

__all__ = [
    "apply_pipeline",
    "apply_to_group",
    "bandpass_filter",
    "bleach_correction",
    "center",
    "common_reference",
    "decimate",
    "gaussian_filter",
    "get_step",
    "highpass_filter",
    "isosbestic_correction",
    "map_segments",
    "normalize_by_quantile",
    "notch_filter",
    "pipeline_step",
    "recording_like",
    "register_step",
    "registered_steps",
    "resample",
    "scale",
    "validate_pipeline",
    "zscore",
]
