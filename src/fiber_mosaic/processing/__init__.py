"""Fiber-photometry processing layer: steps, registry, and pipeline.

The counterpart to :mod:`spikeinterface.preprocessing`, sitting beside
:mod:`fiber_mosaic.core` (the recording layer). Importing this package
registers the built-in steps defined in :mod:`.functions`.
"""

from fiber_mosaic.processing.functions import (
    apply_correction,
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
from fiber_mosaic.processing.pipeline import apply_pipeline, validate_pipeline
from fiber_mosaic.processing.registry import (
    ArrayStep,
    RecordingStep,
    concatenate_segments,
    get_step,
    processing_step,
    register_step,
    registered_steps,
    validate_timeline,
)

__all__ = [
    "ArrayStep",
    "RecordingStep",
    "apply_correction",
    "apply_pipeline",
    "bandpass_filter",
    "bleach_correction",
    "center",
    "common_reference",
    "concatenate_segments",
    "decimate",
    "gaussian_filter",
    "get_step",
    "highpass_filter",
    "isosbestic_correction",
    "normalize_by_quantile",
    "notch_filter",
    "processing_step",
    "register_step",
    "registered_steps",
    "resample",
    "scale",
    "validate_pipeline",
    "validate_timeline",
    "zscore",
]
