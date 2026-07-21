"""Public fiber-photometry processing toolbox.

Holds fiber-mosaic's built-in steps (:func:`bleach_correction`,
:func:`isosbestic_correction`) plus a curated re-export of spikeinterface
preprocessing steps, and registers them all as named pipeline steps.

A step is a callable ``func(x, **params) -> y`` where ``x``/``y`` are a
:class:`~fiber_mosaic.core.base.BaseFiberPhotometryExtractor` or a
:class:`~fiber_mosaic.core.base.FiberPhotometryRecordingGroup`. Single-input
steps take a recording; reference-consuming steps (isosbestic) take the whole
group and pick bands by name.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from spikeinterface.core import BaseRecording
from spikeinterface.preprocessing import (
    bandpass_filter,
    center,
    common_reference,
    decimate,
    gaussian_filter,
    highpass_filter,
    normalize_by_quantile,
    notch_filter,
    resample,
    scale,
    zscore,
)

from fiber_mosaic.core.base import (
    BaseFiberPhotometryExtractor,
    FiberPhotometryRecordingGroup,
)
from fiber_mosaic.processing.helpers import map_segments
from fiber_mosaic.processing.registry import pipeline_step, register_step

#: Baseline magnitudes below this are treated as zero when dividing.
_ZERO_BASELINE_EPS = 1e-12

#: spikeinterface steps re-exported and registered under these names.
_SPIKEINTERFACE_STEPS = {
    "bandpass_filter": bandpass_filter,
    "center": center,
    "common_reference": common_reference,
    "decimate": decimate,
    "gaussian_filter": gaussian_filter,
    "highpass_filter": highpass_filter,
    "normalize_by_quantile": normalize_by_quantile,
    "notch_filter": notch_filter,
    "resample": resample,
    "scale": scale,
    "zscore": zscore,
}


def _apply_correction(
    traces: np.ndarray, baseline: np.ndarray, method: str
) -> np.ndarray:
    """Combine traces and a baseline per the named method.

    Parameters
    ----------
    traces : np.ndarray
        Raw values, shape ``(n_samples, n_fibers)``.
    baseline : np.ndarray
        Baseline to remove, same shape as ``traces``.
    method : {"subtract", "divide", "subtract-divide"}
        ``"subtract"`` returns ``traces - baseline``; ``"divide"`` returns
        ``traces / baseline``; ``"subtract-divide"`` returns
        ``(traces - baseline) / baseline`` (a dF/F-like signal).

    Returns
    -------
    np.ndarray
        The corrected traces. Dividing by a (near-)zero baseline yields NaN.

    Raises
    ------
    ValueError
        If ``method`` is not recognized.
    """
    if method == "subtract":
        return traces - baseline
    if method in ("divide", "subtract-divide"):
        numerator = (
            traces - baseline if method == "subtract-divide" else traces
        )
        safe_baseline = np.where(
            np.abs(baseline) < _ZERO_BASELINE_EPS, np.nan, baseline
        )
        return numerator / safe_baseline
    raise ValueError(f"unknown correction method {method!r}")


def _check_reference_compatible(
    signal: BaseRecording, reference: BaseRecording
) -> None:
    """Validate signal and reference share fibers, segments, samples."""
    if signal.get_num_channels() != reference.get_num_channels():
        raise ValueError("signal and reference must have the same fiber count")
    if signal.get_num_segments() != reference.get_num_segments():
        raise ValueError(
            "signal and reference must have the same segment count"
        )
    for index in range(signal.get_num_segments()):
        if signal.get_num_samples(index) != reference.get_num_samples(index):
            raise ValueError(
                f"segment {index}: signal and reference sample counts differ"
            )


def _correct_reference(
    signal_traces: np.ndarray, reference_traces: np.ndarray, method: str
) -> np.ndarray:
    """Fit the reference to the signal per fiber (OLS) and remove it."""
    signal_traces = signal_traces.astype("float64")
    reference_traces = reference_traces.astype("float64")
    fitted = np.empty_like(signal_traces)
    for fiber in range(signal_traces.shape[1]):
        slope, intercept = np.polyfit(
            reference_traces[:, fiber], signal_traces[:, fiber], 1
        )
        fitted[:, fiber] = slope * reference_traces[:, fiber] + intercept
    return _apply_correction(signal_traces, fitted, method)


@pipeline_step("bleach_correction")
def bleach_correction(
    traces: np.ndarray,
    degree: int = 3,
    method: str = "subtract-divide",
) -> np.ndarray:
    """Remove a per-fiber polynomial photobleaching trend.

    Applied per segment (``by_segment=True``): a low-order polynomial in
    normalized sample index is fitted to each fiber and removed according to
    ``method``. Sample index matches wall-clock time for regularly sampled
    data.

    Parameters
    ----------
    traces : np.ndarray
        One segment's fluorescence, shape ``(n_samples, n_fibers)``.
    degree : int, default: 3
        Degree of the polynomial trend fitted per fiber.
    method : {"subtract", "divide", "subtract-divide"}, \
default: "subtract-divide"
        How the fitted trend is removed (see :func:`_apply_correction`).

    Returns
    -------
    np.ndarray
        The corrected segment traces.
    """
    traces = traces.astype("float64")
    index = np.linspace(0.0, 1.0, traces.shape[0])
    trend = np.vander(index, degree + 1) @ np.polyfit(index, traces, degree)
    return _apply_correction(traces, trend, method)


@pipeline_step("isosbestic_correction", by_segment=False)
def isosbestic_correction(
    group: FiberPhotometryRecordingGroup,
    signal: str,
    reference: str,
    method: str = "subtract-divide",
) -> BaseFiberPhotometryExtractor:
    """Correct a signal band using another band as isosbestic reference.

    For each fiber the reference is scaled to the signal by ordinary least
    squares (slope and intercept); the fitted reference is then removed
    according to ``method``. Segment iteration is handled by
    :func:`~fiber_mosaic.processing.helpers.map_segments`.

    Parameters
    ----------
    group : FiberPhotometryRecordingGroup
        Group holding at least the ``signal`` and ``reference`` bands.
    signal : str
        Band name of the calcium-dependent channel to correct.
    reference : str
        Band name of the isosbestic/control channel.
    method : {"subtract", "divide", "subtract-divide"}, \
default: "subtract-divide"
        How the fitted reference is removed.

    Returns
    -------
    BaseFiberPhotometryExtractor
        The corrected signal band.
    """
    signal_recording = group[signal]
    reference_recording = group[reference]
    _check_reference_compatible(signal_recording, reference_recording)
    return map_segments(
        signal_recording,
        lambda signal_traces, reference_traces: _correct_reference(
            signal_traces, reference_traces, method
        ),
        reference_recording,
    )


def apply_to_group(
    group: FiberPhotometryRecordingGroup,
    function: Callable,
    **params,
) -> FiberPhotometryRecordingGroup:
    """Apply a single-input step to every color, returning a new group.

    Parameters
    ----------
    group : FiberPhotometryRecordingGroup
        Group whose per-color recordings are each processed.
    function : Callable
        A step taking one recording (e.g. :func:`bleach_correction` or
        :func:`zscore`). Reference-consuming steps are not supported here.
    **params
        Keyword parameters forwarded to ``function``.

    Returns
    -------
    FiberPhotometryRecordingGroup
        A new group with the same colors and their processed recordings.
    """
    return FiberPhotometryRecordingGroup(
        {
            color: function(recording, **params)
            for color, recording in group.items()
        }
    )


# register the re-exported spikeinterface steps for use in pipelines by name
for _name, _function in _SPIKEINTERFACE_STEPS.items():
    register_step(_name, _function)


__all__ = [
    "apply_to_group",
    "bandpass_filter",
    "bleach_correction",
    "center",
    "common_reference",
    "decimate",
    "gaussian_filter",
    "highpass_filter",
    "isosbestic_correction",
    "normalize_by_quantile",
    "notch_filter",
    "resample",
    "scale",
    "zscore",
]
