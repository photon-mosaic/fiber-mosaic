"""Public fiber-photometry processing toolbox.

Holds fiber-mosaic's built-in steps (:func:`bleach_correction`,
:func:`isosbestic_correction`) plus a curated re-export of spikeinterface
preprocessing steps, and registers them all as named pipeline steps.

Built-in steps are written against the array contract described in
:mod:`fiber_mosaic.processing.registry`: they receive traces, a
``(n_samples, n_fibers)`` times matrix, and any further input bands as arrays,
and return an array of the same length. The decorator handles segments,
padding, and rebuilding recordings.
"""

from __future__ import annotations

import numpy as np
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

from fiber_mosaic.processing.registry import processing_step, register_step

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


def apply_correction(
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


@processing_step("bleach_correction", contiguous=False)
def bleach_correction(
    traces: np.ndarray,
    times: np.ndarray,
    *,
    degree: int = 3,
    method: str = "subtract-divide",
) -> np.ndarray:
    """Remove a per-fiber polynomial photobleaching trend.

    Tolerates temporal gaps (``contiguous=False``): the decorator hands over
    every segment at once with the real timestamps, so one trend is fitted
    across the whole session rather than independently per segment. Each fiber
    is fitted against its own timestamps, which matters when fibers are
    sampled at individual moments.

    Parameters
    ----------
    traces : np.ndarray
        Fluorescence, shape ``(n_samples, n_fibers)``.
    times : np.ndarray
        Sample times, same shape as ``traces``.
    degree : int, default: 3
        Degree of the polynomial trend fitted per fiber.
    method : {"subtract", "divide", "subtract-divide"}, \
default: "subtract-divide"
        How the fitted trend is removed (see :func:`apply_correction`).

    Returns
    -------
    np.ndarray
        The corrected traces.
    """
    traces = traces.astype("float64")
    trend = np.empty_like(traces)
    for fiber in range(traces.shape[1]):
        fiber_times = times[:, fiber]
        span = fiber_times[-1] - fiber_times[0]
        # normalize the fit variable to [0, 1] to keep the Vandermonde
        # system well conditioned for large absolute timestamps
        index = (fiber_times - fiber_times[0]) / (span or 1.0)
        coefficients = np.polyfit(index, traces[:, fiber], degree)
        trend[:, fiber] = np.vander(index, degree + 1) @ coefficients
    return apply_correction(traces, trend, method)


@processing_step("isosbestic_correction", contiguous=False)
def isosbestic_correction(
    signal: np.ndarray,
    times: np.ndarray,
    reference: np.ndarray,
    *,
    method: str = "subtract-divide",
) -> np.ndarray:
    """Correct a signal band using another band as isosbestic reference.

    For each fiber the reference is scaled to the signal by ordinary least
    squares (slope and intercept); the fitted reference is then removed
    according to ``method``. The fit is sample-wise, so it tolerates temporal
    gaps and ``times`` is unused.

    Parameters
    ----------
    signal : np.ndarray
        Calcium-dependent band, shape ``(n_samples, n_fibers)``.
    times : np.ndarray
        Sample times, same shape. Unused; part of the step contract.
    reference : np.ndarray
        Isosbestic/control band, same shape as ``signal``.
    method : {"subtract", "divide", "subtract-divide"}, \
default: "subtract-divide"
        How the fitted reference is removed (see :func:`apply_correction`).

    Returns
    -------
    np.ndarray
        The corrected signal band.
    """
    signal = signal.astype("float64")
    reference = reference.astype("float64")
    fitted = np.empty_like(signal)
    for fiber in range(signal.shape[1]):
        slope, intercept = np.polyfit(reference[:, fiber], signal[:, fiber], 1)
        fitted[:, fiber] = slope * reference[:, fiber] + intercept
    return apply_correction(signal, fitted, method)


# register the re-exported spikeinterface steps for use in pipelines by name
for _name, _function in _SPIKEINTERFACE_STEPS.items():
    register_step(_name, _function)


__all__ = [
    "apply_correction",
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
