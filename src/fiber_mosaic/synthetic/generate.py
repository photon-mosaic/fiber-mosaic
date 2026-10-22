"""Signal model and generators for synthetic fiber photometry recordings.

The model is deliberately not pure noise: a photobleaching trend and a motion
artifact are *shared* between the calcium-dependent band and the isosbestic
reference, while calcium transients appear only in the calcium-dependent band.
Correction steps are supposed to exploit exactly that structure, so
:func:`simulate_bands` also returns the ground-truth calcium component to check
recovery against.

The numbers here are placeholders chosen to be physically plausible rather
than measured: fluorescence sits on a positive baseline (so that dividing
correction methods stay well behaved) and the artifacts dominate the raw
traces.
"""

from __future__ import annotations

import numpy as np

from fiber_mosaic.core.base import (
    FiberPhotometryRecordingGroup,
    recording_from_traces,
)

#: Default acquisition rate in Hz, typical for fiber photometry.
DEFAULT_SAMPLING_FREQUENCY = 30.0

#: Default resting fluorescence level; positive so "divide" methods are safe.
DEFAULT_BASELINE = 100.0

#: Peak amplitude of the photobleaching trend, in baseline units.
_BLEACH_AMPLITUDE = 30.0

#: Amplitude of the shared motion artifact, in baseline units.
_MOTION_SCALE = 5.0

#: Gains with which the reference band sees the shared bleach and motion.
_REFERENCE_BLEACH_GAIN = 0.8
_REFERENCE_MOTION_GAIN = 0.9

#: Standard deviation of the per-sample measurement noise.
_NOISE_SCALE = 0.5


def smooth_drift(
    num_samples: int,
    rng: np.random.Generator,
    scale: float,
    window: int = 50,
) -> np.ndarray:
    """Return a slow, smooth drift built by smoothing white noise.

    Stands in for a movement or hemodynamic artifact: low frequency, zero
    mean, and normalized so ``scale`` sets its standard deviation.

    Parameters
    ----------
    num_samples : int
        Length of the returned trace.
    rng : np.random.Generator
        Random source.
    scale : float
        Target standard deviation of the drift.
    window : int, default: 50
        Boxcar width used to smooth the white noise; larger is slower.

    Returns
    -------
    np.ndarray
        Drift trace, shape ``(num_samples,)``.
    """
    kernel = np.ones(window) / window
    raw = rng.standard_normal(num_samples + window)
    smoothed = np.convolve(raw, kernel, mode="same")[:num_samples]
    return scale * smoothed / (np.std(smoothed) or 1.0)


def calcium_transients(
    times: np.ndarray,
    rng: np.random.Generator,
    num_events: int = 12,
    decay: float = 0.4,
    amplitude: float = 15.0,
) -> np.ndarray:
    """Return sparse, exponentially decaying calcium-like events.

    Each event rises instantaneously at a random sample and decays with time
    constant ``decay``; overlapping events sum.

    Parameters
    ----------
    times : np.ndarray
        Sample times in seconds, shape ``(num_samples,)``.
    rng : np.random.Generator
        Random source.
    num_events : int, default: 12
        Number of transients to place.
    decay : float, default: 0.4
        Exponential decay constant in seconds.
    amplitude : float, default: 15.0
        Nominal peak amplitude; jittered by a factor in ``[0.5, 1.5]``.

    Returns
    -------
    np.ndarray
        Transient trace, shape ``(num_samples,)``.
    """
    trace = np.zeros_like(times)
    onsets = rng.choice(times.size, size=num_events, replace=False)
    for onset in onsets:
        scale = amplitude * rng.uniform(0.5, 1.5)
        trace[onset:] += scale * np.exp(
            -(times[onset:] - times[onset]) / decay
        )
    return trace


def simulate_bands(
    num_samples: int = 3000,
    num_fibers: int = 3,
    sampling_frequency: float = DEFAULT_SAMPLING_FREQUENCY,
    baseline: float = DEFAULT_BASELINE,
    seed: int | None = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate a calcium-dependent band and its isosbestic reference.

    Both bands share a photobleaching trend and a motion artifact (the
    reference sees them at reduced gain); only the calcium-dependent band
    carries transients.

    Parameters
    ----------
    num_samples : int, default: 3000
        Samples per fiber.
    num_fibers : int, default: 3
        Number of fibers (channels).
    sampling_frequency : float, default: DEFAULT_SAMPLING_FREQUENCY
        Acquisition rate in Hz, used to build the time base.
    baseline : float, default: DEFAULT_BASELINE
        Resting fluorescence level added to both bands.
    seed : int or None, default: 0
        Seed for reproducibility; None draws fresh randomness.

    Returns
    -------
    signal : np.ndarray
        Calcium-dependent band, shape ``(num_samples, num_fibers)``.
    reference : np.ndarray
        Isosbestic reference band, same shape.
    calcium : np.ndarray
        Ground-truth calcium component of ``signal``, same shape. A real
        recording never provides this; it is here to validate correction.
    """
    rng = np.random.default_rng(seed)
    times = np.arange(num_samples) / sampling_frequency

    signal = np.empty((num_samples, num_fibers))
    reference = np.empty((num_samples, num_fibers))
    calcium = np.empty((num_samples, num_fibers))

    for fiber in range(num_fibers):
        # shared by both bands: bleaching decay plus a motion artifact
        bleach = _BLEACH_AMPLITUDE * np.exp(
            -times / times[-1] * rng.uniform(1.5, 3.0)
        )
        motion = smooth_drift(num_samples, rng, scale=_MOTION_SCALE)
        # the biology we want to recover, present in the signal band only
        calcium[:, fiber] = calcium_transients(times, rng)

        signal[:, fiber] = (
            baseline
            + bleach
            + motion
            + calcium[:, fiber]
            + rng.normal(0.0, _NOISE_SCALE, num_samples)
        )
        reference[:, fiber] = (
            baseline
            + _REFERENCE_BLEACH_GAIN * bleach
            + _REFERENCE_MOTION_GAIN * motion
            + rng.normal(0.0, _NOISE_SCALE, num_samples)
        )

    return signal, reference, calcium


def simulate_group(
    signal_color: str = "green",
    reference_color: str = "iso",
    sampling_frequency: float = DEFAULT_SAMPLING_FREQUENCY,
    **kwargs,
) -> tuple[FiberPhotometryRecordingGroup, np.ndarray]:
    """Simulate a two-band group plus the ground-truth calcium.

    Convenience wrapper around :func:`simulate_bands` and
    :func:`~fiber_mosaic.core.base.recording_from_traces`.

    Parameters
    ----------
    signal_color : str, default: "green"
        Band name for the calcium-dependent recording.
    reference_color : str, default: "iso"
        Band name for the isosbestic reference recording.
    sampling_frequency : float, default: DEFAULT_SAMPLING_FREQUENCY
        Acquisition rate in Hz. Passed to both `simulate_bands` and
        `recording_from_traces`, so the two never disagree.
    **kwargs
        Forwarded to :func:`simulate_bands` (``num_samples``, ``num_fibers``,
        ``baseline``, ``seed``).

    Returns
    -------
    group : FiberPhotometryRecordingGroup
        Group holding the signal and reference bands.
    calcium : np.ndarray
        Ground-truth calcium component of the signal band.
    """
    signal, reference, calcium = simulate_bands(
        sampling_frequency=sampling_frequency, **kwargs
    )
    group = FiberPhotometryRecordingGroup(
        {
            signal_color: recording_from_traces(
                signal, signal_color, sampling_frequency
            ),
            reference_color: recording_from_traces(
                reference, reference_color, sampling_frequency
            ),
        }
    )
    return group, calcium


__all__ = [
    "DEFAULT_BASELINE",
    "DEFAULT_SAMPLING_FREQUENCY",
    "calcium_transients",
    "simulate_bands",
    "simulate_group",
    "smooth_drift",
]
