"""Motion correction of dF/F traces with a reference band.

Motion and other calcium-independent artifacts show up in a reference band that
does not carry the sensor signal: usually the isosbestic band, but also a
co-expressed activity-independent fluorophore (e.g. tdTomato or mCherry). It is
a noisy proxy for motion. Per fiber, the signal's dF/F is regressed on the
reference's dF/F with a robust M-estimator, and the scaled reference is
subtracted. The regression runs on 0.05 Hz low-passed traces: noise in the
regressor biases the slope towards zero, increasingly so at higher cutoffs. The
subtracted reference is not filtered, so faster motion is corrected too, at the
cost of the reference's noise. The coefficient is clipped to be non-negative,
since motion changes all bands in the same direction. The intercept is not
subtracted: it cannot be told apart from a genuine tonic signal.

Signal and reference are paired frame by frame, so they need the same fibers,
sampling frequency and length, and are assumed to be uniformly sampled. Bands
acquired one after the other are offset by a fraction of a frame: the
reference is then linearly interpolated to the signal's timestamps. Offsets of
a frame or more are an error.

Inputs are dF/F, e.g. the output of :func:`fiber_mosaic.processing.dff`. The
fit uses ``aind-ophys-utils`` and ``statsmodels``, optional dependencies
imported when a fit runs.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.signal import butter, sosfiltfilt
from spikeinterface.core import BaseRecording
from spikeinterface.preprocessing.basepreprocessor import (
    BasePreprocessorSegment,
)

from fiber_mosaic.core.base import FiberPhotometryRecordingGroup
from fiber_mosaic.processing.baseprocessor import (
    BaseFiberPhotometryPreprocessor,
)

#: Tukey cutoffs, in noise sigmas, for positive / negative residuals.
_C_POS, _C_NEG = 3.0, 4.0
#: Fewest samples the zero-phase low-pass accepts (it pads by 9).
_MIN_SAMPLES = 10


def fit_coefficient(
    signal: np.ndarray,
    reference: np.ndarray,
    fs: float,
    cutoff: float = 0.05,
    c_pos: float = _C_POS,
    c_neg: float = _C_NEG,
) -> float:
    """Return the coefficient scaling the reference to the signal's motion.

    Both traces are low-passed at ``cutoff``, which keeps the reference's noise
    from biasing the slope towards zero, and regressed against each other with
    a robust asymmetric Tukey fit. Non-finite samples are interpolated for the
    filter and left out of the regression.

    Parameters
    ----------
    signal, reference : np.ndarray
        dF/F of one fiber, same length.
    fs : float
        Sampling frequency in Hz.
    cutoff : float, default: 0.05
        Low-pass cutoff in Hz of the traces entering the regression; the
        higher, the more the noise biases the coefficient low.
    c_pos, c_neg : float
        Tukey cutoffs, in noise sigmas, for positive / negative residuals.

    Returns
    -------
    float
        The slope, at least 0; 0 with a warning if fewer than 10 samples are
        finite in both traces or the reference is constant, and NaN with a
        warning if the reference has no finite sample.

    Raises
    ------
    ImportError
        If ``aind-ophys-utils`` is not installed.
    ValueError
        If the shapes differ.
    """
    try:
        from aind_ophys_utils.baseline_fitting import AsymmetricTukeyBiweight
        from statsmodels.api import RLM, add_constant
    except ImportError as error:
        raise ImportError(
            "motion_correction() fits its coefficient with "
            "aind-ophys-utils, which is not installed "
            "(pip install fiber-mosaic[dff])"
        ) from error

    signal = np.asarray(signal, dtype="float64")
    reference = np.asarray(reference, dtype="float64")
    if signal.shape != reference.shape:
        raise ValueError(
            f"signal {signal.shape} and reference {reference.shape} differ "
            "in shape"
        )
    if not np.isfinite(reference).any():
        warnings.warn(
            "the reference has no finite sample, coefficient set to NaN",
            stacklevel=2,
        )
        return np.nan
    valid = np.isfinite(signal) & np.isfinite(reference)
    if valid.sum() < _MIN_SAMPLES or np.ptp(reference[valid]) == 0:
        warnings.warn(
            f"nothing to fit (need {_MIN_SAMPLES} samples finite in both "
            "traces and a reference that is not constant), coefficient set "
            "to 0",
            stacklevel=2,
        )
        return 0.0
    if not valid.all():
        warnings.warn(
            f"leaving {(~valid).sum()} non-finite sample(s) out of the fit",
            stacklevel=2,
        )

    sos = butter(2, cutoff, fs=fs, output="sos")
    x = np.arange(len(signal))
    low_passed = [
        sosfiltfilt(sos, np.interp(x, x[valid], trace[valid]))[valid]
        for trace in (signal, reference)
    ]
    fit = RLM(
        low_passed[0],
        add_constant(low_passed[1]),
        M=AsymmetricTukeyBiweight(c_pos=c_pos, c_neg=c_neg),
    ).fit()
    return float(max(fit.params[1], 0.0))


def _check_paired(recording, reference_recording) -> None:
    """Raise unless the reference matches the signal fiber by fiber."""
    if list(recording.channel_ids) != list(reference_recording.channel_ids):
        raise ValueError(
            "signal and reference need the same fibers in the same order, "
            f"got {list(recording.channel_ids)} and "
            f"{list(reference_recording.channel_ids)}"
        )
    if not np.isclose(  # gross mismatch only; _offsets checks the timestamps
        recording.get_sampling_frequency(),
        reference_recording.get_sampling_frequency(),
        rtol=1e-3,
        atol=0,
    ):
        raise ValueError("signal and reference differ in sampling frequency")
    lengths = [
        [r.get_num_samples(i) for i in range(r.get_num_segments())]
        for r in (recording, reference_recording)
    ]
    if lengths[0] != lengths[1]:
        raise ValueError(
            "signal and reference differ in segments or number of samples, "
            f"got {lengths[0]} and {lengths[1]}"
        )


def _offsets(recording, reference_recording) -> list:
    """Return, per segment, ``None`` or the times of signal and reference.

    ``None`` if the reference is sampled at the signal's times, else the two
    time vectors, e.g. for bands acquired one after the other. Raises if they
    are a frame or more apart: that is a different timebase, not an offset.
    """
    fs = recording.get_sampling_frequency()
    offsets = []
    for index in range(recording.get_num_segments()):
        times = recording.get_times(segment_index=index)
        reference_times = reference_recording.get_times(segment_index=index)
        shift = np.abs(reference_times - times).max(initial=0.0)
        if shift >= 1 / fs:
            raise ValueError(
                f"signal and reference are {shift:.3g} s apart in segment "
                f"{index}, a frame or more: they need the same time base"
            )
        simultaneous = shift <= 1e-6 / fs
        offsets.append(None if simultaneous else (times, reference_times))
    return offsets


def _interpolate(traces, from_times, to_times) -> np.ndarray:
    """Linearly interpolate each fiber's trace onto ``to_times``."""
    return np.stack(
        [np.interp(to_times, from_times, trace) for trace in traces.T], axis=1
    )


class MotionCorrectionSegment(BasePreprocessorSegment):
    """Segment-level motion correction.

    ``times``, if given, is the signal's and the reference's time vector: the
    reference is interpolated to the signal's times.
    """

    def __init__(
        self,
        recording_segment,
        reference_recording_segment,
        coefficients: np.ndarray,
        reference_means: np.ndarray,
        dtype,
        times=None,
    ) -> None:
        BasePreprocessorSegment.__init__(self, recording_segment)
        self.reference_recording_segment = reference_recording_segment
        self.times = times
        self.coefficients = coefficients
        self.reference_means = reference_means
        self.dtype = dtype

    def get_traces(self, start_frame, end_frame, channel_indices):
        """Return the signal minus its scaled, mean-centered reference."""
        signal = self.parent_recording_segment.get_traces(
            start_frame, end_frame, channel_indices
        )
        if self.times is None:
            reference = self.reference_recording_segment.get_traces(
                start_frame, end_frame, channel_indices
            )
        else:
            times, reference_times = self.times
            start = 0 if start_frame is None else start_frame
            end = len(times) if end_frame is None else end_frame
            low, high = max(start - 1, 0), min(end + 1, len(times))
            reference = _interpolate(
                self.reference_recording_segment.get_traces(
                    low, high, channel_indices
                ),
                reference_times[low:high],
                times[start:end],
            )
        coefficients = self.coefficients[channel_indices]
        motion = coefficients * (
            reference - self.reference_means[channel_indices]
        )
        motion = np.where(coefficients == 0, 0.0, motion)  # 0 * NaN is NaN
        return (signal - motion).astype(self.dtype)


class MotionCorrectionRecording(BaseFiberPhotometryPreprocessor):
    """dF/F with the reference's motion component subtracted.

    Each fiber of each segment is fitted once, in ``__init__``, with
    :func:`fit_coefficient`; the coefficients and reference means live in
    ``_kwargs``, so a saved recording reloads without refitting.

    Parameters
    ----------
    recording : BaseRecording
        dF/F of the signal, e.g. ``dff(group)["green"]``.
    reference_recording : BaseRecording
        dF/F of the reference band, same fibers and frames.
    coefficients, reference_means : list of array-like, optional
        Per segment and fiber; fitted when None.
    cutoff, c_pos, c_neg
        See :func:`fit_coefficient`.
    """

    _precomputable_kwarg_names = ["coefficients", "reference_means"]

    def __init__(
        self,
        recording: BaseRecording,
        reference_recording: BaseRecording,
        coefficients=None,
        reference_means=None,
        cutoff: float = 0.05,
        c_pos: float = _C_POS,
        c_neg: float = _C_NEG,
    ) -> None:
        BaseFiberPhotometryPreprocessor.__init__(
            self, recording, dtype="float32"
        )
        _check_paired(recording, reference_recording)
        offsets = _offsets(recording, reference_recording)
        if (coefficients is None) != (reference_means is None):
            raise ValueError(
                "pass both coefficients and reference_means, or neither"
            )

        if coefficients is None:
            fs = recording.get_sampling_frequency()
            coefficients, reference_means = [], []
            for index in range(recording.get_num_segments()):
                signal = recording.get_traces(segment_index=index)
                reference = reference_recording.get_traces(segment_index=index)
                if offsets[index] is not None:
                    reference = _interpolate(
                        reference, offsets[index][1], offsets[index][0]
                    )
                coefficients.append(
                    [
                        fit_coefficient(s, r, fs, cutoff, c_pos, c_neg)
                        for s, r in zip(signal.T, reference.T, strict=True)
                    ]
                )
                finite = np.isfinite(reference)
                mean = np.where(finite, reference, 0.0).sum(axis=0)
                reference_means.append(
                    (mean / np.maximum(finite.sum(axis=0), 1)).tolist()
                )
        coefficients = [np.asarray(c, dtype="float64") for c in coefficients]
        reference_means = [
            np.asarray(m, dtype="float64") for m in reference_means
        ]
        shape = (recording.get_num_channels(),)
        if not (
            len(coefficients)
            == len(reference_means)
            == recording.get_num_segments()
            and all(a.shape == shape for a in coefficients + reference_means)
        ):
            raise ValueError(
                "coefficients and reference_means need one value per fiber "
                "for each segment"
            )

        for segment, reference_segment, coef, mean, times in zip(
            recording.segments,
            reference_recording.segments,
            coefficients,
            reference_means,
            offsets,
            strict=True,
        ):
            self.add_recording_segment(
                MotionCorrectionSegment(
                    segment, reference_segment, coef, mean, self._dtype, times
                )
            )

        self._kwargs = dict(
            recording=recording,
            reference_recording=reference_recording,
            # NaN (no usable reference) becomes null: standard JSON
            coefficients=[
                np.where(np.isnan(c), None, c).tolist() for c in coefficients
            ],
            reference_means=[m.tolist() for m in reference_means],
            cutoff=cutoff,
            c_pos=c_pos,
            c_neg=c_neg,
        )


def motion_correction(
    recording,
    reference_recording: BaseRecording | None = None,
    *,
    reference: str = "iso",
    **kwargs,
) -> MotionCorrectionRecording | dict | FiberPhotometryRecordingGroup:
    """
    Subtract a reference band's motion component from a signal.

    Takes the dF/F of a recording and its reference, or a dict or
    FiberPhotometryRecordingGroup of dF/F recordings: then every recording but
    the one named ``reference`` is corrected with that one, and the result has
    the same container, without the reference.

    Parameters
    ----------
    recording : BaseRecording, dict or FiberPhotometryRecordingGroup
        The signal's dF/F, or the dF/F of all colors, e.g. ``dff(group)``.
    reference_recording : BaseRecording, optional
        The reference's dF/F; only for a single recording.
    reference : str, default: "iso"
        Name of the reference in a dict or group.
    **kwargs
        ``cutoff``, ``c_pos``, ``c_neg``; see :func:`fit_coefficient`.

    Returns
    -------
    MotionCorrectionRecording, dict or FiberPhotometryRecordingGroup
        The corrected signal(s).
    """
    if isinstance(recording, BaseRecording):
        if reference_recording is None:
            raise TypeError("a single recording needs its reference_recording")
        return MotionCorrectionRecording(
            recording, reference_recording, **kwargs
        )
    if not isinstance(recording, FiberPhotometryRecordingGroup | dict):
        raise TypeError(
            "motion_correction() accepts a recording, a dict of "
            "recordings or a FiberPhotometryRecordingGroup"
        )
    if reference_recording is not None:
        raise TypeError("a dict or group is corrected with its `reference`")
    if reference not in recording or len(recording) < 2:
        raise ValueError(
            f"need the reference {reference!r} and at least one more "
            f"recording, got {sorted(recording)}"
        )
    corrected = {
        name: MotionCorrectionRecording(signal, recording[reference], **kwargs)
        for name, signal in recording.items()
        if name != reference
    }
    if isinstance(recording, dict):
        return corrected
    return FiberPhotometryRecordingGroup(corrected)
