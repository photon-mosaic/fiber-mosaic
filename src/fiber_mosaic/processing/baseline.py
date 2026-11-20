"""Baselines: estimate F0, then subtract it (dF) or subtract and divide (dF/F).

A baseline is itself a recording, F0 with the signal's shape. An estimator
returns one, :func:`multiexp_baseline` so far, and
:func:`baseline_correction` combines signal and baseline lazily, so a new
estimator only has to produce F0 and every mode works with it. :func:`dff` is
the one-call shortcut for the multi-exponential baseline with ``mode="dff"``;
unlike the other two it takes a single recording, so it works as a pipeline
step.

    >>> from fiber_mosaic.processing import (
    ...     baseline_correction, multiexp_baseline,
    ... )
    >>> baseline = multiexp_baseline(group)
    >>> dff = baseline_correction(group, baseline)  # (F - F0) / F0
    >>> df = baseline_correction(group, baseline, mode="df")  # F - F0

The multi-exponential baseline is a sum of exponentials -- photobleaching,
optionally with a brightening term -- fitted to the whole trace with a Tukey
biweight loss, so calcium transients do not pull it upward. The fit uses
``aind-ophys-utils``, an optional dependency imported when a fit runs;
evaluating a fitted baseline needs only numpy.
"""

from __future__ import annotations

import warnings

import numpy as np
from spikeinterface.core import BaseRecording
from spikeinterface.preprocessing.basepreprocessor import (
    BasePreprocessorSegment,
)

from fiber_mosaic.core.base import FiberPhotometryRecordingGroup
from fiber_mosaic.core.core_tools import (
    define_function_handling_group_from_class,
)
from fiber_mosaic.processing.baseprocessor import (
    BaseFiberPhotometryPreprocessor,
)

# Applying a baseline

_MODES = {
    "df": lambda signal, baseline: signal - baseline,
    "dff": lambda signal, baseline: (signal - baseline) / baseline,
}


def _layout(recording: BaseRecording) -> tuple[list, list]:
    """Return the fiber ids and the samples per segment."""
    return list(recording.channel_ids), [
        recording.get_num_samples(index)
        for index in range(recording.get_num_segments())
    ]


class BaselineCorrectionSegment(BasePreprocessorSegment):
    """Segment-level baseline correction."""

    def __init__(
        self, recording_segment, baseline_segment, mode: str, dtype
    ) -> None:
        BasePreprocessorSegment.__init__(self, recording_segment)
        self.baseline_segment = baseline_segment
        self.mode = mode
        self.dtype = dtype

    def get_traces(self, start_frame, end_frame, channel_indices):
        """Return the signal combined with its baseline."""
        signal = self.parent_recording_segment.get_traces(
            start_frame, end_frame, channel_indices
        ).astype("float64")
        baseline = self.baseline_segment.get_traces(
            start_frame, end_frame, channel_indices
        ).astype("float64")
        return _MODES[self.mode](signal, baseline).astype(self.dtype)


class BaselineCorrectionRecording(BaseFiberPhotometryPreprocessor):
    """A signal with its baseline subtracted, or subtracted and divided out.

    Parameters
    ----------
    recording : BaseRecording
        The signal.
    baseline_recording : BaseRecording
        Its baseline F0, same fibers and frames.
    mode : {"dff", "df"}, default: "dff"
        ``"dff"`` gives ``(F - F0) / F0``, ``"df"`` gives ``F - F0``.
    """

    def __init__(
        self,
        recording: BaseRecording,
        baseline_recording: BaseRecording,
        mode: str = "dff",
    ) -> None:
        if mode not in _MODES:
            raise ValueError(
                f"mode must be one of {list(_MODES)}, got {mode!r}"
            )
        if _layout(recording) != _layout(baseline_recording):
            raise ValueError(
                "signal and baseline differ in fibers, segments or length"
            )
        BaseFiberPhotometryPreprocessor.__init__(
            self, recording, dtype="float32"
        )
        for segment, baseline_segment in zip(
            recording.segments, baseline_recording.segments, strict=True
        ):
            self.add_recording_segment(
                BaselineCorrectionSegment(
                    segment, baseline_segment, mode, self._dtype
                )
            )

        self._kwargs = dict(
            recording=recording,
            baseline_recording=baseline_recording,
            mode=mode,
        )


def baseline_correction(
    recording, baseline_recording, mode: str = "dff"
) -> BaselineCorrectionRecording | dict | FiberPhotometryRecordingGroup:
    """
    Subtract a baseline from a signal, and optionally divide it out.

    Takes a recording and its baseline, or two dicts or two
    FiberPhotometryRecordingGroups with the same names, which are paired by
    name; the result has the same container.

    Parameters
    ----------
    recording : BaseRecording, dict or FiberPhotometryRecordingGroup
        The signal.
    baseline_recording : same container as ``recording``
        Its baseline F0, e.g. ``multiexp_baseline(recording)``.
    mode : {"dff", "df"}, default: "dff"
        ``"dff"`` gives ``(F - F0) / F0``, ``"df"`` gives ``F - F0``.

    Returns
    -------
    BaselineCorrectionRecording, dict or FiberPhotometryRecordingGroup
        The corrected signal.
    """
    if isinstance(recording, BaseRecording):
        return BaselineCorrectionRecording(recording, baseline_recording, mode)
    if not isinstance(recording, FiberPhotometryRecordingGroup | dict):
        raise TypeError(
            "baseline_correction() accepts a recording, a dict of recordings "
            "or a FiberPhotometryRecordingGroup"
        )
    if set(recording) != set(baseline_recording):
        raise ValueError(
            f"signal names {sorted(recording)} and baseline names "
            f"{sorted(baseline_recording)} differ"
        )
    corrected = {
        name: BaselineCorrectionRecording(
            signal, baseline_recording[name], mode
        )
        for name, signal in recording.items()
    }
    if isinstance(recording, dict):
        return corrected
    return FiberPhotometryRecordingGroup(corrected)


# Multi-exponential baseline

#: Upper bound on the slow bleach tau and the brightening tau [s]; keeps
#: ``exp(-t / tau)`` finite without constraining real (1e4-1e5 s) terms.
_TAU_MAX = 1e7

#: Tukey cutoff, in noise sigmas.
_C = 4.0

#: Geometric annealing steps of the IRLS noise scale.
_SIGMA_ANNEAL_STEPS = 6

#: Early window [s] over which a 3rd bleach exponential is compared.
_T_EVAL_EXP3 = 120.0


def _init_sum_of_exps(trace, n_exp, include_brightening):
    """Return an initial guess and bounds for a ``sum_of_exps`` fit.

    ``b_inf`` starts at the 10th percentile of the last 1000 frames, bounded
    below by a tenth of their mean. Bleach amplitudes are >= 0, brightening
    has a negative amplitude. Amplitudes are capped at the larger of 10x the
    start-to-asymptote difference and half of ``b_inf``, so that nearly
    cancelling slow terms can still form a mid-session hump.
    """
    b_inf = float(np.percentile(trace[-1000:], 10))
    b_inf_lo = float(trace[-1000:].mean() / 10)
    amp = float(trace[:500].mean() - b_inf)
    amp_cap = max(10.0 * max(abs(amp), 1.0), 0.5 * b_inf)

    amp_fracs = {2: [0.7, 0.3], 3: [0.65, 0.30, 0.05]}[n_exp]
    tau_inits = {2: [3600.0, 600.0], 3: [3600.0, 600.0, 30.0]}[n_exp]
    tau_bounds = {
        2: [(300, _TAU_MAX), (1, 5000)],
        3: [(300, _TAU_MAX), (1, 5000), (1, 180)],
    }[n_exp]

    x0 = [b_inf]
    bounds = [(b_inf_lo, np.inf)]
    for frac, tau, tau_bound in zip(
        amp_fracs, tau_inits, tau_bounds, strict=True
    ):
        # a net-brightening trace has negative `amp`; clamp to the >= 0 bound
        x0 += [max(amp * frac, 0.0), tau]
        bounds += [(0, amp_cap), tau_bound]

    if include_brightening:
        x0 += [min(-0.05 * b_inf, 0.0), 2000.0]
        bounds += [(-amp_cap, 0), (60, _TAU_MAX)]

    return np.array(x0), tuple(bounds)


def _sort_bleach_params(params, n_exp):
    """Sort the bleach (amplitude, tau) pairs by tau, descending."""
    pairs = params[1 : 1 + 2 * n_exp].reshape(n_exp, 2)
    sorted_params = params.copy()
    sorted_params[1 : 1 + 2 * n_exp] = pairs[np.argsort(-pairs[:, 1])].ravel()
    return sorted_params


def fit_baseline(
    trace: np.ndarray,
    times: np.ndarray,
    rss_thresh: tuple[float, float] = (0.98, 0.97),
    maxiter: int = 8,
    ds: int = 10,
    noise_method: str = "mad",
) -> np.ndarray:
    """Fit a sum-of-exponentials baseline to one trace.

    Model selection cold-starts at 2 bleach exponentials on a ``ds``-times
    decimated trace, tries adding a brightening term and then a 3rd bleach
    exponential (each kept only if it lowers the RSS by ``rss_thresh``), and
    refits the winner on the full trace with robust IRLS, warm-started from
    the decimated fit. Non-finite samples are left out of the fit.

    Parameters
    ----------
    trace : np.ndarray
        Fluorescence of one fiber.
    times : np.ndarray
        Sample times in seconds, starting near 0 (the amplitudes are anchored
        at ``t = 0``).
    rss_thresh : tuple of float, default: (0.98, 0.97)
        RSS ratios a brightening term and a 3rd exponential must beat.
    maxiter : int, default: 8
        Maximum IRLS iterations of the final fit.
    ds : int, default: 10
        Decimation factor of the model-selection fits.
    noise_method : {"mad", "welch"}, default: "mad"
        Noise estimate fixing the Tukey cutoffs. Welch can sit far below MAD
        on dense-transient traces, putting the cutoffs inside the noise.

    Returns
    -------
    np.ndarray
        Parameters, length 9: ``[b_inf, b1, tau1, b2, tau2, b3, tau3,
        b_bright, tau_bright]``, unused terms (amplitude 0, tau inf).

    Raises
    ------
    ImportError
        If ``aind-ophys-utils`` is not installed.
    ValueError
        If no sample is finite.
    """
    try:
        from aind_ophys_utils.baseline_fitting import (
            TukeyBiweight,
            nonlinear_fit,
            sum_of_exps,
        )
        from aind_ophys_utils.signal_utils import noise_std
    except ImportError as error:
        raise ImportError(
            "the multi-exponential baseline is fitted with aind-ophys-utils, "
            "which is not installed (pip install fiber-mosaic[dff])"
        ) from error

    trace = np.asarray(trace, dtype="float64")
    times = np.asarray(times, dtype="float64")
    valid = np.isfinite(trace) & np.isfinite(times)
    if not valid.any():
        raise ValueError("every sample is non-finite -- nothing to fit")
    if not valid.all():
        warnings.warn(
            f"dropping {(~valid).sum()} non-finite sample(s) before fitting",
            stacklevel=2,
        )
        trace, times = trace[valid], times[valid]

    sigma = float(noise_std(trace, method=noise_method))
    trace_ds, times_ds = trace[::ds], times[::ds]
    n_early = int(np.searchsorted(times_ds, times_ds[0] + _T_EVAL_EXP3))

    kw_ds = dict(model=sum_of_exps, M=None)
    kw_full = dict(
        model=sum_of_exps,
        M=TukeyBiweight(c=_C),
        maxiter=maxiter,
        sigma_anneal_steps=_SIGMA_ANNEAL_STEPS,
    )
    if np.isfinite(sigma) and sigma > 0:
        kw_full["fixed_sigma"] = sigma

    def rss(fit, n=None):
        return float(np.sum((trace_ds[:n] - fit[:n]) ** 2))

    # 1: two bleach exponentials, cold start
    x0, bounds = _init_sum_of_exps(trace_ds, 2, include_brightening=False)
    fit, result = nonlinear_fit(
        trace_ds, times_ds, x0=x0, bounds=bounds, **kw_ds
    )
    best_fit, best_rss = fit, rss(fit)
    params = _sort_bleach_params(result.x, 2)
    n_exp, bright = 2, False

    # 2: add brightening, judged on the whole decimated trace
    x0 = np.concatenate([params, [min(-0.05 * params[0], 0.0), 2000.0]])
    _, bounds = _init_sum_of_exps(trace_ds, 2, include_brightening=True)
    fit, result = nonlinear_fit(
        trace_ds, times_ds, x0=x0, bounds=bounds, **kw_ds
    )
    if rss(fit) < rss_thresh[0] * best_rss:
        best_fit, params, bright = fit, _sort_bleach_params(result.x, 2), True

    # 3: add a 3rd bleach exponential, judged on the early window
    new_exp = np.array([max(0.05 * params[0], 0.0), 50.0])
    x0 = np.concatenate(
        [params[:-2], new_exp, params[-2:]] if bright else [params, new_exp]
    )
    _, bounds = _init_sum_of_exps(trace_ds, 3, include_brightening=bright)
    fit, result = nonlinear_fit(
        trace_ds, times_ds, x0=x0, bounds=bounds, **kw_ds
    )
    params_3 = _sort_bleach_params(result.x, 3)
    if (
        rss(fit, n_early) < rss_thresh[1] * rss(best_fit, n_early)
        and params_3[2 + 2 * n_exp] < 180.0
    ):
        params, n_exp = params_3, 3

    # 4: final robust fit on the full trace, warm-started
    _, bounds = _init_sum_of_exps(trace, n_exp, include_brightening=bright)
    _, result = nonlinear_fit(
        trace, times, x0=params, bounds=bounds, **kw_full
    )
    params = _sort_bleach_params(result.x, n_exp)

    padded = np.array(
        [0.0, 0.0, np.inf, 0.0, np.inf, 0.0, np.inf, 0.0, np.inf]
    )
    padded[0] = params[0]
    padded[1 : 1 + 2 * n_exp] = params[1 : 1 + 2 * n_exp]
    if bright:
        padded[7:] = params[-2:]
    return padded


def _encode_params(params: np.ndarray) -> list[list[float | None]]:
    """Return ``params`` as nested lists with the inf padding as ``None``.

    The padded taus are inf, which JSON can only hold as the non-standard
    ``Infinity``; ``None`` saves as a standard ``null``.
    """
    return [
        [None if value == np.inf else value for value in row]
        for row in params.tolist()
    ]


def _decode_params(params) -> np.ndarray:
    """Return the ``(n_fibers, 9)`` array for :func:`_encode_params`' output.

    ``None`` becomes inf again; arrays pass through unchanged.
    """
    return np.array(
        [
            [np.inf if value is None else value for value in row]
            for row in params
        ],
        dtype="float64",
    )


def _evaluate_baseline(params: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Evaluate fitted baselines, shape ``(len(t), n_fibers)``.

    ``params`` is ``(n_fibers, 9)`` in the layout of :func:`fit_baseline`;
    padded terms have amplitude 0 and tau inf, so they add nothing.
    """
    amplitudes, taus = params[:, 1::2], params[:, 2::2]
    decays = np.exp(-t[:, None, None] / taus)
    return params[:, 0] + (amplitudes * decays).sum(axis=-1)


class MultiExpBaselineSegment(BasePreprocessorSegment):
    """Segment-level multi-exponential baseline."""

    def __init__(
        self,
        recording_segment,
        baseline_params: np.ndarray,
        t_origin: float,
        dtype,
    ) -> None:
        BasePreprocessorSegment.__init__(self, recording_segment)
        self.baseline_params = baseline_params
        self.t_origin = t_origin
        self.dtype = dtype

    def get_traces(self, start_frame, end_frame, channel_indices):
        """Return F0 for the requested frames and fibers."""
        times = self.parent_recording_segment.sample_index_to_time(
            np.arange(start_frame, end_frame)
        )
        baseline = _evaluate_baseline(
            self.baseline_params[channel_indices], times - self.t_origin
        )
        return baseline.astype(self.dtype)


class FiberMultiExpBaselineRecording(BaseFiberPhotometryPreprocessor):
    """Robust multi-exponential baseline F0, fitted per fiber.

    Each fiber of each segment is fitted once, in ``__init__``, with
    :func:`fit_baseline`; the fitted parameters live in ``_kwargs`` (so a
    saved recording reloads without refitting) and ``get_traces`` returns F0.
    Fiber times are read from the parent, so real timestamps enter the fit.

    Parameters
    ----------
    recording : BaseRecording
        Raw fluorescence of one color.
    baseline_params : list of array-like, optional
        Fitted parameters per segment, ``(n_fibers, 9)``; fitted when None.
        Saved with ``None`` for the inf padding, so the JSON is standard.
    t_origins : list of float, optional
        Per-segment time at which the fit's ``t = 0`` sits; the first sample
        time when None.
    rss_thresh, maxiter, ds, noise_method
        See :func:`fit_baseline`.
    """

    _precomputable_kwarg_names = ["baseline_params", "t_origins"]

    def __init__(
        self,
        recording: BaseRecording,
        baseline_params=None,
        t_origins=None,
        rss_thresh: tuple[float, float] = (0.98, 0.97),
        maxiter: int = 8,
        ds: int = 10,
        noise_method: str = "mad",
    ) -> None:
        BaseFiberPhotometryPreprocessor.__init__(
            self, recording, dtype="float32"
        )

        if t_origins is None:
            t_origins = [
                float(recording.get_times(segment_index=index)[0])
                for index in range(recording.get_num_segments())
            ]
        if baseline_params is None:
            baseline_params = []
            for segment_index, t_origin in enumerate(t_origins):
                traces = recording.get_traces(segment_index=segment_index)
                times = recording.get_times(segment_index=segment_index)
                baseline_params.append(
                    np.array(
                        [
                            fit_baseline(
                                trace,
                                times - t_origin,
                                rss_thresh=rss_thresh,
                                maxiter=maxiter,
                                ds=ds,
                                noise_method=noise_method,
                            )
                            for trace in traces.T
                        ]
                    )
                )
        baseline_params = [_decode_params(p) for p in baseline_params]

        for segment, params, t_origin in zip(
            recording.segments, baseline_params, t_origins, strict=True
        ):
            self.add_recording_segment(
                MultiExpBaselineSegment(segment, params, t_origin, self._dtype)
            )

        self._kwargs = dict(
            recording=recording,
            baseline_params=[_encode_params(p) for p in baseline_params],
            t_origins=list(t_origins),
            rss_thresh=list(rss_thresh),
            maxiter=maxiter,
            ds=ds,
            noise_method=noise_method,
        )


multiexp_baseline = define_function_handling_group_from_class(
    FiberMultiExpBaselineRecording, "multiexp_baseline"
)


# dF/F shortcut


class FiberDffRecording(BaselineCorrectionRecording):
    """dF/F against a robust multi-exponential baseline, fitted per fiber.

    Parameters
    ----------
    recording : BaseRecording
        Raw fluorescence of one color.
    baseline_recording : BaseRecording, optional
        A fitted baseline to reuse, e.g. when reloading; fitted when None.
    **fit_kwargs
        Options of the fit, see
        :class:`FiberMultiExpBaselineRecording`.
    """

    def __init__(
        self,
        recording: BaseRecording,
        baseline_recording: BaseRecording | None = None,
        **fit_kwargs,
    ) -> None:
        if baseline_recording is None:
            baseline_recording = FiberMultiExpBaselineRecording(
                recording, **fit_kwargs
            )
        BaselineCorrectionRecording.__init__(
            self, recording, baseline_recording, mode="dff"
        )
        self._kwargs = dict(
            recording=recording, baseline_recording=baseline_recording
        )


dff = define_function_handling_group_from_class(FiberDffRecording, "dff")
