"""
Ported SpikeInterface preprocessing steps.

One place to declare which upstream preprocessors fiber-mosaic offers, and
with which defaults. Each entry gives two things:

- a fiber-aware class (``Fiber...Recording``), for subclassing and isinstance
  checks;
- the public function, which dispatches over a recording, a dict of
  recordings, or a FiberPhotometryRecordingGroup.

Each port is a plain class statement, so SpikeInterface can find it again by
``__module__`` and ``__name__`` when it reloads or unpickles a step. One rule
decides whether a step belongs here:

**Its defaults must make sense for photometry.** Upstream defaults are chosen
for electrophysiology at tens of kilohertz. A port replaces them by
redeclaring ``__init__`` with new defaults; callers still override
explicitly, and the new defaults show up in `help()` and `inspect.signature`.
"""

from __future__ import annotations

from spikeinterface.core import BaseRecording
from spikeinterface.preprocessing.preprocessing_classes import (
    BandpassFilterRecording,
    CenterRecording,
    CommonReferenceRecording,
    GaussianFilterRecording,
    HighpassFilterRecording,
    NormalizeByQuantileRecording,
    NotchFilterRecording,
    ScaleRecording,
    ZScoreRecording,
)

from fiber_mosaic.core.base import FiberPhotometryMixin
from fiber_mosaic.core.core_tools import (
    define_function_handling_group_from_class,
)


class FiberBandpassFilterRecording(
    FiberPhotometryMixin, BandpassFilterRecording
):
    """``BandpassFilterRecording`` with photometry defaults.

    Upstream defaults are 300 Hz / 6 kHz, chosen for spike-band ephys and
    meaningless above the Nyquist frequency of a photometry rig. These are
    conventional starting points, not the right answer for any given
    experiment: 0.1 Hz removes slow drift while leaving calcium transient
    decay (~0.5-2 s) intact, and 5 Hz sits above the fastest calcium kinetics
    while staying below Nyquist even for the ~20 Hz systems. Analyses should
    state the band they used.

    SpikeInterface rejects freq_min below an absolute 100 Hz threshold,
    warning about chunking artifacts. The threshold is not relative to the
    sampling rate, so it fires for every useful photometry band; the margin
    that actually prevents those artifacts is sized from freq_min separately
    (margin_ms="auto") and still applies. ``ignore_low_freq_error`` therefore
    defaults to True.
    """

    def __init__(
        self,
        recording: BaseRecording,
        freq_min: float = 0.1,
        freq_max: float = 5.0,
        margin_ms: float | str = "auto",
        dtype=None,
        ignore_low_freq_error: bool = True,
        _skip_margin_warning_for_old_version: bool = False,
        **filter_kwargs,
    ) -> None:
        BandpassFilterRecording.__init__(
            self,
            recording,
            freq_min=freq_min,
            freq_max=freq_max,
            margin_ms=margin_ms,
            dtype=dtype,
            ignore_low_freq_error=ignore_low_freq_error,
            _skip_margin_warning_for_old_version=(
                _skip_margin_warning_for_old_version
            ),
            **filter_kwargs,
        )


class FiberHighpassFilterRecording(
    FiberPhotometryMixin, HighpassFilterRecording
):
    """``HighpassFilterRecording`` with photometry defaults.

    See :class:`FiberBandpassFilterRecording` for why ``freq_min`` defaults
    to 0.1 Hz and ``ignore_low_freq_error`` to True.
    """

    def __init__(
        self,
        recording: BaseRecording,
        freq_min: float = 0.1,
        margin_ms: float | str = "auto",
        dtype=None,
        ignore_low_freq_error: bool = True,
        _skip_margin_warning_for_old_version: bool = False,
        **filter_kwargs,
    ) -> None:
        HighpassFilterRecording.__init__(
            self,
            recording,
            freq_min=freq_min,
            margin_ms=margin_ms,
            dtype=dtype,
            ignore_low_freq_error=ignore_low_freq_error,
            _skip_margin_warning_for_old_version=(
                _skip_margin_warning_for_old_version
            ),
            **filter_kwargs,
        )


class FiberNotchFilterRecording(FiberPhotometryMixin, NotchFilterRecording):
    """``NotchFilterRecording`` with the fiber API."""


class FiberGaussianFilterRecording(
    FiberPhotometryMixin, GaussianFilterRecording
):
    """``GaussianFilterRecording`` with the fiber API."""


class FiberCenterRecording(FiberPhotometryMixin, CenterRecording):
    """``CenterRecording`` with the fiber API."""


class FiberScaleRecording(FiberPhotometryMixin, ScaleRecording):
    """``ScaleRecording`` with the fiber API."""


class FiberZScoreRecording(FiberPhotometryMixin, ZScoreRecording):
    """``ZScoreRecording`` with the fiber API."""


class FiberNormalizeByQuantileRecording(
    FiberPhotometryMixin, NormalizeByQuantileRecording
):
    """``NormalizeByQuantileRecording`` with the fiber API."""


class FiberCommonReferenceRecording(
    FiberPhotometryMixin, CommonReferenceRecording
):
    """``CommonReferenceRecording`` with the fiber API."""


bandpass_filter = define_function_handling_group_from_class(
    FiberBandpassFilterRecording, "bandpass_filter"
)
highpass_filter = define_function_handling_group_from_class(
    FiberHighpassFilterRecording, "highpass_filter"
)
notch_filter = define_function_handling_group_from_class(
    FiberNotchFilterRecording, "notch_filter"
)
gaussian_filter = define_function_handling_group_from_class(
    FiberGaussianFilterRecording, "gaussian_filter"
)
center = define_function_handling_group_from_class(
    FiberCenterRecording, "center"
)
scale = define_function_handling_group_from_class(FiberScaleRecording, "scale")
zscore = define_function_handling_group_from_class(
    FiberZScoreRecording, "zscore"
)
normalize_by_quantile = define_function_handling_group_from_class(
    FiberNormalizeByQuantileRecording, "normalize_by_quantile"
)
common_reference = define_function_handling_group_from_class(
    FiberCommonReferenceRecording, "common_reference"
)

__all__ = [
    "FiberBandpassFilterRecording",
    "FiberCenterRecording",
    "FiberCommonReferenceRecording",
    "FiberGaussianFilterRecording",
    "FiberHighpassFilterRecording",
    "FiberNormalizeByQuantileRecording",
    "FiberNotchFilterRecording",
    "FiberScaleRecording",
    "FiberZScoreRecording",
    "bandpass_filter",
    "center",
    "common_reference",
    "gaussian_filter",
    "highpass_filter",
    "normalize_by_quantile",
    "notch_filter",
    "scale",
    "zscore",
]
