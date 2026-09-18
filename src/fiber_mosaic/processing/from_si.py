"""
Ported SpikeInterface preprocessing steps.

One place to declare which upstream preprocessors fiber-mosaic offers, and
with which defaults. Each entry gives two things:

- a fiber-aware class (``Fiber...Recording``), for subclassing and isinstance
  checks;
- the public function, which dispatches over a recording, a dict of
  recordings, or a FiberPhotometryRecordingGroup.

Porting a step is one call to :func:`port_from_si`. Two rules decide whether
a step belongs here:

**It must preserve the sample count.** Per-fiber timestamps are inherited from
the parent recording, and that inheritance is only valid while samples line up
one-to-one. `resample` and `decimate` are therefore absent: they need
per-step timestamp rules (subsampling for decimate, interpolation for
resample) before they can be offered.

**Its defaults must make sense for photometry.** Upstream defaults are chosen
for electrophysiology at tens of kilohertz. ``default_kwargs`` replaces them
without touching upstream; callers still override explicitly, and the new
defaults show up in `help()` and `inspect.signature`.
"""

from __future__ import annotations

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

from fiber_mosaic.core.core_tools import (
    define_function_handling_group_from_class,
)
from fiber_mosaic.processing.baseprocessor import as_fiber_preprocessor

# SpikeInterface rejects freq_min below an absolute 100 Hz threshold, warning
# about chunking artifacts. The threshold is not relative to the sampling
# rate, so it fires for every useful photometry band; the margin that actually
# prevents those artifacts is sized from freq_min separately (margin_ms="auto")
# and still applies. Bypass the threshold rather than make every caller do it.
_ALLOW_LOW_FREQ = {"ignore_low_freq_error": True}

# Upstream defaults are 300 Hz / 6 kHz, chosen for spike-band ephys and
# meaningless above the Nyquist frequency of a photometry rig. These are
# conventional starting points, not the right answer for any given
# experiment: 0.1 Hz removes slow drift while leaving calcium transient decay
# (~0.5-2 s) intact, and 5 Hz sits above the fastest calcium kinetics while
# staying below Nyquist even for the ~20 Hz systems. Both remain overridable
# per call, and analyses should state the band they used.
_BANDPASS_DEFAULTS = _ALLOW_LOW_FREQ | {"freq_min": 0.1, "freq_max": 5.0}
_HIGHPASS_DEFAULTS = _ALLOW_LOW_FREQ | {"freq_min": 0.1}


def port_from_si(
    source_class: type, name: str, default_kwargs: dict | None = None
) -> tuple[type, callable]:
    """
    Port one upstream preprocessor into fiber-mosaic.

    Parameters
    ----------
    source_class : type
        The upstream preprocessor class to port.
    name : str
        Public name for the resulting function.
    default_kwargs : dict, optional
        Defaults to replace on the upstream constructor.

    Returns
    -------
    fiber_class : type
        Fiber-aware subclass of ``source_class``.
    function : callable
        Public function dispatching over recording / dict / group.
    """
    fiber_class = as_fiber_preprocessor(
        source_class, default_kwargs=default_kwargs
    )
    function = define_function_handling_group_from_class(fiber_class, name)
    return fiber_class, function


FiberBandpassFilterRecording, bandpass_filter = port_from_si(
    BandpassFilterRecording, "bandpass_filter", _BANDPASS_DEFAULTS
)
FiberHighpassFilterRecording, highpass_filter = port_from_si(
    HighpassFilterRecording, "highpass_filter", _HIGHPASS_DEFAULTS
)
FiberNotchFilterRecording, notch_filter = port_from_si(
    NotchFilterRecording, "notch_filter"
)
FiberGaussianFilterRecording, gaussian_filter = port_from_si(
    GaussianFilterRecording, "gaussian_filter"
)
FiberCenterRecording, center = port_from_si(CenterRecording, "center")
FiberScaleRecording, scale = port_from_si(ScaleRecording, "scale")
FiberZScoreRecording, zscore = port_from_si(ZScoreRecording, "zscore")
FiberNormalizeByQuantileRecording, normalize_by_quantile = port_from_si(
    NormalizeByQuantileRecording, "normalize_by_quantile"
)
FiberCommonReferenceRecording, common_reference = port_from_si(
    CommonReferenceRecording, "common_reference"
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
    "port_from_si",
    "scale",
    "zscore",
]
