"""
Processing module for fiber-mosaic.

Lazy, fiber-native preprocessing built on SpikeInterface's preprocessor
machinery. A step computes in `get_traces()` rather than up front, so
intermediates keep their provenance and can be sliced, saved and reloaded.

Two things live here:

- `baseprocessor`: base classes for fiber-mosaic's own steps, plus
  `as_fiber_preprocessor` for giving an upstream class the fiber API.
- `from_si`: which upstream steps are ported, and with which defaults.

Every ported step takes a single recording, a dict of recordings, or a
FiberPhotometryRecordingGroup, and returns the matching container:

    >>> from fiber_mosaic.processing import bandpass_filter
    >>> filtered = bandpass_filter(group, freq_min=0.1, freq_max=5.0)
    >>> filtered.colors
    ['green', 'iso']
"""

from __future__ import annotations


from fiber_mosaic.processing.from_si import (
    FiberBandpassFilterRecording,
    FiberCenterRecording,
    FiberCommonReferenceRecording,
    FiberGaussianFilterRecording,
    FiberHighpassFilterRecording,
    FiberNormalizeByQuantileRecording,
    FiberNotchFilterRecording,
    FiberScaleRecording,
    FiberZScoreRecording,
    bandpass_filter,
    center,
    common_reference,
    gaussian_filter,
    highpass_filter,
    normalize_by_quantile,
    notch_filter,
    scale,
    zscore,
)

_all_processer_dict = {
    # filter stuff
    FiberBandpassFilterRecording: bandpass_filter,
    FiberHighpassFilterRecording: highpass_filter,
    FiberNotchFilterRecording: notch_filter,
    FiberGaussianFilterRecording: gaussian_filter,
    FiberCommonReferenceRecording: common_reference,
    # gain offset stuff
    FiberNormalizeByQuantileRecording: normalize_by_quantile,
    FiberScaleRecording: scale,
    FiberCenterRecording: center,
    FiberZScoreRecording: zscore,
}
# we control import in the preprocessing init by setting an __all__

# pp_function.__name__ gives the name of the function that users should use
__all__ = [pp_function.__name__ for pp_function in _all_processer_dict.values()]

processor_dict = {pp_class.__name__: pp_function for pp_class, pp_function in _all_processer_dict.items()}
__all__.append("preprocessor_dict")
