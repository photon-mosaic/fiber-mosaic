"""Pytest tests for ``fiber_mosaic.processing.baseprocessor``."""

from __future__ import annotations

import inspect

import numpy as np
import pytest
from spikeinterface.core.numpyextractors import NumpyRecordingSegment
from spikeinterface.preprocessing.preprocessing_classes import (
    BandpassFilterRecording,
    CenterRecording,
    DecimateRecording,
    ScaleRecording,
)

from fiber_mosaic.core.base import (
    BaseFiberPhotometryExtractor,
    FiberPhotometryRecordingGroup,
)
from fiber_mosaic.processing import (
    bandpass_filter,
    highpass_filter,
    port_from_si,
    scale,
)
from fiber_mosaic.processing.baseprocessor import (
    BaseFiberPhotometryPreprocessor,
    BaseFiberPhotometryPreprocessorSegment,
    as_fiber_preprocessor,
)

# A plausible photometry band. No ignore_low_freq_error here: the ported
# bandpass_filter bakes that in, which is what makes these bands usable.
FILTER_KWARGS = {"freq_min": 1.0, "freq_max": 20.0}


def _make_recording(
    color="green",
    fiber_ids=("f0", "f1", "f2"),
    n_samples=500,
    n_segments=1,
    with_times=True,
):
    """Build a recording with per-fiber times offset per fiber."""
    sampling_frequency = 100.0
    rec = BaseFiberPhotometryExtractor(
        sampling_frequency=sampling_frequency,
        fiber_ids=list(fiber_ids),
        color=color,
        dtype="float32",
    )
    rng = np.random.default_rng(0)
    for segment_index in range(n_segments):
        rec.add_segment(
            NumpyRecordingSegment(
                traces=rng.standard_normal((n_samples, len(fiber_ids))).astype(
                    "float32"
                ),
                sampling_frequency=sampling_frequency,
                t_start=None,
            )
        )
        if with_times:
            base = np.arange(n_samples) / sampling_frequency
            offsets = np.arange(len(fiber_ids)) * 0.001
            rec.set_times(
                base[:, None] + offsets + segment_index * 100.0,
                segment_index=segment_index,
            )
    return rec


# ---------------- as_fiber_preprocessor ----------------


def test_wrapped_class_keeps_the_fiber_api():
    """A wrapped upstream preprocessor exposes the fiber-native surface."""
    rec = _make_recording()
    filtered = bandpass_filter(rec, **FILTER_KWARGS)

    assert filtered.color == "green"
    assert filtered.get_num_fibers() == 3
    np.testing.assert_array_equal(
        filtered.get_fiber_ids(), rec.get_fiber_ids()
    )
    assert filtered.get_fluorescence().shape == (500, 3)
    assert "color=green" in repr(filtered)


def test_wrapped_class_stays_lazy():
    """Reading a slice matches slicing the full traces."""
    rec = _make_recording()
    filtered = bandpass_filter(rec, **FILTER_KWARGS)

    np.testing.assert_allclose(
        filtered.get_traces(start_frame=100, end_frame=200),
        filtered.get_traces()[100:200],
        atol=1e-5,
    )


def test_wrapped_class_is_a_subclass_of_the_upstream_one():
    """Wrapping subclasses rather than reimplements, so behaviour is shared."""
    rec = _make_recording()
    filtered = bandpass_filter(rec, **FILTER_KWARGS)

    assert isinstance(filtered, BandpassFilterRecording)
    np.testing.assert_allclose(
        filtered.get_traces(),
        BandpassFilterRecording(
            rec, ignore_low_freq_error=True, **FILTER_KWARGS
        ).get_traces(),
    )


def test_as_fiber_preprocessor_default_and_custom_name():
    """The generated class is named for its source unless told otherwise."""
    assert (
        as_fiber_preprocessor(BandpassFilterRecording).__name__
        == "FiberBandpassFilterRecording"
    )
    assert (
        as_fiber_preprocessor(BandpassFilterRecording, name="Custom").__name__
        == "Custom"
    )


# ---------------- default_kwargs overrides ----------------


def test_ported_filter_bakes_in_the_low_freq_bypass():
    """A photometry band works without the caller repeating the flag.

    Upstream rejects freq_min below an absolute 100 Hz threshold, which sits
    above every useful photometry band.
    """
    rec = _make_recording()
    with pytest.raises(ValueError, match="too low"):
        BandpassFilterRecording(rec, **FILTER_KWARGS)

    assert bandpass_filter(rec, **FILTER_KWARGS).get_num_fibers() == 3


def test_default_kwargs_are_visible_in_the_signature():
    """Overridden defaults show up in help() and inspect.signature."""
    fiber_class = as_fiber_preprocessor(
        BandpassFilterRecording,
        default_kwargs={"freq_min": 0.1, "freq_max": 5.0},
    )
    parameters = inspect.signature(fiber_class).parameters
    assert parameters["freq_min"].default == 0.1
    assert parameters["freq_max"].default == 5.0
    # untouched parameters keep the upstream default
    assert parameters["margin_ms"].default == "auto"


def test_default_kwargs_reach_the_public_function_signature():
    """define_function_handling_group_from_class copies the new defaults."""
    _, function = port_from_si(
        BandpassFilterRecording,
        "narrow_band",
        {"freq_min": 0.1, "freq_max": 5.0, "ignore_low_freq_error": True},
    )
    assert function.__signature__.parameters["freq_min"].default == 0.1


def test_explicit_keyword_beats_the_override():
    """The override is only a default; callers still win."""
    rec = _make_recording()
    fiber_class = as_fiber_preprocessor(
        ScaleRecording, default_kwargs={"gain": 2.0}
    )

    np.testing.assert_allclose(
        fiber_class(rec).get_traces(), rec.get_traces() * 2.0
    )
    np.testing.assert_allclose(
        fiber_class(rec, gain=5.0).get_traces(), rec.get_traces() * 5.0
    )


def test_positional_argument_beats_the_override():
    """A value passed positionally is not overridden either."""
    rec = _make_recording()
    fiber_class = as_fiber_preprocessor(
        ScaleRecording, default_kwargs={"gain": 2.0}
    )

    np.testing.assert_allclose(
        fiber_class(rec, 5.0).get_traces(), rec.get_traces() * 5.0
    )


def test_default_kwargs_rejects_unknown_parameter():
    """A typo is caught at wrap time, not at call time.

    ScaleRecording takes no ``**kwargs``, so an unrecognized name cannot be
    anything but a mistake.
    """
    with pytest.raises(TypeError, match="has no parameter 'gian'"):
        as_fiber_preprocessor(ScaleRecording, default_kwargs={"gian": 2.0})


def test_default_kwargs_rejects_required_parameter():
    """Only parameters that already have a default can be changed."""
    with pytest.raises(TypeError, match="required parameter"):
        as_fiber_preprocessor(
            ScaleRecording, default_kwargs={"recording": None}
        )


def test_default_kwargs_rejects_variadic_parameter():
    """A **kwargs catch-all has no default to replace."""
    with pytest.raises(TypeError, match="variadic parameter"):
        as_fiber_preprocessor(
            CenterRecording, default_kwargs={"random_chunk_kwargs": {}}
        )


def test_unknown_parameter_allowed_when_class_takes_extra_kwargs():
    """Names absorbed by a **kwargs catch-all are passed through."""
    fiber_class = as_fiber_preprocessor(
        BandpassFilterRecording,
        default_kwargs={"ftype": "butter", "ignore_low_freq_error": True},
    )
    rec = _make_recording()
    assert fiber_class(rec, **FILTER_KWARGS).get_num_fibers() == 3


def test_ported_filters_default_to_a_photometry_band():
    """The advertised defaults are usable at photometry sampling rates."""
    bandpass_params = bandpass_filter.__signature__.parameters
    assert bandpass_params["freq_min"].default == 0.1
    assert bandpass_params["freq_max"].default == 5.0
    assert highpass_filter.__signature__.parameters["freq_min"].default == 0.1

    # and they work with no band specified at all, at 100 Hz sampling
    rec = _make_recording()
    assert bandpass_filter(rec).get_fluorescence().shape == (500, 3)


# ---------------- per-fiber times propagation ----------------


def test_times_inherited_through_a_wrapped_upstream_step():
    """Case 3: times resolve via the parent recording."""
    rec = _make_recording()
    filtered = bandpass_filter(rec, **FILTER_KWARGS)

    assert filtered.has_fiber_times()
    np.testing.assert_array_equal(
        filtered.get_fiber_times(), rec.get_fiber_times()
    )


def test_times_inherited_through_a_chain_of_steps():
    """Resolution recurses, so a two-step chain still sees real times."""
    rec = _make_recording()
    chained = scale(bandpass_filter(rec, **FILTER_KWARGS), gain=2.0)

    assert chained.has_fiber_times()
    np.testing.assert_array_equal(
        chained.get_fiber_times(), rec.get_fiber_times()
    )


def test_times_slicing_survives_propagation():
    """Frame and fiber selection work on inherited times too."""
    rec = _make_recording()
    filtered = bandpass_filter(rec, **FILTER_KWARGS)

    np.testing.assert_array_equal(
        filtered.get_fiber_times(
            start_frame=10, end_frame=20, fiber_ids=["f0", "f2"]
        ),
        rec.get_fiber_times(
            start_frame=10, end_frame=20, fiber_ids=["f0", "f2"]
        ),
    )


def test_times_inherited_per_segment():
    """Each segment resolves its own times, not segment 0's."""
    rec = _make_recording(n_segments=2)
    filtered = bandpass_filter(rec, **FILTER_KWARGS)

    for segment_index in range(2):
        np.testing.assert_array_equal(
            filtered.get_fiber_times(segment_index=segment_index),
            rec.get_fiber_times(segment_index=segment_index),
        )


def test_no_times_on_parent_means_no_times_downstream():
    """A source without per-fiber times does not gain them by filtering."""
    rec = _make_recording(with_times=False)
    filtered = bandpass_filter(rec, **FILTER_KWARGS)

    assert not filtered.has_fiber_times()
    assert filtered.get_fiber_times().shape == (500, 3)


def test_shape_changing_step_drops_times_rather_than_misaligning():
    """Decimation invalidates the parent's timestamps, so they are dropped.

    Wrapping a shape-changing step is allowed but not offered as a public
    function: returning the parent's times here would silently pair each
    output sample with the wrong timestamp.
    """
    rec = _make_recording()
    fiber_decimate = as_fiber_preprocessor(DecimateRecording)
    decimated = fiber_decimate(rec, decimation_factor=2)

    assert decimated.get_num_samples() == 250
    assert not decimated.has_fiber_times()


# ---------------- BaseFiberPhotometryPreprocessor ----------------


class _GainSegment(BaseFiberPhotometryPreprocessorSegment):
    """Segment applying a constant gain, to exercise the base classes."""

    def __init__(self, parent_recording_segment, gain):
        BaseFiberPhotometryPreprocessorSegment.__init__(
            self, parent_recording_segment
        )
        self.gain = gain

    def get_traces(self, start_frame, end_frame, channel_indices):
        """Return the parent's traces scaled by ``gain``."""
        traces = self.parent_recording_segment.get_traces(
            start_frame, end_frame, channel_indices
        )
        return traces * self.gain


class _GainRecording(BaseFiberPhotometryPreprocessor):
    """A minimal fiber-mosaic preprocessor built on the base class."""

    def __init__(self, recording, gain=2.0):
        BaseFiberPhotometryPreprocessor.__init__(self, recording)
        for parent_segment in recording.segments:
            self.add_recording_segment(_GainSegment(parent_segment, gain))
        self._kwargs = {"recording": recording, "gain": gain}


def test_own_preprocessor_keeps_fiber_api_and_computes():
    """A subclass gets the fiber API and applies its own computation."""
    rec = _make_recording()
    gained = _GainRecording(rec, gain=3.0)

    assert gained.color == "green"
    assert gained.get_num_fibers() == 3
    np.testing.assert_allclose(
        gained.get_fluorescence(), rec.get_fluorescence() * 3.0
    )


def test_own_preprocessor_inherits_times_at_the_segment_level():
    """Case 2: the segment delegates to its parent segment."""
    rec = _make_recording()
    gained = _GainRecording(rec)

    assert gained.has_fiber_times()
    np.testing.assert_array_equal(
        gained.get_fiber_times(), rec.get_fiber_times()
    )


def test_own_preprocessor_stays_lazy():
    """Computation happens per request, so slices agree with the whole."""
    rec = _make_recording()
    gained = _GainRecording(rec)

    np.testing.assert_allclose(
        gained.get_traces(start_frame=5, end_frame=15),
        gained.get_traces()[5:15],
    )


def test_own_preprocessor_can_override_fibers_and_dtype():
    """Constructor arguments are forwarded to SI's BasePreprocessor."""
    rec = _make_recording()
    processed = BaseFiberPhotometryPreprocessor(rec, dtype="float64")

    assert processed.get_dtype() == np.dtype("float64")
    assert processed.get_num_fibers() == 3


# ---------------- group dispatch through the public functions ----------------


def test_public_function_dispatches_over_a_group():
    """A step applied to a group returns a group of processed colors."""
    green = _make_recording("green")
    iso = _make_recording("iso")
    group = FiberPhotometryRecordingGroup({"green": green, "iso": iso})

    filtered = bandpass_filter(group, **FILTER_KWARGS)

    assert isinstance(filtered, FiberPhotometryRecordingGroup)
    assert filtered.colors == ["green", "iso"]
    assert filtered["green"].color == "green"
    assert filtered["iso"].color == "iso"
    assert filtered["green"].has_fiber_times()


def test_public_function_dispatches_over_a_dict():
    """A dict of colors maps to a dict of processed recordings."""
    recordings = {"green": _make_recording("green")}
    filtered = bandpass_filter(recordings, **FILTER_KWARGS)

    assert set(filtered) == {"green"}
    assert filtered["green"].get_num_fibers() == 3


def test_group_dispatch_requires_shared_fibers():
    """The resulting group is validated like any other group."""
    green = _make_recording("green")
    iso = _make_recording("iso", fiber_ids=("f0", "f1"))
    with pytest.raises(ValueError):
        FiberPhotometryRecordingGroup({"green": green, "iso": iso})
