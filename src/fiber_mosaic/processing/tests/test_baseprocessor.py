"""Pytest tests for ``fiber_mosaic.processing.baseprocessor``."""

from __future__ import annotations

import importlib
import inspect
import pickle

import numpy as np
import pytest
from spikeinterface.core.numpyextractors import (
    NumpyRecording,
    NumpyRecordingSegment,
)
from spikeinterface.preprocessing.basepreprocessor import (
    BasePreprocessorSegment,
)
from spikeinterface.preprocessing.preprocessing_classes import (
    BandpassFilterRecording,
)

from fiber_mosaic.core.base import (
    BaseFiberPhotometryExtractor,
    FiberPhotometryRecordingGroup,
)
from fiber_mosaic.processing import (
    bandpass_filter,
    from_spikeinterface,
    highpass_filter,
    scale,
)
from fiber_mosaic.processing.baseprocessor import (
    BaseFiberPhotometryPreprocessor,
)
from fiber_mosaic.processing.from_spikeinterface import (
    FiberBandpassFilterRecording,
    FiberScaleRecording,
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
    """Build a recording, optionally with an explicit time vector."""
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
            times = np.arange(n_samples) / sampling_frequency
            rec.set_times(
                times + segment_index * 100.0,
                segment_index=segment_index,
                with_warning=False,
            )
    return rec


def _dumpable_recording():
    """A recording SpikeInterface can rebuild from its dict, for pickling.

    ``_make_recording`` adds segments by hand and records no ``_kwargs``, so
    its reloaded copy would have no segments.
    """
    traces = np.random.default_rng(0).standard_normal((500, 3))
    return NumpyRecording([traces.astype("float32")], 100.0)


def _ported_classes():
    """Every Fiber... class exported by ``from_spikeinterface``."""
    for name in from_spikeinterface.__all__:
        if name.startswith("Fiber"):
            yield getattr(from_spikeinterface, name)


# ---------------- ported classes ----------------


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


def test_wrapped_class_survives_pickle() -> None:
    """A ported step can be pickled and restored as the same class."""
    filtered = bandpass_filter(_dumpable_recording(), **FILTER_KWARGS)

    restored = pickle.loads(pickle.dumps(filtered))

    assert type(restored) is type(filtered)
    assert restored._kwargs["freq_min"] == FILTER_KWARGS["freq_min"]
    np.testing.assert_allclose(restored.get_traces(), filtered.get_traces())


@pytest.mark.parametrize("fiber_class", list(_ported_classes()))
def test_every_ported_class_reloads_from_its_dict(fiber_class) -> None:
    """SpikeInterface finds each class again by its module and name."""
    module = importlib.import_module(fiber_class.__module__)
    assert getattr(module, fiber_class.__name__) is fiber_class


def test_class_without_new_defaults_survives_pickle() -> None:
    """An empty port pickles too, through load() of its dict."""
    scaled = scale(_dumpable_recording(), gain=2.0)

    restored = pickle.loads(pickle.dumps(scaled))

    assert type(restored) is FiberScaleRecording
    np.testing.assert_allclose(restored.get_traces(), scaled.get_traces())


# ---------------- photometry defaults ----------------


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
    parameters = inspect.signature(FiberBandpassFilterRecording).parameters
    assert parameters["freq_min"].default == 0.1
    assert parameters["freq_max"].default == 5.0
    # untouched parameters keep the upstream default
    assert parameters["margin_ms"].default == "auto"


def test_explicit_keyword_beats_the_default():
    """The photometry default is only a default; callers still win."""
    filtered = bandpass_filter(_make_recording(), freq_min=2.0)
    assert filtered._kwargs["freq_min"] == 2.0


def test_extra_filter_kwargs_pass_through():
    """Names absorbed by upstream's **filter_kwargs still reach it."""
    filtered = bandpass_filter(_make_recording(), ftype="bessel")
    assert filtered._kwargs["ftype"] == "bessel"


def test_ported_filters_default_to_a_photometry_band():
    """The advertised defaults are usable at photometry sampling rates."""
    bandpass_params = bandpass_filter.__signature__.parameters
    assert bandpass_params["freq_min"].default == 0.1
    assert bandpass_params["freq_max"].default == 5.0
    assert highpass_filter.__signature__.parameters["freq_min"].default == 0.1

    # and they work with no band specified at all, at 100 Hz sampling
    rec = _make_recording()
    assert bandpass_filter(rec).get_fluorescence().shape == (500, 3)
    assert highpass_filter(rec).get_fluorescence().shape == (500, 3)


# ---------------- BaseFiberPhotometryPreprocessor ----------------


class _GainSegment(BasePreprocessorSegment):
    """Segment applying a constant gain, to exercise the base classes."""

    def __init__(self, parent_recording_segment, gain):
        BasePreprocessorSegment.__init__(self, parent_recording_segment)
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
