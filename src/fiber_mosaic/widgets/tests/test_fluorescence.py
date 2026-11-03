"""Pytest tests for ``fiber_mosaic.widgets.fluorescence``."""

from __future__ import annotations

import warnings

import matplotlib
import numpy as np
import pytest
from spikeinterface.core import NumpyRecording
from spikeinterface.core.numpyextractors import NumpyRecordingSegment

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fiber_mosaic.core.base import BaseFiberPhotometryExtractor  # noqa: E402
from fiber_mosaic.processing import zscore  # noqa: E402
from fiber_mosaic.widgets import (  # noqa: E402
    FiberTracesWidget,
    plot_fluorescence,
)

FS = 20.0


def _make_recording(
    color: str = "green",
    n_fibers: int = 3,
    duration_s: float = 120.0,
    baseline: float = 500.0,
    n_segments: int = 1,
) -> BaseFiberPhotometryExtractor:
    """Build a recording of slow sines on a large baseline, per fiber."""
    recording = BaseFiberPhotometryExtractor(
        sampling_frequency=FS,
        fiber_ids=[f"F{index}" for index in range(n_fibers)],
        color=color,
    )
    times = np.arange(int(duration_s * FS)) / FS
    traces = np.stack(
        [baseline + 3 * np.sin(times / 3 + i) for i in range(n_fibers)],
        axis=1,
    )
    for _ in range(n_segments):
        recording.add_segment(
            NumpyRecordingSegment(
                traces=traces, sampling_frequency=FS, t_start=None
            )
        )
    return recording


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_plots_every_fiber_with_fiber_labels() -> None:
    """Each fiber gets a trace, a tick label and the fiber axis label."""
    widget = plot_fluorescence(_make_recording())
    assert isinstance(widget, FiberTracesWidget)
    assert len(widget.ax.get_lines()) == 3
    assert widget.ax.get_ylabel() == "fiber"
    tick_labels = [tick.get_text() for tick in widget.ax.get_yticklabels()]
    assert tick_labels == ["F0", "F1", "F2"]
    assert widget.ax.get_xlabel() == "time (s)"


def test_legend_names_the_color() -> None:
    widget = plot_fluorescence(_make_recording(color="iso"))
    legend = widget.ax.get_legend()
    assert [text.get_text() for text in legend.get_texts()] == ["iso"]


def test_legend_falls_back_when_no_color() -> None:
    """A recording without a color annotation is labelled 'signal'."""
    recording = NumpyRecording(np.zeros((100, 2)), FS)
    widget = plot_fluorescence(recording)
    legend = widget.ax.get_legend()
    assert [text.get_text() for text in legend.get_texts()] == ["signal"]


def test_no_legend() -> None:
    widget = plot_fluorescence(_make_recording(), add_legend=False)
    assert widget.ax.get_legend() is None


def test_fiber_subset() -> None:
    widget = plot_fluorescence(_make_recording(), fiber_ids=["F2"])
    assert len(widget.ax.get_lines()) == 1
    tick_labels = [tick.get_text() for tick in widget.ax.get_yticklabels()]
    assert tick_labels == ["F2"]


def test_hide_fiber_ids() -> None:
    widget = plot_fluorescence(_make_recording(), show_fiber_ids=False)
    assert widget.ax.get_ylabel() == ""
    assert not widget.ax.get_yaxis().get_visible()


def test_default_window_is_sixty_seconds() -> None:
    widget = plot_fluorescence(_make_recording(duration_s=120.0))
    assert widget.ax.get_xlim() == pytest.approx((0.0, 60.0))


def test_default_window_clipped_to_short_segment() -> None:
    """A recording shorter than the window is shown whole, silently."""
    recording = _make_recording(duration_s=10.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        widget = plot_fluorescence(recording)
    assert widget.ax.get_xlim() == pytest.approx(
        (0.0, recording.get_end_time())
    )


def test_explicit_time_range() -> None:
    widget = plot_fluorescence(_make_recording(), time_range=(10.0, 30.0))
    assert widget.ax.get_xlim() == pytest.approx((10.0, 30.0))


def test_multi_segment_requires_segment_index() -> None:
    with pytest.raises(ValueError, match="segment_index"):
        plot_fluorescence(_make_recording(n_segments=2))


def test_multi_segment_with_segment_index() -> None:
    widget = plot_fluorescence(_make_recording(n_segments=2), segment_index=1)
    assert widget.ax.get_xlim() == pytest.approx((0.0, 60.0))


def test_centering_keeps_baseline_traces_readable() -> None:
    """Centered traces sit on their fiber's tick, spaced by their swing."""
    widget = plot_fluorescence(_make_recording(baseline=500.0))
    ticks = widget.ax.get_yticks()
    for line, tick in zip(widget.ax.get_lines(), ticks, strict=True):
        assert np.median(line.get_ydata()) == pytest.approx(tick)
    # Spacing comes from the ~3-unit swing, not the 500-unit baseline.
    assert ticks[1] - ticks[0] < 10


def test_no_centering_keeps_absolute_levels() -> None:
    widget = plot_fluorescence(
        _make_recording(baseline=500.0), center_traces=False
    )
    first = widget.ax.get_lines()[0]
    assert np.median(first.get_ydata()) == pytest.approx(500.0, abs=3)


def test_centering_does_not_change_the_data() -> None:
    recording = _make_recording()
    before = recording.get_fluorescence()
    plot_fluorescence(recording)
    np.testing.assert_array_equal(recording.get_fluorescence(), before)


def test_flat_traces_get_a_unit_spacing() -> None:
    """Constant traces have no amplitude to size the spacing from."""
    recording = NumpyRecording(np.full((100, 2), 7.0), FS)
    widget = plot_fluorescence(recording)
    np.testing.assert_allclose(widget.ax.get_yticks(), [0.0, 1.0])


def test_trace_color() -> None:
    widget = plot_fluorescence(_make_recording(), trace_color="tab:green")
    for line in widget.ax.get_lines():
        assert line.get_color() == "tab:green"


def test_events_are_drawn() -> None:
    widget = plot_fluorescence(
        _make_recording(), events=np.array([5.0, 15.0, 90.0])
    )
    # Two fall in the default 60 s window, plus the three traces.
    assert len(widget.ax.get_lines()) == 3 + 2


def test_draws_into_a_given_axis() -> None:
    _, ax = plt.subplots()
    widget = plot_fluorescence(_make_recording(), ax=ax)
    assert widget.ax is ax


def test_processed_recording() -> None:
    """A processing step's output plots like an extractor's, color kept."""
    widget = plot_fluorescence(zscore(_make_recording(color="green")))
    legend = widget.ax.get_legend()
    assert [text.get_text() for text in legend.get_texts()] == ["green"]
    assert len(widget.ax.get_lines()) == 3


@pytest.mark.parametrize(
    "not_a_recording",
    [np.zeros((100, 2)), {"green": None}, [1, 2, 3]],
)
def test_rejects_non_recordings(not_a_recording) -> None:
    with pytest.raises(TypeError, match="accepts a recording"):
        plot_fluorescence(not_a_recording)


def test_map_mode_is_left_to_upstream() -> None:
    """Centering and the fiber label only apply to line mode."""
    recording = _make_recording()
    widget = FiberTracesWidget(
        recording, mode="map", time_range=(0, 10), show_channel_ids=True
    )
    assert widget.ax.get_ylabel() == ""
    image = np.asarray(widget.ax.get_images()[0].get_array())
    assert np.median(image) == pytest.approx(500.0, abs=3)


@pytest.fixture
def ipympl_backend():
    """Switch matplotlib to ipympl, as ``%matplotlib widget`` would."""
    previous = matplotlib.get_backend()
    matplotlib.use("module://ipympl.backend_nbagg")
    yield
    plt.close("all")
    matplotlib.use(previous)


def test_ipywidgets_backend_keeps_labels_on_redraw(ipympl_backend) -> None:
    """Moving the time slider redraws; the fiber labels must survive it."""
    widget = plot_fluorescence(
        _make_recording(), backend="ipywidgets", display=False
    )
    assert widget.ax.get_ylabel() == "fiber"
    widget.time_slider.value = (200, 600, 0)
    assert widget.ax.get_ylabel() == "fiber"
    tick_labels = [tick.get_text() for tick in widget.ax.get_yticklabels()]
    assert tick_labels == ["F0", "F1", "F2"]


def test_ipywidgets_scale_is_not_undone(ipympl_backend) -> None:
    """The scale control stretches traces against a fixed spacing."""
    widget = plot_fluorescence(
        _make_recording(), backend="ipywidgets", display=False
    )
    ticks_before = widget.ax.get_yticks().copy()
    swing_before = np.ptp(widget.ax.get_lines()[0].get_ydata())
    widget.scaler.value = widget.scaler.value * 2
    np.testing.assert_allclose(widget.ax.get_yticks(), ticks_before)
    swing_after = np.ptp(widget.ax.get_lines()[0].get_ydata())
    assert swing_after == pytest.approx(2 * swing_before)
