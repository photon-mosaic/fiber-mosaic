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

from fiber_mosaic.core.base import (  # noqa: E402
    BaseFiberPhotometryExtractor,
    FiberPhotometryRecordingGroup,
)
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
    t_start: float | None = None,
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
                traces=traces, sampling_frequency=FS, t_start=t_start
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


@pytest.mark.parametrize("not_a_recording", [np.zeros((100, 2)), [1, 2, 3]])
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


# -- dicts ------------------------------------------------------------------


def _legend_texts(ax) -> list[str]:
    return [text.get_text() for text in ax.get_legend().get_texts()]


def test_dict_overlays_stages_on_one_axis() -> None:
    """A dict compares stages of one color: one axis, one color each."""
    recording = _make_recording()
    widget = plot_fluorescence({"raw": recording, "zscore": zscore(recording)})
    assert isinstance(widget, FiberTracesWidget)
    assert len(widget.ax.get_lines()) == 2 * 3
    assert _legend_texts(widget.ax) == ["raw", "zscore"]
    raw_color = widget.ax.get_lines()[0].get_color()
    zscore_color = widget.ax.get_lines()[3].get_color()
    assert raw_color != zscore_color


def test_dict_with_different_timestamps_raises() -> None:
    """Upstream would draw the second layer against the first's times."""
    green = _make_recording(color="green")
    iso = _make_recording(color="iso", t_start=0.025)
    with pytest.raises(ValueError, match="FiberPhotometryRecordingGroup"):
        plot_fluorescence({"green": green, "iso": iso})


def test_dict_with_different_fibers_raises() -> None:
    first = _make_recording(n_fibers=3)
    second = _make_recording(n_fibers=2)
    with pytest.raises(ValueError, match="fiber_ids"):
        plot_fluorescence({"a": first, "b": second})


def test_empty_dict_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        plot_fluorescence({})


def test_dict_of_non_recordings_raises() -> None:
    with pytest.raises(TypeError, match="'green' entry"):
        plot_fluorescence({"green": None})


@pytest.mark.parametrize("target", ["recording", "dict"])
def test_overlay_is_only_for_groups(target) -> None:
    recording = _make_recording()
    value = recording if target == "recording" else {"raw": recording}
    with pytest.raises(ValueError, match="overlay"):
        plot_fluorescence(value, overlay=True)


# -- groups -----------------------------------------------------------------


def _make_group(**kwargs) -> FiberPhotometryRecordingGroup:
    """Green and iso, interleaved: iso samples half a frame later."""
    return FiberPhotometryRecordingGroup(
        {
            "green": _make_recording(color="green", **kwargs),
            "iso": _make_recording(color="iso", t_start=0.025, **kwargs),
        }
    )


def test_group_draws_one_subplot_per_color() -> None:
    widgets = plot_fluorescence(_make_group())
    assert list(widgets) == ["green", "iso"]
    green_ax, iso_ax = widgets["green"].ax, widgets["iso"].ax
    assert green_ax is not iso_ax
    assert green_ax.figure is iso_ax.figure
    assert [green_ax.get_title(), iso_ax.get_title()] == ["green", "iso"]
    assert green_ax.get_legend() is None
    for ax in (green_ax, iso_ax):
        assert len(ax.get_lines()) == 3
        assert ax.get_xlim() == pytest.approx((0.0, 60.0))


def test_group_draws_each_color_on_its_own_timestamps() -> None:
    widgets = plot_fluorescence(_make_group())
    green_x = widgets["green"].ax.get_lines()[0].get_xdata()
    iso_x = widgets["iso"].ax.get_lines()[0].get_xdata()
    assert green_x[0] == pytest.approx(0.0)
    assert iso_x[0] == pytest.approx(0.025)


def test_group_into_given_axes() -> None:
    _, axes = plt.subplots(2, 1)
    widgets = plot_fluorescence(_make_group(), axes=axes)
    assert widgets["green"].ax is axes[0]
    assert widgets["iso"].ax is axes[1]


def test_group_with_wrong_number_of_axes_raises() -> None:
    _, axes = plt.subplots(3, 1)
    with pytest.raises(ValueError, match="one axis per color"):
        plot_fluorescence(_make_group(), axes=axes)


def test_group_default_window_clipped_to_short_segment() -> None:
    widgets = plot_fluorescence(_make_group(duration_s=10.0))
    green = widgets["green"]
    assert green.ax.get_xlim() == pytest.approx((0.0, 60.0))
    assert green.ax.get_lines()[0].get_xdata()[-1] < 10.0


def test_group_requires_segment_index_when_multi_segment() -> None:
    with pytest.raises(ValueError, match="segment_index"):
        plot_fluorescence(_make_group(n_segments=2))


def test_group_overlay_shares_one_axis() -> None:
    widgets = plot_fluorescence(_make_group(), overlay=True)
    ax = widgets["green"].ax
    assert widgets["iso"].ax is ax
    assert len(ax.get_lines()) == 2 * 3
    assert _legend_texts(ax) == ["green", "iso"]
    colors = {line.get_color() for line in ax.get_lines()}
    assert colors == {"C0", "C1"}
    assert ax.get_xlim() == pytest.approx((0.0, 60.0))


def test_group_overlay_lines_fibers_up_across_colors() -> None:
    """Each fiber's traces share a row, even with different amplitudes."""
    group = FiberPhotometryRecordingGroup(
        {
            "green": _make_recording(color="green"),
            "iso": zscore(_make_recording(color="iso", t_start=0.025)),
        }
    )
    ax = plot_fluorescence(group, overlay=True)["green"].ax
    lines = ax.get_lines()
    for fiber in range(3):
        green_row = np.median(lines[fiber].get_ydata())
        iso_row = np.median(lines[3 + fiber].get_ydata())
        assert green_row == pytest.approx(iso_row, abs=1e-6)


def test_group_overlay_without_centering_keeps_shared_spacing() -> None:
    ax = plot_fluorescence(_make_group(), overlay=True, center_traces=False)[
        "green"
    ].ax
    lines = ax.get_lines()
    assert np.median(lines[0].get_ydata()) == pytest.approx(500.0, abs=3)
    np.testing.assert_allclose(
        np.diff([np.median(line.get_ydata()) for line in lines[:3]]),
        np.diff([np.median(line.get_ydata()) for line in lines[3:]]),
        atol=1e-6,
    )


def test_group_overlay_draws_events_once() -> None:
    widgets = plot_fluorescence(
        _make_group(), overlay=True, events=np.array([5.0, 15.0])
    )
    assert len(widgets["green"].ax.get_lines()) == 2 * 3 + 2


def test_group_overlay_into_given_axis() -> None:
    _, ax = plt.subplots()
    widgets = plot_fluorescence(_make_group(), overlay=True, ax=ax)
    assert widgets["green"].ax is ax


def test_group_with_unsupported_backend_raises() -> None:
    with pytest.raises(ValueError, match="matplotlib or ipywidgets"):
        plot_fluorescence(_make_group(), backend="figpack")


def test_group_ipywidgets_draws_one_widget_per_color(ipympl_backend) -> None:
    widgets = plot_fluorescence(
        _make_group(), backend="ipywidgets", display=False
    )
    assert list(widgets) == ["green", "iso"]
    assert widgets["green"].widget is not widgets["iso"].widget


def test_group_ipywidgets_overlay_raises(ipympl_backend) -> None:
    with pytest.raises(ValueError, match="matplotlib backend"):
        plot_fluorescence(
            _make_group(), backend="ipywidgets", overlay=True, display=False
        )
