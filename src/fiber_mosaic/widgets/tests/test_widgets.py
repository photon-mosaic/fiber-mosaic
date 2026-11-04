"""Pytest tests for ``fiber_mosaic.widgets``."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from spikeinterface.core import NumpyRecording
from spikeinterface.core.numpyextractors import NumpyRecordingSegment

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fiber_mosaic.core.base import (  # noqa: E402
    BaseFiberPhotometryExtractor,
    FiberPhotometryRecordingGroup,
)
from fiber_mosaic.processing import zscore  # noqa: E402
from fiber_mosaic.widgets.fluorescence import (  # noqa: E402
    FiberTracesWidget,
    FluorescenceWidget,
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


def _make_group(**kwargs) -> FiberPhotometryRecordingGroup:
    """Green and iso, interleaved: iso samples half a frame later."""
    return FiberPhotometryRecordingGroup(
        {
            "green": _make_recording(color="green", **kwargs),
            "iso": _make_recording(color="iso", t_start=0.025, **kwargs),
        }
    )


def _legend_texts(ax) -> list[str]:
    return [text.get_text() for text in ax.get_legend().get_texts()]


def _tick_labels(ax) -> list[str]:
    return [tick.get_text() for tick in ax.get_yticklabels()]


def _line_colors(ax) -> list[str]:
    return [line.get_color() for line in ax.get_lines()]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def recording():
    return _make_recording()


@pytest.fixture
def group():
    return _make_group()


@pytest.fixture
def ipympl_backend():
    """Switch matplotlib to ipympl, as ``%matplotlib widget`` would."""
    previous = matplotlib.get_backend()
    matplotlib.use("module://ipympl.backend_nbagg")
    yield
    plt.close("all")
    matplotlib.use(previous)


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------


def test_exports():
    from fiber_mosaic.widgets import plot_fluorescence

    assert plot_fluorescence is FluorescenceWidget


# ---------------------------------------------------------------------------
# FluorescenceWidget – single recording, matplotlib backend
# ---------------------------------------------------------------------------


class TestRecordingMatplotlib:
    def test_plots_every_fiber_with_fiber_labels(self, recording):
        """Each fiber gets a trace, a tick label and the fiber axis label."""
        widget = FluorescenceWidget(recording)
        assert len(widget.ax.get_lines()) == 3
        assert widget.ax.get_ylabel() == "fiber"
        assert _tick_labels(widget.ax) == ["F0", "F1", "F2"]
        assert widget.ax.get_xlabel() == "time (s)"
        assert len(widget.traces_widgets) == 1
        assert isinstance(widget.traces_widgets[0], FiberTracesWidget)

    def test_legend_names_the_color(self):
        widget = FluorescenceWidget(_make_recording(color="iso"))
        assert _legend_texts(widget.ax) == ["iso"]

    def test_legend_falls_back_when_no_color(self):
        """A recording without a color annotation is labelled 'signal'."""
        widget = FluorescenceWidget(NumpyRecording(np.zeros((100, 2)), FS))
        assert _legend_texts(widget.ax) == ["signal"]

    def test_no_legend(self, recording):
        widget = FluorescenceWidget(recording, add_legend=False)
        assert widget.ax.get_legend() is None

    def test_fiber_subset(self, recording):
        widget = FluorescenceWidget(recording, fiber_ids=["F2"])
        assert len(widget.ax.get_lines()) == 1
        assert _tick_labels(widget.ax) == ["F2"]

    def test_hide_fiber_ids(self, recording):
        widget = FluorescenceWidget(recording, show_fiber_ids=False)
        assert widget.ax.get_ylabel() == ""
        assert not widget.ax.get_yaxis().get_visible()

    def test_trace_color(self, recording):
        widget = FluorescenceWidget(recording, trace_color="tab:green")
        assert set(_line_colors(widget.ax)) == {"tab:green"}

    def test_events_are_drawn(self, recording):
        widget = FluorescenceWidget(
            recording, events=np.array([5.0, 15.0, 90.0])
        )
        # Two fall in the default 60 s window, plus the three traces.
        assert len(widget.ax.get_lines()) == 3 + 2

    def test_draws_into_a_given_axis(self, recording):
        _, ax = plt.subplots()
        widget = FluorescenceWidget(recording, ax=ax)
        assert widget.ax is ax

    def test_figtitle(self, recording):
        widget = FluorescenceWidget(recording, figtitle="session 1")
        assert widget.figure.get_suptitle() == "session 1"

    def test_processed_recording(self):
        """A processing step's output plots like an extractor's, color kept."""
        widget = FluorescenceWidget(zscore(_make_recording(color="green")))
        assert _legend_texts(widget.ax) == ["green"]
        assert len(widget.ax.get_lines()) == 3

    @pytest.mark.parametrize(
        "not_a_recording", [np.zeros((100, 2)), [1, 2, 3]]
    )
    def test_rejects_non_recordings(self, not_a_recording):
        with pytest.raises(TypeError, match="accepts a recording"):
            FluorescenceWidget(not_a_recording)

    def test_overlay_is_only_for_groups(self, recording):
        with pytest.raises(ValueError, match="overlay"):
            FluorescenceWidget(recording, overlay=True)

    def test_unsupported_backend(self, recording):
        with pytest.raises(AssertionError, match="not available"):
            FluorescenceWidget(recording, backend="figpack")


# ---------------------------------------------------------------------------
# FluorescenceWidget – time window and segments
# ---------------------------------------------------------------------------


class TestTimeRange:
    def test_default_window_is_sixty_seconds(self, recording):
        widget = FluorescenceWidget(recording)
        assert widget.ax.get_xlim() == pytest.approx((0.0, 60.0))

    def test_default_window_clipped_to_short_segment(self):
        """A recording shorter than the window is shown whole, silently."""
        recording = _make_recording(duration_s=10.0)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            widget = FluorescenceWidget(recording)
        assert widget.ax.get_xlim() == pytest.approx(
            (0.0, recording.get_end_time())
        )

    def test_explicit_time_range(self, recording):
        widget = FluorescenceWidget(recording, time_range=(10.0, 30.0))
        assert widget.ax.get_xlim() == pytest.approx((10.0, 30.0))

    def test_multi_segment_requires_segment_index(self):
        with pytest.raises(ValueError, match="segment_index"):
            FluorescenceWidget(_make_recording(n_segments=2))

    def test_multi_segment_with_segment_index(self):
        widget = FluorescenceWidget(
            _make_recording(n_segments=2), segment_index=1
        )
        assert widget.ax.get_xlim() == pytest.approx((0.0, 60.0))


# ---------------------------------------------------------------------------
# FluorescenceWidget – display centering
# ---------------------------------------------------------------------------


class TestCentering:
    def test_keeps_baseline_traces_readable(self):
        """Centered traces sit on their fiber's tick, spaced by their swing."""
        widget = FluorescenceWidget(_make_recording(baseline=500.0))
        ticks = widget.ax.get_yticks()
        for line, tick in zip(widget.ax.get_lines(), ticks, strict=True):
            assert np.median(line.get_ydata()) == pytest.approx(tick)
        # Spacing comes from the ~3-unit swing, not the 500-unit baseline.
        assert ticks[1] - ticks[0] < 10

    def test_off_keeps_absolute_levels(self):
        widget = FluorescenceWidget(
            _make_recording(baseline=500.0), center_traces=False
        )
        first = widget.ax.get_lines()[0]
        assert np.median(first.get_ydata()) == pytest.approx(500.0, abs=3)

    def test_does_not_change_the_data(self, recording):
        before = recording.get_fluorescence()
        FluorescenceWidget(recording)
        np.testing.assert_array_equal(recording.get_fluorescence(), before)

    def test_flat_traces_get_a_unit_spacing(self):
        """Constant traces have no amplitude to size the spacing from."""
        widget = FluorescenceWidget(NumpyRecording(np.full((100, 2), 7.0), FS))
        np.testing.assert_allclose(widget.ax.get_yticks(), [0.0, 1.0])

    def test_map_mode_is_left_to_upstream(self, recording):
        """Centering and the fiber label only apply to line mode."""
        widget = FiberTracesWidget(
            recording, mode="map", time_range=(0, 10), show_channel_ids=True
        )
        assert widget.ax.get_ylabel() == ""
        image = np.asarray(widget.ax.get_images()[0].get_array())
        assert np.median(image) == pytest.approx(500.0, abs=3)


# ---------------------------------------------------------------------------
# FluorescenceWidget – dict of recordings, matplotlib backend
# ---------------------------------------------------------------------------


class TestDictMatplotlib:
    def test_overlays_stages_on_one_axis(self, recording):
        """A dict compares stages of one color: one axis, one color each."""
        widget = FluorescenceWidget(
            {"raw": recording, "zscore": zscore(recording)}
        )
        assert len(widget.ax.get_lines()) == 2 * 3
        assert _legend_texts(widget.ax) == ["raw", "zscore"]
        assert _line_colors(widget.ax) == ["C0"] * 3 + ["C1"] * 3

    def test_ignores_trace_color(self, recording):
        """One color for every layer would hide which layer is which."""
        widget = FluorescenceWidget(
            {"raw": recording, "zscore": zscore(recording)},
            trace_color="tab:green",
        )
        assert _line_colors(widget.ax) == ["C0"] * 3 + ["C1"] * 3

    def test_dict_of_one_keeps_trace_color(self, recording):
        widget = FluorescenceWidget(
            {"raw": recording}, trace_color="tab:green"
        )
        assert set(_line_colors(widget.ax)) == {"tab:green"}

    def test_different_timestamps_raise(self):
        """Upstream would draw the second layer against the first's times."""
        green = _make_recording(color="green")
        iso = _make_recording(color="iso", t_start=0.025)
        with pytest.raises(ValueError, match="FiberPhotometryRecordingGroup"):
            FluorescenceWidget({"green": green, "iso": iso})

    def test_different_fibers_raise(self):
        first = _make_recording(n_fibers=3)
        second = _make_recording(n_fibers=2)
        with pytest.raises(ValueError, match="fiber_ids"):
            FluorescenceWidget({"a": first, "b": second})

    def test_empty_dict_raises(self):
        with pytest.raises(ValueError, match="empty"):
            FluorescenceWidget({})

    def test_dict_of_non_recordings_raises(self):
        with pytest.raises(TypeError, match="'green' entry"):
            FluorescenceWidget({"green": None})

    def test_overlay_is_only_for_groups(self, recording):
        with pytest.raises(ValueError, match="overlay"):
            FluorescenceWidget({"raw": recording}, overlay=True)


# ---------------------------------------------------------------------------
# FluorescenceWidget – group, matplotlib backend
# ---------------------------------------------------------------------------


class TestGroupMatplotlib:
    def test_one_subplot_per_color(self, group):
        widget = FluorescenceWidget(group)
        green, iso = widget.traces_widgets
        assert green.ax is not iso.ax
        assert green.ax.figure is iso.ax.figure is widget.figure
        assert [green.ax.get_title(), iso.ax.get_title()] == ["green", "iso"]
        assert green.ax.get_legend() is None
        for ax in (green.ax, iso.ax):
            assert len(ax.get_lines()) == 3
            assert ax.get_xlim() == pytest.approx((0.0, 60.0))

    def test_each_color_on_its_own_timestamps(self, group):
        green, iso = FluorescenceWidget(group).traces_widgets
        assert green.ax.get_lines()[0].get_xdata()[0] == pytest.approx(0.0)
        assert iso.ax.get_lines()[0].get_xdata()[0] == pytest.approx(0.025)

    def test_stacked_subplots_label_time_once(self, group):
        """Stacked subplots share a time axis, labelled at the bottom only."""
        green, iso = FluorescenceWidget(group).traces_widgets
        assert green.ax.get_xlabel() == ""
        assert iso.ax.get_xlabel() == "time (s)"

    def test_into_given_axes(self, group):
        _, axes = plt.subplots(2, 1)
        green, iso = FluorescenceWidget(group, axes=axes).traces_widgets
        assert green.ax is axes[0]
        assert iso.ax is axes[1]

    def test_given_axes_keep_every_time_label(self, group):
        """Axes the caller laid out may not be stacked; leave their labels."""
        _, axes = plt.subplots(1, 2)
        widget = FluorescenceWidget(group, axes=axes)
        for traces_widget in widget.traces_widgets:
            assert traces_widget.ax.get_xlabel() == "time (s)"

    def test_wrong_number_of_axes_raises(self, group):
        _, axes = plt.subplots(3, 1)
        with pytest.raises(ValueError, match="one axis per color"):
            FluorescenceWidget(group, axes=axes)

    def test_single_axis_for_subplots_raises(self, group):
        _, ax = plt.subplots()
        with pytest.raises(ValueError, match="one axis per color"):
            FluorescenceWidget(group, ax=ax)

    @pytest.mark.parametrize("overlay", [False, True])
    def test_default_window_clipped_to_short_segment(self, overlay):
        """The window ends with the last color to end, not at 60 s."""
        group = _make_group(duration_s=10.0)
        widget = FluorescenceWidget(group, overlay=overlay)
        iso_end = group["iso"].get_end_time()
        for traces_widget in widget.traces_widgets:
            assert traces_widget.ax.get_xlim() == pytest.approx((0.0, iso_end))

    def test_requires_segment_index_when_multi_segment(self):
        with pytest.raises(ValueError, match="segment_index"):
            FluorescenceWidget(_make_group(n_segments=2))


# ---------------------------------------------------------------------------
# FluorescenceWidget – group overlay, matplotlib backend
# ---------------------------------------------------------------------------


class TestGroupOverlay:
    def test_shares_one_axis(self, group):
        widget = FluorescenceWidget(group, overlay=True)
        ax = widget.ax
        assert all(tw.ax is ax for tw in widget.traces_widgets)
        assert len(ax.get_lines()) == 2 * 3
        assert _legend_texts(ax) == ["green", "iso"]
        assert set(_line_colors(ax)) == {"C0", "C1"}
        assert ax.get_xlim() == pytest.approx((0.0, 60.0))

    def test_lines_fibers_up_across_colors(self):
        """Each fiber's traces share a row, even with different amplitudes."""
        group = FiberPhotometryRecordingGroup(
            {
                "green": _make_recording(color="green"),
                "iso": zscore(_make_recording(color="iso", t_start=0.025)),
            }
        )
        lines = FluorescenceWidget(group, overlay=True).ax.get_lines()
        for fiber in range(3):
            green_row = np.median(lines[fiber].get_ydata())
            iso_row = np.median(lines[3 + fiber].get_ydata())
            assert green_row == pytest.approx(iso_row, abs=1e-6)

    def test_without_centering_keeps_shared_spacing(self, group):
        widget = FluorescenceWidget(group, overlay=True, center_traces=False)
        lines = widget.ax.get_lines()
        assert np.median(lines[0].get_ydata()) == pytest.approx(500.0, abs=3)
        np.testing.assert_allclose(
            np.diff([np.median(line.get_ydata()) for line in lines[:3]]),
            np.diff([np.median(line.get_ydata()) for line in lines[3:]]),
            atol=1e-6,
        )

    def test_draws_events_once(self, group):
        widget = FluorescenceWidget(
            group, overlay=True, events=np.array([5.0, 15.0])
        )
        assert len(widget.ax.get_lines()) == 2 * 3 + 2

    def test_into_given_axis(self, group):
        _, ax = plt.subplots()
        widget = FluorescenceWidget(group, overlay=True, ax=ax)
        assert widget.ax is ax


# ---------------------------------------------------------------------------
# FluorescenceWidget – ipywidgets backend
# ---------------------------------------------------------------------------


class TestIpywidgets:
    def test_keeps_labels_on_redraw(self, ipympl_backend, recording):
        """Moving the time slider redraws; the fiber labels must survive it."""
        widget = FluorescenceWidget(
            recording, backend="ipywidgets", display=False
        )
        (traces_widget,) = widget.traces_widgets
        assert widget.widget is traces_widget.widget
        assert traces_widget.ax.get_ylabel() == "fiber"
        traces_widget.time_slider.value = (200, 600, 0)
        assert traces_widget.ax.get_ylabel() == "fiber"
        assert _tick_labels(traces_widget.ax) == ["F0", "F1", "F2"]

    def test_scale_is_not_undone(self, ipympl_backend, recording):
        """The scale control stretches traces against a fixed spacing."""
        widget = FluorescenceWidget(
            recording, backend="ipywidgets", display=False
        )
        (traces_widget,) = widget.traces_widgets
        ax = traces_widget.ax
        ticks_before = ax.get_yticks().copy()
        swing_before = np.ptp(ax.get_lines()[0].get_ydata())
        traces_widget.scaler.value = traces_widget.scaler.value * 2
        np.testing.assert_allclose(ax.get_yticks(), ticks_before)
        swing_after = np.ptp(ax.get_lines()[0].get_ydata())
        assert swing_after == pytest.approx(2 * swing_before)

    def test_dict_layer_keeps_its_color(self, ipympl_backend, recording):
        """Showing one layer alone must not recolor it as the first layer."""
        widget = FluorescenceWidget(
            {"raw": recording, "zscore": zscore(recording)},
            backend="ipywidgets",
            display=False,
        )
        (traces_widget,) = widget.traces_widgets
        traces_widget.layer_selector.value = "zscore"
        assert set(_line_colors(traces_widget.ax)) == {"C1"}

    def test_group_stacks_one_plot_per_color(self, ipympl_backend, group):
        widget = FluorescenceWidget(group, backend="ipywidgets", display=False)
        green, iso = widget.traces_widgets
        label_green, plot_green, label_iso, plot_iso = widget.widget.children
        assert [label_green.value, label_iso.value] == ["green", "iso"]
        assert plot_green is green.widget
        assert plot_iso is iso.widget

    def test_group_overlay_raises(self, ipympl_backend, group):
        with pytest.raises(ValueError, match="matplotlib backend"):
            FluorescenceWidget(
                group, backend="ipywidgets", overlay=True, display=False
            )

    def test_displays_by_default(self, ipympl_backend, recording, monkeypatch):
        shown = []
        monkeypatch.setattr("IPython.display.display", shown.append)
        widget = FluorescenceWidget(recording, backend="ipywidgets")
        assert shown == [widget.widget]
