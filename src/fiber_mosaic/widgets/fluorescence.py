"""
Fiber-native trace plotting.

SpikeInterface's ``TracesWidget`` does the plotting -- backends, interactive
controls, event markers -- but its vocabulary and defaults are
electrophysiology's. This module keeps the machinery and changes the rest:

- FiberTracesWidget: ``TracesWidget`` with fiber labels and display
  centering, applied in ``plot_matplotlib`` so they survive the redraws the
  ipywidgets backend makes on every slider move.
- FluorescenceWidget: the public widget, in fiber-photometry vocabulary,
  exposed as ``plot_fluorescence`` following SpikeInterface's convention.
  It draws each axis with a FiberTracesWidget.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from spikeinterface.core import BaseRecording
from spikeinterface.widgets.base import (
    BaseWidget,
    get_default_plotter_backend,
    to_attr,
)
from spikeinterface.widgets.traces import TracesWidget
from spikeinterface.widgets.utils_matplotlib import make_mpl_figure

from fiber_mosaic.core.base import FiberPhotometryRecordingGroup

# SpikeInterface shows the first second by default, which is only 20-30
# samples on a photometry rig; a minute shows a few transients.
DEFAULT_WINDOW_S = 60.0


class FiberTracesWidget(TracesWidget):
    """``TracesWidget`` with fiber labels and display centering.

    Takes ``TracesWidget``'s arguments unchanged, plus ``center_traces``
    and ``vspacing``. Most callers want :func:`plot_fluorescence` instead,
    which speaks fiber vocabulary and sets photometry defaults.

    Upstream stacks channels at a spacing sized from their absolute
    amplitude. That suits ephys, which is centered on zero, but a raw
    photometry trace sits on a large baseline: fluorescence of ~500 with
    transients of a few units draws as a flat line, and the channel ticks
    land far from their traces. With ``center_traces`` each fiber's median
    over the plotted window is subtracted *for display only* -- absolute
    levels are lost on the plot, which already hides them behind the
    stacking offsets, and the data are untouched.

    Parameters
    ----------
    recording : BaseRecording or dict
        Passed to ``TracesWidget``.
    center_traces : bool, default: True
        Subtract each fiber's median over the plotted window before
        drawing. Only applies in ``"line"`` mode.
    vspacing : float, optional
        Vertical spacing between fibers. If None, sized from the first
        draw (with centering) or by upstream (without). Set it to line up
        several widgets drawn on one axis.
    vspacing_factor : float, default: 1.5
        When sizing the spacing, the multiple of the largest amplitude in
        view to use.
    **kwargs
        Passed to ``TracesWidget``.

    Notes
    -----
    The ipywidgets backend's controls (the channel selector, the layer and
    mode drop-downs) are built inside SpikeInterface and keep its wording.
    """

    def __init__(
        self,
        recording,
        center_traces: bool = True,
        vspacing: float | None = None,
        vspacing_factor: float = 1.5,
        **kwargs,
    ):
        # TracesWidget.__init__ draws immediately, so these must be set
        # before it is called.
        self._center_traces = center_traces
        self._vspacing = vspacing
        self._vspacing_factor = vspacing_factor
        # Upstream's layer palette runs to pale yellows; matplotlib's color
        # cycle stays legible, and matches what an overlaid group uses.
        self._cycle_layer_colors = (
            isinstance(recording, dict)
            and len(recording) > 1
            and kwargs.get("color") is None
        )
        TracesWidget.__init__(
            self, recording, vspacing_factor=vspacing_factor, **kwargs
        )

    def plot_matplotlib(self, data_plot, **backend_kwargs):
        """Draw with upstream's plotter, then apply fiber labels."""
        line_mode = data_plot["mode"] == "line"
        if line_mode:
            data_plot = self._for_display(data_plot)
        TracesWidget.plot_matplotlib(self, data_plot, **backend_kwargs)
        if line_mode and data_plot["show_channel_ids"]:
            self.ax.set_ylabel("fiber")

    def _for_display(self, data_plot: dict) -> dict:
        """Return a copy of ``data_plot`` colored, centered and spaced.

        The spacing is sized once, from the first draw, and then reused:
        the ipywidgets scale control works by scaling the traces against a
        fixed spacing, so resizing it on every redraw would undo the scale.
        """
        data_plot = dict(data_plot)
        if self._cycle_layer_colors:
            # Index into every layer, not just those in view, so a layer
            # keeps its color when the ipywidgets layer selector hides others.
            keys = self.data_plot["layer_keys"]
            data_plot["colors"] = {
                key: dict.fromkeys(fibers, f"C{keys.index(key)}")
                for key, fibers in data_plot["colors"].items()
            }
        if self._center_traces:
            data_plot["list_traces"] = [
                _center(traces) for traces in data_plot["list_traces"]
            ]
            if self._vspacing is None:
                self._vspacing = _spacing(
                    data_plot["list_traces"], self._vspacing_factor
                )
        if self._vspacing is not None:
            data_plot["vspacing"] = self._vspacing
        return data_plot


class FluorescenceWidget(BaseWidget):
    """Widget for visualizing fluorescence traces of one or more fibers.

    Works on any recording -- an extractor's raw output or the result of a
    processing step -- and stacks the selected fibers on a time axis. What
    is drawn depends on the input:

    - a recording: one plot; the legend names its color.
    - a ``dict`` of recordings: one plot, each recording overlaid in its
      own color and named by its key -- e.g. ``{"raw": rec, "dff": dff}``
      to compare processing stages. The recordings must share fibers and
      timestamps.
    - a ``FiberPhotometryRecordingGroup``: one subplot per color, sharing
      the time axis, or with ``overlay=True`` all colors on one axis. Each
      color is drawn on its own timestamps, so colors sampled at different
      times stay aligned.

    Each axis is drawn by a :class:`FiberTracesWidget`; they are kept, in
    drawing order, in ``traces_widgets``.

    Parameters
    ----------
    recording : BaseRecording, dict or FiberPhotometryRecordingGroup
        What to plot.
    segment_index : int, optional
        The segment to plot. Required for multi-segment recordings.
    fiber_ids : list or array-like, optional
        The fibers to plot. If None, plots all fibers.
    time_range : tuple of float, optional
        ``(start, end)`` in seconds. If None, plots the first 60 s of the
        segment, or the whole segment if it is shorter.
    overlay : bool, default: False
        For a group, draw every color on one axis instead of one subplot
        per color. Needs the matplotlib backend.
    show_fiber_ids : bool, default: True
        Label each trace with its fiber ID.
    center_traces : bool, default: True
        Subtract each fiber's median over the window before drawing, so
        traces on a large baseline stay readable. Display only; see
        :class:`FiberTracesWidget`.
    trace_color : str, optional
        A matplotlib color for every trace of a recording, or of each group
        subplot. Default is black. Ignored when several recordings share an
        axis -- a dict, or a group with ``overlay=True`` -- which always
        draws each recording in its own color.
    events : np.ndarray or list of np.ndarray, optional
        Event times in seconds, drawn as vertical lines; a structured array
        with a ``"duration"`` field draws spans instead. For multi-segment
        recordings, one array per segment.
    events_color : str, default: "gray"
        Color of the event markers.
    events_alpha : float, default: 0.5
        Opacity of the event markers.
    scale : float, default: 1.0
        Multiplies the traces before drawing.
    vspacing_factor : float, default: 1.5
        Vertical spacing between fibers, as a multiple of the largest
        amplitude in view.
    add_legend : bool, default: True
        Add a legend naming each recording. Group subplots are titled by
        color instead.
    backend : {"matplotlib", "ipywidgets"}, optional
        Plotting backend. Defaults to SpikeInterface's default backend,
        normally ``"matplotlib"``. ``"ipywidgets"`` needs
        ``%matplotlib widget`` in a notebook, and draws a group as one
        interactive plot per color.
    **backend_kwargs
        Passed to the backend, e.g. ``ax``, ``axes`` (one per group color)
        or ``figsize`` for matplotlib, ``width_cm`` or ``height_cm`` for
        ipywidgets.

    Raises
    ------
    TypeError
        If ``recording`` is not a recording, a dict of recordings, or a
        group.
    ValueError
        If a dict's recordings do not share fibers and timestamps, if
        ``overlay`` is set for anything but a group, if a group is
        overlaid with the ipywidgets backend, or if ``axes`` does not hold
        one axis per group color.

    Examples
    --------
    >>> from fiber_mosaic.processing import zscore
    >>> plot_fluorescence(zscore(recording), fiber_ids=["F0", "F1"])
    >>> plot_fluorescence({"raw": recording, "zscore": zscore(recording)})
    >>> plot_fluorescence(group, overlay=True)
    """

    def __init__(
        self,
        recording: BaseRecording | dict | FiberPhotometryRecordingGroup,
        segment_index: int | None = None,
        fiber_ids: Sequence | None = None,
        time_range: Sequence[float] | None = None,
        overlay: bool = False,
        show_fiber_ids: bool = True,
        center_traces: bool = True,
        trace_color: str | None = None,
        events: np.ndarray | list[np.ndarray] | None = None,
        events_color: str = "gray",
        events_alpha: float = 0.5,
        scale: float = 1.0,
        vspacing_factor: float = 1.5,
        add_legend: bool = True,
        backend: str | None = None,
        **backend_kwargs,
    ):
        draw_options = dict(
            trace_color=trace_color, events=events, add_legend=add_legend
        )
        if isinstance(recording, FiberPhotometryRecordingGroup):
            if overlay and (
                (backend or get_default_plotter_backend()) == "ipywidgets"
            ):
                raise ValueError(
                    "overlay=True needs the matplotlib backend; the "
                    "ipywidgets backend draws a group as one plot per color"
                )
            spacing_options = dict(
                fiber_ids=fiber_ids,
                center_traces=center_traces,
                scale=scale,
                vspacing_factor=vspacing_factor,
            )
            segment_index, time_range, panels = _group_panels(
                recording,
                segment_index,
                time_range,
                overlay,
                draw_options,
                spacing_options,
            )
        else:
            if overlay:
                raise ValueError(
                    "overlay applies to a FiberPhotometryRecordingGroup; a "
                    "recording or a dict is already drawn on one axis"
                )
            segment_index, time_range, panels = _layer_panels(
                recording, segment_index, time_range, draw_options
            )

        # Prepare data for plotting
        data_plot = dict(
            panels=panels,
            segment_index=segment_index,
            fiber_ids=fiber_ids,
            time_range=time_range,
            show_fiber_ids=show_fiber_ids,
            center_traces=center_traces,
            events_color=events_color,
            events_alpha=events_alpha,
            scale=scale,
            vspacing_factor=vspacing_factor,
        )

        BaseWidget.__init__(self, data_plot, backend=backend, **backend_kwargs)

    def plot_matplotlib(self, data_plot, **backend_kwargs):
        """Matplotlib backend: one axis per panel, drawn by TracesWidgets."""
        dp = to_attr(data_plot)

        num_panels = len(dp.panels)
        stacked = num_panels > 1 and all(
            backend_kwargs[key] is None for key in ("figure", "ax", "axes")
        )
        if num_panels > 1:
            backend_kwargs.update(num_axes=num_panels, ncols=1)
        self.figure, self.axes, self.ax = make_mpl_figure(**backend_kwargs)

        axes = [ax for ax in np.asarray(self.axes).ravel() if ax is not None]
        if len(axes) != num_panels:
            raise ValueError(
                f"axes must hold one axis per color ({num_panels}), "
                f"got {len(axes)}"
            )

        self.traces_widgets = []
        for ax, panel in zip(axes, dp.panels, strict=True):
            for draw in panel["draws"]:
                self.traces_widgets.append(
                    FiberTracesWidget(
                        draw["layers"],
                        backend="matplotlib",
                        ax=ax,
                        **_traces_kwargs(dp, draw),
                    )
                )
            if panel["title"] is not None:
                ax.set_title(panel["title"])
            ax.set_xlim(*dp.time_range)

        if stacked:
            # Stacked panels share one time axis, labelled at the bottom;
            # otherwise each label would collide with the next title.
            for ax in axes[:-1]:
                ax.sharex(axes[-1])
                ax.set_xlabel("")

    def plot_ipywidgets(self, data_plot, **backend_kwargs):
        """Interactive ipywidgets plot: one TracesWidget per panel."""
        import ipywidgets as widgets
        from IPython.display import display

        dp = to_attr(data_plot)

        self.traces_widgets = []
        children = []
        for panel in dp.panels:
            # Overlaid panels hold several draws; __init__ rejects them
            # here, as each ipywidgets plot makes its own figure.
            (draw,) = panel["draws"]
            traces_widget = FiberTracesWidget(
                draw["layers"],
                backend="ipywidgets",
                width_cm=backend_kwargs["width_cm"],
                height_cm=backend_kwargs["height_cm"],
                display=False,
                **_traces_kwargs(dp, draw),
            )
            self.traces_widgets.append(traces_widget)
            if panel["title"] is not None:
                children.append(widgets.Label(value=panel["title"]))
            children.append(traces_widget.widget)

        self.widget = (
            children[0] if len(children) == 1 else widgets.VBox(children)
        )

        # Display if requested
        if backend_kwargs["display"]:
            display(self.widget)


plot_fluorescence = FluorescenceWidget


def _layer_panels(
    recording,
    segment_index: int | None,
    time_range: Sequence[float] | None,
    draw_options: dict,
) -> tuple[int, Sequence[float], list[dict]]:
    """Return one panel drawing a recording, or a dict as layers."""
    layers = _as_layers(recording)
    rec0 = next(iter(layers.values()))
    segment_index = _resolve_segment_index(rec0, segment_index)
    _check_shared_timebase(layers, segment_index)
    if time_range is None:
        time_range = _clip(
            _default_window(rec0, segment_index), rec0, segment_index
        )
    draw = dict(draw_options, layers=layers, time_range=time_range)
    if len(layers) > 1:
        # Upstream would color only the first layer and draw the rest
        # black; one color for every layer would hide which is which.
        draw["trace_color"] = None
    return segment_index, time_range, [dict(title=None, draws=[draw])]


def _group_panels(
    group: FiberPhotometryRecordingGroup,
    segment_index: int | None,
    time_range: Sequence[float] | None,
    overlay: bool,
    draw_options: dict,
    spacing_options: dict,
) -> tuple[int, Sequence[float], list[dict]]:
    """Return one panel per color, or one panel overlaying them all."""
    recordings = dict(group.items())
    rec0 = next(iter(recordings.values()))
    segment_index = _resolve_segment_index(rec0, segment_index)
    if time_range is None:
        # Colors start and end within a frame or two of each other; the
        # earliest start and latest end keep every color's samples in view.
        t_start = min(
            rec.get_start_time(segment_index=segment_index)
            for rec in recordings.values()
        )
        t_end = max(
            rec.get_end_time(segment_index=segment_index)
            for rec in recordings.values()
        )
        time_range = (t_start, min(t_start + DEFAULT_WINDOW_S, t_end))

    def draw(color, recording, **overrides):
        return dict(
            draw_options,
            layers={color: recording},
            time_range=_clip(time_range, recording, segment_index),
            **overrides,
        )

    if not overlay:
        panels = [
            dict(title=color, draws=[draw(color, recording, add_legend=False)])
            for color, recording in recordings.items()
        ]
        return segment_index, time_range, panels

    # One spacing for every color, so each fiber's traces share a row.
    vspacing = _shared_vspacing(
        recordings.values(), segment_index, time_range, **spacing_options
    )
    draws = [
        draw(
            color,
            recording,
            vspacing=vspacing,
            trace_color=f"C{index}",
            # Every color shares the axis; draw the events once.
            events=draw_options["events"] if index == 0 else None,
        )
        for index, (color, recording) in enumerate(recordings.items())
    ]
    return segment_index, time_range, [dict(title=None, draws=draws)]


def _traces_kwargs(dp, draw: dict) -> dict:
    """Translate one draw into FiberTracesWidget's (upstream's) arguments."""
    return dict(
        segment_index=dp.segment_index,
        channel_ids=dp.fiber_ids,
        time_range=draw["time_range"],
        mode="line",
        show_channel_ids=dp.show_fiber_ids,
        center_traces=dp.center_traces,
        vspacing=draw.get("vspacing"),
        vspacing_factor=dp.vspacing_factor,
        events=draw["events"],
        events_color=dp.events_color,
        events_alpha=dp.events_alpha,
        color=draw["trace_color"],
        scale=dp.scale,
        add_legend=draw["add_legend"],
    )


def _as_layers(recording) -> dict[str, BaseRecording]:
    """Return ``recording`` as TracesWidget layers, named for the legend."""
    if isinstance(recording, BaseRecording):
        return {recording.get_annotation("color") or "signal": recording}
    if isinstance(recording, dict):
        if not recording:
            raise ValueError("recordings dict cannot be empty")
        for key, value in recording.items():
            if not isinstance(value, BaseRecording):
                raise TypeError(
                    f"plot_fluorescence() got a dict whose {key!r} entry "
                    f"is {type(value)}, not a recording"
                )
        return recording
    raise TypeError(
        "plot_fluorescence() accepts a recording, a dict of recordings, or "
        f"a FiberPhotometryRecordingGroup, got {type(recording)}"
    )


def _resolve_segment_index(
    recording: BaseRecording, segment_index: int | None
) -> int:
    """Default to the only segment, as upstream does, or raise."""
    if segment_index is not None:
        return segment_index
    if recording.get_num_segments() != 1:
        raise ValueError(
            "segment_index is required for multi-segment recordings"
        )
    return 0


def _check_shared_timebase(
    layers: dict[str, BaseRecording], segment_index: int
) -> None:
    """Raise unless every layer has the first layer's fibers and times.

    Upstream reads every layer at the first layer's frames and draws them
    against its times, so a layer on another timebase would be silently
    shifted. Colors usually are on another timebase; a group handles them.
    """
    (key0, rec0), *rest = layers.items()
    fibers0 = list(rec0.get_channel_ids())
    times0 = rec0.get_times(segment_index=segment_index)
    for key, rec in rest:
        if list(rec.get_channel_ids()) != fibers0:
            raise ValueError(
                f"recording {key!r} has different fiber_ids than {key0!r}; "
                "recordings plotted together must share fibers"
            )
        times = rec.get_times(segment_index=segment_index)
        if not np.array_equal(times, times0):
            raise ValueError(
                f"recording {key!r} has different timestamps than {key0!r}. "
                "Recordings in a dict are drawn against shared timestamps; "
                "to plot colors sampled at different times, pass them as a "
                "FiberPhotometryRecordingGroup"
            )


def _default_window(
    recording: BaseRecording, segment_index: int
) -> tuple[float, float]:
    """Return the first `DEFAULT_WINDOW_S` of the segment."""
    t_start = recording.get_start_time(segment_index=segment_index)
    return t_start, t_start + DEFAULT_WINDOW_S


def _clip(
    time_range: Sequence[float], recording: BaseRecording, segment_index: int
) -> tuple[float, float]:
    """Clip ``time_range``'s end to the segment's, which upstream warns on."""
    t_end = recording.get_end_time(segment_index=segment_index)
    return time_range[0], min(time_range[1], t_end)


def _center(traces: np.ndarray) -> np.ndarray:
    """Subtract each fiber's median."""
    return traces - np.median(traces, axis=0)


def _spacing(list_traces: list[np.ndarray], vspacing_factor: float) -> float:
    """Size the spacing from the largest amplitude across ``list_traces``."""
    amplitude = max(float(np.max(np.abs(traces))) for traces in list_traces)
    # A flat trace has no amplitude to size from; any spacing works.
    return amplitude * vspacing_factor if amplitude > 0 else 1.0


def _shared_vspacing(
    recordings,
    segment_index: int,
    time_range: Sequence[float],
    fiber_ids: Sequence | None,
    center_traces: bool,
    scale: float,
    vspacing_factor: float,
) -> float:
    """Size one spacing from every recording's traces in ``time_range``.

    Reads each recording at its own frames for the window, as each widget
    will, and centers and scales them the same way.
    """
    list_traces = []
    for recording in recordings:
        n_frames = recording.get_num_samples(segment_index=segment_index)
        start, end = (
            int(
                np.clip(
                    recording.time_to_sample_index(t, segment_index),
                    0,
                    n_frames,
                )
            )
            for t in time_range
        )
        traces = recording.get_traces(
            segment_index=segment_index,
            start_frame=start,
            end_frame=end,
            channel_ids=fiber_ids,
        )
        traces = traces * scale
        if center_traces:
            traces = _center(traces)
        list_traces.append(traces)
    return _spacing(list_traces, vspacing_factor)
