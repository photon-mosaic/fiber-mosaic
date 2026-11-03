"""
Fiber-native trace plotting.

SpikeInterface's ``TracesWidget`` does the plotting -- backends, interactive
controls, event markers -- but its vocabulary and defaults are
electrophysiology's. This module keeps the machinery and changes the rest:

- FiberTracesWidget: ``TracesWidget`` with fiber labels and display
  centering, applied in ``plot_matplotlib`` so they survive the redraws the
  ipywidgets backend makes on every slider move.
- plot_fluorescence: the public function, in fiber-photometry vocabulary.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from spikeinterface.core import BaseRecording
from spikeinterface.widgets.base import get_default_plotter_backend
from spikeinterface.widgets.traces import TracesWidget

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
        """Return a copy of ``data_plot`` centered and spaced for drawing.

        The spacing is sized once, from the first draw, and then reused:
        the ipywidgets scale control works by scaling the traces against a
        fixed spacing, so resizing it on every redraw would undo the scale.
        """
        list_traces = data_plot["list_traces"]
        if self._center_traces:
            list_traces = [_center(traces) for traces in list_traces]
        if self._vspacing is None:
            if not self._center_traces:
                return data_plot
            self._vspacing = _spacing(list_traces, self._vspacing_factor)
        return {
            **data_plot,
            "list_traces": list_traces,
            "vspacing": self._vspacing,
        }


def plot_fluorescence(
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
) -> FiberTracesWidget | dict[str, FiberTracesWidget]:
    """
    Plot fluorescence traces for one or more fibers.

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
        A matplotlib color for every trace. Default is black, or one color
        per recording when several share an axis; ignored with
        ``overlay=True``, which always colors by color.
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
        interactive widget per color.
    **backend_kwargs
        Passed to the backend, e.g. ``ax`` or ``figsize`` for matplotlib,
        ``width_cm`` or ``height_cm`` for ipywidgets. Group subplots take
        ``axes``, one per color, instead of ``ax``.

    Returns
    -------
    FiberTracesWidget or dict of FiberTracesWidget
        For a recording or a dict, the widget; its ``figure`` and ``ax``
        hold the matplotlib objects. For a group, a dict mapping each
        color to its widget.

    Raises
    ------
    TypeError
        If ``recording`` is not a recording, a dict of recordings, or a
        group.
    ValueError
        If a dict's recordings do not share fibers and timestamps, if
        ``overlay`` is set for anything but a group, or if a group is
        overlaid with a backend other than matplotlib.

    Examples
    --------
    >>> from fiber_mosaic.processing import zscore
    >>> plot_fluorescence(zscore(recording), fiber_ids=["F0", "F1"])
    >>> plot_fluorescence({"raw": recording, "zscore": zscore(recording)})
    >>> plot_fluorescence(group, overlay=True)
    """
    options = dict(
        channel_ids=fiber_ids,
        mode="line",
        show_channel_ids=show_fiber_ids,
        center_traces=center_traces,
        events=events,
        events_color=events_color,
        events_alpha=events_alpha,
        color=trace_color,
        scale=scale,
        vspacing_factor=vspacing_factor,
        add_legend=add_legend,
        backend=backend,
    )
    if isinstance(recording, FiberPhotometryRecordingGroup):
        return _plot_group(
            recording,
            segment_index,
            time_range,
            overlay,
            options,
            backend_kwargs,
        )

    if overlay:
        raise ValueError(
            "overlay applies to a FiberPhotometryRecordingGroup; a "
            "recording or a dict is already drawn on one axis"
        )
    layers = _as_layers(recording)
    rec0 = next(iter(layers.values()))
    segment_index = _resolve_segment_index(rec0, segment_index)
    _check_shared_timebase(layers, segment_index)
    if time_range is None:
        time_range = _clip(
            _default_window(rec0, segment_index), rec0, segment_index
        )
    return FiberTracesWidget(
        layers,
        segment_index=segment_index,
        time_range=time_range,
        **options,
        **backend_kwargs,
    )


def _plot_group(
    group: FiberPhotometryRecordingGroup,
    segment_index: int | None,
    time_range: Sequence[float] | None,
    overlay: bool,
    options: dict,
    backend_kwargs: dict,
) -> dict[str, FiberTracesWidget]:
    """Draw a group as one subplot per color, or overlaid on one axis."""
    recordings = dict(group.items())
    rec0 = next(iter(recordings.values()))
    segment_index = _resolve_segment_index(rec0, segment_index)
    if time_range is None:
        # Colors start within a frame or two of each other; the earliest
        # start keeps every color's first samples in view.
        t_start = min(
            rec.get_start_time(segment_index=segment_index)
            for rec in recordings.values()
        )
        time_range = (t_start, t_start + DEFAULT_WINDOW_S)

    # `label` is the photometry color; `color` in `overrides` is upstream's
    # trace color.
    def draw(label, recording, **overrides):
        kwargs = {**options, **backend_kwargs, **overrides}
        return FiberTracesWidget(
            {label: recording},
            segment_index=segment_index,
            time_range=_clip(time_range, recording, segment_index),
            **kwargs,
        )

    backend = options["backend"] or get_default_plotter_backend()
    if backend == "ipywidgets":
        if overlay:
            raise ValueError(
                "overlay=True needs the matplotlib backend; the ipywidgets "
                "backend draws a group as one widget per color"
            )
        return {
            color: draw(color, recording)
            for color, recording in recordings.items()
        }
    if backend != "matplotlib":
        raise ValueError(
            "a FiberPhotometryRecordingGroup can be drawn with the "
            f"matplotlib or ipywidgets backend, got {backend!r}"
        )

    import matplotlib.pyplot as plt

    backend_kwargs = dict(backend_kwargs)
    figsize = backend_kwargs.pop("figsize", None)
    if overlay:
        ax = backend_kwargs.pop("ax", None)
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        # One spacing for every color, so each fiber's traces share a row.
        vspacing = _shared_vspacing(
            recordings.values(), segment_index, time_range, options
        )
        widgets = _overlay(draw, recordings, ax, vspacing, options["events"])
        ax.set_xlim(*time_range)
        return widgets

    axes = backend_kwargs.pop("axes", None)
    if axes is None:
        _, axes = plt.subplots(
            len(recordings), 1, sharex=True, squeeze=False, figsize=figsize
        )
    return _subplots(draw, recordings, axes, time_range)


def _overlay(
    draw, recordings: dict, ax, vspacing: float, events
) -> dict[str, FiberTracesWidget]:
    """Draw every color on ``ax``, one matplotlib color cycle entry each."""
    widgets = {}
    for index, (color, recording) in enumerate(recordings.items()):
        widgets[color] = draw(
            color,
            recording,
            ax=ax,
            vspacing=vspacing,
            color=f"C{index}",
            # Every color shares the axis; draw the events once.
            events=events if index == 0 else None,
        )
    return widgets


def _subplots(
    draw, recordings: dict, axes, time_range: Sequence[float]
) -> dict[str, FiberTracesWidget]:
    """Draw each color on its own axis, titled by color."""
    axes = np.asarray(axes).ravel()
    if len(axes) != len(recordings):
        raise ValueError(
            f"axes must hold one axis per color ({len(recordings)}), "
            f"got {len(axes)}"
        )
    widgets = {}
    for ax, (color, recording) in zip(axes, recordings.items(), strict=True):
        widgets[color] = draw(color, recording, ax=ax, add_legend=False)
        ax.set_title(color)
        ax.set_xlim(*time_range)
    return widgets


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
    recordings, segment_index: int, time_range: Sequence[float], options: dict
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
            channel_ids=options["channel_ids"],
        )
        traces = traces * options["scale"]
        if options["center_traces"]:
            traces = _center(traces)
        list_traces.append(traces)
    return _spacing(list_traces, options["vspacing_factor"])
