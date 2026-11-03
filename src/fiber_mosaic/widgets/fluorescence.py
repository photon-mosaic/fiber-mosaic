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
from spikeinterface.widgets.traces import TracesWidget

# SpikeInterface shows the first second by default, which is only 20-30
# samples on a photometry rig; a minute shows a few transients.
DEFAULT_WINDOW_S = 60.0


class FiberTracesWidget(TracesWidget):
    """``TracesWidget`` with fiber labels and display centering.

    Takes ``TracesWidget``'s arguments unchanged, plus ``center_traces``.
    Most callers want :func:`plot_fluorescence` instead, which speaks fiber
    vocabulary and sets photometry defaults.

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
    vspacing_factor : float, default: 1.5
        Vertical spacing between fibers, as a multiple of the largest
        amplitude in the first layer (after centering, when enabled).
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
        vspacing_factor: float = 1.5,
        **kwargs,
    ):
        # TracesWidget.__init__ draws immediately, so these must be set
        # before it is called.
        self._center_traces = center_traces
        self._vspacing_factor = vspacing_factor
        self._centered_vspacing = None
        TracesWidget.__init__(
            self, recording, vspacing_factor=vspacing_factor, **kwargs
        )

    def plot_matplotlib(self, data_plot, **backend_kwargs):
        """Draw with upstream's plotter, then apply fiber labels."""
        line_mode = data_plot["mode"] == "line"
        if self._center_traces and line_mode:
            data_plot = self._centered(data_plot)
        TracesWidget.plot_matplotlib(self, data_plot, **backend_kwargs)
        if line_mode and data_plot["show_channel_ids"]:
            self.ax.set_ylabel("fiber")

    def _centered(self, data_plot: dict) -> dict:
        """Return a copy of ``data_plot`` with each fiber median-centered.

        The spacing is sized once, from the first draw, and then reused:
        the ipywidgets scale control works by scaling the traces against a
        fixed spacing, so resizing it on every redraw would undo the scale.
        """
        list_traces = [
            traces - np.median(traces, axis=0)
            for traces in data_plot["list_traces"]
        ]
        if self._centered_vspacing is None:
            amplitude = float(np.max(np.abs(list_traces[0])))
            # A flat trace has no amplitude to size from; any spacing works.
            self._centered_vspacing = (
                amplitude * self._vspacing_factor if amplitude > 0 else 1.0
            )
        return {
            **data_plot,
            "list_traces": list_traces,
            "vspacing": self._centered_vspacing,
        }


def plot_fluorescence(
    recording: BaseRecording,
    segment_index: int | None = None,
    fiber_ids: Sequence | None = None,
    time_range: Sequence[float] | None = None,
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
) -> FiberTracesWidget:
    """
    Plot fluorescence traces for one or more fibers of a recording.

    Works on any recording -- an extractor's raw output or the result of a
    processing step -- and stacks the selected fibers on a shared time
    axis. The legend names the recording's color.

    Parameters
    ----------
    recording : BaseRecording
        The recording to plot.
    segment_index : int, optional
        The segment to plot. Required for multi-segment recordings.
    fiber_ids : list or array-like, optional
        The fibers to plot. If None, plots all fibers.
    time_range : tuple of float, optional
        ``(start, end)`` in seconds. If None, plots the first 60 s of the
        segment, or the whole segment if it is shorter.
    show_fiber_ids : bool, default: True
        Label each trace with its fiber ID.
    center_traces : bool, default: True
        Subtract each fiber's median over the window before drawing, so
        traces on a large baseline stay readable. Display only; see
        :class:`FiberTracesWidget`.
    trace_color : str, optional
        A matplotlib color for every trace. Default is black.
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
        Add a legend naming the recording's color.
    backend : {"matplotlib", "ipywidgets"}, optional
        Plotting backend. Defaults to SpikeInterface's default backend,
        normally ``"matplotlib"``. ``"ipywidgets"`` needs
        ``%matplotlib widget`` in a notebook.
    **backend_kwargs
        Passed to the backend, e.g. ``ax`` or ``figsize`` for matplotlib,
        ``width_cm`` or ``height_cm`` for ipywidgets.

    Returns
    -------
    FiberTracesWidget
        The widget; its ``figure`` and ``ax`` hold the matplotlib objects.

    Raises
    ------
    TypeError
        If ``recording`` is not a recording.

    Examples
    --------
    >>> from fiber_mosaic.processing import zscore
    >>> plot_fluorescence(zscore(recording), fiber_ids=["F0", "F1"])
    """
    if not isinstance(recording, BaseRecording):
        raise TypeError(
            f"plot_fluorescence() accepts a recording, got {type(recording)}"
        )

    if segment_index is None and recording.get_num_segments() == 1:
        segment_index = 0
    if time_range is None and segment_index is not None:
        time_range = _default_time_range(recording, segment_index)

    label = recording.get_annotation("color") or "signal"
    return FiberTracesWidget(
        {label: recording},
        center_traces=center_traces,
        vspacing_factor=vspacing_factor,
        segment_index=segment_index,
        channel_ids=fiber_ids,
        time_range=time_range,
        mode="line",
        show_channel_ids=show_fiber_ids,
        events=events,
        events_color=events_color,
        events_alpha=events_alpha,
        color=trace_color,
        scale=scale,
        add_legend=add_legend,
        backend=backend,
        **backend_kwargs,
    )


def _default_time_range(
    recording: BaseRecording, segment_index: int
) -> tuple[float, float]:
    """Return the first `DEFAULT_WINDOW_S`, clipped to the segment end."""
    t_start = recording.get_start_time(segment_index=segment_index)
    t_end = recording.get_end_time(segment_index=segment_index)
    return t_start, min(t_start + DEFAULT_WINDOW_S, t_end)
