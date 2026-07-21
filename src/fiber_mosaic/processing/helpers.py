"""Helpers for writing processing steps with minimal boilerplate.

A processing step must return a recording, not a bare array. These helpers
let a contributor compute on plain numpy arrays and hand the result back as a
:class:`~fiber_mosaic.core.base.BaseFiberPhotometryExtractor` that carries the
source recording's time base, fibers, and color.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from spikeinterface.core import BaseRecording
from spikeinterface.core.numpyextractors import NumpyRecordingSegment

from fiber_mosaic.core.base import BaseFiberPhotometryExtractor


def recording_like(
    reference: BaseRecording,
    traces: np.ndarray | Sequence[np.ndarray],
    color: str | None = None,
) -> BaseFiberPhotometryExtractor:
    """Build an in-memory fiber recording carrying ``reference``'s metadata.

    Parameters
    ----------
    reference : BaseRecording
        Recording whose sampling rate, fiber ids, color, and explicit
        per-fiber times are copied onto the result.
    traces : np.ndarray or sequence of np.ndarray
        New traces: one ``(n_samples, n_fibers)`` array for a single segment,
        or one array per segment.
    color : str or None, default: None
        Output color; falls back to ``reference``'s color attribute or
        ``"color"`` annotation.

    Returns
    -------
    BaseFiberPhotometryExtractor
        A new recording holding ``traces``.
    """
    if isinstance(traces, np.ndarray):
        traces_list = [traces]
    else:
        traces_list = [np.asarray(segment) for segment in traces]
    resolved_color = (
        color
        or getattr(reference, "color", None)
        or reference.get_annotation("color")
    )
    sampling_frequency = reference.get_sampling_frequency()
    new = BaseFiberPhotometryExtractor(
        sampling_frequency=sampling_frequency,
        fiber_ids=reference.get_channel_ids(),
        color=resolved_color,
        dtype=traces_list[0].dtype,
    )
    for segment_traces in traces_list:
        new.add_segment(
            NumpyRecordingSegment(
                traces=segment_traces,
                sampling_frequency=sampling_frequency,
                t_start=None,
            )
        )
    # carry explicit per-fiber times when the reference has them
    if hasattr(reference, "has_fiber_times"):
        for segment_index in range(new.get_num_segments()):
            if reference.has_fiber_times(segment_index=segment_index):
                new.set_times(
                    reference.get_fiber_times(segment_index=segment_index),
                    segment_index=segment_index,
                    with_warning=False,
                )
    return new


def map_segments(
    recording: BaseRecording,
    function: Callable,
    *others: BaseRecording,
    **params,
) -> BaseFiberPhotometryExtractor:
    """Apply an array transform to each segment, preserving the time base.

    ``function`` maps one segment's ``(n_samples, n_fibers)`` traces (plus the
    matching segment of each recording in ``others``) to a same-length array;
    it is applied independently per segment and the results are wrapped with
    :func:`recording_like` against ``recording``.

    Parameters
    ----------
    recording : BaseRecording
        Primary recording whose segments are transformed and whose metadata
        the result carries.
    function : Callable
        Transform ``function(traces, *other_traces, **params) -> traces``.
    *others : BaseRecording
        Additional recordings; each one's matching segment is passed
        positionally after ``traces``. Must share ``recording``'s segments.
    **params
        Keyword arguments forwarded to ``function``.

    Returns
    -------
    BaseFiberPhotometryExtractor
        The transformed recording.
    """
    outputs = []
    for index in range(recording.get_num_segments()):
        segments = [recording.get_traces(segment_index=index)]
        segments.extend(
            other.get_traces(segment_index=index) for other in others
        )
        outputs.append(np.asarray(function(*segments, **params)))
    return recording_like(recording, outputs)
