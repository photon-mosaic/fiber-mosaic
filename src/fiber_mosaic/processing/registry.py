"""Registry of named processing steps for pipeline lookup.

A processing step is any callable ``func(x, **params) -> y`` where ``x`` and
``y`` are each a :class:`~fiber_mosaic.core.base.BaseFiberPhotometryExtractor`
or a :class:`~fiber_mosaic.core.base.FiberPhotometryRecordingGroup`. Steps are
referenced by name from a pipeline, so the pipeline stays a plain,
JSON-serializable list of dicts.

Contributors register a step with the :func:`pipeline_step` decorator. By
default (``by_segment=True``) the decorator maps the function over each
segment: the function receives one segment's ``(n_samples, n_fibers)`` traces
array and returns a same-length array, and the decorator reassembles the
result into a recording carrying the input's time base and metadata. A step
that needs the whole recording/group at once (e.g. to read sample times, or to
consume a second band) sets ``by_segment=False`` and receives the routed input
directly; it may return a recording/group, or raw traces that get wrapped.
Existing callables or classes are registered as-is with :func:`register_step`.
"""

from __future__ import annotations

import functools
from collections.abc import Callable

import numpy as np
from spikeinterface.core import BaseRecording

from fiber_mosaic.processing.helpers import map_segments, recording_like

#: Registry mapping step name -> processing callable.
_STEP_REGISTRY: dict[str, Callable] = {}


def register_step(
    name: str, function: Callable, overwrite: bool = False
) -> Callable:
    """Register ``function`` under ``name`` for pipeline lookup.

    Parameters
    ----------
    name : str
        Name used to reference the step from a pipeline.
    function : Callable
        The processing callable to register, unchanged.
    overwrite : bool, default: False
        If False, re-registering an existing name raises; if True, the
        previous entry is replaced.

    Returns
    -------
    Callable
        The registered function, unchanged.

    Raises
    ------
    ValueError
        If ``name`` is already registered and ``overwrite`` is False.
    """
    if name in _STEP_REGISTRY and not overwrite:
        raise ValueError(f"step {name!r} is already registered")
    _STEP_REGISTRY[name] = function
    return function


def _as_recording(source: object, result: object) -> object:
    """Wrap array/list step output as a recording; pass others through.

    Parameters
    ----------
    source : object
        The step's input argument, used as the metadata reference when
        wrapping raw traces.
    result : object
        The step's return value.

    Returns
    -------
    object
        ``result`` unchanged when it is already a recording or group;
        otherwise a recording built from the returned traces.

    Raises
    ------
    TypeError
        If ``result`` is raw traces but ``source`` is not a single recording.
    """
    if isinstance(result, (np.ndarray, list, tuple)):
        if not isinstance(source, BaseRecording):
            raise TypeError(
                "a step that returns traces must receive a single recording "
                "as input, not a group"
            )
        return recording_like(source, result)
    return result


def pipeline_step(
    name: str, overwrite: bool = False, by_segment: bool = True
) -> Callable:
    """Register a processing step under ``name`` (decorator form).

    With ``by_segment=True`` (default) the function is called once per segment
    with that segment's ``(n_samples, n_fibers)`` traces array and must return
    a same-length array; the decorator maps over segments and reassembles a
    recording carrying the input's time base and metadata. With
    ``by_segment=False`` the function receives the routed input (a recording or
    group) directly and may return a recording/group, or raw traces that are
    wrapped. The wrapped function is returned, so it stays directly callable.

    Parameters
    ----------
    name : str
        Name used to reference the step from a pipeline.
    overwrite : bool, default: False
        Passed through to :func:`register_step`.
    by_segment : bool, default: True
        If True, map the function over each segment (see above). Set False for
        steps that need the whole recording/group at once.

    Returns
    -------
    Callable
        A decorator that registers and returns the wrapped function.
    """

    def decorator(function: Callable) -> Callable:
        """Wrap ``function`` per ``by_segment``, then register it."""

        @functools.wraps(function)
        def wrapper(source, *args, **kwargs):
            """Run the step, mapping over segments when requested."""
            if by_segment:
                if not isinstance(source, BaseRecording):
                    raise TypeError(
                        "a by_segment step needs a single recording, not a "
                        "group; use by_segment=False for group-level steps"
                    )
                return map_segments(
                    source,
                    lambda traces: function(traces, *args, **kwargs),
                )
            return _as_recording(source, function(source, *args, **kwargs))

        register_step(name, wrapper, overwrite=overwrite)
        return wrapper

    return decorator


def get_step(name: str) -> Callable:
    """Return the processing callable registered under ``name``.

    Parameters
    ----------
    name : str
        A registered step name.

    Returns
    -------
    Callable
        The registered processing callable.

    Raises
    ------
    KeyError
        If ``name`` is not registered.
    """
    if name not in _STEP_REGISTRY:
        raise KeyError(
            f"unknown step {name!r}; registered: {registered_steps()}"
        )
    return _STEP_REGISTRY[name]


def registered_steps() -> list[str]:
    """Return the sorted names of all registered steps."""
    return sorted(_STEP_REGISTRY)
