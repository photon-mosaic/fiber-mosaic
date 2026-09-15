"""
Helpers for exposing preprocessor classes as functions.

SpikeInterface turns a preprocessor class into the public function users call,
via ``define_function_handling_dict_from_class``, which dispatches over a
single recording or a dict of recordings. This module adds the fiber-mosaic
equivalent, which also accepts a
:class:`~fiber_mosaic.core.base.FiberPhotometryRecordingGroup`.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable

from spikeinterface.core import BaseRecording

from fiber_mosaic.core.base import FiberPhotometryRecordingGroup


def define_function_handling_group_from_class(
    source_class: type, name: str
) -> Callable:
    """
    Expose a preprocessor class as a function that also accepts a group.

    The returned function dispatches on its first argument and returns the
    matching container, so a step reads the same whether it is applied to one
    color or to a whole session:

    - a recording -> a preprocessed recording
    - a ``dict`` of recordings -> a dict of preprocessed recordings
    - a ``FiberPhotometryRecordingGroup`` -> a new group, colors preserved

    Parameters
    ----------
    source_class : type
        The preprocessor class to wrap.
    name : str
        Public name for the resulting function.

    Returns
    -------
    Callable
        A function with ``source_class``'s signature and docstring.

    Raises
    ------
    TypeError
        If the first argument is not a recording, a dict, or a group.

    Notes
    -----
    Dispatch broadcasts the step over each color independently, so this is
    only meaningful for steps with a *single* recording input. A step that
    consumes a second band -- an isosbestic reference, say -- combines colors
    rather than mapping over them, and must take them explicitly instead of
    being broadcast.

    Unlike SpikeInterface's dict-only version, the input is located with an
    ``in`` test rather than a truthiness test, so an empty dict dispatches as
    a dict instead of falling through to the positional branch.
    """

    def source_class_or_group(*args, **kwargs):
        """Apply ``source_class`` to a recording, a dict, or a group."""
        if "recording" in kwargs:
            target = kwargs["recording"]
            rest_args = args
            rest_kwargs = {
                key: value
                for key, value in kwargs.items()
                if key != "recording"
            }
        elif args:
            target = args[0]
            rest_args = args[1:]
            rest_kwargs = kwargs
        else:
            raise TypeError(
                f"{name}() requires a recording, a dict of recordings, or a "
                "FiberPhotometryRecordingGroup as its first argument"
            )

        if isinstance(target, BaseRecording):
            return source_class(target, *rest_args, **rest_kwargs)
        if isinstance(target, FiberPhotometryRecordingGroup):
            return FiberPhotometryRecordingGroup(
                {
                    color: source_class(recording, *rest_args, **rest_kwargs)
                    for color, recording in target.items()
                }
            )
        if isinstance(target, dict):
            return {
                key: source_class(recording, *rest_args, **rest_kwargs)
                for key, recording in target.items()
            }
        raise TypeError(
            f"{name}() accepts a recording, a dict of recordings, or a "
            f"FiberPhotometryRecordingGroup, got {type(target)}"
        )

    source_class_or_group.__signature__ = inspect.signature(source_class)
    source_class_or_group.__doc__ = source_class.__doc__
    source_class_or_group.__name__ = name

    return source_class_or_group
