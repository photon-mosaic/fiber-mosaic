"""Declarative DAG pipeline over registered processing steps.

A pipeline is a plain ``list`` of step ``dict``s::

    [
        {"function": "bleach_correction", "input": "green",
         "output": "green_bc", "params": {"degree": 3}},
        {"function": "isosbestic_correction", "output": "green_corr",
         "params": {"signal": "green_bc", "reference": "iso"}},
    ]

:func:`apply_pipeline` threads a :class:`FiberPhotometryRecordingGroup` (the
namespace of named recordings) through the steps. Because steps are referenced
by registered name and carry only JSON scalars, the pipeline round-trips
through ``json.dump`` / ``json.load`` and is reproduced by re-running it on the
raw data.

Routing rule
------------
- ``"input"`` present  -> the named band (a recording) is passed to the step.
- ``"input"`` absent   -> the whole namespace group is passed to the step.
- step returns a recording -> stored in the namespace under ``"output"``.
- step returns a group     -> its bands are merged into the namespace.
"""

from __future__ import annotations

from spikeinterface.core import BaseRecording

from fiber_mosaic.core.base import FiberPhotometryRecordingGroup
from fiber_mosaic.processing.registry import get_step, registered_steps


def validate_pipeline(pipeline: list[dict]) -> None:
    """Validate a pipeline's structure and step registration.

    Parameters
    ----------
    pipeline : list of dict
        The pipeline to validate.

    Raises
    ------
    ValueError
        If the pipeline is not a list, a step is malformed, or a step
        references an unregistered function.
    """
    if not isinstance(pipeline, list):
        raise ValueError(f"pipeline must be a list, got {type(pipeline)}")
    known = set(registered_steps())
    for index, step in enumerate(pipeline):
        if not isinstance(step, dict):
            raise ValueError(f"step {index} must be a dict, got {type(step)}")
        if "function" not in step:
            raise ValueError(f"step {index} missing required key 'function'")
        if step["function"] not in known:
            raise ValueError(
                f"step {index} references unknown function "
                f"{step['function']!r}; registered: {sorted(known)}"
            )
        for key in ("input", "output"):
            if key in step and not isinstance(step[key], str):
                raise ValueError(f"step {index} {key!r} must be a string")
        if not isinstance(step.get("params", {}), dict):
            raise ValueError(f"step {index} 'params' must be a dict")


def _seed_namespace(
    source: FiberPhotometryRecordingGroup | BaseRecording,
) -> FiberPhotometryRecordingGroup:
    """Return the initial namespace group from a group or single recording.

    Parameters
    ----------
    source : FiberPhotometryRecordingGroup or BaseRecording
        Pipeline input. A group is copied; a single recording is wrapped into
        a one-band group keyed by its color (falling back to ``"input"``).

    Returns
    -------
    FiberPhotometryRecordingGroup
        The seeded namespace.

    Raises
    ------
    TypeError
        If ``source`` is neither supported type.
    """
    if isinstance(source, FiberPhotometryRecordingGroup):
        return FiberPhotometryRecordingGroup(dict(source.items()))
    if isinstance(source, BaseRecording):
        name = (
            getattr(source, "color", None)
            or source.get_annotation("color")
            or "input"
        )
        return FiberPhotometryRecordingGroup({name: source})
    raise TypeError(
        "source must be a FiberPhotometryRecordingGroup or a BaseRecording, "
        f"got {type(source)}"
    )


def _resolve_input(
    namespace: FiberPhotometryRecordingGroup, step: dict
) -> FiberPhotometryRecordingGroup | BaseRecording:
    """Return the argument to pass to a step per the routing rule."""
    if "input" not in step:
        return namespace
    name = step["input"]
    if name not in namespace:
        raise ValueError(
            f"step {step['function']!r} reads unknown band {name!r}; "
            f"available: {namespace.colors}"
        )
    return namespace[name]


def _merge_result(
    namespace: FiberPhotometryRecordingGroup, step: dict, result: object
) -> FiberPhotometryRecordingGroup:
    """Fold a step's return value back into the namespace."""
    if isinstance(result, FiberPhotometryRecordingGroup):
        for name, recording in result.items():
            namespace = namespace.with_recording(name, recording)
        return namespace
    if isinstance(result, BaseRecording):
        if "output" not in step:
            raise ValueError(
                f"step {step['function']!r} returned a recording but has no "
                "'output' name"
            )
        return namespace.with_recording(step["output"], result)
    raise TypeError(
        f"step {step['function']!r} returned {type(result)}; expected a "
        "recording or a FiberPhotometryRecordingGroup"
    )


def apply_pipeline(
    source: FiberPhotometryRecordingGroup | BaseRecording,
    pipeline: list[dict],
) -> FiberPhotometryRecordingGroup:
    """Run a pipeline over a group or recording, returning the full namespace.

    Parameters
    ----------
    source : FiberPhotometryRecordingGroup or BaseRecording
        Pipeline input; seeds the namespace (see :func:`_seed_namespace`).
    pipeline : list of dict
        Steps to execute in order (see the module docstring).

    Returns
    -------
    FiberPhotometryRecordingGroup
        The namespace of all bands, both seeded and derived.
    """
    validate_pipeline(pipeline)
    namespace = _seed_namespace(source)
    for step in pipeline:
        function = get_step(step["function"])
        argument = _resolve_input(namespace, step)
        result = function(argument, **step.get("params", {}))
        namespace = _merge_result(namespace, step, result)
    return namespace
