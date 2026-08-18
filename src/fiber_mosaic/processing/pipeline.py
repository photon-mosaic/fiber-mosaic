"""Declarative DAG pipeline over registered processing steps.

A pipeline is a plain ``list`` of step ``dict``s::

    [
        {"function": "bleach_correction", "input": "green",
         "output": "green_bc", "params": {"degree": 3}},
        {"function": "isosbestic_correction",
         "inputs": {"signal": "green_bc", "reference": "iso"},
         "output": "green_corr", "params": {"method": "subtract"}},
    ]

:func:`apply_pipeline` threads a *workspace* -- a plain ``dict`` mapping entry
name to recording -- through the steps. Because steps are referenced by
registered name and carry only strings and JSON scalars, a pipeline round-trips
through ``json.dump`` / ``json.load`` and is reproduced by re-running it on the
raw data.

Wiring
------
Inputs and outputs follow one rule in both directions: **keys are the step's
own input and output names, values are workspace entry names.**

- ``"input": "green"`` -- sugar, valid when the step has a single input.
- ``"inputs": {"signal": "green", "reference": "iso"}`` -- the general form;
  keys must cover exactly the step's input names.
- ``"output": "green_bc"`` -- sugar, valid when the step returns one array.
- ``"outputs": {"low": "low_freq", "high": "high_freq"}`` -- the general form
  for a step returning a dict of arrays.

Because every entry name is declared, the graph is known before execution:
:func:`validate_pipeline` checks input and output names, parameter names, and
that each step only reads entries produced earlier.
"""

from __future__ import annotations

from dataclasses import dataclass

from spikeinterface.core import BaseRecording

from fiber_mosaic.core.base import FiberPhotometryRecordingGroup
from fiber_mosaic.processing.registry import (
    ArrayStep,
    get_step,
    registered_steps,
)


@dataclass(frozen=True)
class _Step:
    """A pipeline step with its wiring parsed into mappings."""

    function: str
    inputs: dict[str, str]
    output: str | None
    outputs: dict[str, str] | None
    params: dict


def _require_entry_mapping(value: object, index: int, key: str) -> dict:
    """Validate ``value`` as a ``{step name: entry name}`` mapping of str."""
    if not isinstance(value, dict):
        raise ValueError(f"step {index} {key!r} must be a dict")
    for step_name, entry_name in value.items():
        if not isinstance(step_name, str) or not isinstance(entry_name, str):
            raise ValueError(
                f"step {index} {key!r} must map step names to entry names, "
                "both strings"
            )
    return dict(value)


def _parse_inputs(step: dict, index: int, spec) -> dict[str, str]:
    """Return the step's ``{input name: entry name}`` inputs, either form."""
    singular = step.get("input")
    plural = step.get("inputs")
    if (singular is None) == (plural is None):
        raise ValueError(
            f"step {index} must declare exactly one of 'input' or 'inputs'"
        )
    if plural is not None:
        inputs = _require_entry_mapping(plural, index, "inputs")
    else:
        if not isinstance(singular, str):
            raise ValueError(f"step {index} 'input' must be a string")
        if len(spec.input_names) != 1:
            raise ValueError(
                f"step {index} uses 'input' but {spec.name!r} has "
                f"{len(spec.input_names)} input names "
                f"{list(spec.input_names)}; use 'inputs'"
            )
        inputs = {spec.input_names[0]: singular}
    expected = set(spec.input_names)
    if set(inputs) != expected:
        raise ValueError(
            f"step {index} ({spec.name!r}) declares inputs "
            f"{sorted(inputs)} but the step's inputs are {sorted(expected)}"
        )
    return inputs


def _parse_outputs(
    step: dict, index: int
) -> tuple[str | None, dict[str, str] | None]:
    """Return the singular output entry, or the ``{name: entry}`` map."""
    singular = step.get("output")
    plural = step.get("outputs")
    if (singular is None) == (plural is None):
        raise ValueError(
            f"step {index} must declare exactly one of 'output' or 'outputs'"
        )
    if plural is not None:
        return None, _require_entry_mapping(plural, index, "outputs")
    if not isinstance(singular, str):
        raise ValueError(f"step {index} 'output' must be a string")
    return singular, None


def _parse_step(raw: object, index: int) -> _Step:
    """Parse one raw step dict, validating its wiring."""
    if not isinstance(raw, dict):
        raise ValueError(f"step {index} must be a dict, got {type(raw)}")
    if "function" not in raw:
        raise ValueError(f"step {index} missing required key 'function'")
    name = raw["function"]
    if name not in set(registered_steps()):
        raise ValueError(
            f"step {index} references unknown function {name!r}; "
            f"registered: {registered_steps()}"
        )
    spec = get_step(name)
    params = raw.get("params", {})
    if not isinstance(params, dict):
        raise ValueError(f"step {index} 'params' must be a dict")
    if isinstance(spec, ArrayStep):
        unknown = set(params) - spec.param_names
        if unknown:
            raise ValueError(
                f"step {index} ({name!r}) got unknown parameter(s) "
                f"{sorted(unknown)}; accepted: {sorted(spec.param_names)}"
            )
    output, outputs = _parse_outputs(raw, index)
    return _Step(
        function=name,
        inputs=_parse_inputs(raw, index, spec),
        output=output,
        outputs=outputs,
        params=dict(params),
    )


def _output_entries(step: _Step) -> list[str]:
    """Return the entry names a step writes into the workspace."""
    if step.output is not None:
        return [step.output]
    return list(step.outputs.values())


def _parse_pipeline(
    pipeline: object, known_entries: set[str] | None = None
) -> list[_Step]:
    """Parse and validate a whole pipeline, returning its steps.

    Parameters
    ----------
    pipeline : object
        The pipeline to validate; must be a list of step dicts.
    known_entries : set of str or None, default: None
        Entry names available before the first step. When given, every step is
        checked to read only entries that exist by that point.

    Returns
    -------
    list of _Step
        The parsed steps, in order.

    Raises
    ------
    ValueError
        If the pipeline is malformed, references an unregistered function,
        mis-declares names or parameters, or reads an unavailable entry.
    """
    if not isinstance(pipeline, list):
        raise ValueError(f"pipeline must be a list, got {type(pipeline)}")
    available = None if known_entries is None else set(known_entries)
    steps = []
    for index, raw in enumerate(pipeline):
        step = _parse_step(raw, index)
        if available is not None:
            for input_name, entry_name in step.inputs.items():
                if entry_name not in available:
                    raise ValueError(
                        f"step {index} ({step.function!r}) reads unknown "
                        f"entry {entry_name!r} for input {input_name!r}; "
                        f"available: {sorted(available)}"
                    )
            available.update(_output_entries(step))
        steps.append(step)
    return steps


def validate_pipeline(
    pipeline: object, known_entries: set[str] | None = None
) -> None:
    """Validate a pipeline's structure, wiring and step registration.

    Parameters
    ----------
    pipeline : object
        The pipeline to validate.
    known_entries : set of str or None, default: None
        Entry names available before the first step. When given, entry
        availability is checked too.

    Raises
    ------
    ValueError
        If the pipeline is invalid.
    """
    _parse_pipeline(pipeline, known_entries=known_entries)


def _seed_workspace(
    source: dict[str, BaseRecording]
    | FiberPhotometryRecordingGroup
    | BaseRecording,
) -> dict[str, BaseRecording]:
    """Return the initial workspace for a pipeline run.

    Seeding is how colors acquire the roles a pipeline refers to. Passing a
    ``dict`` states that binding explicitly, which keeps a pipeline reusable
    across groups whose colors are named differently. A group is accepted as a
    convenience and seeds entries named after its colors.

    Parameters
    ----------
    source : dict, FiberPhotometryRecordingGroup or BaseRecording
        A mapping of workspace name to recording (the explicit form); or a
        group, whose colors become the workspace names; or a single recording,
        keyed by its color (falling back to ``"input"``).

    Returns
    -------
    dict
        Mapping of entry name to recording.

    Raises
    ------
    TypeError
        If ``source`` is not a supported type, or a mapping entry is not a
        name-to-recording pair.
    """
    if isinstance(source, dict):
        for name, recording in source.items():
            if not isinstance(name, str):
                raise TypeError(
                    f"entry names must be strings, got {type(name)}"
                )
            if not isinstance(recording, BaseRecording):
                raise TypeError(
                    f"workspace entry {name!r} must be a recording, got "
                    f"{type(recording)}"
                )
        return dict(source)
    if isinstance(source, FiberPhotometryRecordingGroup):
        return dict(source.items())
    if isinstance(source, BaseRecording):
        name = (
            getattr(source, "color", None)
            or source.get_annotation("color")
            or "input"
        )
        return {name: source}
    raise TypeError(
        "source must be a mapping of names to recordings, a "
        "FiberPhotometryRecordingGroup, or a BaseRecording, got "
        f"{type(source)}"
    )


def _bind_outputs(
    workspace: dict[str, BaseRecording],
    step: _Step,
    results: dict[str, BaseRecording],
    index: int,
) -> None:
    """Store a step's results in the workspace under the declared names."""
    if step.output is not None:
        if len(results) != 1:
            raise ValueError(
                f"step {index} ({step.function!r}) declares a single "
                f"'output' but returned {sorted(results)}; use 'outputs'"
            )
        workspace[step.output] = next(iter(results.values()))
        return
    missing = set(step.outputs) - set(results)
    unexpected = set(results) - set(step.outputs)
    if missing or unexpected:
        raise ValueError(
            f"step {index} ({step.function!r}) declared outputs "
            f"{sorted(step.outputs)} but returned {sorted(results)}"
        )
    for output_name, entry_name in step.outputs.items():
        workspace[entry_name] = results[output_name]


def apply_pipeline(
    source: dict[str, BaseRecording]
    | FiberPhotometryRecordingGroup
    | BaseRecording,
    pipeline: list[dict],
) -> dict[str, BaseRecording]:
    """Run a pipeline over seeded recordings, returning the workspace.

    Pass a ``dict`` to bind colors to the names a pipeline expects::

        apply_pipeline(
            {"raw_signal": group["green"], "raw_reference": group["iso"]},
            pipeline,
        )

    A group may be passed instead, seeding entries named after its colors.

    Parameters
    ----------
    source : dict, FiberPhotometryRecordingGroup or BaseRecording
        Pipeline input; seeds the workspace (see :func:`_seed_workspace`).
    pipeline : list of dict
        Steps to execute in order (see the module docstring).

    Returns
    -------
    dict
        Mapping of entry name to recording, both seeded and derived.
    """
    workspace = _seed_workspace(source)
    steps = _parse_pipeline(pipeline, known_entries=set(workspace))
    for index, step in enumerate(steps):
        spec = get_step(step.function)
        inputs = {
            input_name: workspace[entry_name]
            for input_name, entry_name in step.inputs.items()
        }
        results = spec.run(inputs, dict(step.params))
        _bind_outputs(workspace, step, results, index)
    return workspace
