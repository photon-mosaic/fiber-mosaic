"""
Declarative preprocessing pipelines for fiber photometry.

A pipeline is an ordered list of named steps and their parameters::

    [
        {"name": "bandpass_filter", "params": {"freq_min": 0.2}},
        {"name": "zscore", "params": {"seed": 0}},
    ]

Plain strings and JSON scalars, so the list round-trips through
``json.dump``/``json.load`` and an analysis can record what it ran.

Only the step registry is fiber-mosaic's. SpikeInterface's
``PreprocessingPipeline`` parses the steps and applies them, and
``apply_preprocessing_pipeline`` runs it; both resolve names through a
``function_names_to_functions`` class attribute that upstream documents as
the subclass hook. Subclassing their concrete class rather than
``BasePipeline`` means their apply function accepts ours unchanged.

Groups work unmodified: the pipeline passes its input straight to each step,
and every step in :mod:`fiber_mosaic.processing.from_spikeinterface` already
dispatches over a recording, a dict, or a ``FiberPhotometryRecordingGroup``.
For different steps per recording, with steps that use another recording's
intermediate output, see :func:`apply_processing_pipeline`.

Examples
--------
>>> from fiber_mosaic.processing.pipeline import (
...     FiberProcessingPipeline, apply_processing_pipeline
... )
>>> pipeline = FiberProcessingPipeline(
...     [{"name": "bandpass_filter", "params": {"freq_min": 0.2}}]
... )
>>> processed = apply_processing_pipeline(group, pipeline)
"""

from __future__ import annotations

from spikeinterface.preprocessing.pipeline import PreprocessingPipeline

from fiber_mosaic.core import (
    BaseFiberPhotometryExtractor,
    FiberPhotometryRecordingGroup,
)
from fiber_mosaic.processing import processor_dict

#: Step name -> step function, for every step fiber-mosaic registers in
#: ``processor_dict``. Keyed by the public function name, as upstream does.
pp_names_to_functions = {
    function.__name__: function for function in processor_dict.values()
}

def available_processing_steps() -> list[str]:
    """Return a list of all available fiber-mosaic processing step names."""
    return list(pp_names_to_functions.keys())


class FiberProcessingPipeline(PreprocessingPipeline):
    """
    An ordered list of fiber photometry preprocessing steps.

    SpikeInterface's ``PreprocessingPipeline`` with fiber-mosaic's steps, so
    the result keeps the fiber API and photometry defaults.

    Parameters
    ----------
    preprocessor_list_or_dict : list or dict
        Ordered steps, each ``{"name": str, "params": dict}``; ``params``
        may be omitted. A ``{name: params}`` dict works but upstream
        deprecates it for removal in 0.107.0.

    Notes
    -----
    Pass this object to ``apply_processing_pipeline``, never a bare list:
    upstream turns a list into its own pipeline and silently applies the
    ephys steps instead.

    Parameters go under ``"params"``. Upstream's own docstring says
    ``"kwargs"``, which is read as no parameters at all.

    A step taking a second recording names an earlier step's output as
    ``"pipeline[step_name]"`` (``"raw"`` being the input). Upstream only
    substitutes into parameters whose name contains ``"recording"``.
    """

    function_names_to_functions = pp_names_to_functions


def _resolve_params(step, local, shared):
    """Substitute ``"pipeline[x]"`` params; None if a reference is unbuilt."""
    params = step["params"].copy()
    for key, value in params.items():
        if not (isinstance(value, str) and "pipeline[" in value):
            continue
        if "recording" not in key or key in ("recording", "parent_recording"):
            raise ValueError(
                f"Cannot substitute a recording for argument '{key}' of step "
                f"'{step['name']}'."
            )
        ref = value.split("pipeline[")[-1].split("]")[0]
        if ref in local:
            params[key] = local[ref]
        elif ref in shared:
            params[key] = shared[ref]
        else:
            return None
    params.pop("recording", None)
    params.pop("parent_recording", None)
    return params


def _run_pipelines(recordings, pipelines):
    """Run each recording's steps, interleaved so references resolve."""
    steps = {name: pipelines[name].preprocessor_list for name in pipelines}
    cursors = dict.fromkeys(pipelines, 0)
    current = dict(recordings)
    local = {name: {"raw": recordings[name]} for name in pipelines}
    shared = {}
    function_names_to_functions = (
        FiberProcessingPipeline.function_names_to_functions
    )

    while any(cursors[name] < len(steps[name]) for name in pipelines):
        progressed = False
        for name in pipelines:
            while cursors[name] < len(steps[name]):
                step = steps[name][cursors[name]]
                params = _resolve_params(step, local[name], shared)
                if params is None:
                    break
                function = function_names_to_functions[step["name"]]
                current[name] = function(current[name], **params)
                local[name][step["name"]] = current[name]
                output_name = step.get("output_name")
                if output_name is not None:
                    if output_name in shared:
                        raise ValueError(
                            f"Duplicate output_name '{output_name}'."
                        )
                    shared[output_name] = current[name]
                cursors[name] += 1
                progressed = True
        if not progressed:
            stuck = {
                name: steps[name][cursors[name]]
                for name in pipelines
                if cursors[name] < len(steps[name])
            }
            raise ValueError(
                "Cannot resolve the 'pipeline[...]' references of steps "
                f"{stuck}: "
                "they name an unknown output or form a cycle."
            )
    return current


def _as_pipeline(pipeline):
    if isinstance(pipeline, FiberProcessingPipeline):
        return pipeline
    # a bare list given to upstream would silently resolve to the ephys steps
    return FiberProcessingPipeline(pipeline)


def apply_processing_pipeline(
    recording_or_group: BaseFiberPhotometryExtractor
    | FiberPhotometryRecordingGroup,
    pipeline: list | dict | FiberProcessingPipeline,
):
    """
    Apply a fiber photometry processing pipeline to a recording or group.

    The pipeline can be a list of steps or a ``FiberProcessingPipeline``,
    applied to a recording or to every recording of a group. For a group it
    can also be a dict mapping recording names to their own list of steps;
    recordings without an entry are returned unchanged.

    A step can set ``"output_name"`` so any later step, of any recording,
    can take its output as ``"pipeline[output_name]"`` (in a parameter whose
    name contains ``"recording"``). Within one recording, ``"pipeline[raw]"``
    and ``"pipeline[step_name]"`` also work, as in SpikeInterface. Steps run
    in whatever order these references require.

    Example
    -------
    > bp = {"name": "bandpass_filter", "params": {"freq_min": 0.2}}
    > pipeline_dict = {
    >     "green": [
    >         bp,
    >         {"name": "isosbestic_correction",
    >          "params": {"reference_recording": "pipeline[red_corrected]"}},
    >     ],
    >     "red": [
    >         bp,
    >         {"name": "isosbestic_correction",
    >          "params": {"reference_recording": "pipeline[iso_bandpass]"},
    >          "output_name": "red_corrected"},
    >     ],
    >     "iso": [{**bp, "output_name": "iso_bandpass"}],
    > }
    >
    > processed_group = apply_processing_pipeline(group, pipeline_dict)

    Parameters
    ----------
    recording_or_group : BaseFiberPhotometryExtractor | RecordingGroup
        The recording or ``FiberPhotometryRecordingGroup`` to process.
    pipeline : list | dict | FiberProcessingPipeline
        The processing pipeline to apply.

    Returns
    -------
    BaseFiberPhotometryExtractor | FiberPhotometryRecordingGroup
        The processed recording or group after applying the pipeline.
    """
    if not isinstance(recording_or_group, FiberPhotometryRecordingGroup):
        if isinstance(pipeline, dict):
            raise TypeError(
                "A dict pipeline maps group recording names to steps; "
                "pass a list "
                "or a FiberProcessingPipeline for a single recording."
            )
        # a single recording has no name; "recording" is just its key here
        recordings = {"recording": recording_or_group}
        return _run_pipelines(
            recordings, {"recording": _as_pipeline(pipeline)}
        )["recording"]

    if not isinstance(pipeline, dict):
        from spikeinterface.preprocessing import (
            apply_preprocessing_pipeline as si_apply,
        )

        # every step already dispatches over a group
        return si_apply(recording_or_group, _as_pipeline(pipeline))

    # group with a dict pipeline: every key must be a group recording
    unknown = set(pipeline) - set(recording_or_group.names)
    if unknown:
        raise ValueError(
            f"Pipeline names {sorted(unknown)} are not recordings of the "
            "group "
            f"{recording_or_group.names}."
        )
    recordings = {
        name: recording_or_group[name] for name in recording_or_group.names
    }
    pipelines = {name: _as_pipeline(steps) for name, steps in pipeline.items()}
    return FiberPhotometryRecordingGroup(_run_pipelines(recordings, pipelines))


__all__ = [
    "FiberProcessingPipeline",
    "pp_names_to_functions",
    "apply_processing_pipeline",
]
