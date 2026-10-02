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

Examples
--------
>>> from spikeinterface.preprocessing import apply_preprocessing_pipeline
>>> from fiber_mosaic.processing.pipeline import FiberPreprocessingPipeline
>>> pipeline = FiberPreprocessingPipeline(
...     [{"name": "bandpass_filter", "params": {"freq_min": 0.2}}]
... )
>>> processed = apply_preprocessing_pipeline(group, pipeline)
"""

from __future__ import annotations

from spikeinterface.preprocessing.pipeline import PreprocessingPipeline

from fiber_mosaic.processing import processor_dict

#: Step name -> step function, for every step fiber-mosaic registers in
#: ``processor_dict``. Keyed by the public function name, as upstream does.
pp_names_to_functions = {
    function.__name__: function for function in processor_dict.values()
}


class FiberPreprocessingPipeline(PreprocessingPipeline):
    """
    An ordered list of fiber photometry preprocessing steps.

    SpikeInterface's ``PreprocessingPipeline`` with fiber-mosaic's steps, so
    the result keeps the fiber API, per-fiber timestamps and photometry
    defaults.

    Parameters
    ----------
    preprocessor_list_or_dict : list or dict
        Ordered steps, each ``{"name": str, "params": dict}``; ``params``
        may be omitted. A ``{name: params}`` dict works but upstream
        deprecates it for removal in 0.107.0.

    Notes
    -----
    Pass this object to ``apply_preprocessing_pipeline``, never a bare list:
    upstream turns a list into its own pipeline and silently applies the
    ephys steps instead.

    Parameters go under ``"params"``. Upstream's own docstring says
    ``"kwargs"``, which is read as no parameters at all.

    A step taking a second recording names an earlier step's output as
    ``"pipeline[step_name]"`` (``"raw"`` being the input). Upstream only
    substitutes into parameters whose name contains ``"recording"``.
    """

    function_names_to_functions = pp_names_to_functions


__all__ = ["FiberPreprocessingPipeline", "pp_names_to_functions"]
