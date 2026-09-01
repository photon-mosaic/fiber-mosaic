from spikeinterface.preprocessing.pipeline import ABCPipeline

from fiber_mosaic.core import BaseFiberPhotometryExtractor, FiberPhotometryRecordingGroup

from fiber_mosaic.processing import processor_dict, _all_processer_dict

pp_names_to_functions = {preprocessor.__name__: preprocessor for preprocessor in processor_dict.values()}
pp_names_to_classes = {pp_function.__name__: pp_class for pp_class, pp_function in _all_processer_dict.items()}


class PreprocessingPipeline(ABCPipeline):
    """
    A preprocessing pipeline, containing ordered preprocessing steps.

    Parameters
    ----------
    preprocessor_list_or_dict : dict or list
        Dictionary or list containing preprocessing steps and their kwargs

    Examples
    --------
    Generate a `PreprocessingPipeline` containing a `bandpass_filter` then a
    `common_reference` step. Then apply this to a recording

    >>> from spikeinterface.preprocessing import PreprocessingPipeline
    >>> preprocessor_dict = {'bandpass_filter': {'freq_max': 3000}, 'common_reference': {}}
    >>> my_pipeline = PreprocessingPipeline(preprocessor_dict)
    PreprocessingPipeline:  Raw Recording → bandpass_filter → common_reference → Preprocessed Recording
    >>> my_pipeline._apply(recording)

    """
    function_names_to_functions = pp_names_to_functions
    function_names_to_classes = pp_names_to_classes


def apply_preprocessing_pipeline(
    recording_or_group: BaseFiberPhotometryExtractor | FiberPhotometryRecordingGroup, pipeline: PreprocessingPipeline | list | dict, apply_precomputed_kwargs=True
):
    """
    Creates a preprocessed recording by applying the preprocessing steps in
    `pipeline` to `recording`.

    Parameters
    ----------
    recording_or_group : BaseFiberPhotometryExtractor | FiberPhotometryRecordingGroup
        The initial recording or group
    pipeline : PreprocessingPipeline | list | dict
        Dictionary containing preprocessing steps and their kwargs, a list of preprocessing steps, or a pipeline object.
        If None, the original recording is returned.
    apply_precomputed_kwargs : Bool, default: True
        Some preprocessing steps (e.g. Whitening) contain arguments which are computed
        during preprocessing. If True, we use the arguments which have already been
        computed. If False, we recompute them on application of the pipeline.

    Returns
    -------
    preprocessed_recording : BaseFiberPhotometryExtractor | FiberPhotometryRecordingGroup
        Preprocessed recording or group

    Examples
    --------
    Create a preprocessed recording from a generated recording and a preprocessing pipeline

    >>> from spikeinterface.preprocessing import create_preprocessed
    >>> from spikeinterface.generation import generate_recording
    >>> recording = generate_recording()
    >>> pipeline = [{'name': 'bandpass_filter', 'kwargs': {'freq_max': 3000}}, {'name': 'common_reference', 'kwargs': {}}]
    >>> preprocessed_recording = apply_preprocessing_pipeline(recording, pipeline)
    """

    if isinstance(pipeline, PreprocessingPipeline):
        pipeline = pipeline
    elif isinstance(pipeline, dict):
        pipeline = PreprocessingPipeline(pipeline)
    elif isinstance(pipeline, list):
        pipeline = PreprocessingPipeline(pipeline)
    else:
        raise TypeError("`pipeline` must be a `PreprocessingPipeline`, a list, or a dict")

    preprocessed_recording = pipeline._apply(recording_or_group, apply_precomputed_kwargs)
    return preprocessed_recording