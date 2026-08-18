"""Registry and execution of individual processing steps.

A processing step is a plain function over numpy arrays::

    def step(primary, times, *extra_arrays, **params) -> array | dict

:func:`processing_step` registers it and wraps it so it can be called with
*recordings* instead of arrays: the wrapper reads traces and times off the
inputs, handles segments, padding and reassembly, and hands back recordings
carrying the primary input's metadata.

Two axes are declared on the decorator:

``contiguous``
    ``True`` -- the function needs temporally continuous samples, so it is run
    once per segment. ``False`` -- it tolerates temporal gaps, so it is run
    once over all segments concatenated, with the real timestamps.
``margin``
    Per-segment padding, applied and trimmed by the runner. Only meaningful
    with ``contiguous=True``.

Input names are *derived from the signature*: the first positional parameter
is the primary input, the second is always the times array, further positional
parameters are additional input names, and keyword-only parameters are step
parameters. Nothing about inputs or outputs is declared on the decorator.

Times are read via ``get_fiber_times`` -- a ``(n_samples, n_fibers)`` matrix,
one timestamp per measured value -- so steps needing exact timing work per
fiber. Upstream spikeinterface preprocessors are registered unchanged with
:func:`register_step`.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from spikeinterface.core import BaseRecording
from spikeinterface.core.numpyextractors import NumpyRecordingSegment

from fiber_mosaic.core.base import BaseFiberPhotometryExtractor

#: Registry mapping step name -> step spec.
_STEP_REGISTRY: dict[str, ArrayStep | RecordingStep] = {}


def _has_fiber_times(recording: BaseRecording, segment_index: int) -> bool:
    """Return True when ``recording`` stores explicit per-fiber times."""
    return hasattr(recording, "has_fiber_times") and recording.has_fiber_times(
        segment_index=segment_index
    )


def concatenate_segments(
    reference: BaseRecording,
    traces_list: Sequence[np.ndarray],
) -> BaseFiberPhotometryExtractor:
    """Build a fiber recording from per-segment traces, carrying metadata.

    Parameters
    ----------
    reference : BaseRecording
        Recording whose sampling rate, fiber ids, color and per-segment times
        are copied onto the result.
    traces_list : sequence of np.ndarray
        One ``(n_samples, n_fibers)`` array per output segment.

    Returns
    -------
    BaseFiberPhotometryExtractor
        A recording with one segment per array in ``traces_list``.
    """
    color = getattr(reference, "color", None) or reference.get_annotation(
        "color"
    )
    sampling_frequency = reference.get_sampling_frequency()
    new = BaseFiberPhotometryExtractor(
        sampling_frequency=sampling_frequency,
        fiber_ids=reference.get_channel_ids(),
        color=color,
        dtype=traces_list[0].dtype,
    )
    for index, segment_traces in enumerate(traces_list):
        new.add_segment(
            NumpyRecordingSegment(
                traces=segment_traces,
                sampling_frequency=sampling_frequency,
                t_start=None,
            )
        )
        if _has_fiber_times(reference, index):
            new.set_times(
                reference.get_fiber_times(segment_index=index),
                segment_index=index,
                with_warning=False,
            )
    return new


def _times_for(recording: BaseRecording, segment_index: int) -> np.ndarray:
    """Return one segment's times as ``(n_samples, n_fibers)``.

    Prefers per-fiber times. Plain spikeinterface recordings (for instance the
    output of an upstream preprocessor) have none, so their 1-D time vector is
    broadcast across fibers.

    Parameters
    ----------
    recording : BaseRecording
        Recording to read times from.
    segment_index : int
        Segment to read.

    Returns
    -------
    np.ndarray
        Times, shape ``(n_samples, n_fibers)``.
    """
    if hasattr(recording, "get_fiber_times"):
        return np.asarray(
            recording.get_fiber_times(segment_index=segment_index)
        )
    times = np.asarray(recording.get_times(segment_index=segment_index))
    return np.broadcast_to(
        times[:, np.newaxis], (times.size, recording.get_num_channels())
    ).copy()


def validate_timeline(recording: BaseRecording) -> None:
    """Check that segments describe disjoint, ordered time intervals.

    Every fiber's segments must advance strictly in time: segment ``i + 1``
    must start after segment ``i`` ends. Segments are never silently
    reordered, since out-of-order segments indicate a loading problem.

    Parameters
    ----------
    recording : BaseRecording
        Recording whose segment timeline is checked.

    Raises
    ------
    ValueError
        If any fiber's consecutive segments overlap or are out of order. A
        multi-segment recording with no explicit times trips this, because
        every segment's times then start at zero.
    """
    if recording.get_num_segments() < 2:
        return
    previous_last = None
    for index in range(recording.get_num_segments()):
        times = _times_for(recording, index)
        first, last = times[0], times[-1]
        if previous_last is not None and np.any(first <= previous_last):
            fibers = np.flatnonzero(first <= previous_last).tolist()
            raise ValueError(
                f"segment {index} starts at or before segment {index - 1} "
                f"ends, for fiber index/indices {fibers}; segments must "
                "describe disjoint, ordered time intervals (a multi-segment "
                "recording with no explicit times always fails this)"
            )
        previous_last = last


def _check_compatible(inputs: dict[str, BaseRecording]) -> None:
    """Check that all input recordings share fibers, segments and samples."""
    items = list(inputs.items())
    reference_name, reference = items[0]
    for name, recording in items[1:]:
        if recording.get_num_channels() != reference.get_num_channels():
            raise ValueError(
                f"input {name!r} has a different fiber count than "
                f"{reference_name!r}"
            )
        if recording.get_num_segments() != reference.get_num_segments():
            raise ValueError(
                f"input {name!r} has a different segment count than "
                f"{reference_name!r}"
            )
        for index in range(reference.get_num_segments()):
            same = recording.get_num_samples(
                index
            ) == reference.get_num_samples(index)
            if not same:
                raise ValueError(
                    f"segment {index}: input {name!r} and "
                    f"{reference_name!r} have different sample counts"
                )


def _pad_times(times: np.ndarray, margin: int) -> np.ndarray:
    """Extend times by ``margin`` samples at both ends, per fiber.

    Padded timestamps are extrapolated linearly from each fiber's median
    sampling interval, so a step reading ``times`` sees a plausible clock
    across the padded region.

    Parameters
    ----------
    times : np.ndarray
        Times, shape ``(n_samples, n_fibers)``.
    margin : int
        Samples to add at each end.

    Returns
    -------
    np.ndarray
        Times, shape ``(n_samples + 2 * margin, n_fibers)``.
    """
    step = np.median(np.diff(times, axis=0), axis=0)
    offsets = np.arange(1, margin + 1)[:, np.newaxis]
    head = times[0] - step * offsets[::-1]
    tail = times[-1] + step * offsets
    return np.concatenate([head, times, tail], axis=0)


def _as_output_map(result: object) -> dict[str, np.ndarray]:
    """Return a step's return value as a ``name -> array`` mapping.

    A bare array becomes ``{"result": array}``, so single-output steps keep
    returning arrays while multi-output steps return named arrays.

    Parameters
    ----------
    result : object
        Whatever the step function returned.

    Returns
    -------
    dict
        Mapping of output name to array.

    Raises
    ------
    TypeError
        If the value is neither an array nor a dict of arrays.
    """
    if isinstance(result, dict):
        return {name: np.asarray(value) for name, value in result.items()}
    if isinstance(result, np.ndarray):
        return {"result": result}
    raise TypeError(
        "a processing step must return an array or a dict of arrays; got "
        f"{type(result)}"
    )


def _split_by_lengths(
    array: np.ndarray, lengths: Sequence[int]
) -> list[np.ndarray]:
    """Split ``array`` along axis 0 into blocks of the given lengths."""
    return list(np.split(array, np.cumsum(lengths)[:-1], axis=0))


@dataclass(frozen=True)
class ArrayStep:
    """A contributor-written step operating on numpy arrays.

    Parameters
    ----------
    name : str
        Registry name.
    function : Callable
        The wrapped array function.
    input_names : tuple of str
        Input names, derived from the signature. The first is the
        primary, whose metadata and time base the outputs inherit.
    param_names : frozenset of str
        Names of the keyword-only step parameters.
    contiguous : bool
        If True the function runs once per segment, otherwise once over all
        segments concatenated.
    margin : int
        Per-segment padding, only used when ``contiguous`` is True.
    pad_mode : str
        Padding mode passed to :func:`numpy.pad`.
    """

    name: str
    function: Callable
    input_names: tuple[str, ...]
    param_names: frozenset[str]
    contiguous: bool
    margin: int
    pad_mode: str

    @property
    def primary_input(self) -> str:
        """Name of the input name whose metadata the outputs inherit."""
        return self.input_names[0]

    def run(
        self, inputs: dict[str, BaseRecording], params: dict
    ) -> dict[str, BaseRecording]:
        """Run the step over recordings, returning one per output name.

        Parameters
        ----------
        inputs : dict
            Mapping of input name to recording, covering exactly
            :attr:`input_names`.
        params : dict
            Step parameters, forwarded to the function as keywords.

        Returns
        -------
        dict
            Mapping of output name to recording.

        Raises
        ------
        ValueError
            If names are missing or unexpected, the inputs are incompatible,
            or a returned array has the wrong length.
        """
        missing = set(self.input_names) - set(inputs)
        unexpected = set(inputs) - set(self.input_names)
        if missing or unexpected:
            raise ValueError(
                f"step {self.name!r} expects inputs "
                f"{list(self.input_names)}; missing {sorted(missing)}, "
                f"unexpected {sorted(unexpected)}"
            )
        _check_compatible(inputs)
        primary = inputs[self.primary_input]
        if self.contiguous:
            collected = self._run_per_segment(inputs, primary, params)
        else:
            validate_timeline(primary)
            collected = self._run_over_all(inputs, primary, params)
        return {
            name: concatenate_segments(primary, arrays)
            for name, arrays in collected.items()
        }

    def _traces(
        self, inputs: dict[str, BaseRecording], segment_index: int
    ) -> list[np.ndarray]:
        """Return one segment's traces for every input name, in name order."""
        return [
            inputs[name].get_traces(segment_index=segment_index)
            for name in self.input_names
        ]

    def _run_per_segment(
        self,
        inputs: dict[str, BaseRecording],
        primary: BaseRecording,
        params: dict,
    ) -> dict[str, list[np.ndarray]]:
        """Apply the function to each segment, padding then trimming."""
        collected: dict[str, list[np.ndarray]] = {}
        margin = self.margin
        for index in range(primary.get_num_segments()):
            arrays = self._traces(inputs, index)
            times = _times_for(primary, index)
            length = arrays[0].shape[0]
            if margin:
                if margin >= length:
                    raise ValueError(
                        f"step {self.name!r} needs a margin of {margin} "
                        f"samples but segment {index} has only {length}"
                    )
                arrays = [
                    np.pad(
                        array, ((margin, margin), (0, 0)), mode=self.pad_mode
                    )
                    for array in arrays
                ]
                times = _pad_times(times, margin)
            outputs = _as_output_map(
                self.function(arrays[0], times, *arrays[1:], **params)
            )
            expected = length + 2 * margin
            for name, array in outputs.items():
                if array.shape[0] != expected:
                    raise ValueError(
                        f"step {self.name!r} output {name!r} returned "
                        f"{array.shape[0]} samples for segment {index}, "
                        f"expected {expected}; steps must preserve length"
                    )
                if margin:
                    array = array[margin:-margin]
                collected.setdefault(name, []).append(array)
        return collected

    def _run_over_all(
        self,
        inputs: dict[str, BaseRecording],
        primary: BaseRecording,
        params: dict,
    ) -> dict[str, list[np.ndarray]]:
        """Apply the function once across all segments, then split back."""
        num_segments = primary.get_num_segments()
        lengths = [
            primary.get_num_samples(index) for index in range(num_segments)
        ]
        arrays = [
            np.concatenate(
                [
                    inputs[name].get_traces(segment_index=index)
                    for index in range(num_segments)
                ],
                axis=0,
            )
            for name in self.input_names
        ]
        times = np.concatenate(
            [_times_for(primary, index) for index in range(num_segments)],
            axis=0,
        )
        outputs = _as_output_map(
            self.function(arrays[0], times, *arrays[1:], **params)
        )
        total = sum(lengths)
        collected: dict[str, list[np.ndarray]] = {}
        for name, array in outputs.items():
            if array.shape[0] != total:
                raise ValueError(
                    f"step {self.name!r} output {name!r} returned "
                    f"{array.shape[0]} samples, expected {total}; steps must "
                    "preserve length"
                )
            collected[name] = _split_by_lengths(array, lengths)
        return collected


@dataclass(frozen=True)
class RecordingStep:
    """A recording-to-recording callable registered unchanged.

    Wraps upstream spikeinterface preprocessors, which are lazy and may change
    the sample count -- something the array contract forbids.

    Parameters
    ----------
    name : str
        Registry name.
    function : Callable
        A ``recording -> recording`` callable.
    input_names : tuple of str
        Always the single name ``("recording",)``.
    """

    name: str
    function: Callable
    input_names: tuple[str, ...] = field(default=("recording",))

    @property
    def primary_input(self) -> str:
        """Name of the single input name."""
        return self.input_names[0]

    def run(
        self, inputs: dict[str, BaseRecording], params: dict
    ) -> dict[str, BaseRecording]:
        """Call the wrapped callable on the single input recording.

        Parameters
        ----------
        inputs : dict
            Mapping holding the single input name.
        params : dict
            Keyword arguments forwarded to the callable.

        Returns
        -------
        dict
            ``{"result": recording}``.
        """
        return {"result": self.function(inputs[self.primary_input], **params)}


def _introspect_signature(
    function: Callable,
) -> tuple[tuple[str, ...], frozenset[str]]:
    """Derive input names and parameter names from a signature.

    The first positional parameter is the primary input name, the second is
    the times array (not a name), further positional parameters are extra
    input names, and keyword-only parameters are step parameters.

    Parameters
    ----------
    function : Callable
        The array function to inspect.

    Returns
    -------
    input_names : tuple of str
        Input names, primary first.
    param_names : frozenset of str
        Keyword-only parameter names.

    Raises
    ------
    TypeError
        If the signature uses ``*args``/``**kwargs``, or has fewer than the
        two required positional parameters.
    """
    positional: list[str] = []
    keyword_only: list[str] = []
    for name, parameter in inspect.signature(function).parameters.items():
        if parameter.kind in (
            parameter.POSITIONAL_ONLY,
            parameter.POSITIONAL_OR_KEYWORD,
        ):
            positional.append(name)
        elif parameter.kind is parameter.KEYWORD_ONLY:
            keyword_only.append(name)
        else:
            raise TypeError(
                f"{function.__name__}: *args/**kwargs are not allowed; "
                "declare inputs positionally and parameters keyword-only"
            )
    if len(positional) < 2:
        raise TypeError(
            f"{function.__name__}: a step needs at least two positional "
            "parameters, the primary input and the times array"
        )
    return (positional[0], *positional[2:]), frozenset(keyword_only)


def _register(
    name: str, spec: ArrayStep | RecordingStep, overwrite: bool
) -> None:
    """Store ``spec`` under ``name``, honouring ``overwrite``."""
    if name in _STEP_REGISTRY and not overwrite:
        raise ValueError(f"step {name!r} is already registered")
    _STEP_REGISTRY[name] = spec


def register_step(
    name: str, function: Callable, overwrite: bool = False
) -> Callable:
    """Register a ``recording -> recording`` callable under ``name``.

    Used for upstream spikeinterface preprocessors, which are registered
    unchanged as a :class:`RecordingStep`.

    Parameters
    ----------
    name : str
        Name used to reference the step from a pipeline.
    function : Callable
        The callable to register.
    overwrite : bool, default: False
        If False, re-registering an existing name raises.

    Returns
    -------
    Callable
        ``function``, unchanged.

    Raises
    ------
    ValueError
        If ``name`` is already registered and ``overwrite`` is False.
    """
    _register(name, RecordingStep(name=name, function=function), overwrite)
    return function


def processing_step(
    name: str,
    *,
    contiguous: bool = False,
    margin: int = 0,
    pad_mode: str = "reflect",
    overwrite: bool = False,
) -> Callable:
    """Register an array function as a named processing step (decorator).

    Input names and parameters are derived from the signature; nothing about
    inputs or outputs is declared here. The returned wrapper takes
    *recordings* positionally in name order plus parameters as keywords, so a
    step stays directly callable outside a pipeline.

    Parameters
    ----------
    name : str
        Name used to reference the step from a pipeline.
    contiguous : bool, default: False
        If True the function requires temporally continuous samples and is run
        once per segment. If False it tolerates temporal gaps and is run once
        over all segments concatenated.
    margin : int, default: 0
        Samples of padding added to each segment before the call and trimmed
        from the result afterwards. Only meaningful with ``contiguous=True``.
    pad_mode : str, default: "reflect"
        Padding mode passed to :func:`numpy.pad`.
    overwrite : bool, default: False
        If False, re-registering an existing name raises.

    Returns
    -------
    Callable
        A decorator returning the recording-level wrapper.
    """

    def decorator(function: Callable) -> Callable:
        """Introspect and register ``function``, returning its wrapper."""
        input_names, param_names = _introspect_signature(function)
        spec = ArrayStep(
            name=name,
            function=function,
            input_names=input_names,
            param_names=param_names,
            contiguous=contiguous,
            margin=margin,
            pad_mode=pad_mode,
        )
        _register(name, spec, overwrite)

        @functools.wraps(function)
        def wrapper(*recordings, **params):
            """Run the step on recordings given in name order."""
            if len(recordings) != len(spec.input_names):
                raise TypeError(
                    f"step {name!r} takes {len(spec.input_names)} "
                    f"recording(s) {list(spec.input_names)}, got "
                    f"{len(recordings)}"
                )
            results = spec.run(
                dict(zip(spec.input_names, recordings, strict=True)), params
            )
            if len(results) == 1:
                return next(iter(results.values()))
            return results

        wrapper.spec = spec
        return wrapper

    return decorator


def get_step(name: str) -> ArrayStep | RecordingStep:
    """Return the spec registered under ``name``.

    Parameters
    ----------
    name : str
        A registered step name.

    Returns
    -------
    ArrayStep or RecordingStep
        The registered spec.

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
