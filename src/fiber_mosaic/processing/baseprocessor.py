"""
Fiber-aware SpikeInterface preprocessors.

SpikeInterface preprocessors are lazy recording-to-recording classes: they
compute in `get_traces()` rather than up front, so intermediates keep their
provenance and can be sliced, saved and reloaded. This module makes that
machinery fiber-native:

- BaseFiberPhotometryPreprocessor: base class for fiber-mosaic's own
  preprocessors, adding the fiber API on top of SI's BasePreprocessor.
- as_fiber_preprocessor: gives an existing upstream preprocessor class the
  same fiber-native surface, without reimplementing it.

`BasePreprocessor` propagates metadata with ``copy_metadata(only_main=False)``,
which carries annotations and channel properties but not attributes or
methods. That is why the fiber API lives in
:class:`~fiber_mosaic.core.base.FiberPhotometryMixin` and the color is stored
as an annotation -- both then survive any preprocessing step.
"""

from __future__ import annotations

import inspect

from spikeinterface.core import BaseRecording
from spikeinterface.preprocessing.basepreprocessor import (
    BasePreprocessor,
)

from fiber_mosaic.core.base import (
    FiberPhotometryMixin,
)


class BaseFiberPhotometryPreprocessor(FiberPhotometryMixin, BasePreprocessor):
    """
    Base class for lazy fiber photometry preprocessing steps.

    Combines SpikeInterface's ``BasePreprocessor`` with the fiber-native API,
    so a processing step behaves like any other recording: lazy, sliceable,
    and carrying its own provenance.

    Subclasses implement the work in a segment's ``get_traces()`` and set
    ``self._kwargs``, exactly as upstream preprocessors do. A step that needs
    a fitted model (a bleaching trend, a regression coefficient) should fit it
    in ``__init__`` across all segments and store the coefficients in
    ``_kwargs``, following ``ZScoreRecording``; listing those names in
    ``_precomputable_kwarg_names`` lets them survive `save()`/`load()`.

    Parameters
    ----------
    recording : BaseRecording
        The recording to preprocess.
    sampling_frequency : float, optional
        Output sampling frequency. Defaults to the parent's.
    fiber_ids : list or array-like, optional
        Output fiber IDs. Defaults to the parent's.
    dtype : dtype, optional
        Output dtype. Defaults to the parent's.
    """

    def __init__(
        self,
        recording: BaseRecording,
        sampling_frequency: float | None = None,
        fiber_ids=None,
        dtype=None,
    ):
        BasePreprocessor.__init__(
            self,
            recording,
            sampling_frequency=sampling_frequency,
            channel_ids=fiber_ids,
            dtype=dtype,
        )


def _check_overridable(
    source_class: type, signature: inspect.Signature, names
) -> None:
    """
    Check that every name in ``names`` is a default this class can accept.

    Parameters
    ----------
    source_class : type
        The class whose signature is being overridden, for error messages.
    signature : inspect.Signature
        Signature of ``source_class``.
    names : iterable of str
        Parameter names to override.

    Raises
    ------
    TypeError
        If a name is not a parameter and the class accepts no extra keywords,
        or if it names a required or variadic parameter.
    """
    parameters = signature.parameters
    accepts_extra = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    variadic = (
        inspect.Parameter.VAR_KEYWORD,
        inspect.Parameter.VAR_POSITIONAL,
    )
    for key in names:
        parameter = parameters.get(key)
        if parameter is None:
            if not accepts_extra:
                raise TypeError(
                    f"{source_class.__name__} has no parameter {key!r}; "
                    f"accepted: {sorted(parameters)}"
                )
            continue
        if parameter.kind in variadic:
            raise TypeError(
                f"{key!r} is a variadic parameter of "
                f"{source_class.__name__} and has no default to override"
            )
        if parameter.default is inspect.Parameter.empty:
            raise TypeError(
                f"{key!r} is a required parameter of "
                f"{source_class.__name__}; default_kwargs can only change "
                "parameters that already have a default"
            )


def as_fiber_preprocessor(
    source_class: type,
    name: str | None = None,
    default_kwargs: dict | None = None,
) -> type:
    """
    Give an upstream preprocessor class the fiber-native API.

    Upstream SpikeInterface preprocessors build their segments inside their
    own ``__init__``, so there is no hook for substituting a fiber-aware
    segment class. Mixing the API in at the recording level instead is
    enough, and timestamps propagate through SpikeInterface's own segments.

    Parameters
    ----------
    source_class : type
        An upstream preprocessor class, for example
        ``spikeinterface.preprocessing.preprocessing_classes.BandpassFilterRecording``.
    name : str, optional
        Name for the generated class. Defaults to ``"Fiber" + source name``.
    default_kwargs : dict, optional
        New defaults for the upstream constructor, replacing values chosen for
        electrophysiology. Callers still override them explicitly; only the
        fallback changes. The generated class reports the new defaults through
        `inspect.signature`, so they show up in help and in the public
        function built by
        :func:`~fiber_mosaic.core.core_tools.define_function_handling_group_from_class`.

    Returns
    -------
    type
        A subclass of ``source_class`` that also carries the fiber API.

    Raises
    ------
    TypeError
        If ``default_kwargs`` names a parameter the class does not accept, or
        one that is required or variadic and so has no default to replace.

    Examples
    --------
    >>> FiberBandpass = as_fiber_preprocessor(
    ...     BandpassFilterRecording,
    ...     default_kwargs={"ignore_low_freq_error": True},
    ... )
    """
    if name is None:
        name = f"Fiber{source_class.__name__}"
    if not default_kwargs:
        return type(name, (FiberPhotometryMixin, source_class), {})

    signature = inspect.signature(source_class)
    overrides = dict(default_kwargs)
    _check_overridable(source_class, signature, overrides)

    def __init__(self, *args, **kwargs):
        """Construct the step, filling in fiber photometry defaults."""
        # Anything the caller gave -- by keyword, or positionally under a
        # parameter's own name -- wins over the override.
        supplied = set(kwargs) | set(signature.bind_partial(*args).arguments)
        merged = {
            key: value
            for key, value in overrides.items()
            if key not in supplied
        }
        source_class.__init__(self, *args, **kwargs, **merged)

    parameters = [
        parameter.replace(default=overrides[parameter.name])
        if parameter.name in overrides
        else parameter
        for parameter in signature.parameters.values()
    ]
    namespace = {
        "__init__": __init__,
        "__signature__": signature.replace(parameters=parameters),
    }
    return type(name, (FiberPhotometryMixin, source_class), namespace)
