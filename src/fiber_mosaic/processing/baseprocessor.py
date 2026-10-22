"""
Fiber-aware SpikeInterface preprocessors.

SpikeInterface preprocessors are lazy recording-to-recording classes: they
compute in `get_traces()` rather than up front, so intermediates keep their
provenance and can be sliced, saved and reloaded. This module makes that
machinery fiber-native:

- BaseFiberPhotometryPreprocessor: base class for fiber-mosaic's own
  preprocessors, adding the fiber API on top of SI's BasePreprocessor.

`BasePreprocessor` propagates metadata with ``copy_metadata(only_main=False)``,
which carries annotations and channel properties but not attributes or
methods. That is why the name is stored as an annotation: it survives any
preprocessing step. The fiber API itself lives in
:class:`~fiber_mosaic.core.base.FiberPhotometryMixin`, so only fiber-mosaic's
own classes have it -- subclasses of this module's base class and the ports
in :mod:`~fiber_mosaic.processing.from_spikeinterface`.
"""

from __future__ import annotations

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
