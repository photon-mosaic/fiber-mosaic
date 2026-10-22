"""Synthetic fiber photometry data for demos, tests, and development.

Generates fake recordings from numpy arrays so the rest of the library can be
exercised without real data files. See :mod:`.generate` for the signal model.

``recording_from_traces``, used internally by
:func:`~fiber_mosaic.synthetic.generate.simulate_group`, is a
general-purpose constructor with nothing synthetic-specific about it --
import it from :mod:`fiber_mosaic.core.base` or the top-level
:mod:`fiber_mosaic` package.
"""

from fiber_mosaic.synthetic.generate import (
    DEFAULT_BASELINE,
    DEFAULT_SAMPLING_FREQUENCY,
    calcium_transients,
    simulate_bands,
    simulate_group,
    smooth_drift,
)

__all__ = [
    "DEFAULT_BASELINE",
    "DEFAULT_SAMPLING_FREQUENCY",
    "calcium_transients",
    "simulate_bands",
    "simulate_group",
    "smooth_drift",
]
