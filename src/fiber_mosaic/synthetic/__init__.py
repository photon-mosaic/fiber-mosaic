"""Synthetic fiber photometry data for demos, tests, and development.

Generates fake recordings from numpy arrays so the rest of the library can be
exercised without real data files. See :mod:`.generate` for the signal model.
"""

from fiber_mosaic.synthetic.generate import (
    DEFAULT_BASELINE,
    DEFAULT_SAMPLING_FREQUENCY,
    calcium_transients,
    recording_from_traces,
    simulate_bands,
    simulate_group,
    smooth_drift,
)

__all__ = [
    "DEFAULT_BASELINE",
    "DEFAULT_SAMPLING_FREQUENCY",
    "calcium_transients",
    "recording_from_traces",
    "simulate_bands",
    "simulate_group",
    "smooth_drift",
]
