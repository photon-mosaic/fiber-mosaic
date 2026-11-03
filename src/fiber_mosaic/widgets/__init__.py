"""
Widgets module for fiber-mosaic.

Fiber-native plotting built on SpikeInterface's widgets. The SI machinery
(backends, interactive controls) is kept; the vocabulary and the defaults are
photometry's:

- `fluorescence`: `plot_fluorescence`, which plots one or more fibers of any
  recording -- raw or processed -- and `FiberTracesWidget`, the
  ``TracesWidget`` subclass behind it.

Requires the ``widgets`` extra (``pip install fiber-mosaic[widgets]``). Two
backends are supported: ``"matplotlib"`` for static figures and
``"ipywidgets"`` for interactive scrolling in a notebook, which needs
``%matplotlib widget``:

    >>> from fiber_mosaic.widgets import plot_fluorescence
    >>> plot_fluorescence(recording, fiber_ids=["F0"], time_range=(0, 120))
"""

from __future__ import annotations

from fiber_mosaic.widgets.fluorescence import (
    FiberTracesWidget,
    plot_fluorescence,
)

__all__ = [
    "FiberTracesWidget",
    "plot_fluorescence",
]
