"""
Widgets module for fiber-mosaic.

Fiber-native plotting built on SpikeInterface's widgets. Requires the
``widgets`` extra (``pip install fiber-mosaic[widgets]``).
"""

from .fluorescence import plot_fluorescence

__all__ = ["plot_fluorescence"]
