"""
fiber-mosaic: A library for fiber photometry data analysis.

Built on top of SpikeInterface, fiber-mosaic provides domain-specific
vocabulary and tools for working with fiber photometry recordings.

Key concepts:
- A "fiber" is analogous to a channel in electrophysiology
- A "color" is a single-color recording (one BaseFiberPhotometryExtractor)
- A FiberPhotometryRecordingGroup bundles multiple colors sharing fibers
"""

__version__ = "0.1.0"

from .core.base import (
    BaseFiberPhotometryExtractor,
    FiberPhotometryMixin,
    FiberPhotometryRecordingGroup,
)

# Extractors - import directly for convenience
from .extractors import (
    # CSV
    CsvFiberPhotometryExtractor,
    # DANDI streaming
    DandiFiberPhotometryExtractor,
    # Doric
    DoricFiberPhotometryExtractor,
    # NPM (Neurophotometrics)
    NpmFiberPhotometryExtractor,
    # NWB
    NwbFiberPhotometryExtractor,
    # TDT
    TdtFiberPhotometryExtractor,
    # Format detection
    detect_formats,
    is_dandi_uri,
    parse_dandi_uri,
    read_csv_fiber_photometry,
    read_dandi_fiber_photometry,
    read_doric_fiber_photometry,
    read_npm_fiber_photometry,
    read_nwb_fiber_photometry,
    read_tdt_fiber_photometry,
)

__all__ = [
    # Core classes
    "BaseFiberPhotometryExtractor",
    "FiberPhotometryMixin",
    "FiberPhotometryRecordingGroup",
    # Format detection
    "detect_formats",
    # CSV
    "CsvFiberPhotometryExtractor",
    "read_csv_fiber_photometry",
    # NWB
    "NwbFiberPhotometryExtractor",
    "read_nwb_fiber_photometry",
    # TDT
    "TdtFiberPhotometryExtractor",
    "read_tdt_fiber_photometry",
    # Doric
    "DoricFiberPhotometryExtractor",
    "read_doric_fiber_photometry",
    # NPM
    "NpmFiberPhotometryExtractor",
    "read_npm_fiber_photometry",
    # DANDI
    "DandiFiberPhotometryExtractor",
    "read_dandi_fiber_photometry",
    "is_dandi_uri",
    "parse_dandi_uri",
]
