"""
Fiber photometry data extractors.

This module provides extractors for reading fiber photometry data from various
file formats. Each extractor follows the SpikeInterface pattern of discovery
and lazy loading.

Supported formats:
- CSV: Standard CSV files with timestamps and fluorescence data
- NWB: Neurodata Without Borders files with FiberPhotometryResponseSeries
- TDT: Tucker-Davis Technologies binary format
- Doric: Doric Neuroscience .doric and CSV exports
- NPM: Neurophotometrics multi-channel CSV format
- DANDI: Streaming access to NWB files on the DANDI Archive

Usage:
    from fiber_mosaic.extractors import (
        detect_formats,
        CsvFiberPhotometryExtractor,
        read_csv_fiber_photometry,
    )

    # Auto-detect formats in a folder
    formats = detect_formats("/path/to/data")

    # Read a specific format
    recording = read_csv_fiber_photometry("data.csv", color="green")
"""

from .csv_extractor import (
    CsvFiberPhotometryExtractor,
    read_csv_fiber_photometry,
)
from .dandi_extractor import (
    DandiFiberPhotometryExtractor,
    is_dandi_uri,
    parse_dandi_uri,
    read_dandi_fiber_photometry,
)
from .detect_formats import detect_formats
from .doric_extractor import (
    DoricFiberPhotometryExtractor,
    read_doric_fiber_photometry,
)
from .npm_extractor import (
    NpmFiberPhotometryExtractor,
    read_npm_fiber_photometry,
)
from .nwb_extractor import (
    NwbFiberPhotometryExtractor,
    read_nwb_fiber_photometry,
)
from .tdt_extractor import (
    TdtFiberPhotometryExtractor,
    read_tdt_fiber_photometry,
)

__all__ = [
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
