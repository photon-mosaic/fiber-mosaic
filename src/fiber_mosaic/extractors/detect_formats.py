"""
Format detection utilities for fiber photometry data.

This module provides utilities to automatically detect which acquisition formats
are present in a data folder, enabling automatic selection of the appropriate
extractor.
"""

from __future__ import annotations

import glob
import logging
import os
from pathlib import Path
from typing import Set

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _is_float(value) -> bool:
    """Return True if *value* can be interpreted as a float."""
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _is_event_csv(path: str) -> bool:
    """
    Return True if the CSV file is an event CSV (single column named 'timestamps').

    Parameters
    ----------
    path : str
        Absolute path to a CSV file.

    Returns
    -------
    bool
        True if the file has a single column named 'timestamps'.
    """
    try:
        df = pd.read_csv(path, nrows=0, index_col=False)
        cols = list(df.columns)
        return len(cols) == 1 and cols[0].lower() == "timestamps"
    except Exception:
        return False


def classify_csv_file(path: str) -> str:
    """
    Classify a single CSV file as belonging to one of three modalities.

    Parameters
    ----------
    path : str
        Absolute path to a CSV file.

    Returns
    -------
    str
        One of ``"doric"``, ``"npm"``, or ``"csv"``.

    Raises
    ------
    Exception
        If the file structure is invalid or unrecognized.
    """
    df = pd.read_csv(path, header=None, nrows=2, index_col=False, dtype=str)
    df = df.dropna(axis=1, how="all")
    df_arr = np.array(df).flatten()
    non_numeric = [el for el in df_arr if not _is_float(el)]

    # Doric CSV files have a 2-line all-string header (metadata + units rows)
    # with no numeric values in the first two rows at all.
    if len(non_numeric) == len(df_arr):
        return "doric"

    # File has string headers (or numeric-only headers) — inspect column names
    df = pd.read_csv(path, index_col=False)
    colnames = list(df.columns)

    # Doric v2 files store numeric values as column headers; treat as headerless
    if all(_is_float(c) for c in colnames):
        df = pd.read_csv(path, header=None)
        cols = np.array(list(df.columns), dtype=str)
    else:
        cols = np.array(colnames, dtype=str)

    if len(cols) == 1:
        if cols[0].lower() != "timestamps":
            raise ValueError(
                f"Single-column CSV must have column name 'timestamps', got '{cols[0]}'"
            )
        return "csv"
    elif len(cols) == 3:
        arr1 = np.array(["timestamps", "data", "sampling_rate"])
        arr2 = np.char.lower(cols)
        if (np.sort(arr1) == np.sort(arr2)).all():
            return "csv"
        raise ValueError(
            "3-column CSV must have columns: timestamps, data, sampling_rate"
        )
    elif len(cols) >= 2:
        return "npm"
    else:
        raise ValueError(f"CSV file has unexpected number of columns: {len(cols)}")


def detect_formats(folder_path: str) -> Set[str]:
    """
    Detect all acquisition formats present in a session folder.

    Scans the folder for files matching known fiber photometry formats and
    returns a set of format identifiers.

    Parameters
    ----------
    folder_path : str or Path
        Path to the session folder.

    Returns
    -------
    set of str
        Format strings for all sources found in the folder.
        Possible elements: ``"nwb"``, ``"tdt"``, ``"doric"``, ``"csv"``, ``"npm"``.

    Examples
    --------
    >>> formats = detect_formats("/path/to/session")
    >>> if "nwb" in formats:
    ...     recording = read_nwb_fiber_photometry(...)
    """
    folder_path = str(folder_path)
    formats: Set[str] = set()

    # NWB .nwb files
    if glob.glob(os.path.join(folder_path, "*.nwb")):
        formats.add("nwb")
        logger.debug("Found NWB files in %s", folder_path)

    # TDT .tsq files (binary header)
    if glob.glob(os.path.join(folder_path, "*.tsq")):
        formats.add("tdt")
        logger.debug("Found TDT files in %s", folder_path)

    # Doric .doric files
    if glob.glob(os.path.join(folder_path, "*.doric")):
        formats.add("doric")
        logger.debug("Found Doric files in %s", folder_path)

    # Process CSV files
    csv_paths = glob.glob(os.path.join(folder_path, "*.csv"))

    # Multi-column CSV files can be NPM, Doric CSV exports, or 3-column data_csv files
    non_event_csv_paths = [p for p in csv_paths if not _is_event_csv(p)]
    has_npm_data = False

    if non_event_csv_paths:
        labels = set()
        for p in non_event_csv_paths:
            try:
                labels.add(classify_csv_file(p))
            except Exception as e:
                logger.warning("Could not classify CSV file %s: %s", p, e)

        if "npm" in labels:
            formats.add("npm")
            has_npm_data = True
            logger.debug("Found NPM CSV files in %s", folder_path)
        if "doric" in labels:
            formats.add("doric")
            logger.debug("Found Doric CSV files in %s", folder_path)
        if not has_npm_data and "csv" in labels:
            formats.add("csv")
            logger.debug("Found standard CSV files in %s", folder_path)

    # Single-column timestamp CSVs provide external event timestamps
    # When NPM data is present, suppress NPM-generated split-event files (event*.csv)
    event_csv_paths = [p for p in csv_paths if _is_event_csv(p)]
    if has_npm_data:
        external_event_csv_paths = [
            p for p in event_csv_paths
            if not os.path.basename(p).startswith("event")
        ]
    else:
        external_event_csv_paths = event_csv_paths

    if external_event_csv_paths:
        formats.add("csv")
        logger.debug("Found event CSV files in %s", folder_path)

    return formats
