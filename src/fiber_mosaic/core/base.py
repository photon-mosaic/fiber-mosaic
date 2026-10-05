"""
Core classes for fiber photometry data handling.

This module provides the base classes that form the foundation of fiber-mosaic:

- FiberPhotometryMixin: The fiber-native API, mixable into any SI recording
- BaseFiberPhotometryExtractor: A per-color recording (wraps SI BaseRecording)
- FiberPhotometryRecordingGroup: A container for multiple colors sharing fibers
- recording_from_traces: General-purpose constructor for wrapping plain numpy
  arrays as a fiber-native recording (mixes FiberPhotometryMixin into SI's
  own NumpyRecording -- used by :mod:`fiber_mosaic.synthetic` but not
  specific to synthetic data)
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
from spikeinterface.core import BaseRecording
from spikeinterface.core.numpyextractors import (
    NumpyRecording,
    NumpyRecordingSegment,
)


class FiberPhotometryMixin:
    """
    Fiber-native API for fiber photometry recordings.

    Provides fiber-photometry vocabulary on top of a SpikeInterface
    recording:

    - Channels are called "fibers"
    - `get_fluorescence()` wraps `get_traces()` with fiber-native naming

    This is deliberately separate from BaseFiberPhotometryExtractor, which
    adds the source-side concerns (construction, stream discovery, segment
    loading). Keeping the API in a mixin lets recording-to-recording classes
    -- notably SpikeInterface preprocessors, which cannot inherit from an
    extractor -- expose the same fiber-native surface.

    Notes
    -----
    Must be mixed into a SpikeInterface ``BaseRecording``, on whose
    ``get_channel_ids``, ``get_num_channels``, ``get_num_segments``,
    ``get_traces``, ``get_sampling_frequency``, ``get_dtype`` and
    ``get_annotation`` it relies. It defines no ``__init__``, so it never
    interferes with the host class's construction.

    The color is read from the ``"color"`` annotation rather than from an
    attribute, because ``copy_metadata`` propagates annotations but not
    attributes -- so the color survives SpikeInterface preprocessing steps.
    """

    @property
    def color(self) -> str:
        """Return the color/wavelength identifier for this recording."""
        return self.get_annotation("color")

    @property
    def fiber_ids(self) -> np.ndarray:
        """Return the fiber identifiers (alias for channel_ids)."""
        return self.get_channel_ids()

    def get_fiber_ids(self) -> np.ndarray:
        """Return the fiber identifiers (alias for get_channel_ids)."""
        return self.get_channel_ids()

    def get_num_fibers(self) -> int:
        """Return the number of fibers (alias for get_num_channels)."""
        return self.get_num_channels()

    def get_fluorescence(
        self,
        segment_index: int | None = None,
        start_frame: int | None = None,
        end_frame: int | None = None,
        fiber_ids: Sequence | None = None,
    ) -> np.ndarray:
        """
        Get fluorescence traces for specified fibers and time range.

        This is a fiber-native wrapper around `get_traces()` that uses
        fiber-photometry vocabulary.

        Parameters
        ----------
        segment_index : int, optional
            The segment index. If None and only one segment exists,
            defaults to 0.
        start_frame : int, optional
            The start frame. If None, starts from the beginning.
        end_frame : int, optional
            The end frame (exclusive). If None, reads to the end.
        fiber_ids : list or array-like, optional
            The fiber IDs to retrieve. If None, returns all fibers.

        Returns
        -------
        traces : np.ndarray
            Array of shape (n_samples, n_fibers) containing the
            fluorescence data.
        """
        return self.get_traces(
            segment_index=segment_index,
            start_frame=start_frame,
            end_frame=end_frame,
            channel_ids=fiber_ids,
        )

    def __repr__(self) -> str:
        """Return a one-line summary: class, color, fiber/segment count."""
        n_seg = self.get_num_segments()
        n_fib = self.get_num_fibers()
        sf = self.get_sampling_frequency()
        dtype = self.get_dtype()
        return (
            f"{self.__class__.__name__} | color={self.color} | "
            f"{n_fib} fiber(s) | {n_seg} segment(s) | "
            f"{sf:.1f} Hz | dtype: {dtype}"
        )


class BaseFiberPhotometryExtractor(FiberPhotometryMixin, BaseRecording):
    """
    Base class for fiber photometry recordings.

    This class represents a single-color recording from a fiber photometry
    experiment. It wraps SpikeInterface's BaseRecording, takes its
    fiber-native API from FiberPhotometryMixin, and adds the source-side
    concerns: construction, `get_streams()` for format-specific discovery of
    available data streams, and segment loading.

    Each color in an experiment should be its own BaseFiberPhotometryExtractor
    instance, grouped together via FiberPhotometryRecordingGroup.

    Parameters
    ----------
    sampling_frequency : float
        The nominal sampling frequency in Hz.
    fiber_ids : list or array-like
        Identifiers for each fiber (will be used as channel_ids internally).
    color : str
        The color/wavelength identifier for this recording
        (e.g., "green", "red", "iso").
    dtype : dtype, optional
        The data type of the traces. Default is float64.
    """

    def __init__(
        self,
        sampling_frequency: float,
        fiber_ids: Sequence,
        color: str,
        dtype: np.dtype = np.float64,
    ):
        # Pass fiber_ids as channel_ids to SI's BaseRecording
        BaseRecording.__init__(
            self,
            sampling_frequency=sampling_frequency,
            channel_ids=list(fiber_ids),
            dtype=dtype,
        )
        # Store the color as an annotation, not an attribute, so that
        # copy_metadata carries it across SpikeInterface preprocessing steps.
        self.annotate(color=color)
        # Store kwargs for serialization
        self._kwargs = {
            "sampling_frequency": sampling_frequency,
            "fiber_ids": list(fiber_ids),
            "color": color,
            "dtype": str(dtype),
        }

    @classmethod
    def get_streams(
        cls, file_path: str, **kwargs
    ) -> tuple[list[str], list[str]]:
        """
        Discover available streams in a file.

        This is a discovery hook that file-format-specific readers should
        override. Returns the available stream names and IDs so users can
        choose which to load.

        Parameters
        ----------
        file_path : str
            Path to the file or folder to inspect.
        **kwargs
            Format-specific discovery options. Most readers ignore these;
            some accept keywords that mirror their constructor (e.g. the CSV
            reader accepts ``time_column`` so discovery excludes the same
            column the loader will). Overrides should declare the options
            they support explicitly.

        Returns
        -------
        stream_names : list of str
            Human-readable names of available streams.
        stream_ids : list of str
            Identifiers to pass to the reader's constructor.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.
        """
        raise NotImplementedError(
            f"{cls.__name__}.get_streams() must be implemented by file readers"
        )

    def _add_numpy_segment(
        self,
        traces: np.ndarray,
        timestamps: np.ndarray,
    ) -> None:
        """
        Add an in-memory segment and set its timestamps.

        Convenience method used by all file-format extractors that load
        data into NumPy arrays. Calls :meth:`add_segment` and
        :meth:`set_times` so subclasses do not repeat the boilerplate.

        Parameters
        ----------
        traces : np.ndarray
            2-D array of shape ``(n_samples, n_fibers)``. Cast to float64
            before storing.
        timestamps : np.ndarray
            1-D array of timestamps in seconds, length ``n_samples``.
        """
        t_start = float(timestamps[0]) if len(timestamps) > 0 else 0.0
        segment = NumpyRecordingSegment(
            traces=traces.astype(np.float64),
            sampling_frequency=self.get_sampling_frequency(),
            t_start=t_start,
        )
        self.add_segment(segment)
        self.set_times(timestamps, with_warning=False)


class _FiberNumpyRecording(FiberPhotometryMixin, NumpyRecording):
    """SpikeInterface's ``NumpyRecording`` with the fiber-native API mixed in.

    Not part of the public API -- constructed only by
    :func:`recording_from_traces`, which also sets the ``color`` annotation.
    """


def _segment_t_start(times: np.ndarray) -> float:
    """First-sample time as a scalar, for SI's per-segment ``t_start``.

    Handles only the 1-D and 2-D shapes ``set_times`` itself supports (for
    2-D, one column per fiber -- fiber 0's first sample stands in for the
    segment's start, since ``t_start`` is a single SI-level scalar and
    can't carry per-fiber jitter anyway), and returns 0.0 for anything
    empty or any other shape rather than indexing into it -- checking
    ``size`` before any indexing (instead of after, per-branch) means no
    zero-length axis, in either dimension, can raise. Malformed input is
    left for ``set_times`` (called right after construction, with the same
    ``times``) to reject with its own clear error.
    """
    if times.size == 0 or times.ndim not in (1, 2):
        return 0.0
    return float(times[0] if times.ndim == 1 else times[0, 0])


def _timestamps_per_segment(
    timestamps: ArrayLike | Sequence[ArrayLike], num_segments: int
) -> list[np.ndarray]:
    """Split ``timestamps`` into one array per segment.

    With a single segment, ``timestamps`` is that segment's own array-like
    (1-D or 2-D, plain list or ``np.ndarray``) taken as a whole -- not split
    apart. With more than one, it's a sequence with one array-like per
    segment. Deciding from ``num_segments`` rather than sniffing
    ``timestamps``' own shape is what lets a single 2-D segment be passed as
    nested lists without misreading it as one 1-D segment per row.
    """
    if num_segments == 1:
        segments = [np.asarray(timestamps)]
    else:
        segments = [np.asarray(times) for times in timestamps]

    if len(segments) != num_segments:
        raise ValueError(
            "timestamps must provide one array per segment "
            f"({num_segments} segments, got {len(segments)})"
        )
    return segments


def recording_from_traces(
    traces: np.ndarray | Sequence[np.ndarray],
    color: str,
    sampling_frequency: float = 30.0,
    fiber_ids: Sequence | None = None,
    timestamps: ArrayLike | Sequence[ArrayLike] | None = None,
) -> BaseRecording:
    """Wrap traces arrays as a fiber photometry recording.

    A general-purpose constructor for building a fiber-native recording
    straight from in-memory arrays -- useful for synthetic data, quick
    scripts, or wrapping real data that arrives as arrays plus timestamps,
    without writing a dedicated file-reading subclass.

    Parameters
    ----------
    traces : np.ndarray or sequence of np.ndarray
        One ``(num_samples, num_fibers)`` array for a single segment, or one
        array per segment.
    color : str
        Band label, e.g. ``"green"`` or ``"iso"``.
    sampling_frequency : float, default: 30.0
        Nominal acquisition rate in Hz, used for the SI time base. Still
        required even when ``timestamps`` is given.
    fiber_ids : sequence or None, default: None
        Fiber IDs; defaults to ``"fiber_0" ... "fiber_n"``.
    timestamps : array-like or sequence of array-like, optional
        Real per-sample timestamps, one array per segment matching
        ``traces`` (a bare array-like for a single segment -- a plain list
        is fine too). Passed to :meth:`~FiberPhotometryMixin.set_times`, so
        each array may be 1-D (broadcast to all fibers) or 2-D (one column
        per fiber); raises :exc:`ValueError` if the number of arrays
        doesn't match ``traces``' segment count. Each segment's first
        sample also becomes that segment's SI ``t_start``, so SI's own
        (nominal, evenly-spaced) ``get_times()`` at least starts at the
        right time instead of at zero; only
        :meth:`~FiberPhotometryMixin.get_fiber_times` reflects the exact
        supplied timestamps, e.g. under jitter. Omit ``timestamps`` to fall
        back to nominal timestamps from `sampling_frequency`.

    Returns
    -------
    BaseRecording
        A recording with the fiber-native API mixed in (``color``,
        ``fiber_ids``, ``get_fluorescence()``, ``get_fiber_times()``, ...).

    Notes
    -----
    Built on spikeinterface's own :class:`~spikeinterface.core.NumpyRecording`
    (multi-segment traces, dtype consistency, ``t_start``) with
    :class:`FiberPhotometryMixin` mixed in, rather than reimplementing that
    segment-handling here.
    """
    first = traces if isinstance(traces, np.ndarray) else traces[0]
    if fiber_ids is None:
        fiber_ids = [f"fiber_{index}" for index in range(first.shape[1])]
    if not isinstance(traces, np.ndarray):
        # SI's NumpyRecording requires an actual list, not e.g. a tuple
        traces = list(traces)
    num_segments = 1 if isinstance(traces, np.ndarray) else len(traces)

    segments = None
    t_starts = None
    if timestamps is not None:
        segments = _timestamps_per_segment(timestamps, num_segments)
        t_starts = [_segment_t_start(times) for times in segments]

    recording = _FiberNumpyRecording(
        traces_list=traces,
        sampling_frequency=sampling_frequency,
        t_starts=t_starts,
        channel_ids=list(fiber_ids),
    )
    recording.annotate(color=color)

    if segments is not None:
        for segment_index, times in enumerate(segments):
            recording.set_times(times, segment_index=segment_index)

    return recording


class FiberPhotometryRecordingGroup:
    """
    A container for multiple fiber photometry recordings sharing fibers.

    This class groups several per-color recordings
    (BaseFiberPhotometryExtractor instances) that share the same fiber
    layout but may have different timebases or sample counts.

    This is NOT a BaseRecording because the colors can have different
    timebases, so there's no single trace matrix to return.

    Parameters
    ----------
    recordings : dict
        A dictionary mapping color names to
        BaseFiberPhotometryExtractor instances. All recordings must have
        identical fiber_ids in the same order.

    Examples
    --------
    >>> green = CsvFiberPhotometryExtractor("green.csv", color="green")
    >>> iso = CsvFiberPhotometryExtractor("iso.csv", color="iso")
    >>> group = FiberPhotometryRecordingGroup({"green": green, "iso": iso})
    >>> group.colors
    ['green', 'iso']
    >>> group["green"].get_fluorescence()
    """

    def __init__(self, recordings: dict[str, BaseRecording]):
        if not recordings:
            raise ValueError("recordings dict cannot be empty")

        self._recordings = dict(recordings)
        self._validate_fiber_ids()

    def _validate_fiber_ids(self) -> None:
        """Validate that all recordings have identical fiber_ids."""
        fiber_ids_list = [
            tuple(rec.get_channel_ids()) for rec in self._recordings.values()
        ]
        reference = fiber_ids_list[0]
        for color, fiber_ids in zip(
            self._recordings.keys(), fiber_ids_list, strict=True
        ):
            if fiber_ids != reference:
                raise ValueError(
                    f"Recording '{color}' has different fiber_ids "
                    f"than the first recording. All recordings must "
                    f"share identical fiber_ids in the same order."
                )

    @property
    def colors(self) -> list[str]:
        """Return the list of color names in this group."""
        return list(self._recordings.keys())

    @property
    def fiber_ids(self) -> np.ndarray:
        """Return the shared fiber identifiers."""
        first_rec = next(iter(self._recordings.values()))
        return first_rec.get_channel_ids()

    def get_num_fibers(self) -> int:
        """Return the number of fibers shared by all recordings."""
        return len(self.fiber_ids)

    def get_recording(self, color: str) -> BaseRecording:
        """
        Get the recording for a specific color.

        Parameters
        ----------
        color : str
            The color name.

        Returns
        -------
        BaseRecording
            The recording for the specified color.
        """
        return self._recordings[color]

    # Dict-like access
    def __getitem__(self, color: str) -> BaseRecording:
        """Return the recording for a specific color."""
        return self._recordings[color]

    def __contains__(self, color: str) -> bool:
        """Check if a color exists in this group."""
        return color in self._recordings

    def __len__(self) -> int:
        """Return the number of colors in this group."""
        return len(self._recordings)

    def __iter__(self) -> Iterable[str]:
        """Iterate over color names in this group."""
        return iter(self._recordings)

    def keys(self):
        """Return the color names in this group."""
        return self._recordings.keys()

    def values(self):
        """Return the recordings in this group."""
        return self._recordings.values()

    def items(self):
        """Return (color, recording) pairs in this group."""
        return self._recordings.items()

    def __repr__(self) -> str:
        """Return a multi-line summary listing each per-color recording."""
        lines = [
            f"{self.__class__.__name__} | "
            f"{self.get_num_fibers()} fiber(s), {len(self)} color(s)"
        ]
        for color, rec in self._recordings.items():
            lines.append(f"  [{color}] {rec!r}")
        return "\n".join(lines)
