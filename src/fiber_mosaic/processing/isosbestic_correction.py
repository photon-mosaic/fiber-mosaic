"""Isosbestic correction preprocessing step."""

from __future__ import annotations

from spikeinterface.core import BaseRecording
from spikeinterface.preprocessing.basepreprocessor import (
    BasePreprocessorSegment,
)

from fiber_mosaic.processing.baseprocessor import (
    BaseFiberPhotometryPreprocessor,
)


class IsosbesticCorrectionSegment(BasePreprocessorSegment):
    """Segment-level isosbestic correction."""

    # TODO implement
    #  - the robust regressions (different loss functions, MAE, huber) + IRLS
    #  - choice of correction method: subtract, divide, substract-divide
    #  - output color naming? should we append _isosbestic-corrected

    def __init__(
        self,
        recording_segment,
        reference_recording_segment,
    ) -> None:
        BasePreprocessorSegment.__init__(self, recording_segment)
        self.reference_recording_segment = reference_recording_segment

    def get_traces(
        self,
        start_frame,
        end_frame,
        channel_indices,
    ):
        """Return signal divided by reference (isosbestic correction)."""
        signal = self.parent_recording_segment.get_traces(
            start_frame,
            end_frame,
            channel_indices,
        ).astype("float32", copy=True)

        reference = self.reference_recording_segment.get_traces(
            start_frame,
            end_frame,
            channel_indices,
        )
        # dummy correction for now
        signal /= reference
        return signal


class IsosbesticCorrectionRecording(BaseFiberPhotometryPreprocessor):
    """Recording-level isosbestic correction preprocessor."""

    def __init__(
        self,
        recording: BaseRecording,
        reference_recording: BaseRecording,
    ) -> None:
        BaseFiberPhotometryPreprocessor.__init__(
            self,
            recording,
            dtype="float32",
        )

        for index, segment in enumerate(recording.segments):
            self.add_recording_segment(
                IsosbesticCorrectionSegment(
                    segment,
                    reference_recording.segments[index],
                )
            )

        self._kwargs = dict(
            recording=recording, reference_recording=reference_recording
        )


def isosbestic_correction(
    recording: BaseRecording, reference_recording: BaseRecording
) -> IsosbesticCorrectionRecording:
    """
    Correct a signal recording with its isosbestic reference.

    Takes one signal and one reference, never a group: a group does not
    define which color corrects which.

    Parameters
    ----------
    recording : BaseRecording
        The signal recording, e.g. ``group["green"]``.
    reference_recording : BaseRecording
        The isosbestic reference, e.g. ``group["iso"]``.

    Returns
    -------
    IsosbesticCorrectionRecording
        The corrected signal.
    """
    return IsosbesticCorrectionRecording(recording, reference_recording)


# how to pipeline: specific processing for each branch
# replacement of workspace
