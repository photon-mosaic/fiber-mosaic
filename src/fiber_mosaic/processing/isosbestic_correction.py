from __future__ import annotations

from spikeinterface.core import BaseRecording

from fiber_mosaic.processing.baseprocessor import (
    BaseFiberPhotometryPreprocessor,
    BaseFiberPhotometryPreprocessorSegment,
)


class IsosbesticCorrectionSegment(BaseFiberPhotometryPreprocessorSegment):
    def __init__(
        self,
        recording_segment,
        reference_recording_segment,
    ) -> None:
        BaseFiberPhotometryPreprocessorSegment.__init__(
            self, recording_segment
        )
        self.reference_recording_segment = reference_recording_segment

    def get_traces(
        self,
        start_frame,
        end_frame,
        channel_indices,
    ):
        signal = self.recording_segment.get_traces(
            start_frame,
            end_frame,
            channel_indices,
        ).astype("float32", copy=True)

        reference = self.reference_recording_segment.get_traces(
            start_frame,
            end_frame,
            channel_indices,
        )
        # dummy correction for now, to be controlled with an kwarg
        signal /= reference
        return signal


class IsosbesticCorrectionRecording(BaseFiberPhotometryPreprocessor):
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


# this is probably not the way
def isosbestic_correction(
    recording: BaseRecording,
    reference_recording: BaseRecording,
) -> IsosbesticCorrectionRecording:
    return IsosbesticCorrectionRecording(recording, reference_recording)
