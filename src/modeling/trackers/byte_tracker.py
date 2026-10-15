import supervision as sv
import numpy as np


class ByteTracker:
    """
    Wrapper for ByteTrack using the Supervision library.
    Excellent for recovering 'lost' tracks in occluded/blurry scenarios.
    """

    def __init__(self):
        print("Initializing ByteTrack...")
        # track_activation_threshold: Minimum confidence to start a new track
        # lost_track_buffer: How many frames to remember a lost object
        # minimum_matching_threshold: How strict the overlap matching is
        self.lost_track_buffer = 60
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=self.lost_track_buffer,
            minimum_matching_threshold=0.8,
            frame_rate=30,
        )

    def set_frame_rate(self, frame_rate):
        """Scale ByteTrack's lost-track window to the inference cadence."""
        self.tracker.max_time_lost = max(
            1,
            int(min(max(float(frame_rate), 1.0), 30.0) / 30.0
                * self.lost_track_buffer),
        )

    def update(self, detections_list):
        """
        Input:
            detections_list = list of tuples: ([x1, y1, x2, y2], conf, class_id)
        Output:
            Detections object containing track_id
        """
        if detections_list:
            xyxy = np.asarray([d[0] for d in detections_list], dtype=float)
            confidence = np.asarray([d[1] for d in detections_list], dtype=float)
            class_id = np.asarray([d[2] for d in detections_list], dtype=int)
        else:
            xyxy = np.empty((0, 4), dtype=float)
            confidence = np.empty((0,), dtype=float)
            class_id = np.empty((0,), dtype=int)

        detections = sv.Detections(
            xyxy=xyxy, confidence=confidence, class_id=class_id.astype(int)
        )

        tracked_detections = self.tracker.update_with_detections(detections)

        return tracked_detections
