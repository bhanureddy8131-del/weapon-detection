import logging
import math

import supervision as sv
import numpy as np

logger = logging.getLogger(__name__)


class ByteTracker:
    """
    Wrapper for ByteTrack using the Supervision library.
    Excellent for recovering 'lost' tracks in occluded/blurry scenarios.
    """

    def __init__(self):
        self.lost_track_buffer = 60
        self.frame_rate = 30.0
        self.reset()

    def reset(self):
        """Start a clean tracker for a new source while preserving its cadence."""
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=self.lost_track_buffer,
            minimum_matching_threshold=0.8,
            frame_rate=max(1, int(round(self.frame_rate))),
        )
        self._apply_lost_window()

    def _apply_lost_window(self):
        cadence = min(max(self.frame_rate, 1.0), 30.0)
        self.tracker.max_time_lost = max(
            1,
            int(cadence / 30.0 * self.lost_track_buffer),
        )

    @staticmethod
    def _box_iou(first, second):
        x1 = max(first[0], second[0])
        y1 = max(first[1], second[1])
        x2 = min(first[2], second[2])
        y2 = min(first[3], second[3])
        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        first_area = (first[2] - first[0]) * (first[3] - first[1])
        second_area = (second[2] - second[0]) * (second[3] - second[1])
        union = first_area + second_area - intersection
        return intersection / union if union > 0 else 0.0

    @classmethod
    def _validated_detections(cls, detections_list):
        candidates = []
        for detection in detections_list or []:
            try:
                box, confidence, class_id = detection
                box = np.asarray(box, dtype=np.float64)
                confidence = float(confidence)
                class_id_float = float(class_id)
                if (
                    box.shape != (4,)
                    or not np.isfinite(box).all()
                    or not math.isfinite(confidence)
                    or not math.isfinite(class_id_float)
                    or not 0.0 <= confidence <= 1.0
                    or not class_id_float.is_integer()
                    or box[2] <= box[0]
                    or box[3] <= box[1]
                ):
                    continue
                candidates.append((box, confidence, int(class_id_float)))
            except (TypeError, ValueError):
                continue

        candidates.sort(key=lambda item: item[1], reverse=True)
        accepted = []
        for candidate in candidates:
            box, _, class_id = candidate
            duplicate = any(
                accepted_class_id == class_id
                and cls._box_iou(box, accepted_box) >= 0.95
                for accepted_box, _, accepted_class_id in accepted
            )
            if not duplicate:
                accepted.append(candidate)
        return accepted

    def set_frame_rate(self, frame_rate):
        """Scale ByteTrack's lost-track window to the inference cadence."""
        try:
            frame_rate = float(frame_rate)
        except (TypeError, ValueError):
            frame_rate = 1.0
        if not math.isfinite(frame_rate):
            frame_rate = 1.0
        self.frame_rate = min(max(frame_rate, 1.0), 30.0)
        self._apply_lost_window()

    def update(self, detections_list):
        """
        Input:
            detections_list = list of tuples: ([x1, y1, x2, y2], conf, class_id)
        Output:
            Detections object containing track_id
        """
        raw_count = len(detections_list) if detections_list is not None else 0
        validated = self._validated_detections(detections_list)
        if raw_count > len(validated):
            logger.debug(
                "Ignored %s invalid or duplicate tracker detection(s).",
                raw_count - len(validated),
            )
        if validated:
            xyxy = np.asarray([d[0] for d in validated], dtype=np.float32)
            confidence = np.asarray([d[1] for d in validated], dtype=np.float32)
            class_id = np.asarray([d[2] for d in validated], dtype=int)
        else:
            xyxy = np.empty((0, 4), dtype=np.float32)
            confidence = np.empty((0,), dtype=np.float32)
            class_id = np.empty((0,), dtype=int)

        detections = sv.Detections(
            xyxy=xyxy, confidence=confidence, class_id=class_id.astype(int)
        )

        tracked_detections = self.tracker.update_with_detections(detections)
        tracker_ids = getattr(tracked_detections, "tracker_id", None)
        if tracker_ids is None:
            return tracked_detections[
                np.zeros(len(tracked_detections), dtype=bool)
            ]

        tracker_ids = np.asarray(tracker_ids)
        valid_indices = []
        for index, track_id in enumerate(tracker_ids):
            try:
                numeric_id = float(track_id)
            except (TypeError, ValueError):
                continue
            if math.isfinite(numeric_id) and numeric_id >= 0 and numeric_id.is_integer():
                valid_indices.append(index)

        def detection_confidence(index):
            if confidences is None or index >= len(confidences):
                return 0.0
            try:
                value = float(confidences[index])
            except (TypeError, ValueError):
                return 0.0
            return value if math.isfinite(value) else 0.0

        best_index_by_id = {}
        confidences = getattr(tracked_detections, "confidence", None)
        for index in valid_indices:
            track_id = int(tracker_ids[index])
            confidence = detection_confidence(index)
            previous = best_index_by_id.get(track_id)
            if previous is None:
                best_index_by_id[track_id] = index
                continue
            previous_confidence = detection_confidence(previous)
            if confidence > previous_confidence:
                best_index_by_id[track_id] = index

        selected_indices = np.asarray(
            sorted(best_index_by_id.values()),
            dtype=int,
        )
        if len(selected_indices) < len(tracked_detections):
            logger.debug(
                "Filtered %s invalid or duplicate ByteTrack output(s).",
                len(tracked_detections) - len(selected_indices),
            )
        return tracked_detections[selected_indices]
