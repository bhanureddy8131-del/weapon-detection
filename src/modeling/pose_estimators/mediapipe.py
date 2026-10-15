import logging
from pathlib import Path

import mediapipe as mp
import cv2

from src.config import POSE_MODEL_PATH


class MediaPipeEstimator:
    """
    Wrapper for MediaPipe Pose.
    Extracts 33 skeletal landmarks from a person's image.
    """

    def __init__(self):
        self.available = False
        self.diagnostic = ""
        self.last_elapsed_ms = 0.0
        self.pose = None
        self._task_api = False
        self._mp = mp

        solutions_api = getattr(mp, "solutions", None)
        if solutions_api is not None and hasattr(solutions_api, "pose"):
            try:
                self.mp_pose = solutions_api.pose
                self.mp_drawing = solutions_api.drawing_utils
                self.mp_drawing_styles = solutions_api.drawing_styles
                self.pose = self.mp_pose.Pose(
                    static_image_mode=False,
                    model_complexity=0,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                self.available = True
                self.diagnostic = "MediaPipe Pose legacy API initialized."
                return
            except Exception as exc:
                self.diagnostic = f"Legacy MediaPipe Pose initialization failed: {exc}"
                logging.exception(self.diagnostic)

        model_path = Path(POSE_MODEL_PATH)
        if not model_path.is_file():
            api_note = (
                "The installed MediaPipe package exposes Tasks, not "
                "solutions.pose."
                if solutions_api is None or not hasattr(solutions_api, "pose")
                else "The legacy Pose API could not be initialized."
            )
            self.diagnostic = (
                f"{api_note} Pose Landmarker model is missing at {model_path}. "
                "Download pose_landmarker_lite.task using the command in README.md."
            )
            return

        try:
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python import vision

            options = vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(model_path)),
                running_mode=vision.RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_segmentation_masks=False,
            )
            self.pose = vision.PoseLandmarker.create_from_options(options)
            self._task_api = True
            self.available = True
            self.diagnostic = f"MediaPipe Pose Landmarker loaded from {model_path}."
        except Exception as exc:
            self.pose = None
            self.diagnostic = f"MediaPipe Pose Landmarker initialization failed: {exc}"
            logging.exception(self.diagnostic)

    def predict(self, person_crop):
        """
        Input: BGR Image (Crop of a person)
        Output: List of 33 landmarks [x, y, z, visibility, ...]
                Returns None if no pose detected or estimator unavailable.
        """
        if not self.available or self.pose is None:
            return None, None

        started_at = cv2.getTickCount()
        try:
            img_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
            if self._task_api:
                image = self._mp.Image(
                    image_format=self._mp.ImageFormat.SRGB,
                    data=img_rgb,
                )
                results = self.pose.detect(image)
                pose_landmarks = (
                    results.pose_landmarks[0]
                    if results.pose_landmarks
                    else None
                )
            else:
                results = self.pose.process(img_rgb)
                pose_landmarks = (
                    results.pose_landmarks
                    if results and results.pose_landmarks
                    else None
                )

            if pose_landmarks is None:
                return None, None

            landmark_items = (
                pose_landmarks.landmark
                if hasattr(pose_landmarks, "landmark")
                else pose_landmarks
            )
            landmarks = []
            for landmark in landmark_items:
                landmarks.extend([
                    landmark.x,
                    landmark.y,
                    landmark.z,
                    getattr(landmark, "visibility", 0.0),
                ])
            return landmarks, pose_landmarks
        except Exception as exc:
            self.diagnostic = f"MediaPipe pose inference failed: {exc}"
            logging.exception(self.diagnostic)
            return None, None
        finally:
            elapsed_ticks = cv2.getTickCount() - started_at
            self.last_elapsed_ms = (
                elapsed_ticks / cv2.getTickFrequency() * 1000
            )

    def visualize(self, person_crop, landmarks_object):
        """
        Draws the skeleton on the cropped image in-place.
        """
        if not self.available or landmarks_object is None or self._task_api:
            return person_crop

        try:
            self.mp_drawing.draw_landmarks(
                person_crop,
                landmarks_object,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style(),
            )
        except Exception as exc:
            logging.exception("Could not draw MediaPipe pose landmarks: %s", exc)
        return person_crop

    def close(self):
        if self.pose is not None:
            self.pose.close()
            self.pose = None
            self.available = False
