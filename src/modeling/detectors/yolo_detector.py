from ultralytics import YOLO
from src.utils.base_detector import BaseDetector
import os

class YOLODetector(BaseDetector):

    def __init__(self):
        self.model = None
        self.class_names = None
        print(" YOLODetector initialized.")

    def load_model(self, model_path):
        if not os.path.exists(model_path) or os.path.getsize(model_path) < 10000:
            print(f" Custom model file at {model_path} missing or Git LFS pointer. Falling back to 'yolo11s.pt'...")
            model_path = "yolo11s.pt"

        print(f" Loading YOLO model from: {model_path}...")
        try:
            self.model = YOLO(model_path)
            self.class_names = self.model.names
            print(f" Model loaded successfully!")
            print(f" Classes found: {self.class_names}")
        except Exception as e:
            print(f" Error loading model: {e}")
            raise e

    def predict(self, frame, conf=0.5):
        """
        Returns the raw results object from Ultralytics.
        """
        if self.model is None:
            raise RuntimeError(" Model has not been loaded. Call load_model() first.")
            
        results = self.model.predict(frame, conf=conf, verbose=False)
        return results