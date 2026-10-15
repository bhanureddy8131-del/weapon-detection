from abc import ABC, abstractmethod
from typing import Any

class BaseDetector(ABC):

    @abstractmethod
    def load_model(self, model_path: Any) -> None:
        """Loads the model from a file."""
        pass

    @abstractmethod
    def predict(self, frame: Any) -> Any:
        """Runs inference on a single frame."""
        pass