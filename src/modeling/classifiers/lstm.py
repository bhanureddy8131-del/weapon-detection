from collections import deque
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


POSE_FEATURE_SIZE = 33 * 4
POSE_FEATURE_LAYOUT = "mediapipe_pose_xyz_visibility_v1"
ACTION_CLASSES = frozenset({"normal", "fighting", "shooting"})
ACTION_ARCHITECTURE = "bidirectional_lstm"


class ActionModelCompatibilityError(ValueError):
    """Raised when a checkpoint cannot be verified against inference inputs."""


class PoseSequenceBuffer:
    """Collect consecutive, valid MediaPipe poses for one person track."""

    def __init__(self, sequence_length: int, feature_size: int = POSE_FEATURE_SIZE):
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if feature_size <= 0:
            raise ValueError("feature_size must be positive")
        self.sequence_length = sequence_length
        self.feature_size = feature_size
        self._frames = deque(maxlen=sequence_length)

    def __len__(self):
        return len(self._frames)

    @property
    def ready(self):
        return len(self._frames) == self.sequence_length

    def clear(self):
        self._frames.clear()

    def append(self, features) -> bool:
        if features is None:
            self.clear()
            return False

        try:
            values = np.asarray(features, dtype=np.float32)
        except (TypeError, ValueError):
            self.clear()
            return False

        if (
            values.shape != (self.feature_size,)
            or not np.isfinite(values).all()
        ):
            self.clear()
            return False

        self._frames.append(values.copy())
        return True

    def as_tensor(self, device=None) -> torch.Tensor:
        if not self.ready:
            raise ValueError(
                "Cannot create an action tensor before a complete valid "
                f"sequence is collected ({len(self)}/{self.sequence_length})."
            )
        frames = np.stack(tuple(self._frames)).astype(np.float32, copy=False)
        return torch.from_numpy(frames.copy()).unsqueeze(0).to(device=device)


class BidirectionalLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        dropout_rate: float = 0.4,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        if x.ndim != 3 or x.shape[-1] != self.input_size:
            raise ValueError(
                "Expected LSTM input shaped (batch, sequence, "
                f"{self.input_size}); received {tuple(x.shape)}."
            )
        hidden = x.new_zeros(
            self.num_layers * 2,
            x.size(0),
            self.hidden_size,
        )
        cell = torch.zeros_like(hidden)
        output, _ = self.lstm(x, (hidden, cell))
        return self.fc(output[:, -1, :])


def _validated_class_map(class_map) -> dict[str, int]:
    if not isinstance(class_map, Mapping):
        raise ActionModelCompatibilityError("Checkpoint class_map must be a mapping.")

    normalized = {
        str(name).strip().lower(): int(class_id)
        for name, class_id in class_map.items()
    }
    if set(normalized) != ACTION_CLASSES:
        raise ActionModelCompatibilityError(
            "Checkpoint actions must be exactly "
            f"{sorted(ACTION_CLASSES)}; found {sorted(normalized)}."
        )
    class_ids = list(normalized.values())
    if sorted(class_ids) != list(range(len(ACTION_CLASSES))):
        raise ActionModelCompatibilityError(
            "Checkpoint class_map IDs must be unique and contiguous from zero."
        )
    return normalized


def load_action_model_from_checkpoint(
    checkpoint,
    device,
    sequence_length: int,
    feature_size: int = POSE_FEATURE_SIZE,
) -> tuple[BidirectionalLSTM, dict[int, str]]:
    if not isinstance(checkpoint, Mapping):
        raise ActionModelCompatibilityError("LSTM checkpoint must be a mapping.")

    required_fields = {
        "model_state_dict",
        "class_map",
        "input_size",
        "hidden_size",
        "num_layers",
        "bidirectional",
        "sequence_length",
        "feature_layout",
        "architecture",
        "dropout_rate",
    }
    missing_fields = sorted(required_fields - checkpoint.keys())
    if missing_fields:
        raise ActionModelCompatibilityError(
            "Checkpoint is missing compatibility metadata: "
            + ", ".join(missing_fields)
        )

    class_map = _validated_class_map(checkpoint["class_map"])
    if int(checkpoint["input_size"]) != feature_size:
        raise ActionModelCompatibilityError(
            f"Checkpoint input_size is {checkpoint['input_size']}; "
            f"MediaPipe preprocessing produces {feature_size} features."
        )
    if int(checkpoint["sequence_length"]) != sequence_length:
        raise ActionModelCompatibilityError(
            f"Checkpoint sequence_length is {checkpoint['sequence_length']}; "
            f"configured inference sequence length is {sequence_length}."
        )
    if checkpoint["feature_layout"] != POSE_FEATURE_LAYOUT:
        raise ActionModelCompatibilityError(
            f"Unsupported checkpoint feature layout: {checkpoint['feature_layout']!r}."
        )
    if checkpoint["architecture"] != ACTION_ARCHITECTURE:
        raise ActionModelCompatibilityError(
            f"Unsupported LSTM architecture: {checkpoint['architecture']!r}."
        )
    if checkpoint["bidirectional"] is not True:
        raise ActionModelCompatibilityError(
            "Checkpoint is not bidirectional; it does not match the inference model."
        )

    hidden_size = int(checkpoint["hidden_size"])
    num_layers = int(checkpoint["num_layers"])
    dropout_rate = float(checkpoint["dropout_rate"])
    if hidden_size <= 0 or num_layers <= 0 or not 0.0 <= dropout_rate < 1.0:
        raise ActionModelCompatibilityError(
            "Checkpoint hidden_size, num_layers, or dropout_rate is invalid."
        )

    model = BidirectionalLSTM(
        feature_size,
        hidden_size,
        num_layers,
        len(class_map),
        dropout_rate=dropout_rate,
    ).to(device)
    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    except (RuntimeError, TypeError) as exc:
        raise ActionModelCompatibilityError(
            f"Checkpoint tensor shapes do not match the bidirectional LSTM: {exc}"
        ) from exc

    model.eval()
    inverse_map = {class_id: name for name, class_id in class_map.items()}
    with torch.inference_mode():
        output = model(torch.zeros(1, sequence_length, feature_size, device=device))
    if tuple(output.shape) != (1, len(class_map)):
        raise ActionModelCompatibilityError(
            "Loaded LSTM returned an incompatible output shape: "
            f"{tuple(output.shape)}."
        )
    return model, inverse_map


def load_action_model(
    checkpoint_path: Path | str,
    device,
    sequence_length: int,
    feature_size: int = POSE_FEATURE_SIZE,
) -> tuple[BidirectionalLSTM, dict[int, str]]:
    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.is_file():
        raise FileNotFoundError(
            f"LSTM checkpoint does not exist: {checkpoint_file}"
        )
    try:
        checkpoint = torch.load(
            str(checkpoint_file),
            map_location=device,
            weights_only=True,
        )
    except Exception as exc:
        raise ActionModelCompatibilityError(
            f"Could not read LSTM checkpoint {checkpoint_file}: {exc}"
        ) from exc
    return load_action_model_from_checkpoint(
        checkpoint,
        device,
        sequence_length,
        feature_size,
    )


def format_action_prediction(
    logits: torch.Tensor,
    inverse_class_map: Mapping[int, str],
    confidence_threshold: float,
) -> Optional[dict[str, float | int | str]]:
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be between zero and one.")
    if logits.ndim != 2 or logits.shape[0] != 1:
        raise ValueError(
            f"Expected one logits row shaped (1, classes); received {tuple(logits.shape)}."
        )
    if logits.shape[1] != len(inverse_class_map):
        raise ValueError("LSTM output count does not match the checkpoint class map.")
    if not torch.isfinite(logits).all():
        raise ValueError("LSTM output contains non-finite logits.")

    probabilities = torch.softmax(logits, dim=1)
    confidence, class_index = torch.max(probabilities, dim=1)
    score = float(confidence.item())
    if score < confidence_threshold:
        return None

    class_id = int(class_index.item())
    action = inverse_class_map.get(class_id)
    if action not in ACTION_CLASSES:
        raise ValueError(f"LSTM output class {class_id} is not in the validated class map.")
    return {
        "class": action,
        "class_id": class_id,
        "confidence": score,
    }


def predict_action(
    model: Optional[BidirectionalLSTM],
    sequence: PoseSequenceBuffer,
    inverse_class_map: Mapping[int, str],
    device,
    confidence_threshold: float,
) -> Optional[dict[str, float | int | str]]:
    if model is None or not sequence.ready:
        return None
    with torch.inference_mode():
        logits = model(sequence.as_tensor(device))
    return format_action_prediction(
        logits,
        inverse_class_map,
        confidence_threshold,
    )
