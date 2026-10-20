"""Run a reproducible, non-overwriting SmartVision YOLO training experiment."""

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import ultralytics
import yaml
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_dataset import audit_dataset
from scripts.prepare_detection_dataset import prepare_detection_dataset
from src.config import EXPECTED_DETECTOR_CLASSES, VALIDATION_DATA_PATH


RUNS_ROOT = PROJECT_ROOT / "runs" / "detect"
BASE_CHECKPOINT = PROJECT_ROOT / "yolo11s.pt"
LOCK_PATH = RUNS_ROOT / ".smartvision-training.lock"
MIN_FREE_GB = 2.0


def _class_names(configuration):
    names = configuration.get("names")
    if isinstance(names, dict):
        try:
            return [
                str(names[index]).strip().lower()
                for index in range(len(names))
            ]
        except KeyError:
            return []
    if isinstance(names, list):
        return [str(name).strip().lower() for name in names]
    return []


def _training_device(requested):
    if requested == "auto":
        return 0 if torch.cuda.is_available() else "cpu"
    if requested == "0" and not torch.cuda.is_available():
        raise ValueError("CUDA device 0 was requested, but CUDA is unavailable.")
    if requested != "cpu":
        try:
            device = int(requested)
        except ValueError as exc:
            raise ValueError("Device must be auto, cpu, or a CUDA device index.") from exc
        if not torch.cuda.is_available() or device >= torch.cuda.device_count():
            raise ValueError(f"Requested CUDA device {device} is unavailable.")
        return device
    return "cpu"


def _acquire_training_lock():
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(
            LOCK_PATH,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
        )
    except FileExistsError as exc:
        existing = LOCK_PATH.read_text(encoding="utf-8", errors="replace")
        raise RuntimeError(
            "Another training run may be active. Lock file: "
            f"{LOCK_PATH}. Contents: {existing.strip()}"
        ) from exc
    payload = {
        "pid": os.getpid(),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    with os.fdopen(descriptor, "w", encoding="utf-8") as lock_file:
        json.dump(payload, lock_file)
    return LOCK_PATH


def _validate_configuration(data_yaml):
    with data_yaml.open("r", encoding="utf-8-sig") as file:
        configuration = yaml.safe_load(file)
    actual_classes = _class_names(configuration)
    if actual_classes != list(EXPECTED_DETECTOR_CLASSES):
        raise ValueError(
            f"Dataset class map {actual_classes} does not match "
            f"{list(EXPECTED_DETECTOR_CLASSES)}."
        )
    if configuration.get("nc") != len(EXPECTED_DETECTOR_CLASSES):
        raise ValueError("Dataset nc does not match the six-class contract.")
    return configuration


def _dataset_bytes(data_yaml):
    configuration = _validate_configuration(data_yaml)
    dataset_root = Path(configuration.get("path", data_yaml.parent))
    if not dataset_root.is_absolute():
        dataset_root = data_yaml.parent / dataset_root
    total_bytes = 0
    for split_name in ("train", "val"):
        split_path = Path(configuration[split_name])
        if not split_path.is_absolute():
            split_path = dataset_root / split_path
        if not split_path.is_dir():
            raise FileNotFoundError(
                f"Configured {split_name} split does not exist: {split_path}"
            )
        split_files = {
            path
            for directory in (split_path, split_path.parent / "labels")
            if directory.is_dir()
            for path in directory.rglob("*")
            if path.is_file()
        }
        total_bytes += sum(path.stat().st_size for path in split_files)
    return configuration, total_bytes


def _experiment_configuration(
    arguments,
    device,
    source_data_yaml,
    training_data_yaml,
    data_bytes,
    audit_path,
    preparation_counts,
):
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint_source": str(
            Path(arguments.base_model).expanduser().resolve()
        ),
        "dataset_yaml": str(source_data_yaml),
        "dataset_yaml_sha256": hashlib.sha256(
            source_data_yaml.read_bytes()
        ).hexdigest(),
        "training_dataset_yaml": str(training_data_yaml),
        "dataset_bytes_train_and_val": data_bytes,
        "dataset_audit_report": str(audit_path),
        "prepared_annotation_counts": preparation_counts,
        "expected_classes": list(EXPECTED_DETECTOR_CLASSES),
        "device": str(device),
        "python": sys.version.split()[0],
        "pytorch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "training": {
            "epochs": arguments.epochs,
            "image_size": arguments.image_size,
            "batch": arguments.batch,
            "workers": arguments.workers,
            "patience": arguments.patience,
            "seed": arguments.seed,
            "resume_run": (
                str(Path(arguments.resume_run).resolve())
                if arguments.resume_run
                else None
            ),
            "augmentation": {
                "degrees": arguments.degrees,
                "scale": arguments.scale,
                "translate": arguments.translate,
                "fliplr": arguments.fliplr,
                "mosaic": arguments.mosaic,
                "mixup": arguments.mixup,
                "close_mosaic": arguments.close_mosaic,
            },
        },
    }


def run_training(arguments):
    data_yaml = Path(arguments.data).expanduser()
    base_model = Path(arguments.base_model).expanduser()
    if not data_yaml.is_absolute():
        data_yaml = PROJECT_ROOT / data_yaml
    if not base_model.is_absolute():
        base_model = PROJECT_ROOT / base_model
    data_yaml = data_yaml.resolve()
    base_model = base_model.resolve()
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Dataset configuration is missing: {data_yaml}")

    is_resuming = arguments.resume_run is not None
    run_directory = (
        Path(arguments.resume_run).expanduser().resolve()
        if arguments.resume_run is not None
        else None
    )
    if run_directory is not None:
        last_checkpoint = run_directory / "weights" / "last.pt"
        if not last_checkpoint.is_file():
            raise FileNotFoundError(
                f"Cannot resume: last checkpoint is missing: {last_checkpoint}"
            )
        model_path = last_checkpoint
    else:
        if not base_model.is_file():
            raise FileNotFoundError(
                f"Pretrained base checkpoint is missing: {base_model}"
            )
        model_path = base_model

    if arguments.batch < 1 or arguments.workers < 0:
        raise ValueError("Batch must be positive and workers cannot be negative.")
    if arguments.image_size < 320 or arguments.image_size % 32:
        raise ValueError("Image size must be >=320 and divisible by 32.")
    if min(arguments.epochs, arguments.patience) < 1:
        raise ValueError("Epochs and patience must be positive.")

    _, _source_dataset_bytes = _dataset_bytes(data_yaml)
    audit, audit_path = audit_dataset(
        data_yaml,
        output_directory=PROJECT_ROOT / "runs" / "dataset_audit"
        / f"training-preflight-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}",
    )
    if not audit["audit_passed"]:
        raise RuntimeError(
            "Dataset audit failed; refusing to train. Review "
            f"{audit_path} and fix source annotations intentionally."
        )

    device = _training_device(arguments.device)
    usage = shutil.disk_usage(RUNS_ROOT if RUNS_ROOT.exists() else PROJECT_ROOT)
    free_gb = usage.free / (1024 ** 3)
    print(f"Dataset audit: {audit_path}")
    print(f"Source training/validation bytes: {_source_dataset_bytes:,}")
    print(f"Available disk space: {free_gb:.2f} GiB")
    if free_gb < MIN_FREE_GB:
        raise RuntimeError(
            f"Only {free_gb:.2f} GiB are free; training requires at least "
            f"{MIN_FREE_GB:.1f} GiB of safety margin."
        )
    if free_gb < 5.0:
        print(
            "Resource warning: less than 5 GiB is free. Keep cache disabled "
            "and monitor the output volume."
        )

    lock_path = _acquire_training_lock()
    try:
        if run_directory is not None:
            run_name = run_directory.name
            project = run_directory.parent
        else:
            project = RUNS_ROOT
            run_name = (
                "smartvision_weapon_"
                + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            )
            run_directory = project / run_name
            if run_directory.exists():
                raise FileExistsError(
                    f"Refusing to reuse existing experiment: {run_directory}"
                )

        configuration_path = run_directory / "smartvision_experiment.json"
        if is_resuming:
            if not configuration_path.is_file():
                raise FileNotFoundError(
                    "Resume run does not contain smartvision_experiment.json."
                )
            experiment = json.loads(
                configuration_path.read_text(encoding="utf-8")
            )
            if experiment.get("dataset_yaml") != str(data_yaml):
                raise ValueError(
                    "Resume run was created from another dataset configuration."
                )
            if experiment.get("dataset_yaml_sha256") != hashlib.sha256(
                data_yaml.read_bytes()
            ).hexdigest():
                raise ValueError(
                    "Dataset YAML changed since the selected run; refusing "
                    "to resume with a different dataset definition."
                )
            training_data_yaml = Path(
                experiment.get("training_dataset_yaml", "")
            ).resolve()
            if not training_data_yaml.is_file():
                raise FileNotFoundError(
                    "The detection-format staging dataset for this run is "
                    f"missing: {training_data_yaml}"
                )
            resumed_audit, resumed_audit_path = audit_dataset(
                training_data_yaml,
                output_directory=PROJECT_ROOT / "runs" / "dataset_audit"
                / f"resume-preflight-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}",
            )
            if not resumed_audit["audit_passed"]:
                raise RuntimeError(
                    "Resume dataset audit failed; refusing to resume. "
                    f"Review {resumed_audit_path}."
                )
        else:
            run_directory.mkdir(parents=True, exist_ok=False)
            staging_root = (
                RUNS_ROOT.parent / "prepared_datasets" / run_name
            )
            staging_root.parent.mkdir(parents=True, exist_ok=True)
            staging_root.mkdir(parents=True, exist_ok=False)
            training_data_yaml, preparation_counts = prepare_detection_dataset(
                data_yaml,
                staging_root,
            )
            staged_audit, staged_audit_path = audit_dataset(
                training_data_yaml,
                output_directory=PROJECT_ROOT / "runs" / "dataset_audit"
                / f"staged-preflight-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}",
            )
            if not staged_audit["audit_passed"]:
                raise RuntimeError(
                    "Detection-format staging audit failed; refusing to "
                    f"train. Review {staged_audit_path}."
                )
            _, dataset_bytes = _dataset_bytes(
                training_data_yaml
            )
            experiment = _experiment_configuration(
                arguments,
                device,
                data_yaml,
                training_data_yaml,
                dataset_bytes,
                staged_audit_path,
                preparation_counts,
            )
            configuration_path.write_text(
                json.dumps(experiment, indent=2),
                encoding="utf-8",
            )

        model = YOLO(str(model_path))
        if _class_names({"names": model.names}) != list(
            EXPECTED_DETECTOR_CLASSES
        ) and arguments.resume_run:
            raise ValueError(
                "Resume checkpoint class mapping is incompatible with the dataset."
            )

        train_options = {
            "data": str(training_data_yaml),
            "device": device,
            "project": str(project),
            "name": run_name,
            "exist_ok": bool(arguments.resume_run),
            "save": True,
            "save_period": -1,
            "plots": True,
            "verbose": True,
            "workers": arguments.workers,
            "cache": False,
            "amp": bool(torch.cuda.is_available() and device != "cpu"),
            "seed": arguments.seed,
            "deterministic": True,
            "resume": is_resuming,
        }
        if not is_resuming:
            train_options.update({
                "epochs": arguments.epochs,
                "imgsz": arguments.image_size,
                "batch": arguments.batch,
                "patience": arguments.patience,
                "pretrained": True,
                "degrees": arguments.degrees,
                "scale": arguments.scale,
                "translate": arguments.translate,
                "fliplr": arguments.fliplr,
                "flipud": 0.0,
                "mosaic": arguments.mosaic,
                "mixup": arguments.mixup,
                "close_mosaic": arguments.close_mosaic,
            })

        started = time.perf_counter()
        print(f"Training device: {device}")
        print(f"Base checkpoint: {model_path}")
        print(f"Experiment directory: {run_directory}")
        results = model.train(**train_options)
        elapsed = time.perf_counter() - started

        best = run_directory / "weights" / "best.pt"
        last = run_directory / "weights" / "last.pt"
        if not best.is_file() or not last.is_file():
            raise RuntimeError(
                "Training returned without both weights/best.pt and "
                f"weights/last.pt in {run_directory}."
            )
        if _class_names({"names": YOLO(str(best)).names}) != list(
            EXPECTED_DETECTOR_CLASSES
        ):
            raise RuntimeError("Trained checkpoint class map is incorrect.")

        completion = {
            **experiment,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": elapsed,
            "best_checkpoint": str(best),
            "last_checkpoint": str(last),
            "training_results": (
                str(getattr(results, "save_dir", run_directory))
            ),
            "candidate_selection": (
                "Not selected for production. Evaluate every candidate with "
                "scripts/validate_detector.py using identical settings."
            ),
        }
        (run_directory / "smartvision_training_completion.json").write_text(
            json.dumps(completion, indent=2, default=str),
            encoding="utf-8",
        )
        print(f"Training completed in {elapsed / 3600:.2f} hours.")
        print(f"Best checkpoint: {best}")
        print(f"Last checkpoint: {last}")
        print(
            "This experiment is a candidate only. Run the validation "
            "evaluator before model selection."
        )
    finally:
        try:
            lock_path.unlink(missing_ok=True)
        except OSError:
            print(
                f"Warning: could not remove training lock {lock_path}. "
                "Inspect it before another run."
            )


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(VALIDATION_DATA_PATH))
    parser.add_argument("--base-model", type=Path, default=BASE_CHECKPOINT)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume-run", type=Path)
    parser.add_argument("--degrees", type=float, default=5.0)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--translate", type=float, default=0.1)
    parser.add_argument("--fliplr", type=float, default=0.5)
    parser.add_argument("--mosaic", type=float, default=1.0)
    parser.add_argument("--mixup", type=float, default=0.1)
    parser.add_argument("--close-mosaic", type=int, default=10)
    return parser


def main():
    try:
        run_training(build_parser().parse_args())
    except Exception as exc:
        print(f"Training failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
