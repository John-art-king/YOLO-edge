"""Evaluate the sigmoid and relu YOLOv10 checkpoints on the date dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


COCO_NAMES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    9: "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    12: "parking meter",
    13: "bench",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    29: "frisbee",
    30: "skis",
    31: "snowboard",
    32: "sports ball",
    33: "kite",
    34: "baseball bat",
    35: "baseball glove",
    36: "skateboard",
    37: "surfboard",
    38: "tennis racket",
    39: "bottle",
    40: "wine glass",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table",
    61: "toilet",
    62: "tv",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    68: "microwave",
    69: "oven",
    70: "toaster",
    71: "sink",
    72: "refrigerator",
    73: "book",
    74: "clock",
    75: "vase",
    76: "scissors",
    77: "teddy bear",
    78: "hair drier",
    79: "toothbrush",
}

DEFAULT_MODELS = (
    "runs/sigmoid/best.pt",
    "runs/sigmoid/best.onnx",
    "runs/relu/best.pt",
    "runs/relu/best.onnx",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("date"),
        help="Dataset directory containing images/ and labels/ (default: date).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        type=Path,
        default=[Path(path) for path in DEFAULT_MODELS],
        help="Model paths to evaluate (default: sigmoid/relu best.pt and best.onnx).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Override the validation image size. By default it is read from ONNX models or set to 640 for PT.",
    )
    parser.add_argument("--batch", type=int, default=1, help="Validation batch size (default: 1).")
    parser.add_argument("--workers", type=int, default=0, help="Data loader workers (default: 0 on Windows).")
    parser.add_argument("--device", default="cpu", help="Device, for example cpu, 0, or 0,1 (default: cpu).")
    parser.add_argument(
        "--project",
        type=Path,
        default=Path("runs/model_eval"),
        help="Directory for Ultralytics validation outputs (default: runs/model_eval).",
    )
    parser.add_argument("--name", default=None, help="Optional name for a single validation run.")
    parser.add_argument("--plots", action="store_true", help="Save PR/confusion-matrix/label plots.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check paths and print the planned evaluations without loading models.",
    )
    return parser.parse_args()


def resolve_path(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else (root / path).resolve()


def validate_dataset(data_dir: Path) -> None:
    image_dir = data_dir / "images"
    label_dir = data_dir / "labels"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Missing dataset image directory: {image_dir}")
    if not label_dir.is_dir():
        raise FileNotFoundError(f"Missing dataset label directory: {label_dir}")
    image_count = sum(1 for path in image_dir.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
    label_count = sum(1 for path in label_dir.rglob("*.txt"))
    if image_count == 0:
        raise FileNotFoundError(f"No images found under: {image_dir}")
    if label_count == 0:
        raise FileNotFoundError(f"No YOLO label files found under: {label_dir}")
    print(f"Dataset: {data_dir} ({image_count} images, {label_count} labels)")


def write_dataset_yaml(data_dir: Path, output_dir: Path) -> Path:
    """Create a dataset config without changing the user's dataset directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = output_dir / "date_eval.yaml"
    lines = [f"path: {json.dumps(str(data_dir))}", "train: images", "val: images", "test: images", "names:"]
    lines.extend(f"  {class_id}: {json.dumps(name)}" for class_id, name in COCO_NAMES.items())
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return yaml_path


def onnx_image_size(model_path: Path) -> int | tuple[int, int] | None:
    """Return the model's fixed ONNX input size, or None when it is dynamic."""
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("ONNX evaluation requires onnxruntime. Install it in the active environment.") from exc

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    shape = session.get_inputs()[0].shape
    if len(shape) != 4 or not all(isinstance(value, int) for value in shape[2:]):
        return None
    return int(shape[2]) if shape[2] == shape[3] else (int(shape[2]), int(shape[3]))


def model_image_size(model_path: Path, override: int | None) -> int | tuple[int, int]:
    if override is not None:
        if model_path.suffix.lower() == ".onnx":
            detected = onnx_image_size(model_path)
            if detected is not None and detected != override:
                raise ValueError(
                    f"{model_path.name} has fixed input size {detected}; do not use --imgsz {override}."
                )
        return override
    detected = onnx_image_size(model_path) if model_path.suffix.lower() == ".onnx" else None
    return detected if detected is not None else 640


def json_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def evaluate_model(model_path: Path, data_yaml: Path, args: argparse.Namespace, root: Path) -> dict[str, Any]:
    from ultralytics import YOLOv10

    image_size = model_image_size(model_path, args.imgsz)
    label = args.name or f"{model_path.parent.name}_{model_path.stem}_{model_path.suffix[1:]}"
    print(f"\nEvaluating {model_path} | imgsz={image_size} | device={args.device}")
    model = YOLOv10(str(model_path))
    metrics = model.val(
        data=str(data_yaml),
        imgsz=image_size,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=str(resolve_path(args.project, root)),
        name=label,
        exist_ok=True,
        plots=args.plots,
        verbose=True,
    )
    result: dict[str, Any] = {
        "model": str(model_path),
        "format": model_path.suffix.lower().lstrip("."),
        "imgsz": image_size,
        "metrics": json_value(getattr(metrics, "results_dict", {})),
    }
    box = getattr(metrics, "box", None)
    if box is not None:
        result["precision"] = json_value(getattr(box, "mp", None))
        result["recall"] = json_value(getattr(box, "mr", None))
        result["mAP50"] = json_value(getattr(box, "map50", None))
        result["mAP50-95"] = json_value(getattr(box, "map", None))
    return result


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parent
    data_dir = resolve_path(args.data, root)
    model_paths = [resolve_path(path, root) for path in args.models]
    output_dir = resolve_path(args.project, root)

    validate_dataset(data_dir)
    missing = [path for path in model_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing model file(s):\n" + "\n".join(str(path) for path in missing))

    data_yaml = write_dataset_yaml(data_dir, output_dir)
    print(f"Dataset YAML: {data_yaml}")
    print("Models:")
    for path in model_paths:
        print(f"  - {path}")

    if args.dry_run:
        for path in model_paths:
            print(f"  {path.name}: imgsz={model_image_size(path, args.imgsz)}")
        return 0

    results = []
    failures = []
    for model_path in model_paths:
        try:
            results.append(evaluate_model(model_path, data_yaml, args, root))
        except Exception as exc:  # Keep the other model evaluations running.
            failures.append({"model": str(model_path), "error": f"{type(exc).__name__}: {exc}"})
            print(f"ERROR: {model_path}: {exc}", file=sys.stderr)

    summary = {"dataset": str(data_dir), "results": results, "failures": failures}
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nSummary: {summary_path}")
    if failures:
        print(f"Completed with {len(failures)} failure(s).", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
