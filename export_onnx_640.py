"""Export a trained YOLOv10 model to a validated static-shape ONNX model."""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from ultralytics import YOLOv10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Path to a trained YOLOv10 checkpoint.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output ONNX path. Defaults to <weights>_640x640.onnx.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=13,
        help="ONNX opset version (default: 13).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights = args.weights.expanduser().resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"Weights file not found: {weights}")

    output = (
        args.output.expanduser().resolve()
        if args.output
        else weights.with_name(f"{weights.stem}_640x640.onnx")
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    model = YOLOv10(str(weights))
    exported = Path(
        model.export(
            format="onnx",
            imgsz=(640, 640),
            batch=1,
            device="cpu",
            dynamic=False,
            simplify=True,
            opset=args.opset,
        )
    ).resolve()

    if exported != output:
        output.write_bytes(exported.read_bytes())

    onnx_model = onnx.load(str(output))
    onnx.checker.check_model(onnx_model)
    input_shape = [
        dim.dim_value
        for dim in onnx_model.graph.input[0].type.tensor_type.shape.dim
    ]
    expected_shape = [1, 3, 640, 640]
    if input_shape != expected_shape:
        raise RuntimeError(
            f"Unexpected ONNX input shape: {input_shape}; expected {expected_shape}"
        )

    print(f"Exported: {output}")
    print(f"Input shape: {input_shape}")
    print(f"Classes: {len(model.names)} ({model.names})")


if __name__ == "__main__":
    main()
