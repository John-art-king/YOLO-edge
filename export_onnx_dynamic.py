"""Export a trained YOLOv10 model to a dynamic-shape ONNX model."""

import argparse
import shutil
from pathlib import Path

import onnx
from ultralytics import YOLOv10


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights",
        default="runs/best.pt",
        help="Path to the trained .pt model.",
    )
    parser.add_argument("--output", default=None, help="Output ONNX path.")
    return parser.parse_args()


def main():
    args = parse_args()
    weights = Path(args.weights).resolve()
    output = Path(args.output).resolve() if args.output else weights.with_name(f"{weights.stem}_dynamic.onnx")

    model = YOLOv10(str(weights))
    exported = Path(
        model.export(
            format="onnx",
            imgsz=(640, 640),
            batch=1,
            device="cpu",
            dynamic=True,
            simplify=True,
            opset=13,
        )
    )

    if exported.resolve() != output:
        shutil.copy2(exported, output)

    onnx_model = onnx.load(str(output))
    onnx.checker.check_model(onnx_model)
    dims = onnx_model.graph.input[0].type.tensor_type.shape.dim
    shape = [dim.dim_param or dim.dim_value for dim in dims]
    expected = ["batch", 3, "height", "width"]
    if shape != expected:
        raise RuntimeError(f"Unexpected dynamic ONNX input shape: {shape}")

    print(f"Exported: {output}")
    print(f"Input shape: {shape}")
    print("Runtime height and width should normally be multiples of the model stride (32).")


if __name__ == "__main__":
    main()
