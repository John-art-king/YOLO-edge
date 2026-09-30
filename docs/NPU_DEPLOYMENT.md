# NPU Deployment Checklist

EdgeV10-NPU produces a static-shape ONNX interchange model. The final conversion command is specific to the target chip and vendor SDK, so record the toolchain details with every benchmark.

## Export

```bash
python export_onnx_640.py \
  --weights path/to/best.pt \
  --output artifacts/yolov10_640x640.onnx
```

The exporter validates the ONNX graph and requires the input shape `[1, 3, 640, 640]`.

## Conversion Checklist

- [ ] Record the NPU chip and board model.
- [ ] Record the compiler, SDK, driver, firmware, and runtime versions.
- [ ] Confirm all ONNX operators are supported by the compiler.
- [ ] Select FP16, INT8, or another target precision.
- [ ] Keep the calibration dataset representative of deployment data.
- [ ] Compare accuracy before and after quantization.
- [ ] Validate output decoding and class names on the target board.
- [ ] Separate preprocessing, NPU inference, and postprocessing timing.

## Benchmark Template

| Field | Value |
|---|---|
| Model | `yolov10-...` |
| Input shape | `1x3x640x640` |
| Precision | `FP16 / INT8` |
| NPU / board | `fill in` |
| Compiler / SDK | `fill in` |
| Driver / firmware | `fill in` |
| Calibration set | `fill in` |
| Warm-up iterations | `fill in` |
| Measured iterations | `fill in` |
| Preprocess latency | `fill in` |
| NPU inference latency | `fill in` |
| Postprocess latency | `fill in` |
| End-to-end latency | `fill in` |
| End-to-end FPS | `fill in` |
| Validation metric | `fill in` |

## Reporting Rules

Use the same input size, batch size, confidence threshold, and postprocessing implementation when comparing model variants. A latency number without the device, SDK version, precision, and timing scope is not reproducible.
