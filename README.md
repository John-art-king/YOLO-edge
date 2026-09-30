# EdgeV10-NPU

**Hardware-Aware YOLOv10 Acceleration Toolkit for Edge AI and NPU Deployment**

[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![ONNX](https://img.shields.io/badge/Export-ONNX-005CED.svg)](https://onnx.ai/)

EdgeV10-NPU is a deployment-oriented YOLOv10 research project for edge inference. It adapts the YOLOv10 architecture and export workflow for NPU-friendly execution, with a focus on static-shape ONNX export, activation compatibility, reproducible validation, and practical board-side deployment.

> This repository is an independent derivative work based on the Ultralytics YOLO codebase and the YOLOv10 research implementation. It is not an official Ultralytics or YOLOv10 release.

## Why EdgeV10-NPU?

Many edge NPUs do not support every operation or activation equally well. EdgeV10-NPU focuses on the parts that usually determine whether a detector can move from PyTorch to a real device:

- NPU-friendly activation and attention implementations
- **NPU-Aware Dynamic Global Reweighting (NADGR)** for lightweight feature recalibration
- Fixed `640x640`, batch-1 ONNX export for predictable deployment
- Export validation with ONNX graph checks
- ReLU-oriented model variants for accelerator compatibility
- Reproducible model-structure tests
- A clean starting point for INT8 quantization and vendor SDK integration

## Deployment Pipeline

```text
YOLOv10 training
      |
      v
NPU-friendly model definition
      |
      v
Static-shape ONNX export
      |
      v
Vendor quantization / compiler SDK
      |
      v
NPU runtime on edge device
```

## Repository Layout

```text
EdgeV10-NPU/
├── ultralytics/                 # YOLOv10 and NPU-friendly model implementation
├── ultralytics/cfg/models/v10/  # YOLOv10 model configurations
├── export_onnx_640.py           # Static 640x640 ONNX export and validation
├── export_onnx_dynamic.py       # Dynamic-shape ONNX export
├── test_models.py               # PT/ONNX evaluation utility
├── tests/                       # Model-structure and regression tests
├── examples/                    # Inference examples inherited from the base project
├── docs/                        # Deployment notes
├── requirements.txt
└── LICENSE
```

## Quick Start

### 1. Create an environment

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate       # Windows PowerShell

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

The pinned PyTorch and ONNX Runtime versions in `requirements.txt` are a reference environment. For a vendor NPU SDK, install the SDK runtime and compiler according to that vendor's documentation.

### 2. Run a PyTorch smoke test

```bash
python -c "from ultralytics import YOLOv10; print('EdgeV10-NPU import OK')"
```

### 3. Export a static ONNX model

```bash
python export_onnx_640.py \
  --weights path/to/best.pt \
  --output artifacts/yolov10_640x640.onnx
```

The exporter checks the ONNX graph and verifies the input shape is `[1, 3, 640, 640]`.

### 4. Run the model-structure tests

```bash
pytest tests/test_relu_activations.py -q
```

## NPU Deployment

The ONNX model is the interchange artifact. The final compilation command depends on the target NPU, SDK, and runtime. A typical deployment flow is:

1. Export a fixed-shape ONNX model.
2. Validate the graph with ONNX checker and ONNX Runtime.
3. Run the vendor quantizer or compiler.
4. Deploy the compiled model to the target board.
5. Benchmark preprocessing, inference, postprocessing, and end-to-end latency separately.

See [docs/NPU_DEPLOYMENT.md](docs/NPU_DEPLOYMENT.md) for the deployment checklist and benchmark reporting template.

## Benchmark Reporting

Please report measurements with enough context to reproduce them. Do not compare FPS values without recording the device and runtime configuration.

| Model | Activation | Precision | Input | Device / SDK | Inference latency | End-to-end FPS |
|---|---|---|---|---|---:|---:|
| YOLOv10-NPU | ReLU | INT8 | 640x640 | To be filled | To be filled | To be filled |

Recommended benchmark details:

- NPU chip and board name
- SDK, driver, firmware, and compiler versions
- Precision and calibration dataset
- Input shape and batch size
- Warm-up iterations and measured iterations
- Whether preprocessing and postprocessing are included
- Validation dataset and accuracy metric

## Model Variants

The model configuration files are under `ultralytics/cfg/models/v10/`. Custom training scripts and checkpoints should remain outside the source repository or be distributed through GitHub Releases, Hugging Face, or ModelScope.

## Roadmap

- [ ] Add a vendor-specific NPU conversion example
- [ ] Add INT8 calibration and accuracy validation scripts
- [ ] Publish reproducible board-side latency tables
- [ ] Add C++ runtime examples
- [ ] Add automated ONNX export and smoke tests in CI

## Acknowledgements

This project builds on the open-source [Ultralytics](https://github.com/ultralytics/ultralytics) codebase and the original [YOLOv10](https://github.com/THU-MIG/yolov10) implementation. Please review the upstream licenses and attribution requirements before redistributing models or commercial products.

## License

The source code is released under the [AGPL-3.0 license](LICENSE), subject to the licenses and attribution notices of the upstream projects and third-party components.
