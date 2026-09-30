# Changelog

## Unreleased

- Reorganized the project as EdgeV10-NPU, a hardware-aware YOLOv10 deployment toolkit.
- Removed local datasets, checkpoints, training runs, caches, logs, and generated artifacts from the public package.
- Added an English and Chinese project README focused on NPU deployment.
- Generalized the static ONNX exporter to accept arbitrary checkpoint paths and class counts.
- Added an NPU deployment checklist and benchmark reporting template.
- Renamed the NPU-friendly feature module to `NADGR` (`NPU-Aware Dynamic Global Reweighting`).
