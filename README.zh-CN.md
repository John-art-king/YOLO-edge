# EdgeV10-NPU

**面向边缘 AI 与 NPU 部署的硬件感知 YOLOv10 加速工具包**

[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![ONNX](https://img.shields.io/badge/Export-ONNX-005CED.svg)](https://onnx.ai/)

EdgeV10-NPU 是一个面向边缘推理的 YOLOv10 部署项目，重点解决从 PyTorch 模型到真实 NPU 板端部署过程中常见的算子、激活函数、静态输入尺寸、ONNX 导出和验证问题。

> 本项目是基于 Ultralytics YOLO 代码库和 YOLOv10 研究实现的独立衍生项目，不是 Ultralytics 或 YOLOv10 官方发布版本。

## 项目特点

- 面向 NPU 的激活函数与注意力模块实现
- **NPU-Aware Dynamic Global Reweighting（NADGR）**，即 NPU 感知动态全局重标定模块
- 支持固定 `640x640`、batch-1 的静态 ONNX 导出
- 导出后自动进行 ONNX 图检查
- 提供 ReLU 方向的模型结构适配
- 提供模型结构回归测试
- 便于继续接入 INT8 量化和芯片厂商 SDK

## 部署流程

```text
YOLOv10 训练
    ↓
NPU 友好模型结构
    ↓
静态尺寸 ONNX 导出
    ↓
芯片厂商量化 / 编译 SDK
    ↓
边缘设备 NPU Runtime 推理
```

## 目录结构

```text
EdgeV10-NPU/
├── ultralytics/                 # YOLOv10 与 NPU 友好模型实现
├── ultralytics/cfg/models/v10/  # YOLOv10 模型配置
├── export_onnx_640.py           # 固定 640x640 ONNX 导出与校验
├── export_onnx_dynamic.py       # 动态尺寸 ONNX 导出
├── test_models.py               # PT/ONNX 评估工具
├── tests/                       # 模型结构与回归测试
├── examples/                    # 推理示例
├── docs/                        # 部署说明
├── requirements.txt
└── LICENSE
```

## 快速开始

### 1. 创建环境

```bash
python -m venv .venv
.venv\Scripts\activate       # Windows PowerShell
# source .venv/bin/activate    # Linux/macOS

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

如果使用具体芯片的 NPU SDK，请按照芯片厂商文档额外安装编译器、运行时和驱动。

### 2. 检查源码导入

```bash
python -c "from ultralytics import YOLOv10; print('EdgeV10-NPU import OK')"
```

### 3. 导出固定尺寸 ONNX

```bash
python export_onnx_640.py \
  --weights path/to/best.pt \
  --output artifacts/yolov10_640x640.onnx
```

导出脚本会检查 ONNX 图，并验证输入尺寸为 `[1, 3, 640, 640]`。

### 4. 运行模型结构测试

```bash
pytest tests/test_relu_activations.py -q
```

## NPU 部署说明

ONNX 是模型交换格式，最终编译命令取决于目标 NPU、SDK 和 Runtime。推荐流程如下：

1. 导出固定输入尺寸的 ONNX 模型。
2. 使用 ONNX checker 和 ONNX Runtime 验证模型。
3. 使用芯片厂商量化工具或编译器生成板端模型。
4. 在目标板卡上运行 Runtime。
5. 分别记录预处理、NPU 推理、后处理和端到端延迟。

详细清单和性能记录模板见 [docs/NPU_DEPLOYMENT.md](docs/NPU_DEPLOYMENT.md)。

## 性能记录模板

| 模型 | 激活函数 | 精度 | 输入尺寸 | 设备 / SDK | 推理延迟 | 端到端 FPS |
|---|---|---|---|---|---:|---:|
| YOLOv10-NPU | ReLU | INT8 | 640x640 | 待填写 | 待填写 | 待填写 |

发布正式结果时，请同时记录：

- NPU 芯片和开发板型号
- SDK、驱动、固件和编译器版本
- 精度、校准数据集和量化方式
- 输入尺寸和 batch size
- 预热次数与实际测试次数
- 是否包含预处理和后处理
- 验证数据集与精度指标

## 路线图

- [ ] 增加具体芯片的 NPU 转换示例
- [ ] 增加 INT8 校准与精度验证脚本
- [ ] 发布可复现的板端延迟表
- [ ] 增加 C++ Runtime 示例
- [ ] 增加自动 ONNX 导出和 CI 冒烟测试

## 致谢

本项目基于开源的 [Ultralytics](https://github.com/ultralytics/ultralytics) 代码库和 [YOLOv10](https://github.com/THU-MIG/yolov10) 实现构建。重新发布模型或用于商业产品前，请仔细核对上游项目和第三方组件的许可证及署名要求。

## 许可证

源代码采用 [AGPL-3.0](LICENSE) 许可证，同时受上游项目和第三方组件的许可证及署名要求约束。
