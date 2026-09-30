from pathlib import Path


MODULES_DIR = Path(__file__).parents[1] / "ultralytics" / "nn" / "modules"


def test_attention_modules_use_relu_instead_of_sigmoid():
    conv_source = (MODULES_DIR / "conv.py").read_text(encoding="utf-8")
    block_source = (MODULES_DIR / "block.py").read_text(encoding="utf-8")

    assert "self.act = nn.Sigmoid()" not in conv_source
    assert "aw = aw.sigmoid()" not in block_source
    assert "self.act = nn.ReLU()" in conv_source
    assert "self.act(aw)" in block_source


def test_convolutional_model_activations_use_relu_instead_of_silu():
    conv_source = (MODULES_DIR / "conv.py").read_text(encoding="utf-8")
    block_source = (MODULES_DIR / "block.py").read_text(encoding="utf-8")

    assert "default_act = nn.SiLU()" not in conv_source
    assert "self.act = nn.SiLU()" not in block_source
    assert conv_source.count("default_act = nn.ReLU()") == 3


def test_psa_uses_nadgr():
    block_source = (MODULES_DIR / "block.py").read_text(encoding="utf-8")
    nadgr_source = block_source.split("class NADGR", 1)[1].split("class PSA", 1)[0]

    assert "RepVGGDW(dim)" in nadgr_source
    assert "nn.AdaptiveAvgPool2d(1)" in nadgr_source
    assert nadgr_source.count("nn.ReLU(inplace=True)") == 2
    assert "nn.Sigmoid()" not in nadgr_source
    assert "x * gate" in nadgr_source
    assert " @ " not in nadgr_source
    assert "matmul" not in nadgr_source.lower()
    assert "self.nadgr = NADGR(self.c)" in block_source


def test_psa_npu_attention_preserves_shape_and_gradients():
    import torch

    from ultralytics.nn.modules.block import NADGR, PSA

    model = PSA(512, 512).train()
    assert isinstance(model.nadgr, NADGR)

    x = torch.randn(1, 512, 20, 20, requires_grad=True)
    output = model(x)
    output.mean().backward()

    assert output.shape == x.shape
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
