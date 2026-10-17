from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from ml.training.attention_lora import LowRankWeight, attention_lora_deltas, enable_attention_lora, merge_attention_lora


class FakeAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_proj_weight = nn.Parameter(torch.randn(12, 4))
        self.out_proj = nn.Linear(4, 4, bias=False)

    def forward(self, x):
        return self.out_proj(nn.functional.linear(x, self.in_proj_weight)[:, :4])


def fake_model():
    model = nn.Module()
    layer = nn.Module()
    layer.self_attn = FakeAttention()
    model.transformer = nn.Module()
    model.transformer.layers = nn.ModuleList([layer])
    model.linear = nn.Linear(4, 3)
    return model


def test_adapter_handles_direct_weight_access_and_initially_preserves_output():
    model = fake_model()
    x = torch.randn(3, 4)
    before = model.transformer.layers[0].self_attn(x).detach()
    parameters, handles, metadata = enable_attention_lora(model, last_layers=1, rank=2, alpha=2)
    assert torch.equal(before, model.transformer.layers[0].self_attn(x))
    assert len(parameters) == 4
    assert metadata["outputHeadFrozen"] is True
    assert not model.linear.weight.requires_grad
    assert all(row["changedEffectiveParameters"] == 0 for row in attention_lora_deltas(handles))


def test_only_adapters_update_and_merge_restores_plain_checkpoint_and_same_output():
    model = fake_model()
    original = {name: value.detach().clone() for name, value in model.state_dict().items()}
    parameters, handles, _ = enable_attention_lora(model, last_layers=1, rank=2, alpha=2)
    optimizer = torch.optim.AdamW(parameters, lr=.01, weight_decay=0)
    x = torch.randn(3, 4)
    for _ in range(2):
        optimizer.zero_grad()
        model.transformer.layers[0].self_attn(x).square().sum().backward()
        optimizer.step()
    assert any(row["changedEffectiveParameters"] > 0 for row in attention_lora_deltas(handles))
    expected = model.transformer.layers[0].self_attn(x).detach()
    report = merge_attention_lora(model, handles)
    assert report["effectiveMatricesBitwisePreserved"] is True
    assert set(model.state_dict()) == set(original)
    assert torch.equal(expected, model.transformer.layers[0].self_attn(x))
    assert torch.equal(original["linear.weight"], model.linear.weight)


def test_effective_weight_precision_does_not_depend_on_autocast():
    weight = torch.randn(4, 4)
    adapter = LowRankWeight(weight, 2, 2)
    with torch.no_grad():
        adapter.B.fill_(.001)
    expected = adapter(weight)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        actual = adapter(weight)
    assert actual.dtype == torch.float32
    assert torch.equal(expected, actual)


def test_invalid_rank_layer_count_and_base_dtype_are_rejected():
    with pytest.raises(ValueError, match="rank or alpha"):
        LowRankWeight(torch.randn(4, 4), 0, 8)
    with pytest.raises(ValueError, match="one to four"):
        enable_attention_lora(fake_model(), last_layers=2, rank=2, alpha=2)
    with pytest.raises(ValueError, match="float32"):
        LowRankWeight(torch.randn(4, 4).half(), 2, 2)
