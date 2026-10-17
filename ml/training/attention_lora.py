"""Small, zero-initialized attention adapters for isolated piano experiments.

PyTorch parametrizations cover both direct F.linear(..., in_proj_weight) and
Linear modules. Base matrices stay frozen; accepted research adapters can be
merged into ordinary checkpoint keys without changing the production loader.
"""

from __future__ import annotations

import hashlib
import math

import torch
from torch import nn
from torch.nn.utils import parametrize


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()


class LowRankWeight(nn.Module):
    def __init__(self, weight, rank: int, alpha: float):
        super().__init__()
        if weight.ndim != 2 or weight.dtype != torch.float32:
            raise ValueError("Attention adapters require a two-dimensional float32 base weight")
        if not 1 <= rank <= 64 or not math.isfinite(alpha) or not 0 < alpha <= 128:
            raise ValueError("Invalid bounded LoRA rank or alpha")
        self.scale = float(alpha) / rank
        self.A = nn.Parameter(torch.empty(rank, weight.shape[1], device=weight.device, dtype=weight.dtype))
        self.B = nn.Parameter(torch.zeros(weight.shape[0], rank, device=weight.device, dtype=weight.dtype))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, original):
        # Compute the effective weight consistently in FP32, independent of
        # BF16 training or FP16 inference autocast. Merge on the same device.
        with torch.autocast(device_type=original.device.type, enabled=False):
            return original + self.scale * (self.B @ self.A)


def enable_attention_lora(model, *, last_layers: int, rank: int, alpha: float):
    layers = model.transformer.layers
    if not 1 <= last_layers <= min(4, len(layers)):
        raise ValueError("The bounded attention-adapter experiment supports one to four final layers")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    handles, parameters, targets = [], [], []
    for index in range(len(layers) - last_layers, len(layers)):
        attention = getattr(layers[index], "self_attn", None)
        if attention is None or not hasattr(attention, "in_proj_weight") or not hasattr(attention, "out_proj"):
            raise ValueError("Unsupported streaming-attention layout")
        for module, attribute in ((attention, "in_proj_weight"), (attention.out_proj, "weight")):
            if parametrize.is_parametrized(module, attribute):
                raise ValueError("Refusing to stack an attention adapter onto an existing parametrization")
            module_name = next(name for name, value in model.named_modules() if value is module)
            original = getattr(module, attribute)
            before = tensor_hash(original)
            adapter = LowRankWeight(original, rank, alpha)
            parametrize.register_parametrization(module, attribute, adapter)
            if tensor_hash(getattr(module, attribute)) != before:
                raise ValueError("Zero-initialized attention adapter changed a base weight")
            handles.append((module, attribute, adapter, before))
            parameters.extend(adapter.parameters())
            targets.append({"name": f"{module_name}.{attribute}", "shape": list(original.shape), "originalSha256": before})
    metadata = {"schema": "polymath-attention-lora-v1", "lastLayers": last_layers, "rank": rank, "alpha": float(alpha), "scale": float(alpha) / rank, "trainableParameters": sum(p.numel() for p in parameters), "baseParametersFrozen": True, "outputHeadFrozen": True, "conditioningFrozen": True, "effectiveWeightComputation": "FP32 on the model device", "targets": targets}
    return parameters, handles, metadata


def attention_lora_deltas(handles):
    reports = []
    for module, attribute, adapter, expected_hash in handles:
        original = module.parametrizations[attribute].original
        if tensor_hash(original) != expected_hash:
            raise ValueError("A frozen attention base matrix changed")
        effective = getattr(module, attribute).detach()
        delta = effective - original.detach()
        if not torch.isfinite(delta).all():
            raise ValueError("Non-finite attention-adapter update")
        rms = float(delta.square().mean().sqrt())
        base_rms = float(original.detach().square().mean().sqrt())
        reports.append({"attribute": attribute, "baseMatrixBitwiseUnchanged": True, "changedEffectiveParameters": int(torch.count_nonzero(delta)), "rmsDelta": rms, "relativeRmsDelta": rms / max(1e-12, base_rms), "maximumAbsoluteDelta": float(delta.abs().max()), "effectiveWeightSha256": tensor_hash(effective)})
    return reports


def merge_attention_lora(model, handles):
    reports = attention_lora_deltas(handles)
    with torch.no_grad():
        for module, attribute, _adapter, _hash in handles:
            expected = tensor_hash(getattr(module, attribute))
            parametrize.remove_parametrizations(module, attribute, leave_parametrized=True)
            if tensor_hash(getattr(module, attribute)) != expected:
                raise ValueError("Attention-adapter merge changed the effective weight")
    if any(".parametrizations." in name for name in model.state_dict()):
        raise ValueError("An unmerged parametrization remains in the checkpoint")
    return {"ordinaryCheckpointKeysRestored": True, "mergedOnModelDeviceInFP32": True, "effectiveMatricesBitwisePreserved": True, "matrices": reports}
