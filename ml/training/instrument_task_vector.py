"""Auditable single-row task-vector tuning with all other cue rows frozen.

The active embedding row is observed at the released conditioner's embedding
lookup, never inferred from a guessed class-index offset. The optimizer must
give this matrix zero weight decay; the gradient hook zeros every inactive row.
"""

from __future__ import annotations

import hashlib
from typing import Any


class TaskVectorError(ValueError):
    pass


def enable_instrument_task_vector(transcription: Any, condition: str | None = None):
    import torch

    model = transcription._model
    try:
        conditioner = model.condition_provider.conditioners["instrument_group"]
        embedding = conditioner.embed
        parameter = embedding.weight
    except (AttributeError, KeyError) as exc:
        raise TaskVectorError("Unsupported instrument-conditioning implementation") from exc
    observed = []

    def observe(_module, arguments):
        observed.extend(int(value) for value in arguments[0].detach().cpu().reshape(-1).tolist())

    observer = embedding.register_forward_pre_hook(observe)
    try:
        with torch.no_grad():
            conditioner(conditioner.tokenize([condition]))
    finally:
        observer.remove()
    rows = sorted(set(observed))
    if len(rows) != 1 or not 0 <= rows[0] < parameter.shape[0]:
        raise TaskVectorError("Task-vector tuning requires exactly one observed embedding row")
    row = rows[0]
    name = next((name for name, value in model.named_parameters() if value is parameter), None)
    if name is None:
        raise TaskVectorError("Task-vector parameter is not present in the model state")
    parameter.requires_grad_(True)

    def only_active_row(gradient):
        if not torch.isfinite(gradient[row]).all():
            raise TaskVectorError("Non-finite piano task-vector gradient")
        masked = torch.zeros_like(gradient)
        masked[row] = gradient[row]
        return masked

    hook = parameter.register_hook(only_active_row)
    metadata = {
        "schema": "polymath-single-row-instrument-task-vector-v1",
        "parameter": name,
        "conditionValue": condition,
        "activeRow": row,
        "rowObservedAtActualEmbeddingLookup": True,
        "matrixRows": int(parameter.shape[0]),
        "activeParameters": int(parameter.shape[1]),
        "inactiveRowsGradientMasked": True,
        "requiredWeightDecay": 0.0,
    }
    return parameter, hook, metadata


def task_vector_delta(parameter, baseline, active_row: int) -> dict[str, Any]:
    import torch

    current = parameter.detach().float().cpu()
    before = baseline.detach().float().cpu()
    inactive_unchanged = torch.equal(current[:active_row].view(torch.int32), before[:active_row].view(torch.int32)) and torch.equal(current[active_row + 1:].view(torch.int32), before[active_row + 1:].view(torch.int32))
    if not inactive_unchanged:
        raise TaskVectorError("An inactive instrument-conditioning row changed")
    delta = current[active_row] - before[active_row]
    return {
        "activeRow": active_row,
        "inactiveRowsBitwiseUnchanged": True,
        "changedActiveParameters": int(torch.count_nonzero(delta)),
        "rmsDelta": float(delta.square().mean().sqrt()),
        "maximumAbsoluteDelta": float(delta.abs().max()),
        "normBefore": float(before[active_row].norm()),
        "normAfter": float(current[active_row].norm()),
        "rowBeforeSha256": hashlib.sha256(before[active_row].numpy().tobytes()).hexdigest(),
        "rowAfterSha256": hashlib.sha256(current[active_row].numpy().tobytes()).hexdigest(),
    }
