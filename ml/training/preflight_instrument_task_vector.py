"""Check real task-vector gradients and peak GPU memory without any updates."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random

from ml.training.train_muscriptor_piano import (
    audit_records,
    clip_loss,
    configure_trainable_parameters,
    deterministic_seed,
    read_jsonl,
    sha256_file,
)
from ml.training.muscriptor_tokens import encode_instrument_clip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--base-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Preflight output already exists: {args.out}")
    if sha256_file(args.base) != args.base_sha256:
        raise ValueError("Base checkpoint changed before gradient preflight")
    records = read_jsonl(args.train_manifest)
    audit = audit_records(records)
    chosen = max(records, key=lambda record: len(encode_instrument_clip(
        record["notes"], duration_seconds=float(record["durationSeconds"]),
        instrument=record["instrumentFocus"],
    )))
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import numpy as np
    import torch
    from muscriptor import TranscriptionModel
    from ml.training.instrument_task_vector import enable_instrument_task_vector, task_vector_delta

    seed = deterministic_seed("polymath-phase100-v003-unconditioned")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    model = TranscriptionModel.load_model(args.base, device="cuda")
    head_parameters = configure_trainable_parameters(model._model, 0)
    parameter, hook, metadata = enable_instrument_task_vector(model, None)
    before = parameter.detach().cpu().clone()
    try:
        model._model.train()
        torch.cuda.reset_peak_memory_stats()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = clip_loss(model, chosen, "cuda", conditioning_mode="unconditioned")
        loss.backward()
        row = metadata["activeRow"]
        gradient = parameter.grad
        if gradient is None or not torch.isfinite(gradient[row]).all() or float(gradient[row].norm()) <= 0:
            raise ValueError("No finite nonzero gradient reached the task vector")
        inactive_gradient_zero = bool(torch.count_nonzero(gradient[:row]) == 0 and torch.count_nonzero(gradient[row + 1:]) == 0)
        if not inactive_gradient_zero:
            raise ValueError("A frozen conditioning row received a gradient")
        allowed_ids = {id(value) for value in [*head_parameters, parameter]}
        other_gradients_absent = all(value.grad is None for value in model._model.parameters() if id(value) not in allowed_ids)
        if not other_gradients_absent:
            raise ValueError("An unintended foundation parameter received a gradient")
        delta = task_vector_delta(parameter, before, row)
        if delta["changedActiveParameters"]:
            raise ValueError("Preflight changed a model weight despite creating no optimizer")
        peak = torch.cuda.max_memory_allocated()
        total = torch.cuda.get_device_properties(0).total_memory
        report = {
            "schema": "polymath-instrument-task-vector-gradient-preflight-v1",
            "optimizerCreated": False,
            "weightsUpdated": False,
            "baseCheckpointSha256": args.base_sha256,
            "trainManifestSha256": sha256_file(args.train_manifest),
            "trainAudit": audit,
            "largestTokenClipId": chosen["clipId"],
            "largestTokenCount": len(encode_instrument_clip(chosen["notes"], duration_seconds=float(chosen["durationSeconds"]), instrument=chosen["instrumentFocus"])),
            "loss": float(loss.detach().cpu()),
            "taskVector": metadata,
            "taskVectorGradientNorm": float(gradient[row].norm()),
            "taskVectorGradientNonzeroParameters": int(torch.count_nonzero(gradient[row])),
            "inactiveGradientRowsZero": inactive_gradient_zero,
            "otherFoundationGradientsAbsent": other_gradients_absent,
            "noUpdateDelta": delta,
            "gpu": torch.cuda.get_device_name(0),
            "peakAllocatedBytes": peak,
            "totalGpuBytes": total,
            "peakGpuFraction": peak / total,
            "verdict": "PASS" if peak / total < 0.90 else "MEMORY_HEADROOM_REQUIRES_REVIEW",
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
        if report["verdict"] != "PASS":
            raise SystemExit(1)
    finally:
        hook.remove()


if __name__ == "__main__":
    main()
