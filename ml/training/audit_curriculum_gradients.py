"""Measure paired original/synthetic training-gradient agreement without updates.

This is a small, offline curriculum diagnostic, not a training run or an
accuracy estimate. No optimizer is constructed and no checkpoint is saved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .muscriptor_tokens import encode_instrument_clip


PREFIXES = {
    "original": "wane-original-p000-supervision-v002-",
    "p100": "wane-ideal-p100-curriculum-v001-",
    "p075": "wane-mix-p075-curriculum-v001-",
    "p050": "wane-mix-p050-domain-bridge-v001-",
    "p025": "wane-mix-p025-domain-bridge-v001-",
}


def target_digest(row):
    tokens = encode_instrument_clip(row["notes"], duration_seconds=float(row["durationSeconds"]), instrument=row["instrumentFocus"])
    return hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()


def paired_groups(records, count, *, minimum_notes=0, minimum_duration=0):
    if not 2 <= count <= 12:
        raise ValueError("Bounded gradient diagnostic requires 2 to 12 time positions")
    grouped = {}
    for row in records:
        variant = next((name for name, prefix in PREFIXES.items() if row["clipId"].startswith(prefix)), None)
        if variant is None:
            continue
        key = (round(float(row["sourceStart"]), 6), round(float(row["durationSeconds"]), 6))
        if variant in grouped.setdefault(key, {}):
            raise ValueError("Duplicate curriculum variant at one time position")
        grouped[key][variant] = row
    complete = []
    for key, variants in sorted(grouped.items()):
        if set(variants) != set(PREFIXES):
            raise ValueError("Paired curriculum is incomplete at a time position")
        hashes = {target_digest(row) for row in variants.values()}
        if len(hashes) != 1:
            raise ValueError("Paired audio variants do not encode exactly the same target tokens")
        if len(variants["original"]["notes"]) < minimum_notes or key[1] < minimum_duration:
            continue
        complete.append((key, variants))
    if len(complete) < count:
        raise ValueError("Insufficient complete paired time positions")
    indices = np.linspace(0, len(complete) - 1, count).round().astype(int)
    return [complete[index] for index in indices]


def cosine(left, right):
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Invalid gradient vectors")
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    if denominator <= 0:
        raise ValueError("Zero gradient cannot define directional agreement")
    return float(np.clip(np.dot(left, right) / denominator, -1, 1))


def gradient_summary(original, synthetic):
    original, synthetic = np.asarray(original, dtype=np.float64), np.asarray(synthetic, dtype=np.float64)
    agreement = cosine(original, synthetic)
    return {
        "cosineOriginalVersusSynthetic": agreement,
        "firstOrderDirectionalConflict": agreement < 0,
        "originalGradientNorm": float(np.linalg.norm(original)),
        "syntheticGradientNorm": float(np.linalg.norm(synthetic)),
        "syntheticToOriginalNormRatio": float(np.linalg.norm(synthetic) / np.linalg.norm(original)),
        "combinedProjectionOntoOriginal": float(np.dot(original + synthetic, original) / np.dot(original, original)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--pairs", type=int, default=6)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Append-only gradient audit output already exists")
    config = json.loads(args.experiment.read_text())
    plan = json.loads(args.plan.read_text())
    if plan["experiment"] != args.experiment.as_posix() or plan["pairedPositions"] != args.pairs:
        raise ValueError("Gradient diagnostic command does not match its frozen plan")
    data = Path("/runpod-volume/training") / config["dataset"]["id"]
    base = Path("/runpod-volume/models/muscriptor-tester/phase46-v017/model.safetensors")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    from .train_muscriptor_piano import audit_records, clip_loss, read_jsonl, sha256_file
    from .run_piano_task_vector_experiment import audit_separation, verify
    from .full_song_target_development import load_config
    from .evaluate_checkpoint import configure_piano_task_evaluation_runtime

    verify(base, config["baseCheckpoint"]["sha256"])
    train_path, validation_path = data / "prepared-train.jsonl", data / "prepared-validation.jsonl"
    verify(train_path, config["dataset"]["trainManifestSha256"])
    verify(validation_path, config["dataset"]["validationManifestSha256"])
    records, validation = read_jsonl(train_path), read_jsonl(validation_path)
    input_audit = audit_records(records)
    target_config = load_config(Path(config["fullSongTargetConfig"]))
    separation = audit_separation(records, validation, target_config["windows"])
    groups = paired_groups(records, args.pairs, minimum_notes=5, minimum_duration=4.0)
    if [key[0] for key, _variants in groups] != plan["sourceStartsSeconds"]:
        raise ValueError("Paired source positions differ from the frozen plan")
    import fcntl
    with (data.parent / "polymath-research-gpu.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        configure_piano_task_evaluation_runtime()
        import torch
        from muscriptor import TranscriptionModel
        from .attention_lora import attention_lora_deltas, enable_attention_lora

        model = TranscriptionModel.load_model(base, device="cuda")
        hyper = config["frozenHyperparameters"]
        parameters, handles, adapter = enable_attention_lora(model._model, last_layers=hyper["attentionLoRALayers"], rank=hyper["attentionLoRARank"], alpha=hyper["attentionLoRAAlpha"])
        # Eval mode removes dropout noise. Gradients are still enabled; this
        # is a local diagnostic at the unchanged base, not minibatch replay.
        model._model.eval()
        torch.cuda.reset_peak_memory_stats()
        allowed = {id(parameter) for parameter in parameters}
        pair_reports, aggregate_original, aggregate_synthetic = [], None, None
        for (start, duration), variants in groups:
            vectors, variant_reports = {}, {}
            for name, row in variants.items():
                model._model.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = clip_loss(model, row, "cuda", apply_example_weight=False, conditioning_mode="unconditioned", timing_weight=hyper["timingTokenWeight"], note_on_weight=hyper["noteOnTokenWeight"], note_off_weight=hyper["noteOffTokenWeight"], eos_weight=hyper["eosTokenWeight"])
                loss.backward()
                if any(parameter.grad is not None for parameter in model._model.parameters() if id(parameter) not in allowed):
                    raise ValueError("A frozen foundation parameter received a gradient")
                if any(parameter.grad is None or not torch.isfinite(parameter.grad).all() for parameter in parameters):
                    raise ValueError("Invalid adapter gradient")
                vector = torch.cat([parameter.grad.detach().float().flatten() for parameter in parameters]).cpu().numpy().astype(np.float64)
                if np.linalg.norm(vector) <= 0:
                    raise ValueError("No adapter gradient")
                vectors[name] = vector
                variant_reports[name] = {"clipId": row["clipId"], "audioSha256": sha256_file(Path(row["audioClip"])), "unweightedLoss": float(loss.detach()), "exampleWeight": float(row.get("exampleWeight", 1.0)), "unweightedGradientNorm": float(np.linalg.norm(vector)), "cosineVersusOriginal": None}
                del loss
            for name, vector in vectors.items():
                variant_reports[name]["cosineVersusOriginal"] = cosine(vectors["original"], vector)
            original = vectors["original"] * variant_reports["original"]["exampleWeight"]
            synthetic = sum(vector * variant_reports[name]["exampleWeight"] for name, vector in vectors.items() if name != "original")
            aggregate_original = original.copy() if aggregate_original is None else aggregate_original + original
            aggregate_synthetic = synthetic.copy() if aggregate_synthetic is None else aggregate_synthetic + synthetic
            report = {"sourceStart": start, "durationSeconds": duration, "targetTokensSha256": target_digest(variants["original"]), "variants": variant_reports, **gradient_summary(original, synthetic)}
            pair_reports.append(report)
            print(json.dumps({"completedPairs": len(pair_reports), "totalPairs": len(groups), "sourceStart": start, "cosine": report["cosineOriginalVersusSynthetic"]}), flush=True)
        model._model.zero_grad(set_to_none=True)
        deltas = attention_lora_deltas(handles)
        if any(row["changedEffectiveParameters"] != 0 for row in deltas):
            raise ValueError("A no-optimizer gradient diagnostic changed the effective model")
        verify(base, config["baseCheckpoint"]["sha256"])
        result = {
            "schema": "polymath-curriculum-gradient-audit-v1", "optimizerCreated": False, "weightsUpdated": False, "productionChanged": False,
            "implementationSha256": sha256_file(Path(__file__)), "planSha256": sha256_file(args.plan), "experimentSha256": sha256_file(args.experiment), "baseCheckpointSha256": config["baseCheckpoint"]["sha256"], "trainManifestSha256": sha256_file(train_path),
            "inputAudit": input_audit, "separationAudit": separation, "adapter": adapter,
            "sampling": "Six (or explicit bounded count) evenly spaced complete paired training positions with at least five notes and four seconds; identical target tokens required; all source/validation time exclusions preserved",
            "gradientMode": "Eval-mode, BF16, teacher forcing, unconditioned audio, zero-initialized adapters; actual example weights applied only when combining gradients",
            "pairs": pair_reports, "aggregate": gradient_summary(aggregate_original, aggregate_synthetic), "zeroUpdateMatrixDeltas": deltas,
            "peakGpuFraction": torch.cuda.max_memory_allocated() / torch.cuda.get_device_properties(0).total_memory,
            "interpretation": "First-order SGD direction evidence at one unchanged base and adapter initialization, not an Adam update simulation, causal proof of curriculum failure, model accuracy or permission to change production. A negative cosine can motivate a separately frozen ablation; a positive cosine does not prove decoded quality improves.",
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")
        print(json.dumps({"outSha256": sha256_file(args.out), "aggregate": result["aggregate"], "weightsUpdated": False}), flush=True)


if __name__ == "__main__":
    main()
