"""Verify zero-update decoder fidelity, adapter gradients and memory headroom."""

import argparse
import json
import os
from pathlib import Path

from ml.training.train_muscriptor_piano import audit_records, clip_loss, read_jsonl, sha256_file
from ml.training.muscriptor_tokens import encode_instrument_clip


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Attention-adapter preflight folder already exists")
    base = Path("/runpod-volume/models/muscriptor-tester/phase46-v017/model.safetensors")
    base_hash = "a078a3a3dfdf35b2641de676ab9026b3475af6c009e4042a0d704b2cdd38afe0"
    data = Path("/runpod-volume/training/phase-105-piano-task-vector-v001")
    train_path, val_path = data / "prepared-train.jsonl", data / "prepared-validation.jsonl"
    cached = Path("/runpod-volume/training/phase-103-decoder-baseline-audit-v001")
    audio = Path("/runpod-volume/jobs/phase45-foundation-eval-20260916/wane-gbt-original.wav")
    hashes = {base: base_hash, train_path: "53e68be733d4061236be478d38bf6fe667ebe506d96ef08c92203280a2631d12", val_path: "c48aa0b292f4baaa8b63412e2bacd0c56572ed659010289f790a4eff24ccb117", cached / "fresh-baseline-full-raw.json": "1aa8a996d36a36f99aad1b7c83fee9fe7890375520b483028fd0e7a9d14b294e", cached / "fresh-baseline-panel.json": "a6f0ced022400aff6eb3b458c16381709dc5ca6b2bacdc343614cf8dd6088f6a", audio: "c1cb84e5b11ae5314c297e96f82469e88922add794d3f9c111122ecb8278b142"}
    for path, expected in hashes.items():
        if sha256_file(path) != expected:
            raise ValueError(f"Attention preflight input changed: {path}")
    train, validation = read_jsonl(train_path), read_jsonl(val_path)
    audit = audit_records(train)
    chosen = max(train, key=lambda row: len(encode_instrument_clip(row["notes"], duration_seconds=float(row["durationSeconds"]), instrument=row["instrumentFocus"])))
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    from ml.training.evaluate_checkpoint import configure_piano_task_evaluation_runtime, evaluate_loaded_transcription
    configure_piano_task_evaluation_runtime()
    import torch
    from muscriptor import TranscriptionModel
    from ml.training.attention_lora import attention_lora_deltas, enable_attention_lora
    from ml.training.audit_decoder_baseline import compare_events
    from ml.training.decoded_checkpoint_gate import select_decoded_gate_records
    from ml.training.transcribe_batch import transcribe_one
    model = TranscriptionModel.load_model(base, device="cuda")
    parameters, handles, metadata = enable_attention_lora(model._model, last_layers=1, rank=8, alpha=8)
    model._model.train()
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss = clip_loss(model, chosen, "cuda", conditioning_mode="unconditioned")
    loss.backward()
    gradients = []
    for _module, attribute, adapter, _hash in handles:
        if adapter.B.grad is None or not torch.isfinite(adapter.B.grad).all() or float(adapter.B.grad.norm()) <= 0:
            raise ValueError("No finite nonzero gradient reached an attention adapter")
        if adapter.A.grad is None or torch.count_nonzero(adapter.A.grad):
            raise ValueError("Zero-initialized B should initially give A exactly zero gradient")
        gradients.append({"attribute": attribute, "BGradientNorm": float(adapter.B.grad.norm()), "BNonzeroGradientParameters": int(torch.count_nonzero(adapter.B.grad)), "AInitialGradientExactlyZero": True})
    allowed = {id(value) for value in parameters}
    if not all(value.grad is None for value in model._model.parameters() if id(value) not in allowed):
        raise ValueError("A frozen foundation parameter received a gradient")
    peak = torch.cuda.max_memory_allocated()
    total = torch.cuda.get_device_properties(0).total_memory
    del loss
    model._model.zero_grad(set_to_none=True)
    model._model.eval()
    args.out.mkdir(parents=True)
    print(json.dumps({"stage": "zero-update-full-song-reproduction", "peakGradientMemoryFraction": peak / total}), flush=True)
    raw = transcribe_one(model, audio, title="Zero-update attention-adapter fidelity preflight", instruments=None, model_id="phase107-attention-preflight-v001", provider="Polymath no-optimizer research audit")
    (args.out / "zero-update-full-raw.json").write_text(json.dumps(raw, indent=2) + "\n")
    full_comparison = compare_events(json.loads((cached / "fresh-baseline-full-raw.json").read_text())["notes"], raw["notes"])
    records = select_decoded_gate_records(validation, maximum_per_song=6)
    panel = evaluate_loaded_transcription(model, records, instruments=None, include_raw_predictions=True)
    (args.out / "zero-update-panel.json").write_text(json.dumps(panel, indent=2) + "\n")
    expected_panel = json.loads((cached / "fresh-baseline-panel.json").read_text())
    expected = {row["clipId"]: row["notes"] for row in expected_panel["decodedClips"]}
    panel_comparisons = {row["clipId"]: compare_events(expected[row["clipId"]], row["notes"]) for row in panel["decodedClips"]}
    deltas = attention_lora_deltas(handles)
    unchanged = all(row["changedEffectiveParameters"] == 0 for row in deltas)
    passed = full_comparison["identical"] and all(row["identical"] for row in panel_comparisons.values()) and unchanged and peak / total < .90
    report = {"schema": "polymath-attention-lora-preflight-v1", "optimizerCreated": False, "weightsUpdated": False, "baseCheckpointSha256": base_hash, "trainManifestSha256": sha256_file(train_path), "trainAudit": audit, "largestTokenClip": chosen["clipId"], "adapter": metadata, "gradientEvidence": gradients, "otherFoundationGradientsAbsent": True, "zeroUpdateMatrixDeltas": deltas, "fullSongReproduction": full_comparison, "panelReproduction": panel_comparisons, "peakGradientGpuBytes": peak, "peakGradientGpuFraction": peak / total, "gpu": torch.cuda.get_device_name(0), "verdict": "PASS" if passed else "REJECT_PREFLIGHT"}
    (args.out / "preflight.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
