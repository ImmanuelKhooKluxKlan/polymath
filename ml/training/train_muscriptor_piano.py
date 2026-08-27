"""Conservative supervised fine-tuning for a MuScriptor-compatible piano checkpoint.

This adapter intentionally trains only the final transformer block and output
head first. It defaults to an audit-only dry run and refuses to overwrite a
checkpoint. Use it on a CUDA RunPod after the reviewed dataset quality gate has
passed; never point ``--out`` at ``models/original``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

from ml.training.muscriptor_tokens import encode_piano_clip, teacher_forcing_pair


RIGHTS_ACKNOWLEDGEMENT = "I_HAVE_TRAINING_RIGHTS"


class TrainingError(RuntimeError):
    """Raised before any optimizer update when a safety invariant fails."""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TrainingError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(record, dict):
                raise TrainingError(f"{path}:{line_number}: record must be an object")
            records.append(record)
    if not records:
        raise TrainingError(f"Prepared manifest has no clips: {path}")
    return records


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def audit_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    songs: set[str] = set()
    token_counts: list[int] = []
    audio_seconds = 0.0
    for index, record in enumerate(records):
        clip_id = str(record.get("clipId") or f"clip-{index}")
        audio = Path(str(record.get("audioClip") or "")).resolve()
        if not audio.is_file() or audio.stat().st_size <= 44:
            raise TrainingError(f"{clip_id}: prepared audio clip is missing")
        duration = float(record.get("durationSeconds") or 0)
        if not 0 < duration <= 5.001:
            raise TrainingError(f"{clip_id}: duration must be at most five seconds")
        notes = record.get("notes")
        if not isinstance(notes, list) or not notes:
            raise TrainingError(f"{clip_id}: reviewed piano labels are missing")
        tokens = encode_piano_clip(notes, duration_seconds=duration)
        if len(tokens) > 2000:
            raise TrainingError(f"{clip_id}: {len(tokens)} tokens exceed MuScriptor's 2000-token segment limit")
        token_counts.append(len(tokens))
        audio_seconds += duration
        songs.add(str(record.get("songId") or "unknown"))
    return {
        "clips": len(records),
        "songs": len(songs),
        "audioSeconds": round(audio_seconds, 3),
        "minimumTokens": min(token_counts),
        "maximumTokens": max(token_counts),
        "averageTokens": round(sum(token_counts) / len(token_counts), 2),
    }


def load_audio_clip(path: Path, device: str):
    import numpy as np
    import soundfile as sf
    import torch

    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if sample_rate != 16000:
        raise TrainingError(f"Prepared clip must be 16 kHz, got {sample_rate}: {path}")
    mono = np.asarray(audio, dtype=np.float32).mean(axis=1)
    expected = 5 * sample_rate
    if mono.shape[0] < expected:
        mono = np.pad(mono, (0, expected - mono.shape[0]))
    elif mono.shape[0] > expected:
        mono = mono[:expected]
    return torch.from_numpy(mono).unsqueeze(0).to(device)


def configure_trainable_parameters(model, train_last_layers: int) -> list[Any]:
    if train_last_layers < 1:
        raise TrainingError("At least one final transformer layer must be trainable")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    layers = model.transformer.layers
    if train_last_layers > len(layers):
        raise TrainingError(f"Checkpoint has only {len(layers)} transformer layers")
    for layer in layers[-train_last_layers:]:
        for parameter in layer.parameters():
            parameter.requires_grad_(True)
    for module in (model.out_norm, model.linear):
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def clip_loss(transcription, record: dict[str, Any], device: str):
    import torch
    import torch.nn.functional as functional

    duration = float(record["durationSeconds"])
    tokens = encode_piano_clip(record["notes"], duration_seconds=duration)
    inputs, targets = teacher_forcing_pair(tokens)
    input_tensor = torch.tensor([inputs], dtype=torch.long, device=device)
    target_tensor = torch.tensor([targets], dtype=torch.long, device=device)
    wav = load_audio_clip(Path(record["audioClip"]), device)
    conditions = transcription._build_conditions(wav, "0")
    provider = transcription._model.condition_provider
    prepared = provider.tokenize(conditions)
    condition_tensors = provider(prepared)
    logits = transcription._model(input_tensor, condition_tensors, first_step=True)
    return functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), target_tensor.reshape(-1))


def evaluate_loss(transcription, records: list[dict[str, Any]], device: str, precision: str) -> float:
    import torch

    transcription._model.eval()
    losses: list[float] = []
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    with torch.no_grad():
        for record in records:
            with torch.autocast(device_type="cuda", dtype=dtype, enabled=device.startswith("cuda")):
                losses.append(float(clip_loss(transcription, record, device).detach().cpu()))
    return sum(losses) / max(1, len(losses))


def save_checkpoint(transcription, base: Path, output: Path, metadata: dict[str, Any]) -> None:
    from safetensors.torch import save_file

    if output.exists() and any(output.iterdir()):
        raise TrainingError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / "model.safetensors.tmp"
    state = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in transcription._model.state_dict().items()
    }
    save_file(state, str(temporary), metadata={"format": "pt", "polymath": "piano-supervised"})
    os.replace(temporary, output / "model.safetensors")
    config = base.parent / "config.json"
    if not config.is_file():
        raise TrainingError(f"config.json is missing beside the base checkpoint: {config}")
    shutil.copy2(config, output / "config.json")
    (output / "training-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8",
    )


def train(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from muscriptor import TranscriptionModel

    if not torch.cuda.is_available():
        raise TrainingError("CUDA GPU is required for the 1.3B checkpoint")
    if args.rights_acknowledgement != RIGHTS_ACKNOWLEDGEMENT:
        raise TrainingError(f"Execution requires --rights-acknowledgement {RIGHTS_ACKNOWLEDGEMENT}")
    if not args.base.is_file():
        raise TrainingError(f"Base checkpoint does not exist: {args.base}")
    if not (args.base.parent / "config.json").is_file():
        raise TrainingError(f"config.json is missing beside the base checkpoint: {args.base.parent}")
    if args.out == args.base.parent or args.base.parent in args.out.parents:
        raise TrainingError("Output must be outside the immutable base-checkpoint directory")
    if args.out.exists() and (not args.out.is_dir() or any(args.out.iterdir())):
        raise TrainingError(f"Output path must be a new or empty directory: {args.out}")
    train_records = read_jsonl(args.train_manifest)
    validation_records = read_jsonl(args.validation_manifest)
    train_audit = audit_records(train_records)
    validation_audit = audit_records(validation_records)
    if train_audit["songs"] < args.minimum_train_songs:
        raise TrainingError(
            f"Only {train_audit['songs']} training songs; minimum is {args.minimum_train_songs}. "
            "Do not overfit the foundation checkpoint to a handful of songs."
        )

    device = "cuda"
    transcription = TranscriptionModel.load_model(args.base, device=device)
    parameters = configure_trainable_parameters(transcription._model, args.train_last_layers)
    trainable_names = {
        name for name, parameter in transcription._model.named_parameters()
        if parameter.requires_grad
    }
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
    baseline_validation_loss = evaluate_loss(transcription, validation_records, device, args.precision)
    best_validation_loss = baseline_validation_loss
    best_state = None
    step = 0
    accumulated = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        transcription._model.train()
        for record in train_records:
            with torch.autocast(device_type="cuda", dtype=dtype):
                loss = clip_loss(transcription, record, device) / args.gradient_accumulation
            loss.backward()
            step += 1
            accumulated += 1
            if accumulated == args.gradient_accumulation:
                torch.nn.utils.clip_grad_norm_(parameters, args.gradient_clip_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                accumulated = 0
        if accumulated:
            torch.nn.utils.clip_grad_norm_(parameters, args.gradient_clip_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            accumulated = 0
        validation_loss = evaluate_loss(transcription, validation_records, device, args.precision)
        print(json.dumps({"epoch": epoch + 1, "validationLoss": validation_loss}), flush=True)
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = {
                name: parameter.detach().cpu().clone()
                for name, parameter in transcription._model.state_dict().items()
                if name in trainable_names
            }
    if best_state is None:
        raise TrainingError("Validation loss never improved; no candidate checkpoint was written")
    transcription._model.load_state_dict(best_state, strict=False)
    metadata = {
        "schema": "polymath-muscriptor-training-run-v1",
        "baseCheckpoint": str(args.base),
        "baseSha256": sha256_file(args.base),
        "trainManifest": str(args.train_manifest),
        "validationManifest": str(args.validation_manifest),
        "trainAudit": train_audit,
        "validationAudit": validation_audit,
        "trainLastLayers": args.train_last_layers,
        "learningRate": args.learning_rate,
        "epochs": args.epochs,
        "baselineValidationLoss": baseline_validation_loss,
        "bestValidationLoss": best_validation_loss,
        "commercialUseAllowed": False,
        "note": "Candidate only. Promotion requires frozen note-F1 tests and manual listening.",
    }
    save_checkpoint(transcription, args.base, args.out, metadata)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="Allow optimizer updates; otherwise audit only")
    parser.add_argument("--rights-acknowledgement", default="")
    parser.add_argument("--minimum-train-songs", type=int, default=20)
    parser.add_argument("--train-last-layers", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--precision", choices=("bf16", "fp16"), default="bf16")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.train_manifest = args.train_manifest.resolve()
    args.validation_manifest = args.validation_manifest.resolve()
    args.base = args.base.resolve()
    args.out = args.out.resolve()
    train_records = read_jsonl(args.train_manifest)
    validation_records = read_jsonl(args.validation_manifest)
    audit = {
        "mode": "execute" if args.execute else "audit-only",
        "train": audit_records(train_records),
        "validation": audit_records(validation_records),
    }
    if not args.execute:
        print(json.dumps(audit, indent=2))
        return
    print(json.dumps(train(args), indent=2))


if __name__ == "__main__":
    try:
        main()
    except TrainingError as exc:
        raise SystemExit(str(exc)) from exc
