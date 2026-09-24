"""Conservative supervised fine-tuning for a MuScriptor-compatible piano checkpoint.

This adapter intentionally trains only a requested number of final transformer
blocks plus the output head (zero blocks means head-only). It defaults to an
audit-only dry run and refuses to overwrite a checkpoint. Use it on a CUDA
RunPod after the reviewed dataset quality gate has passed; never point
``--out`` at ``models/original``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
from pathlib import Path
from typing import Any

from ml.training.muscriptor_tokens import (
    EOS_ID,
    PITCH_BASE,
    SHIFT_BASE,
    TIE_ID,
    VELOCITY_BASE,
    canonical_instrument_name,
    encode_instrument_clip,
    instrument_group_id,
    teacher_forcing_pair,
)


RIGHTS_ACKNOWLEDGEMENT = "I_HAVE_TRAINING_RIGHTS"
CONDITIONING_MODES = {"instrument", "unconditioned"}


class TrainingError(RuntimeError):
    """Raised before any optimizer update when a safety invariant fails."""


def deterministic_seed(label: str) -> int:
    """Map the human-readable run seed to a stable 32-bit RNG seed."""

    digest = hashlib.sha256(str(label).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big", signed=False)


def conditioning_group(instrument: str, mode: str) -> str | None:
    """Return the exact MuScriptor text condition used for a training clip.

    The production piano-reduction route first listens to an unconstrained full
    mix.  Route-specific checkpoints therefore need an explicit unconditioned
    training option; training with the acoustic-piano condition and decoding
    without it is a distribution mismatch that can look good on clip loss while
    catastrophically changing the instruments emitted for real songs.
    """

    normalized = str(mode or "instrument").strip().lower()
    if normalized not in CONDITIONING_MODES:
        raise TrainingError(
            "conditioningMode must be instrument or unconditioned"
        )
    return None if normalized == "unconditioned" else str(instrument_group_id(instrument))


def target_token_weights(
    tokens: list[int],
    timing_weight: float = 1.15,
    note_off_weight: float = 1.25,
    eos_weight: float = 1.20,
    note_on_weight: float = 1.0,
) -> list[float]:
    """Rebalance onset, timing, and stopping errors without changing clip scale.

    MuScriptor represents both note-on and note-off keys with the same pitch
    token; the preceding velocity state says which one it is.  Weighting the
    off-state pitch as well as its velocity token specifically targets chopped
    or stuck durations.  The optional on-state weight applies to both its state
    token and subsequent pitches, allowing a recall-focused experiment without
    conflating onset and release events.  We normalize to mean 1 so this changes
    *which* errors matter inside a clip, not the clip's overall learning rate.
    """

    if min(timing_weight, note_off_weight, eos_weight, note_on_weight) <= 0:
        raise TrainingError("Token-loss weights must be positive")
    velocity_state: int | None = None
    weights: list[float] = []
    for token in tokens:
        weight = 1.0
        if token == EOS_ID:
            weight = eos_weight
        elif SHIFT_BASE <= token < PITCH_BASE:
            weight = timing_weight
        elif token == VELOCITY_BASE:
            velocity_state = 0
            weight = note_off_weight
        elif token == VELOCITY_BASE + 1:
            velocity_state = 1
            weight = note_on_weight
        elif PITCH_BASE <= token < VELOCITY_BASE and velocity_state == 0:
            weight = note_off_weight
        elif PITCH_BASE <= token < VELOCITY_BASE and velocity_state == 1:
            weight = note_on_weight
        elif token == TIE_ID:
            weight = timing_weight
        weights.append(weight)
    average = sum(weights) / max(1, len(weights))
    return [weight / average for weight in weights]


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
    instruments: set[str] = set()
    token_counts: list[int] = []
    audio_seconds = 0.0
    negative_examples = 0
    for index, record in enumerate(records):
        clip_id = str(record.get("clipId") or f"clip-{index}")
        audio = Path(str(record.get("audioClip") or "")).resolve()
        local_audio = Path(str(record.get("localAudioSource") or "")).resolve()
        if (not audio.is_file() or audio.stat().st_size <= 44) and local_audio.is_file():
            # Composed manifests deliberately point audioClip at the RunPod
            # mount. Keep a provenance-only local copy path so the exact same
            # manifest can still pass a desktop preflight before upload.
            audio = local_audio
        if not audio.is_file() or audio.stat().st_size <= 44:
            raise TrainingError(f"{clip_id}: prepared audio clip is missing")
        duration = float(record.get("durationSeconds") or 0)
        if not 0 < duration <= 5.001:
            raise TrainingError(f"{clip_id}: duration must be at most five seconds")
        notes = record.get("notes")
        if not isinstance(notes, list):
            raise TrainingError(f"{clip_id}: reviewed instrument labels are missing")
        instrument = canonical_instrument_name(
            record.get("instrumentFocus") or "acoustic_piano"
        )
        is_negative = bool(record.get("isNegativeExample"))
        if not notes and not (
            is_negative and record.get("targetState") == "reviewed-silence"
        ):
            raise TrainingError(
                f"{clip_id}: an empty target is allowed only for explicitly reviewed silence"
            )
        if notes and is_negative:
            raise TrainingError(f"{clip_id}: a negative example cannot contain notes")
        weight = float(record.get("exampleWeight", 1.0))
        if not 0 < weight <= 1:
            raise TrainingError(f"{clip_id}: exampleWeight must be within (0, 1]")
        tokens = encode_instrument_clip(
            notes,
            duration_seconds=duration,
            instrument=instrument,
        )
        if len(tokens) > 2000:
            raise TrainingError(f"{clip_id}: {len(tokens)} tokens exceed MuScriptor's 2000-token segment limit")
        token_counts.append(len(tokens))
        audio_seconds += duration
        songs.add(str(record.get("songId") or "unknown"))
        instruments.add(instrument)
        negative_examples += is_negative
    return {
        "clips": len(records),
        "songs": len(songs),
        "instruments": sorted(instruments),
        "positiveExamples": len(records) - negative_examples,
        "negativeExamples": negative_examples,
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
    if train_last_layers < 0:
        raise TrainingError("The number of trainable final transformer layers cannot be negative")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    layers = model.transformer.layers
    if train_last_layers > len(layers):
        raise TrainingError(f"Checkpoint has only {len(layers)} transformer layers")
    if train_last_layers:
        for layer in layers[-train_last_layers:]:
            for parameter in layer.parameters():
                parameter.requires_grad_(True)
    for module in (model.out_norm, model.linear):
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def should_run_periodic_validation(optimizer_step: int, interval_steps: int) -> bool:
    """Return whether this optimizer update is a requested safety checkpoint.

    ``interval_steps=0`` preserves the original epoch-only behaviour.  The
    interval counts optimizer updates rather than clips, so changing gradient
    accumulation does not silently multiply the number of model changes made
    between decoded-note checks.
    """

    if interval_steps < 0:
        raise TrainingError("Validation interval steps cannot be negative")
    return interval_steps > 0 and optimizer_step > 0 and optimizer_step % interval_steps == 0


def decoded_selection_rank(
    metrics: dict[str, Any],
    validation_loss: float,
) -> tuple[float, float, float, float, float]:
    """Rank safe checkpoints by decoded music quality, then token loss.

    Teacher-forced loss remains an eligibility gate, but it is only a proxy for
    the product output.  Once a checkpoint passes the decoded safety gate, the
    strict 100 ms note F1 and recall select the winner.  The 250 ms values and
    lower validation loss break exact ties deterministically.
    """

    try:
        at_100 = metrics["100ms"]
        at_250 = metrics["250ms"]
        values = (
            float(at_100["microF1"]),
            float(at_100["recall"]),
            float(at_250["microF1"]),
            float(at_250["recall"]),
            -float(validation_loss),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise TrainingError("Decoded checkpoint metrics are incomplete") from exc
    if not all(math.isfinite(value) for value in values):
        raise TrainingError("Decoded checkpoint metrics must be finite")
    return values


def summarize_weight_delta(
    model,
    baseline: dict[str, Any],
    trainable_names: set[str],
) -> dict[str, Any]:
    """Measure exactly how far every trainable tensor moved from the base.

    Only trainable tensors are cloned before optimization. This keeps the audit
    detailed without duplicating the full 5.47 GB checkpoint in CPU memory.
    """

    import torch

    tensors: list[dict[str, Any]] = []
    total_parameters = 0
    total_changed = 0
    total_squared_delta = 0.0
    maximum_absolute_delta = 0.0
    current_state = model.state_dict()
    for name in sorted(trainable_names):
        before = baseline[name].float()
        after = current_state[name].detach().float().cpu()
        delta = after - before
        absolute = delta.abs()
        parameters = delta.numel()
        changed = int(torch.count_nonzero(delta).item())
        squared_sum = float(torch.sum(delta * delta).item())
        base_squared_sum = float(torch.sum(before * before).item())
        rms = (squared_sum / max(1, parameters)) ** 0.5
        base_rms = (base_squared_sum / max(1, parameters)) ** 0.5
        max_abs = float(absolute.max().item()) if parameters else 0.0
        tensors.append({
            "name": name,
            "parameters": parameters,
            "changedParameters": changed,
            "changedPercent": round(changed / max(1, parameters) * 100, 6),
            "meanAbsoluteDelta": float(absolute.mean().item()) if parameters else 0.0,
            "rmsDelta": rms,
            "relativeRmsDeltaPercent": rms / max(base_rms, 1e-12) * 100,
            "maximumAbsoluteDelta": max_abs,
        })
        total_parameters += parameters
        total_changed += changed
        total_squared_delta += squared_sum
        maximum_absolute_delta = max(maximum_absolute_delta, max_abs)
    tensors.sort(key=lambda item: item["rmsDelta"], reverse=True)
    return {
        "schema": "polymath-weight-delta-v1",
        "trainableTensorCount": len(tensors),
        "trainableParameters": total_parameters,
        "changedParameters": total_changed,
        "changedPercent": round(total_changed / max(1, total_parameters) * 100, 6),
        "overallRmsDelta": (total_squared_delta / max(1, total_parameters)) ** 0.5,
        "maximumAbsoluteDelta": maximum_absolute_delta,
        "tensors": tensors,
    }


def clip_loss(
    transcription,
    record: dict[str, Any],
    device: str,
    apply_example_weight: bool = False,
    timing_weight: float = 1.15,
    note_off_weight: float = 1.25,
    eos_weight: float = 1.20,
    note_on_weight: float = 1.0,
    conditioning_mode: str = "instrument",
):
    import torch
    import torch.nn.functional as functional

    duration = float(record["durationSeconds"])
    instrument = canonical_instrument_name(
        record.get("instrumentFocus") or "acoustic_piano"
    )
    tokens = encode_instrument_clip(
        record["notes"],
        duration_seconds=duration,
        instrument=instrument,
    )
    inputs, targets = teacher_forcing_pair(
        tokens,
        initial_token_id=int(transcription._model.initial_token_id),
    )
    input_tensor = torch.tensor([inputs], dtype=torch.long, device=device)
    target_tensor = torch.tensor([targets], dtype=torch.long, device=device)
    wav = load_audio_clip(Path(record["audioClip"]), device)
    conditions = transcription._build_conditions(
        wav,
        conditioning_group(instrument, conditioning_mode),
    )
    provider = transcription._model.condition_provider
    prepared = provider.tokenize(conditions)
    condition_tensors = provider(prepared)
    logits = transcription._model(input_tensor, condition_tensors, first_step=True)
    per_token_loss = functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        target_tensor.reshape(-1),
        reduction="none",
    )
    weights = torch.tensor(
        target_token_weights(
            tokens,
            timing_weight,
            note_off_weight,
            eos_weight,
            note_on_weight,
        ),
        dtype=per_token_loss.dtype,
        device=device,
    )
    loss = (per_token_loss * weights).mean()
    if apply_example_weight:
        loss = loss * float(record.get("exampleWeight", 1.0))
    return loss


def evaluate_loss(
    transcription,
    records: list[dict[str, Any]],
    device: str,
    precision: str,
    conditioning_mode: str = "instrument",
) -> float:
    import torch

    transcription._model.eval()
    losses: list[float] = []
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    with torch.no_grad():
        for record in records:
            with torch.autocast(device_type="cuda", dtype=dtype, enabled=device.startswith("cuda")):
                losses.append(float(
                    clip_loss(
                        transcription,
                        record,
                        device,
                        conditioning_mode=conditioning_mode,
                    ).detach().cpu()
                ))
    return sum(losses) / max(1, len(losses))


def load_cached_decoded_metrics(
    evaluation_path: Path,
    records: list[dict[str, Any]],
    *,
    base_checkpoint: Path,
    validation_manifest: Path,
    instruments: tuple[str, ...],
) -> dict[str, Any]:
    """Reuse an immutable full-validation decode for a selected safety panel."""

    from ml.training.evaluate_checkpoint import evaluate_decoded_predictions

    if not evaluation_path.is_file():
        raise TrainingError(f"Cached baseline evaluation does not exist: {evaluation_path}")
    try:
        payload = json.loads(evaluation_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainingError(f"Cached baseline evaluation is unreadable: {evaluation_path}") from exc
    if payload.get("schema") != "polymath-checkpoint-evaluation-v1":
        raise TrainingError("Cached baseline evaluation has an unexpected schema")
    if Path(str(payload.get("checkpoint") or "")).resolve() != base_checkpoint.resolve():
        raise TrainingError("Cached baseline evaluation belongs to a different checkpoint")
    if Path(str(payload.get("validationManifest") or "")).resolve() != validation_manifest.resolve():
        raise TrainingError("Cached baseline evaluation belongs to a different validation manifest")
    if tuple(payload.get("instrumentConstraint") or ()) != instruments:
        raise TrainingError("Cached baseline evaluation used a different instrument constraint")
    decoded = (payload.get("metrics") or {}).get("decodedClips")
    if not isinstance(decoded, list):
        raise TrainingError("Cached baseline evaluation does not contain raw decoded clips")
    by_clip: dict[str, dict[str, Any]] = {}
    for item in decoded:
        if not isinstance(item, dict) or not str(item.get("clipId") or ""):
            raise TrainingError("Cached baseline evaluation contains an invalid clip entry")
        clip_id = str(item["clipId"])
        if clip_id in by_clip:
            raise TrainingError(f"Cached baseline evaluation repeats clip {clip_id}")
        by_clip[clip_id] = item

    predictions: list[list[dict[str, Any]]] = []
    for record in records:
        clip_id = str(record.get("clipId") or "")
        item = by_clip.get(clip_id)
        if item is None:
            raise TrainingError(f"Cached baseline evaluation is missing clip {clip_id}")
        if str(item.get("songId") or "unknown") != str(record.get("songId") or "unknown"):
            raise TrainingError(f"Cached baseline song mismatch for clip {clip_id}")
        if abs(float(item.get("sourceStart") or 0) - float(record.get("sourceStart") or 0)) > 1e-6:
            raise TrainingError(f"Cached baseline timing mismatch for clip {clip_id}")
        notes = item.get("notes")
        if not isinstance(notes, list):
            raise TrainingError(f"Cached baseline notes are missing for clip {clip_id}")
        predictions.append(notes)
    return evaluate_decoded_predictions(records, predictions)


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


def train(args: argparse.Namespace, progress_callback=None) -> dict[str, Any]:
    # Set cuBLAS determinism before importing/initializing CUDA in this process.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    import numpy as np
    import torch
    from muscriptor import TranscriptionModel
    from ml.training.decoded_checkpoint_gate import (
        decoded_checkpoint_gate,
        select_decoded_gate_records,
    )
    from ml.training.evaluate_checkpoint import evaluate_loaded_transcription

    numeric_seed = deterministic_seed(args.seed)
    random.seed(numeric_seed)
    np.random.seed(numeric_seed)
    torch.manual_seed(numeric_seed)
    torch.cuda.manual_seed_all(numeric_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)

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
    conditioning_mode = str(
        getattr(args, "conditioning_mode", "instrument") or "instrument"
    ).strip().lower()
    if conditioning_mode not in CONDITIONING_MODES:
        raise TrainingError("conditioningMode must be instrument or unconditioned")
    train_records = read_jsonl(args.train_manifest)
    validation_records = read_jsonl(args.validation_manifest)
    train_audit = audit_records(train_records)
    validation_audit = audit_records(validation_records)
    if len(train_audit["instruments"]) != 1:
        raise TrainingError(
            "One checkpoint run must target exactly one instrument; found "
            + ", ".join(train_audit["instruments"])
        )
    if validation_audit["instruments"] != train_audit["instruments"]:
        raise TrainingError("Training and validation instrument focus must match")
    if train_audit["songs"] < args.minimum_train_songs:
        raise TrainingError(
            f"Only {train_audit['songs']} training songs; minimum is {args.minimum_train_songs}. "
            "Do not overfit the foundation checkpoint to a handful of songs."
        )
    if args.gradient_accumulation < 1:
        raise TrainingError("Gradient accumulation must be at least one")
    validation_interval_steps = int(getattr(args, "validation_interval_steps", 0))
    if validation_interval_steps < 0:
        raise TrainingError("Validation interval steps cannot be negative")

    device = "cuda"
    transcription = TranscriptionModel.load_model(args.base, device=device)
    parameters = configure_trainable_parameters(transcription._model, args.train_last_layers)
    trainable_names = {
        name for name, parameter in transcription._model.named_parameters()
        if parameter.requires_grad
    }
    baseline_trainable_state = {
        name: parameter.detach().float().cpu().clone()
        for name, parameter in transcription._model.state_dict().items()
        if name in trainable_names
    }
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
    baseline_validation_loss = evaluate_loss(
        transcription,
        validation_records,
        device,
        args.precision,
        conditioning_mode,
    )
    if progress_callback:
        progress_callback(
            f"Baseline validation loss: {baseline_validation_loss:.6f}",
        )
    decoded_gate_clips_per_song = int(getattr(args, "decoded_gate_clips_per_song", 6))
    decoded_gate_records = select_decoded_gate_records(
        validation_records,
        maximum_per_song=decoded_gate_clips_per_song,
    )
    decoded_instruments = (
        None
        if conditioning_mode == "unconditioned"
        else (train_audit["instruments"][0],)
    )
    if progress_callback:
        progress_callback(
            f"Preparing baseline safety panel: {len(decoded_gate_records)} clips across "
            f"{len({str(row.get('songId') or 'unknown') for row in decoded_gate_records})} songs",
        )
    cached_baseline_evaluation = getattr(args, "baseline_decoded_evaluation", None)
    if cached_baseline_evaluation:
        baseline_decoded_metrics = load_cached_decoded_metrics(
            Path(cached_baseline_evaluation),
            decoded_gate_records,
            base_checkpoint=args.base,
            validation_manifest=args.validation_manifest,
            instruments=tuple(decoded_instruments or ()),
        )
        baseline_decoded_source = str(Path(cached_baseline_evaluation))
        if progress_callback:
            progress_callback("Reused the frozen full-validation baseline decode")
    else:
        baseline_decoded_metrics = evaluate_loaded_transcription(
            transcription,
            decoded_gate_records,
            progress_callback,
            decoded_instruments,
        )
        baseline_decoded_source = "decoded-live"
    selected_validation_loss = baseline_validation_loss
    lowest_validation_loss = baseline_validation_loss
    best_decoded_metrics = None
    best_selection_rank = None
    best_checkpoint_index = None
    best_state = None
    epoch_decisions: list[dict[str, Any]] = []
    checkpoint_decisions: list[dict[str, Any]] = []
    step = 0
    optimizer_step = 0
    accumulated = 0
    last_validation_optimizer_step = -1

    def assess_candidate(epoch_number: int, trigger: str) -> dict[str, Any]:
        """Evaluate one reversible snapshot against loss and decoded-note gates."""

        nonlocal selected_validation_loss, lowest_validation_loss
        nonlocal best_decoded_metrics, best_selection_rank, best_checkpoint_index
        nonlocal best_state
        nonlocal last_validation_optimizer_step

        validation_loss = evaluate_loss(
            transcription,
            validation_records,
            device,
            args.precision,
            conditioning_mode,
        )
        decision: dict[str, Any] = {
            "checkpointIndex": len(checkpoint_decisions) + 1,
            "epoch": epoch_number,
            "optimizerStep": optimizer_step,
            "trigger": trigger,
            "validationLoss": validation_loss,
            "lossImprovedVersusBaseline": validation_loss < baseline_validation_loss,
        }
        lowest_validation_loss = min(lowest_validation_loss, validation_loss)
        if progress_callback:
            progress_callback(
                f"Epoch {epoch_number}/{args.epochs}; optimizer step {optimizer_step}; "
                f"validation loss {validation_loss:.6f}",
            )
        if validation_loss < baseline_validation_loss:
            if progress_callback:
                progress_callback(
                    f"Checkpoint {decision['checkpointIndex']}: loss improved; "
                    "decoding the multi-song safety panel",
                )
            candidate_decoded_metrics = evaluate_loaded_transcription(
                transcription,
                decoded_gate_records,
                progress_callback,
                decoded_instruments,
            )
            gate = decoded_checkpoint_gate(
                baseline_decoded_metrics,
                candidate_decoded_metrics,
                maximum_aggregate_f1_regression=float(
                    getattr(args, "decoded_gate_max_aggregate_f1_regression", 0.001)
                ),
                maximum_aggregate_recall_regression=float(
                    getattr(args, "decoded_gate_max_aggregate_recall_regression", 0.002)
                ),
                maximum_per_song_f1_regression=float(
                    getattr(args, "decoded_gate_max_per_song_f1_regression", 0.010)
                ),
            )
            decision["decodedGate"] = gate
            if gate["passed"]:
                decision["eligible"] = True
                rank = decoded_selection_rank(candidate_decoded_metrics, validation_loss)
                decision["selectionRank"] = list(rank)
                decision["becameSelectionLeader"] = (
                    best_selection_rank is None or rank > best_selection_rank
                )
                if decision["becameSelectionLeader"]:
                    best_selection_rank = rank
                    best_checkpoint_index = decision["checkpointIndex"]
                    selected_validation_loss = validation_loss
                    best_decoded_metrics = candidate_decoded_metrics
                    best_state = {
                        name: parameter.detach().cpu().clone()
                        for name, parameter in transcription._model.state_dict().items()
                        if name in trainable_names
                    }
            else:
                decision["eligible"] = False
                decision["becameSelectionLeader"] = False
                decision["rejectionReason"] = "decoded-note safety gate failed"
        else:
            decision["eligible"] = False
            decision["becameSelectionLeader"] = False
            decision["rejectionReason"] = (
                "validation loss did not beat the immutable baseline"
            )
        checkpoint_decisions.append(decision)
        if trigger == "epoch-end":
            epoch_decisions.append(decision)
        last_validation_optimizer_step = optimizer_step
        print(json.dumps(decision), flush=True)
        transcription._model.train()
        return decision

    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        transcription._model.train()
        epoch_records = list(train_records)
        random.Random(f"{args.seed}:{epoch}").shuffle(epoch_records)
        for record in epoch_records:
            with torch.autocast(device_type="cuda", dtype=dtype):
                loss = clip_loss(
                    transcription,
                    record,
                    device,
                    apply_example_weight=True,
                    timing_weight=args.timing_token_weight,
                    note_off_weight=args.note_off_token_weight,
                    eos_weight=args.eos_token_weight,
                    note_on_weight=args.note_on_token_weight,
                    conditioning_mode=conditioning_mode,
                ) / args.gradient_accumulation
            loss.backward()
            step += 1
            accumulated += 1
            if accumulated == args.gradient_accumulation:
                torch.nn.utils.clip_grad_norm_(parameters, args.gradient_clip_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                accumulated = 0
                optimizer_step += 1
                if should_run_periodic_validation(
                    optimizer_step, validation_interval_steps
                ):
                    assess_candidate(epoch + 1, "optimizer-interval")
        if accumulated:
            torch.nn.utils.clip_grad_norm_(parameters, args.gradient_clip_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            accumulated = 0
            optimizer_step += 1
            if should_run_periodic_validation(
                optimizer_step, validation_interval_steps
            ):
                assess_candidate(epoch + 1, "optimizer-interval")
        if last_validation_optimizer_step != optimizer_step:
            assess_candidate(epoch + 1, "epoch-end")
        elif checkpoint_decisions and (
            not epoch_decisions or epoch_decisions[-1] is not checkpoint_decisions[-1]
        ):
            # The periodic checkpoint happened exactly on the final update of
            # this epoch. Reuse it as the epoch summary without decoding twice.
            epoch_decisions.append(checkpoint_decisions[-1])
    if best_state is None:
        raise TrainingError(
            "No evaluated checkpoint improved validation loss while passing decoded "
            "multi-song note gates; "
            "no candidate checkpoint was written"
        )
    for decision in checkpoint_decisions:
        decision["selected"] = decision["checkpointIndex"] == best_checkpoint_index
    transcription._model.load_state_dict(best_state, strict=False)
    weight_delta = summarize_weight_delta(
        transcription._model,
        baseline_trainable_state,
        trainable_names,
    )
    metadata = {
        "schema": "polymath-muscriptor-training-run-v1",
        "baseCheckpoint": str(args.base),
        "baseSha256": sha256_file(args.base),
        "trainManifest": str(args.train_manifest),
        "validationManifest": str(args.validation_manifest),
        "trainAudit": train_audit,
        "validationAudit": validation_audit,
        "instrumentFocus": train_audit["instruments"][0],
        "conditioningMode": conditioning_mode,
        "trainLastLayers": args.train_last_layers,
        "learningRate": args.learning_rate,
        "timingTokenWeight": args.timing_token_weight,
        "noteOnTokenWeight": args.note_on_token_weight,
        "noteOffTokenWeight": args.note_off_token_weight,
        "eosTokenWeight": args.eos_token_weight,
        "epochs": args.epochs,
        "validationIntervalSteps": validation_interval_steps,
        "optimizerSteps": optimizer_step,
        "seed": args.seed,
        "reproducibility": {
            "numericSeed": numeric_seed,
            "torchVersion": str(torch.__version__),
            "cudaRuntimeVersion": str(torch.version.cuda),
            "gpu": torch.cuda.get_device_name(0),
            "cublasWorkspaceConfig": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "deterministicAlgorithms": True,
            "cudnnBenchmark": False,
            "cudnnDeterministic": True,
            "allowTf32": False,
        },
        "baselineValidationLoss": baseline_validation_loss,
        "bestValidationLoss": selected_validation_loss,
        "selectedValidationLoss": selected_validation_loss,
        "lowestObservedValidationLoss": lowest_validation_loss,
        "decodedCheckpointSelection": {
            "schema": "polymath-decoded-checkpoint-selection-v2",
            "ranking": [
                "100ms.microF1",
                "100ms.recall",
                "250ms.microF1",
                "250ms.recall",
                "negativeValidationLoss",
            ],
            "selectedCheckpointIndex": best_checkpoint_index,
            "selectedRank": list(best_selection_rank) if best_selection_rank else None,
            "clipsPerSong": decoded_gate_clips_per_song,
            "records": [
                {
                    "clipId": str(record.get("clipId") or ""),
                    "songId": str(record.get("songId") or "unknown"),
                    "sourceStart": float(record.get("sourceStart") or 0),
                    "isNegativeExample": bool(record.get("isNegativeExample")),
                }
                for record in decoded_gate_records
            ],
            "instrumentConstraint": list(decoded_instruments or ()),
            "baselineDecodedSource": baseline_decoded_source,
            "baseline": baseline_decoded_metrics,
            "bestCandidate": best_decoded_metrics,
            "epochDecisions": epoch_decisions,
            "checkpointDecisions": checkpoint_decisions,
        },
        "weightDelta": weight_delta,
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
    parser.add_argument(
        "--baseline-decoded-evaluation",
        type=Path,
        help="Reuse raw original predictions from a matching frozen full-validation evaluation.",
    )
    parser.add_argument("--execute", action="store_true", help="Allow optimizer updates; otherwise audit only")
    parser.add_argument("--rights-acknowledgement", default="")
    parser.add_argument("--minimum-train-songs", type=int, default=20)
    parser.add_argument("--train-last-layers", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument(
        "--validation-interval-steps",
        type=int,
        default=0,
        help=(
            "Run frozen loss and decoded-note gates every N optimizer updates; "
            "zero keeps epoch-only validation."
        ),
    )
    parser.add_argument("--timing-token-weight", type=float, default=1.15)
    parser.add_argument("--note-on-token-weight", type=float, default=1.0)
    parser.add_argument("--note-off-token-weight", type=float, default=1.25)
    parser.add_argument("--eos-token-weight", type=float, default=1.20)
    parser.add_argument(
        "--conditioning-mode",
        choices=sorted(CONDITIONING_MODES),
        default="instrument",
        help="Use unconditioned only for a route decoded without an instrument constraint.",
    )
    parser.add_argument("--precision", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--seed", default="polymath-piano-phase1-v001")
    parser.add_argument(
        "--decoded-gate-clips-per-song",
        type=int,
        default=6,
        help="Evenly spread clips decoded per validation song after a loss improvement.",
    )
    parser.add_argument("--decoded-gate-max-aggregate-f1-regression", type=float, default=0.001)
    parser.add_argument("--decoded-gate-max-aggregate-recall-regression", type=float, default=0.002)
    parser.add_argument("--decoded-gate-max-per-song-f1-regression", type=float, default=0.010)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.train_manifest = args.train_manifest.resolve()
    args.validation_manifest = args.validation_manifest.resolve()
    args.base = args.base.resolve()
    args.out = args.out.resolve()
    if args.baseline_decoded_evaluation:
        args.baseline_decoded_evaluation = args.baseline_decoded_evaluation.resolve()
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
