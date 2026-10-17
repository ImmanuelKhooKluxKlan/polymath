"""Add low-weight original-song audio supervision without changing its labels.

The reference labels come from a reviewed piano arrangement, while the audio
for each new example is the original full mix used by production inference.
This deliberately small domain bridge teaches the existing transcription model
which arrangement target is expected from vocals plus accompaniment. It does
not claim that the labels are literal source separation ground truth.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any, Iterable

from ml.training.prepare_audio_clips import prepare_manifest


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        rows.append(value)
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
        newline="\n",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def start_label(value: float) -> str:
    return f"{round(value * 1000):07d}ms"


def build_original_domain_rows(
    train: list[dict[str, Any]],
    *,
    target_song_id: str,
    reference_variant_prefix: str,
    original_variant_prefix: str,
    source_media: str,
    example_weight: float,
    maximum_target_training_end: float | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not target_song_id:
        raise ValueError("target_song_id is required")
    if not reference_variant_prefix or not original_variant_prefix:
        raise ValueError("Both variant prefixes are required")
    if reference_variant_prefix == original_variant_prefix:
        raise ValueError("Original and reference variant prefixes must differ")
    if not str(source_media).strip():
        raise ValueError("source_media is required")
    if not math.isfinite(example_weight) or example_weight <= 0:
        raise ValueError("example_weight must be finite and positive")
    if maximum_target_training_end is not None and (
        not math.isfinite(maximum_target_training_end)
        or maximum_target_training_end <= 0
    ):
        raise ValueError("maximum_target_training_end must be finite and positive")

    existing_ids = [str(row.get("clipId") or "") for row in train]
    if len(existing_ids) != len(set(existing_ids)):
        raise ValueError("Training manifest contains duplicate clip IDs")
    if any(clip_id.startswith(original_variant_prefix) for clip_id in existing_ids):
        raise ValueError("Original-domain variant already exists in the training manifest")

    references: dict[float, dict[str, Any]] = {}
    for row in train:
        if str(row.get("songId") or "") != target_song_id:
            continue
        if not str(row.get("clipId") or "").startswith(reference_variant_prefix):
            continue
        start = float(row.get("sourceStart") or 0)
        duration = float(row.get("durationSeconds") or 0)
        if not math.isfinite(start) or not math.isfinite(duration) or duration <= 0:
            raise ValueError(
                f"Invalid reference interval for {row.get('clipId')}: "
                f"start={start}, duration={duration}"
            )
        if start in references:
            raise ValueError(f"Reference variant repeats sourceStart {start}")
        if (
            maximum_target_training_end is not None
            and start + duration > maximum_target_training_end + 1e-9
        ):
            raise ValueError(
                f"Reference interval at {start} extends beyond the trusted "
                f"target-label cutoff ({maximum_target_training_end})"
            )
        references[start] = row

    if not references:
        raise ValueError("No matching reference variants were found")

    variants: list[dict[str, Any]] = []
    for start, source in sorted(references.items()):
        record = copy.deepcopy(source)
        record.update({
            "clipId": f"{original_variant_prefix}{start_label(start)}",
            "sourceMedia": str(source_media),
            "exampleWeight": example_weight,
            "split": "train",
            "originalDomainVariant": {
                "schema": "polymath-original-domain-supervision-v1",
                "referenceClipId": str(source.get("clipId") or ""),
                "audioDomain": "original-full-mix",
                "labelDomain": "reviewed-piano-arrangement",
                "pianoRatio": 0.0,
                "sourceRatio": 1.0,
                "perFileNormalization": False,
            },
        })
        record.pop("audioClip", None)
        record.pop("localAudioSource", None)
        # Original recordings are independent of a synthetic reference render.
        record.pop("audioLabelCoherence", None)
        record.pop("audioRebuildRequired", None)
        variants.append(record)

    generated_ids = [str(row["clipId"]) for row in variants]
    if len(generated_ids) != len(set(generated_ids)):
        raise ValueError("Generated original-domain clip IDs are not unique")
    if set(generated_ids).intersection(existing_ids):
        raise ValueError("Generated original-domain clip IDs collide with training data")

    summary = {
        "schema": "polymath-original-domain-supervision-receipt-v1",
        "targetSongId": target_song_id,
        "referenceVariantPrefix": reference_variant_prefix,
        "originalVariantPrefix": original_variant_prefix,
        "sourceMedia": str(source_media),
        "exampleWeight": example_weight,
        "maximumTargetTrainingEnd": maximum_target_training_end,
        "sourceTrainClips": len(train),
        "originalDomainClips": len(variants),
        "sourceStarts": [float(row["sourceStart"]) for row in variants],
        "interpretation": (
            "Low-weight domain bridge from original full-mix audio to reviewed piano-"
            "arrangement labels. This is training supervision, not certification data."
        ),
    }
    return variants, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--target-song-id", required=True)
    parser.add_argument("--reference-variant-prefix", required=True)
    parser.add_argument("--original-variant-prefix", required=True)
    parser.add_argument("--source-media", required=True)
    parser.add_argument("--example-weight", type=float, required=True)
    parser.add_argument("--maximum-target-training-end", type=float)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument(
        "--runtime-root",
        help="RunPod root written into rendered audioClip paths.",
    )
    args = parser.parse_args()

    train_path = args.train_manifest.resolve()
    validation_path = args.validation_manifest.resolve()
    train = read_jsonl(train_path)
    variants, summary = build_original_domain_rows(
        train,
        target_song_id=args.target_song_id.strip(),
        reference_variant_prefix=args.reference_variant_prefix.strip(),
        original_variant_prefix=args.original_variant_prefix.strip(),
        source_media=args.source_media.strip(),
        example_weight=args.example_weight,
        maximum_target_training_end=args.maximum_target_training_end,
    )

    destination = args.out.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    source_manifest = destination / "original-domain.jsonl"
    write_jsonl(source_manifest, variants)
    rendered = prepare_manifest(
        source_manifest,
        destination,
        args.ffmpeg.resolve(),
        overwrite=False,
        manifest_audio_root=args.runtime_root,
    )
    prepared_variants_path = Path(rendered["preparedManifest"])
    prepared_variants = read_jsonl(prepared_variants_path)

    prepared_train = destination / "prepared-train.jsonl"
    prepared_validation = destination / "prepared-validation.jsonl"
    write_jsonl(prepared_train, [*train, *prepared_variants])
    shutil.copyfile(validation_path, prepared_validation)

    summary["finalTrainClips"] = len(train) + len(prepared_variants)
    summary["validationClips"] = len(read_jsonl(validation_path))
    summary["sources"] = {
        "trainManifest": {"path": str(train_path), "sha256": sha256(train_path)},
        "validationManifest": {
            "path": str(validation_path),
            "sha256": sha256(validation_path),
        },
    }
    summary["outputs"] = {
        "originalDomainSource": {
            "path": str(source_manifest),
            "sha256": sha256(source_manifest),
        },
        "originalDomainPrepared": {
            "path": str(prepared_variants_path),
            "sha256": sha256(prepared_variants_path),
        },
        "trainManifest": {"path": str(prepared_train), "sha256": sha256(prepared_train)},
        "validationManifest": {
            "path": str(prepared_validation),
            "sha256": sha256(prepared_validation),
        },
    }
    receipt = destination / "original-domain-supervision-receipt.json"
    receipt.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(destination), **summary}, indent=2))


if __name__ == "__main__":
    main()
