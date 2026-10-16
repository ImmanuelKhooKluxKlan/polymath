"""Create a leakage-aware, target-domain development split.

Fine-tuning on several audio mixtures from one song can make ordinary clip
validation misleading.  This utility withholds explicitly chosen time windows
from *every* training mixture of that composition, removes neighbouring clips
whose audio overlaps those windows, and builds a development manifest using
the original full-mix audio with the same reviewed note labels.

The result is an opened development set for checkpoint selection.  It is not
an untouched holdout and must never be described as certification accuracy.
"""

from __future__ import annotations

import argparse
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


def intervals_overlap(
    left_start: float,
    left_duration: float,
    right_start: float,
    right_duration: float,
) -> bool:
    """Return true only when half-open audio intervals overlap in time."""

    return max(left_start, right_start) < min(
        left_start + left_duration,
        right_start + right_duration,
    )


def start_label(value: float) -> str:
    milliseconds = round(value * 1000)
    return f"{milliseconds:07d}ms"


def build_split(
    train: list[dict[str, Any]],
    validation: list[dict[str, Any]],
    *,
    target_song_id: str,
    reference_variant_prefix: str,
    target_development_song_id: str,
    source_media: str,
    development_starts: list[float],
    maximum_target_training_end: float | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if not target_song_id or not target_development_song_id:
        raise ValueError("Target and target-development song IDs are required")
    if target_song_id == target_development_song_id:
        raise ValueError("Development song ID must be visibly distinct from training")
    if not reference_variant_prefix:
        raise ValueError("reference_variant_prefix is required")
    if not str(source_media).strip():
        raise ValueError("source_media is required")
    if len(development_starts) < 2:
        raise ValueError("At least two development starts are required")
    if any(not math.isfinite(value) or value < 0 for value in development_starts):
        raise ValueError("Development starts must be finite and non-negative")
    if len(set(development_starts)) != len(development_starts):
        raise ValueError("Development starts must be unique")
    if maximum_target_training_end is not None:
        if (
            not math.isfinite(maximum_target_training_end)
            or maximum_target_training_end <= 0
        ):
            raise ValueError("maximum_target_training_end must be finite and positive")

    existing_song_ids = {
        str(row.get("songId") or "") for row in [*train, *validation]
    }
    if target_development_song_id in existing_song_ids:
        raise ValueError("Development song ID already exists in the source manifests")

    target_rows = [
        row for row in train if str(row.get("songId") or "") == target_song_id
    ]
    if not target_rows:
        raise ValueError(f"Target song is absent from training: {target_song_id}")
    reference_by_start: dict[float, dict[str, Any]] = {}
    for row in target_rows:
        if str(row.get("clipId") or "").startswith(reference_variant_prefix):
            start = float(row.get("sourceStart") or 0)
            if start in reference_by_start:
                raise ValueError(f"Reference variant repeats sourceStart {start}")
            reference_by_start[start] = row

    missing = [start for start in development_starts if start not in reference_by_start]
    if missing:
        raise ValueError(
            "Reference labels are missing development starts: "
            + ", ".join(str(value) for value in missing)
        )

    development: list[dict[str, Any]] = []
    development_intervals: list[tuple[float, float]] = []
    for start in sorted(development_starts):
        source = reference_by_start[start]
        duration = float(source.get("durationSeconds") or 0)
        if not 0 < duration <= 5.001:
            raise ValueError(f"Invalid reference duration at {start}: {duration}")
        if (
            maximum_target_training_end is not None
            and start + duration > maximum_target_training_end + 1e-9
        ):
            raise ValueError(
                f"Development interval at {start} extends beyond the trusted "
                f"target-label cutoff ({maximum_target_training_end})"
            )
        clip_id = f"target-original-dev-{start_label(start)}"
        record = dict(source)
        record.update({
            "clipId": clip_id,
            "songId": target_development_song_id,
            "sourceMedia": str(source_media),
            "sourceStart": start,
            "split": "target-development",
            "exampleWeight": 1.0,
            "developmentSplit": {
                "schema": "polymath-target-development-split-v1",
                "sourceTrainingSongId": target_song_id,
                "referenceClipId": str(source.get("clipId") or ""),
                "audioDomain": "original-full-mix",
                "selectionStatus": "opened-development",
            },
        })
        record.pop("audioClip", None)
        record.pop("localAudioSource", None)
        development.append(record)
        development_intervals.append((start, duration))

    kept: list[dict[str, Any]] = []
    removed_overlapping: list[dict[str, Any]] = []
    removed_after_cutoff: list[dict[str, Any]] = []
    for row in train:
        if str(row.get("songId") or "") != target_song_id:
            kept.append(row)
            continue
        start = float(row.get("sourceStart") or 0)
        duration = float(row.get("durationSeconds") or 0)
        if not math.isfinite(start) or not math.isfinite(duration) or duration <= 0:
            raise ValueError(
                f"Invalid target training interval for {row.get('clipId')}: "
                f"start={start}, duration={duration}"
            )
        if (
            maximum_target_training_end is not None
            and start + duration > maximum_target_training_end + 1e-9
        ):
            removed_after_cutoff.append(row)
            continue
        overlaps = any(
            intervals_overlap(start, duration, dev_start, dev_duration)
            for dev_start, dev_duration in development_intervals
        )
        if overlaps:
            removed_overlapping.append(row)
        else:
            kept.append(row)

    leaking = []
    for row in kept:
        if str(row.get("songId") or "") != target_song_id:
            continue
        start = float(row.get("sourceStart") or 0)
        duration = float(row.get("durationSeconds") or 0)
        if any(
            intervals_overlap(start, duration, dev_start, dev_duration)
            for dev_start, dev_duration in development_intervals
        ):
            leaking.append(str(row.get("clipId") or ""))
    if leaking:
        raise ValueError("Temporal leakage remained after filtering: " + ", ".join(leaking[:5]))

    removed_overlap_starts = sorted(
        {float(row.get("sourceStart") or 0) for row in removed_overlapping}
    )
    removed_cutoff_starts = sorted(
        {float(row.get("sourceStart") or 0) for row in removed_after_cutoff}
    )
    summary = {
        "schema": "polymath-target-development-split-receipt-v2",
        "targetTrainingSongId": target_song_id,
        "targetDevelopmentSongId": target_development_song_id,
        "referenceVariantPrefix": reference_variant_prefix,
        "sourceMedia": str(source_media),
        "developmentStarts": sorted(development_starts),
        "developmentClips": len(development),
        "trainClipsBefore": len(train),
        "trainClipsAfter": len(kept),
        "targetTrainClipsBefore": len(target_rows),
        "targetTrainClipsAfter": sum(
            str(row.get("songId") or "") == target_song_id for row in kept
        ),
        "removedOverlappingClips": len(removed_overlapping),
        "removedOverlappingSourceStarts": removed_overlap_starts,
        "maximumTargetTrainingEnd": maximum_target_training_end,
        "removedAfterTrustedCutoffClips": len(removed_after_cutoff),
        "removedAfterTrustedCutoffSourceStarts": removed_cutoff_starts,
        "validationClips": len(validation),
        "temporalOverlapCountAfterFiltering": 0,
        "intervalConvention": "half-open [start,start+duration)",
        "interpretation": (
            "Opened target-domain development split used for model selection. "
            "It is not an untouched holdout or production accuracy claim."
        ),
    }
    return kept, development, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--target-song-id", required=True)
    parser.add_argument("--reference-variant-prefix", required=True)
    parser.add_argument("--target-development-song-id", required=True)
    parser.add_argument("--source-media", required=True)
    parser.add_argument("--development-start", type=float, action="append", required=True)
    parser.add_argument(
        "--maximum-target-training-end",
        type=float,
        help=(
            "Exclude every target-song clip whose half-open interval extends past this "
            "trusted-label boundary. Other compositions are unaffected."
        ),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--runtime-root",
        help="RunPod root written into rendered audioClip paths.",
    )
    parser.add_argument(
        "--ffmpeg",
        type=Path,
        help="When supplied, render the development clips and freeze their manifest hash.",
    )
    args = parser.parse_args()

    train_path = args.train_manifest.resolve()
    validation_path = args.validation_manifest.resolve()
    train = read_jsonl(train_path)
    validation = read_jsonl(validation_path)
    filtered, development, summary = build_split(
        train,
        validation,
        target_song_id=args.target_song_id.strip(),
        reference_variant_prefix=args.reference_variant_prefix.strip(),
        target_development_song_id=args.target_development_song_id.strip(),
        source_media=args.source_media.strip(),
        development_starts=args.development_start,
        maximum_target_training_end=args.maximum_target_training_end,
    )

    destination = args.out.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    prepared_train = destination / "prepared-train.jsonl"
    prepared_validation = destination / "prepared-validation.jsonl"
    target_source = destination / "target-development.jsonl"
    write_jsonl(prepared_train, filtered)
    shutil.copyfile(validation_path, prepared_validation)
    write_jsonl(target_source, development)

    prepared_target = None
    if args.ffmpeg:
        rendered = prepare_manifest(
            target_source,
            destination,
            args.ffmpeg.resolve(),
            overwrite=False,
            manifest_audio_root=args.runtime_root,
        )
        prepared_target = Path(rendered["preparedManifest"])

    summary["sources"] = {
        "trainManifest": {"path": str(train_path), "sha256": sha256(train_path)},
        "validationManifest": {
            "path": str(validation_path),
            "sha256": sha256(validation_path),
        },
    }
    summary["outputs"] = {
        "trainManifest": {"path": str(prepared_train), "sha256": sha256(prepared_train)},
        "validationManifest": {
            "path": str(prepared_validation),
            "sha256": sha256(prepared_validation),
        },
        "targetDevelopmentSource": {
            "path": str(target_source),
            "sha256": sha256(target_source),
        },
        "targetDevelopmentPrepared": (
            {"path": str(prepared_target), "sha256": sha256(prepared_target)}
            if prepared_target
            else None
        ),
    }
    summary["rendered"] = prepared_target is not None
    receipt = destination / "development-split-receipt.json"
    receipt.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(destination), **summary}, indent=2))


if __name__ == "__main__":
    main()
