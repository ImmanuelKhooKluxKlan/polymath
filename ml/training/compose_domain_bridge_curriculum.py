"""Compose a staged audio-domain bridge without inventing new labels.

The same trusted note labels can be paired with progressively harder audio
mixtures (for example 75% piano, then 50%, then 25%).  This utility appends a
chosen set of already-prepared bridge clips to a broad foundation manifest,
keeps every mixture under one composition identity, and scales its loss weight
explicitly.  Validation is copied unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        records.append(value)
    return records


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            for record in records
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


def parse_variant(value: str) -> tuple[str, float]:
    prefix, separator, raw_weight = value.partition("=")
    prefix = prefix.strip()
    if not separator or not prefix:
        raise ValueError("Each --variant must look like song-prefix=weight")
    weight = float(raw_weight)
    if not 0.0 < weight <= 1.0:
        raise ValueError("Bridge weights must be within (0, 1]")
    return prefix, weight


def compose(
    base_train: list[dict[str, Any]],
    validation: list[dict[str, Any]],
    bridge: list[dict[str, Any]],
    variants: list[tuple[str, float]],
    target_song_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if not base_train or not validation:
        raise ValueError("Base train and validation manifests must both be non-empty")
    if not target_song_id:
        raise ValueError("target_song_id is required")
    if target_song_id not in {str(record.get("songId") or "") for record in base_train}:
        raise ValueError(f"Target composition is absent from base train: {target_song_id}")

    selected: list[dict[str, Any]] = []
    variant_counts: dict[str, int] = {}
    variant_weight_totals: dict[str, float] = {}
    for prefix, multiplier in variants:
        matching = [
            record for record in bridge
            if str(record.get("songId") or "").startswith(prefix)
        ]
        if not matching:
            raise ValueError(f"No bridge clips matched variant prefix: {prefix}")
        variant_counts[prefix] = len(matching)
        variant_weight_totals[prefix] = 0.0
        for source in matching:
            record = dict(source)
            original_song_id = str(record.get("songId") or "")
            original_weight = float(record.get("exampleWeight", 1.0))
            if not 0.0 < original_weight <= 1.0:
                raise ValueError(f"{record.get('clipId')}: invalid source exampleWeight")
            adjusted_weight = original_weight * multiplier
            record["songId"] = target_song_id
            record["exampleWeight"] = adjusted_weight
            record["domainBridge"] = {
                "sourceSongId": original_song_id,
                "variantPrefix": prefix,
                "weightMultiplier": multiplier,
            }
            selected.append(record)
            variant_weight_totals[prefix] += adjusted_weight

    train = [dict(record) for record in base_train] + selected
    copied_validation = [dict(record) for record in validation]
    seen_clips: set[str] = set()
    song_splits: dict[str, str] = {}
    for split, records in (("train", train), ("validation", copied_validation)):
        for record in records:
            clip_id = str(record.get("clipId") or "").strip()
            song_id = str(record.get("songId") or "").strip()
            audio_clip = str(record.get("audioClip") or "").replace("\\", "/")
            if not clip_id or not song_id:
                raise ValueError("Every record requires clipId and songId")
            if clip_id in seen_clips:
                raise ValueError(f"Duplicate clipId: {clip_id}")
            if not audio_clip.startswith("/runpod-volume/"):
                raise ValueError(f"{clip_id}: audioClip is not on the RunPod volume")
            previous = song_splits.get(song_id)
            if previous and previous != split:
                raise ValueError(f"Composition leakage for {song_id}: {previous} and {split}")
            seen_clips.add(clip_id)
            song_splits[song_id] = split

    summary = {
        "schema": "polymath-domain-bridge-composition-v1",
        "targetSongId": target_song_id,
        "baseTrainClips": len(base_train),
        "bridgeClips": len(selected),
        "trainClips": len(train),
        "validationClips": len(copied_validation),
        "trainCompositions": len({str(record["songId"]) for record in train}),
        "validationCompositions": len(
            {str(record["songId"]) for record in copied_validation}
        ),
        "variants": [
            {
                "prefix": prefix,
                "weightMultiplier": multiplier,
                "clips": variant_counts[prefix],
                "effectiveWeight": round(variant_weight_totals[prefix], 6),
            }
            for prefix, multiplier in variants
        ],
        "leakageRule": "All audio mixtures of one composition share targetSongId before split validation.",
        "warning": "Private research curriculum; this does not establish commercial rights or held-out performance.",
    }
    return train, copied_validation, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-train", type=Path, required=True)
    parser.add_argument("--base-validation", type=Path, required=True)
    parser.add_argument("--bridge-manifest", type=Path, required=True)
    parser.add_argument("--variant", action="append", required=True)
    parser.add_argument("--target-song-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    sources = {
        "baseTrain": args.base_train.resolve(),
        "baseValidation": args.base_validation.resolve(),
        "bridgeManifest": args.bridge_manifest.resolve(),
    }
    variants = [parse_variant(value) for value in args.variant]
    train, validation, summary = compose(
        read_jsonl(sources["baseTrain"]),
        read_jsonl(sources["baseValidation"]),
        read_jsonl(sources["bridgeManifest"]),
        variants,
        args.target_song_id.strip(),
    )
    destination = args.out.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    write_jsonl(destination / "prepared-train.jsonl", train)
    write_jsonl(destination / "prepared-validation.jsonl", validation)
    summary["sources"] = {
        name: {"path": str(path), "sha256": sha256(path)}
        for name, path in sources.items()
    }
    (destination / "composition-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(destination), **summary}, indent=2))


if __name__ == "__main__":
    main()
