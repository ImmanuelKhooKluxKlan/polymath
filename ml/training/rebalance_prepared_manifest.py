"""Create a candidate-only training manifest with explicit corpus balancing.

The foundation corpus is intentionally broad, but a large synthetic subset can
overwhelm the much smaller reviewed pianist-style subset.  This utility lowers
the contribution of non-priority songs without duplicating clips or touching
their audio.  The source manifest remains immutable and every change is audited.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def rebalance_records(
    records: Iterable[dict[str, Any]],
    *,
    priority_song_ids: set[str],
    background_weight: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not 0.0 < background_weight <= 1.0:
        raise ValueError("background_weight must be within (0, 1]")
    if not priority_song_ids:
        raise ValueError("At least one priority song is required")

    output: list[dict[str, Any]] = []
    clips_by_song: Counter[str] = Counter()
    effective_weight_by_song: Counter[str] = Counter()
    found_priority: set[str] = set()
    changed = 0
    for source in records:
        record = dict(source)
        song_id = str(record.get("songId") or "").strip()
        if not song_id:
            raise ValueError("Manifest record is missing songId")
        is_priority = song_id in priority_song_ids
        if is_priority:
            found_priority.add(song_id)
        requested_weight = 1.0 if is_priority else background_weight
        original_weight = float(record.get("exampleWeight", 1.0))
        new_weight = min(original_weight, requested_weight)
        if abs(new_weight - original_weight) > 1e-12:
            changed += 1
        record["exampleWeight"] = new_weight
        output.append(record)
        clips_by_song[song_id] += 1
        effective_weight_by_song[song_id] += new_weight

    missing = sorted(priority_song_ids - found_priority)
    if missing:
        raise ValueError(f"Priority songs not present in manifest: {', '.join(missing)}")
    audit = {
        "schema": "polymath-prepared-manifest-rebalance-v1",
        "prioritySongIds": sorted(priority_song_ids),
        "backgroundWeight": background_weight,
        "records": len(output),
        "changedRecords": changed,
        "clipsBySong": dict(sorted(clips_by_song.items())),
        "effectiveWeightBySong": {
            key: round(value, 6)
            for key, value in sorted(effective_weight_by_song.items())
        },
        "warning": (
            "Candidate-only weighting experiment. It does not create new evidence "
            "or make an opened song an untouched holdout."
        ),
    }
    return output, audit


def equalize_song_weights(
    records: Iterable[dict[str, Any]],
    *,
    target_effective_weight: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Scale each song to the same total example weight.

    Scaling preserves the relative weights inside a song (for example, a
    reviewed-silence clip remains lighter than a positive clip).  A target is
    rejected when it would require any clip weight to exceed the trainer's
    maximum of 1.0; silently clipping it would make the corpus unequal again.
    """
    if not 0.0 < target_effective_weight:
        raise ValueError("target_effective_weight must be positive")

    source_records = [dict(record) for record in records]
    if not source_records:
        raise ValueError("At least one record is required")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    original_totals: Counter[str] = Counter()
    for record in source_records:
        song_id = str(record.get("songId") or "").strip()
        clip_id = str(record.get("clipId") or "").strip()
        if not song_id or not clip_id:
            raise ValueError("Every manifest record requires songId and clipId")
        weight = float(record.get("exampleWeight", 1.0))
        if not 0.0 < weight <= 1.0:
            raise ValueError(f"{clip_id}: exampleWeight must be within (0, 1]")
        grouped[song_id].append(record)
        original_totals[song_id] += weight

    scales: dict[str, float] = {}
    for song_id, song_records in grouped.items():
        scale = target_effective_weight / float(original_totals[song_id])
        maximum_scaled_weight = max(
            float(record.get("exampleWeight", 1.0)) * scale
            for record in song_records
        )
        if maximum_scaled_weight > 1.0 + 1e-12:
            raise ValueError(
                f"{song_id}: target {target_effective_weight:g} would require "
                f"exampleWeight {maximum_scaled_weight:.6f} above 1.0"
            )
        scales[song_id] = scale

    output: list[dict[str, Any]] = []
    achieved_totals: Counter[str] = Counter()
    changed = 0
    for record in source_records:
        song_id = str(record["songId"]).strip()
        original = float(record.get("exampleWeight", 1.0))
        updated = original * scales[song_id]
        if abs(updated - original) > 1e-12:
            changed += 1
        record["exampleWeight"] = updated
        output.append(record)
        achieved_totals[song_id] += updated

    audit = {
        "schema": "polymath-prepared-manifest-song-equalization-v1",
        "targetEffectiveWeightPerSong": target_effective_weight,
        "records": len(output),
        "songs": len(grouped),
        "changedRecords": changed,
        "clipsBySong": {
            song_id: len(song_records)
            for song_id, song_records in sorted(grouped.items())
        },
        "originalEffectiveWeightBySong": {
            key: round(value, 9) for key, value in sorted(original_totals.items())
        },
        "scaleBySong": {
            key: round(value, 12) for key, value in sorted(scales.items())
        },
        "effectiveWeightBySong": {
            key: round(value, 9) for key, value in sorted(achieved_totals.items())
        },
        "warning": (
            "Candidate-only corpus balancing. Equal song influence is not a "
            "substitute for multi-domain decoded-note evaluation."
        ),
    }
    return output, audit


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        records.append(value)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--priority-song", action="append", default=[])
    parser.add_argument("--background-weight", type=float, default=0.1)
    parser.add_argument("--target-effective-weight-per-song", type=float)
    args = parser.parse_args()

    source_records = read_jsonl(Path(args.input))
    if args.target_effective_weight_per_song is not None:
        if args.priority_song:
            raise ValueError(
                "--priority-song cannot be combined with "
                "--target-effective-weight-per-song"
            )
        records, audit = equalize_song_weights(
            source_records,
            target_effective_weight=args.target_effective_weight_per_song,
        )
    else:
        records, audit = rebalance_records(
            source_records,
            priority_song_ids={
                str(value).strip() for value in args.priority_song if str(value).strip()
            },
            background_weight=args.background_weight,
        )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
    )
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path), "report": str(report_path), **audit}, indent=2))


if __name__ == "__main__":
    main()
