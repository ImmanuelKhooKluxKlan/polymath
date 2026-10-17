"""Replace one prepared composition's labels without touching its audio.

This is useful when a stronger, independently reviewed supervision package is
created for recorded source audio. Synthetic audio rendered from the old
labels must be rebuilt; provenance-marked synthetic clips receive an explicit
blocking flag until that rebuild completes. Every matching clip keeps its
audio path, split, and identity. A
clip is relabelled only when its complete time range remains inside approved
quality windows in the replacement package; otherwise the command fails
closed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from ml.training.dataset_builder import (
    ALLOWED_WINDOW_STATUS,
    SCHEMA,
    clip_notes,
    windows_overlapping,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def relabel(
    records: list[dict[str, Any]],
    package: dict[str, Any],
    *,
    target_song_id: str,
    package_path: Path,
    target_weight_multiplier: float = 1.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if package.get("schema") != SCHEMA:
        raise ValueError(f"replacement package schema must be {SCHEMA!r}")
    notes = package.get("notes")
    alignment = package.get("alignment")
    windows = alignment.get("qualityWindows") if isinstance(alignment, dict) else None
    if not isinstance(notes, list) or not notes:
        raise ValueError("replacement package has no notes")
    if not isinstance(windows, list) or not windows:
        raise ValueError("replacement package has no alignment quality windows")
    if not 0.0 < target_weight_multiplier <= 1.0:
        raise ValueError("target_weight_multiplier must be within (0, 1]")

    output: list[dict[str, Any]] = []
    touched = 0
    before_notes = 0
    after_notes = 0
    rebuild_required = 0
    package_digest = sha256(package_path)
    for source in records:
        if str(source.get("songId") or "") != target_song_id:
            output.append(dict(source))
            continue
        start = float(source.get("sourceStart") or 0.0)
        duration = float(source.get("durationSeconds") or 0.0)
        if duration <= 0:
            raise ValueError(f"{source.get('clipId')}: durationSeconds must be positive")
        end = start + duration
        overlap = windows_overlapping(start, end, windows)
        if not overlap:
            raise ValueError(f"{source.get('clipId')}: no replacement quality window")
        unsafe = [
            window
            for window in overlap
            if str(window.get("status") or "") not in ALLOWED_WINDOW_STATUS
        ]
        if unsafe:
            ids = [str(window.get("id") or "unknown") for window in unsafe]
            raise ValueError(
                f"{source.get('clipId')}: replacement overlaps unapproved windows {ids}"
            )
        instrument = str(source.get("instrumentFocus") or "acoustic_piano")
        replacement = clip_notes(notes, start, end, instrument)
        if not replacement:
            raise ValueError(f"{source.get('clipId')}: replacement produced no labels")
        bridge = source.get("domainBridge")
        positive_weight = target_weight_multiplier * (
            float(bridge.get("weightMultiplier", 1.0))
            if isinstance(bridge, dict)
            else 1.0
        )
        if not 0.0 < positive_weight <= 1.0:
            raise ValueError(
                f"{source.get('clipId')}: invalid domain-bridge positive weight"
            )
        before_notes += len(source.get("notes") or [])
        after_notes += len(replacement)
        touched += 1
        coherence = source.get("audioLabelCoherence")
        needs_rebuild = bool(source.get("audioRebuildRequired")) or (
            isinstance(coherence, dict)
            and coherence.get("labelSourceSha256") != package_digest
        )
        rebuild_required += int(needs_rebuild)
        output.append({
            **source,
            "notes": replacement,
            "targetState": "notes",
            "isNegativeExample": False,
            "exampleWeight": positive_weight,
            "qualityWindowIds": [window.get("id") for window in overlap],
            "supervisionPackage": str(package_path.resolve()),
            "audioRebuildRequired": needs_rebuild,
            "relabeling": {
                "schema": "polymath-prepared-composition-relabel-v1",
                "replacementSupervisionSha256": package_digest,
                "audioReused": True,
                "previousTargetState": source.get("targetState"),
                "previousNegativeExample": bool(source.get("isNegativeExample")),
                "previousExampleWeight": float(source.get("exampleWeight", 1.0)),
            },
        })
    if touched == 0:
        raise ValueError(f"target song id is absent: {target_song_id}")
    return output, {
        "targetSongId": target_song_id,
        "relabelledClips": touched,
        "notesBefore": before_notes,
        "notesAfter": after_notes,
        "noteDelta": after_notes - before_notes,
        "targetWeightMultiplier": target_weight_multiplier,
        "syntheticClipsRequiringAudioRebuild": rebuild_required,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--supervision-package", type=Path, required=True)
    parser.add_argument("--target-song-id", required=True)
    parser.add_argument("--target-weight-multiplier", type=float, default=1.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sources = {
        "trainManifest": args.train_manifest.resolve(),
        "validationManifest": args.validation_manifest.resolve(),
        "supervisionPackage": args.supervision_package.resolve(),
    }
    package = json.loads(sources["supervisionPackage"].read_text(encoding="utf-8"))
    train, summary = relabel(
        read_jsonl(sources["trainManifest"]),
        package,
        target_song_id=args.target_song_id.strip(),
        package_path=sources["supervisionPackage"],
        target_weight_multiplier=args.target_weight_multiplier,
    )
    validation = read_jsonl(sources["validationManifest"])
    destination = args.out.resolve()
    if destination.exists():
        raise ValueError(f"refusing to overwrite existing output: {destination}")
    destination.mkdir(parents=True)
    write_jsonl(destination / "prepared-train.jsonl", train)
    write_jsonl(destination / "prepared-validation.jsonl", validation)
    receipt = {
        "schema": "polymath-prepared-composition-relabel-receipt-v1",
        **summary,
        "trainClips": len(train),
        "validationClips": len(validation),
        "sources": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in sources.items()
        },
        "outputs": {
            "trainSha256": sha256(destination / "prepared-train.jsonl"),
            "validationSha256": sha256(destination / "prepared-validation.jsonl"),
        },
        "safety": {
            "audioMutated": False,
            "validationMutated": False,
            "unapprovedWindowsAllowed": False,
        },
    }
    (destination / "relabel-receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(destination), **receipt}, indent=2))


if __name__ == "__main__":
    main()
