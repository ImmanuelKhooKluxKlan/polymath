"""Build leakage-safe five-second training manifests from reviewed supervision packages.

This module does not train the neural network. It converts human-reviewed alignment
packages into deterministic clip records that a MuScriptor-compatible trainer can
consume. Keeping this boundary separate prevents uncertain alignment regions from
silently becoming false ground truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "polymath-supervision-package-v1"
ALLOWED_WINDOW_STATUS = {"trusted", "accepted-manually", "neutral"}


class DatasetError(ValueError):
    """Raised when an input package is unsafe or inconsistent."""


@dataclass(frozen=True)
class SongEntry:
    song_id: str
    source_media: Path
    supervision_package: Path
    allowed_for_training: bool
    rights_note: str
    forced_split: str | None = None


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetError(f"Could not read JSON {path}: {exc}") from exc


def sha256_file(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_split(song_id: str, seed: str, train_share: float, validation_share: float) -> str:
    value = int(hashlib.sha256(f"{seed}:{song_id}".encode("utf-8")).hexdigest()[:16], 16) / 2**64
    if value < train_share:
        return "train"
    if value < train_share + validation_share:
        return "validation"
    return "test"


def load_index(path: Path) -> list[SongEntry]:
    payload = read_json(path)
    items = payload.get("songs") if isinstance(payload, dict) else None
    if not isinstance(items, list) or not items:
        raise DatasetError("Training index must contain a non-empty 'songs' array.")
    base = path.parent
    entries: list[SongEntry] = []
    seen: set[str] = set()
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise DatasetError(f"Song entry {index + 1} is not an object.")
        song_id = str(item.get("songId") or "").strip()
        if not song_id or song_id in seen:
            raise DatasetError(f"Song entry {index + 1} has a missing or duplicate songId.")
        seen.add(song_id)
        source = (base / str(item.get("sourceMedia") or "")).resolve()
        package = (base / str(item.get("supervisionPackage") or "")).resolve()
        rights = item.get("rights") if isinstance(item.get("rights"), dict) else {}
        entries.append(SongEntry(
            song_id=song_id,
            source_media=source,
            supervision_package=package,
            allowed_for_training=bool(rights.get("allowedForTraining", False)),
            rights_note=str(rights.get("note") or "").strip(),
            forced_split=str(item.get("split") or "").strip().lower() or None,
        ))
    return entries


def validate_package(entry: SongEntry, package: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if package.get("schema") != SCHEMA:
        errors.append(f"schema must be {SCHEMA!r}")
    if not entry.source_media.is_file():
        errors.append(f"source media is missing: {entry.source_media}")
    if not entry.allowed_for_training:
        errors.append("rights.allowedForTraining is not true")
    if not entry.rights_note:
        errors.append("rights.note is empty; record ownership/licence/consent evidence")

    timeline = package.get("timeline") if isinstance(package.get("timeline"), dict) else {}
    source_duration = timeline.get("sourceDurationSeconds")
    if not isinstance(source_duration, (int, float)) or source_duration <= 0:
        errors.append("timeline.sourceDurationSeconds must be a positive number")

    alignment = package.get("alignment") if isinstance(package.get("alignment"), dict) else {}
    windows = alignment.get("qualityWindows")
    if not isinstance(windows, list) or not windows:
        errors.append("alignment.qualityWindows must be a non-empty array")
        windows = []
    notes = package.get("notes")
    if not isinstance(notes, list) or not notes:
        errors.append("notes must be a non-empty array")
        notes = []

    previous_time = -1.0
    for note_index, note in enumerate(notes):
        if not isinstance(note, dict):
            errors.append(f"note {note_index} is not an object")
            continue
        midi = note.get("midi")
        start = note.get("time")
        duration = note.get("duration")
        if not isinstance(midi, int) or midi < 0 or midi > 127:
            errors.append(f"note {note_index} has invalid MIDI pitch")
        if not isinstance(start, (int, float)) or start < 0:
            errors.append(f"note {note_index} has invalid start time")
            continue
        if start + 1e-7 < previous_time:
            errors.append("notes are not sorted by source time")
        previous_time = max(previous_time, float(start))
        if not isinstance(duration, (int, float)) or duration <= 0:
            errors.append(f"note {note_index} has invalid duration")
        elif isinstance(source_duration, (int, float)) and start + duration > source_duration + 0.02:
            errors.append(f"note {note_index} exceeds the source timeline")
        if note.get("trainingEligible") and note.get("qualityStatus") not in {"trusted", "accepted-manually"}:
            errors.append(f"note {note_index} is eligible but its qualityStatus is not trusted/accepted")
        if len(errors) >= 100:
            errors.append("validation stopped after 100 errors")
            break
    return errors


def windows_overlapping(start: float, end: float, windows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [window for window in windows if (
        float(window.get("sourceStart", 0)) < end
        and float(window.get("sourceEnd", 0)) > start
    )]


def clip_notes(notes: Iterable[dict[str, Any]], start: float, end: float) -> list[dict[str, Any]]:
    clipped: list[dict[str, Any]] = []
    for note in notes:
        note_start = float(note.get("time", 0))
        note_end = note_start + float(note.get("duration", 0))
        if note_start >= end or note_end <= start or not note.get("trainingEligible"):
            continue
        local_start = max(0.0, note_start - start)
        local_end = min(end, note_end) - start
        clipped.append({
            "midi": int(note["midi"]),
            "time": round(local_start, 6),
            "duration": round(max(0.01, local_end - local_start), 6),
            "velocity": round(float(note.get("velocity", 0.75)), 4),
            "instrument": str(note.get("instrument") or "acoustic_piano"),
            "continuedFromPreviousClip": note_start < start,
            "continuesIntoNextClip": note_end > end,
            "qualityWindowId": note.get("qualityWindowId"),
        })
    return sorted(clipped, key=lambda note: (note["time"], note["midi"]))


def build_song_clips(
    entry: SongEntry,
    package: dict[str, Any],
    split: str,
    clip_seconds: float,
    hop_seconds: float,
    minimum_notes: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    duration = float(package["timeline"]["sourceDurationSeconds"])
    windows = package["alignment"]["qualityWindows"]
    notes = package["notes"]
    clips: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    index = 0
    start = 0.0
    while start < duration - 0.01:
        end = min(duration, start + clip_seconds)
        overlap = windows_overlapping(start, end, windows)
        unsafe = [window for window in overlap if window.get("status") not in ALLOWED_WINDOW_STATUS]
        labels = clip_notes(notes, start, end)
        clip_id = f"{entry.song_id}-{index:05d}"
        if unsafe:
            rejected.append({
                "clipId": clip_id,
                "sourceStart": round(start, 6),
                "sourceEnd": round(end, 6),
                "reason": "overlaps-unapproved-window",
                "windowIds": [window.get("id") for window in unsafe],
            })
        elif len(labels) < minimum_notes:
            rejected.append({
                "clipId": clip_id,
                "sourceStart": round(start, 6),
                "sourceEnd": round(end, 6),
                "reason": "too-few-eligible-notes",
                "eligibleNotes": len(labels),
            })
        else:
            clips.append({
                "schema": "polymath-training-clip-v1",
                "clipId": clip_id,
                "songId": entry.song_id,
                "split": split,
                "sourceMedia": str(entry.source_media),
                "sourceStart": round(start, 6),
                "durationSeconds": round(end - start, 6),
                "sampleRate": 16000,
                "instrumentFocus": "piano",
                "notes": labels,
                "qualityWindowIds": [window.get("id") for window in overlap],
                "supervisionPackage": str(entry.supervision_package),
            })
        index += 1
        start += hop_seconds
    return clips, rejected


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def build_dataset(
    index_path: Path,
    output_directory: Path,
    clip_seconds: float = 5.0,
    hop_seconds: float = 5.0,
    minimum_notes: int = 1,
    seed: str = "polymath-piano-v001",
    train_share: float = 0.8,
    validation_share: float = 0.1,
) -> dict[str, Any]:
    if clip_seconds <= 0 or hop_seconds <= 0:
        raise DatasetError("clip_seconds and hop_seconds must be positive")
    if not 0 < train_share < 1 or not 0 <= validation_share < 1 or train_share + validation_share >= 1:
        raise DatasetError("split shares must leave non-zero train and test ranges")
    entries = load_index(index_path)
    output_directory.mkdir(parents=True, exist_ok=True)
    records: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    rejected: list[dict[str, Any]] = []
    songs: list[dict[str, Any]] = []
    errors: list[str] = []

    for entry in entries:
        package = read_json(entry.supervision_package)
        if not isinstance(package, dict):
            errors.append(f"{entry.song_id}: supervision package is not an object")
            continue
        package_errors = validate_package(entry, package)
        if package_errors:
            errors.extend(f"{entry.song_id}: {message}" for message in package_errors)
            continue
        split = entry.forced_split or deterministic_split(entry.song_id, seed, train_share, validation_share)
        if split not in records:
            errors.append(f"{entry.song_id}: split must be train, validation, or test")
            continue
        clips, song_rejected = build_song_clips(
            entry, package, split, clip_seconds, hop_seconds, minimum_notes,
        )
        records[split].extend(clips)
        rejected.extend({"songId": entry.song_id, **item} for item in song_rejected)
        songs.append({
            "songId": entry.song_id,
            "split": split,
            "sourceMedia": str(entry.source_media),
            "sourceSha256": sha256_file(entry.source_media),
            "supervisionPackage": str(entry.supervision_package),
            "supervisionSha256": sha256_file(entry.supervision_package),
            "acceptedClips": len(clips),
            "rejectedClips": len(song_rejected),
            "rightsNote": entry.rights_note,
        })

    if errors:
        raise DatasetError("Dataset validation failed:\n- " + "\n- ".join(errors))
    for split, split_records in records.items():
        write_jsonl(output_directory / f"{split}.jsonl", split_records)
    write_jsonl(output_directory / "rejected-clips.jsonl", rejected)
    summary = {
        "schema": "polymath-dataset-summary-v1",
        "sourceIndex": str(index_path.resolve()),
        "seed": seed,
        "clipSeconds": clip_seconds,
        "hopSeconds": hop_seconds,
        "minimumNotes": minimum_notes,
        "songs": songs,
        "counts": {split: len(split_records) for split, split_records in records.items()},
        "rejectedClips": len(rejected),
        "leakageRule": "Every songId belongs to exactly one split before clips are created.",
    }
    (output_directory / "dataset-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True, help="Training index JSON")
    parser.add_argument("--out", type=Path, required=True, help="Output dataset-manifest directory")
    parser.add_argument("--clip-seconds", type=float, default=5.0)
    parser.add_argument("--hop-seconds", type=float, default=5.0)
    parser.add_argument("--minimum-notes", type=int, default=1)
    parser.add_argument("--seed", default="polymath-piano-v001")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        summary = build_dataset(
            args.index.resolve(), args.out.resolve(), args.clip_seconds,
            args.hop_seconds, args.minimum_notes, args.seed,
        )
    except DatasetError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
