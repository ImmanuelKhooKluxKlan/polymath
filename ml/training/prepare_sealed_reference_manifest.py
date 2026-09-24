"""Open authorized MIDI labels without changing locked inference coordinates.

The output copies every clip identity from the hash-locked primary inference
manifest and adds only clipped reference notes.  It refuses to parse MIDI until
both label authorization and label-materialization receipts exist.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from ml.training.prepare_maestro_curriculum import read_midi_notes
from ml.training.prepare_overlap_validation import clip_song_notes
from ml.training.rescore_song_timelines import sha256_file
from ml.training.sealed_exam_custody import (
    SealedExamError,
    read_json,
    read_jsonl,
    utc_now,
    validate_plan,
    verified_receipt,
    verify_locked_artifacts,
    write_new_json,
)


class SealedReferenceError(RuntimeError):
    """Raised when authorized labels cannot be paired without leakage."""


IDENTITY_FIELDS = (
    "clipId",
    "songId",
    "split",
    "sourceMedia",
    "sourceAudioSha256",
    "sourceStart",
    "durationSeconds",
    "audioClip",
)


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def manifest_identity(records: list[dict[str, Any]]) -> str:
    identities = [
        {key: record.get(key) for key in IDENTITY_FIELDS}
        for record in records
    ]
    return canonical_hash(identities)


def write_jsonl_new(path: Path, records: list[dict[str, Any]]) -> None:
    if path.exists():
        raise SealedReferenceError(f"Refusing to overwrite sealed reference: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
    )


def prepare_reference_manifest(
    *,
    plan_path: Path,
    repo_root: Path,
    custody_dir: Path,
    primary_manifest_path: Path,
    output_manifest_path: Path,
    receipt_path: Path,
    midi_reader: Callable[[Path, float], list[dict[str, Any]]] = read_midi_notes,
) -> dict[str, Any]:
    if output_manifest_path.exists():
        raise SealedReferenceError(f"Refusing to overwrite sealed reference: {output_manifest_path}")
    if receipt_path.exists():
        raise SealedReferenceError(f"Refusing to overwrite label receipt: {receipt_path}")
    plan, summary = validate_plan(plan_path, repo_root, require_still_sealed=False)
    plan_hash = sha256_file(plan_path)
    predictions = verified_receipt(
        custody_dir, "02-PREDICTIONS-LOCKED.json", plan_hash
    )
    verified_receipt(custody_dir, "03-LABEL-AUTHORIZATION.json", plan_hash)
    verify_locked_artifacts(predictions)
    label_materialization_path = custody_dir / "03A-LABELS-MATERIALIZED.json"
    label_materialization = read_json(label_materialization_path)
    if label_materialization.get("schema") != "polymath-sealed-exam-labels-materialized-v1":
        raise SealedReferenceError("Label materialization receipt is missing or invalid")
    if label_materialization.get("planSha256") != plan_hash:
        raise SealedReferenceError("Label materialization belongs to another plan")
    locked_primary = predictions["artifacts"]["primaryManifest"]
    if sha256_file(primary_manifest_path) != locked_primary["sha256"]:
        raise SealedReferenceError("Primary manifest differs from the locked prediction manifest")
    primary_records = read_jsonl(primary_manifest_path)
    if any(record.get("labelsPresent") is not False for record in primary_records):
        raise SealedReferenceError("Locked primary manifest is not explicitly unlabeled")
    if any(record.get("notes") not in (None, []) for record in primary_records):
        raise SealedReferenceError("Locked primary manifest already contains notes")

    materialized = {
        str(Path(str(record.get("path") or "")).resolve()): str(record.get("sha256") or "")
        for record in (label_materialization.get("files") or [])
        if isinstance(record, dict)
    }
    notes_by_song: dict[str, list[dict[str, Any]]] = {}
    midi_hashes: dict[str, str] = {}
    for song in summary["testSongs"]:
        midi_path = (
            Path(str(plan["selection"]["localDatasetRoot"])) / str(song["midiFilename"])
        ).resolve()
        actual_hash = sha256_file(midi_path) if midi_path.is_file() else ""
        if not actual_hash or materialized.get(str(midi_path)) != actual_hash:
            raise SealedReferenceError("A selected MIDI label differs from its custody receipt")
        song_id = str(song["songId"])
        notes = midi_reader(midi_path, float(song["durationSeconds"]))
        if not notes:
            raise SealedReferenceError(f"MIDI label contains no notes for {song_id}")
        notes_by_song[song_id] = notes
        midi_hashes[song_id] = actual_hash

    labeled: list[dict[str, Any]] = []
    observed_songs: set[str] = set()
    for record in primary_records:
        song_id = str(record.get("songId") or "")
        if song_id not in notes_by_song:
            raise SealedReferenceError("Primary manifest contains a song outside the selection")
        start = float(record.get("sourceStart") or 0)
        end = start + float(record.get("durationSeconds") or 0)
        notes = clip_song_notes(notes_by_song[song_id], start, end)
        labeled.append({
            **record,
            "notes": notes,
            "labelsPresent": True,
            "targetState": "sealed-reference-notes" if notes else "reviewed-silence",
            "referenceProvenance": {
                "dataset": "MAESTRO v3.0.0",
                "midiSha256": midi_hashes[song_id],
                "alignment": "official simultaneous Disklavier audio/MIDI capture",
                "labelsOpenedAfterPredictionsLocked": True,
            },
        })
        observed_songs.add(song_id)
    if observed_songs != set(notes_by_song):
        raise SealedReferenceError("Primary manifest does not cover all eight selected songs")
    locked_identity = manifest_identity(primary_records)
    labeled_identity = manifest_identity(labeled)
    if labeled_identity != locked_identity:
        raise SealedReferenceError("Adding labels changed locked clip coordinates")
    write_jsonl_new(output_manifest_path, labeled)
    receipt = {
        "schema": "polymath-sealed-reference-manifest-v1",
        "createdAtUtc": utc_now(),
        "planSha256": plan_hash,
        "predictionReceiptSha256": sha256_file(
            custody_dir / "02-PREDICTIONS-LOCKED.json"
        ),
        "labelAuthorizationSha256": sha256_file(
            custody_dir / "03-LABEL-AUTHORIZATION.json"
        ),
        "labelMaterializationSha256": sha256_file(label_materialization_path),
        "lockedPrimaryManifestSha256": locked_primary["sha256"],
        "outputManifest": str(output_manifest_path.resolve()),
        "outputManifestSha256": sha256_file(output_manifest_path),
        "lockedIdentitySha256": locked_identity,
        "outputIdentitySha256": labeled_identity,
        "identityPreserved": True,
        "songs": len(observed_songs),
        "clips": len(labeled),
        "labelsRead": True,
        "predictionsLockedBeforeLabels": True,
        "sealedTestConsumed": False,
        "productionPromotionAllowed": False,
    }
    write_new_json(receipt_path, receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--custody-dir", type=Path, required=True)
    parser.add_argument("--primary-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    try:
        result = prepare_reference_manifest(
            plan_path=args.plan.resolve(),
            repo_root=args.repo_root.resolve(),
            custody_dir=args.custody_dir.resolve(),
            primary_manifest_path=args.primary_manifest.resolve(),
            output_manifest_path=args.out.resolve(),
            receipt_path=args.receipt.resolve(),
        )
    except (SealedExamError, SealedReferenceError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
