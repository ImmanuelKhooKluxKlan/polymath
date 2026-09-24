"""Prepare label-free primary and overlap clips for the sealed piano exam.

This command is valid only after the custody chain has written its audio-only
authorization.  It reads selected audio, refuses to proceed if any selected
MIDI label is present, and writes manifests that explicitly contain no labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from ml.training.prepare_audio_clips import render_clip, resolve_ffmpeg
from ml.training.prepare_unlabeled_inference import window_starts
from ml.training.rescore_song_timelines import sha256_file
from ml.training.sealed_exam_custody import (
    SealedExamError,
    read_json,
    validate_plan,
    verified_receipt,
)


class SealedAudioPreparationError(RuntimeError):
    """Raised when sealed audio cannot be prepared without labels."""


def write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=False)
    path.write_text(
        "".join(json.dumps(value, separators=(",", ":")) + "\n" for value in values),
        encoding="utf-8",
    )


def build_records(
    songs: list[dict[str, Any]],
    dataset_root: Path,
    pass_name: str,
    offset_seconds: float,
    window_seconds: float,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for song in songs:
        source = dataset_root / str(song["audioFilename"])
        source_hash = sha256_file(source)
        duration = float(song["durationSeconds"])
        for index, start in enumerate(window_starts(duration, window_seconds, offset_seconds)):
            clip_duration = min(window_seconds, duration - start)
            records.append({
                "schema": "polymath-unlabeled-inference-clip-v1",
                "clipId": f"{song['songId']}-{pass_name}-{index:05d}",
                "songId": str(song["songId"]),
                "split": "test",
                "sourceMedia": str(source.resolve()),
                "sourceAudioSha256": source_hash,
                "sourceStart": round(start, 6),
                "durationSeconds": round(clip_duration, 6),
                "sampleRate": 16000,
                "instrumentFocus": "acoustic_piano",
                "notes": [],
                "labelsPresent": False,
                "targetState": "sealed-unlabeled-inference-only",
                "inferencePass": {
                    "name": pass_name,
                    "offsetSeconds": offset_seconds,
                    "windowSeconds": window_seconds,
                },
            })
    return records


def prepare_sealed_audio_passes(
    *,
    plan_path: Path,
    repo_root: Path,
    custody_dir: Path,
    dataset_root: Path,
    output_root: Path,
    remote_root: PurePosixPath,
    ffmpeg: Path,
    renderer: Callable[[Path, dict[str, Any], Path], None] = render_clip,
) -> dict[str, Any]:
    plan, summary = validate_plan(plan_path, repo_root, require_still_sealed=False)
    plan_hash = sha256_file(plan_path)
    authorization = verified_receipt(
        custody_dir, "01-AUDIO-AUTHORIZATION.json", plan_hash
    )
    if authorization.get("authorizedAssetClass") != "audio-only":
        raise SealedAudioPreparationError("Custody receipt does not authorize audio")
    if (custody_dir / "03-LABEL-AUTHORIZATION.json").exists():
        raise SealedAudioPreparationError("Audio inference must finish before label authorization")
    materialization_path = custody_dir / "01A-AUDIO-MATERIALIZED.json"
    materialization = read_json(materialization_path)
    if materialization.get("schema") != "polymath-sealed-exam-audio-materialized-v1":
        raise SealedAudioPreparationError("Audio materialization receipt is missing or invalid")
    if materialization.get("planSha256") != plan_hash:
        raise SealedAudioPreparationError("Audio materialization belongs to another plan")
    materialized_files = {
        str(Path(str(record.get("path") or "")).resolve()): str(record.get("sha256") or "")
        for record in (materialization.get("files") or [])
        if isinstance(record, dict)
    }
    if output_root.exists():
        raise SealedAudioPreparationError(f"Refusing to overwrite sealed audio pack: {output_root}")
    frozen_dataset = Path(str(plan["selection"]["localDatasetRoot"])).resolve()
    if dataset_root.resolve() != frozen_dataset:
        raise SealedAudioPreparationError("Dataset root differs from the frozen plan")
    songs = summary["testSongs"]
    missing_audio = [song["songId"] for song in songs if not (dataset_root / song["audioFilename"]).is_file()]
    if missing_audio:
        raise SealedAudioPreparationError(
            f"Selected audio is not materialized for {len(missing_audio)} songs"
        )
    for song in songs:
        source = (dataset_root / song["audioFilename"]).resolve()
        if materialized_files.get(str(source)) != sha256_file(source):
            raise SealedAudioPreparationError("Selected audio differs from its custody receipt")
    present_labels = [song["songId"] for song in songs if (dataset_root / song["midiFilename"]).is_file()]
    if present_labels:
        raise SealedAudioPreparationError(
            f"Refusing audio inference because {len(present_labels)} MIDI labels are present"
        )
    policy = plan["fixedPolicy"]
    window_seconds = float(policy.get("windowSeconds") or 5.0)
    overlap_offset = float(policy.get("overlapOffsetSeconds") or 2.5)
    output_root.mkdir(parents=True)
    pass_receipts: dict[str, Any] = {}
    try:
        for pass_name, offset in (("primary", 0.0), ("overlap", overlap_offset)):
            records = build_records(songs, dataset_root, pass_name, offset, window_seconds)
            audio_root = output_root / pass_name / "audio"
            audio_root.mkdir(parents=True, exist_ok=False)
            for record in records:
                destination = audio_root / f"{record['clipId']}.wav"
                renderer(ffmpeg, record, destination)
                record["localAudioSource"] = str(destination.resolve())
                record["audioClip"] = str(
                    remote_root / pass_name / "audio" / destination.name
                )
            manifest = output_root / pass_name / "manifest.jsonl"
            manifest.write_text(
                "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records),
                encoding="utf-8",
            )
            pass_receipts[pass_name] = {
                "clips": len(records),
                "offsetSeconds": offset,
                "manifest": str(manifest.resolve()),
                "manifestSha256": sha256_file(manifest),
                "audioBytes": sum(path.stat().st_size for path in audio_root.glob("*.wav")),
            }
    except Exception:
        # Leave partial evidence in place; never delete or silently retry a
        # partially opened sealed run.  The operator must audit the incident.
        raise
    receipt = {
        "schema": "polymath-sealed-audio-inference-pack-v1",
        "createdAtUtc": authorization["createdAtUtc"],
        "planSha256": plan_hash,
        "audioAuthorizationSha256": sha256_file(
            custody_dir / "01-AUDIO-AUTHORIZATION.json"
        ),
        "audioMaterializationSha256": sha256_file(materialization_path),
        "testSongs": len(songs),
        "labelsPresent": False,
        "labelsRead": False,
        "windowSeconds": window_seconds,
        "overlapOffsetSeconds": overlap_offset,
        "passes": pass_receipts,
        "sealedTestConsumed": False,
        "productionPromotionAllowed": False,
    }
    receipt_path = output_root / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--custody-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--ffmpeg", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    try:
        result = prepare_sealed_audio_passes(
            plan_path=args.plan.resolve(),
            repo_root=args.repo_root.resolve(),
            custody_dir=args.custody_dir.resolve(),
            dataset_root=args.dataset_root.resolve(),
            output_root=args.out_root.resolve(),
            remote_root=PurePosixPath(args.remote_root),
            ffmpeg=resolve_ffmpeg(args.ffmpeg),
        )
    except (SealedExamError, SealedAudioPreparationError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
