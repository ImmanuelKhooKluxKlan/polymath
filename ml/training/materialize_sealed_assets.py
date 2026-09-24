"""Materialize only the asset class authorized by sealed-exam custody.

Audio and MIDI labels are fetched in separate, irreversible stages.  Audio may
be fetched after receipt 01; labels may be fetched only after receipt 03 proves
that every prediction artifact was already hash-locked.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

from ml.training.rescore_song_timelines import sha256_file
from ml.training.sealed_exam_custody import (
    SealedExamError,
    receipt_path,
    utc_now,
    validate_plan,
    verified_receipt,
    verify_locked_artifacts,
    write_new_json,
)


class SealedAssetError(RuntimeError):
    """Raised when an unauthorized or non-reproducible fetch is attempted."""


ASSET_RECEIPTS = {
    "audio": (
        "01A-AUDIO-MATERIALIZED.json",
        "polymath-sealed-exam-audio-materialized-v1",
    ),
    "labels": (
        "03A-LABELS-MATERIALIZED.json",
        "polymath-sealed-exam-labels-materialized-v1",
    ),
}


def default_downloader(**kwargs: Any) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - environment dependency
        raise SealedAssetError("huggingface_hub is required to fetch sealed assets") from exc
    snapshot_download(**kwargs)


def materialize_assets(
    *,
    plan_path: Path,
    repo_root: Path,
    custody_dir: Path,
    dataset_root: Path,
    kind: str,
    downloader: Callable[..., None] = default_downloader,
) -> dict[str, Any]:
    if kind not in ASSET_RECEIPTS:
        raise ValueError("kind must be audio or labels")
    plan, summary = validate_plan(plan_path, repo_root, require_still_sealed=False)
    plan_hash = sha256_file(plan_path)
    frozen_dataset = Path(str(plan["selection"]["localDatasetRoot"])).resolve()
    if dataset_root.resolve() != frozen_dataset:
        raise SealedAssetError("Dataset root differs from the frozen plan")
    if kind == "audio":
        verified_receipt(custody_dir, "01-AUDIO-AUTHORIZATION.json", plan_hash)
        if (custody_dir / "02-PREDICTIONS-LOCKED.json").exists():
            raise SealedAssetError("Audio materialization cannot begin after predictions lock")
        field = "audioFilename"
        forbidden_field = "midiFilename"
        predecessor = custody_dir / "01-AUDIO-AUTHORIZATION.json"
    else:
        predictions = verified_receipt(
            custody_dir, "02-PREDICTIONS-LOCKED.json", plan_hash
        )
        verified_receipt(custody_dir, "03-LABEL-AUTHORIZATION.json", plan_hash)
        verify_locked_artifacts(predictions)
        field = "midiFilename"
        forbidden_field = ""
        predecessor = custody_dir / "03-LABEL-AUTHORIZATION.json"
    receipt_name, receipt_schema = ASSET_RECEIPTS[kind]
    receipt_out = custody_dir / receipt_name
    if receipt_out.exists():
        raise SealedAssetError(f"Sealed {kind} assets were already materialized")
    songs = summary["testSongs"]
    targets = [dataset_root / str(song[field]) for song in songs]
    if any(path.exists() for path in targets):
        raise SealedAssetError(f"Selected sealed {kind} files already exist")
    if forbidden_field:
        forbidden = [dataset_root / str(song[forbidden_field]) for song in songs]
        if any(path.exists() for path in forbidden):
            raise SealedAssetError("MIDI labels appeared before predictions were locked")
    patterns = [str(song[field]).replace("\\", "/") for song in songs]
    mirror = summary["manifest"].get("mirror") or {}
    repo_id = str(mirror.get("repoId") or "").strip()
    if not repo_id:
        raise SealedAssetError("Selection manifest has no frozen dataset mirror")
    dataset_root.mkdir(parents=True, exist_ok=True)
    downloader(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=patterns,
        local_dir=str(dataset_root),
    )
    missing = [path for path in targets if not path.is_file()]
    if missing:
        raise SealedAssetError(f"Authorized fetch left {len(missing)} selected files missing")
    if forbidden_field and any(
        (dataset_root / str(song[forbidden_field])).exists() for song in songs
    ):
        raise SealedAssetError("Audio-only fetch unexpectedly materialized MIDI labels")
    payload = {
        "schema": receipt_schema,
        "createdAtUtc": utc_now(),
        "candidateVersion": plan["candidateVersion"],
        "planSha256": plan_hash,
        "previousReceiptSha256": sha256_file(predecessor),
        "assetClass": "audio-only" if kind == "audio" else "midi-reference-labels",
        "files": [
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in targets
        ],
        "fileCount": len(targets),
        "labelsRead": False,
        "predictionsLockedBeforeMaterialization": kind == "labels",
        "sealedTestConsumed": False,
        "productionPromotionAllowed": False,
    }
    write_new_json(receipt_out, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--custody-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--kind", choices=("audio", "labels"), required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    try:
        result = materialize_assets(
            plan_path=args.plan.resolve(),
            repo_root=args.repo_root.resolve(),
            custody_dir=args.custody_dir.resolve(),
            dataset_root=args.dataset_root.resolve(),
            kind=args.kind,
        )
    except (SealedExamError, SealedAssetError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
