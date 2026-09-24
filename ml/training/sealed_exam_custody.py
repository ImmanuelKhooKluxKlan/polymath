"""Fail-closed custody receipts for the single-use Phase 94 sealed exam.

The workflow deliberately separates audio inference from label access:

1. ``begin-audio`` is allowed only after a committed blind transfer PASS.
2. ``lock-predictions`` hashes unlabeled manifests and decoded outputs.
3. ``authorize-labels`` permits the MIDI references to be materialized only
   after the predictions are immutable.
4. ``finalize`` consumes the test exactly once and records the frozen gate.

This module does not download or decode data.  It is the evidence guard around
those operations.  Its receipts are append-only and hash-chain to the frozen
plan.  A passing research exam never grants commercial or production rights.
"""

from __future__ import annotations

import argparse
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from ml.training.rescore_song_timelines import sha256_file


PLAN_SCHEMA = "polymath-sealed-direct-piano-exam-plan-v1"
BLIND_RESULT_SCHEMA = "polymath-opened-song-overlap-transfer-verdict-v1"
RECEIPT_SCHEMAS = {
    "01-AUDIO-AUTHORIZATION.json": "polymath-sealed-exam-audio-authorization-v1",
    "02-PREDICTIONS-LOCKED.json": "polymath-sealed-exam-predictions-locked-v1",
    "03-LABEL-AUTHORIZATION.json": "polymath-sealed-exam-label-authorization-v1",
    "04-FINAL-RESULT.json": "polymath-sealed-exam-final-result-v1",
}


class SealedExamError(RuntimeError):
    """Raised when the sealed-exam evidence chain is invalid."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SealedExamError(f"Cannot read JSON evidence: {path}") from exc
    if not isinstance(value, dict):
        raise SealedExamError(f"Evidence must be one JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise SealedExamError(f"{path}:{line_number} is not an object")
                records.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise SealedExamError(f"Cannot read JSONL evidence: {path}") from exc
    return records


def write_new_json(path: Path, value: dict[str, Any]) -> None:
    """Write one receipt without ever replacing an existing receipt."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise SealedExamError(f"Refusing to replace custody receipt: {path}")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    payload = (json.dumps(value, indent=2) + "\n").encode("utf-8")
    descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        # On Windows rename fails when the destination already exists, which
        # preserves the append-only receipt contract even under a race.
        temporary.rename(path)
    finally:
        temporary.unlink(missing_ok=True)


def required_string(value: dict[str, Any], key: str, context: str) -> str:
    output = str(value.get(key) or "").strip()
    if not output:
        raise SealedExamError(f"{context} has no {key}")
    return output


def resolve_artifact(path_value: str, repo_root: Path) -> Path:
    path = Path(path_value)
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def verify_artifact_table(plan: dict[str, Any], repo_root: Path) -> None:
    artifacts = plan.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise SealedExamError("Frozen plan has no artifact table")
    for name, record in artifacts.items():
        if not isinstance(record, dict):
            raise SealedExamError(f"Artifact {name} is not an object")
        path = resolve_artifact(required_string(record, "path", f"artifact {name}"), repo_root)
        expected = required_string(record, "sha256", f"artifact {name}").lower()
        if not path.is_file():
            raise SealedExamError(f"Frozen artifact is missing: {name}")
        if sha256_file(path).lower() != expected:
            raise SealedExamError(f"Frozen artifact hash changed: {name}")


def selection_summary(plan: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    selection = plan.get("selection") or {}
    path = resolve_artifact(
        required_string(selection, "manifestPath", "selection"), repo_root
    )
    expected_hash = required_string(selection, "manifestSha256", "selection").lower()
    if not path.is_file() or sha256_file(path).lower() != expected_hash:
        raise SealedExamError("Selection manifest is missing or changed")
    manifest = read_json(path)
    songs = manifest.get("songs")
    if not isinstance(songs, list):
        raise SealedExamError("Selection manifest has no songs array")
    test = [song for song in songs if isinstance(song, dict) and song.get("split") == "test"]
    expected_songs = int(selection.get("expectedTestSongs") or 0)
    if len(test) != expected_songs or expected_songs <= 0:
        raise SealedExamError("Sealed test song count differs from the frozen plan")
    if any(song.get("sealed") is not True for song in test):
        raise SealedExamError("Every selected test song must be marked sealed")
    components: dict[str, set[str]] = {}
    for split in ("train", "validation", "test"):
        components[split] = {
            str(person)
            for song in songs
            if isinstance(song, dict) and song.get("split") == split
            for person in (song.get("composerComponents") or [])
        }
    if (
        components["train"] & components["validation"]
        or components["train"] & components["test"]
        or components["validation"] & components["test"]
    ):
        raise SealedExamError("Individual people overlap across evidence splits")
    expected_people = int(selection.get("expectedTestPeople") or 0)
    if len(components["test"]) != expected_people or expected_people <= 0:
        raise SealedExamError("Sealed test person count differs from the frozen plan")
    return {
        "path": path,
        "manifest": manifest,
        "testSongs": test,
        "testPeople": len(components["test"]),
    }


def verify_still_sealed(plan: dict[str, Any], summary: dict[str, Any], repo_root: Path) -> None:
    selection = plan["selection"]
    dataset_root = resolve_artifact(
        required_string(selection, "localDatasetRoot", "selection"), repo_root
    )
    present_pairs = 0
    for song in summary["testSongs"]:
        audio = dataset_root / required_string(song, "audioFilename", "test song")
        midi = dataset_root / required_string(song, "midiFilename", "test song")
        if audio.is_file() or midi.is_file():
            present_pairs += 1
    if present_pairs:
        raise SealedExamError(
            f"Sealed source material is already present for {present_pairs} test songs"
        )
    prepared = resolve_artifact(
        required_string(selection, "preparedRoot", "selection"), repo_root
    )
    forbidden = [
        prepared / "prepared-test.jsonl",
        prepared / "audio" / "test",
    ]
    if any(path.is_file() or (path.is_dir() and any(path.iterdir())) for path in forbidden):
        raise SealedExamError("Prepared sealed-test material already exists")
    empty_manifest = selection.get("emptyTestManifestPath")
    if empty_manifest:
        path = resolve_artifact(str(empty_manifest), repo_root)
        if path.is_file() and any(line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines()):
            raise SealedExamError("The sealed test manifest already contains records")


def validate_plan(plan_path: Path, repo_root: Path, *, require_still_sealed: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    plan = read_json(plan_path)
    if plan.get("schema") != PLAN_SCHEMA:
        raise SealedExamError("Unsupported sealed-exam plan schema")
    if plan.get("status") != "frozen-before-blind-reveal-and-test-materialization":
        raise SealedExamError("Sealed-exam plan was not frozen at the required time")
    if plan.get("sealedTestOpened") is not False:
        raise SealedExamError("Frozen plan does not declare the test sealed")
    policy = plan.get("fixedPolicy") or {}
    if float(policy.get("radiusSeconds") or -1) != 0.6:
        raise SealedExamError("Unexpected frozen overlap radius")
    if float(policy.get("onsetMatchToleranceSeconds") or -1) != 0.1:
        raise SealedExamError("Unexpected frozen onset-pairing tolerance")
    acceptance = plan.get("acceptance") or {}
    if float(acceptance.get("microF1_100msMinimum") or -1) != 0.9:
        raise SealedExamError("The sealed 100 ms target must be frozen at 0.90")
    if float(acceptance.get("songBootstrapLower95_100msMinimum") or -1) != 0.9:
        raise SealedExamError("The sealed lower confidence bound must be frozen at 0.90")
    if acceptance.get("candidateMustBeatImmutableOriginal") is not True:
        raise SealedExamError("The candidate must beat the immutable baseline")
    verify_artifact_table(plan, repo_root)
    summary = selection_summary(plan, repo_root)
    if require_still_sealed:
        verify_still_sealed(plan, summary, repo_root)
    return plan, summary


def preflight(plan_path: Path, repo_root: Path) -> dict[str, Any]:
    plan, summary = validate_plan(plan_path, repo_root, require_still_sealed=True)
    return {
        "status": "PASS",
        "candidateVersion": plan["candidateVersion"],
        "planSha256": sha256_file(plan_path),
        "testSongs": len(summary["testSongs"]),
        "testPeople": summary["testPeople"],
        "sourceTestFilesPresent": 0,
        "preparedTestOpened": False,
        "labelsRead": False,
    }


def validate_blind_result(plan: dict[str, Any], result: dict[str, Any]) -> None:
    if result.get("schema") != BLIND_RESULT_SCHEMA:
        raise SealedExamError("Unsupported blind-transfer verdict schema")
    eligibility = plan.get("eligibility") or {}
    if result.get("candidateVersion") != eligibility.get("openedTransferCandidateVersion"):
        raise SealedExamError("Blind verdict belongs to another transfer candidate")
    if result.get("mappingCommitment") != eligibility.get("mappingCommitment"):
        raise SealedExamError("Blind verdict uses another mapping commitment")
    if result.get("status") != "PASS" or result.get("selectedRole") != "candidate":
        raise SealedExamError("Blind transfer did not select the candidate")
    if result.get("decision") != "eligible-for-single-use-sealed-test-freeze":
        raise SealedExamError("Blind verdict did not authorize a sealed research exam")
    checks = result.get("checks") or {}
    required = {
        "commitment_verified",
        "structural_safety_gate",
        "candidate_preferred",
        "no_blocker_heard",
    }
    if set(checks) != required or not all(checks.values()):
        raise SealedExamError("Blind verdict does not pass every required check")
    if result.get("sealedTestOpened") is not False:
        raise SealedExamError("Blind verdict says the sealed test was already opened")
    if result.get("productionPromotionAllowed") is not False:
        raise SealedExamError("Research verdict cannot authorize production")


def receipt_path(custody_dir: Path, name: str) -> Path:
    if name not in RECEIPT_SCHEMAS:
        raise SealedExamError(f"Unknown receipt: {name}")
    return custody_dir / name


def verified_receipt(custody_dir: Path, name: str, plan_hash: str) -> dict[str, Any]:
    path = receipt_path(custody_dir, name)
    value = read_json(path)
    if value.get("schema") != RECEIPT_SCHEMAS[name]:
        raise SealedExamError(f"Unexpected receipt schema: {name}")
    if value.get("planSha256") != plan_hash:
        raise SealedExamError(f"Receipt belongs to another frozen plan: {name}")
    return value


def assert_absent(custody_dir: Path, names: Iterable[str]) -> None:
    present = [name for name in names if receipt_path(custody_dir, name).exists()]
    if present:
        raise SealedExamError(f"Custody stage has already advanced: {', '.join(present)}")


def verify_locked_artifacts(predictions: dict[str, Any]) -> None:
    artifacts = predictions.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise SealedExamError("Prediction receipt has no artifact table")
    for name, record in artifacts.items():
        if not isinstance(record, dict):
            raise SealedExamError(f"Locked prediction artifact is malformed: {name}")
        path = Path(required_string(record, "path", f"locked artifact {name}"))
        expected = required_string(record, "sha256", f"locked artifact {name}")
        if not path.is_file() or sha256_file(path) != expected:
            raise SealedExamError(f"Locked prediction artifact changed: {name}")


def begin_audio(
    plan_path: Path,
    blind_result_path: Path,
    repo_root: Path,
    custody_dir: Path,
) -> dict[str, Any]:
    plan, summary = validate_plan(plan_path, repo_root, require_still_sealed=True)
    validate_blind_result(plan, read_json(blind_result_path))
    assert_absent(custody_dir, RECEIPT_SCHEMAS)
    payload = {
        "schema": RECEIPT_SCHEMAS["01-AUDIO-AUTHORIZATION.json"],
        "createdAtUtc": utc_now(),
        "candidateVersion": plan["candidateVersion"],
        "planSha256": sha256_file(plan_path),
        "blindResultSha256": sha256_file(blind_result_path),
        "selectionManifestSha256": plan["selection"]["manifestSha256"],
        "authorizedTestSongs": len(summary["testSongs"]),
        "authorizedAssetClass": "audio-only",
        "predictionsLocked": False,
        "labelsAuthorized": False,
        "labelsRead": False,
        "sealedTestConsumed": False,
        "productionPromotionAllowed": False,
    }
    write_new_json(receipt_path(custody_dir, "01-AUDIO-AUTHORIZATION.json"), payload)
    return payload


def validate_unlabeled_manifest(path: Path, expected_songs: int) -> tuple[list[dict[str, Any]], set[str]]:
    records = read_jsonl(path)
    if not records:
        raise SealedExamError(f"Unlabeled manifest is empty: {path}")
    song_ids: set[str] = set()
    for record in records:
        if record.get("split") != "test":
            raise SealedExamError("Prediction manifests must contain only the sealed test split")
        if record.get("labelsPresent") is not False:
            raise SealedExamError("Prediction manifest does not explicitly disable labels")
        if record.get("notes") not in (None, []):
            raise SealedExamError("Prediction manifest contains reference notes")
        song_ids.add(required_string(record, "songId", "prediction record"))
    if len(song_ids) != expected_songs:
        raise SealedExamError("Prediction manifest song count differs from the frozen plan")
    return records, song_ids


def validate_unlabeled_evaluation(
    path: Path,
    records: list[dict[str, Any]],
    expected_checkpoint: str,
    expected_checkpoint_sha256: str,
) -> dict[str, Any]:
    value = read_json(path)
    if value.get("schema") != "polymath-checkpoint-evaluation-v1":
        raise SealedExamError("Unexpected checkpoint evaluation schema")
    checkpoint = str(value.get("checkpoint") or "")
    if expected_checkpoint not in checkpoint:
        raise SealedExamError("Decoded evaluation does not use the frozen checkpoint")
    if str(value.get("checkpointSha256") or "").lower() != expected_checkpoint_sha256.lower():
        raise SealedExamError("Decoded evaluation checkpoint hash differs from the frozen checkpoint")
    decoded = (value.get("metrics") or {}).get("decodedClips")
    if not isinstance(decoded, list) or len(decoded) != len(records):
        raise SealedExamError("Decoded clip count differs from its unlabeled manifest")
    reference_counts = [
        float(((value.get("metrics") or {}).get(tolerance) or {}).get("referenceNotes") or 0)
        for tolerance in ("50ms", "100ms", "250ms")
    ]
    if any(reference_counts):
        raise SealedExamError("Decoded evaluation already contains reference-note metrics")
    return value


def lock_predictions(
    plan_path: Path,
    repo_root: Path,
    custody_dir: Path,
    primary_manifest_path: Path,
    overlap_manifest_path: Path,
    primary_evaluation_path: Path,
    overlap_evaluation_path: Path,
    candidate_output_path: Path,
) -> dict[str, Any]:
    plan, summary = validate_plan(plan_path, repo_root, require_still_sealed=False)
    plan_hash = sha256_file(plan_path)
    audio_receipt = verified_receipt(
        custody_dir, "01-AUDIO-AUTHORIZATION.json", plan_hash
    )
    assert_absent(custody_dir, list(RECEIPT_SCHEMAS)[1:])
    expected_songs = len(summary["testSongs"])
    primary_records, primary_songs = validate_unlabeled_manifest(
        primary_manifest_path, expected_songs
    )
    overlap_records, overlap_songs = validate_unlabeled_manifest(
        overlap_manifest_path, expected_songs
    )
    if primary_songs != overlap_songs:
        raise SealedExamError("Primary and overlap manifests contain different songs")
    checkpoint_fragment = required_string(
        plan.get("checkpoint") or {}, "runtimeIdentity", "checkpoint"
    )
    checkpoint_sha256 = required_string(
        plan.get("checkpoint") or {}, "sha256", "checkpoint"
    )
    validate_unlabeled_evaluation(
        primary_evaluation_path, primary_records, checkpoint_fragment, checkpoint_sha256
    )
    validate_unlabeled_evaluation(
        overlap_evaluation_path, overlap_records, checkpoint_fragment, checkpoint_sha256
    )
    candidate = read_json(candidate_output_path)
    if candidate.get("schema") != "polymath-fixed-overlap-inference-v1":
        raise SealedExamError("Unexpected fixed-policy candidate schema")
    if candidate.get("labelsPresent") is not False or candidate.get("metricsIncluded") is not False:
        raise SealedExamError("Candidate output must be locked before labels and metrics exist")
    if int(candidate.get("songCount") or 0) != expected_songs:
        raise SealedExamError("Candidate output song count differs from the frozen plan")
    policy = candidate.get("policy") or {}
    if float(policy.get("radiusSeconds") or -1) != float(plan["fixedPolicy"]["radiusSeconds"]):
        raise SealedExamError("Candidate output changed the frozen overlap radius")
    if float(policy.get("onsetMatchToleranceSeconds") or -1) != float(
        plan["fixedPolicy"]["onsetMatchToleranceSeconds"]
    ):
        raise SealedExamError("Candidate output changed the onset-pairing tolerance")
    artifacts = {
        "primaryManifest": primary_manifest_path,
        "overlapManifest": overlap_manifest_path,
        "primaryEvaluation": primary_evaluation_path,
        "overlapEvaluation": overlap_evaluation_path,
        "candidateOutput": candidate_output_path,
    }
    payload = {
        "schema": RECEIPT_SCHEMAS["02-PREDICTIONS-LOCKED.json"],
        "createdAtUtc": utc_now(),
        "candidateVersion": plan["candidateVersion"],
        "planSha256": plan_hash,
        "previousReceiptSha256": sha256_file(
            receipt_path(custody_dir, "01-AUDIO-AUTHORIZATION.json")
        ),
        "blindResultSha256": audio_receipt["blindResultSha256"],
        "songCount": expected_songs,
        "primaryClips": len(primary_records),
        "overlapClips": len(overlap_records),
        "artifacts": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in artifacts.items()
        },
        "predictionsLocked": True,
        "labelsAuthorized": False,
        "labelsRead": False,
        "accuracyMetricsIncluded": False,
        "sealedTestConsumed": False,
        "productionPromotionAllowed": False,
    }
    write_new_json(receipt_path(custody_dir, "02-PREDICTIONS-LOCKED.json"), payload)
    return payload


def authorize_labels(plan_path: Path, repo_root: Path, custody_dir: Path) -> dict[str, Any]:
    plan, _ = validate_plan(plan_path, repo_root, require_still_sealed=False)
    plan_hash = sha256_file(plan_path)
    verified_receipt(custody_dir, "01-AUDIO-AUTHORIZATION.json", plan_hash)
    predictions = verified_receipt(custody_dir, "02-PREDICTIONS-LOCKED.json", plan_hash)
    assert_absent(custody_dir, list(RECEIPT_SCHEMAS)[2:])
    if predictions.get("predictionsLocked") is not True:
        raise SealedExamError("Predictions are not frozen")
    verify_locked_artifacts(predictions)
    payload = {
        "schema": RECEIPT_SCHEMAS["03-LABEL-AUTHORIZATION.json"],
        "createdAtUtc": utc_now(),
        "candidateVersion": plan["candidateVersion"],
        "planSha256": plan_hash,
        "previousReceiptSha256": sha256_file(
            receipt_path(custody_dir, "02-PREDICTIONS-LOCKED.json")
        ),
        "predictionArtifactHashes": {
            key: value["sha256"] for key, value in predictions["artifacts"].items()
        },
        "authorizedAssetClass": "midi-reference-labels",
        "predictionsLockedBeforeAuthorization": True,
        "labelsAuthorized": True,
        "sealedTestConsumed": False,
        "productionPromotionAllowed": False,
    }
    write_new_json(receipt_path(custody_dir, "03-LABEL-AUTHORIZATION.json"), payload)
    return payload


def finalize_exam(
    plan_path: Path,
    repo_root: Path,
    custody_dir: Path,
    baseline_score_path: Path,
    candidate_score_path: Path,
    gate_result_path: Path,
    reference_receipt_path: Path,
) -> dict[str, Any]:
    plan, summary = validate_plan(plan_path, repo_root, require_still_sealed=False)
    plan_hash = sha256_file(plan_path)
    verified_receipt(custody_dir, "01-AUDIO-AUTHORIZATION.json", plan_hash)
    predictions = verified_receipt(custody_dir, "02-PREDICTIONS-LOCKED.json", plan_hash)
    verify_locked_artifacts(predictions)
    verified_receipt(custody_dir, "03-LABEL-AUTHORIZATION.json", plan_hash)
    assert_absent(custody_dir, ["04-FINAL-RESULT.json"])
    labels_materialized_path = custody_dir / "03A-LABELS-MATERIALIZED.json"
    labels_materialized = read_json(labels_materialized_path)
    if labels_materialized.get("schema") != "polymath-sealed-exam-labels-materialized-v1":
        raise SealedExamError("Label materialization receipt is missing or invalid")
    if labels_materialized.get("planSha256") != plan_hash:
        raise SealedExamError("Label materialization belongs to another plan")
    reference_receipt = read_json(reference_receipt_path)
    if reference_receipt.get("schema") != "polymath-sealed-reference-manifest-v1":
        raise SealedExamError("Reference receipt uses an unexpected schema")
    if reference_receipt.get("planSha256") != plan_hash:
        raise SealedExamError("Reference receipt belongs to another plan")
    if reference_receipt.get("identityPreserved") is not True:
        raise SealedExamError("Reference preparation did not preserve locked clip identity")
    if reference_receipt.get("predictionsLockedBeforeLabels") is not True:
        raise SealedExamError("Reference labels were not opened after prediction lock")
    if reference_receipt.get("lockedPrimaryManifestSha256") != predictions["artifacts"]["primaryManifest"]["sha256"]:
        raise SealedExamError("Reference labels do not belong to the locked primary manifest")
    if reference_receipt.get("labelMaterializationSha256") != sha256_file(labels_materialized_path):
        raise SealedExamError("Reference receipt is not bound to the label materialization")
    reference_manifest_path = Path(
        required_string(reference_receipt, "outputManifest", "reference receipt")
    )
    reference_manifest_hash = required_string(
        reference_receipt, "outputManifestSha256", "reference receipt"
    )
    if not reference_manifest_path.is_file() or sha256_file(reference_manifest_path) != reference_manifest_hash:
        raise SealedExamError("Opened reference manifest changed after preparation")
    baseline = read_json(baseline_score_path)
    candidate = read_json(candidate_score_path)
    gate = read_json(gate_result_path)
    expected_schema = "polymath-stitched-song-timeline-score-v1"
    if baseline.get("schema") != expected_schema or candidate.get("schema") != expected_schema:
        raise SealedExamError("Final scores use an unexpected schema")
    expected_songs = len(summary["testSongs"])
    if len(baseline.get("songIds") or []) != expected_songs:
        raise SealedExamError("Baseline final score does not contain every sealed song")
    if baseline.get("songIds") != candidate.get("songIds"):
        raise SealedExamError("Baseline and candidate final scores use different songs")
    if gate.get("schema") != "polymath-phase94-full-validation-gate-result-v1":
        raise SealedExamError("Final gate result uses an unexpected schema")
    if gate.get("candidateVersion") != plan.get("candidateVersion"):
        raise SealedExamError("Final gate result belongs to another candidate")
    if gate.get("baselineSha256") != sha256_file(baseline_score_path):
        raise SealedExamError("Final gate is not bound to the supplied baseline score")
    if gate.get("candidateSha256") != sha256_file(candidate_score_path):
        raise SealedExamError("Final gate is not bound to the supplied candidate score")
    locked_artifacts = predictions["artifacts"]
    if baseline.get("sourceEvaluationSha256") != locked_artifacts["primaryEvaluation"]["sha256"]:
        raise SealedExamError("Baseline score is not bound to the locked primary prediction")
    if candidate.get("sourcePredictionSha256") != locked_artifacts["candidateOutput"]["sha256"]:
        raise SealedExamError("Candidate score is not bound to the locked overlap prediction")
    reference_hash = candidate.get("validationManifestSha256")
    if not reference_hash or baseline.get("validationManifestSha256") != reference_hash:
        raise SealedExamError("Baseline and candidate were not scored against the same labels")
    if reference_hash != reference_manifest_hash:
        raise SealedExamError("Scores are not bound to the authorized reference manifest")
    gate_artifact_name = required_string(
        plan.get("acceptance") or {}, "fullGateArtifactName", "acceptance"
    )
    frozen_gate = (plan.get("artifacts") or {}).get(gate_artifact_name) or {}
    if gate.get("gateSha256") != required_string(
        frozen_gate, "sha256", f"artifact {gate_artifact_name}"
    ):
        raise SealedExamError("Final result was not evaluated with the frozen safety gate")
    score = float(candidate["metrics"]["100ms"]["microF1"])
    baseline_score = float(baseline["metrics"]["100ms"]["microF1"])
    lower95 = float(candidate["songClusterBootstrap95"]["100ms"]["lower95"])
    acceptance = plan["acceptance"]
    checks = {
        "all_eight_songs": len(candidate["songIds"]) == expected_songs,
        "candidate_beats_immutable_original": score > baseline_score,
        "micro_f1_100ms": score >= float(acceptance["microF1_100msMinimum"]),
        "lower95_100ms": lower95 >= float(
            acceptance["songBootstrapLower95_100msMinimum"]
        ),
        "complete_safety_gate": bool(gate.get("researchGatePassed")),
        "gate_certification": bool((gate.get("certification") or {}).get("passed")),
    }
    passed = all(checks.values())
    payload = {
        "schema": RECEIPT_SCHEMAS["04-FINAL-RESULT.json"],
        "createdAtUtc": utc_now(),
        "candidateVersion": plan["candidateVersion"],
        "planSha256": plan_hash,
        "previousReceiptSha256": sha256_file(
            receipt_path(custody_dir, "03-LABEL-AUTHORIZATION.json")
        ),
        "predictionReceiptSha256": sha256_file(
            receipt_path(custody_dir, "02-PREDICTIONS-LOCKED.json")
        ),
        "lockedPredictionHashes": {
            key: value["sha256"] for key, value in predictions["artifacts"].items()
        },
        "scoreArtifacts": {
            "baseline": sha256_file(baseline_score_path),
            "candidate": sha256_file(candidate_score_path),
            "gate": sha256_file(gate_result_path),
            "referenceReceipt": sha256_file(reference_receipt_path),
            "referenceManifest": reference_manifest_hash,
        },
        "metrics": {"microF1_100ms": score, "songBootstrapLower95_100ms": lower95},
        "checks": checks,
        "failedChecks": [name for name, value in checks.items() if not value],
        "status": "PASS" if passed else "FAIL",
        "sealedTestConsumed": True,
        "testMayNotBeReusedForSelection": True,
        "accuracyClaimEligible": passed,
        "commercialRightsCleared": False,
        "productionPromotionAllowed": False,
    }
    write_new_json(receipt_path(custody_dir, "04-FINAL-RESULT.json"), payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    subparsers = parser.add_subparsers(dest="command", required=True)
    common: dict[str, argparse.ArgumentParser] = {}
    for name in ("preflight", "begin-audio", "lock-predictions", "authorize-labels", "finalize"):
        current = subparsers.add_parser(name)
        current.add_argument("--plan", type=Path, required=True)
        if name != "preflight":
            current.add_argument("--custody-dir", type=Path, required=True)
        common[name] = current
    common["begin-audio"].add_argument("--blind-result", type=Path, required=True)
    for flag in (
        "primary-manifest",
        "overlap-manifest",
        "primary-evaluation",
        "overlap-evaluation",
        "candidate-output",
    ):
        common["lock-predictions"].add_argument(f"--{flag}", type=Path, required=True)
    for flag in ("baseline-score", "candidate-score", "gate-result"):
        common["finalize"].add_argument(f"--{flag}", type=Path, required=True)
    common["finalize"].add_argument("--reference-receipt", type=Path, required=True)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    plan_path = args.plan.resolve()
    if args.command == "preflight":
        result = preflight(plan_path, repo_root)
    elif args.command == "begin-audio":
        result = begin_audio(
            plan_path, args.blind_result.resolve(), repo_root, args.custody_dir.resolve()
        )
    elif args.command == "lock-predictions":
        result = lock_predictions(
            plan_path,
            repo_root,
            args.custody_dir.resolve(),
            args.primary_manifest.resolve(),
            args.overlap_manifest.resolve(),
            args.primary_evaluation.resolve(),
            args.overlap_evaluation.resolve(),
            args.candidate_output.resolve(),
        )
    elif args.command == "authorize-labels":
        result = authorize_labels(plan_path, repo_root, args.custody_dir.resolve())
    else:
        result = finalize_exam(
            plan_path,
            repo_root,
            args.custody_dir.resolve(),
            args.baseline_score.resolve(),
            args.candidate_score.resolve(),
            args.gate_result.resolve(),
            args.reference_receipt.resolve(),
        )
    print(json.dumps(result, indent=2))
    return 0 if result.get("status", "PASS") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
