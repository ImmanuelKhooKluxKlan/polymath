from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml.training.materialize_sealed_assets import materialize_assets
from ml.training.rescore_song_timelines import sha256_file
from ml.training.sealed_exam_custody import (
    SealedExamError,
    authorize_labels,
    begin_audio,
    finalize_exam,
    lock_predictions,
    preflight,
)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def write_jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(value) + "\n" for value in values), encoding="utf-8")


def fixture(tmp_path: Path) -> dict[str, Path]:
    dataset = tmp_path / "source"
    prepared = tmp_path / "prepared"
    dataset.mkdir()
    prepared.mkdir()
    empty_test = tmp_path / "test.jsonl"
    empty_test.write_text("", encoding="utf-8")
    songs = [
        {
            "songId": "train-0",
            "split": "train",
            "composerComponents": ["train-person"],
            "sealed": False,
            "audioFilename": "train.wav",
            "midiFilename": "train.midi",
        },
        {
            "songId": "validation-0",
            "split": "validation",
            "composerComponents": ["validation-person"],
            "sealed": False,
            "audioFilename": "validation.wav",
            "midiFilename": "validation.midi",
        },
    ]
    for index in range(8):
        songs.append({
            "songId": f"test-{index}",
            "split": "test",
            "composerComponents": [f"test-person-{index % 4}"],
            "sealed": True,
            "durationSeconds": 10.0,
            "audioFilename": f"test-{index}.wav",
            "midiFilename": f"test-{index}.midi",
        })
    selection = tmp_path / "selection.json"
    write_json(selection, {
        "schema": "selection",
        "mirror": {"repoId": "example/maestro"},
        "songs": songs,
    })
    frozen_artifact = tmp_path / "policy.py"
    frozen_artifact.write_text("frozen-policy\n", encoding="utf-8")
    gate_config = tmp_path / "sealed-gate.json"
    gate_config.write_text('{"threshold":0.9}\n', encoding="utf-8")
    plan = tmp_path / "plan.json"
    write_json(plan, {
        "schema": "polymath-sealed-direct-piano-exam-plan-v1",
        "candidateVersion": "phase94-overlap-release-preserving-v003",
        "status": "frozen-before-blind-reveal-and-test-materialization",
        "selection": {
            "manifestPath": str(selection),
            "manifestSha256": sha256_file(selection),
            "expectedTestSongs": 8,
            "expectedTestPeople": 4,
            "localDatasetRoot": str(dataset),
            "preparedRoot": str(prepared),
            "emptyTestManifestPath": str(empty_test),
        },
        "artifacts": {
            "fixedPolicyCode": {
                "path": str(frozen_artifact),
                "sha256": sha256_file(frozen_artifact),
            },
            "fullGate": {
                "path": str(gate_config),
                "sha256": sha256_file(gate_config),
            },
        },
        "checkpoint": {
            "runtimeIdentity": "/models/original/model.safetensors",
            "sha256": "frozen-checkpoint-sha256",
        },
        "fixedPolicy": {
            "radiusSeconds": 0.6,
            "onsetMatchToleranceSeconds": 0.1,
        },
        "eligibility": {
            "openedTransferCandidateVersion": "opened-kiss-me-v005",
            "mappingCommitment": "frozen-commitment",
        },
        "acceptance": {
            "microF1_100msMinimum": 0.9,
            "songBootstrapLower95_100msMinimum": 0.9,
            "candidateMustBeatImmutableOriginal": True,
            "fullGateArtifactName": "fullGate",
        },
        "sealedTestOpened": False,
    })
    blind = tmp_path / "blind.json"
    write_json(blind, {
        "schema": "polymath-opened-song-overlap-transfer-verdict-v1",
        "candidateVersion": "opened-kiss-me-v005",
        "mappingCommitment": "frozen-commitment",
        "status": "PASS",
        "selectedRole": "candidate",
        "decision": "eligible-for-single-use-sealed-test-freeze",
        "checks": {
            "commitment_verified": True,
            "structural_safety_gate": True,
            "candidate_preferred": True,
            "no_blocker_heard": True,
        },
        "sealedTestOpened": False,
        "productionPromotionAllowed": False,
    })
    return {
        "plan": plan,
        "blind": blind,
        "dataset": dataset,
        "prepared": prepared,
        "empty_test": empty_test,
        "custody": tmp_path / "custody",
        "repo": tmp_path,
    }


def inference_artifacts(paths: dict[str, Path], *, labeled: bool = False) -> dict[str, Path]:
    root = paths["repo"] / "inference"
    primary_records = []
    overlap_records = []
    for index in range(8):
        common = {
            "songId": f"test-{index}",
            "split": "test",
            "labelsPresent": False,
            "notes": [{"midi": 60}] if labeled and index == 0 else [],
            "durationSeconds": 5.0,
            "sourceMedia": f"test-{index}.wav",
            "sourceAudioSha256": f"audio-{index}",
            "audioClip": f"/remote/test-{index}.wav",
        }
        primary_records.append({**common, "clipId": f"p-{index}", "sourceStart": 0})
        overlap_records.append({**common, "clipId": f"o-{index}", "sourceStart": 2.5})
    primary_manifest = root / "primary.jsonl"
    overlap_manifest = root / "overlap.jsonl"
    write_jsonl(primary_manifest, primary_records)
    write_jsonl(overlap_manifest, overlap_records)

    def evaluation(records: list[dict]) -> dict:
        return {
            "schema": "polymath-checkpoint-evaluation-v1",
            "checkpoint": "/models/original/model.safetensors",
            "checkpointSha256": "frozen-checkpoint-sha256",
            "metrics": {
                "50ms": {"referenceNotes": 0},
                "100ms": {"referenceNotes": 0},
                "250ms": {"referenceNotes": 0},
                "decodedClips": [
                    {"clipId": row["clipId"], "songId": row["songId"], "notes": []}
                    for row in records
                ],
            },
        }

    primary_eval = root / "primary-evaluation.json"
    overlap_eval = root / "overlap-evaluation.json"
    write_json(primary_eval, evaluation(primary_records))
    write_json(overlap_eval, evaluation(overlap_records))
    candidate = root / "candidate.json"
    write_json(candidate, {
        "schema": "polymath-fixed-overlap-inference-v1",
        "songCount": 8,
        "labelsPresent": False,
        "metricsIncluded": False,
        "policy": {
            "radiusSeconds": 0.6,
            "onsetMatchToleranceSeconds": 0.1,
        },
    })
    return {
        "primary_manifest": primary_manifest,
        "overlap_manifest": overlap_manifest,
        "primary_eval": primary_eval,
        "overlap_eval": overlap_eval,
        "candidate": candidate,
    }


def begin_and_lock(paths: dict[str, Path]) -> dict[str, Path]:
    begin_audio(paths["plan"], paths["blind"], paths["repo"], paths["custody"])
    artifacts = inference_artifacts(paths)
    lock_predictions(
        paths["plan"],
        paths["repo"],
        paths["custody"],
        artifacts["primary_manifest"],
        artifacts["overlap_manifest"],
        artifacts["primary_eval"],
        artifacts["overlap_eval"],
        artifacts["candidate"],
    )
    return artifacts


def materialize_test_labels(paths: dict[str, Path]) -> None:
    def downloader(**kwargs) -> None:
        root = Path(kwargs["local_dir"])
        for relative in kwargs["allow_patterns"]:
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(("label:" + relative).encode("utf-8"))

    materialize_assets(
        plan_path=paths["plan"], repo_root=paths["repo"],
        custody_dir=paths["custody"], dataset_root=paths["dataset"],
        kind="labels", downloader=downloader,
    )


def scoring_artifacts(paths: dict[str, Path], *, score: float = 0.92, lower95: float = 0.91):
    root = paths["repo"] / "scores"
    song_ids = [f"test-{index}" for index in range(8)]
    reference_manifest = root / "reference.jsonl"
    reference_manifest.parent.mkdir(parents=True, exist_ok=True)
    reference_manifest.write_text('{"sealed":"reference"}\n', encoding="utf-8")
    reference_hash = sha256_file(reference_manifest)
    prediction_receipt = json.loads(
        (paths["custody"] / "02-PREDICTIONS-LOCKED.json").read_text()
    )
    reference_receipt = root / "reference-receipt.json"
    write_json(reference_receipt, {
        "schema": "polymath-sealed-reference-manifest-v1",
        "planSha256": sha256_file(paths["plan"]),
        "labelMaterializationSha256": sha256_file(
            paths["custody"] / "03A-LABELS-MATERIALIZED.json"
        ),
        "lockedPrimaryManifestSha256": prediction_receipt["artifacts"]["primaryManifest"]["sha256"],
        "outputManifest": str(reference_manifest.resolve()),
        "outputManifestSha256": reference_hash,
        "identityPreserved": True,
        "predictionsLockedBeforeLabels": True,
    })
    baseline = root / "baseline.json"
    candidate = root / "candidate.json"
    write_json(baseline, {
        "schema": "polymath-stitched-song-timeline-score-v1",
        "songIds": song_ids,
        "metrics": {"100ms": {"microF1": 0.89}},
        "songClusterBootstrap95": {"100ms": {"lower95": 0.87}},
        "sourceEvaluationSha256": sha256_file(paths["repo"] / "inference" / "primary-evaluation.json"),
        "validationManifestSha256": reference_hash,
    })
    write_json(candidate, {
        "schema": "polymath-stitched-song-timeline-score-v1",
        "songIds": song_ids,
        "metrics": {"100ms": {"microF1": score}},
        "songClusterBootstrap95": {"100ms": {"lower95": lower95}},
        "sourcePredictionSha256": sha256_file(paths["repo"] / "inference" / "candidate.json"),
        "validationManifestSha256": reference_hash,
    })
    gate = root / "gate.json"
    write_json(gate, {
        "schema": "polymath-phase94-full-validation-gate-result-v1",
        "candidateVersion": "phase94-overlap-release-preserving-v003",
        "baselineSha256": sha256_file(baseline),
        "candidateSha256": sha256_file(candidate),
        "researchGatePassed": True,
        "certification": {"passed": True},
        "gateSha256": sha256_file(paths["repo"] / "sealed-gate.json"),
    })
    return baseline, candidate, gate, reference_receipt


def test_preflight_proves_selected_source_and_labels_are_absent(tmp_path: Path) -> None:
    paths = fixture(tmp_path)
    result = preflight(paths["plan"], paths["repo"])
    assert result["status"] == "PASS"
    assert result["testSongs"] == 8
    assert result["testPeople"] == 4
    assert result["labelsRead"] is False


def test_preflight_fails_if_any_selected_source_file_exists(tmp_path: Path) -> None:
    paths = fixture(tmp_path)
    (paths["dataset"] / "test-0.wav").write_bytes(b"opened")
    with pytest.raises(SealedExamError, match="already present"):
        preflight(paths["plan"], paths["repo"])


def test_begin_requires_blind_candidate_win_and_is_single_use(tmp_path: Path) -> None:
    paths = fixture(tmp_path)
    blind = json.loads(paths["blind"].read_text())
    blind["status"] = "FAIL"
    blind["selectedRole"] = "baseline"
    write_json(paths["blind"], blind)
    with pytest.raises(SealedExamError, match="did not select"):
        begin_audio(paths["plan"], paths["blind"], paths["repo"], paths["custody"])
    blind["status"] = "PASS"
    blind["selectedRole"] = "candidate"
    write_json(paths["blind"], blind)
    begin_audio(paths["plan"], paths["blind"], paths["repo"], paths["custody"])
    with pytest.raises(SealedExamError, match="already advanced"):
        begin_audio(paths["plan"], paths["blind"], paths["repo"], paths["custody"])


def test_labels_cannot_be_authorized_before_predictions(tmp_path: Path) -> None:
    paths = fixture(tmp_path)
    begin_audio(paths["plan"], paths["blind"], paths["repo"], paths["custody"])
    with pytest.raises(SealedExamError):
        authorize_labels(paths["plan"], paths["repo"], paths["custody"])


def test_lock_rejects_reference_notes_in_prediction_manifest(tmp_path: Path) -> None:
    paths = fixture(tmp_path)
    begin_audio(paths["plan"], paths["blind"], paths["repo"], paths["custody"])
    artifacts = inference_artifacts(paths, labeled=True)
    with pytest.raises(SealedExamError, match="reference notes"):
        lock_predictions(
            paths["plan"], paths["repo"], paths["custody"],
            artifacts["primary_manifest"], artifacts["overlap_manifest"],
            artifacts["primary_eval"], artifacts["overlap_eval"], artifacts["candidate"],
        )


def test_prediction_mutation_blocks_label_authorization(tmp_path: Path) -> None:
    paths = fixture(tmp_path)
    artifacts = begin_and_lock(paths)
    artifacts["candidate"].write_text("{}\n", encoding="utf-8")
    with pytest.raises(SealedExamError, match="changed"):
        authorize_labels(paths["plan"], paths["repo"], paths["custody"])


def test_complete_chain_can_pass_research_but_never_authorizes_production(tmp_path: Path) -> None:
    paths = fixture(tmp_path)
    begin_and_lock(paths)
    authorize_labels(paths["plan"], paths["repo"], paths["custody"])
    materialize_test_labels(paths)
    baseline, candidate, gate, reference_receipt = scoring_artifacts(paths)
    result = finalize_exam(
        paths["plan"], paths["repo"], paths["custody"], baseline, candidate, gate,
        reference_receipt,
    )
    assert result["status"] == "PASS"
    assert result["sealedTestConsumed"] is True
    assert result["accuracyClaimEligible"] is True
    assert result["productionPromotionAllowed"] is False
    assert result["commercialRightsCleared"] is False
    with pytest.raises(SealedExamError, match="already advanced"):
        finalize_exam(
            paths["plan"], paths["repo"], paths["custody"], baseline, candidate, gate,
            reference_receipt,
        )


def test_lower_confidence_bound_failure_consumes_test_without_certifying(tmp_path: Path) -> None:
    paths = fixture(tmp_path)
    begin_and_lock(paths)
    authorize_labels(paths["plan"], paths["repo"], paths["custody"])
    materialize_test_labels(paths)
    baseline, candidate, gate, reference_receipt = scoring_artifacts(
        paths, score=0.92, lower95=0.899
    )
    result = finalize_exam(
        paths["plan"], paths["repo"], paths["custody"], baseline, candidate, gate,
        reference_receipt,
    )
    assert result["status"] == "FAIL"
    assert result["sealedTestConsumed"] is True
    assert result["accuracyClaimEligible"] is False
    assert "lower95_100ms" in result["failedChecks"]
