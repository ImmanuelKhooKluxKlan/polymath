from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ml.training.finalize_blind_listening_verdict import canonical_hash
from ml.training.finalize_overlap_transfer_verdict import finalize


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_evidence(tmp_path: Path) -> dict[str, Path]:
    pack = tmp_path / "pack"
    pack.mkdir()
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    baseline.write_text('{"notes":[]}\n', encoding="utf-8")
    candidate.write_text('{"notes":[1]}\n', encoding="utf-8")
    payload = {
        "salt": "fixed-test-salt",
        "mapping": {"A": "candidate", "B": "baseline"},
        "baselineSha256": digest(baseline),
        "candidateSha256": digest(candidate),
    }
    commitment = canonical_hash(payload)
    (pack / "MAPPING-COMMITMENT.sha256").write_text(commitment + "\n", encoding="utf-8")
    (pack / "SEALED-MAPPING.json").write_text(json.dumps({
        "schema": "polymath-real-video-blind-map-v1",
        "commitment": commitment,
        "commitmentPayload": payload,
        "inputs": {
            "baseline": {"path": str(baseline)},
            "candidate": {"path": str(candidate)},
        },
    }), encoding="utf-8")
    (pack / "AUDIO-RENDER.json").write_text('{"sampleRate":44100}\n', encoding="utf-8")
    (pack / "LISTEN.html").write_text("<html>blind test</html>\n", encoding="utf-8")
    (pack / "02-A.wav").write_bytes(b"candidate-audio")
    (pack / "03-B.wav").write_bytes(b"baseline-audio")
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps({
        "schema": "polymath-candidate-freeze-v1",
        "candidateVersion": "phase94-opened-v005",
    }), encoding="utf-8")
    pre = tmp_path / "pre.json"
    pre.write_text(json.dumps({
        "schema": "polymath-opened-song-pre-listening-evidence-v1",
        "candidateVersion": "phase94-opened-v005",
        "artifacts": {
            "freezeSha256": digest(freeze),
            "primaryPlayableSha256": digest(baseline),
            "candidatePlayableSha256": digest(candidate),
            "audioRenderManifestSha256": digest(pack / "AUDIO-RENDER.json"),
            "listenPageSha256": digest(pack / "LISTEN.html"),
            "sealedMappingSha256": digest(pack / "SEALED-MAPPING.json"),
        },
        "outputs": {
            "mappingCommitment": commitment,
            "versionA": {"sha256": digest(pack / "02-A.wav")},
            "versionB": {"sha256": digest(pack / "03-B.wav")},
        },
        "blindState": {"mappingInspected": False},
    }), encoding="utf-8")
    structural = tmp_path / "structural.json"
    structural_freeze = tmp_path / "structural-freeze.json"
    structural_freeze.write_text(json.dumps({
        "schema": "polymath-candidate-freeze-v1",
        "candidateVersion": "phase94-structural-v006",
    }), encoding="utf-8")
    structural.write_text(json.dumps({
        "schema": "polymath-unlabeled-transfer-safety-evidence-v1",
        "candidateVersion": "phase94-structural-v006",
        "artifacts": {
            "freezeSha256": digest(structural_freeze),
            "primaryPlayableSha256": digest(baseline),
            "candidatePlayableSha256": digest(candidate),
            "mappingCommitment": commitment,
        },
        "gate": {"structuralSafetyGatePassed": True},
        "blindMappingInspected": False,
    }), encoding="utf-8")
    return {
        "pack": pack,
        "freeze": freeze,
        "pre": pre,
        "structural_freeze": structural_freeze,
        "structural": structural,
    }


def run_finalize(paths: dict[str, Path], tmp_path: Path, choice: str, name: str, **kwargs):
    return finalize(
        freeze_path=paths["freeze"],
        pre_listening_path=paths["pre"],
        structural_freeze_path=paths["structural_freeze"],
        structural_result_path=paths["structural"],
        pack=paths["pack"],
        choice=choice,
        reviewer="owner",
        output_dir=tmp_path / name,
        **kwargs,
    )


def test_candidate_choice_passes_and_choice_precedes_reveal(tmp_path: Path) -> None:
    paths = build_evidence(tmp_path)
    output = tmp_path / "candidate-win"
    result = run_finalize(paths, tmp_path, "A", "candidate-win")
    assert result["status"] == "PASS"
    assert result["selectedRole"] == "candidate"
    pre_choice = json.loads((output / "PRE-REVEAL-CHOICE.json").read_text())
    assert pre_choice["mappingOpenedAtRecording"] is False
    assert (output / "REVEALED-MAPPING.json").is_file()


@pytest.mark.parametrize(("choice", "role"), [("B", "baseline"), ("TIE", "tie")])
def test_non_candidate_choice_fails_closed(tmp_path: Path, choice: str, role: str) -> None:
    paths = build_evidence(tmp_path)
    result = run_finalize(paths, tmp_path, choice, f"loss-{choice}")
    assert result["status"] == "FAIL"
    assert result["selectedRole"] == role
    assert "candidate_preferred" in result["failedChecks"]


def test_blocker_fails_even_when_candidate_wins(tmp_path: Path) -> None:
    paths = build_evidence(tmp_path)
    result = run_finalize(paths, tmp_path, "A", "blocker", blocker_heard=True)
    assert result["status"] == "FAIL"
    assert "no_blocker_heard" in result["failedChecks"]


def test_tampered_commitment_fails_after_prechoice_before_reveal(tmp_path: Path) -> None:
    paths = build_evidence(tmp_path)
    (paths["pack"] / "MAPPING-COMMITMENT.sha256").write_text("0" * 64 + "\n")
    output = tmp_path / "tampered"
    with pytest.raises(Exception, match="commitment"):
        run_finalize(paths, tmp_path, "A", "tampered")
    # Input verification rejects the altered pack before opening its mapping.
    assert not (output / "REVEALED-MAPPING.json").exists()
