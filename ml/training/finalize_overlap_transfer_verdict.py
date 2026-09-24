"""Finalize the Phase 94 opened-song blind overlap-transfer verdict.

The supplied A/B/TIE choice is atomically recorded before this module reads
the sealed mapping.  It then verifies the commitment, frozen artifact hashes,
and structural transfer gate.  PASS means only that the fixed development
winner cleared this opened-song perceptual gate; it never authorizes a sealed
test result, commercial use, or production deployment by itself.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ml.training.finalize_blind_listening_verdict import (
    read_json,
    sha256_file,
    verify_sealed_mapping,
)


class OverlapVerdictError(RuntimeError):
    """Raised when blind-transfer evidence cannot be finalized safely."""


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def required_string(payload: dict[str, Any], key: str, context: str) -> str:
    value = str(payload.get(key) or "").strip()
    if not value:
        raise OverlapVerdictError(f"{context} has no {key}")
    return value


def verify_pre_reveal_inputs(
    *,
    freeze_path: Path,
    pre_listening_path: Path,
    structural_freeze_path: Path,
    structural_result_path: Path,
    pack: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], str]:
    freeze = read_json(freeze_path)
    pre_listening = read_json(pre_listening_path)
    structural_freeze = read_json(structural_freeze_path)
    structural = read_json(structural_result_path)
    if freeze.get("schema") != "polymath-candidate-freeze-v1":
        raise OverlapVerdictError("Unsupported opened-transfer freeze schema")
    if pre_listening.get("schema") != "polymath-opened-song-pre-listening-evidence-v1":
        raise OverlapVerdictError("Unsupported pre-listening evidence schema")
    if structural_freeze.get("schema") != "polymath-candidate-freeze-v1":
        raise OverlapVerdictError("Unsupported structural freeze schema")
    if structural.get("schema") != "polymath-unlabeled-transfer-safety-evidence-v1":
        raise OverlapVerdictError("Unsupported structural evidence schema")
    versions = {
        required_string(freeze, "candidateVersion", "freeze"),
        required_string(pre_listening, "candidateVersion", "pre-listening evidence"),
    }
    # The structural audit has its own version but must bind to the same
    # playable artifacts and commitment below.
    if len(versions) != 1:
        raise OverlapVerdictError("Freeze and pre-listening candidate versions differ")
    if required_string(
        structural_freeze, "candidateVersion", "structural freeze"
    ) != required_string(structural, "candidateVersion", "structural evidence"):
        raise OverlapVerdictError("Structural freeze and result candidate versions differ")
    expected_freeze_hash = required_string(
        pre_listening.get("artifacts") or {}, "freezeSha256", "pre-listening artifacts"
    )
    if sha256_file(freeze_path) != expected_freeze_hash:
        raise OverlapVerdictError("Opened-transfer freeze hash no longer matches")
    if bool((pre_listening.get("blindState") or {}).get("mappingInspected")):
        raise OverlapVerdictError("Pre-listening evidence says the mapping was already inspected")
    if bool(structural.get("blindMappingInspected")):
        raise OverlapVerdictError("Structural evidence says the mapping was already inspected")
    structural_artifacts = structural.get("artifacts") or {}
    expected_structural_freeze_hash = required_string(
        structural_artifacts, "freezeSha256", "structural artifacts"
    )
    if sha256_file(structural_freeze_path) != expected_structural_freeze_hash:
        raise OverlapVerdictError("Structural freeze hash no longer matches")
    commitment = required_string(
        pre_listening.get("outputs") or {}, "mappingCommitment", "pre-listening outputs"
    )
    pack_commitment = (pack / "MAPPING-COMMITMENT.sha256").read_text(encoding="utf-8").strip()
    if pack_commitment != commitment:
        raise OverlapVerdictError("Listening-pack commitment no longer matches pre-listening evidence")
    pre_artifacts = pre_listening.get("artifacts") or {}
    for key in ("primaryPlayableSha256", "candidatePlayableSha256"):
        if required_string(structural_artifacts, key, "structural artifacts") != required_string(
            pre_artifacts, key, "pre-listening artifacts"
        ):
            raise OverlapVerdictError(f"Structural and pre-listening {key} differ")
    if required_string(structural_artifacts, "mappingCommitment", "structural artifacts") != commitment:
        raise OverlapVerdictError("Structural evidence uses a different listening commitment")

    # Hashing the sealed map does not reveal it.  These checks detect any pack
    # mutation before the listener's choice is persisted and before JSON is
    # parsed from SEALED-MAPPING.json.
    pack_artifacts = {
        "audioRenderManifestSha256": pack / "AUDIO-RENDER.json",
        "listenPageSha256": pack / "LISTEN.html",
        "sealedMappingSha256": pack / "SEALED-MAPPING.json",
    }
    for key, path in pack_artifacts.items():
        if not path.is_file():
            raise OverlapVerdictError(f"Listening pack is missing {path.name}")
        if sha256_file(path) != required_string(pre_artifacts, key, "pre-listening artifacts"):
            raise OverlapVerdictError(f"Listening-pack {path.name} hash no longer matches")
    blind_outputs = pre_listening.get("outputs") or {}
    for version, filename in (("versionA", "02-A.wav"), ("versionB", "03-B.wav")):
        path = pack / filename
        if not path.is_file():
            raise OverlapVerdictError(f"Listening pack is missing {filename}")
        expected = required_string(blind_outputs.get(version) or {}, "sha256", version)
        if sha256_file(path) != expected:
            raise OverlapVerdictError(f"Listening-pack {filename} hash no longer matches")
    return freeze, pre_listening, structural_freeze, structural, commitment


def finalize(
    *,
    freeze_path: Path,
    pre_listening_path: Path,
    structural_freeze_path: Path,
    structural_result_path: Path,
    pack: Path,
    choice: str,
    reviewer: str,
    output_dir: Path,
    notes: str = "",
    blocker_heard: bool = False,
) -> dict[str, Any]:
    choice = str(choice).strip().upper()
    if choice not in {"A", "B", "TIE"}:
        raise ValueError("choice must be A, B, or TIE")
    reviewer = str(reviewer).strip()
    if not reviewer:
        raise ValueError("reviewer is required")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite final blind evidence: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    freeze, pre_listening, structural_freeze, structural, commitment = verify_pre_reveal_inputs(
        freeze_path=freeze_path,
        pre_listening_path=pre_listening_path,
        structural_freeze_path=structural_freeze_path,
        structural_result_path=structural_result_path,
        pack=pack,
    )
    recorded_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    pre_reveal_choice = {
        "schema": "polymath-pre-reveal-overlap-transfer-choice-v1",
        "recordedAtUtc": recorded_at,
        "choice": choice,
        "reviewer": reviewer,
        "notes": str(notes)[:1000],
        "blockerHeard": bool(blocker_heard),
        "mappingOpenedAtRecording": False,
        "listeningPackCommitment": commitment,
        "freezeSha256": sha256_file(freeze_path),
        "preListeningEvidenceSha256": sha256_file(pre_listening_path),
        "structuralFreezeSha256": sha256_file(structural_freeze_path),
        "structuralEvidenceSha256": sha256_file(structural_result_path),
    }
    atomic_json(output_dir / "PRE-REVEAL-CHOICE.json", pre_reveal_choice)

    # This is intentionally the first operation that opens SEALED-MAPPING.json.
    mapping = verify_sealed_mapping(pack)
    sealed = read_json(pack / "SEALED-MAPPING.json")
    commitment_payload = sealed["commitmentPayload"]
    pre_artifacts = pre_listening["artifacts"]
    if str(commitment_payload.get("baselineSha256") or "") != str(
        pre_artifacts.get("primaryPlayableSha256") or ""
    ):
        raise OverlapVerdictError("Sealed baseline is not the frozen primary playable artifact")
    if str(commitment_payload.get("candidateSha256") or "") != str(
        pre_artifacts.get("candidatePlayableSha256") or ""
    ):
        raise OverlapVerdictError("Sealed candidate is not the frozen overlap playable artifact")

    selected_role = mapping.get(choice) if choice in {"A", "B"} else "tie"
    checks = {
        "commitment_verified": True,
        "structural_safety_gate": bool((structural.get("gate") or {}).get("structuralSafetyGatePassed")),
        "candidate_preferred": selected_role == "candidate",
        "no_blocker_heard": not blocker_heard,
    }
    failed = [name for name, passed in checks.items() if not passed]
    result = {
        "schema": "polymath-opened-song-overlap-transfer-verdict-v1",
        "candidateVersion": freeze["candidateVersion"],
        "recordedAtUtc": recorded_at,
        "reviewer": reviewer,
        "blindChoice": choice,
        "selectedRole": selected_role,
        "notes": str(notes)[:1000],
        "blockerHeard": bool(blocker_heard),
        "mapping": mapping,
        "mappingCommitment": commitment,
        "checks": checks,
        "failedChecks": failed,
        "status": "PASS" if not failed else "FAIL",
        "decision": (
            "eligible-for-single-use-sealed-test-freeze"
            if not failed
            else "do-not-open-sealed-test"
        ),
        "accuracyEvidenceAllowed": False,
        "sealedTestOpened": False,
        "productionPromotionAllowed": False,
    }
    atomic_json(output_dir / "REVEALED-MAPPING.json", {
        "schema": "polymath-verified-overlap-transfer-map-v1",
        "warning": "Written only after PRE-REVEAL-CHOICE.json.",
        "commitment": commitment,
        "mapping": mapping,
    })
    atomic_json(output_dir / "RESULT.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--pre-listening", type=Path, required=True)
    parser.add_argument("--structural-freeze", type=Path, required=True)
    parser.add_argument("--structural-result", type=Path, required=True)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--choice", required=True, choices=("A", "B", "TIE"))
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--notes", default="")
    parser.add_argument("--blocker-heard", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = finalize(
        freeze_path=args.freeze.resolve(),
        pre_listening_path=args.pre_listening.resolve(),
        structural_freeze_path=args.structural_freeze.resolve(),
        structural_result_path=args.structural_result.resolve(),
        pack=args.pack.resolve(),
        choice=args.choice,
        reviewer=args.reviewer,
        output_dir=args.output_dir.resolve(),
        notes=args.notes,
        blocker_heard=args.blocker_heard,
    )
    print(json.dumps({
        "status": result["status"],
        "selectedRole": result["selectedRole"],
        "failedChecks": result["failedChecks"],
        "sealedTestOpened": False,
    }, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
