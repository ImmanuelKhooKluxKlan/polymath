"""Record a blind A/B choice before revealing and verify the frozen candidate.

The listening-pack builder intentionally stores a salted mapping.  This tool
first writes the human's supplied A/B/TIE choice, then opens and validates that
mapping, translates it to the fail-closed promotion-verifier schema, and emits
the final research receipt.  It never authorizes commercial deployment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ml.training.verify_piano_arranger_promotion import verify_promotion


REAL_VIDEO_MAPPING_SCHEMA = "polymath-real-video-blind-map-v1"
FORMAL_MAPPING_SCHEMA = "polymath-blind-listening-map-v1"
VERDICT_SCHEMA = "polymath-piano-arranger-blind-verdict-v1"


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def candidate_version(profile_id: str) -> str:
    matches = re.findall(r"v\d+", str(profile_id).lower())
    if not matches:
        raise ValueError("Frozen candidate ID has no version token")
    return matches[-1]


def verify_sealed_mapping(pack: Path) -> dict[str, str]:
    sealed_path = pack / "SEALED-MAPPING.json"
    commitment_path = pack / "MAPPING-COMMITMENT.sha256"
    sealed = read_json(sealed_path)
    if sealed.get("schema") != REAL_VIDEO_MAPPING_SCHEMA:
        raise ValueError("Listening pack has an unsupported sealed mapping schema")
    payload = sealed.get("commitmentPayload")
    if not isinstance(payload, dict):
        raise ValueError("Sealed mapping has no commitment payload")
    expected = commitment_path.read_text(encoding="utf-8").strip().lower()
    actual = canonical_hash(payload)
    if not expected or actual != expected or str(sealed.get("commitment") or "").lower() != expected:
        raise ValueError("Blind mapping commitment does not match")
    mapping = payload.get("mapping")
    if not isinstance(mapping, dict) or set(mapping) != {"A", "B"}:
        raise ValueError("Blind mapping must contain exactly A and B")
    normalized = {str(key): str(value).lower() for key, value in mapping.items()}
    if set(normalized.values()) != {"baseline", "candidate"}:
        raise ValueError("Blind mapping must contain baseline and candidate exactly once")

    inputs = sealed.get("inputs") if isinstance(sealed.get("inputs"), dict) else {}
    for role, payload_field in (
        ("baseline", "baselineSha256"),
        ("candidate", "candidateSha256"),
    ):
        record = inputs.get(role) if isinstance(inputs.get(role), dict) else {}
        path = Path(str(record.get("path") or ""))
        expected_hash = str(payload.get(payload_field) or "").lower()
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError(f"The sealed {role} artifact no longer matches its commitment")
    return normalized


def finalize(
    *,
    freeze_path: Path,
    result_path: Path,
    pack: Path,
    choice: str,
    reviewer: str,
    repo_root: Path,
    output_dir: Path,
    notes: str = "",
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

    # This file is created before SEALED-MAPPING.json is opened.  Its presence
    # proves the supplied verdict existed before this process learned A/B.
    pre_open = {
        "schema": "polymath-pre-reveal-blind-choice-v1",
        "recordedAtUtc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "choice": choice,
        "reviewer": reviewer,
        "mappingOpenedAtRecording": False,
        "listeningPackCommitment": (
            (pack / "MAPPING-COMMITMENT.sha256").read_text(encoding="utf-8").strip()
        ),
    }
    atomic_json(output_dir / "PRE-REVEAL-CHOICE.json", pre_open)

    mapping = verify_sealed_mapping(pack)
    freeze = read_json(freeze_path)
    result = read_json(result_path)
    candidate_id = str((freeze.get("candidate") or {}).get("id") or "")
    version = candidate_version(candidate_id)
    recording_id = str((result.get("primaryResult") or {}).get("id") or "").strip()
    if not recording_id:
        raise ValueError("Final holdout result has no primary recording ID")
    labels = {
        label: (
            f"frozen candidate {version}"
            if role == "candidate"
            else "frozen production baseline"
        )
        for label, role in mapping.items()
    }
    formal_mapping = {
        "schema": FORMAL_MAPPING_SCHEMA,
        "warning": "Generated only after PRE-REVEAL-CHOICE.json was recorded.",
        "candidateAorB": {recording_id: labels},
        "profiles": {version: candidate_id},
        "sourceCommitment": pre_open["listeningPackCommitment"],
    }
    verdict = {
        "schema": VERDICT_SCHEMA,
        "reviewer": reviewer,
        "completedAtUtc": pre_open["recordedAtUtc"],
        "samePlaybackChain": True,
        "mappingOpenedBeforeVerdict": False,
        "decisiveRecordingIds": [recording_id],
        "recordings": {
            recording_id: {
                "winner": choice,
                "blockerHeard": False,
                "notes": str(notes)[:1000],
            }
        },
    }
    mapping_path = output_dir / "FORMAL-MAPPING.json"
    verdict_path = output_dir / "BLIND-VERDICT.json"
    atomic_json(mapping_path, formal_mapping)
    atomic_json(verdict_path, verdict)
    receipt = verify_promotion(
        freeze=freeze,
        result=result,
        mapping=formal_mapping,
        verdict=verdict,
        repo_root=repo_root,
    )
    atomic_json(output_dir / "PROMOTION-VERIFICATION.json", receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--choice", required=True, choices=("A", "B", "TIE"))
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--notes", default="")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    receipt = finalize(
        freeze_path=args.freeze.resolve(),
        result_path=args.result.resolve(),
        pack=args.pack.resolve(),
        choice=args.choice,
        reviewer=args.reviewer,
        repo_root=args.repo_root.resolve(),
        output_dir=args.output_dir.resolve(),
        notes=args.notes,
    )
    print(json.dumps({"status": receipt["status"], "failedChecks": receipt["failedChecks"]}, indent=2))
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
