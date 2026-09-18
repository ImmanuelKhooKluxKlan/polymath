"""Fail-closed verification for a frozen piano-arranger promotion.

This tool does not deploy a profile.  It proves that one exact frozen profile
and runtime passed the recorded automated evaluation and the predeclared blind
listening gate.  Commercial deployment remains a separate licensing decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FREEZE_SCHEMA = "polymath-piano-arranger-candidate-freeze-v1"
RESULT_SCHEMA = "polymath-piano-arranger-final-holdout-result-v1"
MAPPING_SCHEMA = "polymath-blind-listening-map-v1"
VERDICT_SCHEMA = "polymath-piano-arranger-blind-verdict-v1"
RECEIPT_SCHEMA = "polymath-piano-arranger-promotion-verification-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _version_token(profile_id: str) -> str:
    matches = re.findall(r"v\d+", str(profile_id).lower())
    if not matches:
        raise ValueError("The frozen candidate ID must contain a version such as v031")
    return matches[-1]


def _record_check(checks: dict[str, bool], failures: list[str], name: str, passed: bool) -> None:
    checks[name] = bool(passed)
    if not passed:
        failures.append(name)


def _valid_completed_at(value: Any) -> bool:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return False
    return parsed.tzinfo is not None


def verify_promotion(
    *,
    freeze: dict[str, Any],
    result: dict[str, Any],
    mapping: dict[str, Any],
    verdict: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    failures: list[str] = []

    _record_check(checks, failures, "freeze_schema", freeze.get("schema") == FREEZE_SCHEMA)
    _record_check(checks, failures, "result_schema", result.get("schema") == RESULT_SCHEMA)
    _record_check(checks, failures, "mapping_schema", mapping.get("schema") == MAPPING_SCHEMA)
    _record_check(checks, failures, "verdict_schema", verdict.get("schema") == VERDICT_SCHEMA)

    frozen_candidate = freeze.get("candidate") if isinstance(freeze.get("candidate"), dict) else {}
    result_candidate = result.get("candidate") if isinstance(result.get("candidate"), dict) else {}
    candidate_id = str(frozen_candidate.get("id") or "")
    candidate_path = Path(str(frozen_candidate.get("path") or ""))
    expected_candidate_hash = str(frozen_candidate.get("fileSha256") or "").lower()

    _record_check(checks, failures, "candidate_file_exists", candidate_path.is_file())
    actual_candidate_hash = sha256_file(candidate_path) if candidate_path.is_file() else ""
    _record_check(
        checks,
        failures,
        "candidate_hash_matches_freeze",
        bool(expected_candidate_hash) and actual_candidate_hash == expected_candidate_hash,
    )
    _record_check(
        checks,
        failures,
        "result_identifies_frozen_candidate",
        result_candidate.get("id") == candidate_id
        and str(result_candidate.get("fileSha256") or "").lower() == expected_candidate_hash,
    )
    _record_check(
        checks,
        failures,
        "evaluation_used_frozen_runtime",
        result_candidate.get("profileHashMatchedFreezeAtScoring") is True
        and result_candidate.get("runtimeHashesMatchedFreezeAtScoring") is True
        and result_candidate.get("retunedAfterFirstHoldoutScore") is False,
    )

    runtime_files = freeze.get("runtime", {}).get("files", {})
    runtime_ok = isinstance(runtime_files, dict) and bool(runtime_files)
    runtime_actual: dict[str, str] = {}
    if runtime_ok:
        for relative_path, expected_hash in runtime_files.items():
            path = repo_root / str(relative_path)
            if not path.is_file():
                runtime_ok = False
                continue
            actual_hash = sha256_file(path)
            runtime_actual[str(relative_path)] = actual_hash
            if actual_hash != str(expected_hash).lower():
                runtime_ok = False
    _record_check(checks, failures, "runtime_hashes_still_match_freeze", runtime_ok)

    primary = result.get("primaryResult") if isinstance(result.get("primaryResult"), dict) else {}
    baseline_metrics = primary.get("baseline") if isinstance(primary.get("baseline"), dict) else {}
    candidate_metrics = primary.get("candidate") if isinstance(primary.get("candidate"), dict) else {}
    _record_check(
        checks,
        failures,
        "all_automated_promotion_gates_passed",
        primary.get("decision") == "PROMOTE" and primary.get("allAutomatedPromotionGatesPassed") is True,
    )
    _record_check(
        checks,
        failures,
        "heldout_exact_100ms_improved",
        float(candidate_metrics.get("exactF1At100ms", -1))
        > float(baseline_metrics.get("exactF1At100ms", 2)),
    )
    _record_check(
        checks,
        failures,
        "heldout_duration_not_regressed",
        float(candidate_metrics.get("physicalDurationMaeSeconds", 2))
        <= float(baseline_metrics.get("physicalDurationMaeSeconds", -1)),
    )
    _record_check(
        checks,
        failures,
        "heldout_cutoffs_not_regressed",
        float(candidate_metrics.get("physicalSevereCutoffRate", 2))
        <= float(baseline_metrics.get("physicalSevereCutoffRate", -1)),
    )
    _record_check(
        checks,
        failures,
        "heldout_retriggers_not_regressed",
        int(candidate_metrics.get("rapidRetriggersUnder100ms", 10**9))
        <= int(baseline_metrics.get("rapidRetriggersUnder100ms", -1)),
    )

    _record_check(checks, failures, "reviewer_recorded", bool(str(verdict.get("reviewer") or "").strip()))
    _record_check(checks, failures, "review_timestamp_valid", _valid_completed_at(verdict.get("completedAtUtc")))
    _record_check(checks, failures, "same_playback_chain", verdict.get("samePlaybackChain") is True)
    _record_check(
        checks,
        failures,
        "mapping_was_sealed_during_review",
        verdict.get("mappingOpenedBeforeVerdict") is False,
    )

    map_rows = mapping.get("candidateAorB") if isinstance(mapping.get("candidateAorB"), dict) else {}
    verdict_rows = verdict.get("recordings") if isinstance(verdict.get("recordings"), dict) else {}
    candidate_token = _version_token(candidate_id) if candidate_id else ""
    decisive_ids = {
        str(item).strip()
        for item in verdict.get("decisiveRecordingIds", [])
        if str(item).strip()
    } if isinstance(verdict.get("decisiveRecordingIds"), list) else set()
    _record_check(checks, failures, "at_least_one_decisive_recording", bool(decisive_ids))

    all_rows_complete = bool(map_rows) and set(verdict_rows) == set(map_rows)
    no_blockers = all_rows_complete
    decisive_candidate_wins = bool(decisive_ids) and decisive_ids.issubset(map_rows)
    supporting_has_no_baseline_win = all_rows_complete
    resolved_rows: dict[str, Any] = {}
    if all_rows_complete:
        for recording_id, choices in map_rows.items():
            review = verdict_rows.get(recording_id)
            if not isinstance(choices, dict) or not isinstance(review, dict):
                all_rows_complete = False
                break
            winner = str(review.get("winner") or "").strip().upper()
            if winner not in {"A", "B", "TIE"} or not isinstance(review.get("blockerHeard"), bool):
                all_rows_complete = False
                break
            no_blockers = no_blockers and review["blockerHeard"] is False
            chosen_label = "tie" if winner == "TIE" else str(choices.get(winner) or "").lower()
            chosen_candidate = winner != "TIE" and candidate_token in chosen_label
            if recording_id in decisive_ids:
                decisive_candidate_wins = decisive_candidate_wins and chosen_candidate
            elif winner != "TIE":
                supporting_has_no_baseline_win = supporting_has_no_baseline_win and chosen_candidate
            resolved_rows[recording_id] = {
                "winner": winner,
                "winnerWasCandidate": chosen_candidate,
                "blockerHeard": review["blockerHeard"],
                "notes": str(review.get("notes") or "")[:1000],
            }
    if not all_rows_complete:
        no_blockers = False
        decisive_candidate_wins = False
        supporting_has_no_baseline_win = False

    _record_check(checks, failures, "blind_rows_complete", all_rows_complete)
    _record_check(checks, failures, "blind_review_has_no_blocker", no_blockers)
    _record_check(checks, failures, "candidate_won_every_decisive_recording", decisive_candidate_wins)
    _record_check(checks, failures, "candidate_did_not_lose_supporting_recordings", supporting_has_no_baseline_win)

    passed = not failures
    return {
        "schema": RECEIPT_SCHEMA,
        "generatedAtUtc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "PASS" if passed else "FAIL",
        "candidate": {
            "id": candidate_id,
            "fileSha256": actual_candidate_hash or None,
        },
        "scope": "research-only",
        "commercialDeploymentAuthorized": False,
        "checks": checks,
        "failedChecks": failures,
        "blindReview": resolved_rows,
        "runtimeFileSha256": runtime_actual,
        "notice": (
            "Technical promotion gates passed. Commercial deployment still requires separate licence clearance."
            if passed
            else "No promotion is authorized. Correct the failed evidence or evaluate a new frozen candidate."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--verdict", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    receipt = verify_promotion(
        freeze=read_json(args.freeze),
        result=read_json(args.result),
        mapping=read_json(args.mapping),
        verdict=read_json(args.verdict),
        repo_root=args.repo_root.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "failedChecks": receipt["failedChecks"]}, indent=2))
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
