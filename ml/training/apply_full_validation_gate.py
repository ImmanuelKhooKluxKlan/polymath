"""Apply a pre-frozen Phase94 onset, duration, and stability gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ml.training.rescore_song_timelines import sha256_file


class FullValidationGateError(RuntimeError):
    """Raised when frozen gate evidence cannot be paired safely."""


def _metric(payload: dict[str, Any], *path: str) -> float:
    value: Any = payload
    try:
        for part in path:
            value = value[part]
        return float(value)
    except (KeyError, TypeError, ValueError) as exc:
        raise FullValidationGateError(f"Missing metric: {'.'.join(path)}") from exc


def evaluate_gate(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    gate: dict[str, Any],
) -> dict[str, Any]:
    if baseline.get("schema") != "polymath-stitched-song-timeline-score-v1":
        raise FullValidationGateError("Baseline has an unexpected schema")
    if candidate.get("schema") != "polymath-stitched-song-timeline-score-v1":
        raise FullValidationGateError("Candidate has an unexpected schema")
    if gate.get("schema") != "polymath-phase94-full-validation-gate-v1":
        raise FullValidationGateError("Gate has an unexpected schema")
    if baseline.get("songIds") != candidate.get("songIds"):
        raise FullValidationGateError("Baseline and candidate song identities differ")

    policy = gate["requiredChecks"]
    base_100 = baseline["metrics"]["100ms"]
    cand_100 = candidate["metrics"]["100ms"]
    base_diag = baseline["diagnostics100ms"]
    cand_diag = candidate["diagnostics100ms"]

    base_matched = max(1.0, float(base_diag["matchedNotes"]))
    cand_matched = max(1.0, float(cand_diag["matchedNotes"]))
    base_predicted = max(1.0, float(base_diag["predictedNotes"]))
    cand_predicted = max(1.0, float(cand_diag["predictedNotes"]))
    rates = {
        "baseline": {
            "cutOffPer1000Matched": 1000 * float(base_diag["cutOffNotes"]) / base_matched,
            "overlongPer1000Matched": 1000 * float(base_diag["overlongNotes"]) / base_matched,
            "rapidRetriggersPer1000Predicted": 1000 * float(base_diag["rapidRetriggers"]) / base_predicted,
        },
        "candidate": {
            "cutOffPer1000Matched": 1000 * float(cand_diag["cutOffNotes"]) / cand_matched,
            "overlongPer1000Matched": 1000 * float(cand_diag["overlongNotes"]) / cand_matched,
            "rapidRetriggersPer1000Predicted": 1000 * float(cand_diag["rapidRetriggers"]) / cand_predicted,
        },
    }
    per_song_deltas = {
        song_id: round(
            _metric(candidate, "perSong", song_id, "100ms", "microF1")
            - _metric(baseline, "perSong", song_id, "100ms", "microF1"),
            6,
        )
        for song_id in baseline["songIds"]
    }
    checks = {
        "microF1_100ms_exceeds_original": (
            float(cand_100["microF1"]) > float(base_100["microF1"])
        ),
        "microF1_100ms_minimum": (
            float(cand_100["microF1"]) >= float(policy["microF1_100msMinimum"])
        ),
        "precision_100ms": (
            float(cand_100["precision"])
            >= float(base_100["precision"])
            - float(policy["precision100msMaximumRegression"])
        ),
        "recall_100ms": (
            float(cand_100["recall"])
            >= float(base_100["recall"])
            - float(policy["recall100msMaximumRegression"])
        ),
        "per_song_100ms": (
            min(per_song_deltas.values(), default=0.0)
            >= -float(policy["maximumPerSongF1Regression"])
        ),
        "onset_and_offset_f1": (
            _metric(cand_diag, "onsetAndOffset", "f1")
            >= _metric(base_diag, "onsetAndOffset", "f1")
            - float(policy["onsetAndOffsetF1MaximumRegression"])
        ),
        "frame_f1": (
            _metric(cand_diag, "frame", "f1")
            >= _metric(base_diag, "frame", "f1")
            - float(policy["frameF1MaximumRegression"])
        ),
        "cutoff_rate": (
            rates["candidate"]["cutOffPer1000Matched"]
            <= rates["baseline"]["cutOffPer1000Matched"]
            + float(policy["cutOffPer1000MatchedMaximumIncrease"])
        ),
        "overlong_rate": (
            rates["candidate"]["overlongPer1000Matched"]
            <= rates["baseline"]["overlongPer1000Matched"]
            + float(policy["overlongPer1000MatchedMaximumIncrease"])
        ),
        "rapid_retrigger_rate": (
            rates["candidate"]["rapidRetriggersPer1000Predicted"]
            <= rates["baseline"]["rapidRetriggersPer1000Predicted"]
            + float(policy["rapidRetriggersPer1000PredictedMaximumIncrease"])
        ),
    }
    lower95 = _metric(candidate, "songClusterBootstrap95", "100ms", "lower95")
    certification_minimum = float(
        gate["certificationCheck"]["songBootstrapLower95_100msMinimum"]
    )
    return {
        "schema": "polymath-phase94-full-validation-gate-result-v1",
        "candidateVersion": gate.get("candidateVersion"),
        "checks": checks,
        "failedChecks": [name for name, passed in checks.items() if not passed],
        "researchGatePassed": all(checks.values()),
        "certification": {
            "lower95": lower95,
            "minimum": certification_minimum,
            "passed": lower95 >= certification_minimum,
        },
        "metrics": {
            "baseline100ms": base_100,
            "candidate100ms": cand_100,
            "candidateMinusBaseline100msF1": round(
                float(cand_100["microF1"]) - float(base_100["microF1"]), 6,
            ),
            "perSong100msF1Delta": per_song_deltas,
            "worstPerSong100msF1Delta": min(per_song_deltas.values(), default=0.0),
            "ratesPer1000": rates,
            "baselineOnsetAndOffsetF1": _metric(base_diag, "onsetAndOffset", "f1"),
            "candidateOnsetAndOffsetF1": _metric(cand_diag, "onsetAndOffset", "f1"),
            "baselineFrameF1": _metric(base_diag, "frame", "f1"),
            "candidateFrameF1": _metric(cand_diag, "frame", "f1"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    baseline_path = args.baseline.resolve()
    candidate_path = args.candidate.resolve()
    gate_path = args.gate.resolve()
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    expected_baseline_hash = str(
        (gate.get("immutableOriginal") or {}).get("evaluationSha256") or ""
    )
    actual_baseline_hash = sha256_file(baseline_path)
    if expected_baseline_hash and actual_baseline_hash != expected_baseline_hash:
        raise FullValidationGateError("Frozen baseline SHA-256 does not match")
    result = evaluate_gate(
        json.loads(baseline_path.read_text(encoding="utf-8")),
        json.loads(candidate_path.read_text(encoding="utf-8")),
        gate,
    )
    result.update({
        "baseline": str(baseline_path),
        "baselineSha256": actual_baseline_hash,
        "candidate": str(candidate_path),
        "candidateSha256": sha256_file(candidate_path),
        "gate": str(gate_path),
        "gateSha256": sha256_file(gate_path),
    })
    destination = args.out.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(destination),
        "researchGatePassed": result["researchGatePassed"],
        "certificationPassed": result["certification"]["passed"],
        "failedChecks": result["failedChecks"],
    }, indent=2))


if __name__ == "__main__":
    main()
