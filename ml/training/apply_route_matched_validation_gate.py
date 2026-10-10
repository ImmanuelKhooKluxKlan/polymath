"""Fail-closed validation gate for route-matched MuScriptor checkpoints.

This gate is deliberately separate from checkpoint training.  It binds the
training receipt, paired decoded evaluation, and continuous-song rescoring so
that a lower teacher-forced loss or a good five-second clip score cannot move a
candidate forward on its own.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ml.training.rescore_song_timelines import sha256_file


class RouteMatchedGateError(RuntimeError):
    """Raised when gate evidence is incomplete, inconsistent, or mutated."""


def _metric(payload: dict[str, Any], *path: str) -> float:
    value: Any = payload
    try:
        for part in path:
            value = value[part]
        return float(value)
    except (KeyError, TypeError, ValueError) as exc:
        raise RouteMatchedGateError(f"Missing metric: {'.'.join(path)}") from exc


def _rate_summary(timeline: dict[str, Any]) -> dict[str, float]:
    diagnostics = timeline["diagnostics100ms"]
    matched = max(1.0, float(diagnostics["matchedNotes"]))
    predicted = max(1.0, float(diagnostics["predictedNotes"]))
    return {
        "cutOffPer1000Matched": (
            1000.0 * float(diagnostics["cutOffNotes"]) / matched
        ),
        "overlongPer1000Matched": (
            1000.0 * float(diagnostics["overlongNotes"]) / matched
        ),
        "rapidRetriggersPer1000Predicted": (
            1000.0 * float(diagnostics["rapidRetriggers"]) / predicted
        ),
    }


def _validate_inputs(
    comparison: dict[str, Any],
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    training: dict[str, Any],
    gate: dict[str, Any],
) -> None:
    if comparison.get("schema") != "polymath-checkpoint-comparison-v1":
        raise RouteMatchedGateError("Comparison has an unexpected schema")
    for label, timeline in (("baseline", baseline), ("candidate", candidate)):
        if timeline.get("schema") != "polymath-stitched-song-timeline-score-v1":
            raise RouteMatchedGateError(f"{label.title()} timeline has an unexpected schema")
    if training.get("schema") != "polymath-muscriptor-training-run-v1":
        raise RouteMatchedGateError("Training receipt has an unexpected schema")
    if gate.get("schema") != "polymath-route-matched-validation-gate-v1":
        raise RouteMatchedGateError("Gate has an unexpected schema")
    if baseline.get("songIds") != candidate.get("songIds"):
        raise RouteMatchedGateError("Baseline and candidate song identities differ")


def evaluate_gate(
    comparison: dict[str, Any],
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    training: dict[str, Any],
    gate: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate the frozen research gate without mutating any model or route."""

    _validate_inputs(comparison, baseline, candidate, training, gate)
    policy = gate["requiredChecks"]
    route = gate["route"]

    baseline_continuous = baseline["metrics"]
    candidate_continuous = candidate["metrics"]
    baseline_diagnostics = baseline["diagnostics100ms"]
    candidate_diagnostics = candidate["diagnostics100ms"]
    rates = {
        "baseline": _rate_summary(baseline),
        "candidate": _rate_summary(candidate),
    }
    per_song_deltas = {
        song_id: round(
            _metric(candidate, "perSong", song_id, "100ms", "microF1")
            - _metric(baseline, "perSong", song_id, "100ms", "microF1"),
            6,
        )
        for song_id in baseline["songIds"]
    }

    baseline_loss = _metric(training, "baselineValidationLoss")
    candidate_loss = _metric(training, "selectedValidationLoss")
    continuous_100_delta = (
        _metric(candidate_continuous, "100ms", "microF1")
        - _metric(baseline_continuous, "100ms", "microF1")
    )
    checks = {
        "conditioning_mode_matches_live_route": (
            str(training.get("conditioningMode") or "")
            == str(route["conditioningMode"])
        ),
        "instrument_constraint_matches_live_route": (
            list(comparison.get("instrumentConstraint") or [])
            == list(route["instrumentConstraint"])
        ),
        "minimum_complete_songs": (
            int(candidate.get("songs") or 0) >= int(policy["minimumCompleteSongs"])
        ),
        "route_matched_validation_loss_improves": candidate_loss < baseline_loss,
        "clip_local_100ms_f1": (
            _metric(comparison, "candidate", "100ms", "microF1")
            >= _metric(comparison, "baseline", "100ms", "microF1")
            - float(policy["clipLocal100msF1MaximumRegression"])
        ),
        "clip_local_250ms_f1": (
            _metric(comparison, "candidate", "250ms", "microF1")
            >= _metric(comparison, "baseline", "250ms", "microF1")
            - float(policy["clipLocal250msF1MaximumRegression"])
        ),
        "continuous_100ms_f1_strictly_improves": (
            continuous_100_delta
            > float(policy["continuous100msF1MinimumImprovement"])
        ),
        "continuous_100ms_precision": (
            _metric(candidate_continuous, "100ms", "precision")
            >= _metric(baseline_continuous, "100ms", "precision")
            - float(policy["continuous100msPrecisionMaximumRegression"])
        ),
        "continuous_100ms_recall": (
            _metric(candidate_continuous, "100ms", "recall")
            >= _metric(baseline_continuous, "100ms", "recall")
            - float(policy["continuous100msRecallMaximumRegression"])
        ),
        "continuous_250ms_f1": (
            _metric(candidate_continuous, "250ms", "microF1")
            >= _metric(baseline_continuous, "250ms", "microF1")
            - float(policy["continuous250msF1MaximumRegression"])
        ),
        "song_bootstrap_lower95": (
            _metric(candidate, "songClusterBootstrap95", "100ms", "lower95")
            >= _metric(baseline, "songClusterBootstrap95", "100ms", "lower95")
            - float(policy["songBootstrapLower95MaximumRegression"])
        ),
        "per_song_100ms_f1": (
            min(per_song_deltas.values(), default=float("-inf"))
            >= -float(policy["maximumPerSong100msF1Regression"])
        ),
        "onset_and_offset_f1": (
            _metric(candidate_diagnostics, "onsetAndOffset", "f1")
            >= _metric(baseline_diagnostics, "onsetAndOffset", "f1")
            - float(policy["onsetAndOffsetF1MaximumRegression"])
        ),
        "frame_f1": (
            _metric(candidate_diagnostics, "frame", "f1")
            >= _metric(baseline_diagnostics, "frame", "f1")
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
    failed = [name for name, passed in checks.items() if not passed]
    automated_pass = not failed
    return {
        "schema": "polymath-route-matched-validation-gate-result-v1",
        "candidateVersion": gate.get("candidateVersion"),
        "checks": checks,
        "failedChecks": failed,
        "automatedResearchGatePassed": automated_pass,
        "canProceedToListeningCanary": automated_pass,
        "productionEligible": False,
        "manualGatesStillRequired": list(gate.get("manualGatesStillRequired") or []),
        "metrics": {
            "trainingLoss": {
                "baseline": baseline_loss,
                "candidate": candidate_loss,
                "delta": round(candidate_loss - baseline_loss, 9),
            },
            "clipLocal": {
                "baseline100msF1": _metric(comparison, "baseline", "100ms", "microF1"),
                "candidate100msF1": _metric(comparison, "candidate", "100ms", "microF1"),
                "baseline250msF1": _metric(comparison, "baseline", "250ms", "microF1"),
                "candidate250msF1": _metric(comparison, "candidate", "250ms", "microF1"),
            },
            "continuousSong": {
                "baseline100ms": baseline_continuous["100ms"],
                "candidate100ms": candidate_continuous["100ms"],
                "candidateMinusBaseline100msF1": round(continuous_100_delta, 6),
                "baseline250ms": baseline_continuous["250ms"],
                "candidate250ms": candidate_continuous["250ms"],
                "perSong100msF1Delta": per_song_deltas,
                "worstPerSong100msF1Delta": min(
                    per_song_deltas.values(), default=None
                ),
                "baselineBootstrap100ms": baseline["songClusterBootstrap95"]["100ms"],
                "candidateBootstrap100ms": candidate["songClusterBootstrap95"]["100ms"],
            },
            "ratesPer1000": rates,
            "baselineOnsetAndOffsetF1": _metric(
                baseline_diagnostics, "onsetAndOffset", "f1"
            ),
            "candidateOnsetAndOffsetF1": _metric(
                candidate_diagnostics, "onsetAndOffset", "f1"
            ),
            "baselineFrameF1": _metric(baseline_diagnostics, "frame", "f1"),
            "candidateFrameF1": _metric(candidate_diagnostics, "frame", "f1"),
        },
    }


def _load(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RouteMatchedGateError(f"Cannot read {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RouteMatchedGateError(f"Expected a JSON object in {path}")
    return payload


def _verify_artifact_binding(
    comparison_path: Path,
    comparison: dict[str, Any],
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    training: dict[str, Any],
    gate: dict[str, Any],
) -> dict[str, Any]:
    comparison_hash = sha256_file(comparison_path)
    for label, timeline, expected_side in (
        ("baseline", baseline, "baseline"),
        ("candidate", candidate, "candidate"),
    ):
        if timeline.get("sourceEvaluationSha256") != comparison_hash:
            raise RouteMatchedGateError(
                f"{label.title()} timeline is not bound to this comparison"
            )
        if timeline.get("sourceEvaluationSide") != expected_side:
            raise RouteMatchedGateError(
                f"{label.title()} timeline has the wrong comparison side"
            )
    baseline_manifest_hash = str(baseline.get("validationManifestSha256") or "")
    candidate_manifest_hash = str(candidate.get("validationManifestSha256") or "")
    if not baseline_manifest_hash or baseline_manifest_hash != candidate_manifest_hash:
        raise RouteMatchedGateError("Timeline validation-manifest hashes differ")
    expected_manifest_hash = str(
        (gate.get("immutableInputs") or {}).get("validationManifestSha256") or ""
    )
    if expected_manifest_hash and baseline_manifest_hash != expected_manifest_hash:
        raise RouteMatchedGateError("Frozen validation-manifest SHA-256 does not match")
    expected_base_hash = str(
        (gate.get("immutableInputs") or {}).get("baseCheckpointSha256") or ""
    )
    comparison_base_hash = str(comparison.get("baseCheckpointSha256") or "")
    training_base_hash = str(training.get("baseSha256") or "")
    if expected_base_hash and (
        comparison_base_hash != expected_base_hash
        or training_base_hash != expected_base_hash
    ):
        raise RouteMatchedGateError("Frozen base-checkpoint SHA-256 does not match")
    return {
        "comparisonSha256": comparison_hash,
        "validationManifestSha256": baseline_manifest_hash,
        "baseCheckpointSha256": comparison_base_hash,
        "candidateCheckpointSha256": comparison.get("candidateCheckpointSha256"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--baseline-timeline", type=Path, required=True)
    parser.add_argument("--candidate-timeline", type=Path, required=True)
    parser.add_argument("--training-metadata", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--fail-on-rejection",
        action="store_true",
        help="Exit with status 2 after writing evidence when any check fails.",
    )
    args = parser.parse_args()

    comparison_path = args.comparison.resolve()
    baseline_path = args.baseline_timeline.resolve()
    candidate_path = args.candidate_timeline.resolve()
    training_path = args.training_metadata.resolve()
    gate_path = args.gate.resolve()
    comparison = _load(comparison_path)
    baseline = _load(baseline_path)
    candidate = _load(candidate_path)
    training = _load(training_path)
    gate = _load(gate_path)
    integrity = _verify_artifact_binding(
        comparison_path, comparison, baseline, candidate, training, gate
    )
    result = evaluate_gate(comparison, baseline, candidate, training, gate)
    result.update({
        "integrity": integrity,
        "artifacts": {
            "comparison": str(comparison_path),
            "baselineTimeline": str(baseline_path),
            "candidateTimeline": str(candidate_path),
            "trainingMetadata": str(training_path),
            "gate": str(gate_path),
        },
        "artifactSha256": {
            "baselineTimeline": sha256_file(baseline_path),
            "candidateTimeline": sha256_file(candidate_path),
            "trainingMetadata": sha256_file(training_path),
            "gate": sha256_file(gate_path),
        },
    })
    destination = args.out.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(destination),
        "automatedResearchGatePassed": result["automatedResearchGatePassed"],
        "canProceedToListeningCanary": result["canProceedToListeningCanary"],
        "productionEligible": result["productionEligible"],
        "failedChecks": result["failedChecks"],
    }, indent=2))
    if args.fail_on_rejection and not result["automatedResearchGatePassed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
