from __future__ import annotations

import copy

import pytest

from ml.training.search_piano_arranger_policy import (
    gate_result,
    make_trial_profile,
    parse_grid,
)


def selector(threshold: float, offset: float = 0.0) -> dict:
    return {
        "type": "standardized-logistic-v1",
        "featureNames": ["bias", "midi"],
        "weights": [1.0 + offset, 2.0 + offset],
        "means": [0.0, 1.0],
        "scales": [1.0, 2.0],
        "threshold": threshold,
    }


def template() -> dict:
    return {
        "id": "old",
        "profileSha256": "old-hash",
        "decoder": {
            "adaptiveSourceDensity": {
                "enabled": True,
                "highDensityMultiplier": 2.0,
                "shortSourceDurationWeight": 0.92,
                "longSourceDurationWeight": 0.92,
            },
            "adaptiveSelectionBlend": {
                "enabled": True,
                "highSourceBaseShare": 0.8,
                "highSourceSelectionModel": selector(0.4),
            },
        },
    }


def test_parse_grid_deduplicates_and_rejects_empty() -> None:
    assert parse_grid("0.8, 0.9,0.8") == (0.8, 0.9)
    with pytest.raises(Exception):
        parse_grid(" , ")


def test_trial_profile_is_bounded_hashed_and_does_not_mutate_template() -> None:
    original = template()
    untouched = copy.deepcopy(original)
    result = make_trial_profile(
        original,
        base_selection_model=selector(0.50),
        other_selection_model=selector(0.30, 1.0),
        high_base_share=0.75,
        high_density=1.85,
        threshold_offset=0.02,
        quota_backfill=0.6,
        short_source_duration_weight=0.85,
        long_source_duration_weight=0.98,
        profile_id="trial",
        created_at="2026-09-12T00:00:00+00:00",
        bass_legato_bridge_seconds=0.7,
        harmony_legato_bridge_seconds=0.5,
        bass_physical_hold_seconds=1.2,
        harmony_physical_hold_seconds=1.0,
    )

    assert original == untouched
    assert result["id"] == "trial"
    assert result["decoder"]["adaptiveSourceDensity"]["highDensityMultiplier"] == 1.85
    assert result["decoder"]["adaptiveSourceDensity"]["shortSourceDurationWeight"] == 0.85
    assert result["decoder"]["adaptiveSourceDensity"]["longSourceDurationWeight"] == 0.98
    assert result["decoder"]["adaptiveSelectionBlend"]["highSourceBaseShare"] == 0.75
    assert result["decoder"]["adaptiveSelectionBlend"]["highSourceSelectionModel"]["threshold"] == pytest.approx(0.47)
    assert result["decoder"]["adaptiveSourceDensity"]["lowQuotaBackfillRatio"] == 1.0
    assert result["decoder"]["adaptiveSourceDensity"]["highQuotaBackfillRatio"] == 0.6
    assert result["decoder"]["adaptiveSourceDensity"]["nonVocalQuotaBackfillRatio"] == 1.0
    assert result["decoder"]["physicalPerformance"] == {
        "maximumLegatoBridgeSeconds": {
            "melody": 0.9,
            "bass": 0.7,
            "harmony": 0.5,
        },
        "maximumPhysicalHoldSeconds": {
            "melody": 1.35,
            "bass": 1.2,
            "harmony": 1.0,
        },
    }
    assert len(result["profileSha256"]) == 64


def metric_fixture(*, exact100: float, cutoff: float, retriggers: int = 0) -> dict:
    def score(value: float, *, recall: float | None = None) -> dict:
        return {
            "f1": value,
            "precision": value,
            "recall": value if recall is None else recall,
        }

    return {
        "exactPitchOnset50ms": score(exact100 - 0.02),
        "pitchClassOnset50ms": score(exact100 + 0.02),
        "exactPitchOnset100ms": score(exact100),
        "pitchClassOnset100ms": score(exact100 + 0.08),
        "exactPitchOnset250ms": score(exact100 + 0.10),
        "pitchClassOnset250ms": score(exact100 + 0.25),
        "duration": {"medianAbsoluteErrorSeconds": 0.14},
        "visualDuration": {"medianAbsoluteErrorSeconds": 0.12, "severeCutoffRate": cutoff},
        "physicalDuration": {"medianAbsoluteErrorSeconds": 0.12, "severeCutoffRate": cutoff},
        "rapidRetriggersUnder100ms": retriggers,
    }


def test_gate_result_requires_strict_timing_gain_and_safe_cutoffs() -> None:
    baseline = metric_fixture(exact100=0.20, cutoff=0.16)
    passing = metric_fixture(exact100=0.205, cutoff=0.17)
    failing = metric_fixture(exact100=0.204, cutoff=0.20)

    passing_gates, _ = gate_result(baseline, passing)
    failing_gates, _ = gate_result(baseline, failing)

    assert all(passing_gates.values())
    assert not failing_gates["exactF1_100ms_improves"]
    assert not failing_gates["visual_cutoff_rate_not_worse"]
    assert not failing_gates["physical_cutoff_rate_not_worse"]


def test_gate_rejects_f1_gain_that_deletes_supported_melody() -> None:
    baseline = metric_fixture(exact100=0.20, cutoff=0.16)
    candidate = metric_fixture(exact100=0.21, cutoff=0.16)
    candidate["exactPitchOnset100ms"]["recall"] = 0.18
    candidate["pitchClassOnset250ms"]["recall"] = 0.40
    baseline["pitchClassOnset250ms"]["recall"] = 0.45

    gates, _ = gate_result(baseline, candidate)

    assert not gates["exactRecall_100ms_improves"]
    assert not gates["pitchClassRecall_250ms_does_not_regress"]


def test_trial_profile_can_scope_short_holds_to_sparse_vocal_sources() -> None:
    result = make_trial_profile(
        template(),
        base_selection_model=selector(0.50),
        other_selection_model=selector(0.30, 1.0),
        high_base_share=1.0,
        high_density=1.7,
        threshold_offset=0.0,
        quota_backfill=0.65,
        short_source_duration_weight=0.85,
        long_source_duration_weight=0.92,
        profile_id="adaptive-physical",
        created_at="2026-09-12T00:00:00+00:00",
        bass_legato_bridge_seconds=0.25,
        harmony_legato_bridge_seconds=0.8,
        bass_physical_hold_seconds=2.2,
        harmony_physical_hold_seconds=2.6,
        physical_adaptation_low_nps=5.0,
        physical_adaptation_high_nps=10.0,
    )
    physical = result["decoder"]["physicalPerformance"]
    assert physical["maximumLegatoBridgeSeconds"]["bass"] == 1.8
    assert physical["adaptiveSourceDensity"]["lowMaximumLegatoBridgeSeconds"]["bass"] == 0.25
    assert physical["adaptiveSourceDensity"]["highMaximumLegatoBridgeSeconds"]["bass"] == 1.8
