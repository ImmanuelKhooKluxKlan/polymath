from ml.training.evaluate_raw_support_recovery_loso import duration_gate, weighted_delta


def test_weighted_delta_uses_reference_note_counts():
    rows = [
        {"evaluation": {"referenceNotes": 100, "deltas": {"quality": 0.1}}},
        {"evaluation": {"referenceNotes": 300, "deltas": {"quality": -0.1}}},
    ]
    assert weighted_delta(rows, "quality") == -0.05


def test_duration_gate_uses_existing_ten_percent_and_cutoff_tolerances():
    baseline = {
        "durationMedianAbsoluteErrorSeconds": 0.2,
        "visualSevereCutoffRate": 0.1,
        "physicalDurationMedianAbsoluteErrorSeconds": 0.2,
        "physicalSevereCutoffRate": 0.1,
        "rapidRetriggersUnder100ms": 10,
    }
    candidate = {
        "durationMedianAbsoluteErrorSeconds": 0.22,
        "visualSevereCutoffRate": 0.11,
        "physicalDurationMedianAbsoluteErrorSeconds": 0.22,
        "physicalSevereCutoffRate": 0.11,
        "rapidRetriggersUnder100ms": 11,
    }
    assert all(duration_gate(baseline, candidate).values())
