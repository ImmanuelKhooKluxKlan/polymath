from ml.training.search_checkpoint_consensus_loso import (
    consensus_notes,
    one_to_one_matches,
)


def note(midi: int, time: float, duration: float = 0.2) -> dict:
    return {
        "midi": midi,
        "time": time,
        "duration": duration,
        "visualDuration": duration,
        "audioDuration": duration,
        "velocity": 0.7,
    }


def test_one_to_one_matches_does_not_reuse_repeated_notes() -> None:
    left = [note(60, 1.0), note(60, 1.08)]
    right = [note(60, 1.03), note(60, 1.11)]
    assert one_to_one_matches(left, right, 0.06) == [(0, 0), (1, 1)]


def test_consensus_blends_only_when_both_checkpoints_support_event() -> None:
    incumbent = [note(60, 1.0), note(62, 2.0)]
    checkpoint_a = [note(60, 1.1), note(62, 2.1)]
    checkpoint_b = [note(60, 1.2)]
    output, diagnostics = consensus_notes(
        incumbent,
        checkpoint_a,
        checkpoint_b,
        match_radius_seconds=0.25,
        onset_blend=0.5,
        require_both_for_onset=True,
        add_corroborated_missing=False,
        corroboration_radius_seconds=0.05,
        incumbent_exclusion_radius_seconds=0.1,
    )
    assert output[0]["time"] == 1.075
    assert output[1]["time"] == 2.0
    assert diagnostics["shiftedIncumbentNotes"] == 1


def test_consensus_adds_only_two_checkpoint_missing_event() -> None:
    incumbent = [note(60, 1.0)]
    checkpoint_a = [note(60, 1.01), note(64, 2.0), note(67, 3.0)]
    checkpoint_b = [note(60, 0.99), note(64, 2.03)]
    output, diagnostics = consensus_notes(
        incumbent,
        checkpoint_a,
        checkpoint_b,
        match_radius_seconds=0.08,
        onset_blend=0.0,
        require_both_for_onset=True,
        add_corroborated_missing=True,
        corroboration_radius_seconds=0.05,
        incumbent_exclusion_radius_seconds=0.1,
    )
    assert [(item["midi"], round(item["time"], 3)) for item in output] == [
        (60, 1.0),
        (64, 2.015),
    ]
    assert diagnostics["corroboratedAdditions"] == 1


def test_consensus_does_not_add_event_covered_by_incumbent() -> None:
    incumbent = [note(64, 2.0)]
    checkpoint_a = [note(64, 2.12)]
    checkpoint_b = [note(64, 2.14)]
    output, diagnostics = consensus_notes(
        incumbent,
        checkpoint_a,
        checkpoint_b,
        match_radius_seconds=0.05,
        onset_blend=0.0,
        require_both_for_onset=True,
        add_corroborated_missing=True,
        corroboration_radius_seconds=0.05,
        incumbent_exclusion_radius_seconds=0.15,
    )
    assert len(output) == 1
    assert diagnostics["corroboratedAdditions"] == 0
