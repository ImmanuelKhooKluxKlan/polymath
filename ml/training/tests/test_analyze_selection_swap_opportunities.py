import sys
from pathlib import Path


TRAINING_DIR = Path(__file__).resolve().parents[1]
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from analyze_selection_swap_opportunities import (  # noqa: E402
    pair_local_swaps,
    reconcile_equivalent_events,
)


def note(index, midi, time, instrument="clean_electric_guitar"):
    return {
        "sourceIndex": index,
        "midi": midi,
        "time": time,
        "duration": 0.2,
        "velocity": 0.7,
        "instrument": instrument,
    }


def test_duplicate_source_event_is_not_counted_as_a_swap_error():
    notes = {0: note(0, 60, 1.0), 1: note(1, 60, 1.012)}
    missed, unsupported, matches = reconcile_equivalent_events(
        {0}, {1}, notes, radius_seconds=0.035
    )
    assert not missed
    assert not unsupported
    assert matches[0]["kind"] == "same-pitch"


def test_octave_equivalent_event_is_reconciled_but_labelled_separately():
    notes = {0: note(0, 48, 1.0), 1: note(1, 60, 1.01)}
    missed, unsupported, matches = reconcile_equivalent_events(
        {0}, {1}, notes, radius_seconds=0.035
    )
    assert not missed
    assert not unsupported
    assert matches[0]["kind"] == "same-pitch-class"


def test_local_swap_prefers_same_onset_and_same_family():
    notes = {
        0: note(0, 52, 1.0),
        1: note(1, 64, 1.01),
        2: note(2, 55, 1.06),
    }
    pairs, remaining_missed, remaining_unsupported = pair_local_swaps(
        {0}, {1, 2}, notes, radius_seconds=0.08, onset_radius_seconds=0.035
    )
    assert pairs == [(0, 1)]
    assert not remaining_missed
    assert remaining_unsupported == {2}
