import random

from ml.training.audit_piano_reference_support import maximum_time_matches, support_audit
from ml.training.evaluate_piano_arranger import greedy_matches


def note(time, midi=60):
    return {"time": time, "midi": midi, "duration": .2, "velocity": .75}


def brute_match_count(reference, observed, tolerance):
    def recurse(index, used):
        if index == len(reference):
            return 0
        best = recurse(index + 1, used)
        for right, candidate in enumerate(observed):
            if right not in used and reference[index]["midi"] == candidate["midi"] and abs(reference[index]["time"] - candidate["time"]) <= tolerance + 1e-12:
                best = max(best, 1 + recurse(index + 1, used | {right}))
        return best
    return recurse(0, set())


def test_nearest_pairing_can_undercount_dense_repeats():
    target = [note(.1), note(.18)]
    observed = [note(.01), note(.11)]
    assert len(greedy_matches(target, observed, .1)) == 1
    assert len(maximum_time_matches(target, observed, .1)) == 2


def test_maximum_matches_agree_with_exhaustive_small_bipartite_search():
    randomizer = random.Random(105)
    for _ in range(100):
        target = [note(randomizer.randrange(10) / 10, randomizer.choice((60, 64))) for _ in range(5)]
        observed = [note(randomizer.randrange(10) / 10, randomizer.choice((60, 64))) for _ in range(5)]
        assert len(maximum_time_matches(target, observed, .1)) == brute_match_count(target, observed, .1)


def test_octave_oracle_is_explicit_and_one_to_one():
    assert maximum_time_matches([note(0, 60)], [note(0, 72)], .1) == []
    assert len(maximum_time_matches([note(0, 60), note(0, 72)], [note(0, 84)], .1, octave_equivalent=True)) == 1


def test_diagnostic_separates_note_support_loss_and_gains_without_mutation():
    target, raw, arranged = [note(0), note(1)], [note(0)], [note(1)]
    result = support_audit(target, {"raw": raw, "arranged": arranged})
    assert result["pairedStageSupport"]["sameTargetExact100msSupportLostAfterProcessing"] == 1
    assert result["pairedStageSupport"]["sameTargetExact100msSupportGainedAfterProcessing"] == 1
    assert raw == [note(0)]
