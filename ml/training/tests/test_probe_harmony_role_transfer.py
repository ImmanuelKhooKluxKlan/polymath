import copy

from ml.training.probe_harmony_role_transfer import gates, playback_signature


def test_observability_ignores_metadata_but_detects_actual_playback_change():
    baseline = {"notes": [{"midi": 60, "time": 1, "duration": .5, "source": "old"}]}
    candidate = copy.deepcopy(baseline)
    candidate["notes"][0]["source"] = "new"
    assert playback_signature(baseline) == playback_signature(candidate)
    candidate["notes"][0]["midi"] = 72
    assert playback_signature(baseline) != playback_signature(candidate)


def metrics():
    return {"exactPitchOnset100ms": {"f1": .3, "recall": .4}, "exactPitchOnset250ms": {"f1": .4, "recall": .5}, "physicalDuration": {"medianAbsoluteErrorSeconds": .2, "severeCutoffRate": .1}, "rapidRetriggersUnder100ms": 0}


def test_unchanged_control_passes_but_extra_duplicates_fail():
    assert all(gates(metrics(), metrics(), 0).values())
    assert not gates(metrics(), metrics(), 1)["noDuplicatePitchOnsets"]


def test_note_gain_cannot_hide_hold_or_recall_regression():
    before, after = metrics(), copy.deepcopy(metrics())
    after["exactPitchOnset100ms"]["f1"] += .1
    after["exactPitchOnset250ms"]["recall"] -= .001
    after["physicalDuration"]["severeCutoffRate"] += .03
    checks = gates(before, after, 0)
    assert not checks["250ms_recall_not_worse"]
    assert not checks["physicalCutoffRateNotWorse"]
