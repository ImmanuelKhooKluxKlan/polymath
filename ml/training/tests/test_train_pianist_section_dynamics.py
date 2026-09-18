from ml.training.train_pianist_section_dynamics import (
    apply_profile,
    fit_profile,
    sha256_json,
)


def note(midi, time, velocity, generated_by=None):
    value = {
        "midi": midi,
        "time": time,
        "duration": 0.2,
        "velocity": velocity,
    }
    if generated_by:
        value["generatedBy"] = generated_by
    return value


def profile():
    value = {
        "id": "touch-test",
        "application": {
            "startSeconds": 1.0,
            "endSeconds": 2.0,
            "velocityByChordSize": {"1": 0.4, "2": 0.9},
        },
    }
    value["profileSha256"] = sha256_json(value)
    return value


def test_apply_sets_one_coherent_velocity_per_gesture():
    candidate = {
        "notes": [note(60, 1.2, 0.2), note(64, 1.2, 0.8), note(67, 2.1, 0.5)]
    }
    output, diagnostics = apply_profile(candidate, profile())
    assert [item["velocity"] for item in output["notes"][:2]] == [0.9, 0.9]
    assert output["notes"][2]["velocity"] == 0.5
    assert diagnostics["changedGroups"] == 1


def test_apply_freezes_prior_pianist_motifs():
    candidate = {
        "notes": [note(60, 1.2, 0.55, "pianist-repetition-adapter-v1")]
    }
    output, diagnostics = apply_profile(candidate, profile())
    assert output["notes"][0]["velocity"] == 0.55
    assert diagnostics["frozenGeneratedGroups"] == 1


def test_fit_rejects_overlapping_train_and_test_ranges():
    reference = [[note(60, 0.0, 0.6)], [note(62, 0.2, 0.7)]]
    try:
        fit_profile(
            reference,
            profile_id="x",
            song_id="song",
            training_start=0.0,
            training_end=1.0,
            application_start=0.5,
            application_end=2.0,
        )
    except ValueError as error:
        assert "Ranges must be ordered" in str(error)
    else:  # pragma: no cover
        raise AssertionError("Expected a leakage-boundary error")
