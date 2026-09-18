from ml.training.train_pianist_cyclic_arpeggio_adapter import (
    apply_profile,
    detect_cycle_offset,
    evaluation_gate,
    fit_profile,
    monotonic_onset_matches,
    nearest_anchor,
    predicted_group,
    sha256_json,
)


def note(midi, time=0.0, instrument="clean_electric_guitar", velocity=0.7):
    return {
        "midi": midi,
        "note": "x",
        "time": time,
        "duration": 0.2,
        "velocity": velocity,
        "instrument": instrument,
    }


def test_nearest_anchor_prefers_same_pitch_class():
    anchor, rank = nearest_anchor([50, 51, 58, 62, 67], 74 - 12)
    assert anchor == 62
    assert rank == 3


def test_cycle_offset_tracks_a_human_timing_shift_from_raw_input():
    template = [
        [note(48, 1.0)],
        [note(55, 1.25)],
        [note(60, 1.5)],
        [note(64, 1.75)],
    ]
    repeated = [
        [note(48, 5.85)],
        [note(55, 6.10)],
        [note(60, 6.35)],
        [note(64, 6.60)],
    ]
    offset, evidence, _ = detect_cycle_offset(
        template + repeated,
        template_start=1.0,
        duration=1.0,
        expected_offset=5.0,
        search_radius=0.25,
    )
    assert abs(offset - 4.85) <= 0.01
    assert evidence["meanPitchClassF1"] == 1.0


def test_cycle_offset_can_cross_fit_an_earlier_occurrence():
    earlier = [
        [note(48, 1.0)],
        [note(55, 1.25)],
        [note(60, 1.5)],
        [note(64, 1.75)],
    ]
    template = [
        [note(48, 5.85)],
        [note(55, 6.10)],
        [note(60, 6.35)],
        [note(64, 6.60)],
    ]
    offset, evidence, _ = detect_cycle_offset(
        earlier + template,
        template_start=5.85,
        duration=1.0,
        expected_offset=-5.0,
        search_radius=0.25,
    )
    assert abs(offset - (-4.85)) <= 0.01
    assert evidence["meanPitchClassF1"] == 1.0


def test_onset_matcher_skips_an_extra_existing_strike():
    generated = [1.00, 1.30, 1.60]
    existing = [0.99, 1.29, 1.45, 1.61]
    assert monotonic_onset_matches(
        generated, existing, maximum_distance=0.12
    ) == [(0, 0), (1, 1), (2, 3)]


def test_gate_accepts_safe_expressive_gain_without_inventing_exact_chords():
    baseline = {
        "coverageAdjustedPitchClassF1": 0.80,
        "coverageAdjustedExactPitchClassRate": 0.65,
        "coverageAdjustedOccupancyAccuracy": 0.70,
        "onsetMaeSeconds": 0.02,
        "gestureVelocityMae": 0.10,
        "exactKeyDurationMaeSeconds": 0.20,
        "sequenceScore": 4.0,
    }
    candidate = {
        "coverageAdjustedPitchClassF1": 0.82,
        "coverageAdjustedExactPitchClassRate": 0.65,
        "coverageAdjustedOccupancyAccuracy": 0.80,
        "onsetMaeSeconds": 0.02,
        "gestureVelocityMae": 0.07,
        "exactKeyDurationMaeSeconds": 0.08,
        "sequenceScore": 3.0,
    }
    passed, basis = evaluation_gate(baseline, candidate)
    assert passed is True
    assert basis == "safe-expressive-improvement"


def test_persistent_pitch_class_is_retained_when_destination_supports_it():
    gesture = {
        "pitchClasses": [2, 10],
        "selectors": [
            {
                "sourceRank": 2,
                "sourceDeltaSemitones": 12,
                "originalPitchClass": 10,
                "originalMidi": 70,
                "note": note(70),
            },
            {
                "sourceRank": 3,
                "sourceDeltaSemitones": 12,
                "originalPitchClass": 2,
                "originalMidi": 74,
                "note": note(74),
            },
        ],
    }
    result = predicted_group(
        gesture,
        [49, 51, 58, 61, 63, 67, 77],
        persistent_pitch_classes={10},
        destination_cycle_pitch_classes={1, 3, 5, 10},
        equivalent_pitch_classes=None,
    )
    assert {item["midi"] % 12 for item in result} == {1, 10}


def test_equivalent_phase_reuses_pitch_class_identity():
    gesture = {
        "pitchClasses": [2, 10],
        "selectors": [
            {
                "sourceRank": 0,
                "sourceDeltaSemitones": 12,
                "originalPitchClass": 10,
                "originalMidi": 70,
                "note": note(70),
            }
        ],
    }
    result = predicted_group(
        gesture,
        [53, 60],
        persistent_pitch_classes={10},
        destination_cycle_pitch_classes={0, 5, 10},
        equivalent_pitch_classes={1, 10},
    )
    assert {item["midi"] % 12 for item in result} == {1, 10}


def test_confidence_guard_keeps_candidate_when_source_cycle_is_empty():
    profile = {
        "id": "test",
        "cycle": {
            "templateStartSeconds": 1.0,
            "durationSeconds": 1.0,
            "persistentPitchClasses": [],
            "gestures": [
                {
                    "relativeTime": 0.0,
                    "pitchClassSignature": "0",
                    "pitchClasses": [0],
                    "velocity": 0.7,
                    "selectors": [],
                }
            ],
        },
        "sourceSupport": {"lagSeconds": 0.0, "radiusSeconds": 0.1},
        "profileSha256": "",
    }
    profile["profileSha256"] = sha256_json(profile)
    candidate = {"notes": [note(60, 2.0)]}
    output, diagnostics = apply_profile(
        candidate, profile, {"notes": []}, expected_offset_seconds=0.75
    )
    assert diagnostics["applied"] is False
    assert diagnostics["expectedOffsetSource"] == "explicit"
    assert diagnostics["expectedOffsetSeconds"] == 0.75
    assert output["notes"] == candidate["notes"]


def test_source_window_overrides_must_be_paired():
    reference = [[note(60, 0.0)], [note(62, 0.2)], [note(64, 0.4)], [note(65, 0.6)]]
    source = [[note(48, 0.0)], [note(50, 0.2)], [note(52, 0.4)], [note(53, 0.6)]]
    try:
        fit_profile(
            reference,
            source,
            profile_id="test",
            song_id="song",
            template_start=0.0,
            cycle_duration=0.8,
            source_to_output_semitones=12,
            source_lag_seconds=0.0,
        )
    except ValueError as error:
        assert "must be supplied together" in str(error)
    else:  # pragma: no cover - protects the validation contract.
        raise AssertionError("Expected a paired-source-window validation error")


def test_explicit_source_window_records_validation_provenance():
    reference = [
        [note(60, 0.0)],
        [note(62, 0.2)],
        [note(64, 0.4)],
        [note(65, 0.6)],
    ]
    source = [
        [note(48, 0.0)],
        [note(50, 0.2)],
        [note(52, 0.4)],
        [note(53, 0.6)],
    ]
    profile = fit_profile(
        reference,
        source,
        profile_id="test",
        song_id="song",
        template_start=0.0,
        cycle_duration=0.8,
        source_to_output_semitones=12,
        source_lag_seconds=0.0,
        source_radius_seconds=0.1,
    )
    assert profile["sourceSupport"]["selectionUsedDestinationTarget"] is True
    assert (
        profile["sourceSupport"]["selectionProvenance"]
        == "explicit-validation-setting"
    )
