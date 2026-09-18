from ml.training.apply_pianist_register_router import apply_router, feature_records
from ml.training.fit_pianist_register_router import FEATURE_NAMES, SHIFT_CLASSES


def test_feature_records_preserve_payload_indices_and_gesture_positions():
    payload = {
        "notes": [
            {
                "midi": 48,
                "time": 1.0,
                "duration": 0.3,
                "velocity": 0.6,
                "hand": "left",
                "arrangementRole": "bass",
                "sourceInstrument": "bass",
                "sourceMidiBeforeArrangement": 36,
            },
            {
                "midi": 60,
                "time": 1.0,
                "duration": 0.3,
                "velocity": 0.6,
                "hand": "right",
                "arrangementRole": "harmony",
                "sourceInstrument": "piano",
                "sourceMidiBeforeArrangement": 60,
            },
        ]
    }
    records, indices = feature_records(payload)
    assert indices == [0, 1]
    assert records[0]["gestureSize"] == 2
    assert records[0]["isLowest"] is True
    assert records[1]["isHighest"] is True


def test_router_can_be_scoped_to_recovered_notes_only():
    classes = {}
    for shift in SHIFT_CLASSES:
        weights = [0.0] * len(FEATURE_NAMES)
        weights[0] = 10.0 if shift == 12 else -10.0
        classes[str(shift)] = {
            "weights": weights,
            "means": [0.0] * len(FEATURE_NAMES),
            "scales": [1.0] * len(FEATURE_NAMES),
        }
    profile = {
        "schema": "polymath-pianist-register-router-v1",
        "id": "scope-test",
        "model": {
            "featureNames": list(FEATURE_NAMES),
            "classes": classes,
        },
        "policy": {
            "minimumConfidence": 0.5,
            "minimumMargin": 0.5,
            "minimumMidi": 33,
            "maximumMidi": 108,
        },
    }
    payload = {
        "notes": [
            {
                "midi": 48,
                "time": 1.0,
                "duration": 0.3,
                "velocity": 0.6,
                "hand": "left",
                "arrangementRole": "harmony",
                "sourceInstrument": "piano",
                "generatedBy": "raw-support-recovery-v1",
            },
            {
                "midi": 52,
                "time": 2.0,
                "duration": 0.3,
                "velocity": 0.6,
                "hand": "left",
                "arrangementRole": "harmony",
                "sourceInstrument": "piano",
            },
        ]
    }

    output, diagnostics = apply_router(
        payload,
        profile,
        only_generated_by="raw-support-recovery-v1",
    )

    assert output["notes"][0]["midi"] == 60
    assert output["notes"][1]["midi"] == 52
    assert diagnostics["acceptedChanges"] == 1
    assert diagnostics["skippedOutsideScope"] == 1
