from ml.training.apply_paired_gesture_duration import apply_paired_duration
from server.piano_gesture_calibration import GESTURE_DURATION_FEATURE_NAMES


def test_paired_duration_operator_preserves_notes_and_rebuilds_holds():
    notes = [
        {
            "time": 0.0,
            "midi": 48,
            "note": "C3",
            "duration": 0.8,
            "scoreDuration": 0.8,
            "audioDuration": 1.2,
            "velocity": 0.5,
            "hand": "left",
            "voice": "bass",
            "arrangementRole": "bass",
        },
        {
            "time": 0.2,
            "midi": 72,
            "note": "C5",
            "duration": 0.7,
            "scoreDuration": 0.7,
            "audioDuration": 1.0,
            "velocity": 0.8,
            "hand": "right",
            "voice": "melody",
            "arrangementRole": "melody",
        },
    ]
    payload = {"instrument": "piano", "bpm": 100, "notes": notes}
    feature_count = len(GESTURE_DURATION_FEATURE_NAMES)
    profile = {
        "decoder": {
            "gestureDynamics": {
                "enabled": True,
                "onsetWindowSeconds": 0.035,
                "pairedDurationCalibration": {
                    "enabled": True,
                    "id": "test-duration",
                    "featureNames": list(GESTURE_DURATION_FEATURE_NAMES),
                    "weights": [-0.2] + [0.0] * (feature_count - 1),
                    "means": [0.0] * feature_count,
                    "scales": [1.0] * feature_count,
                    "blend": 1.0,
                    "maximumLogCorrection": 0.5,
                },
            }
        }
    }

    output, diagnostics = apply_paired_duration(payload, profile)

    assert diagnostics["applied"] is True
    assert diagnostics["changedNotes"] == 2
    assert [(note["time"], note["midi"], note["velocity"]) for note in output["notes"]] == [
        (0.0, 48, 0.5),
        (0.2, 72, 0.8),
    ]
    assert all("audioDuration" in note for note in output["notes"])
    assert output["pianoArrangement"]["pairedDurationResearchOnly"] is True
