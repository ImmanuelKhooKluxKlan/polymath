from ml.training.apply_authored_duration_grid import apply_duration_grid


def test_duration_grid_preserves_symbolic_events_and_rebuilds_physical_holds():
    payload = {
        "instrument": "piano",
        "bpm": 100,
        "notes": [
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
        ],
    }
    profile = {
        "decoder": {
            "authoredTwoHandStyle": {
                "durationGrid": {
                    "enabled": True,
                    "writtenReleaseRatio": 0.9,
                    "maximumUnits": 4,
                    "lowerHistogram": [{"units": 1, "share": 1.0}],
                    "upperHistogram": [{"units": 2, "share": 1.0}],
                }
            }
        }
    }

    output, diagnostics = apply_duration_grid(payload, profile)

    assert diagnostics["applied"] is True
    assert [(note["time"], note["midi"], note["velocity"]) for note in output["notes"]] == [
        (0.0, 48, 0.5),
        (0.2, 72, 0.8),
    ]
    assert all("audioDuration" in note for note in output["notes"])
    assert output["pianoArrangement"]["durationGridResearchOnly"] is True
