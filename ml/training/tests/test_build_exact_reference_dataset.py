from pathlib import Path

from ml.training.build_exact_reference_dataset import build_exact_supervision_package


def test_exact_package_clips_notes_to_audio_clock(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"hashable")
    package = build_exact_supervision_package(
        song_id="piece-a",
        source_media=source,
        duration=10.0,
        provenance={"dataset": "test"},
        target_payload={
            "notes": [
                {"midi": 60, "time": 4.9, "duration": 0.4, "velocity": 100},
                {"midi": 64, "time": 9.9, "duration": 1.0, "velocity": 0.5},
                {"midi": 67, "time": 10.1, "duration": 1.0, "velocity": 0.5},
            ]
        },
    )

    assert package["schema"] == "polymath-supervision-package-v1"
    assert len(package["alignment"]["qualityWindows"]) == 2
    assert len(package["notes"]) == 2
    assert package["notes"][0]["qualityWindowId"] == "w00000"
    assert package["notes"][0]["velocity"] == round(100 / 127, 4)
    assert package["notes"][1]["duration"] == 0.1
    assert all(note["trainingEligible"] for note in package["notes"])
