import hashlib
import json

import pytest

from ml.training.full_song_target_development import load_config, score_payload


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def note(midi, time):
    return {"midi": midi, "time": time, "duration": 0.5, "velocity": 0.7}


def test_full_song_score_ignores_training_windows_and_preserves_context(tmp_path):
    audio = tmp_path / "song.wav"
    audio.write_bytes(b"RIFF-frozen-audio")
    reference = tmp_path / "reference.json"
    alignment = tmp_path / "alignment.json"
    baseline = tmp_path / "baseline.json"
    receipt = tmp_path / "split.json"
    reference_hash = write_json(reference, {"notes": [note(60, 2), note(64, 12)]})
    alignment_hash = write_json(alignment, {"anchors": [], "metrics": {}})
    baseline_hash = write_json(baseline, {"notes": [note(60, 2), note(64, 12)]})
    receipt_hash = write_json(receipt, {"developmentStarts": [10]})
    config_path = tmp_path / "config.json"
    write_json(config_path, {
        "schema": "polymath-full-song-target-development-v1",
        "audio": {"path": str(audio), "sha256": hashlib.sha256(audio.read_bytes()).hexdigest()},
        "baselineRaw": {"path": str(baseline), "sha256": baseline_hash},
        "reference": {"path": str(reference), "sha256": reference_hash},
        "alignmentReport": {"path": str(alignment), "sha256": alignment_hash},
        "splitReceipt": {"path": str(receipt), "sha256": receipt_hash},
        "referenceAlreadyAligned": True,
        "windows": [{"start": 10, "end": 15}, {"start": 20, "end": 25}],
    })

    config = load_config(config_path)
    score = score_payload({"notes": [note(60, 2), note(64, 12)]}, config)

    assert score["referenceNotes"] == 1
    assert score["predictedNotes"] == 1
    assert score["100ms"]["microF1"] == 1.0


def test_config_rejects_overlapping_windows(tmp_path):
    config = tmp_path / "config.json"
    write_json(config, {
        "schema": "polymath-full-song-target-development-v1",
        "windows": [{"start": 0, "end": 5}, {"start": 4, "end": 6}],
    })
    with pytest.raises(ValueError, match="cannot overlap"):
        load_config(config)
