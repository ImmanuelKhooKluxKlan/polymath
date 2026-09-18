import json
from pathlib import Path

import pytest

from ml.training.compose_linked_prepared_dataset import compose


def write_manifest(path: Path, *records: dict) -> Path:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return path


def record(clip: str, song: str, remote: str) -> dict:
    return {
        "clipId": clip,
        "songId": song,
        "split": "stale",
        "audioClip": remote,
        "durationSeconds": 5,
    }


def test_composes_remote_audio_without_copying(tmp_path: Path) -> None:
    (tmp_path / "audio" / "train").mkdir(parents=True)
    (tmp_path / "audio" / "train" / "a-1.wav").write_bytes(b"RIFF-not-a-real-test-wave")
    train = write_manifest(
        tmp_path / "train.jsonl",
        record("a-1", "a", "/runpod-volume/training/base/a.wav"),
    )
    validation = write_manifest(
        tmp_path / "validation.jsonl",
        record("b-1", "b", "/runpod-volume/training/hard/b.wav"),
    )
    output, summary = compose([train], [validation])
    assert output["train"][0]["split"] == "train"
    assert output["train"][0]["localAudioSource"].endswith("a-1.wav")
    assert output["validation"][0]["split"] == "validation"
    assert summary["splits"]["train"]["clips"] == 1
    assert summary["splits"]["validation"]["songs"] == ["b"]


def test_rejects_song_leakage(tmp_path: Path) -> None:
    train = write_manifest(
        tmp_path / "train.jsonl",
        record("a-1", "same", "/runpod-volume/training/base/a.wav"),
    )
    validation = write_manifest(
        tmp_path / "validation.jsonl",
        record("a-2", "same", "/runpod-volume/training/base/b.wav"),
    )
    with pytest.raises(ValueError, match="song leakage"):
        compose([train], [validation])


def test_rejects_local_audio_path(tmp_path: Path) -> None:
    train = write_manifest(tmp_path / "train.jsonl", record("a-1", "a", "C:/a.wav"))
    validation = write_manifest(
        tmp_path / "validation.jsonl",
        record("b-1", "b", "/runpod-volume/training/base/b.wav"),
    )
    with pytest.raises(ValueError, match="/runpod-volume"):
        compose([train], [validation])
