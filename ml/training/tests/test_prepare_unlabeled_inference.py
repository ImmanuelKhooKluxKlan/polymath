import json
import wave
from pathlib import Path, PurePosixPath

import pytest

from ml.training.prepare_unlabeled_inference import (
    UnlabeledInferenceError,
    prepare_unlabeled_inference,
    window_starts,
)


def write_silence(path: Path, seconds: float, sample_rate: int = 16000) -> None:
    frames = round(seconds * sample_rate)
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(b"\x00\x00" * frames)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_window_starts_keeps_short_final_window() -> None:
    assert window_starts(12.2, 5.0, 0.0) == [0.0, 5.0, 10.0]
    assert window_starts(12.2, 5.0, 2.5) == [2.5, 7.5]


def test_prepare_unlabeled_pack_has_two_hashable_passes(tmp_path: Path) -> None:
    source = tmp_path / "song.wav"
    output = tmp_path / "pack"
    write_silence(source, 12.2)
    receipt = prepare_unlabeled_inference(
        source,
        output,
        PurePosixPath("/workspace/opened/kiss-me"),
        song_id="kiss-me",
    )
    assert receipt["labelsPresent"] is False
    assert receipt["accuracyEvidenceAllowed"] is False
    assert receipt["passes"]["primary"]["clips"] == 3
    assert receipt["passes"]["overlap"]["clips"] == 2
    primary = read_jsonl(output / "primary" / "manifest.jsonl")
    overlap = read_jsonl(output / "overlap" / "manifest.jsonl")
    assert [record["sourceStart"] for record in primary] == [0.0, 5.0, 10.0]
    assert [record["sourceStart"] for record in overlap] == [2.5, 7.5]
    assert primary[-1]["durationSeconds"] == 2.2
    assert overlap[-1]["durationSeconds"] == 4.7
    assert all(record["notes"] == [] for record in [*primary, *overlap])
    assert all(record["labelsPresent"] is False for record in [*primary, *overlap])
    assert primary[0]["audioClip"].startswith("/workspace/opened/kiss-me/primary/audio/")
    for pass_name in ("primary", "overlap"):
        for artifact in receipt["passes"][pass_name]["files"]:
            assert artifact["sha256"]


def test_prepare_refuses_to_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "song.wav"
    output = tmp_path / "pack"
    write_silence(source, 1.0)
    output.mkdir()
    with pytest.raises(UnlabeledInferenceError, match="Refusing to overwrite"):
        prepare_unlabeled_inference(
            source,
            output,
            PurePosixPath("/workspace/opened/song"),
            song_id="song",
        )


def test_prepare_requires_canonical_wave_format(tmp_path: Path) -> None:
    source = tmp_path / "song.wav"
    output = tmp_path / "pack"
    write_silence(source, 1.0, sample_rate=8000)
    with pytest.raises(UnlabeledInferenceError, match="mono 16 kHz"):
        prepare_unlabeled_inference(
            source,
            output,
            PurePosixPath("/workspace/opened/song"),
            song_id="song",
        )
