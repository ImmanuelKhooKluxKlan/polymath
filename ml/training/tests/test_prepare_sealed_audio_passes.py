from __future__ import annotations

import json
from pathlib import Path, PurePosixPath

import pytest

from ml.training.prepare_sealed_audio_passes import (
    SealedAudioPreparationError,
    build_records,
    prepare_sealed_audio_passes,
)
from ml.training.materialize_sealed_assets import materialize_assets
from ml.training.sealed_exam_custody import begin_audio
from ml.training.tests.test_sealed_exam_custody import fixture


def materialize_fake_audio(paths: dict[str, Path]) -> None:
    selection = json.loads((paths["repo"] / "selection.json").read_text())
    for song in selection["songs"]:
        if song["split"] == "test":
            (paths["dataset"] / song["audioFilename"]).write_bytes(b"source-audio")


def fake_renderer(_ffmpeg: Path, _record: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(b"R" * 64)


def fake_downloader(**kwargs) -> None:
    root = Path(kwargs["local_dir"])
    for relative in kwargs["allow_patterns"]:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"source-audio")


def test_build_records_is_label_free_and_shifted(tmp_path: Path) -> None:
    audio = tmp_path / "song.wav"
    audio.write_bytes(b"audio")
    songs = [{
        "songId": "sealed-song",
        "audioFilename": "song.wav",
        "durationSeconds": 12.0,
    }]
    records = build_records(songs, tmp_path, "overlap", 2.5, 5.0)
    assert [row["sourceStart"] for row in records] == [2.5, 7.5]
    assert all(row["split"] == "test" for row in records)
    assert all(row["labelsPresent"] is False and row["notes"] == [] for row in records)


def test_prepares_both_passes_only_after_audio_authorization(tmp_path: Path) -> None:
    paths = fixture(tmp_path)
    # begin-audio normally proves files absent immediately before download. The
    # fixture therefore authorizes first, then simulates the audio-only fetch.
    begin_audio(paths["plan"], paths["blind"], paths["repo"], paths["custody"])
    materialize_assets(
        plan_path=paths["plan"], repo_root=paths["repo"],
        custody_dir=paths["custody"], dataset_root=paths["dataset"],
        kind="audio", downloader=fake_downloader,
    )
    result = prepare_sealed_audio_passes(
        plan_path=paths["plan"],
        repo_root=paths["repo"],
        custody_dir=paths["custody"],
        dataset_root=paths["dataset"],
        output_root=tmp_path / "pack",
        remote_root=PurePosixPath("/remote/sealed"),
        ffmpeg=Path("fake-ffmpeg"),
        renderer=fake_renderer,
    )
    assert result["testSongs"] == 8
    assert result["labelsPresent"] is False
    assert result["passes"]["primary"]["clips"] == 16
    assert result["passes"]["overlap"]["clips"] == 16


def test_refuses_to_prepare_when_any_midi_label_is_present(tmp_path: Path) -> None:
    paths = fixture(tmp_path)
    begin_audio(paths["plan"], paths["blind"], paths["repo"], paths["custody"])
    materialize_assets(
        plan_path=paths["plan"], repo_root=paths["repo"],
        custody_dir=paths["custody"], dataset_root=paths["dataset"],
        kind="audio", downloader=fake_downloader,
    )
    (paths["dataset"] / "test-0.midi").write_bytes(b"label")
    with pytest.raises(SealedAudioPreparationError, match="MIDI labels are present"):
        prepare_sealed_audio_passes(
            plan_path=paths["plan"], repo_root=paths["repo"],
            custody_dir=paths["custody"], dataset_root=paths["dataset"],
            output_root=tmp_path / "pack", remote_root=PurePosixPath("/remote"),
            ffmpeg=Path("fake"), renderer=fake_renderer,
        )


def test_refuses_without_custody_authorization(tmp_path: Path) -> None:
    paths = fixture(tmp_path)
    materialize_fake_audio(paths)
    with pytest.raises(Exception):
        prepare_sealed_audio_passes(
            plan_path=paths["plan"], repo_root=paths["repo"],
            custody_dir=paths["custody"], dataset_root=paths["dataset"],
            output_root=tmp_path / "pack", remote_root=PurePosixPath("/remote"),
            ffmpeg=Path("fake"), renderer=fake_renderer,
        )
