from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml.training.materialize_sealed_assets import SealedAssetError, materialize_assets
from ml.training.sealed_exam_custody import authorize_labels, begin_audio, lock_predictions
from ml.training.tests.test_sealed_exam_custody import fixture, inference_artifacts


def fake_downloader(**kwargs) -> None:
    root = Path(kwargs["local_dir"])
    for relative in kwargs["allow_patterns"]:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(("asset:" + relative).encode("utf-8"))


def add_mirror(paths: dict[str, Path]) -> None:
    selection_path = paths["repo"] / "selection.json"
    selection = json.loads(selection_path.read_text())
    selection["mirror"] = {"repoId": "example/maestro"}
    selection_path.write_text(json.dumps(selection) + "\n")
    from ml.training.rescore_song_timelines import sha256_file
    plan = json.loads(paths["plan"].read_text())
    plan["selection"]["manifestSha256"] = sha256_file(selection_path)
    paths["plan"].write_text(json.dumps(plan) + "\n")


def test_audio_stage_fetches_only_audio_and_writes_receipt(tmp_path: Path) -> None:
    paths = fixture(tmp_path)
    add_mirror(paths)
    begin_audio(paths["plan"], paths["blind"], paths["repo"], paths["custody"])
    result = materialize_assets(
        plan_path=paths["plan"], repo_root=paths["repo"],
        custody_dir=paths["custody"], dataset_root=paths["dataset"],
        kind="audio", downloader=fake_downloader,
    )
    assert result["fileCount"] == 8
    assert result["assetClass"] == "audio-only"
    assert not list(paths["dataset"].glob("*.midi"))
    assert (paths["custody"] / "01A-AUDIO-MATERIALIZED.json").is_file()
    with pytest.raises(SealedAssetError, match="already materialized"):
        materialize_assets(
            plan_path=paths["plan"], repo_root=paths["repo"],
            custody_dir=paths["custody"], dataset_root=paths["dataset"],
            kind="audio", downloader=fake_downloader,
        )


def test_labels_cannot_be_fetched_before_predictions_lock(tmp_path: Path) -> None:
    paths = fixture(tmp_path)
    add_mirror(paths)
    begin_audio(paths["plan"], paths["blind"], paths["repo"], paths["custody"])
    with pytest.raises(Exception):
        materialize_assets(
            plan_path=paths["plan"], repo_root=paths["repo"],
            custody_dir=paths["custody"], dataset_root=paths["dataset"],
            kind="labels", downloader=fake_downloader,
        )


def test_labels_fetch_only_after_locked_prediction_hashes(tmp_path: Path) -> None:
    paths = fixture(tmp_path)
    add_mirror(paths)
    begin_audio(paths["plan"], paths["blind"], paths["repo"], paths["custody"])
    artifacts = inference_artifacts(paths)
    lock_predictions(
        paths["plan"], paths["repo"], paths["custody"],
        artifacts["primary_manifest"], artifacts["overlap_manifest"],
        artifacts["primary_eval"], artifacts["overlap_eval"], artifacts["candidate"],
    )
    authorize_labels(paths["plan"], paths["repo"], paths["custody"])
    result = materialize_assets(
        plan_path=paths["plan"], repo_root=paths["repo"],
        custody_dir=paths["custody"], dataset_root=paths["dataset"],
        kind="labels", downloader=fake_downloader,
    )
    assert result["fileCount"] == 8
    assert result["predictionsLockedBeforeMaterialization"] is True
    assert len(list(paths["dataset"].glob("*.midi"))) == 8


def test_prediction_mutation_blocks_label_fetch(tmp_path: Path) -> None:
    paths = fixture(tmp_path)
    add_mirror(paths)
    begin_audio(paths["plan"], paths["blind"], paths["repo"], paths["custody"])
    artifacts = inference_artifacts(paths)
    lock_predictions(
        paths["plan"], paths["repo"], paths["custody"],
        artifacts["primary_manifest"], artifacts["overlap_manifest"],
        artifacts["primary_eval"], artifacts["overlap_eval"], artifacts["candidate"],
    )
    authorize_labels(paths["plan"], paths["repo"], paths["custody"])
    artifacts["candidate"].write_text("{}\n")
    with pytest.raises(Exception, match="changed"):
        materialize_assets(
            plan_path=paths["plan"], repo_root=paths["repo"],
            custody_dir=paths["custody"], dataset_root=paths["dataset"],
            kind="labels", downloader=fake_downloader,
        )
