from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml.training.materialize_sealed_assets import materialize_assets
from ml.training.prepare_sealed_reference_manifest import (
    SealedReferenceError,
    prepare_reference_manifest,
)
from ml.training.sealed_exam_custody import authorize_labels, begin_audio, lock_predictions
from ml.training.tests.test_materialize_sealed_assets import fake_downloader
from ml.training.tests.test_sealed_exam_custody import fixture, inference_artifacts


def fake_midi_reader(_path: Path, _duration: float) -> list[dict]:
    return [{
        "midi": 60,
        "time": 0.1,
        "duration": 0.4,
        "velocity": 0.7,
        "instrument": "acoustic_piano",
    }]


def opened_fixture(tmp_path: Path):
    paths = fixture(tmp_path)
    begin_audio(paths["plan"], paths["blind"], paths["repo"], paths["custody"])
    artifacts = inference_artifacts(paths)
    lock_predictions(
        paths["plan"], paths["repo"], paths["custody"],
        artifacts["primary_manifest"], artifacts["overlap_manifest"],
        artifacts["primary_eval"], artifacts["overlap_eval"], artifacts["candidate"],
    )
    authorize_labels(paths["plan"], paths["repo"], paths["custody"])
    materialize_assets(
        plan_path=paths["plan"], repo_root=paths["repo"],
        custody_dir=paths["custody"], dataset_root=paths["dataset"],
        kind="labels", downloader=fake_downloader,
    )
    return paths, artifacts


def test_adds_only_labels_and_preserves_every_locked_identity(tmp_path: Path) -> None:
    paths, artifacts = opened_fixture(tmp_path)
    output = tmp_path / "references" / "test.jsonl"
    receipt = tmp_path / "references" / "receipt.json"
    result = prepare_reference_manifest(
        plan_path=paths["plan"], repo_root=paths["repo"],
        custody_dir=paths["custody"], primary_manifest_path=artifacts["primary_manifest"],
        output_manifest_path=output, receipt_path=receipt, midi_reader=fake_midi_reader,
    )
    assert result["identityPreserved"] is True
    assert result["songs"] == 8
    assert result["predictionsLockedBeforeLabels"] is True
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert all(row["labelsPresent"] is True for row in rows)
    assert all(row["notes"] for row in rows)


def test_refuses_before_label_materialization(tmp_path: Path) -> None:
    paths = fixture(tmp_path)
    begin_audio(paths["plan"], paths["blind"], paths["repo"], paths["custody"])
    artifacts = inference_artifacts(paths)
    lock_predictions(
        paths["plan"], paths["repo"], paths["custody"],
        artifacts["primary_manifest"], artifacts["overlap_manifest"],
        artifacts["primary_eval"], artifacts["overlap_eval"], artifacts["candidate"],
    )
    authorize_labels(paths["plan"], paths["repo"], paths["custody"])
    with pytest.raises(Exception):
        prepare_reference_manifest(
            plan_path=paths["plan"], repo_root=paths["repo"],
            custody_dir=paths["custody"], primary_manifest_path=artifacts["primary_manifest"],
            output_manifest_path=tmp_path / "out.jsonl", receipt_path=tmp_path / "receipt.json",
            midi_reader=fake_midi_reader,
        )


def test_refuses_changed_locked_primary_manifest(tmp_path: Path) -> None:
    paths, artifacts = opened_fixture(tmp_path)
    artifacts["primary_manifest"].write_text("{}\n", encoding="utf-8")
    with pytest.raises(Exception, match="changed|differs"):
        prepare_reference_manifest(
            plan_path=paths["plan"], repo_root=paths["repo"],
            custody_dir=paths["custody"], primary_manifest_path=artifacts["primary_manifest"],
            output_manifest_path=tmp_path / "out.jsonl", receipt_path=tmp_path / "receipt.json",
            midi_reader=fake_midi_reader,
        )


def test_refuses_midi_changed_after_materialization_receipt(tmp_path: Path) -> None:
    paths, artifacts = opened_fixture(tmp_path)
    (paths["dataset"] / "test-0.midi").write_bytes(b"tampered")
    with pytest.raises(SealedReferenceError, match="differs"):
        prepare_reference_manifest(
            plan_path=paths["plan"], repo_root=paths["repo"],
            custody_dir=paths["custody"], primary_manifest_path=artifacts["primary_manifest"],
            output_manifest_path=tmp_path / "out.jsonl", receipt_path=tmp_path / "receipt.json",
            midi_reader=fake_midi_reader,
        )
