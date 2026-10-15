from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml.training.relabel_prepared_composition import relabel


def package(status: str = "trusted") -> dict:
    return {
        "schema": "polymath-supervision-package-v1",
        "alignment": {
            "qualityWindows": [
                {"id": "w1", "sourceStart": 0.0, "sourceEnd": 5.0, "status": status}
            ]
        },
        "notes": [
            {
                "midi": 60,
                "time": 1.0,
                "duration": 0.5,
                "velocity": 0.8,
                "instrument": "acoustic_piano",
                "qualityWindowId": "w1",
                "qualityStatus": status,
                "trainingEligible": status == "trusted",
            }
        ],
    }


def record(song_id: str, *, negative: bool = False) -> dict:
    return {
        "clipId": f"{song_id}-0",
        "songId": song_id,
        "sourceStart": 0.0,
        "durationSeconds": 5.0,
        "instrumentFocus": "acoustic_piano",
        "audioClip": "/runpod-volume/existing.wav",
        "notes": [] if negative else [{"midi": 61, "time": 0.0, "duration": 0.1}],
        "targetState": "reviewed-silence" if negative else "notes",
        "isNegativeExample": negative,
        "exampleWeight": 0.35 if negative else 1.0,
    }


def test_relabels_only_target_and_reuses_audio(tmp_path: Path) -> None:
    package_path = tmp_path / "labels.json"
    package_path.write_text(json.dumps(package()), encoding="utf-8")
    untouched = record("other")
    result, summary = relabel(
        [record("target"), untouched],
        package(),
        target_song_id="target",
        package_path=package_path,
    )
    assert summary["relabelledClips"] == 1
    assert result[0]["audioClip"] == "/runpod-volume/existing.wav"
    assert result[0]["notes"][0]["midi"] == 60
    assert result[1] == untouched


def test_visually_confirmed_notes_convert_old_negative_to_positive(tmp_path: Path) -> None:
    package_path = tmp_path / "labels.json"
    package_path.write_text(json.dumps(package()), encoding="utf-8")
    result, _summary = relabel(
        [record("target", negative=True)],
        package(),
        target_song_id="target",
        package_path=package_path,
    )
    assert result[0]["targetState"] == "notes"
    assert result[0]["isNegativeExample"] is False
    assert result[0]["exampleWeight"] == 1.0
    assert result[0]["relabeling"]["previousNegativeExample"] is True


def test_target_weight_multiplier_preserves_domain_bridge_ratio(tmp_path: Path) -> None:
    package_path = tmp_path / "labels.json"
    package_path.write_text(json.dumps(package()), encoding="utf-8")
    source = record("target")
    source["domainBridge"] = {"weightMultiplier": 0.5}
    result, summary = relabel(
        [source],
        package(),
        target_song_id="target",
        package_path=package_path,
        target_weight_multiplier=0.15,
    )
    assert result[0]["exampleWeight"] == pytest.approx(0.075)
    assert summary["targetWeightMultiplier"] == 0.15


def test_fails_closed_on_review_window(tmp_path: Path) -> None:
    package_path = tmp_path / "labels.json"
    package_path.write_text(json.dumps(package("review")), encoding="utf-8")
    with pytest.raises(ValueError, match="unapproved"):
        relabel(
            [record("target")],
            package("review"),
            target_song_id="target",
            package_path=package_path,
        )
