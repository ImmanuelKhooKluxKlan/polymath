import pytest

from ml.training.create_variant_weight_ablation import reweight_variant


def row(clip_id, weight=1.0):
    return {
        "clipId": clip_id,
        "songId": "song",
        "audioClip": f"/runpod-volume/{clip_id}.wav",
        "exampleWeight": weight,
        "notes": [{"midi": 60, "time": 0.1, "duration": 0.2}],
    }


def test_reweight_variant_preserves_order_identity_audio_and_labels():
    source = [row("ordinary"), row("p000-a", 0.075), row("later"), row("p000-b", 0.075)]

    output, summary = reweight_variant(
        source,
        clip_prefix="p000-",
        replacement_weight=1e-12,
    )

    assert [item["clipId"] for item in output] == [item["clipId"] for item in source]
    assert [item["audioClip"] for item in output] == [item["audioClip"] for item in source]
    assert [item["notes"] for item in output] == [item["notes"] for item in source]
    assert [item["exampleWeight"] for item in output] == [1.0, 1e-12, 1.0, 1e-12]
    assert source[1]["exampleWeight"] == 0.075
    assert summary["matchingRowIndicesZeroBased"] == [1, 3]
    assert summary["recordOrderPreserved"] is True


def test_reweight_variant_requires_match_and_positive_weight():
    with pytest.raises(ValueError, match="No clip IDs matched"):
        reweight_variant([row("ordinary")], clip_prefix="p000-", replacement_weight=1e-12)

    with pytest.raises(ValueError, match="within"):
        reweight_variant([row("p000-a")], clip_prefix="p000-", replacement_weight=0)


def test_variant_weight_can_increase_original_audio_exposure_without_duplication():
    source = [row("other", 0.1), row("p000-a", 0.075), row("p000-b", 0.075)]
    output, summary = reweight_variant(source, clip_prefix="p000-", replacement_weight=0.5)
    assert len(output) == len(source)
    assert summary["weightMass"]["selectedBefore"] == pytest.approx(0.15)
    assert summary["weightMass"]["selectedAfter"] == pytest.approx(1.0)
    assert summary["weightMass"]["manifestBefore"] == pytest.approx(0.25)
    assert summary["weightMass"]["manifestAfter"] == pytest.approx(1.1)
    assert [item["notes"] for item in output] == [item["notes"] for item in source]
    assert [item["audioClip"] for item in output] == [item["audioClip"] for item in source]
