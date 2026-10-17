import copy

import pytest

from ml.training.rebuild_prepared_audio_variants import merge_rendered_rows, rebuild_rows


def row(clip_id):
    return {
        "clipId": clip_id, "songId": "song", "sourceStart": 5.0,
        "durationSeconds": 5.0, "exampleWeight": 0.15, "split": "train",
        "audioClip": f"/old/{clip_id}.wav", "localAudioSource": f"D:/{clip_id}.wav",
        "notes": [{"midi": 72, "time": 0.5, "duration": 0.2}],
        "relabeling": {"replacementSupervisionSha256": "a" * 64},
    }


def test_audio_rebuild_preserves_labels_weights_and_training_order():
    original = [row("unrelated"), row("p100-a"), row("p075-a")]
    selected, counts = rebuild_rows(
        original, {"p100-": "/new/piano.wav", "p075-": "/new/mix.wav"},
        label_source_sha256="a" * 64,
    )
    assert counts == {"p100-": 1, "p075-": 1}
    assert all("audioClip" not in r and "localAudioSource" not in r for r in selected)
    rendered = [{**r, "audioClip": f"/new/{r['clipId']}.wav"} for r in reversed(selected)]
    merged = merge_rendered_rows(original, rendered)
    assert [r["clipId"] for r in merged] == [r["clipId"] for r in original]
    assert [r["notes"] for r in merged] == [r["notes"] for r in original]
    assert [r["exampleWeight"] for r in merged] == [r["exampleWeight"] for r in original]
    assert merged[0] == original[0]
    assert merged[1]["sourceMedia"] == "/new/piano.wav"


def test_audio_rebuild_rejects_label_source_mismatch_and_ambiguous_prefix():
    with pytest.raises(ValueError, match="label source does not match"):
        rebuild_rows([row("p100-a")], {"p100-": "/new.wav"}, label_source_sha256="b" * 64)
    with pytest.raises(ValueError, match="Ambiguous"):
        rebuild_rows([row("p100-a")], {"p100-": "/new.wav", "p": "/other.wav"}, label_source_sha256="a" * 64)


def test_merge_rejects_accidental_label_or_weight_changes():
    original = [row("p100-a")]
    edited = copy.deepcopy(original)
    edited[0]["notes"][0]["midi"] = 73
    with pytest.raises(ValueError, match="frozen field notes"):
        merge_rendered_rows(original, edited)
    edited = copy.deepcopy(original)
    edited[0]["exampleWeight"] = 1.0
    with pytest.raises(ValueError, match="frozen field exampleWeight"):
        merge_rendered_rows(original, edited)
