import pytest

from ml.training.add_original_domain_supervision import build_original_domain_rows


def row(clip_id, start, *, song="target", weight=0.15):
    return {
        "clipId": clip_id,
        "songId": song,
        "sourceStart": float(start),
        "durationSeconds": 5.0,
        "sourceMedia": "/old/piano.wav",
        "audioClip": f"/old/{clip_id}.wav",
        "localAudioSource": f"C:/{clip_id}.wav",
        "exampleWeight": weight,
        "split": "train",
        "notes": [{"midi": 72, "time": 1.0, "duration": 0.5}],
    }


def test_build_original_domain_rows_copies_exact_labels_and_changes_audio_domain():
    train = [
        row("ideal-p100-000", 0),
        row("mix-p050-000", 0),
        row("ideal-p100-005", 5),
        row("unrelated", 0, song="other"),
    ]

    variants, summary = build_original_domain_rows(
        train,
        target_song_id="target",
        reference_variant_prefix="ideal-p100-",
        original_variant_prefix="original-p000-",
        source_media="/runpod-volume/original.wav",
        example_weight=0.075,
        maximum_target_training_end=10.0,
    )

    assert [item["sourceStart"] for item in variants] == [0.0, 5.0]
    assert [item["clipId"] for item in variants] == [
        "original-p000-0000000ms",
        "original-p000-0005000ms",
    ]
    assert all(item["sourceMedia"] == "/runpod-volume/original.wav" for item in variants)
    assert all(item["exampleWeight"] == 0.075 for item in variants)
    assert all("audioClip" not in item for item in variants)
    assert all("localAudioSource" not in item for item in variants)
    assert variants[0]["notes"] == train[0]["notes"]
    assert variants[0]["notes"] is not train[0]["notes"]
    assert summary["originalDomainClips"] == 2


def test_build_original_domain_rows_refuses_repeated_reference_start():
    with pytest.raises(ValueError, match="repeats sourceStart"):
        build_original_domain_rows(
            [row("ideal-p100-a", 0), row("ideal-p100-b", 0)],
            target_song_id="target",
            reference_variant_prefix="ideal-p100-",
            original_variant_prefix="original-p000-",
            source_media="/runpod-volume/original.wav",
            example_weight=0.075,
        )


def test_build_original_domain_rows_refuses_existing_variant_and_bad_tail():
    with pytest.raises(ValueError, match="already exists"):
        build_original_domain_rows(
            [row("ideal-p100-a", 0), row("original-p000-old", 0)],
            target_song_id="target",
            reference_variant_prefix="ideal-p100-",
            original_variant_prefix="original-p000-",
            source_media="/runpod-volume/original.wav",
            example_weight=0.075,
        )

    with pytest.raises(ValueError, match="extends beyond the trusted"):
        build_original_domain_rows(
            [row("ideal-p100-late", 147.5)],
            target_song_id="target",
            reference_variant_prefix="ideal-p100-",
            original_variant_prefix="original-p000-",
            source_media="/runpod-volume/original.wav",
            example_weight=0.075,
            maximum_target_training_end=150.0,
        )
