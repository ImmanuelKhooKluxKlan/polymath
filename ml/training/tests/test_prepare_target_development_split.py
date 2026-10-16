import pytest

from ml.training.prepare_target_development_split import (
    build_split,
    intervals_overlap,
)


def row(clip_id, song, start, *, prefix=False):
    return {
        "clipId": ("ideal-" if prefix else "mix-") + clip_id,
        "songId": song,
        "sourceStart": float(start),
        "durationSeconds": 5.0,
        "audioClip": f"/runpod-volume/{clip_id}.wav",
        "localAudioSource": f"C:/{clip_id}.wav",
        "instrumentFocus": "acoustic_piano",
        "exampleWeight": 0.1,
        "notes": [{"midi": 72, "time": 1.0, "duration": 0.5}],
        "split": "train",
    }


def test_half_open_intervals_allow_touching_boundaries():
    assert intervals_overlap(0, 5, 2.5, 5) is True
    assert intervals_overlap(0, 5, 5, 5) is False


def test_build_split_removes_every_overlapping_audio_variant():
    train = [row("other", "other-song", 0)]
    for start in (0, 2.5, 5, 7.5, 10):
        train.append(row(f"ref-{start}", "target", start, prefix=True))
        train.append(row(f"bridge-{start}", "target", start))

    kept, development, summary = build_split(
        train,
        [row("validation", "validation-song", 0)],
        target_song_id="target",
        reference_variant_prefix="ideal-",
        target_development_song_id="target-original-development",
        source_media="/runpod-volume/original.wav",
        development_starts=[5.0, 10.0],
    )

    assert [item["clipId"] for item in kept] == ["mix-other", "ideal-ref-0", "mix-bridge-0"]
    assert [item["sourceStart"] for item in development] == [5.0, 10.0]
    assert all(item["sourceMedia"] == "/runpod-volume/original.wav" for item in development)
    assert all("audioClip" not in item for item in development)
    assert summary["removedOverlappingClips"] == 8
    assert summary["temporalOverlapCountAfterFiltering"] == 0


def test_build_split_refuses_missing_reference_start():
    with pytest.raises(ValueError, match="missing development starts"):
        build_split(
            [row("ref-0", "target", 0, prefix=True)],
            [row("validation", "validation-song", 0)],
            target_song_id="target",
            reference_variant_prefix="ideal-",
            target_development_song_id="target-original-development",
            source_media="/runpod-volume/original.wav",
            development_starts=[0.0, 5.0],
        )


def test_build_split_enforces_trusted_label_cutoff_only_for_target_song():
    train = [row("other-late", "other-song", 200)]
    for start in (0, 5, 145, 147.5, 150):
        train.append(row(f"ref-{start}", "target", start, prefix=True))
        train.append(row(f"bridge-{start}", "target", start))

    kept, development, summary = build_split(
        train,
        [row("validation", "validation-song", 0)],
        target_song_id="target",
        reference_variant_prefix="ideal-",
        target_development_song_id="target-original-development",
        source_media="/runpod-volume/original.wav",
        development_starts=[0.0, 145.0],
        maximum_target_training_end=150.0,
    )

    assert [item["clipId"] for item in kept] == [
        "mix-other-late",
        "ideal-ref-5",
        "mix-bridge-5",
    ]
    assert [item["sourceStart"] for item in development] == [0.0, 145.0]
    assert summary["removedOverlappingClips"] == 4
    assert summary["removedAfterTrustedCutoffClips"] == 4
    assert summary["removedAfterTrustedCutoffSourceStarts"] == [147.5, 150.0]
    assert summary["maximumTargetTrainingEnd"] == 150.0


def test_build_split_refuses_development_window_after_trusted_cutoff():
    with pytest.raises(ValueError, match="extends beyond the trusted"):
        build_split(
            [
                row("ref-0", "target", 0, prefix=True),
                row("ref-147.5", "target", 147.5, prefix=True),
            ],
            [row("validation", "validation-song", 0)],
            target_song_id="target",
            reference_variant_prefix="ideal-",
            target_development_song_id="target-original-development",
            source_media="/runpod-volume/original.wav",
            development_starts=[0.0, 147.5],
            maximum_target_training_end=150.0,
        )
