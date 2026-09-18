from __future__ import annotations

import pytest

from ml.training.filter_prepared_manifest import filter_records


def test_filter_records_preserves_selected_records_verbatim() -> None:
    records = [
        {"clipId": "a-1", "songId": "a", "audioClip": "/runpod-volume/a.wav"},
        {"clipId": "b-1", "songId": "b", "audioClip": "/runpod-volume/b.wav"},
        {"clipId": "a-2", "songId": "a", "audioClip": "/runpod-volume/a2.wav"},
    ]

    selected, report = filter_records(records, include_song_ids={"a"})

    assert selected == [records[0], records[2]]
    assert selected[0] is records[0]
    assert report["clipsBySong"] == {"a": 2}


def test_filter_records_rejects_missing_requested_song() -> None:
    with pytest.raises(ValueError, match="absent"):
        filter_records(
            [{"clipId": "a-1", "songId": "a"}],
            include_song_ids={"missing"},
        )


def test_filter_records_rejects_empty_request() -> None:
    with pytest.raises(ValueError, match="At least one"):
        filter_records([], include_song_ids=set())


def test_filter_records_spreads_a_clip_cap_across_the_whole_song() -> None:
    records = [
        {"clipId": f"a-{index}", "songId": "a", "sourceStart": index * 5}
        for index in range(10)
    ]
    selected, report = filter_records(
        records,
        include_song_ids={"a"},
        maximum_clips_per_song=3,
    )

    assert [record["clipId"] for record in selected] == ["a-0", "a-4", "a-9"]
    assert report["clipsBySong"] == {"a": 3}
    assert report["availableClipsBySong"] == {"a": 10}
