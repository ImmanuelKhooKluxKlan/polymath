from ml.training.collapse_direct_piano_duplicates import (
    collapse_direct_piano_duplicates,
)


def note(midi: int, time: float, duration: float = 0.2, velocity: float = 0.7):
    return {
        "midi": midi,
        "time": time,
        "duration": duration,
        "velocity": velocity,
        "instrument": "acoustic_piano",
    }


def test_merges_only_sub_65ms_same_pitch_fragments():
    cleaned, report = collapse_direct_piano_duplicates(
        [note(60, 1.0, 0.1), note(60, 1.04, 0.3, 0.8)]
    )

    assert len(cleaned) == 1
    assert cleaned[0]["time"] == 1.0
    assert cleaned[0]["duration"] == 0.34
    assert cleaned[0]["velocity"] == 0.8
    assert report["duplicatesRemoved"] == 1


def test_preserves_65ms_and_90ms_musical_repeats():
    cleaned, report = collapse_direct_piano_duplicates(
        [note(60, 1.0), note(60, 1.065), note(60, 1.155)]
    )

    assert len(cleaned) == 3
    assert report["duplicatesRemoved"] == 0
    assert report["minimumPreservedFastGapSeconds"] == 0.065


def test_never_merges_different_pitches():
    cleaned, report = collapse_direct_piano_duplicates(
        [note(60, 1.0), note(64, 1.0), note(67, 1.01)]
    )

    assert [item["midi"] for item in cleaned] == [60, 64, 67]
    assert report["duplicatesRemoved"] == 0


def test_rejects_a_window_that_would_erase_fast_music():
    try:
        collapse_direct_piano_duplicates([note(60, 0.0)], duplicate_seconds=0.1)
    except ValueError as error:
        assert "0.035 and 0.075" in str(error)
    else:
        raise AssertionError("Expected an unsafe duplicate window to be rejected")
