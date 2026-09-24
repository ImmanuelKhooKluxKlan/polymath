from pathlib import Path

import mido

from ml.training.prepare_maestro_curriculum import (
    balanced_selection,
    composer_components,
    component_disjoint_selection,
    disjoint_composer_owners,
    quality_windows,
    read_midi_notes,
    selection_manifest,
)


def row(split: str, composer: str, index: int, duration: float = 120.0) -> dict:
    return {
        "split": split,
        "canonical_composer": composer,
        "canonical_title": f"Piece {index}",
        "audio_filename": f"audio-{index}.wav",
        "midi_filename": f"midi-{index}.midi",
        "duration": duration,
        "year": "2020",
    }


def test_balanced_selection_is_deterministic_and_composer_diverse() -> None:
    rows = [
        row("train", composer, index)
        for index, composer in enumerate(["A", "A", "B", "B", "C", "C"])
    ]
    first = balanced_selection(
        rows,
        split="train",
        count=3,
        seed="fixed",
        minimum_duration_seconds=90,
        maximum_duration_seconds=300,
    )
    second = balanced_selection(
        rows,
        split="train",
        count=3,
        seed="fixed",
        minimum_duration_seconds=90,
        maximum_duration_seconds=300,
    )
    assert [item["audio_filename"] for item in first] == [item["audio_filename"] for item in second]
    assert {item["canonical_composer"] for item in first} == {"A", "B", "C"}


def test_quality_windows_cover_remainder_without_gap() -> None:
    windows = quality_windows(12.25)
    assert [(item["sourceStart"], item["sourceEnd"]) for item in windows] == [
        (0.0, 5.0),
        (5.0, 10.0),
        (10.0, 12.25),
    ]


def test_disjoint_composer_owners_prioritizes_validation_breadth() -> None:
    rows = [
        row("train", "Shared A", 0),
        row("validation", "Shared A", 1),
        row("test", "Shared A", 2),
        row("train", "Shared B", 3),
        row("validation", "Shared B", 4),
        row("test", "Shared B", 5),
        row("validation", "Validation only", 6),
        row("test", "Test only", 7),
        row("train", "Train only", 8),
    ]
    owners = disjoint_composer_owners(
        rows,
        counts={"train": 1, "validation": 2, "test": 2},
        seed="fixed",
        minimum_duration_seconds=90,
        maximum_duration_seconds=300,
    )
    assert len(owners["validation"]) == 2
    assert "Validation only" in owners["validation"]
    assert len(owners["test"]) == 2
    assert "Test only" in owners["test"]
    assert owners["train"].isdisjoint(owners["validation"] | owners["test"])


def test_composite_composer_credit_cannot_leak_one_person_across_splits() -> None:
    rows = [
        row("train", "Robert Schumann / Franz Liszt", 0),
        row("train", "Train only", 1),
        row("validation", "Franz Liszt", 2),
        row("validation", "Validation only", 3),
        row("test", "Robert Schumann", 4),
        row("test", "Test only", 5),
    ]
    owners = disjoint_composer_owners(
        rows,
        counts={"train": 1, "validation": 1, "test": 1},
        seed="fixed",
        minimum_duration_seconds=90,
        maximum_duration_seconds=300,
    )
    people = {
        split: set().union(*(composer_components(credit) for credit in credits))
        if credits else set()
        for split, credits in owners.items()
    }
    assert people["train"].isdisjoint(people["validation"])
    assert people["train"].isdisjoint(people["test"])
    assert people["validation"].isdisjoint(people["test"])


def test_component_resplit_meets_counts_without_indirect_person_leakage() -> None:
    rows = []
    credits = [
        "A / B", "B / C", "D", "E", "F", "G", "H", "I", "J", "K",
    ]
    for index, credit in enumerate(credits):
        for recording in range(4):
            rows.append(row(("train", "validation", "test")[recording % 3], credit, index * 10 + recording))
    selected = component_disjoint_selection(
        rows,
        counts={"train": 8, "validation": 4, "test": 4},
        seed="fixed",
        minimum_duration_seconds=90,
        maximum_duration_seconds=300,
    )
    assert {split: len(items) for split, items in selected.items()} == {
        "train": 8,
        "validation": 4,
        "test": 4,
    }
    people = {
        split: {
            person
            for item in items
            for person in composer_components(item["canonical_composer"])
        }
        for split, items in selected.items()
    }
    assert people["train"].isdisjoint(people["validation"])
    assert people["train"].isdisjoint(people["test"])
    assert people["validation"].isdisjoint(people["test"])
    assert all(len(people[split]) >= 3 for split in people)


def test_read_midi_notes_respects_tempo_and_velocity(tmp_path: Path) -> None:
    path = tmp_path / "example.midi"
    midi = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=500_000, time=0))
    track.append(mido.Message("note_on", note=60, velocity=100, time=0))
    track.append(mido.Message("note_off", note=60, velocity=0, time=480))
    midi.save(path)
    notes = read_midi_notes(path, 2.0)
    assert len(notes) == 1
    assert notes[0]["midi"] == 60
    assert notes[0]["time"] == 0.0
    assert notes[0]["duration"] == 0.5
    assert notes[0]["velocity"] == round(100 / 127, 4)


def test_selection_manifest_excludes_previously_opened_audio(tmp_path: Path) -> None:
    metadata = tmp_path / "maestro.csv"
    metadata.write_text(
        "canonical_composer,canonical_title,split,year,midi_filename,audio_filename,duration\n"
        "A,Opened,train,2020,opened.midi,opened.wav,120\n"
        "B,Fresh train,train,2020,fresh.midi,fresh.wav,120\n"
        "C,Fresh validation,validation,2020,val.midi,val.wav,120\n"
        "D,Fresh test,test,2020,test.midi,test.wav,120\n",
        encoding="utf-8",
    )
    manifest = selection_manifest(
        metadata,
        train_count=1,
        validation_count=1,
        reserve_test_count=1,
        seed="fixed",
        minimum_duration_seconds=90,
        maximum_duration_seconds=300,
        mirror_repo="example/repo",
        excluded_audio_filenames={"opened.wav"},
    )
    assert "opened.wav" not in {song["audioFilename"] for song in manifest["songs"]}
    assert manifest["selection"]["excludedPreviouslyOpenedRecordings"] == 1


def test_selection_manifest_is_composer_disjoint_across_splits(tmp_path: Path) -> None:
    metadata = tmp_path / "maestro.csv"
    rows = [
        "canonical_composer,canonical_title,split,year,midi_filename,audio_filename,duration"
    ]
    index = 0
    for split, composers in {
        "train": ["Shared", "Train A", "Train B"],
        "validation": ["Shared", "Validation A", "Validation B"],
        "test": ["Shared", "Test A", "Test B"],
    }.items():
        for composer in composers:
            rows.append(
                f"{composer},Piece {index},{split},2020,midi-{index}.midi,audio-{index}.wav,120"
            )
            index += 1
    metadata.write_text("\n".join(rows) + "\n", encoding="utf-8")

    manifest = selection_manifest(
        metadata,
        train_count=2,
        validation_count=2,
        reserve_test_count=2,
        seed="fixed",
        minimum_duration_seconds=90,
        maximum_duration_seconds=300,
        mirror_repo="example/repo",
    )
    composers = {
        split: {song["composer"] for song in manifest["songs"] if song["split"] == split}
        for split in ("train", "validation", "test")
    }
    assert composers["train"].isdisjoint(composers["validation"])
    assert composers["train"].isdisjoint(composers["test"])
    assert composers["validation"].isdisjoint(composers["test"])
    assert manifest["selection"]["composerDisjointAcrossSelectedSplits"] is True
    assert manifest["selection"]["composerDisjointUnit"] == (
        "individual-person-within-composite-credit"
    )
