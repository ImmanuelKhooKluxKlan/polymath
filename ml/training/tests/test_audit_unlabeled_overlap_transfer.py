import json
import wave
from pathlib import Path

from ml.training.audit_unlabeled_overlap_transfer import audit_transfer, compare_notes


def note(midi: int, time: float, duration: float = 0.4) -> dict:
    return {
        "midi": midi,
        "time": time,
        "duration": duration,
        "velocity": 0.75,
        "instrument": "acoustic_piano",
    }


def payload(notes: list[dict], duration: float = 10.0) -> dict:
    return {"durationSeconds": duration, "notes": notes}


def write_wav(path: Path, frames: int = 800, value: int = 1000) -> None:
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(2)
        writer.setsampwidth(2)
        writer.setframerate(8000)
        writer.writeframes(int(value).to_bytes(2, "little", signed=True) * 2 * frames)


def plan() -> dict:
    return {
        "candidateVersion": "test",
        "matching": {
            "unchangedOnsetToleranceSeconds": 0.000001,
            "unchangedDurationToleranceSeconds": 0.000001,
            "samePitchMovedPairToleranceSeconds": 0.1,
        },
        "requiredChecks": {
            "invalidPrimaryNotesMaximum": 0,
            "invalidCandidateNotesMaximum": 0,
            "changedOnsetBoundaryViolationsMaximum": 0,
            "boundaryRadiusSeconds": 0.6,
            "maximumPairedOnsetShiftSeconds": 0.1,
            "candidateNoteCountRatioMinimum": 0.5,
            "candidateNoteCountRatioMaximum": 2.0,
            "rapidRetriggers75msPer1000MaximumIncrease": 1000,
            "rapidRetriggers100msMaximum": "primary * 1.05 + 1",
            "durationP95RatioMaximum": 1.1,
            "notesLongerThan2sMaximum": "primary * 1.10 + 2",
            "maximumCandidateDurationSeconds": "max(primary maximum + 0.10, 5.0)",
            "renderDurationMismatchFramesMaximum": 0,
            "absoluteRmsDifferenceDbMaximum": 0.5,
            "absolutePeakMaximum": 0.98,
            "clippedSamplesMaximum": 0,
        },
    }


def test_compare_notes_distinguishes_unchanged_shifted_removed_and_added() -> None:
    primary = [
        {**note(64, 2.0), "index": 0},
        {**note(60, 4.9, 0.6), "index": 1},
        {**note(67, 5.2), "index": 2},
    ]
    candidate = [
        {**note(64, 2.0), "index": 0},
        {**note(60, 4.95, 0.55), "index": 1},
        {**note(69, 5.1), "index": 2},
    ]
    result = compare_notes(
        primary,
        candidate,
        unchanged_onset_tolerance=1e-6,
        unchanged_duration_tolerance=1e-6,
        moved_pair_tolerance=0.1,
        boundary_radius=0.6,
    )
    assert result["unchangedPairs"] == 1
    assert result["movedPairs"] == 1
    assert result["removedPrimary"] == 1
    assert result["addedCandidate"] == 1
    assert result["changedOnsetBoundaryViolations"] == 0
    assert result["maximumPairedOnsetShiftSeconds"] == 0.05
    assert result["maximumPairedReleaseShiftSeconds"] == 0.0


def test_audit_passes_boundary_scoped_safe_transfer(tmp_path: Path) -> None:
    first = tmp_path / "a.wav"
    second = tmp_path / "b.wav"
    write_wav(first)
    write_wav(second)
    primary = payload([note(64, 2.0), note(60, 4.9, 0.6), note(67, 5.2)])
    candidate = payload([note(64, 2.0), note(60, 4.95, 0.55), note(69, 5.1)])
    result = audit_transfer(primary, candidate, first, second, plan())
    assert result["structuralSafetyGatePassed"] is True
    assert result["failedChecks"] == []
    assert result["blindMappingInspected"] is False
    assert result["accuracyEvidenceAllowed"] is False


def test_audit_rejects_change_far_from_boundary(tmp_path: Path) -> None:
    first = tmp_path / "a.wav"
    second = tmp_path / "b.wav"
    write_wav(first)
    write_wav(second)
    primary = payload([note(64, 2.0)])
    candidate = payload([note(64, 2.05)])
    result = audit_transfer(primary, candidate, first, second, plan())
    assert result["structuralSafetyGatePassed"] is False
    assert result["failedChecks"] == ["changed_onsets_boundary_scoped"]


def test_audio_loudness_and_length_are_gated(tmp_path: Path) -> None:
    first = tmp_path / "a.wav"
    second = tmp_path / "b.wav"
    write_wav(first, frames=800, value=1000)
    write_wav(second, frames=799, value=4000)
    result = audit_transfer(payload([note(60, 1.0)]), payload([note(60, 1.0)]), first, second, plan())
    assert result["checks"]["render_duration"] is False
    assert result["checks"]["render_rms"] is False
    assert result["structuralSafetyGatePassed"] is False
