"""Audit fixed overlap decoding on an opened, unlabeled song.

The audit deliberately does not read a reference score or the sealed A/B
mapping.  It checks inference invariants that remain meaningful without ground
truth: changes stay near the frozen clip boundaries, paired onset shifts stay
within the frozen matching tolerance, durations/retriggers do not explode, and
the two blind renders are technically comparable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import wave
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


class TransferAuditError(RuntimeError):
    """Raised when an immutable transfer input or audit invariant is invalid."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def boundary_distance(time_seconds: float, window_seconds: float = 5.0) -> float:
    nearest = round(time_seconds / window_seconds) * window_seconds
    return abs(time_seconds - nearest)


def normalize_notes(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    duration_seconds = float(payload.get("durationSeconds") or 0)
    notes: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    raw_notes = payload.get("notes")
    if not isinstance(raw_notes, list):
        raise TransferAuditError("Playable payload does not contain a notes list")
    for index, raw in enumerate(raw_notes):
        reason = None
        if not isinstance(raw, dict):
            invalid.append({"index": index, "reason": "not-an-object"})
            continue
        try:
            midi = int(raw.get("midi"))
            time = float(raw.get("time"))
            duration = float(raw.get("duration"))
            velocity = float(raw.get("velocity", 0.75))
        except (TypeError, ValueError):
            invalid.append({"index": index, "reason": "non-numeric-note"})
            continue
        instrument = str(raw.get("instrument") or "acoustic_piano")
        if not all(math.isfinite(value) for value in (time, duration, velocity)):
            reason = "non-finite-value"
        elif not 21 <= midi <= 108:
            reason = "outside-88-key-range"
        elif time < 0 or (duration_seconds > 0 and time > duration_seconds + 1e-6):
            reason = "onset-outside-song"
        elif duration <= 0:
            reason = "non-positive-duration"
        elif not 0 < velocity <= 1.5:
            reason = "invalid-velocity"
        if reason:
            invalid.append({"index": index, "reason": reason, "note": raw})
            continue
        notes.append({
            "index": index,
            "midi": midi,
            "time": time,
            "duration": duration,
            "velocity": velocity,
            "instrument": instrument,
        })
    notes.sort(key=lambda note: (note["time"], note["instrument"], note["midi"], note["index"]))
    return notes, invalid


def _pair_notes(
    primary: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    primary_indices: set[int],
    candidate_indices: set[int],
    *,
    onset_tolerance: float,
    duration_tolerance: float | None,
) -> list[tuple[int, int]]:
    choices: list[tuple[float, float, float, float, int, int]] = []
    for primary_index in primary_indices:
        left = primary[primary_index]
        for candidate_index in candidate_indices:
            right = candidate[candidate_index]
            if (left["instrument"], left["midi"]) != (right["instrument"], right["midi"]):
                continue
            onset_delta = abs(left["time"] - right["time"])
            duration_delta = abs(left["duration"] - right["duration"])
            if onset_delta > onset_tolerance + 1e-12:
                continue
            if duration_tolerance is not None and duration_delta > duration_tolerance + 1e-12:
                continue
            choices.append((
                onset_delta,
                duration_delta,
                left["time"],
                right["time"],
                primary_index,
                candidate_index,
            ))
    used_primary: set[int] = set()
    used_candidate: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for *_, primary_index, candidate_index in sorted(choices):
        if primary_index in used_primary or candidate_index in used_candidate:
            continue
        used_primary.add(primary_index)
        used_candidate.add(candidate_index)
        pairs.append((primary_index, candidate_index))
    return pairs


def compare_notes(
    primary: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    *,
    unchanged_onset_tolerance: float,
    unchanged_duration_tolerance: float,
    moved_pair_tolerance: float,
    boundary_radius: float,
) -> dict[str, Any]:
    primary_remaining = set(range(len(primary)))
    candidate_remaining = set(range(len(candidate)))
    unchanged = _pair_notes(
        primary,
        candidate,
        primary_remaining,
        candidate_remaining,
        onset_tolerance=unchanged_onset_tolerance,
        duration_tolerance=unchanged_duration_tolerance,
    )
    primary_remaining -= {left for left, _ in unchanged}
    candidate_remaining -= {right for _, right in unchanged}
    moved = _pair_notes(
        primary,
        candidate,
        primary_remaining,
        candidate_remaining,
        onset_tolerance=moved_pair_tolerance,
        duration_tolerance=None,
    )
    primary_remaining -= {left for left, _ in moved}
    candidate_remaining -= {right for _, right in moved}

    violations: list[dict[str, Any]] = []
    shifts: list[float] = []
    release_deltas: list[float] = []
    moved_examples: list[dict[str, Any]] = []
    for primary_index, candidate_index in moved:
        left = primary[primary_index]
        right = candidate[candidate_index]
        left_distance = boundary_distance(left["time"])
        right_distance = boundary_distance(right["time"])
        shift = right["time"] - left["time"]
        release_delta = (
            right["time"] + right["duration"]
            - left["time"] - left["duration"]
        )
        shifts.append(abs(shift))
        release_deltas.append(abs(release_delta))
        row = {
            "midi": left["midi"],
            "primaryTime": round(left["time"], 6),
            "candidateTime": round(right["time"], 6),
            "onsetShiftSeconds": round(shift, 6),
            "releaseShiftSeconds": round(release_delta, 6),
            "primaryBoundaryDistanceSeconds": round(left_distance, 6),
            "candidateBoundaryDistanceSeconds": round(right_distance, 6),
        }
        if len(moved_examples) < 20:
            moved_examples.append(row)
        if left_distance > boundary_radius + 1e-9 or right_distance > boundary_radius + 1e-9:
            violations.append({"kind": "moved-pair", **row})

    removed_examples: list[dict[str, Any]] = []
    for index in sorted(primary_remaining, key=lambda item: primary[item]["time"]):
        note = primary[index]
        distance = boundary_distance(note["time"])
        row = {
            "kind": "removed-primary",
            "midi": note["midi"],
            "time": round(note["time"], 6),
            "boundaryDistanceSeconds": round(distance, 6),
        }
        if len(removed_examples) < 20:
            removed_examples.append(row)
        if distance > boundary_radius + 1e-9:
            violations.append(row)

    added_examples: list[dict[str, Any]] = []
    for index in sorted(candidate_remaining, key=lambda item: candidate[item]["time"]):
        note = candidate[index]
        distance = boundary_distance(note["time"])
        row = {
            "kind": "added-candidate",
            "midi": note["midi"],
            "time": round(note["time"], 6),
            "boundaryDistanceSeconds": round(distance, 6),
        }
        if len(added_examples) < 20:
            added_examples.append(row)
        if distance > boundary_radius + 1e-9:
            violations.append(row)

    return {
        "unchangedPairs": len(unchanged),
        "movedPairs": len(moved),
        "removedPrimary": len(primary_remaining),
        "addedCandidate": len(candidate_remaining),
        "changedOnsetBoundaryViolations": len(violations),
        "maximumPairedOnsetShiftSeconds": round(max(shifts, default=0.0), 6),
        "p95PairedOnsetShiftSeconds": round(percentile(shifts, 0.95), 6),
        "maximumPairedReleaseShiftSeconds": round(max(release_deltas, default=0.0), 6),
        "p95PairedReleaseShiftSeconds": round(percentile(release_deltas, 0.95), 6),
        "examples": {
            "moved": moved_examples,
            "removed": removed_examples,
            "added": added_examples,
            "boundaryViolations": violations[:20],
        },
    }


def retrigger_count(notes: list[dict[str, Any]], threshold: float) -> int:
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for note in notes:
        grouped[(note["instrument"], note["midi"])].append(note["time"])
    return sum(
        current - previous <= threshold + 1e-12
        for times in grouped.values()
        for previous, current in zip(sorted(times), sorted(times)[1:])
    )


def note_summary(notes: list[dict[str, Any]]) -> dict[str, Any]:
    durations = [note["duration"] for note in notes]
    retriggers_75 = retrigger_count(notes, 0.075)
    retriggers_100 = retrigger_count(notes, 0.100)
    return {
        "notes": len(notes),
        "durationMedianSeconds": round(percentile(durations, 0.5), 6),
        "durationP95Seconds": round(percentile(durations, 0.95), 6),
        "maximumDurationSeconds": round(max(durations, default=0.0), 6),
        "notesLongerThan2Seconds": sum(duration > 2.0 for duration in durations),
        "rapidRetriggers75ms": retriggers_75,
        "rapidRetriggers75msPer1000": round(1000 * retriggers_75 / max(1, len(notes)), 6),
        "rapidRetriggers100ms": retriggers_100,
        "rapidRetriggers100msPer1000": round(1000 * retriggers_100 / max(1, len(notes)), 6),
    }


def wav_summary(path: Path) -> dict[str, Any]:
    with wave.open(str(path), "rb") as reader:
        channels = reader.getnchannels()
        sample_width = reader.getsampwidth()
        sample_rate = reader.getframerate()
        frames = reader.getnframes()
        compression = reader.getcomptype()
        raw = reader.readframes(frames)
    if sample_width != 2 or compression != "NONE":
        raise TransferAuditError(f"Expected uncompressed PCM16 WAV: {path}")
    pcm = np.frombuffer(raw, dtype="<i2").astype(np.int32)
    normalized = pcm.astype(np.float64) / 32768.0
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "channels": channels,
        "sampleWidthBytes": sample_width,
        "sampleRate": sample_rate,
        "frames": frames,
        "seconds": round(frames / sample_rate, 6),
        "peak": round(float(np.max(np.abs(normalized))) if normalized.size else 0.0, 6),
        "rms": round(float(np.sqrt(np.mean(np.square(normalized)))) if normalized.size else 0.0, 6),
        "clippedSamples": int(np.count_nonzero(np.abs(pcm) >= 32767)),
    }


def rms_difference_db(first: float, second: float) -> float:
    if first <= 0 and second <= 0:
        return 0.0
    if first <= 0 or second <= 0:
        return math.inf
    return abs(20 * math.log10(first / second))


def audit_transfer(
    primary_payload: dict[str, Any],
    candidate_payload: dict[str, Any],
    version_a_wav: Path,
    version_b_wav: Path,
    plan: dict[str, Any],
) -> dict[str, Any]:
    primary, invalid_primary = normalize_notes(primary_payload)
    candidate, invalid_candidate = normalize_notes(candidate_payload)
    matching = plan["matching"]
    limits = plan["requiredChecks"]
    differences = compare_notes(
        primary,
        candidate,
        unchanged_onset_tolerance=float(matching["unchangedOnsetToleranceSeconds"]),
        unchanged_duration_tolerance=float(matching["unchangedDurationToleranceSeconds"]),
        moved_pair_tolerance=float(matching["samePitchMovedPairToleranceSeconds"]),
        boundary_radius=float(limits["boundaryRadiusSeconds"]),
    )
    primary_summary = note_summary(primary)
    candidate_summary = note_summary(candidate)
    note_count_ratio = len(candidate) / max(1, len(primary))
    duration_p95_ratio = (
        candidate_summary["durationP95Seconds"]
        / max(0.01, primary_summary["durationP95Seconds"])
    )
    audio_a = wav_summary(version_a_wav)
    audio_b = wav_summary(version_b_wav)
    rms_db = rms_difference_db(audio_a["rms"], audio_b["rms"])
    retrigger_75_increase = (
        candidate_summary["rapidRetriggers75msPer1000"]
        - primary_summary["rapidRetriggers75msPer1000"]
    )
    checks = {
        "invalid_primary_notes": (
            len(invalid_primary) <= int(limits["invalidPrimaryNotesMaximum"])
        ),
        "invalid_candidate_notes": (
            len(invalid_candidate) <= int(limits["invalidCandidateNotesMaximum"])
        ),
        "changed_onsets_boundary_scoped": (
            differences["changedOnsetBoundaryViolations"]
            <= int(limits["changedOnsetBoundaryViolationsMaximum"])
        ),
        "paired_onset_shift": (
            differences["maximumPairedOnsetShiftSeconds"]
            <= float(limits["maximumPairedOnsetShiftSeconds"]) + 1e-9
        ),
        "note_count_ratio": (
            float(limits["candidateNoteCountRatioMinimum"])
            <= note_count_ratio
            <= float(limits["candidateNoteCountRatioMaximum"])
        ),
        "rapid_retriggers_75ms": (
            retrigger_75_increase
            <= float(limits["rapidRetriggers75msPer1000MaximumIncrease"]) + 1e-9
        ),
        "rapid_retriggers_100ms": (
            candidate_summary["rapidRetriggers100ms"]
            <= primary_summary["rapidRetriggers100ms"] * 1.05 + 1
        ),
        "duration_p95": duration_p95_ratio <= float(limits["durationP95RatioMaximum"]) + 1e-9,
        "notes_longer_than_2s": (
            candidate_summary["notesLongerThan2Seconds"]
            <= primary_summary["notesLongerThan2Seconds"] * 1.10 + 2
        ),
        "maximum_duration": (
            candidate_summary["maximumDurationSeconds"]
            <= max(primary_summary["maximumDurationSeconds"] + 0.10, 5.0) + 1e-9
        ),
        "render_duration": (
            abs(audio_a["frames"] - audio_b["frames"])
            <= int(limits["renderDurationMismatchFramesMaximum"])
        ),
        "render_rms": rms_db <= float(limits["absoluteRmsDifferenceDbMaximum"]) + 1e-9,
        "render_peak": max(audio_a["peak"], audio_b["peak"]) <= float(limits["absolutePeakMaximum"]),
        "render_clipping": (
            max(audio_a["clippedSamples"], audio_b["clippedSamples"])
            <= int(limits["clippedSamplesMaximum"])
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "schema": "polymath-unlabeled-transfer-safety-result-v1",
        "candidateVersion": plan["candidateVersion"],
        "noteComparison": differences,
        "primary": primary_summary,
        "candidate": candidate_summary,
        "derived": {
            "candidateNoteCountRatio": round(note_count_ratio, 6),
            "durationP95Ratio": round(duration_p95_ratio, 6),
            "rapidRetriggers75msPer1000Increase": round(retrigger_75_increase, 6),
            "absoluteRmsDifferenceDb": round(rms_db, 6),
            "renderDurationMismatchFrames": abs(audio_a["frames"] - audio_b["frames"]),
        },
        "audio": {"versionA": audio_a, "versionB": audio_b},
        "invalidNotes": {
            "primary": invalid_primary[:20],
            "candidate": invalid_candidate[:20],
        },
        "checks": checks,
        "failedChecks": failed,
        "structuralSafetyGatePassed": not failed,
        "blindMappingInspected": False,
        "labelsPresent": False,
        "accuracyEvidenceAllowed": False,
        "sealedTestOpened": False,
        "productionPromotionAllowed": False,
    }


def verify_input_hashes(
    plan: dict[str, Any], primary: Path, candidate: Path, version_a: Path, version_b: Path,
) -> None:
    expected = plan["immutableInputs"]
    pairs = {
        "primaryPlayableSha256": primary,
        "candidatePlayableSha256": candidate,
        "versionA_wavSha256": version_a,
        "versionB_wavSha256": version_b,
    }
    for key, path in pairs.items():
        observed = sha256_file(path)
        if observed != str(expected[key]):
            raise TransferAuditError(f"{key} mismatch: {observed}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--version-a-wav", type=Path, required=True)
    parser.add_argument("--version-b-wav", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    paths = {
        "primary": args.primary.resolve(),
        "candidate": args.candidate.resolve(),
        "versionA": args.version_a_wav.resolve(),
        "versionB": args.version_b_wav.resolve(),
        "plan": args.plan.resolve(),
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name} does not exist: {path}")
    plan = json.loads(paths["plan"].read_text(encoding="utf-8"))
    verify_input_hashes(
        plan, paths["primary"], paths["candidate"], paths["versionA"], paths["versionB"],
    )
    result = audit_transfer(
        json.loads(paths["primary"].read_text(encoding="utf-8")),
        json.loads(paths["candidate"].read_text(encoding="utf-8")),
        paths["versionA"],
        paths["versionB"],
        plan,
    )
    result["sources"] = {
        name: {"path": str(path), "sha256": sha256_file(path)}
        for name, path in paths.items()
    }
    destination = args.out.resolve()
    if destination.exists():
        raise TransferAuditError(f"Refusing to overwrite result: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(destination),
        "sha256": sha256_file(destination),
        "structuralSafetyGatePassed": result["structuralSafetyGatePassed"],
        "failedChecks": result["failedChecks"],
        "blindMappingInspected": False,
    }, indent=2))


if __name__ == "__main__":
    main()
