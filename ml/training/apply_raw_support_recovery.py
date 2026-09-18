"""Conservatively supplement an arranged piano score from raw pitched support.

This decoder does not rebuild or replace the listener-approved arrangement.
It scores raw low-register events with a frozen selector, collapses duplicate
stem evidence by pitch class/onset, and adds only high-confidence support that
is absent from the arranged score.  Per-onset caps and collision checks keep
the experiment from restoring the dense full-mix "machine gun" texture.
"""

from __future__ import annotations

import argparse
import bisect
import copy
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:  # Package and direct-script execution.
    from server.piano_arranger_adapter import (
        instrument_family,
        normalize_source_notes,
        selection_scores,
    )
except ImportError:  # pragma: no cover
    import sys

    REPO_ROOT = Path(__file__).resolve().parents[2]
    SERVER_ROOT = REPO_ROOT / "server"
    if str(SERVER_ROOT) not in sys.path:
        sys.path.insert(0, str(SERVER_ROOT))
    from piano_arranger_adapter import (  # type: ignore
        instrument_family,
        normalize_source_notes,
        selection_scores,
    )


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def selector_model(profile: dict[str, Any]) -> dict[str, Any]:
    nested = (
        (profile.get("decoder") or {})
        .get("leftHandAccompaniment", {})
        .get("selectionModel")
    )
    if isinstance(nested, dict) and nested:
        return nested
    direct = profile.get("selectionModel")
    if isinstance(direct, dict) and direct:
        return direct
    raise ValueError("The profile contains no usable selection model.")


def nearby(times: list[float], time: float, radius: float) -> bool:
    position = bisect.bisect_left(times, time)
    return any(
        0 <= index < len(times) and abs(times[index] - time) <= radius
        for index in (position - 1, position)
    )


def fold_midi(midi: int, minimum: int, maximum: int, shift: int) -> int:
    value = int(midi) + int(shift)
    while value < minimum:
        value += 12
    while value > maximum:
        value -= 12
    return max(minimum, min(maximum, value))


def within_recovery_register(
    family: str,
    midi: int,
    maximum_source_midi: int,
    maximum_piano_source_midi: int | None,
) -> bool:
    if int(midi) > int(maximum_source_midi):
        return False
    return not (
        family == "piano"
        and maximum_piano_source_midi is not None
        and int(midi) > int(maximum_piano_source_midi)
    )


def deduplicate_proposals(
    proposals: list[dict[str, Any]], cluster_seconds: float
) -> list[dict[str, Any]]:
    """Keep one strongest source event per nearby pitch-class cluster."""

    selected: list[dict[str, Any]] = []
    for pitch_class in range(12):
        values = sorted(
            (item for item in proposals if int(item["midi"]) % 12 == pitch_class),
            key=lambda item: (float(item["time"]), -float(item["score"])),
        )
        cluster: list[dict[str, Any]] = []
        cluster_start = -999.0
        for item in values:
            time = float(item["time"])
            if cluster and time - cluster_start > cluster_seconds:
                selected.append(max(cluster, key=lambda row: float(row["score"])))
                cluster = []
            if not cluster:
                cluster_start = time
            cluster.append(item)
        if cluster:
            selected.append(max(cluster, key=lambda row: float(row["score"])))
    return sorted(selected, key=lambda item: (float(item["time"]), int(item["midi"])))


def supplement_score(
    candidate: dict[str, Any],
    source: dict[str, Any],
    profile: dict[str, Any],
    *,
    threshold: float,
    coverage_radius: float,
    cluster_seconds: float,
    onset_window: float,
    maximum_additions_per_onset: int,
    maximum_source_midi: int,
    register_shift: int,
    minimum_output_midi: int,
    maximum_output_midi: int,
    gesture_anchor_radius: float = 0.0,
    maximum_piano_source_midi: int | None = 52,
    style_velocity_blend: float = 0.75,
    style_duration_blend: float = 0.65,
) -> tuple[dict[str, Any], dict[str, Any]]:
    output = copy.deepcopy(candidate)
    output_notes = output.get("notes")
    if not isinstance(output_notes, list):
        raise ValueError("Candidate has no notes list.")
    raw_notes = normalize_source_notes(source.get("notes") or [])
    model = selector_model(profile)
    scores = selection_scores(raw_notes, {"selectionModel": model})
    candidate_times_by_pitch_class: dict[int, list[float]] = defaultdict(list)
    exact_times_by_midi: dict[int, list[float]] = defaultdict(list)
    candidate_onset_times: list[float] = []
    for note in output_notes:
        try:
            midi = int(round(float(note.get("midi", note.get("pitch")))))
            time = float(note.get("time", note.get("startTime", note.get("start"))))
        except (TypeError, ValueError):
            continue
        candidate_times_by_pitch_class[midi % 12].append(time)
        exact_times_by_midi[midi].append(time)
        candidate_onset_times.append(time)
    for values in candidate_times_by_pitch_class.values():
        values.sort()
    for values in exact_times_by_midi.values():
        values.sort()
    candidate_onset_times.sort()

    proposals: list[dict[str, Any]] = []
    rejected_voice = rejected_register = rejected_piano_register = 0
    rejected_score = rejected_covered = 0
    for note, score in zip(raw_notes, scores):
        family = instrument_family(str(note.get("instrument") or ""))
        if family == "voice":
            rejected_voice += 1
            continue
        if not within_recovery_register(
            family,
            int(note["midi"]),
            maximum_source_midi,
            maximum_piano_source_midi,
        ):
            if (
                family == "piano"
                and maximum_piano_source_midi is not None
                and int(note["midi"]) > maximum_piano_source_midi
            ):
                rejected_piano_register += 1
            rejected_register += 1
            continue
        if float(score) < threshold:
            rejected_score += 1
            continue
        if nearby(
            candidate_times_by_pitch_class[int(note["midi"]) % 12],
            float(note["time"]),
            coverage_radius,
        ):
            rejected_covered += 1
            continue
        proposals.append({**note, "score": float(score), "family": family})
    deduplicated = deduplicate_proposals(proposals, cluster_seconds)

    onset_buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in deduplicated:
        onset_buckets[int(round(float(item["time"]) / onset_window))].append(item)
    accepted: list[dict[str, Any]] = []
    rejected_cap = rejected_collision = rejected_unanchored = 0
    family_counts: Counter[str] = Counter()
    roles = profile.get("roles") or {}
    for bucket in sorted(onset_buckets):
        ranked = sorted(
            onset_buckets[bucket],
            key=lambda item: (
                float(item["score"]),
                float(item.get("velocity", 0.0)),
                float(item.get("duration", 0.0)),
            ),
            reverse=True,
        )
        for position, item in enumerate(ranked):
            if position >= maximum_additions_per_onset:
                rejected_cap += 1
                continue
            time = float(item["time"])
            if gesture_anchor_radius > 0 and not nearby(
                candidate_onset_times, time, gesture_anchor_radius
            ):
                rejected_unanchored += 1
                continue
            midi = fold_midi(
                int(item["midi"]),
                minimum_output_midi,
                maximum_output_midi,
                register_shift,
            )
            if nearby(exact_times_by_midi[midi], time, cluster_seconds):
                rejected_collision += 1
                continue
            family = str(item["family"])
            role = "bass" if family == "bass" else "harmony"
            role_style = roles.get(role) or {}
            source_duration = max(0.05, float(item.get("duration", 0.32)))
            duration_scale = max(0.25, float(role_style.get("durationScale", 1.0)))
            predicted_duration = source_duration * duration_scale
            target_duration = max(
                0.05, float(role_style.get("medianDuration", predicted_duration))
            )
            duration = (1.0 - style_duration_blend) * predicted_duration + (
                style_duration_blend * target_duration
            )
            duration = max(0.14, min(1.10, duration))
            source_velocity = max(0.05, min(1.0, float(item.get("velocity", 0.72))))
            target_velocity = max(
                0.05, min(1.0, float(role_style.get("velocity", source_velocity)))
            )
            velocity = (1.0 - style_velocity_blend) * source_velocity + (
                style_velocity_blend * target_velocity
            )
            velocity = max(0.20, min(0.95, velocity))
            note = {
                "midi": midi,
                "time": round(time, 6),
                "duration": round(duration, 6),
                "scoreDuration": round(duration, 6),
                "audioDuration": round(max(duration, min(1.35, duration + 0.14)), 6),
                "velocity": round(velocity, 6),
                "hand": "left",
                "arrangementRole": role,
                "sourceInstrument": item.get("instrument"),
                "sourceIndex": item.get("sourceIndex"),
                "sourceMidiBeforeArrangement": int(item["midi"]),
                "generatedBy": "raw-support-recovery-v1",
                "rawSupportProbability": round(float(item["score"]), 6),
            }
            accepted.append(note)
            exact_times_by_midi[midi].append(time)
            exact_times_by_midi[midi].sort()
            family_counts[family] += 1
    output_notes.extend(accepted)
    output_notes.sort(
        key=lambda item: (
            float(item.get("time", item.get("startTime", item.get("start", 0.0)))),
            int(round(float(item.get("midi", item.get("pitch", 0))))),
        )
    )
    diagnostics = {
        "profile": profile.get("id"),
        "threshold": threshold,
        "coverageRadiusSeconds": coverage_radius,
        "clusterSeconds": cluster_seconds,
        "onsetWindowSeconds": onset_window,
        "maximumAdditionsPerOnset": maximum_additions_per_onset,
        "gestureAnchorRadiusSeconds": round(gesture_anchor_radius, 4),
        "maximumPianoSourceMidi": maximum_piano_source_midi,
        "styleVelocityBlend": round(style_velocity_blend, 4),
        "styleDurationBlend": round(style_duration_blend, 4),
        "rawNotes": len(raw_notes),
        "qualifiedBeforeCoverage": len(proposals) + rejected_covered,
        "uncoveredProposals": len(proposals),
        "deduplicatedProposals": len(deduplicated),
        "acceptedAdditions": len(accepted),
        "acceptedByFamily": dict(family_counts),
        "rejectedVoice": rejected_voice,
        "rejectedRegister": rejected_register,
        "rejectedPianoRegister": rejected_piano_register,
        "rejectedBelowThreshold": rejected_score,
        "rejectedAlreadyCovered": rejected_covered,
        "rejectedOnsetCap": rejected_cap,
        "rejectedWithoutGestureAnchor": rejected_unanchored,
        "rejectedCollision": rejected_collision,
        "warning": "Calibration/research operator; promotion requires whole-song transfer and sealed listening evidence.",
    }
    output.setdefault("diagnostics", {})["rawSupportRecovery"] = diagnostics
    return output, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", default="")
    parser.add_argument("--threshold", type=float, default=0.93)
    parser.add_argument("--coverage-radius-seconds", type=float, default=0.12)
    parser.add_argument("--cluster-seconds", type=float, default=0.08)
    parser.add_argument("--onset-window-seconds", type=float, default=0.10)
    parser.add_argument("--maximum-additions-per-onset", type=int, default=2)
    parser.add_argument(
        "--gesture-anchor-radius-seconds",
        type=float,
        default=0.0,
        help=(
            "When positive, recover only notes near an existing arranged onset. "
            "This completes gestures without inventing unsupported new attacks."
        ),
    )
    parser.add_argument("--maximum-source-midi", type=int, default=59)
    parser.add_argument(
        "--maximum-piano-source-midi",
        type=int,
        default=52,
        help=(
            "Stricter ceiling for piano-labelled recovery evidence. Low piano "
            "events can support the left hand; upper repeated stem events are excluded."
        ),
    )
    parser.add_argument(
        "--register-shift-semitones",
        type=int,
        default=0,
        help=(
            "Octave-only research override for recovered notes. The default keeps "
            "the source register and folds only notes outside the playable range."
        ),
    )
    parser.add_argument("--minimum-output-midi", type=int, default=33)
    parser.add_argument("--maximum-output-midi", type=int, default=71)
    parser.add_argument("--style-velocity-blend", type=float, default=0.75)
    parser.add_argument("--style-duration-blend", type=float, default=0.65)
    args = parser.parse_args()
    if int(args.register_shift_semitones) % 12:
        parser.error("--register-shift-semitones must be an octave multiple")
    output, diagnostics = supplement_score(
        load_json(Path(args.candidate).resolve()),
        load_json(Path(args.source).resolve()),
        load_json(Path(args.profile).resolve()),
        threshold=max(0.0, min(1.0, float(args.threshold))),
        coverage_radius=max(0.01, float(args.coverage_radius_seconds)),
        cluster_seconds=max(0.01, float(args.cluster_seconds)),
        onset_window=max(0.01, float(args.onset_window_seconds)),
        maximum_additions_per_onset=max(1, int(args.maximum_additions_per_onset)),
        maximum_source_midi=int(args.maximum_source_midi),
        register_shift=int(args.register_shift_semitones),
        minimum_output_midi=int(args.minimum_output_midi),
        maximum_output_midi=int(args.maximum_output_midi),
        gesture_anchor_radius=max(0.0, float(args.gesture_anchor_radius_seconds)),
        maximum_piano_source_midi=int(args.maximum_piano_source_midi),
        style_velocity_blend=max(0.0, min(1.0, float(args.style_velocity_blend))),
        style_duration_blend=max(0.0, min(1.0, float(args.style_duration_blend))),
    )
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    if args.report:
        report_path = Path(args.report).resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path), **diagnostics}, indent=2))


if __name__ == "__main__":
    main()
