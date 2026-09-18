"""Fuse an instrument-focused pass into a full-mix transcription.

The full pass carries accompaniment and instrument context.  A second pass,
conditioned on one instrument family (normally ``voice``), can recover a
cleaner melody contour.  This tool replaces only that family and records the
operation; it never edits either source artifact in place.

This is an evaluation tool.  A focused pass must earn promotion on untouched
full-song evidence before the web route may enable it.
"""

from __future__ import annotations

import argparse
import bisect
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

from ml.training.decode_focused_vocal_melody import decode_focused_vocal_melody


PERCUSSION_INSTRUMENTS = {"drums", "timpani", "percussion"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_instruments(values: Iterable[str]) -> set[str]:
    return {str(value).strip().lower() for value in values if str(value).strip()}


def valid_note(note: Any) -> bool:
    if not isinstance(note, dict):
        return False
    try:
        midi = int(round(float(note.get("midi", note.get("pitch")))))
        onset = float(note.get("time", note.get("startTime", note.get("start"))))
        duration = float(note.get("duration", 0.2))
    except (TypeError, ValueError):
        return False
    return (
        0 <= midi <= 127
        and onset >= 0
        and duration > 0
        and math.isfinite(onset)
        and math.isfinite(duration)
    )


def note_instrument(note: dict[str, Any]) -> str:
    return str(note.get("instrument") or "").strip().lower()


def note_pitch_key(note: dict[str, Any], pitch_mode: str) -> int:
    midi = int(round(float(note.get("midi", note.get("pitch")))))
    return midi % 12 if pitch_mode == "pitch-class" else midi


def time_index(
    notes: list[dict[str, Any]], pitch_mode: str
) -> dict[int, list[float]]:
    indexed: dict[int, list[float]] = {}
    for note in notes:
        indexed.setdefault(note_pitch_key(note, pitch_mode), []).append(
            float(note["time"])
        )
    for times in indexed.values():
        times.sort()
    return indexed


def has_nearby_support(
    note: dict[str, Any],
    indexed_times: dict[int, list[float]],
    tolerance_seconds: float,
    pitch_mode: str,
) -> bool:
    times = indexed_times.get(note_pitch_key(note, pitch_mode), [])
    if not times:
        return False
    onset = float(note["time"])
    position = bisect.bisect_left(times, onset)
    return any(
        0 <= candidate < len(times)
        and abs(times[candidate] - onset) <= tolerance_seconds
        for candidate in (position - 1, position)
    )


def payload_duration(notes: list[dict[str, Any]]) -> float:
    return max(
        (float(note.get("time", 0.0)) + float(note.get("duration", 0.0)) for note in notes),
        default=0.0,
    )


def applied_onset_delay(payload: dict[str, Any]) -> float:
    for container_name in ("diagnostics", "transcriptionDiagnostics", "beatGrid"):
        container = payload.get(container_name)
        if not isinstance(container, dict):
            continue
        for field_name in ("onsetDelayAppliedSeconds", "onsetDelaySeconds"):
            try:
                delay = float(container.get(field_name))
            except (TypeError, ValueError):
                continue
            if math.isfinite(delay):
                return max(-0.10, min(0.10, delay))
    return 0.0


def fuse_focused_transcription(
    primary: dict[str, Any],
    focused: dict[str, Any],
    target_instruments: Iterable[str] = ("voice",),
    *,
    known_shared_audio: bool = False,
    strategy: str = "replace-target-family",
    support_scope: str = "primary-pitched",
    fallback_support_scope: str | None = None,
    support_pitch_mode: str = "exact",
    support_tolerance_seconds: float = 0.10,
    minimum_pass_support_ratio: float = 0.65,
    minimum_primary_target_notes_for_fallback: int = 0,
    minimum_primary_target_ratio_for_fallback: float = 0.0,
    retain_unmatched_primary: bool = True,
    focused_melody_decoder: dict[str, Any] | None = None,
) -> dict[str, Any]:
    targets = clean_instruments(target_instruments)
    if not targets:
        raise ValueError("At least one focused instrument is required.")
    if strategy not in {"replace-target-family", "corroborated-union"}:
        raise ValueError("Unsupported focused-transcription fusion strategy.")
    if support_pitch_mode not in {"exact", "pitch-class"}:
        raise ValueError("support_pitch_mode must be exact or pitch-class.")
    if support_scope not in {"primary-pitched", "primary-target"}:
        raise ValueError("support_scope must be primary-pitched or primary-target.")
    if fallback_support_scope not in {None, "primary-pitched", "primary-target"}:
        raise ValueError(
            "fallback_support_scope must be primary-pitched, primary-target, or omitted."
        )
    if not 0.01 <= support_tolerance_seconds <= 0.50:
        raise ValueError("support_tolerance_seconds must be between 0.01 and 0.50.")
    if not 0.0 <= minimum_pass_support_ratio <= 1.0:
        raise ValueError("minimum_pass_support_ratio must be between zero and one.")
    if minimum_primary_target_notes_for_fallback < 0:
        raise ValueError("minimum_primary_target_notes_for_fallback cannot be negative.")
    if not 0.0 <= minimum_primary_target_ratio_for_fallback <= 1.0:
        raise ValueError(
            "minimum_primary_target_ratio_for_fallback must be between zero and one."
        )
    primary_notes = [copy.deepcopy(note) for note in primary.get("notes", []) if valid_note(note)]
    focused_notes = [copy.deepcopy(note) for note in focused.get("notes", []) if valid_note(note)]
    # A standalone transcription run may subtract its own beat-grid onset
    # delay. Put the focused pass onto the primary pass's clock before fusing;
    # otherwise a few milliseconds of preprocessing drift pollute the musical
    # comparison. In the integrated worker both values are zero because fusion
    # happens before the one shared tempo-detection call.
    primary_delay = applied_onset_delay(primary)
    focused_delay = applied_onset_delay(focused)
    timebase_shift = focused_delay - primary_delay
    if timebase_shift:
        for note in focused_notes:
            note["time"] = round(max(0.0, float(note["time"]) + timebase_shift), 4)
    replacements = [note for note in focused_notes if note_instrument(note) in targets]
    if not replacements:
        raise ValueError("The focused pass contains no valid notes for the requested instruments.")

    primary_duration = payload_duration(primary_notes)
    focused_duration = payload_duration(focused_notes)
    # Both passes are generated from the exact same prepared audio.  A large
    # duration disagreement therefore signals a wrong/mismatched artifact.
    if primary_duration and focused_duration:
        duration_ratio = focused_duration / primary_duration
        if not known_shared_audio and (duration_ratio < 0.80 or duration_ratio > 1.20):
            raise ValueError("Primary and focused transcriptions do not appear to share a timeline.")

    primary_targets = [note for note in primary_notes if note_instrument(note) in targets]
    non_targets = [note for note in primary_notes if note_instrument(note) not in targets]
    focused_candidates = replacements
    supported_focused = replacements
    support_ratio = 1.0
    pass_accepted = True
    effective_support_scope = support_scope
    fallback_scope_used = False
    retained_primary_targets: list[dict[str, Any]] = []
    melody_decoder_diagnostics: dict[str, Any] | None = None
    if strategy == "corroborated-union":
        all_pitched_primary = [
            note
            for note in primary_notes
            if note_instrument(note) not in PERCUSSION_INSTRUMENTS
        ]
        primary_target_ratio = len(primary_targets) / max(1, len(all_pitched_primary))
        fallback_evidence_sufficient = (
            len(primary_targets) >= minimum_primary_target_notes_for_fallback
            and primary_target_ratio >= minimum_primary_target_ratio_for_fallback
        )

        def supported_for_scope(scope: str) -> list[dict[str, Any]]:
            support_notes = (
                primary_targets if scope == "primary-target" else all_pitched_primary
            )
            support = time_index(support_notes, support_pitch_mode)
            return [
                note
                for note in focused_candidates
                if has_nearby_support(
                    note,
                    support,
                    support_tolerance_seconds,
                    support_pitch_mode,
                )
            ]

        supported_focused = supported_for_scope(support_scope)
        support_ratio = len(supported_focused) / max(1, len(focused_candidates))
        pass_accepted = support_ratio >= minimum_pass_support_ratio
        if (
            not pass_accepted
            and fallback_support_scope
            and fallback_support_scope != support_scope
            and fallback_evidence_sufficient
        ):
            fallback_supported = supported_for_scope(fallback_support_scope)
            fallback_ratio = len(fallback_supported) / max(1, len(focused_candidates))
            if fallback_ratio >= minimum_pass_support_ratio:
                supported_focused = fallback_supported
                support_ratio = fallback_ratio
                pass_accepted = True
                effective_support_scope = fallback_support_scope
                fallback_scope_used = True
        if pass_accepted and retain_unmatched_primary:
            focused_exact = time_index(supported_focused, "exact")
            retained_primary_targets = [
                note
                for note in primary_targets
                if not has_nearby_support(
                    note,
                    focused_exact,
                    support_tolerance_seconds,
                    "exact",
                )
            ]
        elif not pass_accepted:
            retained_primary_targets = primary_targets
            supported_focused = []

        if (
            pass_accepted
            and targets == {"voice"}
            and isinstance(focused_melody_decoder, dict)
            and focused_melody_decoder.get("enabled")
        ):
            decoder_config = {
                key: value
                for key, value in focused_melody_decoder.items()
                if key not in {"enabled", "decode_all_candidates_after_acceptance"}
            }
            decoder_input = supported_focused
            activation_span = max(primary_duration, 1.0)
            activation_density = len(supported_focused) / activation_span
            minimum_decoder_density = float(
                focused_melody_decoder.get("minimum_input_notes_per_second", 0.0)
                or 0.0
            )
            decode_all_candidates = bool(
                focused_melody_decoder.get(
                    "decode_all_candidates_after_acceptance", False
                )
            )
            if (
                decode_all_candidates
                and activation_density >= minimum_decoder_density
            ):
                decoder_input = focused_candidates
                # The activation decision was made from corroborated evidence,
                # not the larger candidate set. Avoid applying the density gate
                # a second time to a different population.
                decoder_config["minimum_input_notes_per_second"] = 0.0
            supported_focused, melody_decoder_diagnostics = (
                decode_focused_vocal_melody(
                    decoder_input,
                    primary_notes,
                    config=decoder_config,
                )
            )
            melody_decoder_diagnostics["activationNotes"] = len(
                supported_for_scope(effective_support_scope)
            )
            melody_decoder_diagnostics["activationNotesPerSecond"] = round(
                activation_density, 6
            )
            melody_decoder_diagnostics["activationMinimumNotesPerSecond"] = round(
                minimum_decoder_density, 6
            )
            melody_decoder_diagnostics[
                "decodedAllFocusedCandidatesAfterAcceptance"
            ] = decoder_input is focused_candidates
            if (
                melody_decoder_diagnostics.get("applied")
                and focused_melody_decoder.get(
                    "retain_unmatched_primary_when_applied", True
                )
            ):
                recovered_primary: list[dict[str, Any]] = []
                anchors_overriding_candidates = 0
                for source_anchor in sorted(
                    primary_targets,
                    key=lambda note: (float(note["time"]), note_pitch_key(note, "exact")),
                ):
                    anchor = copy.deepcopy(source_anchor)
                    onset = float(anchor["time"])
                    nearby = [
                        (index, note)
                        for index, note in enumerate(supported_focused)
                        if abs(float(note["time"]) - onset)
                        <= support_tolerance_seconds
                    ]
                    if any(
                        note_pitch_key(note, "exact")
                        == note_pitch_key(anchor, "exact")
                        for _index, note in nearby
                    ):
                        continue
                    if nearby:
                        replace_index, _replaced = min(
                            nearby,
                            key=lambda item: (
                                abs(float(item[1]["time"]) - onset),
                                abs(
                                    note_pitch_key(item[1], "exact")
                                    - note_pitch_key(anchor, "exact")
                                ),
                            ),
                        )
                        supported_focused.pop(replace_index)
                        anchors_overriding_candidates += 1
                    anchor["focusedMelodyDecoded"] = True
                    anchor["focusedMelodyAnchorRecovered"] = True
                    recovered_primary.append(anchor)
                retained_primary_targets = recovered_primary
                melody_decoder_diagnostics["primaryAnchorNotesRecovered"] = len(
                    recovered_primary
                )
                melody_decoder_diagnostics[
                    "primaryAnchorsOverridingFocusedCandidates"
                ] = anchors_overriding_candidates

    output = copy.deepcopy(primary)
    if strategy == "corroborated-union" and not pass_accepted:
        # A rejected enhancement must be a literal fallback. Reordering equal
        # onset events can change downstream tie-breaking, so preserve the
        # primary payload's original notes and instrument groups exactly.
        merged_notes = copy.deepcopy(primary.get("notes", []))
    else:
        retained = non_targets + retained_primary_targets
        merged_notes = retained + supported_focused
        merged_notes.sort(
            key=lambda note: (
                float(note.get("time", 0.0)),
                int(round(float(note.get("midi", note.get("pitch", 0))))),
                note_instrument(note),
            )
        )
        output["instrumentGroups"] = sorted(
            {note_instrument(note) for note in merged_notes if note_instrument(note)}
        )
    output["notes"] = merged_notes
    output["focusedTranscriptionFusion"] = {
        "schema": "polymath-focused-transcription-fusion-v1",
        "strategy": strategy,
        "targetInstruments": sorted(targets),
        "applied": pass_accepted,
        "primaryNotes": len(primary_notes),
        "primaryTargetNotes": len(primary_targets),
        "primaryTargetNotesRemoved": len(primary_targets) - len(retained_primary_targets),
        "primaryTargetNotesRetained": len(retained_primary_targets),
        "focusedCandidateNotes": len(focused_candidates),
        "focusedNotesInserted": len(supported_focused),
        "focusedSupportRatio": round(support_ratio, 6),
        "minimumPassSupportRatio": round(minimum_pass_support_ratio, 6),
        "supportScope": support_scope,
        "fallbackSupportScope": fallback_support_scope,
        "effectiveSupportScope": effective_support_scope,
        "fallbackSupportScopeUsed": fallback_scope_used,
        "minimumPrimaryTargetNotesForFallback": int(
            minimum_primary_target_notes_for_fallback
        ),
        "minimumPrimaryTargetRatioForFallback": round(
            minimum_primary_target_ratio_for_fallback, 6
        ),
        "primaryTargetRatioAmongPitched": round(
            len(primary_targets)
            / max(
                1,
                sum(
                    note_instrument(note) not in PERCUSSION_INSTRUMENTS
                    for note in primary_notes
                ),
            ),
            6,
        ),
        "fallbackEvidenceSufficient": (
            len(primary_targets) >= minimum_primary_target_notes_for_fallback
            and len(primary_targets)
            / max(
                1,
                sum(
                    note_instrument(note) not in PERCUSSION_INSTRUMENTS
                    for note in primary_notes
                ),
            )
            >= minimum_primary_target_ratio_for_fallback
        ),
        "supportPitchMode": support_pitch_mode,
        "supportToleranceSeconds": round(support_tolerance_seconds, 6),
        "retainUnmatchedPrimary": bool(retain_unmatched_primary),
        "outputNotes": len(merged_notes),
        "timelineDurationRatio": round(focused_duration / max(primary_duration, 1e-9), 6),
        "timelineValidation": "same-audio-call" if known_shared_audio else "note-span-heuristic",
        "timebaseShiftSeconds": round(timebase_shift, 6),
        "melodyDecoder": melody_decoder_diagnostics,
    }
    return output


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", required=True)
    parser.add_argument("--focused", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--instruments", default="voice")
    parser.add_argument(
        "--strategy",
        choices=("replace-target-family", "corroborated-union"),
        default="replace-target-family",
    )
    parser.add_argument(
        "--support-pitch-mode", choices=("exact", "pitch-class"), default="exact"
    )
    parser.add_argument(
        "--support-scope",
        choices=("primary-pitched", "primary-target"),
        default="primary-pitched",
    )
    parser.add_argument(
        "--fallback-support-scope",
        choices=("primary-pitched", "primary-target"),
    )
    parser.add_argument("--support-tolerance-seconds", type=float, default=0.10)
    parser.add_argument("--minimum-pass-support-ratio", type=float, default=0.65)
    parser.add_argument("--known-shared-audio", action="store_true")
    parser.add_argument(
        "--minimum-primary-target-notes-for-fallback", type=int, default=0
    )
    parser.add_argument(
        "--minimum-primary-target-ratio-for-fallback", type=float, default=0.0
    )
    parser.add_argument(
        "--discard-unmatched-primary", action="store_true"
    )
    parser.add_argument(
        "--decode-focused-melody",
        action="store_true",
        help="Collapse an accepted voice pass into one held monophonic melody path.",
    )
    parser.add_argument(
        "--melody-decoder-config",
        metavar="PATH",
        help="Optional path to a JSON object overriding bounded research decoder settings.",
    )
    args = parser.parse_args()

    primary_path = Path(args.primary).resolve()
    focused_path = Path(args.focused).resolve()
    output_path = Path(args.output).resolve()
    for path in (primary_path, focused_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    instruments = args.instruments.split(",")
    melody_decoder_config: dict[str, Any] = {
        "enabled": args.decode_focused_melody
    }
    if args.melody_decoder_config:
        configured_decoder = load_json(Path(args.melody_decoder_config).resolve())
        melody_decoder_config.update(configured_decoder)
        melody_decoder_config["enabled"] = True

    result = fuse_focused_transcription(
        load_json(primary_path),
        load_json(focused_path),
        instruments,
        known_shared_audio=args.known_shared_audio,
        strategy=args.strategy,
        support_scope=args.support_scope,
        fallback_support_scope=args.fallback_support_scope,
        support_pitch_mode=args.support_pitch_mode,
        support_tolerance_seconds=args.support_tolerance_seconds,
        minimum_pass_support_ratio=args.minimum_pass_support_ratio,
        minimum_primary_target_notes_for_fallback=(
            args.minimum_primary_target_notes_for_fallback
        ),
        minimum_primary_target_ratio_for_fallback=(
            args.minimum_primary_target_ratio_for_fallback
        ),
        retain_unmatched_primary=not args.discard_unmatched_primary,
        focused_melody_decoder=melody_decoder_config,
    )
    result["focusedTranscriptionFusion"]["inputs"] = {
        "primary": {"path": str(primary_path), "sha256": sha256_file(primary_path)},
        "focused": {"path": str(focused_path), "sha256": sha256_file(focused_path)},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(output_path)
    print(json.dumps({"output": str(output_path), **result["focusedTranscriptionFusion"]}, indent=2))


if __name__ == "__main__":
    main()
