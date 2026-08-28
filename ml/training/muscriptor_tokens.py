"""MuScriptor-compatible MT3 token encoding for reviewed piano clips.

The released MuScriptor package exposes its inference tokenizer but keeps the
note-to-token encoder only in its tests. This small, independently testable
adapter follows that public MIT-licensed event layout exactly for piano data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


PAD_ID = 0
EOS_ID = 1
UNK_ID = 2
SHIFT_BASE = 3
MAX_SHIFT_STEPS = 1001
PITCH_BASE = SHIFT_BASE + MAX_SHIFT_STEPS
VELOCITY_BASE = PITCH_BASE + 128
TIE_ID = VELOCITY_BASE + 2
PROGRAM_BASE = TIE_ID + 1
DRUM_BASE = PROGRAM_BASE + 130
VOCAB_SIZE = DRUM_BASE + 128
MODEL_CARDINALITY = 1395
INITIAL_TOKEN_ID = MODEL_CARDINALITY
FRAME_RATE = 100
PIANO_PROGRAM = 0


class TokenEncodingError(ValueError):
    """Raised when reviewed labels cannot be represented safely."""


@dataclass(frozen=True)
class PianoEvent:
    time: float
    velocity: int
    pitch: int
    program: int = PIANO_PROGRAM


def _event_id(kind: str, value: int = 0) -> int:
    if kind == "EOS":
        return EOS_ID
    if kind == "shift" and 0 <= value < MAX_SHIFT_STEPS:
        return SHIFT_BASE + value
    if kind == "pitch" and 0 <= value <= 127:
        return PITCH_BASE + value
    if kind == "velocity" and value in (0, 1):
        return VELOCITY_BASE + value
    if kind == "tie" and value == 0:
        return TIE_ID
    if kind == "program" and 0 <= value <= 129:
        return PROGRAM_BASE + value
    raise TokenEncodingError(f"Unsupported MuScriptor event: {kind}={value}")


def _validated_notes(notes: Iterable[dict[str, Any]], duration_seconds: float) -> list[dict[str, Any]]:
    clean: list[dict[str, Any]] = []
    for index, note in enumerate(notes):
        if not isinstance(note, dict):
            raise TokenEncodingError(f"Note {index} is not an object")
        midi = note.get("midi")
        onset = note.get("time")
        duration = note.get("duration")
        if not isinstance(midi, int) or not 0 <= midi <= 127:
            raise TokenEncodingError(f"Note {index} has invalid MIDI pitch")
        if not isinstance(onset, (int, float)) or not 0 <= float(onset) <= duration_seconds:
            raise TokenEncodingError(f"Note {index} has invalid local onset")
        if not isinstance(duration, (int, float)) or float(duration) <= 0:
            raise TokenEncodingError(f"Note {index} has invalid duration")
        end = min(duration_seconds, float(onset) + float(duration))
        if end <= float(onset) and not note.get("continuedFromPreviousClip"):
            raise TokenEncodingError(f"Note {index} ends before it begins")
        clean.append({
            "midi": midi,
            "time": float(onset),
            "end": end,
            "continued": bool(note.get("continuedFromPreviousClip")),
            "continues": bool(note.get("continuesIntoNextClip")),
        })
    return clean


def encode_piano_clip(
    notes: Iterable[dict[str, Any]],
    duration_seconds: float = 5.0,
    include_eos: bool = True,
) -> list[int]:
    """Encode one reviewed clip into the released model's token IDs.

    Notes already sounding at the clip boundary become the MT3 tie prologue.
    Notes continuing beyond the right boundary intentionally omit their note-off
    event, allowing the following clip's tie prologue to carry the sustain.
    """
    if not 0 < duration_seconds <= 10:
        raise TokenEncodingError("Clip duration must be within (0, 10] seconds")
    clean = _validated_notes(notes, duration_seconds)
    tied_pitches = sorted({note["midi"] for note in clean if note["continued"]})
    tokens: list[int] = []
    program_state: int | None = None
    for pitch in tied_pitches:
        if program_state != PIANO_PROGRAM:
            tokens.append(_event_id("program", PIANO_PROGRAM))
            program_state = PIANO_PROGRAM
        tokens.append(_event_id("pitch", pitch))
    tokens.append(TIE_ID)

    events: list[PianoEvent] = []
    for note in clean:
        if not note["continued"]:
            events.append(PianoEvent(note["time"], 1, note["midi"]))
        if not note["continues"]:
            events.append(PianoEvent(note["end"], 0, note["midi"]))
    events.sort(key=lambda event: (
        round(event.time * FRAME_RATE), False, event.program, event.velocity, event.pitch,
    ))

    tick_state = 0
    velocity_state: int | None = None
    for event in events:
        tick = round(event.time * FRAME_RATE)
        if tick < tick_state:
            raise TokenEncodingError("Piano events are not monotonic")
        if tick > tick_state:
            tokens.append(_event_id("shift", tick))
            tick_state = tick
        if program_state != event.program:
            tokens.append(_event_id("program", event.program))
            program_state = event.program
        if velocity_state != event.velocity:
            tokens.append(_event_id("velocity", event.velocity))
            velocity_state = event.velocity
        tokens.append(_event_id("pitch", event.pitch))
    if include_eos:
        tokens.append(EOS_ID)
    if any(token < 0 or token >= MODEL_CARDINALITY for token in tokens):
        raise TokenEncodingError("Encoded token is outside the model cardinality")
    return tokens


def teacher_forcing_pair(
    tokens: list[int],
    initial_token_id: int = INITIAL_TOKEN_ID,
) -> tuple[list[int], list[int]]:
    """Return causal inputs and next-token labels for cross-entropy training."""
    if not tokens or tokens[-1] != EOS_ID:
        raise TokenEncodingError("Teacher-forcing targets must end with EOS")
    if initial_token_id < VOCAB_SIZE:
        raise TokenEncodingError("Initial token must sit outside the event vocabulary")
    return [initial_token_id, *tokens[:-1]], list(tokens)
