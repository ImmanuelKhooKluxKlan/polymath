"""MuScriptor-compatible MT3 token encoding for one reviewed instrument.

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

# These IDs and representative programs come from MuScriptor 0.2.2's public
# MT3_FULL_PLUS vocabulary.  Training one instrument at a time is deliberate:
# it lets the conditioning signal, target program token, and evaluation filter
# agree instead of silently mixing several causes of an error.
INSTRUMENT_GROUPS: dict[str, tuple[int, int]] = {
    "acoustic_piano": (0, 0),
    "electric_piano": (1, 2),
    "chromatic_percussion": (2, 8),
    "organ": (3, 16),
    "acoustic_guitar": (4, 24),
    "clean_electric_guitar": (5, 26),
    "distorted_electric_guitar": (6, 29),
    "acoustic_bass": (7, 32),
    "electric_bass": (8, 33),
    "violin": (9, 40),
    "viola": (10, 41),
    "cello": (11, 42),
    "contrabass": (12, 43),
    "orchestral_harp": (13, 46),
    "timpani": (14, 47),
    "string_ensemble": (15, 48),
    "synth_strings": (16, 50),
    "voice": (17, 52),
    "orchestra_hit": (18, 55),
    "trumpet": (19, 56),
    "trombone": (20, 57),
    "tuba": (21, 58),
    "french_horn": (22, 60),
    "brass_section": (23, 61),
    "soprano_and_alto_sax": (24, 64),
    "tenor_sax": (25, 66),
    "baritone_sax": (26, 67),
    "oboe": (27, 68),
    "english_horn": (28, 69),
    "bassoon": (29, 70),
    "clarinet": (30, 71),
    "flutes": (31, 72),
    "synth_lead": (32, 80),
    "synth_pad": (33, 88),
}

INSTRUMENT_ALIASES = {
    "piano": "acoustic_piano",
    "acoustic grand piano": "acoustic_piano",
    "acoustic_grand_piano": "acoustic_piano",
    "electric guitar": "clean_electric_guitar",
    "guitar": "acoustic_guitar",
    "bass": "acoustic_bass",
    "vocals": "voice",
    "vocal": "voice",
}


class TokenEncodingError(ValueError):
    """Raised when reviewed labels cannot be represented safely."""


@dataclass(frozen=True)
class InstrumentEvent:
    time: float
    velocity: int
    pitch: int
    program: int


def canonical_instrument_name(value: Any) -> str:
    name = str(value or "acoustic_piano").strip().lower().replace("-", "_")
    name = INSTRUMENT_ALIASES.get(name, name.replace(" ", "_"))
    if name not in INSTRUMENT_GROUPS:
        raise TokenEncodingError(
            f"Unsupported individual instrument {value!r}; expected one of "
            + ", ".join(INSTRUMENT_GROUPS)
        )
    return name


def instrument_group_id(value: Any) -> int:
    return INSTRUMENT_GROUPS[canonical_instrument_name(value)][0]


def instrument_program(value: Any) -> int:
    return INSTRUMENT_GROUPS[canonical_instrument_name(value)][1]


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


def _validated_notes(
    notes: Iterable[dict[str, Any]],
    duration_seconds: float,
    instrument: str,
) -> list[dict[str, Any]]:
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
        note_instrument = canonical_instrument_name(note.get("instrument") or instrument)
        if note_instrument != instrument:
            raise TokenEncodingError(
                f"Note {index} belongs to {note_instrument}, not the clip's {instrument} focus"
            )
        clean.append({
            "midi": midi,
            "time": float(onset),
            "end": end,
            "continued": bool(note.get("continuedFromPreviousClip")),
            "continues": bool(note.get("continuesIntoNextClip")),
        })
    return _trim_same_key_overlaps(clean)


def _trim_same_key_overlaps(notes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Match MuScriptor's public tokenizer invariant for repeated keys.

    A MIDI key cannot be independently active twice for the same instrument.
    Duplicate onsets are collapsed, and an earlier note is stopped at the next
    strike.  Without this normalization, a later note-off can accidentally
    terminate the new strike and teach the model the chopped/stuttering pattern
    we are trying to remove.
    """

    normalized: list[dict[str, Any]] = []
    by_pitch: dict[int, list[dict[str, Any]]] = {}
    for note in notes:
        by_pitch.setdefault(note["midi"], []).append(dict(note))
    for pitch_notes in by_pitch.values():
        ordered = sorted(pitch_notes, key=lambda note: (note["time"], -note["end"]))
        deduplicated: list[dict[str, Any]] = []
        for note in ordered:
            if deduplicated and abs(note["time"] - deduplicated[-1]["time"]) < 1e-7:
                previous = deduplicated[-1]
                previous["end"] = max(previous["end"], note["end"])
                previous["continues"] = previous["continues"] or note["continues"]
                previous["continued"] = previous["continued"] or note["continued"]
                continue
            deduplicated.append(note)
        for previous, following in zip(deduplicated, deduplicated[1:]):
            if previous["end"] > following["time"]:
                previous["end"] = following["time"]
                previous["continues"] = False
        normalized.extend(note for note in deduplicated if note["end"] > note["time"])
    return sorted(normalized, key=lambda note: (note["time"], note["midi"]))


def encode_instrument_clip(
    notes: Iterable[dict[str, Any]],
    duration_seconds: float = 5.0,
    include_eos: bool = True,
    instrument: str = "acoustic_piano",
) -> list[int]:
    """Encode one reviewed individual-instrument clip into model token IDs.

    Notes already sounding at the clip boundary become the MT3 tie prologue.
    Notes continuing beyond the right boundary intentionally omit their note-off
    event, allowing the following clip's tie prologue to carry the sustain.
    """
    if not 0 < duration_seconds <= 10:
        raise TokenEncodingError("Clip duration must be within (0, 10] seconds")
    instrument = canonical_instrument_name(instrument)
    program = instrument_program(instrument)
    clean = _validated_notes(notes, duration_seconds, instrument)
    tied_pitches = sorted({note["midi"] for note in clean if note["continued"]})
    tokens: list[int] = []
    program_state: int | None = None
    for pitch in tied_pitches:
        if program_state != program:
            tokens.append(_event_id("program", program))
            program_state = program
        tokens.append(_event_id("pitch", pitch))
    tokens.append(TIE_ID)

    events: list[InstrumentEvent] = []
    for note in clean:
        if not note["continued"]:
            events.append(InstrumentEvent(note["time"], 1, note["midi"], program))
        if not note["continues"]:
            events.append(InstrumentEvent(note["end"], 0, note["midi"], program))
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


def encode_piano_clip(
    notes: Iterable[dict[str, Any]],
    duration_seconds: float = 5.0,
    include_eos: bool = True,
) -> list[int]:
    """Backward-compatible acoustic-piano wrapper."""

    return encode_instrument_clip(
        notes,
        duration_seconds=duration_seconds,
        include_eos=include_eos,
        instrument="acoustic_piano",
    )


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
