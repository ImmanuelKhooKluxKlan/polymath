"""Music-domain helpers shared by the symbolic and vision readers."""

from __future__ import annotations

from dataclasses import dataclass


NOTE_TO_SEMITONE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
DIATONIC = ("C", "D", "E", "F", "G", "A", "B")
SHARP_ORDER = ("F", "C", "G", "D", "A", "E", "B")
FLAT_ORDER = ("B", "E", "A", "D", "G", "C", "F")
MAJOR_KEYS = {
    -7: "Cb major", -6: "Gb major", -5: "Db major", -4: "Ab major",
    -3: "Eb major", -2: "Bb major", -1: "F major", 0: "C major",
    1: "G major", 2: "D major", 3: "A major", 4: "E major",
    5: "B major", 6: "F# major", 7: "C# major",
}
MINOR_KEYS = {
    -7: "Ab minor", -6: "Eb minor", -5: "Bb minor", -4: "F minor",
    -3: "C minor", -2: "G minor", -1: "D minor", 0: "A minor",
    1: "E minor", 2: "B minor", 3: "F# minor", 4: "C# minor",
    5: "G# minor", 6: "D# minor", 7: "A# minor",
}


@dataclass(frozen=True)
class TimeSignature:
    numerator: int = 4
    denominator: int = 4

    @property
    def measure_quarter_beats(self) -> float:
        return self.numerator * 4.0 / self.denominator


def pitch_to_midi(step: str, octave: int, alter: int = 0) -> int:
    return (int(octave) + 1) * 12 + NOTE_TO_SEMITONE[step.upper()] + int(alter)


def midi_to_name(midi: int) -> str:
    names = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
    midi = max(0, min(127, int(round(midi))))
    return f"{names[midi % 12]}{midi // 12 - 1}"


def diatonic_shift(step: str, octave: int, amount: int) -> tuple[str, int]:
    index = DIATONIC.index(step.upper()) + int(amount)
    shifted_octave = int(octave) + index // 7
    return DIATONIC[index % 7], shifted_octave


def staff_step_to_midi(clef: str, step_from_bottom_line: int, key_fifths: int = 0) -> int:
    # The bottom line is E4 in treble clef and G2 in bass clef.
    base = ("G", 2) if clef == "bass" else ("E", 4)
    step, octave = diatonic_shift(*base, int(step_from_bottom_line))
    altered = 0
    if key_fifths > 0 and step in SHARP_ORDER[: min(7, key_fifths)]:
        altered = 1
    elif key_fifths < 0 and step in FLAT_ORDER[: min(7, -key_fifths)]:
        altered = -1
    return pitch_to_midi(step, octave, altered)


def key_name(fifths: int, mode: str = "major") -> str:
    fifths = max(-7, min(7, int(fifths)))
    return (MINOR_KEYS if str(mode).lower() == "minor" else MAJOR_KEYS)[fifths]


def merge_tied_notes(notes: list[dict]) -> list[dict]:
    """Merge MusicXML tie chains without combining ordinary repeated notes."""
    output: list[dict] = []
    active: dict[tuple, dict] = {}
    for note in sorted(notes, key=lambda item: (item["time"], item.get("midi", 0))):
        key = (note.get("midi"), note.get("voice", ""), note.get("staff", 1))
        tie_stop = bool(note.pop("_tie_stop", False))
        tie_start = bool(note.pop("_tie_start", False))
        if tie_stop and key in active:
            prior = active[key]
            prior["duration"] = round(max(
                prior["duration"], note["time"] + note["duration"] - prior["time"],
            ), 6)
            if not tie_start:
                active.pop(key, None)
            continue
        output.append(note)
        if tie_start:
            active[key] = note
    return output
