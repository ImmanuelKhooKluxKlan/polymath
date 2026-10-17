"""Research-only source tags for a piano-output decoding of the same audio.

This operator changes no pitches, onsets, durations, or velocities. It attaches
corroborating broad-pass instrument labels so the existing arranger can tell a
likely singer line from accompaniment. Its labels are model evidence, not
verified source-separation truth. No reference MIDI is accepted by the API.
"""

from __future__ import annotations

import bisect
from collections import Counter, defaultdict
import copy
import math
from typing import Any


def annotate_source_evidence(
    piano: dict[str, Any], broad: dict[str, Any], *,
    known_shared_audio: bool = False, voice_only: bool = True,
    tolerance_seconds: float = 0.08, maximum_octave_distance: int = 0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if known_shared_audio is not True:
        raise ValueError("Source evidence requires a verified shared audio clock")
    if not math.isfinite(tolerance_seconds) or not 0 < tolerance_seconds <= 0.25:
        raise ValueError("Source-evidence onset tolerance must be within (0, 0.25]")
    if maximum_octave_distance not in (0, 12):
        raise ValueError("Only exact-register or one-octave corroboration is supported")
    output = copy.deepcopy(piano)
    notes = output.get("notes") or []
    if any(str(note.get("instrument") or "").lower() not in {"piano", "acoustic_piano"} for note in notes):
        raise ValueError("Expected a piano-only decoder pass")
    groups: dict[int, list[tuple[float, int, dict[str, Any]]]] = defaultdict(list)
    for source_index, source in enumerate(broad.get("notes") or []):
        instrument = str(source.get("instrument") or "").lower()
        if instrument in {"drums", "timpani"} or not instrument:
            continue
        if voice_only and instrument != "voice":
            continue
        midi, time = int(source["midi"]), float(source["time"])
        if not 0 <= midi <= 127 or not math.isfinite(time) or time < 0:
            raise ValueError("Invalid broad-pass source event")
        groups[midi % 12].append((time, source_index, source))
    for group in groups.values():
        group.sort(key=lambda item: (item[0], item[1]))
    times = {key: [item[0] for item in group] for key, group in groups.items()}
    edges = []
    for index, note in enumerate(notes):
        midi, time = int(note["midi"]), float(note["time"])
        if not 0 <= midi <= 127 or not math.isfinite(time) or time < 0:
            raise ValueError("Invalid piano-pass event")
        group = groups.get(midi % 12, [])
        clock = times.get(midi % 12, [])
        low = bisect.bisect_left(clock, time - tolerance_seconds)
        high = bisect.bisect_right(clock, time + tolerance_seconds)
        for source_time, source_index, source in group[low:high]:
            register_distance = abs(int(source["midi"]) - midi)
            if register_distance > maximum_octave_distance:
                continue
            voice_priority = 0 if source["instrument"] == "voice" else 1
            edges.append((register_distance, abs(source_time - time), voice_priority, index, source_index, source))
    used_notes, used_sources = set(), set()
    tagged = Counter()
    for _register_distance, error, _priority, index, source_index, source in sorted(edges, key=lambda edge: edge[:5]):
        if index in used_notes or source_index in used_sources:
            continue
        note = notes[index]
        note["pianoOutputInstrumentBeforeSourceRouting"] = note["instrument"]
        note["instrument"] = source["instrument"]
        note["sourceInstrument"] = source["instrument"]
        note["sourceEvidenceIndex"] = source_index
        note["sourceEvidenceMidi"] = int(source["midi"])
        note["sourceEvidenceOnsetErrorSeconds"] = round(error, 6)
        used_notes.add(index)
        used_sources.add(source_index)
        tagged[str(source["instrument"])] += 1
    output["instrumentGroups"] = sorted({str(note["instrument"]) for note in notes})
    diagnostics = {
        "schema": "polymath-piano-source-evidence-routing-v1",
        "voiceOnly": voice_only,
        "toleranceSeconds": tolerance_seconds,
        "maximumOctaveDistance": maximum_octave_distance,
        "pianoNotes": len(notes),
        "taggedNotes": len(used_notes),
        "taggedByInstrument": dict(tagged),
        "sourceEvidenceUsedAtMostOnce": True,
        "pitchesOnsetsDurationsVelocitiesPreserved": True,
        "referenceAtInference": False,
        "interpretation": "Corroborating model source labels only; not verified instrumental ground truth.",
    }
    output["sourceEvidenceRouting"] = diagnostics
    return output, diagnostics
