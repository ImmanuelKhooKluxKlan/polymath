"""Exact local conversion when a PDF contains an embedded MusicXML source."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import pypdfium2 as pdfium

from .music import TimeSignature, key_name, merge_tied_notes, midi_to_name, pitch_to_midi
from .performance import shape_piano_performance


DYNAMIC_VELOCITIES = {
    "ppp": 0.34, "pp": 0.42, "p": 0.52,
    "mp": 0.64, "mf": 0.76,
    "f": 0.86, "ff": 0.93, "fff": 0.98,
}


def _strip_namespaces(root: ET.Element) -> ET.Element:
    for element in root.iter():
        element.tag = element.tag.rsplit("}", 1)[-1]
    return root


def _mxl_xml(payload: bytes) -> bytes | None:
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            container = ET.fromstring(archive.read("META-INF/container.xml"))
            rootfile = next((node for node in container.iter() if node.tag.endswith("rootfile")), None)
            if rootfile is None:
                return None
            return archive.read(rootfile.attrib["full-path"])
    except (KeyError, OSError, ET.ParseError, zipfile.BadZipFile):
        return None


def find_embedded_musicxml(pdf_path: str | Path) -> tuple[bytes, str] | None:
    with pdfium.PdfDocument(str(pdf_path)) as document:
        for index in range(document.count_attachments()):
            attachment = document.get_attachment(index)
            name = attachment.get_name()
            payload = bytes(attachment.get_data())
            suffix = Path(name).suffix.lower()
            if suffix in {".musicxml", ".xml"} and b"score-" in payload[:2000]:
                return payload, name
            if suffix == ".mxl":
                xml = _mxl_xml(payload)
                if xml:
                    return xml, name
    return None


def _text(node: ET.Element | None, path: str, fallback: str = "") -> str:
    found = node.find(path) if node is not None else None
    return str(found.text or "").strip() if found is not None else fallback


def _choose_part(root: ET.Element, instrument: str) -> tuple[str, str]:
    part_names = {}
    for score_part in root.findall("./part-list/score-part"):
        part_names[score_part.attrib.get("id", "")] = _text(score_part, "part-name", "Unknown part")
    needles = {
        "piano": ("piano", "keyboard", "grand"),
        "synth": ("synth", "keyboard"),
        "guitar": ("guitar",),
        "electric_guitar": ("electric guitar", "guitar"),
        "ukulele": ("ukulele", "uke"),
        "upright_bass": ("bass",),
        "drums": ("drum", "percussion"),
    }.get(instrument, (instrument.replace("_", " "),))
    for part_id, name in part_names.items():
        if any(needle in name.lower() for needle in needles):
            return part_id, name
    first = next(iter(part_names.items()), ("", "Unknown part"))
    return first


def parse_musicxml(payload: bytes, instrument: str, source_name: str = "embedded.musicxml") -> dict:
    try:
        root = _strip_namespaces(ET.fromstring(payload))
    except ET.ParseError as error:
        raise ValueError(f"Embedded MusicXML is invalid: {error}") from error
    if root.tag != "score-partwise":
        raise ValueError("Only score-partwise embedded MusicXML is currently supported.")

    part_id, part_name = _choose_part(root, instrument)
    part = root.find(f"./part[@id='{part_id}']")
    if part is None:
        raise ValueError("The embedded MusicXML has no playable part.")

    title = _text(root, "movement-title") or _text(root, "work/work-title")
    composer_node = next((
        creator for creator in root.findall("./identification/creator")
        if creator.attrib.get("type", "").lower() in {"composer", "arranger"}
    ), None)
    composer = str(composer_node.text or "").strip() if composer_node is not None else ""
    divisions = 1.0
    tempo = 120.0
    time_signature = TimeSignature()
    fifths = 0
    mode = "major"
    absolute_beats = 0.0
    note_rows: list[dict] = []
    rest_events: list[dict] = []
    tempo_events: list[tuple[float, float]] = [(0.0, tempo)]
    pedal_rows: list[dict] = []
    current_velocity = 0.76
    current_dynamic_mark = "mf"
    active_slur_voices: set[tuple[str, int]] = set()
    measure_count = 0

    for measure in part.findall("measure"):
        measure_count += 1
        attributes = measure.find("attributes")
        if attributes is not None:
            divisions = max(1e-6, float(_text(attributes, "divisions", str(divisions))))
            numerator = int(float(_text(attributes, "time/beats", str(time_signature.numerator))))
            denominator = int(float(_text(attributes, "time/beat-type", str(time_signature.denominator))))
            time_signature = TimeSignature(numerator, denominator)
            fifths = int(float(_text(attributes, "key/fifths", str(fifths))))
            mode = _text(attributes, "key/mode", mode)

        measure_cursor = 0.0
        last_onset_by_voice: dict[tuple[str, int], float] = {}
        measure_extent = time_signature.measure_quarter_beats
        for child in list(measure):
            if child.tag == "direction":
                sound = child.find("sound")
                metronome = child.find("./direction-type/metronome/per-minute")
                raw_tempo = sound.attrib.get("tempo") if sound is not None else None
                raw_tempo = raw_tempo or (metronome.text if metronome is not None else None)
                try:
                    candidate = float(raw_tempo)
                    if 20 <= candidate <= 300:
                        tempo = candidate
                        tempo_events.append((absolute_beats + measure_cursor, tempo))
                except (TypeError, ValueError):
                    pass
                offset = float(_text(child, "offset", "0") or 0) / divisions
                direction_beat = absolute_beats + max(0.0, measure_cursor + offset)
                dynamic = child.find("./direction-type/dynamics")
                if dynamic is not None and list(dynamic):
                    mark = "".join(node.tag.lower() for node in list(dynamic))
                    current_velocity = DYNAMIC_VELOCITIES.get(mark, current_velocity)
                    if mark in DYNAMIC_VELOCITIES:
                        current_dynamic_mark = mark
                pedal = child.find("./direction-type/pedal")
                pedal_type = pedal.attrib.get("type", "") if pedal is not None else ""
                damper = sound.attrib.get("damper-pedal") if sound is not None else None
                if pedal_type in {"start", "resume"} or str(damper).lower() in {"yes", "true", "1"}:
                    pedal_rows.append({"beat": direction_beat, "down": True, "kind": "start"})
                elif pedal_type in {"stop", "discontinue"} or str(damper).lower() in {"no", "false", "0"}:
                    pedal_rows.append({"beat": direction_beat, "down": False, "kind": "stop"})
                elif pedal_type == "change":
                    pedal_rows.extend([
                        {"beat": direction_beat, "down": False, "kind": "change-up"},
                        {"beat": direction_beat, "down": True, "kind": "change-down", "delay": 0.045},
                    ])
                continue
            if child.tag in {"backup", "forward"}:
                amount = float(_text(child, "duration", "0")) / divisions
                measure_cursor += -amount if child.tag == "backup" else amount
                measure_cursor = max(0.0, measure_cursor)
                continue
            if child.tag != "note":
                continue
            duration_beats = max(0.0, float(_text(child, "duration", "0")) / divisions)
            voice = _text(child, "voice", "1")
            staff = max(1, int(float(_text(child, "staff", "1"))))
            voice_key = (voice, staff)
            chord = child.find("chord") is not None
            onset = last_onset_by_voice.get(voice_key, measure_cursor) if chord else measure_cursor
            if not chord:
                last_onset_by_voice[voice_key] = onset
            measure_extent = max(measure_extent, onset + duration_beats)
            if child.find("rest") is not None:
                rest_events.append({"beat": absolute_beats + onset, "durationBeats": duration_beats})
            else:
                pitch = child.find("pitch")
                if pitch is not None:
                    step = _text(pitch, "step", "C").upper()
                    octave = int(float(_text(pitch, "octave", "4")))
                    alter = int(round(float(_text(pitch, "alter", "0"))))
                    midi = pitch_to_midi(step, octave, alter)
                    tie_types = {tie.attrib.get("type", "") for tie in child.findall("tie")}
                    slur_types = {slur.attrib.get("type", "") for slur in child.findall("./notations/slur")}
                    legato = voice_key in active_slur_voices or bool(slur_types & {"start", "stop", "continue"})
                    articulation = _articulation(child)
                    if legato and "legato" not in articulation:
                        articulation = f"{articulation} legato slur".strip()
                    note_rows.append({
                        "beat": absolute_beats + onset,
                        "durationBeats": max(duration_beats, 0.0625),
                        "midi": midi,
                        "velocity": current_velocity,
                        "dynamic": current_dynamic_mark,
                        "hand": "left" if instrument == "piano" and staff > 1 else "right",
                        "voice": f"{part_name} voice {voice}",
                        "staff": staff,
                        "measure": measure_count,
                        "articulation": articulation,
                        "_tie_start": "start" in tie_types,
                        "_tie_stop": "stop" in tie_types,
                    })
                    if "start" in slur_types:
                        active_slur_voices.add(voice_key)
                    if "stop" in slur_types or "discontinue" in slur_types:
                        active_slur_voices.discard(voice_key)
            if not chord:
                measure_cursor += duration_beats
        absolute_beats += max(measure_extent, time_signature.measure_quarter_beats)

    # A printed tempo at beat zero replaces the default rather than creating two
    # contradictory events at the same musical position.
    tempo_events = sorted({beat: value for beat, value in tempo_events}.items())

    def beat_to_seconds(beat: float) -> float:
        elapsed = 0.0
        previous_beat, previous_tempo = tempo_events[0]
        for change_beat, change_tempo in tempo_events[1:]:
            if change_beat >= beat:
                break
            elapsed += max(0.0, change_beat - previous_beat) * 60.0 / previous_tempo
            previous_beat, previous_tempo = change_beat, change_tempo
        return elapsed + max(0.0, beat - previous_beat) * 60.0 / previous_tempo

    notes = []
    for source in note_rows:
        beat = source["beat"]
        duration_beats = source["durationBeats"]
        start = beat_to_seconds(beat)
        row = {
            key: value for key, value in source.items()
            if key not in {"beat", "durationBeats"}
        }
        row["time"] = round(start, 6)
        row["duration"] = round(max(0.01, beat_to_seconds(beat + duration_beats) - start), 6)
        row["note"] = midi_to_name(row["midi"])
        notes.append(row)

    notes = merge_tied_notes(notes)
    for note in notes:
        note.pop("staff", None)
    events = [{
        "type": "rest", "time": round(beat_to_seconds(rest["beat"]), 6),
        "duration": round(max(0.01, beat_to_seconds(rest["beat"] + rest["durationBeats"]) - beat_to_seconds(rest["beat"])), 6),
        "notes": [], "chord": "", "direction": "", "velocity": 0.0,
    } for rest in rest_events]
    if not notes:
        raise ValueError("The embedded MusicXML part contains no pitched notes.")
    pedals = [{
        "id": f"musicxml-pedal-{index}",
        "time": round(beat_to_seconds(row["beat"]) + float(row.get("delay", 0)), 6),
        "down": bool(row["down"]),
        "value": 127 if row["down"] else 0,
        "controller": 64,
        "source": "embedded-musicxml",
        "inferred": False,
    } for index, row in enumerate(pedal_rows)]
    result = {
        "isInstrumentalMusicSheet": True,
        "rejectionReason": "",
        "title": title,
        "composer": composer,
        "instrument": instrument,
        "bpm": tempo_events[0][1],
        "timeSignature": {"numerator": time_signature.numerator, "denominator": time_signature.denominator},
        "keySignature": key_name(fifths, mode),
        "notes": notes,
        "events": events,
        "tabs": [],
        "pedals": pedals,
        "warnings": [],
        "confidence": 0.995,
        "omrDiagnostics": {
            "engine": "embedded-musicxml",
            "source": source_name,
            "part": part_name,
            "measures": measure_count,
            "notes": len(notes),
            "tempoChanges": len(tempo_events),
            "printedPedalEvents": len(pedals),
        },
    }
    result = shape_piano_performance(result, infer_pedal=instrument == "piano")
    result["omrDiagnostics"]["pianoPerformance"] = result.get("pianoPerformance", {})
    return result


def _articulation(note: ET.Element) -> str:
    articulations = note.find("./notations/articulations")
    names = [
        node.tag.replace("-", " ")
        for node in (list(articulations) if articulations is not None else [])
    ]
    if note.find("./notations/slur[@type='start']") is not None:
        names.append("legato slur")
    return " ".join(names)[:80]
