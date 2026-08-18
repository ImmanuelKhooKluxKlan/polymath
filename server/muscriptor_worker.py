"""MuScriptor audio-to-ready-sheet worker for Polymath Musician.

The Node server supplies local paths only. Progress is emitted as JSON lines on
stdout so the web job can expose the model's native five-second chunk progress.
MuScriptor model weights remain subject to their CC BY-NC 4.0 license.
"""

import argparse
import json
from pathlib import Path

from muscriptor import TranscriptionModel


NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def midi_to_note(midi: int) -> str:
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


def emit(message: dict) -> None:
    print(json.dumps(message, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default="Uploaded recording")
    parser.add_argument("--instrument", default="band")
    parser.add_argument("--model", choices=("small", "medium", "large"), default="large")
    parser.add_argument("--instruments", default="")
    args = parser.parse_args()

    instruments = [value.strip() for value in args.instruments.split(",") if value.strip()] or None
    emit({"type": "stage", "stage": f"Loading MuScriptor {args.model.title()}"})
    model = TranscriptionModel.load_model(args.model)
    emit({"type": "stage", "stage": "Listening for notes and instruments"})

    starts = {}
    notes = []
    progress = {"completed": 0, "total": 0}

    for event in model.transcribe(args.input, instruments=instruments):
        if hasattr(event, "start_time") and hasattr(event, "pitch"):
            starts[int(event.index)] = event
        elif hasattr(event, "end_time") and hasattr(event, "start_event"):
            start = event.start_event
            start_time = max(0.0, float(start.start_time))
            end_time = max(start_time + 0.04, float(event.end_time))
            midi = int(start.pitch)
            notes.append({
                "midi": midi,
                "note": midi_to_note(midi),
                "time": round(start_time, 4),
                "duration": round(end_time - start_time, 4),
                "velocity": 0.78,
                "hand": "left" if midi < 60 else "right",
                "instrument": str(start.instrument),
                "source": f"muscriptor-{args.model}",
            })
            starts.pop(int(start.index), None)
        elif hasattr(event, "completed") and hasattr(event, "total"):
            progress = {"completed": int(event.completed), "total": int(event.total)}
            emit({"type": "progress", **progress})

    for start in starts.values():
        midi = int(start.pitch)
        notes.append({
            "midi": midi,
            "note": midi_to_note(midi),
            "time": round(max(0.0, float(start.start_time)), 4),
            "duration": 0.4,
            "velocity": 0.7,
            "hand": "left" if midi < 60 else "right",
            "instrument": str(start.instrument),
            "source": f"muscriptor-{args.model}",
        })

    notes.sort(key=lambda note: (note["time"], note["midi"], note["instrument"]))
    if not notes:
        raise RuntimeError("MuScriptor could not detect playable notes in this recording.")

    payload = {
        "title": args.title.strip() or "Uploaded recording",
        "composer": "MuScriptor transcription",
        "instrument": args.instrument,
        "bpm": 120,
        "notes": notes,
        "instrumentGroups": sorted({note["instrument"] for note in notes}),
        "sourceType": "muscriptor-audio-transcription",
        "readyToPlayFormat": "polymath-musician-json-v1",
        "transcriptionProvider": f"MuScriptor {args.model.title()}",
        "modelLicense": "CC-BY-NC-4.0",
        "progress": progress,
    }
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    emit({"type": "complete", "notes": len(notes), "instrumentGroups": payload["instrumentGroups"]})


if __name__ == "__main__":
    main()
