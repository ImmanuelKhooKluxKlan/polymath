#!/usr/bin/env python3
"""Run one MuScriptor checkpoint over a reproducible batch of audio files.

This is deliberately independent from the Serverless handler.  It lets a
temporary research Pod create baseline/candidate predictions without changing
the live endpoint or embedding audio in API requests.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from muscriptor import TranscriptionModel


NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def midi_to_note(midi: int) -> str:
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


def transcribe_one(
    model: TranscriptionModel,
    audio_path: Path,
    *,
    title: str,
    instruments: list[str] | None,
    model_id: str,
    provider: str,
) -> dict[str, Any]:
    starts: dict[int, Any] = {}
    notes: list[dict[str, Any]] = []
    progress = {"completed": 0, "total": 0}
    started_at = time.monotonic()

    for event in model.transcribe(str(audio_path), instruments=instruments or None):
        if hasattr(event, "start_time") and hasattr(event, "pitch"):
            starts[int(event.index)] = event
        elif hasattr(event, "end_time") and hasattr(event, "start_event"):
            start = event.start_event
            start_time = max(0.0, float(start.start_time))
            end_time = max(start_time + 0.04, float(event.end_time))
            midi = int(start.pitch)
            notes.append(
                {
                    "midi": midi,
                    "note": midi_to_note(midi),
                    "time": round(start_time, 4),
                    "duration": round(end_time - start_time, 4),
                    "velocity": 0.78,
                    "hand": "left" if midi < 60 else "right",
                    "instrument": str(start.instrument),
                    "source": model_id,
                }
            )
            starts.pop(int(start.index), None)
        elif hasattr(event, "completed") and hasattr(event, "total"):
            progress = {"completed": int(event.completed), "total": int(event.total)}
            print(
                f"[{title}] {progress['completed']}/{progress['total']} sections",
                flush=True,
            )

    for start in starts.values():
        midi = int(start.pitch)
        notes.append(
            {
                "midi": midi,
                "note": midi_to_note(midi),
                "time": round(max(0.0, float(start.start_time)), 4),
                "duration": 0.4,
                "velocity": 0.7,
                "hand": "left" if midi < 60 else "right",
                "instrument": str(start.instrument),
                "source": model_id,
            }
        )

    notes.sort(key=lambda note: (note["time"], note["midi"], note["instrument"]))
    if not notes:
        raise RuntimeError(f"MuScriptor detected no playable notes in {audio_path}")
    elapsed = round(time.monotonic() - started_at, 3)
    return {
        "title": title[:120],
        "composer": "MuScriptor transcription",
        "instrument": "band",
        "bpm": 120,
        "notes": notes,
        "instrumentGroups": sorted({note["instrument"] for note in notes}),
        "sourceType": "muscriptor-audio-transcription",
        "readyToPlayFormat": "polymath-musician-json-v1",
        "transcriptionProvider": provider,
        "modelLicense": "CC-BY-NC-4.0",
        "progress": progress,
        "researchProvenance": {
            "modelId": model_id,
            "audioPath": str(audio_path),
            "inferenceSeconds": elapsed,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--provider", default="MuScriptor research Pod")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip complete outputs from this exact model ID and continue the batch",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    items = json.loads(args.manifest.read_text("utf-8"))
    if not isinstance(items, list) or not items:
        raise ValueError("manifest must contain a non-empty JSON array")

    print(f"Loading checkpoint {args.model_id}: {args.model}", flush=True)
    model = TranscriptionModel.load_model(args.model)
    print(f"Loaded checkpoint {args.model_id}", flush=True)

    summary: list[dict[str, Any]] = []
    for index, item in enumerate(items, start=1):
        audio_path = Path(item["audio"]).expanduser().resolve()
        output_path = Path(item["output"]).expanduser().resolve()
        title = str(item.get("title") or audio_path.stem)
        instruments = [str(value) for value in item.get("instruments") or []]
        if output_path.exists() and args.resume:
            existing = json.loads(output_path.read_text("utf-8"))
            existing_notes = existing.get("notes") or []
            existing_model = (
                existing.get("researchProvenance") or {}
            ).get("modelId")
            if existing_notes and existing_model == args.model_id:
                summary.append(
                    {
                        "title": title,
                        "output": str(output_path),
                        "notes": len(existing_notes),
                        "status": "verified-existing",
                    }
                )
                print(f"skipping verified existing output {output_path}", flush=True)
                continue
            raise RuntimeError(
                f"resume found an untrusted or mismatched output: {output_path}"
            )
        if output_path.exists() and not args.force:
            raise FileExistsError(f"refusing to overwrite {output_path}")
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[{index}/{len(items)}] {title}", flush=True)
        result = transcribe_one(
            model,
            audio_path,
            title=title,
            instruments=instruments,
            model_id=args.model_id,
            provider=args.provider,
        )
        temporary = output_path.with_suffix(output_path.suffix + ".partial")
        temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
        temporary.replace(output_path)
        summary.append(
            {
                "title": title,
                "output": str(output_path),
                "notes": len(result["notes"]),
                "inferenceSeconds": result["researchProvenance"]["inferenceSeconds"],
            }
        )
        print(
            f"wrote {output_path} ({len(result['notes'])} notes)",
            flush=True,
        )

    print(json.dumps({"completed": summary}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
