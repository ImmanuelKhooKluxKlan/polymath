'''RunPod Serverless worker for MuScriptor Large audio transcription.'''

from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path
from typing import Any

import runpod
from muscriptor import TranscriptionModel


MODEL_NAME = os.environ.get('MUSCRIPTOR_MODEL', 'large').strip().lower()
VOLUME_JOB_ROOT = Path(
    os.environ.get('MUSCRIPTOR_JOB_ROOT', '/runpod-volume/jobs')
).resolve()
NOTE_NAMES = ('C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B')
VALID_MODELS = {'small', 'medium', 'large'}

if MODEL_NAME not in VALID_MODELS:
    raise RuntimeError(f'Unsupported MuScriptor model: {MODEL_NAME}')


def load_model() -> TranscriptionModel:
    print(f'Loading MuScriptor {MODEL_NAME.title()} into GPU memory', flush=True)
    loaded = TranscriptionModel.load_model(MODEL_NAME)
    print(f'MuScriptor {MODEL_NAME.title()} is ready', flush=True)
    return loaded


MODEL = load_model()


def midi_to_note(midi: int) -> str:
    return f'{NOTE_NAMES[midi % 12]}{midi // 12 - 1}'


def safe_volume_job_path(value: Any) -> Path:
    candidate = Path(str(value or '')).resolve()
    if not candidate.is_relative_to(VOLUME_JOB_ROOT):
        raise ValueError('audio_path must be inside /runpod-volume/jobs')
    if candidate.suffix.lower() != '.wav':
        raise ValueError('Only prepared WAV files are accepted')
    if not candidate.is_file():
        raise FileNotFoundError('Prepared audio was not found on the network volume')
    return candidate


def temporary_audio_path(job: dict[str, Any], encoded: str) -> Path:
    job_id = re.sub(r'[^a-zA-Z0-9_-]+', '-', str(job.get('id') or 'local'))[:80]
    destination = Path('/tmp') / f'{job_id}.wav'
    try:
        audio = base64.b64decode(encoded, validate=True)
    except Exception as error:
        raise ValueError('audio_base64 is not valid Base64') from error
    if not audio or len(audio) > 20 * 1024 * 1024:
        raise ValueError('Embedded audio must be between 1 byte and 20 MB')
    destination.write_bytes(audio)
    return destination


def transcribe(job: dict[str, Any], job_input: dict[str, Any], audio_path: Path) -> dict[str, Any]:
    requested = job_input.get('instruments') or []
    instruments = [str(value).strip() for value in requested if str(value).strip()] or None
    runpod.serverless.progress_update(job, 'Listening for notes and instruments')

    starts: dict[int, Any] = {}
    notes: list[dict[str, Any]] = []
    progress = {'completed': 0, 'total': 0}

    for event in MODEL.transcribe(str(audio_path), instruments=instruments):
        if hasattr(event, 'start_time') and hasattr(event, 'pitch'):
            starts[int(event.index)] = event
        elif hasattr(event, 'end_time') and hasattr(event, 'start_event'):
            start = event.start_event
            start_time = max(0.0, float(start.start_time))
            end_time = max(start_time + 0.04, float(event.end_time))
            midi = int(start.pitch)
            notes.append({
                'midi': midi,
                'note': midi_to_note(midi),
                'time': round(start_time, 4),
                'duration': round(end_time - start_time, 4),
                'velocity': 0.78,
                'hand': 'left' if midi < 60 else 'right',
                'instrument': str(start.instrument),
                'source': f'muscriptor-{MODEL_NAME}-runpod-serverless',
            })
            starts.pop(int(start.index), None)
        elif hasattr(event, 'completed') and hasattr(event, 'total'):
            progress = {'completed': int(event.completed), 'total': int(event.total)}
            runpod.serverless.progress_update(
                job,
                f'Transcribing {progress["completed"]} of {progress["total"]} audio sections',
            )

    for start in starts.values():
        midi = int(start.pitch)
        notes.append({
            'midi': midi,
            'note': midi_to_note(midi),
            'time': round(max(0.0, float(start.start_time)), 4),
            'duration': 0.4,
            'velocity': 0.7,
            'hand': 'left' if midi < 60 else 'right',
            'instrument': str(start.instrument),
            'source': f'muscriptor-{MODEL_NAME}-runpod-serverless',
        })

    notes.sort(key=lambda note: (note['time'], note['midi'], note['instrument']))
    if not notes:
        raise RuntimeError('MuScriptor could not detect playable notes in this recording')

    return {
        'title': str(job_input.get('title') or 'Uploaded recording')[:120],
        'composer': 'MuScriptor transcription',
        'instrument': str(job_input.get('instrument') or 'band'),
        'bpm': 120,
        'notes': notes,
        'instrumentGroups': sorted({note['instrument'] for note in notes}),
        'sourceType': 'muscriptor-audio-transcription',
        'readyToPlayFormat': 'polymath-musician-json-v1',
        'transcriptionProvider': f'MuScriptor {MODEL_NAME.title()} on RunPod Serverless GPU',
        'modelLicense': 'CC-BY-NC-4.0',
        'progress': progress,
    }


def handler(job: dict[str, Any]) -> dict[str, Any]:
    job_input = job.get('input') or {}
    encoded = str(job_input.get('audio_base64') or '').strip()
    delete_audio = bool(job_input.get('delete_audio', True))
    if encoded:
        audio_path = temporary_audio_path(job, encoded)
        delete_audio = True
    else:
        audio_path = safe_volume_job_path(job_input.get('audio_path'))

    try:
        return transcribe(job, job_input, audio_path)
    finally:
        if delete_audio:
            audio_path.unlink(missing_ok=True)


if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})
