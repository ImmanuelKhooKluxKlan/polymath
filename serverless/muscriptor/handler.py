'''RunPod Serverless worker for MuScriptor Large audio transcription.'''

from __future__ import annotations

import base64
import gc
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

import runpod
from huggingface_hub import hf_hub_download
from muscriptor import TranscriptionModel


MODEL_NAME = os.environ.get('MUSCRIPTOR_MODEL', 'large').strip().lower()
MODEL_WEIGHTS_PATH = os.environ.get('MUSCRIPTOR_WEIGHTS_PATH', '').strip()
MODEL_SOURCE_VALUE = (
    os.environ.get('MUSCRIPTOR_MODEL_SOURCE', '').strip() or MODEL_WEIGHTS_PATH
)
VOLUME_JOB_ROOT = Path(
    os.environ.get('MUSCRIPTOR_JOB_ROOT', '/runpod-volume/jobs')
).resolve()
TRAINING_ROOT = Path('/runpod-volume/training').resolve()
ORIGINAL_MODEL_ROOT = Path('/runpod-volume/models/original').resolve()
TEST_MODEL_ROOT = Path('/runpod-volume/models/muscriptor-tester').resolve()
NOTE_NAMES = ('C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B')
VALID_MODELS = {'small', 'medium', 'large'}
BOOTSTRAP_REPO = 'MuScriptor/muscriptor-large'
BOOTSTRAP_REVISION = os.environ.get(
    'MUSCRIPTOR_BOOTSTRAP_REVISION',
    '8809fdfbed2affa7ade94a7059e746e3880720e7',
).strip()

if not MODEL_SOURCE_VALUE and MODEL_NAME not in VALID_MODELS:
    raise RuntimeError(f'Unsupported Polymath model: {MODEL_NAME}')

if MODEL_SOURCE_VALUE:
    if MODEL_SOURCE_VALUE.startswith('hf://'):
        if not MODEL_SOURCE_VALUE.lower().endswith('.safetensors'):
            raise RuntimeError('MUSCRIPTOR_MODEL_SOURCE must identify a .safetensors file')
        MODEL_SOURCE = MODEL_SOURCE_VALUE
        MODEL_LABEL = f'custom Hugging Face model ({MODEL_SOURCE_VALUE[5:]})'
    else:
        MODEL_SOURCE = Path(MODEL_SOURCE_VALUE).expanduser().resolve()
        if MODEL_SOURCE.suffix.lower() != '.safetensors':
            raise RuntimeError('Custom Polymath weights must be a .safetensors file')
        if not MODEL_SOURCE.is_file():
            raise RuntimeError(f'Custom Polymath weights were not found: {MODEL_SOURCE}')
        if not MODEL_SOURCE.with_name('config.json').is_file():
            raise RuntimeError('Custom Polymath weights require config.json in the same folder')
        MODEL_LABEL = f'custom volume model ({MODEL_SOURCE.name})'
    MODEL_SOURCE_ID = 'muscriptor-custom-runpod-serverless'
    MODEL_PROVIDER = 'Custom Polymath model on RunPod Serverless GPU'
else:
    MODEL_SOURCE = MODEL_NAME
    MODEL_LABEL = MODEL_NAME.title()
    MODEL_SOURCE_ID = f'muscriptor-{MODEL_NAME}-runpod-serverless'
    MODEL_PROVIDER = f'Polymath {MODEL_NAME.title()} on RunPod Serverless GPU'


def load_model() -> TranscriptionModel:
    print(f'Loading Polymath {MODEL_LABEL} into GPU memory', flush=True)
    loaded = TranscriptionModel.load_model(MODEL_SOURCE)
    print(f'Polymath {MODEL_LABEL} is ready', flush=True)
    return loaded


MODEL: TranscriptionModel | None = None


def get_model() -> TranscriptionModel:
    global MODEL
    if MODEL is None:
        MODEL = load_model()
    return MODEL


def copy_checkpoint_file(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size == source.stat().st_size:
            return 'verified-existing'
        raise RuntimeError(f'Refusing to overwrite mismatched checkpoint: {destination}')
    temporary = destination.with_name(f'.{destination.name}.{os.getpid()}.uploading')
    try:
        shutil.copy2(source, temporary)
        if temporary.stat().st_size != source.stat().st_size:
            raise RuntimeError(f'Checkpoint copy verification failed: {destination}')
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return 'created'


def bootstrap_model_copies(job: dict[str, Any], job_input: dict[str, Any]) -> dict[str, Any]:
    version = str(job_input.get('version') or 'v001').strip().lower()
    if not re.fullmatch(r'v\d{3,}', version):
        raise ValueError('Bootstrap version must look like v001, v002, and so on')

    runpod.serverless.progress_update(job, 'Locating cached Polymath Large weights')
    weights = Path(hf_hub_download(
        repo_id=BOOTSTRAP_REPO,
        filename='model.safetensors',
        revision=BOOTSTRAP_REVISION,
    ))
    config = Path(hf_hub_download(
        repo_id=BOOTSTRAP_REPO,
        filename='config.json',
        revision=BOOTSTRAP_REVISION,
    ))

    original = Path('/runpod-volume/models/original')
    tester = Path('/runpod-volume/models/muscriptor-tester') / version
    runpod.serverless.progress_update(job, 'Creating immutable original checkpoint')
    results = {
        str(original / weights.name): copy_checkpoint_file(weights, original / weights.name),
        str(original / config.name): copy_checkpoint_file(config, original / config.name),
    }
    runpod.serverless.progress_update(job, f'Creating tester checkpoint {version}')
    results.update({
        str(tester / weights.name): copy_checkpoint_file(original / weights.name, tester / weights.name),
        str(tester / config.name): copy_checkpoint_file(original / config.name, tester / config.name),
    })
    return {
        'action': 'bootstrap-model-copies',
        'model': BOOTSTRAP_REPO,
        'revision': BOOTSTRAP_REVISION,
        'version': version,
        'weightsBytes': weights.stat().st_size,
        'files': results,
        'testerWeightsPath': str(tester / weights.name),
    }


def safe_training_file(dataset_id: str, filename: str) -> Path:
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{2,50}', dataset_id):
        raise ValueError('Training dataset id contains unsupported characters')
    candidate = (TRAINING_ROOT / dataset_id / filename).resolve()
    if not candidate.is_relative_to(TRAINING_ROOT) or not candidate.is_file():
        raise FileNotFoundError(f'Training file was not found: {filename}')
    return candidate


def resolve_training_base(version: Any) -> tuple[Path, Path, str]:
    """Resolve an immutable original or an append-only tester checkpoint.

    Continuing experiments from the current incumbent is essential: restarting
    every phase from the public MuScriptor checkpoint discards improvements made
    by earlier phases.  Only version-shaped directories below TEST_MODEL_ROOT
    are accepted, so a job cannot use an arbitrary filesystem path.
    """

    requested = str(version or 'original').strip().lower()
    if requested in {'', 'original'}:
        root = ORIGINAL_MODEL_ROOT.resolve()
        label = 'original'
    else:
        if not re.fullmatch(r'phase\d+-v\d{3,}', requested):
            raise ValueError('Base version must be original or look like phase1-v001')
        root = (TEST_MODEL_ROOT / requested).resolve()
        if not root.is_relative_to(TEST_MODEL_ROOT):
            raise ValueError('Base checkpoint escaped the tester model directory')
        label = requested

    weights = root / 'model.safetensors'
    config = root / 'config.json'
    if not weights.is_file() or not config.is_file():
        raise FileNotFoundError(f'Base checkpoint is incomplete: {label}')
    return weights, config, label


def train_piano_candidate(job: dict[str, Any], job_input: dict[str, Any]) -> dict[str, Any]:
    from argparse import Namespace

    import torch

    from ml.training.train_muscriptor_piano import (
        RIGHTS_ACKNOWLEDGEMENT,
        train,
    )

    dataset_id = str(job_input.get('dataset_id') or '').strip().lower()
    version = str(job_input.get('version') or '').strip().lower()
    if not re.fullmatch(r'phase\d+-v\d{3,}', version):
        raise ValueError('Training version must look like phase1-v001')
    if str(job_input.get('rights_acknowledgement') or '') != RIGHTS_ACKNOWLEDGEMENT:
        raise ValueError('The explicit private-training rights acknowledgement is missing')

    epochs = int(job_input.get('epochs') or 1)
    train_last_layers = int(job_input.get('train_last_layers') or 1)
    learning_rate = float(job_input.get('learning_rate') or 2e-6)
    if epochs < 1 or epochs > 3:
        raise ValueError('Experimental training accepts between one and three epochs')
    if train_last_layers < 1 or train_last_layers > 2:
        raise ValueError('Experimental training accepts one or two final layers')
    if learning_rate <= 0 or learning_rate > 1e-5:
        raise ValueError('Experimental learning rate must be above zero and at most 1e-5')

    train_manifest = safe_training_file(dataset_id, 'prepared-train.jsonl')
    validation_manifest = safe_training_file(dataset_id, 'prepared-validation.jsonl')
    base, config, base_version = resolve_training_base(job_input.get('base_version'))
    output = (TEST_MODEL_ROOT / version).resolve()
    if not output.is_relative_to(TEST_MODEL_ROOT):
        raise ValueError('Candidate output escaped the tester model directory')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Candidate version already exists: {version}')

    global MODEL
    MODEL = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    runpod.serverless.progress_update(
        job,
        f'Auditing clips and loading append-only base checkpoint {base_version}',
    )

    args = Namespace(
        train_manifest=train_manifest,
        validation_manifest=validation_manifest,
        base=base,
        out=output,
        execute=True,
        rights_acknowledgement=RIGHTS_ACKNOWLEDGEMENT,
        # The CLI and worker must enforce the same generalization floor.  Tiny
        # two/three-song runs are useful only as manual overfit smoke tests and
        # repeatedly produced candidates that lowered loss while regressing
        # decoded piano quality.
        minimum_train_songs=20,
        train_last_layers=train_last_layers,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=0.01,
        gradient_accumulation=8,
        gradient_clip_norm=1.0,
        timing_token_weight=float(job_input.get('timing_token_weight') or 1.15),
        note_off_token_weight=float(job_input.get('note_off_token_weight') or 1.25),
        eos_token_weight=float(job_input.get('eos_token_weight') or 1.20),
        precision='bf16',
        seed=f'polymath-{version}',
    )
    metadata = train(
        args,
        progress_callback=lambda message: runpod.serverless.progress_update(job, message),
    )
    runpod.serverless.progress_update(job, f'Candidate {version} saved; original remains unchanged')
    return {
        'action': 'train-piano-candidate',
        'datasetId': dataset_id,
        'version': version,
        'baseVersion': base_version,
        'candidatePath': str(output),
        'commercialUseAllowed': False,
        'metadata': metadata,
    }


def evaluate_piano_candidate(job: dict[str, Any], job_input: dict[str, Any]) -> dict[str, Any]:
    from ml.training.evaluate_checkpoint import compare_checkpoints, save_comparison
    from ml.training.muscriptor_tokens import canonical_instrument_name

    dataset_id = str(job_input.get('dataset_id') or '').strip().lower()
    version = str(job_input.get('version') or '').strip().lower()
    instrument = canonical_instrument_name(
        job_input.get('instrument') or 'acoustic_piano'
    )
    if not re.fullmatch(r'phase\d+-v\d{3,}', version):
        raise ValueError('Evaluation version must look like phase1-v001')
    validation_manifest = safe_training_file(dataset_id, 'prepared-validation.jsonl')
    base, _config, baseline_version = resolve_training_base(
        job_input.get('baseline_version')
    )
    candidate_root = (TEST_MODEL_ROOT / version).resolve()
    candidate = candidate_root / 'model.safetensors'
    if not candidate_root.is_relative_to(TEST_MODEL_ROOT):
        raise ValueError('Candidate path escaped the tester model directory')
    if not base.is_file() or not candidate.is_file():
        raise FileNotFoundError('Original or candidate checkpoint is missing')

    global MODEL
    MODEL = None
    gc.collect()
    result = compare_checkpoints(
        base,
        candidate,
        validation_manifest,
        progress_callback=lambda message: runpod.serverless.progress_update(job, message),
        instruments=(instrument,),
    )
    destination = candidate_root / f'evaluation-{dataset_id}.json'
    save_comparison(result, destination)
    # Full raw per-clip predictions stay on the private network volume for
    # reproducible pattern analysis. Keep the serverless response compact.
    response_result = {
        **result,
        'baseline': {key: value for key, value in result['baseline'].items() if key != 'decodedClips'},
        'candidate': {key: value for key, value in result['candidate'].items() if key != 'decodedClips'},
    }
    return {
        'action': 'evaluate-piano-candidate',
        'datasetId': dataset_id,
        'version': version,
        'baselineVersion': baseline_version,
        'instrument': instrument,
        'evaluationPath': str(destination),
        **response_result,
    }


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

    model = get_model()
    for event in model.transcribe(str(audio_path), instruments=instruments):
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
                'source': MODEL_SOURCE_ID,
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
            'source': MODEL_SOURCE_ID,
        })

    notes.sort(key=lambda note: (note['time'], note['midi'], note['instrument']))
    if not notes:
        raise RuntimeError('Polymath could not detect playable notes in this recording')

    return {
        'title': str(job_input.get('title') or 'Uploaded recording')[:120],
        'composer': 'Polymath transcription',
        'instrument': str(job_input.get('instrument') or 'band'),
        'bpm': 120,
        'notes': notes,
        'instrumentGroups': sorted({note['instrument'] for note in notes}),
        'sourceType': 'muscriptor-audio-transcription',
        'readyToPlayFormat': 'polymath-musician-json-v1',
        'transcriptionProvider': MODEL_PROVIDER,
        'modelLicense': 'CC-BY-NC-4.0',
        'progress': progress,
    }


def handler(job: dict[str, Any]) -> dict[str, Any]:
    job_input = job.get('input') or {}
    if job_input.get('action') == 'bootstrap_model_copies':
        return bootstrap_model_copies(job, job_input)
    if job_input.get('action') == 'train_piano_candidate':
        return train_piano_candidate(job, job_input)
    if job_input.get('action') == 'evaluate_piano_candidate':
        return evaluate_piano_candidate(job, job_input)
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
