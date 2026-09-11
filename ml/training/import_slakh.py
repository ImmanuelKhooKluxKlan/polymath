"""Import aligned Slakh mixtures as auditable piano-route supervision.

Slakh distributes the exact per-stem MIDI used to synthesize each audio stem.
That lets us build timing labels without fitting an alignment to model output.
The factual multi-instrument MIDI can either supervise the real piano stems or
pass through a frozen arranger profile to create a deterministic piano-reduction
teacher target.  This script never trains a checkpoint; it only creates source
JSON, target JSON, supervision packages, and a leakage-safe training index.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import wave
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "polymath-slakh-import-v1"
SUPERVISION_SCHEMA = "polymath-supervision-package-v1"
LICENSE_ID = "CC-BY-4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
DATASET_URL = "https://zenodo.org/records/4603870"
DEFAULT_SPLIT_COUNTS = (16, 2, 2)
WINDOW_SECONDS = 5.0


class SlakhImportError(ValueError):
    """Raised when a Slakh track cannot become trustworthy supervision."""


@dataclass(frozen=True)
class StemInfo:
    stem_id: str
    midi_path: Path
    program: int
    instrument: str
    is_drum: bool


def sha256_file(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_group_for_program(program: int, *, is_drum: bool = False) -> str:
    """Map a zero-based General MIDI program to MuScriptor's public groups."""
    if is_drum:
        return "drums"
    if not 0 <= program <= 127:
        raise SlakhImportError(f"MIDI program must be in [0, 127], got {program}")
    if program <= 3:
        return "acoustic_piano"
    if program <= 7:
        return "electric_piano"
    if program <= 15:
        return "chromatic_percussion"
    if program <= 23:
        return "organ"
    if program <= 25:
        return "acoustic_guitar"
    if program <= 28:
        return "clean_electric_guitar"
    if program <= 31:
        return "distorted_electric_guitar"
    if program == 32:
        return "acoustic_bass"
    if program <= 39:
        return "electric_bass"
    if program == 40:
        return "violin"
    if program == 41:
        return "viola"
    if program == 42:
        return "cello"
    if program == 43:
        return "contrabass"
    if program in {44, 45, 48, 49}:
        return "string_ensemble"
    if program == 46:
        return "orchestral_harp"
    if program == 47:
        return "timpani"
    if program <= 51:
        return "synth_strings"
    if program <= 54:
        return "voice"
    if program == 55:
        return "orchestra_hit"
    if program == 56:
        return "trumpet"
    if program == 57:
        return "trombone"
    if program == 58:
        return "tuba"
    if program == 60:
        return "french_horn"
    if program <= 63:
        return "brass_section"
    if program <= 65:
        return "soprano_and_alto_sax"
    if program == 66:
        return "tenor_sax"
    if program == 67:
        return "baritone_sax"
    if program == 68:
        return "oboe"
    if program == 69:
        return "english_horn"
    if program == 70:
        return "bassoon"
    if program == 71:
        return "clarinet"
    if program <= 79:
        return "flutes"
    if program <= 87:
        return "synth_lead"
    return "synth_pad"


def wave_duration_seconds(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as handle:
            frames = handle.getnframes()
            rate = handle.getframerate()
    except (OSError, wave.Error) as exc:
        raise SlakhImportError(f"Could not read WAV duration for {path}: {exc}") from exc
    if rate <= 0 or frames <= 0:
        raise SlakhImportError(f"WAV contains no usable samples: {path}")
    return frames / rate


def read_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - exercised on training images
        raise SlakhImportError("PyYAML is required; install requirements-training.txt") from exc
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise SlakhImportError(f"Could not parse {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SlakhImportError(f"Metadata must be an object: {path}")
    return payload


def discover_stems(track_directory: Path, metadata: dict[str, Any]) -> list[StemInfo]:
    raw_stems = metadata.get("stems")
    if not isinstance(raw_stems, dict) or not raw_stems:
        raise SlakhImportError(f"No stems recorded in {track_directory / 'metadata.yaml'}")
    midi_directory = track_directory / str(metadata.get("midi_dir") or "MIDI")
    audio_directory = track_directory / str(metadata.get("audio_dir") or "stems")
    stems: list[StemInfo] = []
    for stem_id, raw in sorted(raw_stems.items()):
        if not isinstance(raw, dict):
            continue
        midi_path = midi_directory / f"{stem_id}.mid"
        audio_path = audio_directory / f"{stem_id}.wav"
        # BabySlakh's converted prototype contains stale false values for
        # midi_saved/audio_rendered even though both files are present. The
        # immutable artifacts are stronger evidence than those legacy flags.
        # Requiring both also prevents a MIDI-only source from becoming a label
        # for sound that was never included in the mixture.
        if not midi_path.is_file() or not audio_path.is_file():
            continue
        try:
            program = int(raw.get("program_num", 0))
        except (TypeError, ValueError) as exc:
            raise SlakhImportError(f"{track_directory.name}/{stem_id} has no valid program") from exc
        is_drum = bool(raw.get("is_drum", False))
        stems.append(
            StemInfo(
                stem_id=str(stem_id),
                midi_path=midi_path,
                program=program,
                instrument=canonical_group_for_program(program, is_drum=is_drum),
                is_drum=is_drum,
            )
        )
    if not stems:
        raise SlakhImportError(f"No rendered MIDI stems found in {track_directory}")
    return stems


def _finish_note(
    notes: list[dict[str, Any]],
    active: dict[tuple[int, int], list[tuple[float, int]]],
    key: tuple[int, int],
    end: float,
    stem: StemInfo,
) -> None:
    starts = active.get(key)
    if not starts:
        return
    start, velocity = starts.pop(0)
    if not starts:
        active.pop(key, None)
    duration = end - start
    if duration < 0.01:
        return
    notes.append(
        {
            "midi": key[1],
            "time": round(start, 6),
            "duration": round(duration, 6),
            "velocity": round(max(0.05, min(1.0, velocity / 127.0)), 4),
            "instrument": stem.instrument,
            "sourceStem": stem.stem_id,
            "sourceProgram": stem.program,
        }
    )


def read_midi_notes(stem: StemInfo, maximum_duration: float) -> list[dict[str, Any]]:
    try:
        import mido
    except ImportError as exc:  # pragma: no cover - exercised on training images
        raise SlakhImportError("mido is required; install requirements-training.txt") from exc
    try:
        midi = mido.MidiFile(stem.midi_path)
        messages = mido.merge_tracks(midi.tracks)
    except (OSError, EOFError, ValueError) as exc:
        raise SlakhImportError(f"Could not parse MIDI {stem.midi_path}: {exc}") from exc
    tempo = 500_000
    elapsed = 0.0
    active: dict[tuple[int, int], list[tuple[float, int]]] = defaultdict(list)
    notes: list[dict[str, Any]] = []
    for message in messages:
        elapsed += mido.tick2second(message.time, midi.ticks_per_beat, tempo)
        if message.type == "set_tempo":
            tempo = message.tempo
            continue
        if message.type not in {"note_on", "note_off"}:
            continue
        channel = int(getattr(message, "channel", 0))
        pitch = int(message.note)
        key = (channel, pitch)
        is_on = message.type == "note_on" and int(message.velocity) > 0
        if is_on:
            # A physical key cannot have two independent active strikes. Close
            # a malformed overlap at the retrigger before opening the new one.
            while active.get(key):
                _finish_note(notes, active, key, elapsed, stem)
            active[key].append((elapsed, int(message.velocity)))
        else:
            _finish_note(notes, active, key, elapsed, stem)
    safe_end = min(maximum_duration, max(elapsed, 0.01))
    for key in list(active):
        while active.get(key):
            _finish_note(notes, active, key, safe_end, stem)
    return [
        note
        for note in sorted(notes, key=lambda item: (item["time"], item["midi"], item["sourceStem"]))
        if 0 <= note["time"] < maximum_duration and 0 <= note["midi"] <= 127
    ]


def factual_payload(track_id: str, duration: float, stems: Iterable[StemInfo]) -> dict[str, Any]:
    notes: list[dict[str, Any]] = []
    for stem in stems:
        notes.extend(read_midi_notes(stem, duration))
    notes.sort(key=lambda item: (item["time"], item["midi"], item["sourceStem"]))
    return {
        "schema": SCHEMA,
        "title": track_id,
        "sourceType": "slakh-exact-aligned-midi",
        "notes": notes,
        "instrumentGroups": sorted({note["instrument"] for note in notes}),
        "timeline": {"durationSeconds": round(duration, 6)},
    }


def piano_stem_target(source: dict[str, Any]) -> dict[str, Any]:
    notes = [
        {
            **note,
            "instrument": "acoustic_piano",
        }
        for note in source.get("notes", [])
        if note.get("instrument") in {"acoustic_piano", "electric_piano"}
    ]
    return {
        **source,
        "sourceType": "slakh-exact-piano-stem-target",
        "instrument": "piano",
        "instrumentGroups": ["acoustic_piano"],
        "notes": notes,
    }


def arranger_target(
    source: dict[str, Any],
    profile_path: Path,
    density_multiplier: float | None,
) -> dict[str, Any]:
    server_root = Path(__file__).resolve().parents[2] / "server"
    if str(server_root) not in sys.path:
        sys.path.insert(0, str(server_root))
    from piano_arranger import arrange_payload

    profile = json.loads(profile_path.read_text(encoding="utf-8-sig"))
    if density_multiplier is not None:
        profile.setdefault("decoder", {})["preCleanupDensityMultiplier"] = max(
            0.5, min(3.0, density_multiplier)
        )
    arranged = arrange_payload(source, mode="instrumental", style_profile=profile)
    arranged["sourceType"] = "slakh-deterministic-piano-reduction-target"
    return arranged


def target_notes_for_training(target: dict[str, Any], duration: float) -> list[dict[str, Any]]:
    notes: list[dict[str, Any]] = []
    for item in target.get("notes", []):
        try:
            pitch = int(round(float(item["midi"])))
            start = max(0.0, float(item["time"]))
            note_duration = max(0.01, float(item.get("duration", 0.2)))
            velocity = max(0.05, min(1.0, float(item.get("velocity", 0.75))))
        except (KeyError, TypeError, ValueError):
            continue
        if not 0 <= pitch <= 127 or start >= duration:
            continue
        end = min(duration, start + note_duration)
        window_index = int(start // WINDOW_SECONDS)
        notes.append(
            {
                "midi": pitch,
                "time": round(start, 6),
                "duration": round(max(0.01, end - start), 6),
                "velocity": round(velocity, 4),
                "instrument": "acoustic_piano",
                "qualityStatus": "trusted",
                "qualityWindowId": f"w{window_index:05d}",
                "trainingEligible": True,
            }
        )
    return sorted(notes, key=lambda item: (item["time"], item["midi"]))


def exact_quality_windows(duration: float) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    index = 0
    start = 0.0
    while start < duration - 1e-7:
        end = min(duration, start + WINDOW_SECONDS)
        windows.append(
            {
                "id": f"w{index:05d}",
                "sourceStart": round(start, 6),
                "sourceEnd": round(end, 6),
                "referenceStart": round(start, 6),
                "referenceEnd": round(end, 6),
                "status": "trusted",
                "trainingEligible": True,
                "reason": "Exact stem MIDI used by Slakh to synthesize the aligned audio.",
            }
        )
        index += 1
        start = end
    return windows


def supervision_package(
    track_id: str,
    mix_path: Path,
    target: dict[str, Any],
    duration: float,
    target_mode: str,
) -> dict[str, Any]:
    notes = target_notes_for_training(target, duration)
    if not notes:
        raise SlakhImportError(f"{track_id} produced no piano target notes")
    return {
        "schema": SUPERVISION_SCHEMA,
        "songId": f"babyslakh-{track_id.lower()}",
        "timeline": {
            "sourceDurationSeconds": round(duration, 6),
            "referenceDurationSeconds": round(duration, 6),
            "hardOuterClock": "source-audio",
        },
        "alignment": {
            "method": "slakh-exact-synthesis-midi",
            "qualityWindows": exact_quality_windows(duration),
            "metrics": {
                "alignmentCoverage": 1.0,
                "timingGroundTruth": True,
                "targetMode": target_mode,
            },
        },
        "provenance": {
            "dataset": "BabySlakh",
            "datasetUrl": DATASET_URL,
            "sourceMedia": str(mix_path.resolve()),
            "sourceSha256": sha256_file(mix_path),
            "license": LICENSE_ID,
            "licenseUrl": LICENSE_URL,
            "attribution": "Manilow et al., Cutting Music Source Separation Some Slakh (WASPAA 2019)",
            "labelPolicy": target_mode,
        },
        "notes": notes,
    }


def composition_identity(track_directory: Path, metadata: dict[str, Any]) -> str:
    uuid = str(metadata.get("UUID") or metadata.get("uuid") or "").strip().lower()
    if uuid:
        return f"uuid:{uuid}"
    all_sources = track_directory / "all_src.mid"
    if all_sources.is_file():
        return f"sha256:{sha256_file(all_sources)}"
    raise SlakhImportError(f"{track_directory.name} has no UUID or all_src.mid identity")


def fixed_splits(track_ids: list[str], train_count: int, validation_count: int) -> dict[str, str]:
    if train_count < 1 or validation_count < 1:
        raise SlakhImportError("Train and validation counts must both be positive")
    if train_count + validation_count >= len(track_ids):
        raise SlakhImportError("Split counts must leave at least one test song")
    ordered = sorted(track_ids)
    return {
        track_id: (
            "train"
            if index < train_count
            else "validation"
            if index < train_count + validation_count
            else "test"
        )
        for index, track_id in enumerate(ordered)
    }


def import_dataset(
    dataset_root: Path,
    output_root: Path,
    *,
    target_mode: str,
    profile_path: Path | None,
    density_multiplier: float | None,
    train_count: int,
    validation_count: int,
) -> dict[str, Any]:
    track_directories = sorted(
        path for path in dataset_root.rglob("Track[0-9][0-9][0-9][0-9][0-9]")
        if path.is_dir() and (path / "metadata.yaml").is_file()
    )
    if not track_directories:
        raise SlakhImportError(f"No Slakh TrackXXXXX directories found below {dataset_root}")
    records: list[tuple[Path, dict[str, Any], str]] = []
    identities: dict[str, str] = {}
    duplicate_tracks: list[dict[str, str]] = []
    for track_directory in track_directories:
        metadata = read_yaml(track_directory / "metadata.yaml")
        identity = composition_identity(track_directory, metadata)
        if identity in identities:
            duplicate_tracks.append(
                {
                    "track": track_directory.name,
                    "duplicateOf": identities[identity],
                    "identity": identity,
                }
            )
            continue
        identities[identity] = track_directory.name
        records.append((track_directory, metadata, identity))
    split_by_track = fixed_splits(
        [track_directory.name for track_directory, _metadata, _identity in records],
        train_count,
        validation_count,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    index_entries: list[dict[str, Any]] = []
    imported: list[dict[str, Any]] = []
    for track_directory, metadata, identity in records:
        track_id = track_directory.name
        mix_path = track_directory / "mix.wav"
        if not mix_path.is_file():
            raise SlakhImportError(f"Missing mixture WAV: {mix_path}")
        duration = wave_duration_seconds(mix_path)
        stems = discover_stems(track_directory, metadata)
        source = factual_payload(track_id, duration, stems)
        if target_mode == "piano-stems":
            target = piano_stem_target(source)
        elif target_mode == "arranger-teacher":
            if profile_path is None:
                raise SlakhImportError("--profile is required for arranger-teacher targets")
            target = arranger_target(source, profile_path, density_multiplier)
        else:  # pragma: no cover - argparse prevents this
            raise SlakhImportError(f"Unsupported target mode: {target_mode}")
        track_output = output_root / track_id
        track_output.mkdir(parents=True, exist_ok=True)
        source_path = track_output / "factual-source.json"
        target_path = track_output / "piano-target.json"
        package_path = track_output / "supervision-package.json"
        source_path.write_text(json.dumps(source, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        target_path.write_text(json.dumps(target, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        package = supervision_package(track_id, mix_path, target, duration, target_mode)
        package_path.write_text(json.dumps(package, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        split = split_by_track[track_id]
        index_entries.append(
            {
                "songId": f"babyslakh-{track_id.lower()}",
                "sourceMedia": str(mix_path.resolve()),
                "supervisionPackage": str(package_path.resolve()),
                "instrumentFocus": "acoustic_piano",
                "split": split,
                "rights": {
                    "allowedForTraining": True,
                    "note": f"BabySlakh {LICENSE_ID}; {DATASET_URL}; attribution recorded in package.",
                },
            }
        )
        imported.append(
            {
                "track": track_id,
                "compositionIdentity": identity,
                "split": split,
                "durationSeconds": round(duration, 3),
                "sourceNotes": len(source["notes"]),
                "targetNotes": len(package["notes"]),
                "stems": len(stems),
            }
        )
    training_index = {
        "schema": "polymath-training-index-v1",
        "dataset": "BabySlakh",
        "licence": {"id": LICENSE_ID, "url": LICENSE_URL, "source": DATASET_URL},
        "songs": index_entries,
    }
    index_path = output_root / "training-index.json"
    index_path.write_text(json.dumps(training_index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        "schema": SCHEMA,
        "datasetRoot": str(dataset_root.resolve()),
        "outputRoot": str(output_root.resolve()),
        "trainingIndex": str(index_path.resolve()),
        "targetMode": target_mode,
        "profile": str(profile_path.resolve()) if profile_path else None,
        "densityMultiplier": density_multiplier,
        "tracksDiscovered": len(track_directories),
        "uniqueCompositions": len(records),
        "duplicatesSkipped": duplicate_tracks,
        "splitCounts": {
            split: sum(item["split"] == split for item in imported)
            for split in ("train", "validation", "test")
        },
        "tracks": imported,
        "licence": {"id": LICENSE_ID, "url": LICENSE_URL, "source": DATASET_URL},
        "leakagePolicy": "Duplicate composition identities are skipped before fixed song-level splits.",
    }
    (output_root / "import-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--target-mode",
        choices=("piano-stems", "arranger-teacher"),
        default="arranger-teacher",
    )
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--density-multiplier", type=float)
    parser.add_argument("--train-count", type=int, default=DEFAULT_SPLIT_COUNTS[0])
    parser.add_argument("--validation-count", type=int, default=DEFAULT_SPLIT_COUNTS[1])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        summary = import_dataset(
            args.dataset_root.resolve(),
            args.output_root.resolve(),
            target_mode=args.target_mode,
            profile_path=args.profile.resolve() if args.profile else None,
            density_multiplier=args.density_multiplier,
            train_count=args.train_count,
            validation_count=args.validation_count,
        )
    except SlakhImportError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
