"""Resolve a precommitted singer-melody holdout without inspecting audio.

The MDB-melody-synth archive contains full mixes and one resynthesized melody
annotation per track, but the archive filename alone does not say whether that
melody is a singer or an instrument.  This resolver joins archive membership to
the official MedleyDB YAML metadata and permits only an explicitly vocal melody
stem.  It reads no audio samples or f0 values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml


MIX_PATTERN = re.compile(
    r"^MDB-melody-synth/audio_mix/(?P<track>.+)_MIX_melsynth\.wav$"
)
ANNOTATION_PATTERN = re.compile(
    r"^MDB-melody-synth/annotation_melody/"
    r"(?P<track>.+)_STEM_(?P<stem>\d+)\.RESYN\.csv$"
)


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def is_resource_fork(path: str) -> bool:
    return any(part.startswith("._") for part in path.replace("\\", "/").split("/"))


def archive_index(member_names: Iterable[str]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for raw_name in member_names:
        name = str(raw_name).replace("\\", "/")
        if is_resource_fork(name):
            continue
        mix_match = MIX_PATTERN.fullmatch(name)
        if mix_match:
            indexed.setdefault(mix_match.group("track"), {"annotations": []})[
                "mix"
            ] = name
            continue
        annotation_match = ANNOTATION_PATTERN.fullmatch(name)
        if annotation_match:
            indexed.setdefault(
                annotation_match.group("track"), {"annotations": []}
            )["annotations"].append(
                {
                    "path": name,
                    "stem": int(annotation_match.group("stem")),
                }
            )
    return indexed


def normalized_instruments(value: Any) -> list[str]:
    values = value if isinstance(value, list) else [value]
    return [str(item or "").strip().lower() for item in values if str(item or "").strip()]


def is_vocal_melody(metadata: dict[str, Any], stem_number: int) -> tuple[bool, list[str]]:
    instrumental = str(metadata.get("instrumental", "")).strip().lower()
    if instrumental not in {"no", "false", "0"}:
        return False, []
    stem = (metadata.get("stems") or {}).get(f"S{stem_number:02d}")
    if not isinstance(stem, dict):
        return False, []
    if str(stem.get("component", "")).strip().lower() != "melody":
        return False, []
    instruments = normalized_instruments(stem.get("instrument"))
    vocal = any("singer" in item or "vocal" in item for item in instruments)
    return vocal, instruments


def eligible_vocal_rows(
    indexed: dict[str, dict[str, Any]], metadata_directory: Path
) -> list[dict[str, Any]]:
    eligible: list[dict[str, Any]] = []
    for track, archive_files in indexed.items():
        annotations = archive_files.get("annotations") or []
        if not archive_files.get("mix") or len(annotations) != 1:
            continue
        metadata_path = metadata_directory / f"{track}_METADATA.yaml"
        if not metadata_path.is_file():
            continue
        metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}
        annotation = annotations[0]
        vocal, instruments = is_vocal_melody(metadata, int(annotation["stem"]))
        if not vocal:
            continue
        eligible.append(
            {
                "trackId": track,
                "mixArchivePath": archive_files["mix"],
                "annotationArchivePath": annotation["path"],
                "annotationStemNumber": int(annotation["stem"]),
                "melodyInstruments": instruments,
                "metadataPath": str(metadata_path.resolve()),
            }
        )
    return sorted(eligible, key=lambda row: row["mixArchivePath"])


def resolve_selection(
    archive: Path, metadata_directory: Path, seed: int
) -> dict[str, Any]:
    with tarfile.open(archive, "r:gz") as handle:
        indexed = archive_index(member.name for member in handle)
    eligible = eligible_vocal_rows(indexed, metadata_directory)
    if not eligible:
        raise ValueError("No singer-melody full mixes satisfy the frozen eligibility rule.")
    index = seed % len(eligible)
    return {
        "schema": "polymath-medleydb-vocal-holdout-resolution-v1",
        "resolvedAtUtc": datetime.now(timezone.utc).isoformat(),
        "method": "metadata-only; no audio samples or annotation values read",
        "seedUnsignedInteger": seed,
        "eligibleVocalTrackCount": len(eligible),
        "calculation": f"{seed} modulo {len(eligible)}",
        "zeroBasedIndex": index,
        "eligiblePoolSha256": canonical_hash(eligible),
        "selected": eligible[index],
        "eligiblePool": eligible,
    }


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--seed", required=True, help="Unsigned integer or hexadecimal value")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    seed = int(args.seed, 0)
    if seed < 0 or seed > 0xFFFFFFFF:
        parser.error("--seed must fit in an unsigned 32-bit integer")
    result = resolve_selection(
        args.archive.resolve(), args.metadata_dir.resolve(), seed
    )
    atomic_json(args.output.resolve(), result)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "eligibleVocalTrackCount": result["eligibleVocalTrackCount"],
                "zeroBasedIndex": result["zeroBasedIndex"],
                "selectedTrackId": result["selected"]["trackId"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
