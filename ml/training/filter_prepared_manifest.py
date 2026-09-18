"""Filter a prepared JSONL manifest without changing clip provenance.

This utility is intentionally small: it selects complete songs from an
already-audited prepared manifest while preserving every record verbatim.
It is useful when composing linked RunPod datasets whose audio objects already
exist under immutable volume paths.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), 1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: record is not an object")
        records.append(value)
    return records


def filter_records(
    records: Iterable[dict[str, Any]],
    *,
    include_song_ids: set[str],
    maximum_clips_per_song: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    requested = {value.strip() for value in include_song_ids if value.strip()}
    if not requested:
        raise ValueError("At least one included song ID is required")
    if maximum_clips_per_song is not None and maximum_clips_per_song < 1:
        raise ValueError("maximum_clips_per_song must be at least 1")
    selected: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for record in records:
        song_id = str(record.get("songId") or "").strip()
        if not song_id:
            raise ValueError("Manifest record is missing songId")
        if song_id in requested:
            selected.append(record)
            counts[song_id] += 1
    missing = sorted(requested - set(counts))
    if missing:
        raise ValueError(f"Requested songs are absent: {', '.join(missing)}")
    if not selected:
        raise ValueError("Filtering produced an empty manifest")
    available_counts = Counter(counts)
    if maximum_clips_per_song is not None:
        by_song = {
            song_id: [record for record in selected if record["songId"] == song_id]
            for song_id in requested
        }
        keep_ids: set[str] = set()
        for song_id, song_records in by_song.items():
            count = min(maximum_clips_per_song, len(song_records))
            if count == 1:
                chosen = [song_records[len(song_records) // 2]]
            elif count == len(song_records):
                chosen = song_records
            else:
                chosen = [
                    song_records[round(index * (len(song_records) - 1) / (count - 1))]
                    for index in range(count)
                ]
            for record in chosen:
                clip_id = str(record.get("clipId") or "").strip()
                if not clip_id:
                    raise ValueError("Manifest record is missing clipId")
                keep_ids.add(clip_id)
        selected = [record for record in selected if str(record.get("clipId")) in keep_ids]
        counts = Counter(str(record["songId"]) for record in selected)
    return selected, {
        "schema": "polymath-prepared-manifest-filter-v1",
        "includedSongIds": sorted(requested),
        "records": len(selected),
        "clipsBySong": dict(sorted(counts.items())),
        "availableClipsBySong": dict(sorted(available_counts.items())),
        "maximumClipsPerSong": maximum_clips_per_song,
        "provenancePolicy": "Records and immutable audioClip paths are preserved verbatim.",
    }


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            for record in records
        ),
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--include-song", action="append", required=True)
    parser.add_argument("--maximum-clips-per-song", type=int)
    args = parser.parse_args()

    records, report = filter_records(
        read_jsonl(args.input.resolve()),
        include_song_ids=set(args.include_song),
        maximum_clips_per_song=args.maximum_clips_per_song,
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output, records)
    report_path = args.report.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "report": str(report_path), **report}, indent=2))


if __name__ == "__main__":
    main()
