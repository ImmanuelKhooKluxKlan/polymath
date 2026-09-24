"""Select, download, and import a leakage-safe MAESTRO piano curriculum.

The official MAESTRO split is preserved.  Test rows are reserved in the
selection manifest but are not downloaded or parsed unless explicitly asked,
which keeps them sealed while train/validation data are prepared.  MAESTRO is
CC BY-NC-SA 4.0, so every generated artifact is marked non-commercial research
only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import unicodedata
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "polymath-maestro-curriculum-v2"
SUPERVISION_SCHEMA = "polymath-supervision-package-v1"
LICENSE_ID = "CC-BY-NC-SA-4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by-nc-sa/4.0/"
DATASET_URL = "https://magenta.tensorflow.org/datasets/maestro"
DEFAULT_MIRROR = "ddPn08/maestro-v3.0.0"


class MaestroCurriculumError(ValueError):
    pass


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path, block_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(seed: str, *parts: str) -> str:
    return hashlib.sha256("://:".join((seed, *parts)).encode("utf-8")).hexdigest()


def composer_components(value: str) -> frozenset[str]:
    """Return the individual people represented by one MAESTRO credit.

    MAESTRO uses a slash for composer/arranger credits such as
    ``Robert Schumann / Franz Liszt``.  Treating that complete display string
    as one composer leaks Liszt into a split that also contains the plain
    ``Franz Liszt`` credit.  Comparison keys are Unicode-normalized and
    case-folded while the original display credit remains untouched.
    """

    components = {
        unicodedata.normalize("NFKC", part).strip().casefold()
        for part in str(value or "").split("/")
        if part.strip()
    }
    if not components:
        raise MaestroCurriculumError("Composer credit must name at least one person")
    return frozenset(components)


def read_metadata(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "canonical_composer",
        "canonical_title",
        "split",
        "year",
        "midi_filename",
        "audio_filename",
        "duration",
    }
    if not rows or not required.issubset(rows[0]):
        raise MaestroCurriculumError(f"Metadata is missing required columns: {path}")
    normalized: list[dict[str, Any]] = []
    for row in rows:
        try:
            duration = float(row["duration"])
        except (TypeError, ValueError):
            continue
        if not math.isfinite(duration) or duration <= 0:
            continue
        normalized.append({**row, "duration": duration})
    return normalized


def balanced_selection(
    rows: Iterable[dict[str, Any]],
    *,
    split: str,
    count: int,
    seed: str,
    minimum_duration_seconds: float,
    maximum_duration_seconds: float,
    excluded_composers: set[str] | None = None,
    allowed_composers: set[str] | None = None,
) -> list[dict[str, Any]]:
    composer_exclusions = set(excluded_composers or set())
    composer_allowlist = set(allowed_composers) if allowed_composers is not None else None
    eligible = [
        row
        for row in rows
        if str(row["split"]).strip().lower() == split
        and str(row["canonical_composer"]) not in composer_exclusions
        and (
            composer_allowlist is None
            or str(row["canonical_composer"]) in composer_allowlist
        )
        and minimum_duration_seconds <= float(row["duration"]) <= maximum_duration_seconds
    ]
    by_composer: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        by_composer[str(row["canonical_composer"])].append(row)
    for composer, composer_rows in by_composer.items():
        composer_rows.sort(
            key=lambda row: stable_hash(seed, split, composer, str(row["audio_filename"]))
        )
    composers = sorted(by_composer, key=lambda name: stable_hash(seed, split, name))
    selected: list[dict[str, Any]] = []
    round_index = 0
    while len(selected) < count:
        added = 0
        for composer in composers:
            options = by_composer[composer]
            if round_index >= len(options):
                continue
            selected.append(options[round_index])
            added += 1
            if len(selected) == count:
                break
        if added == 0:
            break
        round_index += 1
    if len(selected) != count:
        raise MaestroCurriculumError(
            f"Requested {count} {split} rows but only {len(selected)} satisfy the duration bounds "
            f"after excluding {len(composer_exclusions)} reserved composers"
        )
    return selected


def balanced_pool_selection(
    rows: Iterable[dict[str, Any]],
    *,
    count: int,
    seed: str,
    evidence_split: str,
) -> list[dict[str, Any]]:
    """Choose exact rows from an already isolated composer-component pool."""

    by_credit: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_credit[str(row["canonical_composer"])].append(row)
    for credit, credit_rows in by_credit.items():
        credit_rows.sort(
            key=lambda row: stable_hash(
                seed, evidence_split, credit, str(row["audio_filename"])
            )
        )
    credits = sorted(
        by_credit,
        key=lambda credit: stable_hash(seed, evidence_split, credit),
    )
    selected: list[dict[str, Any]] = []
    round_index = 0
    while len(selected) < count:
        added = 0
        for credit in credits:
            options = by_credit[credit]
            if round_index >= len(options):
                continue
            selected.append(options[round_index])
            added += 1
            if len(selected) == count:
                break
        if added == 0:
            break
        round_index += 1
    if len(selected) != count:
        raise MaestroCurriculumError(
            f"Requested {count} {evidence_split} rows but isolated pool has only "
            f"{len(selected)} eligible recordings"
        )
    return selected


def _composer_component_groups(
    rows: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build connected people groups so indirect composite credits cannot leak."""

    source_rows = list(rows)
    parent: dict[str, str] = {}

    def find(person: str) -> str:
        parent.setdefault(person, person)
        if parent[person] != person:
            parent[person] = find(parent[person])
        return parent[person]

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        # Root choice is lexical so group identities do not depend on CSV order.
        low, high = sorted((left_root, right_root))
        parent[high] = low

    for row in source_rows:
        people = sorted(composer_components(row["canonical_composer"]))
        for person in people:
            find(person)
        for person in people[1:]:
            union(people[0], person)

    people_by_root: dict[str, set[str]] = defaultdict(set)
    for person in sorted(parent):
        people_by_root[find(person)].add(person)
    rows_by_root: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        person = min(composer_components(row["canonical_composer"]))
        rows_by_root[find(person)].append(row)
    return [
        {
            "key": " | ".join(sorted(people_by_root[root])),
            "people": frozenset(people_by_root[root]),
            "rows": rows_by_root[root],
        }
        for root in sorted(rows_by_root)
    ]


def component_disjoint_selection(
    rows: Iterable[dict[str, Any]],
    *,
    counts: dict[str, int],
    seed: str,
    minimum_duration_seconds: float,
    maximum_duration_seconds: float,
) -> dict[str, list[dict[str, Any]]]:
    """Create new evidence splits from connected individual-person groups.

    MAESTRO's official train/validation/test partitions are recording-disjoint,
    but not individual-composer-disjoint.  For a generalization benchmark, a
    connected component of composer/arranger credits is therefore atomic: the
    whole group may feed exactly one evidence split.
    """

    split_order = ("train", "validation", "test")
    if set(counts) != set(split_order) or any(counts[split] < 1 for split in split_order):
        raise MaestroCurriculumError("Component-disjoint selection needs positive train/validation/test counts")
    eligible = [
        row
        for row in rows
        if minimum_duration_seconds <= float(row["duration"]) <= maximum_duration_seconds
    ]
    groups = _composer_component_groups(eligible)
    groups.sort(key=lambda group: stable_hash(seed, "component-group", group["key"]))
    if len(groups) < len(split_order):
        raise MaestroCurriculumError("Not enough disconnected composer groups for three evidence splits")

    base_group_targets = {
        "train": min(counts["train"], 5),
        "validation": min(counts["validation"], 4),
        "test": min(counts["test"], 4),
    }
    usable_group_count = min(len(groups), sum(counts.values()))
    minimum_groups = {split: 1 for split in split_order}
    remaining = usable_group_count - len(split_order)
    while remaining > 0 and minimum_groups != base_group_targets:
        for split in ("train", "validation", "test"):
            if remaining <= 0:
                break
            if minimum_groups[split] < base_group_targets[split]:
                minimum_groups[split] += 1
                remaining -= 1
    # When more independent groups exist, spend them on test breadth first,
    # then validation breadth, then training breadth. Every assigned group is
    # represented by at least one selected recording below.
    while remaining > 0:
        added = False
        for split in ("test", "validation", "train"):
            if remaining <= 0:
                break
            if minimum_groups[split] < counts[split]:
                minimum_groups[split] += 1
                remaining -= 1
                added = True
        if not added:
            break

    @lru_cache(maxsize=None)
    def allocate(
        index: int,
        train_needed: int,
        validation_needed: int,
        test_needed: int,
        train_groups_needed: int,
        validation_groups_needed: int,
        test_groups_needed: int,
    ) -> tuple[tuple[int, str], ...] | None:
        row_needs = (train_needed, validation_needed, test_needed)
        group_needs = (
            train_groups_needed,
            validation_groups_needed,
            test_groups_needed,
        )
        if not any(row_needs) and not any(group_needs):
            return ()
        if index >= len(groups):
            return None
        groups_left = len(groups) - index
        if groups_left < sum(group_needs):
            return None
        if sum(len(group["rows"]) for group in groups[index:]) < sum(row_needs):
            return None

        group = groups[index]
        ranked_splits = sorted(
            range(len(split_order)),
            key=lambda split_index: (
                -(group_needs[split_index] > 0),
                -(row_needs[split_index] / max(1, counts[split_order[split_index]])),
                stable_hash(seed, "component-owner", group["key"], split_order[split_index]),
            ),
        )
        for split_index in ranked_splits:
            if row_needs[split_index] <= 0 and group_needs[split_index] <= 0:
                continue
            next_rows = list(row_needs)
            next_groups = list(group_needs)
            next_rows[split_index] = max(0, next_rows[split_index] - len(group["rows"]))
            next_groups[split_index] = max(0, next_groups[split_index] - 1)
            tail = allocate(index + 1, *next_rows, *next_groups)
            if tail is not None:
                return ((index, split_order[split_index]),) + tail
        return allocate(index + 1, *row_needs, *group_needs)

    assignment = allocate(
        0,
        counts["train"],
        counts["validation"],
        counts["test"],
        minimum_groups["train"],
        minimum_groups["validation"],
        minimum_groups["test"],
    )
    if assignment is None:
        raise MaestroCurriculumError(
            "Could not allocate individual-composer components to all evidence splits"
        )

    owned_groups: dict[str, list[dict[str, Any]]] = {
        split: [] for split in split_order
    }
    for group_index, split in assignment:
        owned_groups[split].append(groups[group_index])

    selected: dict[str, list[dict[str, Any]]] = {}
    for split in split_order:
        # Guarantee that every promised component contributes at least one row.
        primary_rows = [
            min(
                group["rows"],
                key=lambda row: stable_hash(
                    seed, split, group["key"], str(row["audio_filename"])
                ),
            )
            for group in owned_groups[split]
        ]
        primary_audio = {str(row["audio_filename"]) for row in primary_rows}
        remaining_pool = [
            row
            for group in owned_groups[split]
            for row in group["rows"]
            if str(row["audio_filename"]) not in primary_audio
        ]
        selected[split] = primary_rows + balanced_pool_selection(
            remaining_pool,
            count=counts[split] - len(primary_rows),
            seed=seed,
            evidence_split=split,
        ) if counts[split] > len(primary_rows) else primary_rows
        selected[split].sort(
            key=lambda row: stable_hash(seed, split, str(row["audio_filename"]))
        )
    return selected


def song_id(row: dict[str, Any], evidence_split: str | None = None) -> str:
    composer = re.sub(r"[^a-z0-9]+", "-", str(row["canonical_composer"]).lower()).strip("-")
    digest = hashlib.sha256(str(row["audio_filename"]).encode("utf-8")).hexdigest()[:10]
    split = evidence_split or str(row["split"])
    return f"maestro-{split}-{composer[:28]}-{row['year']}-{digest}"


def disjoint_composer_owners(
    rows: Iterable[dict[str, Any]],
    *,
    counts: dict[str, int],
    seed: str,
    minimum_duration_seconds: float,
    maximum_duration_seconds: float,
) -> dict[str, set[str]]:
    """Assign each selected composer to exactly one evidence split.

    Validation receives first choice because its decoded metrics choose epochs.
    Test receives the remaining eligible composers, and training uses the
    remainder. Within each split, scarce composers are preferred before common
    composers so the other splits retain as much diversity as possible.
    """

    eligible = [
        row
        for row in rows
        if minimum_duration_seconds <= float(row["duration"]) <= maximum_duration_seconds
        and str(row["split"]).strip().lower() in counts
    ]
    memberships: dict[str, set[str]] = defaultdict(set)
    for row in eligible:
        memberships[str(row["canonical_composer"])].add(
            str(row["split"]).strip().lower()
        )

    components_by_credit = {
        credit: composer_components(credit) for credit in memberships
    }
    conflict_degree = {
        credit: sum(
            bool(components_by_credit[credit] & other_components)
            for other_credit, other_components in components_by_credit.items()
            if other_credit != credit
        )
        for credit in memberships
    }

    owners: dict[str, set[str]] = {split: set() for split in counts}
    claimed_components: set[str] = set()
    for split in ("validation", "test"):
        candidates = [
            credit
            for credit, splits in memberships.items()
            if split in splits
            and components_by_credit[credit].isdisjoint(claimed_components)
        ]
        candidates.sort(
            key=lambda credit: (
                len(memberships[credit]),
                conflict_degree[credit],
                stable_hash(seed, "composer-owner", split, credit),
            )
        )
        owners[split].update(candidates[: min(counts[split], len(candidates))])
        claimed_components.update(
            component
            for credit in owners[split]
            for component in components_by_credit[credit]
        )

    owners["train"] = {
        credit
        for credit, splits in memberships.items()
        if "train" in splits
        and components_by_credit[credit].isdisjoint(claimed_components)
    }

    split_components = {
        split: {
            component
            for credit in credits
            for component in components_by_credit[credit]
        }
        for split, credits in owners.items()
    }
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = split_components[left] & split_components[right]
        if overlap:
            raise MaestroCurriculumError(
                f"Individual composer leakage between {left} and {right}: "
                + ", ".join(sorted(overlap))
            )
    return owners


def selection_manifest(
    metadata_path: Path,
    *,
    train_count: int,
    validation_count: int,
    reserve_test_count: int,
    seed: str,
    minimum_duration_seconds: float,
    maximum_duration_seconds: float,
    mirror_repo: str,
    excluded_audio_filenames: set[str] | None = None,
    split_policy: str = "official",
) -> dict[str, Any]:
    rows = read_metadata(metadata_path)
    exclusions = set(excluded_audio_filenames or set())
    rows = [row for row in rows if str(row["audio_filename"]) not in exclusions]
    counts = {
        "train": train_count,
        "validation": validation_count,
        "test": reserve_test_count,
    }
    if split_policy == "individual-composer-components":
        selected_rows = component_disjoint_selection(
            rows,
            counts=counts,
            seed=seed,
            minimum_duration_seconds=minimum_duration_seconds,
            maximum_duration_seconds=maximum_duration_seconds,
        )
    elif split_policy == "official":
        # Preserve MAESTRO's official split while preventing one individual in
        # a composite composer/arranger credit from crossing evidence splits.
        composer_owners = disjoint_composer_owners(
            rows,
            counts=counts,
            seed=seed,
            minimum_duration_seconds=minimum_duration_seconds,
            maximum_duration_seconds=maximum_duration_seconds,
        )
        selected_rows = {}
        for split in ("train", "validation", "test"):
            selected_rows[split] = balanced_selection(
                rows,
                split=split,
                count=counts[split],
                seed=seed,
                minimum_duration_seconds=minimum_duration_seconds,
                maximum_duration_seconds=maximum_duration_seconds,
                allowed_composers=composer_owners[split],
            )
    else:
        raise MaestroCurriculumError(f"Unknown split policy: {split_policy}")

    selected: list[dict[str, Any]] = []
    for split in ("train", "validation", "test"):
        for row in selected_rows[split]:
            selected.append(
                {
                    "songId": song_id(row, split),
                    "split": split,
                    "sourceSplit": str(row["split"]).strip().lower(),
                    "composer": row["canonical_composer"],
                    "composerComponents": sorted(composer_components(row["canonical_composer"])),
                    "title": row["canonical_title"],
                    "year": int(row["year"]),
                    "durationSeconds": round(float(row["duration"]), 9),
                    "audioFilename": row["audio_filename"],
                    "midiFilename": row["midi_filename"],
                    "sealed": split == "test",
                }
            )
    return {
        "schema": SCHEMA,
        "seed": seed,
        "sourceMetadata": str(metadata_path.resolve()),
        "sourceMetadataSha256": sha256_file(metadata_path),
        "mirror": {"provider": "Hugging Face dataset mirror", "repoId": mirror_repo},
        "dataset": {"name": "MAESTRO v3.0.0", "url": DATASET_URL},
        "license": {
            "id": LICENSE_ID,
            "url": LICENSE_URL,
            "commercialUseAllowed": False,
            "restriction": "Non-commercial research only; share-alike applies to adaptations.",
        },
        "selection": {
            "minimumDurationSeconds": minimum_duration_seconds,
            "maximumDurationSeconds": maximum_duration_seconds,
            "composerBalanced": True,
            "composerDisjointAcrossSelectedSplits": True,
            "composerDisjointUnit": "individual-person-within-composite-credit",
            "selectedComposerCounts": {
                split: len({str(row["canonical_composer"]) for row in selected_rows[split]})
                for split in ("train", "validation", "test")
            },
            "selectedComposerComponentCounts": {
                split: len({
                    component
                    for row in selected_rows[split]
                    for component in composer_components(row["canonical_composer"])
                })
                for split in ("train", "validation", "test")
            },
            "splitPolicy": split_policy,
            "officialSplitsPreserved": split_policy == "official",
            "excludedPreviouslyOpenedRecordings": len(exclusions),
            "counts": counts,
        },
        "songs": selected,
        "sealedTestPolicy": (
            "Test filenames are frozen now. Test audio/MIDI must not be downloaded or parsed until "
            "the candidate checkpoint and acceptance thresholds are frozen."
        ),
    }


def download_materialized_splits(
    manifest: dict[str, Any],
    dataset_root: Path,
    splits: set[str],
) -> dict[str, Any]:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - environment dependency
        raise MaestroCurriculumError("huggingface_hub is required for --download") from exc
    if "test" in splits:
        raise MaestroCurriculumError(
            "Refusing to download the sealed test split during curriculum preparation"
        )
    songs = [song for song in manifest["songs"] if song["split"] in splits]
    patterns = sorted(
        {filename for song in songs for filename in (song["audioFilename"], song["midiFilename"])}
    )
    dataset_root.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=manifest["mirror"]["repoId"],
        repo_type="dataset",
        allow_patterns=patterns,
        local_dir=str(dataset_root),
    )
    missing = [pattern for pattern in patterns if not (dataset_root / pattern).is_file()]
    if missing:
        raise MaestroCurriculumError(f"Download finished with {len(missing)} missing files")
    return {
        "splits": sorted(splits),
        "songs": len(songs),
        "files": len(patterns),
        "bytes": sum((dataset_root / pattern).stat().st_size for pattern in patterns),
    }


def finish_note(
    notes: list[dict[str, Any]],
    active: dict[tuple[int, int], list[tuple[float, int]]],
    key: tuple[int, int],
    end: float,
    maximum_duration: float,
) -> None:
    starts = active.get(key)
    if not starts:
        return
    start, velocity = starts.pop(0)
    if not starts:
        active.pop(key, None)
    if start >= maximum_duration:
        return
    clipped_end = min(maximum_duration, end)
    duration = clipped_end - start
    if duration < 0.01:
        return
    notes.append(
        {
            "midi": key[1],
            "time": round(start, 6),
            "duration": round(duration, 6),
            "velocity": round(max(0.05, min(1.0, velocity / 127.0)), 4),
            "instrument": "acoustic_piano",
        }
    )


def read_midi_notes(path: Path, maximum_duration: float) -> list[dict[str, Any]]:
    try:
        import mido
    except ImportError as exc:  # pragma: no cover - environment dependency
        raise MaestroCurriculumError("mido is required to import MAESTRO") from exc
    midi = mido.MidiFile(path)
    tempo = 500_000
    elapsed = 0.0
    active: dict[tuple[int, int], list[tuple[float, int]]] = defaultdict(list)
    notes: list[dict[str, Any]] = []
    for message in mido.merge_tracks(midi.tracks):
        elapsed += mido.tick2second(message.time, midi.ticks_per_beat, tempo)
        if message.type == "set_tempo":
            tempo = int(message.tempo)
            continue
        if elapsed > maximum_duration + 1.0:
            break
        if message.type == "note_on" and int(message.velocity) > 0:
            key = (int(getattr(message, "channel", 0)), int(message.note))
            active[key].append((max(0.0, elapsed), int(message.velocity)))
        elif message.type in {"note_off", "note_on"}:
            key = (int(getattr(message, "channel", 0)), int(message.note))
            finish_note(notes, active, key, elapsed, maximum_duration)
    for key in list(active):
        while active.get(key):
            finish_note(notes, active, key, maximum_duration, maximum_duration)
    return sorted(notes, key=lambda note: (note["time"], note["midi"]))


def quality_windows(duration: float, window_seconds: float = 5.0) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    index = 0
    start = 0.0
    while start < duration - 1e-9:
        end = min(duration, start + window_seconds)
        windows.append(
            {
                "id": f"exact-{index:05d}",
                "sourceStart": round(start, 6),
                "sourceEnd": round(end, 6),
                "status": "trusted",
                "reason": "Official MAESTRO audio/MIDI pair aligned during capture (~3 ms).",
            }
        )
        index += 1
        start = end
    return windows


def supervision_package(song: dict[str, Any], midi_path: Path) -> dict[str, Any]:
    duration = float(song["durationSeconds"])
    windows = quality_windows(duration)
    notes = read_midi_notes(midi_path, duration)
    if not notes:
        raise MaestroCurriculumError(f"No piano notes parsed from {midi_path}")
    for note in notes:
        window_index = min(len(windows) - 1, int(float(note["time"]) // 5.0))
        note.update(
            {
                "trainingEligible": True,
                "qualityStatus": "trusted",
                "qualityWindowId": windows[window_index]["id"],
            }
        )
    return {
        "schema": SUPERVISION_SCHEMA,
        "song": {
            "id": song["songId"],
            "title": song["title"],
            "composer": song["composer"],
        },
        "timeline": {"sourceDurationSeconds": duration, "clock": "maestro-audio-seconds"},
        "alignment": {
            "method": "official-maestro-capture-alignment",
            "estimatedAlignmentPrecisionSeconds": 0.003,
            "qualityWindows": windows,
        },
        "notes": notes,
        "provenance": {
            "dataset": "MAESTRO v3.0.0",
            "datasetUrl": DATASET_URL,
            "midiFilename": song["midiFilename"],
            "audioFilename": song["audioFilename"],
            "license": LICENSE_ID,
            "licenseUrl": LICENSE_URL,
            "commercialUseAllowed": False,
        },
    }


def import_materialized_splits(
    manifest: dict[str, Any],
    dataset_root: Path,
    output_root: Path,
    splits: set[str],
) -> dict[str, Any]:
    if "test" in splits:
        raise MaestroCurriculumError(
            "Refusing to parse sealed test labels before a candidate is frozen"
        )
    supervision_root = output_root / "supervision"
    supervision_root.mkdir(parents=True, exist_ok=True)
    index_songs: list[dict[str, Any]] = []
    summary_songs: list[dict[str, Any]] = []
    for song in manifest["songs"]:
        if song["split"] not in splits:
            continue
        audio_path = dataset_root / song["audioFilename"]
        midi_path = dataset_root / song["midiFilename"]
        if not audio_path.is_file() or not midi_path.is_file():
            raise MaestroCurriculumError(f"Missing downloaded pair for {song['songId']}")
        package_path = supervision_root / f"{song['songId']}.json"
        package = supervision_package(song, midi_path)
        atomic_json(package_path, package)
        index_songs.append(
            {
                "songId": song["songId"],
                "split": song["split"],
                "sourceMedia": str(audio_path.resolve()),
                "supervisionPackage": str(package_path.resolve()),
                "instrumentFocus": "acoustic_piano",
                "rights": {
                    "allowedForTraining": True,
                    "note": (
                        "MAESTRO v3.0.0, CC BY-NC-SA 4.0. Non-commercial research training only; "
                        "not authorized for commercial checkpoint promotion."
                    ),
                },
            }
        )
        summary_songs.append(
            {
                "songId": song["songId"],
                "split": song["split"],
                "composer": song["composer"],
                "durationSeconds": song["durationSeconds"],
                "notes": len(package["notes"]),
                "audioBytes": audio_path.stat().st_size,
                "audioSha256": sha256_file(audio_path),
                "midiBytes": midi_path.stat().st_size,
                "midiSha256": sha256_file(midi_path),
                "supervisionSha256": sha256_file(package_path),
            }
        )
    index = {
        "schema": "polymath-training-index-v1",
        "purpose": "MAESTRO direct-piano foundation research curriculum",
        "commercialUseAllowed": False,
        "songs": index_songs,
    }
    atomic_json(output_root / "training-index.json", index)
    summary = {
        "schema": "polymath-maestro-import-summary-v1",
        "commercialUseAllowed": False,
        "splits": sorted(splits),
        "songs": summary_songs,
        "counts": {
            split: sum(song["split"] == split for song in summary_songs) for split in sorted(splits)
        },
        "totalAudioSeconds": round(sum(float(song["durationSeconds"]) for song in summary_songs), 3),
        "totalNotes": sum(int(song["notes"]) for song in summary_songs),
    }
    atomic_json(output_root / "import-summary.json", summary)
    return summary


def split_names(value: str) -> set[str]:
    output = {item.strip().lower() for item in value.split(",") if item.strip()}
    if not output or not output.issubset({"train", "validation", "test"}):
        raise MaestroCurriculumError("splits must contain train, validation, and/or test")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--train-count", type=int, default=30)
    parser.add_argument("--validation-count", type=int, default=6)
    parser.add_argument("--reserve-test-count", type=int, default=8)
    parser.add_argument("--minimum-duration-seconds", type=float, default=90.0)
    parser.add_argument("--maximum-duration-seconds", type=float, default=300.0)
    parser.add_argument("--seed", default="polymath-maestro-direct-v001")
    parser.add_argument("--mirror-repo", default=DEFAULT_MIRROR)
    parser.add_argument(
        "--split-policy",
        choices=("official", "individual-composer-components"),
        default="official",
        help=(
            "Use individual-composer-components for a strict unseen-person benchmark; "
            "this intentionally replaces MAESTRO's official split assignment."
        ),
    )
    parser.add_argument(
        "--exclude-registry",
        type=Path,
        help="JSON object with an audioFilenames array of previously opened recordings.",
    )
    parser.add_argument(
        "--exclude-selection",
        type=Path,
        action="append",
        default=[],
        help=(
            "Prior selection manifest whose opened roles must not reappear. "
            "May be supplied more than once."
        ),
    )
    parser.add_argument(
        "--exclude-selection-splits",
        default="train,validation",
        help="Comma-separated roles excluded from every --exclude-selection manifest.",
    )
    parser.add_argument("--materialize-splits", default="train,validation")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--import", dest="run_import", action="store_true")
    args = parser.parse_args()

    selection_path = args.selection or args.output_root / "selection-manifest.json"
    if selection_path.is_file():
        manifest = json.loads(selection_path.read_text(encoding="utf-8-sig"))
        if manifest.get("schema") != SCHEMA:
            raise MaestroCurriculumError(f"Unexpected selection schema: {selection_path}")
    else:
        exclusions: set[str] = set()
        if args.exclude_registry:
            registry = json.loads(args.exclude_registry.read_text(encoding="utf-8-sig"))
            values = registry.get("audioFilenames") if isinstance(registry, dict) else None
            if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
                raise MaestroCurriculumError("exclude registry requires an audioFilenames string array")
            exclusions = set(values)
        excluded_selection_splits = split_names(args.exclude_selection_splits)
        for prior_path in args.exclude_selection:
            prior = json.loads(prior_path.read_text(encoding="utf-8-sig"))
            songs = prior.get("songs") if isinstance(prior, dict) else None
            if not isinstance(songs, list):
                raise MaestroCurriculumError(
                    f"exclude selection requires a songs array: {prior_path}"
                )
            for song in songs:
                if not isinstance(song, dict):
                    raise MaestroCurriculumError(
                        f"exclude selection has an invalid song: {prior_path}"
                    )
                if str(song.get("split") or "").strip().lower() not in excluded_selection_splits:
                    continue
                filename = song.get("audioFilename")
                if not isinstance(filename, str) or not filename:
                    raise MaestroCurriculumError(
                        f"exclude selection song has no audioFilename: {prior_path}"
                    )
                exclusions.add(filename)
        manifest = selection_manifest(
            args.metadata,
            train_count=args.train_count,
            validation_count=args.validation_count,
            reserve_test_count=args.reserve_test_count,
            seed=args.seed,
            minimum_duration_seconds=args.minimum_duration_seconds,
            maximum_duration_seconds=args.maximum_duration_seconds,
            mirror_repo=args.mirror_repo,
            excluded_audio_filenames=exclusions,
            split_policy=args.split_policy,
        )
        atomic_json(selection_path, manifest)
    splits = split_names(args.materialize_splits)
    result: dict[str, Any] = {
        "selection": str(selection_path.resolve()),
        "selectedCounts": manifest["selection"]["counts"],
        "sealedTestSongs": sum(song["split"] == "test" for song in manifest["songs"]),
    }
    if args.download:
        result["download"] = download_materialized_splits(manifest, args.dataset_root, splits)
    if args.run_import:
        result["import"] = import_materialized_splits(
            manifest, args.dataset_root, args.output_root, splits
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
