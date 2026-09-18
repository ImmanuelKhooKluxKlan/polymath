"""Search whole-song holdout policies for conservative raw-support swaps.

Every evaluated song uses a selector profile trained without that complete
song.  The operator freezes onset, gesture size, duration, and velocity and
changes only a bounded number of left-hand pitch classes.  Trials are kept in
memory; only the leaderboard and winning candidates are written to disk.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from .apply_raw_support_gesture_swap import apply_swaps
from .apply_raw_support_recovery import load_json
from .evaluate_pianist_candidate_pair import evaluate_pair


def numeric_grid(value: str) -> tuple[float, ...]:
    values = tuple(
        dict.fromkeys(float(item.strip()) for item in value.split(",") if item.strip())
    )
    if not values or any(not math.isfinite(item) for item in values):
        raise argparse.ArgumentTypeError("grid requires finite numeric values")
    return values


def integer_grid(value: str) -> tuple[int, ...]:
    values = tuple(
        dict.fromkeys(int(item.strip()) for item in value.split(",") if item.strip())
    )
    if not values:
        raise argparse.ArgumentTypeError("grid requires integer values")
    return values


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def weighted_delta(rows: list[dict[str, Any]], key: str) -> float:
    denominator = sum(int(row["evaluation"]["referenceNotes"]) for row in rows)
    numerator = sum(
        float(row["evaluation"]["deltas"][key])
        * int(row["evaluation"]["referenceNotes"])
        for row in rows
    )
    return round(numerator / max(1, denominator), 6)


def policy_key(row: dict[str, Any]) -> tuple[float, ...]:
    weighted = row["weightedDeltas"]
    worst = row["worstSongDeltas"]
    return (
        float(row["passesStructuralGate"]),
        float(weighted["quality"] > 0),
        float(weighted["exactF1_100ms"]),
        float(weighted["exactF1_250ms"]),
        float(weighted["pitchClassF1_250ms"]),
        float(worst["quality"]),
        -float(row["replacements"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--profiles-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--thresholds", type=numeric_grid, default=(0.65, 0.75, 0.85, 0.93)
    )
    parser.add_argument(
        "--minimum-gains", type=numeric_grid, default=(0.10, 0.20, 0.30)
    )
    parser.add_argument(
        "--source-radii", type=numeric_grid, default=(0.08, 0.12, 0.18)
    )
    parser.add_argument(
        "--maximum-replacements", type=integer_grid, default=(1,)
    )
    parser.add_argument("--register-shifts", type=integer_grid, default=(0, 12))
    parser.add_argument("--source-field", default="source")
    parser.add_argument("--source-families", default="")
    parser.add_argument("--replace-hands", default="left")
    parser.add_argument("--replace-roles", default="bass,harmony")
    parser.add_argument("--minimum-source-midi", type=int, default=0)
    parser.add_argument("--maximum-source-midi", type=int, default=59)
    parser.add_argument("--minimum-output-midi", type=int, default=33)
    parser.add_argument("--maximum-output-midi", type=int, default=71)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    profiles_root = args.profiles_root.resolve()
    output_root = args.output_root.resolve()
    manifest = load_json(manifest_path)
    source_families = {
        value.strip().lower()
        for value in args.source_families.split(",")
        if value.strip()
    }
    replace_hands = {
        value.strip().lower()
        for value in args.replace_hands.split(",")
        if value.strip()
    }
    replace_roles = {
        value.strip().lower()
        for value in args.replace_roles.split(",")
        if value.strip()
    }
    if not replace_hands or not replace_roles:
        raise ValueError("replace hands and roles cannot be empty")
    prepared: list[dict[str, Any]] = []
    for row in manifest.get("songs") or []:
        song_id = str(row["id"])
        paths = {
            "reference": Path(str(row["reference"])).resolve(),
            "alignment": Path(str(row["alignment"])).resolve(),
            "source": Path(str(row[args.source_field])).resolve(),
            "baseline": Path(str(row["candidate"])).resolve(),
            "profile": (profiles_root / song_id / "profile.json").resolve(),
        }
        for label, path in paths.items():
            if not path.is_file():
                raise FileNotFoundError(f"{song_id} {label} is missing: {path}")
        prepared.append(
            {
                "id": song_id,
                "row": row,
                "paths": {key: str(path) for key, path in paths.items()},
                "reference": load_json(paths["reference"]),
                "alignment": load_json(paths["alignment"]),
                "source": load_json(paths["source"]),
                "baseline": load_json(paths["baseline"]),
                "profile": load_json(paths["profile"]),
            }
        )

    metric_keys = (
        "quality",
        "exactF1_100ms",
        "exactF1_250ms",
        "pitchClassF1_250ms",
        "pitchClassRecall_250ms",
        "velocityMeanAbsoluteError",
        "chordSizeDistance",
        "handOccupancyDistance",
    )
    leaderboard: list[dict[str, Any]] = []
    artifacts: dict[int, dict[str, dict[str, Any]]] = {}
    for threshold in args.thresholds:
        for minimum_gain in args.minimum_gains:
            for source_radius in args.source_radii:
                for maximum_replacements in args.maximum_replacements:
                    for register_shift in args.register_shifts:
                        songs: list[dict[str, Any]] = []
                        candidates: dict[str, dict[str, Any]] = {}
                        for item in prepared:
                            row = item["row"]
                            candidate, diagnostics = apply_swaps(
                                item["baseline"],
                                item["source"],
                                item["profile"],
                                threshold=max(0.0, min(1.0, threshold)),
                                minimum_gain=max(0.0, minimum_gain),
                                source_radius=max(0.01, source_radius),
                                maximum_replacements_per_gesture=max(
                                    1, maximum_replacements
                                ),
                                maximum_source_midi=int(args.maximum_source_midi),
                                register_shift=register_shift,
                                minimum_output_midi=int(args.minimum_output_midi),
                                maximum_output_midi=int(args.maximum_output_midi),
                                minimum_source_midi=int(args.minimum_source_midi),
                                source_families=source_families or None,
                                replace_hands=replace_hands,
                                replace_roles=replace_roles,
                            )
                            evaluation = evaluate_pair(
                                item["reference"],
                                item["alignment"],
                                item["baseline"],
                                candidate,
                                reference_already_aligned=bool(
                                    row.get("referenceAlreadyAligned")
                                ),
                                reference_transpose_semitones=int(
                                    row.get("referenceTransposeSemitones") or 0
                                ),
                                reference_end_seconds=(
                                    float(row["referenceEndSeconds"])
                                    if row.get("referenceEndSeconds") is not None
                                    else None
                                ),
                                candidate_end_seconds=(
                                    float(row["candidateEndSeconds"])
                                    if row.get("candidateEndSeconds") is not None
                                    else None
                                ),
                            )
                            songs.append(
                                {
                                    "id": item["id"],
                                    "replacements": diagnostics["replacements"],
                                    "evaluation": evaluation,
                                }
                            )
                            candidates[item["id"]] = candidate

                        weighted = {
                            key: weighted_delta(songs, key) for key in metric_keys
                        }
                        worst = {
                            key: round(
                                min(
                                    float(song["evaluation"]["deltas"][key])
                                    for song in songs
                                ),
                                6,
                            )
                            for key in metric_keys
                        }
                        retrigger_increase = sum(
                            max(
                                0,
                                int(song["evaluation"]["candidate"]["rapidRetriggersUnder100ms"])
                                - int(song["evaluation"]["baseline"]["rapidRetriggersUnder100ms"]),
                            )
                            for song in songs
                        )
                        trial = {
                            "policy": {
                                "threshold": threshold,
                                "minimumProbabilityGain": minimum_gain,
                                "sourceRadiusSeconds": source_radius,
                                "maximumReplacementsPerGesture": maximum_replacements,
                                "registerShiftSemitones": register_shift,
                                "sourceField": args.source_field,
                                "sourceFamilies": sorted(source_families),
                                "replaceHands": sorted(replace_hands),
                                "replaceRoles": sorted(replace_roles),
                                "sourceMidiRange": [
                                    int(args.minimum_source_midi),
                                    int(args.maximum_source_midi),
                                ],
                                "outputMidiRange": [
                                    int(args.minimum_output_midi),
                                    int(args.maximum_output_midi),
                                ],
                            },
                            "weightedDeltas": weighted,
                            "worstSongDeltas": worst,
                            "replacements": sum(
                                int(song["replacements"]) for song in songs
                            ),
                            "rapidRetriggerIncrease": retrigger_increase,
                            "songs": [
                                {
                                    "id": song["id"],
                                    "replacements": song["replacements"],
                                    "deltas": song["evaluation"]["deltas"],
                                }
                                for song in songs
                            ],
                        }
                        trial["passesStructuralGate"] = bool(
                            weighted["quality"] > 0
                            and weighted["exactF1_100ms"] > 0
                            and weighted["exactF1_250ms"] >= -0.002
                            and weighted["pitchClassF1_250ms"] > 0
                            and worst["quality"] >= -0.0005
                            and worst["exactF1_100ms"] >= -0.002
                            and retrigger_increase == 0
                        )
                        leaderboard.append(trial)
                        artifacts[id(trial)] = candidates

    leaderboard.sort(key=policy_key, reverse=True)
    top = leaderboard[: max(1, args.top_k)]
    winner = top[0]
    winner_candidates = artifacts[id(winner)]
    for song_id, payload in winner_candidates.items():
        atomic_json(output_root / "best-candidates" / f"{song_id}.json", payload)
    report = {
        "schema": "polymath-raw-support-gesture-swap-search-v1",
        "evidenceBoundary": (
            "Every development song used a selector trained without that complete "
            "song. Policy selection used only songs listed in this manifest; any "
            "transfer song must remain absent from both fitting and policy search."
        ),
        "developmentSongs": [item["id"] for item in prepared],
        "manifest": str(manifest_path),
        "profilesRoot": str(profiles_root),
        "trials": len(leaderboard),
        "best": winner,
        "top": top,
        "decision": (
            "CANDIDATE_FOR_UNSEEN_TRANSFER"
            if winner["passesStructuralGate"]
            else "REJECT_OR_RESEARCH_ONLY"
        ),
    }
    atomic_json(output_root / "report.json", report)
    print(
        json.dumps(
            {
                "report": str((output_root / "report.json").resolve()),
                "trials": len(leaderboard),
                "best": winner,
                "decision": report["decision"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
