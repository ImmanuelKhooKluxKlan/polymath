"""Decoded-note safety gates for MuScriptor checkpoint selection.

Teacher-forced validation loss measures token prediction with the correct
history supplied to the model.  Production decoding does not have that luxury:
one changed token can alter the rest of an autoregressive event stream.  This
module therefore selects a small, song-balanced validation panel and rejects a
checkpoint when its *decoded notes* regress, even if its loss improved.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable


DEFAULT_GATE_TOLERANCES_MS = (100, 250)


def _evenly_spaced(records: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0 or not records:
        return []
    if len(records) <= limit:
        return list(records)
    if limit == 1:
        return [records[len(records) // 2]]
    indices = {
        round(position * (len(records) - 1) / (limit - 1))
        for position in range(limit)
    }
    # Rounding can theoretically collapse neighbouring indices. Fill any gap
    # deterministically from the start so the requested budget is respected.
    for index in range(len(records)):
        if len(indices) >= limit:
            break
        indices.add(index)
    return [records[index] for index in sorted(indices)[:limit]]


def select_decoded_gate_records(
    records: Iterable[dict[str, Any]],
    maximum_per_song: int = 6,
) -> list[dict[str, Any]]:
    """Select an evenly spread, deterministic panel from every song.

    If a song contains reviewed silence, one negative clip is deliberately
    retained.  Otherwise an evenly spaced sample could miss the only evidence
    that reveals hallucinated notes in quiet sections.
    """

    if maximum_per_song < 2:
        raise ValueError("Decoded validation needs at least two clips per song")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, record in enumerate(records):
        song_id = str(record.get("songId") or "unknown")
        copied = dict(record)
        copied.setdefault("_decodedGateOriginalIndex", index)
        grouped[song_id].append(copied)

    selected: list[dict[str, Any]] = []
    for song_id in sorted(grouped):
        ordered = sorted(
            grouped[song_id],
            key=lambda item: (
                float(item.get("sourceStart") or 0),
                str(item.get("clipId") or ""),
                int(item["_decodedGateOriginalIndex"]),
            ),
        )
        if len(ordered) <= maximum_per_song:
            chosen = ordered
        else:
            negatives = [item for item in ordered if item.get("isNegativeExample")]
            positives = [item for item in ordered if not item.get("isNegativeExample")]
            if negatives and positives:
                chosen = [*_evenly_spaced(positives, maximum_per_song - 1), negatives[len(negatives) // 2]]
                chosen.sort(key=lambda item: (
                    float(item.get("sourceStart") or 0),
                    str(item.get("clipId") or ""),
                ))
            else:
                chosen = _evenly_spaced(ordered, maximum_per_song)
        for item in chosen:
            item.pop("_decodedGateOriginalIndex", None)
        selected.extend(chosen)
    return selected


def _metric(metrics: dict[str, Any], tolerance_ms: int, name: str) -> float:
    try:
        return float(metrics[f"{tolerance_ms}ms"][name])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Decoded metrics are missing {tolerance_ms}ms.{name}"
        ) from exc


def decoded_checkpoint_gate(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    maximum_aggregate_f1_regression: float = 0.001,
    maximum_aggregate_recall_regression: float = 0.002,
    maximum_per_song_f1_regression: float = 0.010,
    tolerances_ms: tuple[int, ...] = DEFAULT_GATE_TOLERANCES_MS,
) -> dict[str, Any]:
    """Return an auditable pass/fail verdict for decoded checkpoint metrics."""

    if min(
        maximum_aggregate_f1_regression,
        maximum_aggregate_recall_regression,
        maximum_per_song_f1_regression,
    ) < 0:
        raise ValueError("Decoded checkpoint regression allowances cannot be negative")

    checks: dict[str, bool] = {}
    aggregate_deltas: dict[str, Any] = {}
    for tolerance_ms in tolerances_ms:
        baseline_f1 = _metric(baseline, tolerance_ms, "microF1")
        candidate_f1 = _metric(candidate, tolerance_ms, "microF1")
        baseline_recall = _metric(baseline, tolerance_ms, "recall")
        candidate_recall = _metric(candidate, tolerance_ms, "recall")
        key = f"{tolerance_ms}ms"
        aggregate_deltas[key] = {
            "microF1": round(candidate_f1 - baseline_f1, 6),
            "recall": round(candidate_recall - baseline_recall, 6),
        }
        checks[f"aggregate_{key}_f1"] = (
            candidate_f1 + maximum_aggregate_f1_regression >= baseline_f1
        )
        checks[f"aggregate_{key}_recall"] = (
            candidate_recall + maximum_aggregate_recall_regression >= baseline_recall
        )

    baseline_songs = baseline.get("perSongClipScores") or {}
    candidate_songs = candidate.get("perSongClipScores") or {}
    if set(baseline_songs) != set(candidate_songs):
        raise ValueError("Baseline and candidate decoded metrics cover different songs")
    per_song_deltas: dict[str, Any] = {}
    worst_song_regression = 0.0
    for song_id in sorted(baseline_songs):
        per_song_deltas[song_id] = {}
        for tolerance_ms in tolerances_ms:
            key = f"{tolerance_ms}ms"
            try:
                baseline_f1 = float(baseline_songs[song_id][key]["microF1"])
                candidate_f1 = float(candidate_songs[song_id][key]["microF1"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Per-song decoded metrics are missing {song_id}.{key}.microF1"
                ) from exc
            delta = candidate_f1 - baseline_f1
            per_song_deltas[song_id][key] = {"microF1": round(delta, 6)}
            worst_song_regression = min(worst_song_regression, delta)
            checks[f"song_{song_id}_{key}_f1"] = (
                candidate_f1 + maximum_per_song_f1_regression >= baseline_f1
            )

    failed = sorted(name for name, passed in checks.items() if not passed)
    return {
        "schema": "polymath-decoded-checkpoint-gate-v1",
        "passed": not failed,
        "policy": {
            "tolerancesMs": list(tolerances_ms),
            "maximumAggregateF1Regression": maximum_aggregate_f1_regression,
            "maximumAggregateRecallRegression": maximum_aggregate_recall_regression,
            "maximumPerSongF1Regression": maximum_per_song_f1_regression,
        },
        "aggregateCandidateMinusBaseline": aggregate_deltas,
        "perSongCandidateMinusBaseline": per_song_deltas,
        "worstPerSongF1Delta": round(worst_song_regression, 6),
        "checks": checks,
        "failedChecks": failed,
    }
