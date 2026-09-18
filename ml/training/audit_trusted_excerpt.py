"""Fail closed when a rejected tail leaks into pianist-style training data.

This utility is intentionally generic: it checks a trimmed reference target,
its source-timeline labels, the alignment boundary, chord prototypes, and any
additional JSON artifacts for known superseded path fragments.  A report is
written even when a check fails, then the command exits non-zero.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


EPSILON = 1e-6


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value: Any, fallback: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return fallback
    return result if math.isfinite(result) else fallback


def note_time(note: dict[str, Any]) -> float:
    return finite(note.get("time", note.get("startTime", note.get("start"))))


def note_end(note: dict[str, Any]) -> float:
    return note_time(note) + max(0.0, finite(note.get("duration"), 0.0))


def nested_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from nested_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from nested_strings(item)


def normalized_path_text(value: str) -> str:
    return value.replace("\\", "/").lower()


def boundary_from_alignment(alignment: dict[str, Any], cutoff: float) -> float:
    anchors = [item for item in alignment.get("anchors") or [] if isinstance(item, dict)]
    exact = [
        item
        for item in anchors
        if abs(finite(item.get("referenceTime", item.get("targetTime"))) - cutoff)
        <= EPSILON
    ]
    if not exact:
        raise ValueError(f"Alignment has no explicit boundary anchor at {cutoff}")
    return finite(exact[-1].get("observedTime", exact[-1].get("sourceTime")))


def make_check(name: str, passed: bool, **evidence: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "evidence": evidence}


def audit(
    *,
    trimmed_target: dict[str, Any],
    aligned_target: dict[str, Any],
    alignment: dict[str, Any],
    cutoff: float,
    song_id: str,
    chord_libraries: list[dict[str, Any]],
    scanned_artifacts: list[tuple[str, dict[str, Any]]],
    forbidden_fragments: list[str],
) -> dict[str, Any]:
    target_notes = [item for item in trimmed_target.get("notes") or [] if isinstance(item, dict)]
    aligned_notes = [item for item in aligned_target.get("notes") or [] if isinstance(item, dict)]
    source_boundary = boundary_from_alignment(alignment, cutoff)

    target_onsets = [note_time(note) for note in target_notes]
    target_ends = [note_end(note) for note in target_notes]
    original_onsets = [finite(note.get("originalTime")) for note in aligned_notes]
    aligned_ends = [note_end(note) for note in aligned_notes]

    excerpt = trimmed_target.get("trainingExcerpt") or {}
    anchors = [item for item in alignment.get("anchors") or [] if isinstance(item, dict)]
    reference_anchor_times = [
        finite(item.get("referenceTime", item.get("targetTime"))) for item in anchors
    ]
    windows = [
        item for item in alignment.get("qualityWindows") or [] if isinstance(item, dict)
    ]
    reference_window_ends = [
        finite(item.get("referenceEnd", item.get("targetEndSeconds"))) for item in windows
    ]

    checks = [
        make_check(
            "trimmed-target-onsets-before-exclusive-cutoff",
            bool(target_onsets) and max(target_onsets) < cutoff,
            notes=len(target_notes),
            maximumOnset=max(target_onsets, default=None),
            cutoffExclusive=cutoff,
        ),
        make_check(
            "trimmed-target-durations-clipped-at-cutoff",
            bool(target_ends) and max(target_ends) <= cutoff + EPSILON,
            maximumEnd=max(target_ends, default=None),
            cutoff=cutoff,
        ),
        make_check(
            "trim-provenance-records-exact-boundary",
            abs(finite(excerpt.get("endSeconds")) - cutoff) <= EPSILON
            and excerpt.get("rule") is not None,
            excerptEnd=excerpt.get("endSeconds"),
            rule=excerpt.get("rule"),
        ),
        make_check(
            "alignment-has-no-post-cutoff-anchors",
            bool(reference_anchor_times)
            and max(reference_anchor_times) <= cutoff + EPSILON,
            anchors=len(reference_anchor_times),
            maximumReferenceAnchor=max(reference_anchor_times, default=None),
        ),
        make_check(
            "alignment-windows-stop-at-cutoff",
            bool(reference_window_ends)
            and max(reference_window_ends) <= cutoff + EPSILON,
            windows=len(reference_window_ends),
            maximumReferenceWindowEnd=max(reference_window_ends, default=None),
        ),
        make_check(
            "aligned-labels-retain-only-pre-cutoff-original-onsets",
            bool(original_onsets)
            and all(math.isfinite(value) for value in original_onsets)
            and max(original_onsets) < cutoff,
            notes=len(aligned_notes),
            maximumOriginalOnset=max(original_onsets, default=None),
        ),
        make_check(
            "aligned-label-durations-stop-at-source-boundary",
            bool(aligned_ends) and max(aligned_ends) <= source_boundary + EPSILON,
            maximumMappedEnd=max(aligned_ends, default=None),
            sourceBoundary=source_boundary,
        ),
    ]

    chord_rows: list[dict[str, Any]] = []
    for index, library in enumerate(chord_libraries):
        prototypes = [
            item
            for item in library.get("prototypes") or []
            if isinstance(item, dict) and str(item.get("songId")) == song_id
        ]
        times = [finite(item.get("time")) for item in prototypes]
        row = {
            "libraryIndex": index,
            "songPrototypes": len(times),
            "maximumSourceTime": max(times, default=None),
        }
        chord_rows.append(row)
        checks.append(
            make_check(
                f"chord-library-{index}-has-no-post-boundary-{song_id}-prototype",
                bool(times) and max(times) <= source_boundary + EPSILON,
                **row,
                sourceBoundary=source_boundary,
            )
        )

    normalized_forbidden = [normalized_path_text(item) for item in forbidden_fragments]
    scans: list[dict[str, Any]] = []
    for name, payload in scanned_artifacts:
        matches = sorted(
            {
                value
                for value in nested_strings(payload)
                if any(fragment in normalized_path_text(value) for fragment in normalized_forbidden)
            }
        )
        scans.append({"artifact": name, "forbiddenMatches": matches})
        checks.append(
            make_check(
                f"artifact-{name}-does-not-reference-superseded-target",
                not matches,
                forbiddenMatches=matches,
            )
        )

    return {
        "schema": "polymath-trusted-excerpt-audit-v1",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "songId": song_id,
        "referenceBoundary": {
            "startSecondsInclusive": 0.0,
            "endSecondsExclusive": cutoff,
            "mappedSourceEndSeconds": source_boundary,
        },
        "counts": {
            "trimmedTargetNotes": len(target_notes),
            "alignedTargetNotes": len(aligned_notes),
            "eligibleAlignedTargetNotes": sum(
                item.get("trainingEligible") is not False for item in aligned_notes
            ),
            "chordLibraries": chord_rows,
        },
        "artifactScans": scans,
        "checks": checks,
        "decision": "PASS" if all(item["passed"] for item in checks) else "FAIL",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trimmed-target", required=True)
    parser.add_argument("--aligned-target", required=True)
    parser.add_argument("--alignment", required=True)
    parser.add_argument("--reference-end-exclusive", required=True, type=float)
    parser.add_argument("--song-id", required=True)
    parser.add_argument("--chord-library", action="append", default=[])
    parser.add_argument("--artifact", action="append", default=[])
    parser.add_argument("--forbidden-path-fragment", action="append", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    paths = {
        "trimmedTarget": Path(args.trimmed_target).resolve(),
        "alignedTarget": Path(args.aligned_target).resolve(),
        "alignment": Path(args.alignment).resolve(),
    }
    chord_paths = [Path(value).resolve() for value in args.chord_library]
    artifact_paths = [Path(value).resolve() for value in args.artifact]
    for path in [*paths.values(), *chord_paths, *artifact_paths]:
        if not path.is_file():
            raise FileNotFoundError(path)

    report = audit(
        trimmed_target=load_json(paths["trimmedTarget"]),
        aligned_target=load_json(paths["alignedTarget"]),
        alignment=load_json(paths["alignment"]),
        cutoff=args.reference_end_exclusive,
        song_id=args.song_id,
        chord_libraries=[load_json(path) for path in chord_paths],
        scanned_artifacts=[(path.name, load_json(path)) for path in artifact_paths],
        forbidden_fragments=args.forbidden_path_fragment,
    )
    report["inputs"] = {
        name: {"path": str(path), "sha256": sha256_file(path)}
        for name, path in paths.items()
    }
    report["inputs"]["chordLibraries"] = [
        {"path": str(path), "sha256": sha256_file(path)} for path in chord_paths
    ]
    report["inputs"]["artifacts"] = [
        {"path": str(path), "sha256": sha256_file(path)} for path in artifact_paths
    ]

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "decision": report["decision"]}, indent=2))
    if report["decision"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
