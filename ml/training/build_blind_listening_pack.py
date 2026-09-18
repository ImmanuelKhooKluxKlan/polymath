"""Build a sealed A/B listening pack for one real full-mix challenge."""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .evaluate_piano_arranger import (
        load_json,
        map_reference_notes,
        monotonic_anchors,
        normalize_notes,
    )
    from .render_piano_json_audio import RELEASE_SECONDS, render_payload, sha256_file
except ImportError:  # Allow direct execution from the ml/training directory.
    from evaluate_piano_arranger import (
        load_json,
        map_reference_notes,
        monotonic_anchors,
        normalize_notes,
    )
    from render_piano_json_audio import RELEASE_SECONDS, render_payload, sha256_file


def atomic_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def copy_wav_excerpt(source: Path, destination: Path, duration_seconds: float) -> dict[str, Any]:
    """Copy only the requested opening of a PCM WAV without re-encoding it."""

    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    with wave.open(str(source), "rb") as reader:
        parameters = reader.getparams()
        requested_frames = int(round(duration_seconds * parameters.framerate))
        frame_count = min(parameters.nframes, requested_frames)
        frames = reader.readframes(frame_count)
    with wave.open(str(destination), "wb") as writer:
        writer.setparams(parameters)
        writer.setnframes(frame_count)
        writer.writeframes(frames)
    return {
        "file": destination.name,
        "frames": frame_count,
        "sampleRate": parameters.framerate,
        "seconds": round(frame_count / parameters.framerate, 6),
        "sourceSeconds": round(parameters.nframes / parameters.framerate, 6),
        "truncated": frame_count < parameters.nframes,
        "sha256": sha256_file(destination),
    }


def mapped_reference_payload(
    reference_path: Path,
    alignment_path: Path | None,
    transpose: int,
    *,
    reference_already_aligned: bool = False,
) -> dict[str, Any]:
    reference_payload = load_json(reference_path)
    reference = normalize_notes(reference_payload, transpose_semitones=transpose)
    if reference_already_aligned:
        notes = reference
    else:
        if alignment_path is None:
            raise ValueError("An alignment is required for an unaligned reference")
        alignment = load_json(alignment_path)
        notes = map_reference_notes(reference, monotonic_anchors(alignment))
    payload = {
        "schema": "polymath-mapped-listening-reference-v1",
        "title": "Aligned Pianella-style pseudo-reference",
        "notes": notes,
        "sourceReferenceSha256": sha256_file(reference_path),
        "referenceTransposeSemitones": transpose,
        "referenceAlreadyAligned": reference_already_aligned,
    }
    if alignment_path is not None:
        payload["frozenAlignmentSha256"] = sha256_file(alignment_path)
    return payload


def write_html(path: Path, *, has_reference: bool) -> None:
    reference_section = (
        '<section><h2>2. Pianella-style reference</h2><p>This is an aligned '
        'pseudo-reference from a separate piano performance. It guides style, but '
        'it is not perfect ground truth.</p><audio controls preload="metadata" '
        'src="01-REFERENCE.wav"></audio></section>'
        if has_reference
        else '<section><h2>2. Listening target</h2><p>There is no separate piano '
        'reference for this holdout. Judge whether each version captures the sung '
        'melody, harmony, rhythm, and natural piano phrasing of the original full '
        'mix.</p></section>'
    )
    atomic_text(
        path,
        """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Polymath real-video blind listening</title>
<style>body{font-family:system-ui,sans-serif;max-width:760px;margin:32px auto;padding:0 18px;line-height:1.5;background:#f5f3ff;color:#211a3a}section{background:white;border:1px solid #dcd5ff;border-radius:16px;padding:18px;margin:14px 0}audio{width:100%}h1,h2{margin:.2em 0 .55em}.note{color:#5b5374}.choice{display:grid;grid-template-columns:1fr 1fr;gap:14px}@media(max-width:600px){.choice{grid-template-columns:1fr}}</style></head>
<body><h1>Real music-video A/B test</h1><p class="note">Do not open <code>SEALED-MAPPING.json</code> until you record your choice.</p>
<section><h2>1. Original full mix</h2><p>Listen for the vocal melody, beat, bass movement, and chord rhythm.</p><audio controls preload="metadata" src="00-ORIGINAL-FULL-MIX.wav"></audio></section>
__REFERENCE_SECTION__
<div class="choice"><section><h2>3A. Version A</h2><audio controls preload="metadata" src="02-A.wav"></audio></section><section><h2>3B. Version B</h2><audio controls preload="metadata" src="03-B.wav"></audio></section></div>
<section><h2>Score before revealing</h2><p>Pick A, B, or tie. Check: recognizable sung melody, timing, natural holds, clean retriggers, balanced left hand, and whether it sounds like one pianist rather than stacked stems.</p></section></body></html>""".replace(
            "__REFERENCE_SECTION__", reference_section
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--reference")
    parser.add_argument("--alignment")
    parser.add_argument(
        "--reference-already-aligned",
        action="store_true",
        help="Render reference note times directly instead of mapping them through an alignment again.",
    )
    parser.add_argument("--source-audio", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--duration-seconds", required=True, type=float)
    parser.add_argument("--reference-transpose-semitones", type=int, default=0)
    args = parser.parse_args()

    if args.reference_already_aligned:
        if not args.reference:
            parser.error("--reference-already-aligned requires --reference")
        if args.alignment:
            parser.error(
                "Do not supply --alignment when --reference-already-aligned is set"
            )
    elif bool(args.reference) != bool(args.alignment):
        parser.error("--reference and --alignment must be supplied together")

    paths = {
        "baseline": Path(args.baseline).resolve(),
        "candidate": Path(args.candidate).resolve(),
        "sourceAudio": Path(args.source_audio).resolve(),
    }
    if args.reference:
        paths["reference"] = Path(args.reference).resolve()
    if args.alignment:
        paths["alignment"] = Path(args.alignment).resolve()
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name} does not exist: {path}")
    samples = Path(args.samples).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty listening pack: {output}")
    output.mkdir(parents=True, exist_ok=True)

    salt = secrets.token_hex(32)
    candidate_is_a = bool(secrets.randbits(1))
    mapping = {
        "A": "candidate" if candidate_is_a else "baseline",
        "B": "baseline" if candidate_is_a else "candidate",
    }
    commitment_payload = {
        "salt": salt,
        "mapping": mapping,
        "baselineSha256": sha256_file(paths["baseline"]),
        "candidateSha256": sha256_file(paths["candidate"]),
    }
    commitment = canonical_hash(commitment_payload)
    atomic_text(output / "MAPPING-COMMITMENT.sha256", commitment + "\n")

    source_destination = output / "00-ORIGINAL-FULL-MIX.wav"
    source_render = copy_wav_excerpt(
        paths["sourceAudio"], source_destination, args.duration_seconds
    )
    render_rows = []
    if "reference" in paths:
        reference_payload = mapped_reference_payload(
            paths["reference"],
            paths.get("alignment"),
            args.reference_transpose_semitones,
            reference_already_aligned=args.reference_already_aligned,
        )
        render_rows.append(
            render_payload(
                reference_payload,
                samples,
                output / "01-REFERENCE.wav",
                duration_seconds=args.duration_seconds,
                hard_stop_seconds=args.duration_seconds,
            )
        )
    payloads = {
        "baseline": load_json(paths["baseline"]),
        "candidate": load_json(paths["candidate"]),
    }
    for label, filename in (("A", "02-A.wav"), ("B", "03-B.wav")):
        render_rows.append(
            render_payload(
                payloads[mapping[label]],
                samples,
                output / filename,
                duration_seconds=args.duration_seconds,
                hard_stop_seconds=args.duration_seconds,
            )
        )

    sealed = {
        "schema": "polymath-real-video-blind-map-v1",
        "createdAtUtc": datetime.now(timezone.utc).isoformat(),
        "warning": "Do not inspect until the listener has recorded A, B, or TIE.",
        "commitment": commitment,
        "commitmentPayload": commitment_payload,
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
    }
    atomic_text(output / "SEALED-MAPPING.json", json.dumps(sealed, indent=2) + "\n")
    render_manifest = {
        "schema": "polymath-fixed-piano-render-v2",
        "renderer": "Iowa University medium-dynamic 88-key samples; fixed velocity, envelope, master gain, and soft limiter",
        "perFileNormalization": False,
        "releaseSeconds": RELEASE_SECONDS,
        "durationSeconds": args.duration_seconds,
        "files": [
            {
                "file": Path(row["file"]).name,
                "sha256": row["sha256"],
                "seconds": row["seconds"],
                "peak": row["peak"],
                "rms": row["rms"],
            }
            for row in render_rows
        ],
        "originalFullMix": source_render,
    }
    atomic_text(output / "AUDIO-RENDER.json", json.dumps(render_manifest, indent=2) + "\n")
    atomic_text(
        output / "README-FIRST.md",
        """# Polymath real music-video blind review

Open `LISTEN.html`. Do not open `SEALED-MAPPING.json` first.

1. Hear the original full mix.
2. If supplied, hear the aligned piano-style reference.
3. Compare A and B at the same volume and on the same headphones/speakers.
4. Record A, B, or TIE in `SCORECARD.md`.
5. Only then open the sealed mapping.

This is private research evidence. Musical listening matters alongside
automated scores, especially when no independent piano reference exists.
""",
    )
    atomic_text(
        output / "SCORECARD.md",
        """# Blind verdict

Choice (A / B / TIE):

Melody recognition (0-5):
Timing (0-5):
Natural holds (0-5):
Retrigger cleanliness (0-5):
One-pianist musicality (0-5):

Notes:
""",
    )
    write_html(output / "LISTEN.html", has_reference="reference" in paths)
    print(
        json.dumps(
            {
                "output": str(output),
                "commitment": commitment,
                "files": 4 if "reference" in paths else 3,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
