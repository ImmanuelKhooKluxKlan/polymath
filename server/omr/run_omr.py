#!/usr/bin/env python3
"""CLI boundary used by the Node job queue."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from polymath_omr import OmrError, transcribe_pdf


def main() -> int:
    parser = argparse.ArgumentParser(description="Translate a PDF music score without an external API.")
    parser.add_argument("--input", required=True, help="Source PDF path")
    parser.add_argument("--output", required=True, help="Destination ready-to-play JSON path")
    parser.add_argument("--instrument", default="piano")
    parser.add_argument("--filename", default="")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--max-pages", type=int, default=20)
    args = parser.parse_args()
    try:
        result = transcribe_pdf(
            args.input, args.instrument, filename=args.filename or Path(args.input).name,
            dpi=max(180, min(450, args.dpi)), max_pages=max(1, min(200, args.max_pages)),
        )
        result["translatedAt"] = datetime.now(timezone.utc).isoformat()
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({
            "ok": True,
            "output": str(destination),
            "confidence": result.get("confidence", 0),
            "notes": len(result.get("notes", [])),
            "engine": result.get("omrDiagnostics", {}).get("engine", "unknown"),
        }))
        return 0
    except OmrError as error:
        print(json.dumps({"ok": False, "error": str(error)}), file=sys.stderr)
        return 2
    except Exception as error:  # Keep internal details in server logs, not HTTP responses.
        print(json.dumps({"ok": False, "error": f"Local OMR failed: {error}"}), file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
