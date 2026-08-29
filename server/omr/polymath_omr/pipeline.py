"""High-level provider-free PDF-to-playable-sheet pipeline."""

from __future__ import annotations

from pathlib import Path

from .musicxml import find_embedded_musicxml, parse_musicxml
from .vision import analyze_page, reconstruct, render_pages


class OmrError(RuntimeError):
    """A user-safe optical music recognition failure."""


def transcribe_pdf(
    pdf_path: str | Path,
    instrument: str = "piano",
    *,
    filename: str | None = None,
    dpi: int = 300,
    max_pages: int = 20,
) -> dict:
    source = Path(pdf_path)
    if not source.is_file():
        raise OmrError("The PDF source file does not exist.")
    if source.read_bytes()[:5] != b"%PDF-":
        raise OmrError("The selected source is not a valid PDF file.")
    display_name = filename or source.name
    try:
        embedded = find_embedded_musicxml(source)
        if embedded:
            payload, embedded_name = embedded
            result = parse_musicxml(payload, instrument, embedded_name)
        else:
            rendered = render_pages(source, dpi=dpi, max_pages=max_pages)
            pages = [
                analyze_page(
                    gray, text, index + 1, instrument, symbols, source_width, source_height,
                )
                for index, (gray, text, symbols, source_width, source_height) in enumerate(rendered)
            ]
            result = reconstruct(pages, display_name, instrument)
    except (OSError, ValueError) as error:
        raise OmrError(str(error)) from error
    result["sourcePdf"] = display_name
    result["readyToPlayFormat"] = "polymath-musician-json-v1"
    result["translationProvider"] = "Polymath Local OMR"
    result["modelLicense"] = "Proprietary Polymath code; no external inference API"
    return result
