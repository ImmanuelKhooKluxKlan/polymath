"""Polymath Musician's local optical-music-recognition engine.

Keep the heavy PDF and computer-vision stack lazy.  The piano arranger imports
``polymath_omr.performance`` but does not need OpenCV or PDFium; eagerly loading
the full OMR pipeline made lightweight inference workers depend on both.
"""

from typing import Any


__all__ = ["OmrError", "transcribe_pdf"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from .pipeline import OmrError, transcribe_pdf

        return {"OmrError": OmrError, "transcribe_pdf": transcribe_pdf}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
