"""Polymath Musician's local optical-music-recognition engine."""

from .pipeline import OmrError, transcribe_pdf

__all__ = ["OmrError", "transcribe_pdf"]
