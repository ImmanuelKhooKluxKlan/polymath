"""Deterministic computer-vision reader for engraved sheet-music PDFs.

The detector deliberately exposes intermediate evidence. Low-confidence marks
are rejected instead of being turned into invented notes.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import pypdfium2 as pdfium

from .music import TimeSignature, key_name, midi_to_name, staff_step_to_midi


@dataclass
class Staff:
    page: int
    index: int
    lines: tuple[float, float, float, float, float]
    spacing: float
    x1: int
    x2: int
    system: int = -1
    clef: str = "treble"

    @property
    def top(self) -> float:
        return self.lines[0]

    @property
    def bottom(self) -> float:
        return self.lines[-1]


@dataclass
class NoteMark:
    page: int
    staff_index: int
    system: int
    x: float
    y: float
    midi: int
    duration_beats: float
    confidence: float
    filled: bool
    stem: bool
    beam_count: int
    dotted: bool
    hand: str


@dataclass
class PageAnalysis:
    gray: np.ndarray
    binary: np.ndarray
    horizontal_mask: np.ndarray
    staffs: list[Staff]
    deskew_degrees: float
    text: str
    source_width: float
    source_height: float
    smufl_symbols: list[dict]


def _odd(value: int) -> int:
    value = max(3, int(value))
    return value if value % 2 else value + 1


def _pdfium_text_and_symbols(page: pdfium.PdfPage) -> tuple[str, list[dict]]:
    text_page = page.get_textpage()
    page_height = page.get_height()
    symbols = []
    character_count = text_page.count_chars()
    for index in range(character_count):
        value = text_page.get_text_range(index, 1)
        if len(value) != 1 or not 0xE000 <= ord(value) <= 0xF8FF:
            continue
        text_object = text_page.get_textobj(index)
        if text_object is None:
            continue
        matrix = text_object.get_matrix()
        left, bottom, right, top = text_page.get_charbox(index)
        font = text_object.get_font()
        symbols.append({
            "codepoint": ord(value),
            "x": float(matrix.e),
            # PDF coordinates start at bottom-left; vision starts at top-left.
            "y": float(page_height - matrix.f),
            "bbox": (float(left), float(page_height - top), float(right), float(page_height - bottom)),
            "font": str(font.get_base_name()),
            "size": float(text_object.get_font_size()),
        })
    text = text_page.get_text_range()
    text_page.close()
    return text, symbols


def render_pages(pdf_path: str | Path, dpi: int = 300, max_pages: int = 20) -> list[tuple]:
    rendered: list[tuple] = []
    with pdfium.PdfDocument(str(pdf_path)) as document:
        if len(document) == 0:
            raise ValueError("The PDF contains no pages.")
        if len(document) > max_pages:
            raise ValueError(f"The PDF has {len(document)} pages; the local reader accepts at most {max_pages}.")
        for page_index in range(len(document)):
            page = document[page_index]
            page_width, page_height = page.get_size()
            # Some notation programs export A4 content in page units five times
            # larger than PDF points. A blind 300-DPI multiplier would create a
            # 200-megapixel page. Cap the longest edge while retaining enough
            # pixels for notehead and beam geometry.
            requested_scale = dpi / 72.0
            scale = min(requested_scale, 5000.0 / max(page_width, page_height))
            bitmap = page.render(scale=scale, grayscale=True)
            gray = np.asarray(bitmap.to_pil().convert("L"), dtype=np.uint8)
            text, symbols = _pdfium_text_and_symbols(page)
            rendered.append((
                gray.copy(), text, symbols, float(page_width), float(page_height),
            ))
            bitmap.close()
            page.close()
    return rendered


def deskew(gray: np.ndarray) -> tuple[np.ndarray, float]:
    preview_scale = min(1.0, 1800.0 / max(gray.shape))
    preview = cv2.resize(gray, None, fx=preview_scale, fy=preview_scale, interpolation=cv2.INTER_AREA)
    edges = cv2.Canny(preview, 60, 180)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 720, threshold=max(70, preview.shape[1] // 12),
        minLineLength=max(120, preview.shape[1] // 5), maxLineGap=max(12, preview.shape[1] // 100),
    )
    angles = []
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            if abs(angle) <= 6:
                angles.append(angle)
    angle = float(np.median(angles)) if angles else 0.0
    if abs(angle) < 0.08:
        return gray, angle
    height, width = gray.shape
    rotation = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    corrected = cv2.warpAffine(gray, rotation, (width, height), flags=cv2.INTER_CUBIC, borderValue=255)
    return corrected, angle


def binarize(gray: np.ndarray) -> np.ndarray:
    # Adaptive thresholding survives shadows and uneven phone scans better than
    # one global threshold. Ink is represented as 255 for morphology operations.
    block = _odd(min(81, max(31, round(min(gray.shape) / 32))))
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, 13,
    )


def horizontal_staff_mask(binary: np.ndarray) -> np.ndarray:
    width = binary.shape[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(35, width // 45), 1))
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)


def _runs(indices: np.ndarray) -> list[tuple[int, int]]:
    if len(indices) == 0:
        return []
    splits = np.where(np.diff(indices) > 1)[0] + 1
    groups = np.split(indices, splits)
    return [(int(group[0]), int(group[-1])) for group in groups if len(group)]


def _candidate_line_centers(mask: np.ndarray) -> list[float]:
    score = np.count_nonzero(mask, axis=1) / max(1, mask.shape[1])
    positive = score[score > 0]
    if not len(positive):
        return []
    threshold = max(0.055, min(0.30, float(np.percentile(positive, 70)) * 0.55))
    return [(start + end) / 2 for start, end in _runs(np.flatnonzero(score >= threshold))]


def _staff_groups(candidates: list[float], page_height: int) -> list[tuple[float, ...]]:
    if len(candidates) < 5:
        return []
    candidates_array = np.asarray(candidates, dtype=float)
    proposals: list[tuple[float, tuple[float, ...]]] = []
    min_spacing = max(4.0, page_height / 900.0)
    max_spacing = max(24.0, page_height / 55.0)
    for start in candidates:
        for next_line in candidates:
            spacing = next_line - start
            if not min_spacing <= spacing <= max_spacing:
                continue
            chosen = []
            errors = []
            for index in range(5):
                target = start + index * spacing
                nearest_index = int(np.argmin(np.abs(candidates_array - target)))
                nearest = float(candidates_array[nearest_index])
                error = abs(nearest - target)
                if error > max(2.2, spacing * 0.26):
                    break
                chosen.append(nearest)
                errors.append(error)
            if len(set(chosen)) == 5:
                intervals = np.diff(chosen)
                consistency = float(np.std(intervals) / max(1.0, np.mean(intervals)))
                score = sum(errors) / spacing + consistency * 5
                proposals.append((score, tuple(chosen)))
    accepted: list[tuple[float, ...]] = []
    for _, proposal in sorted(proposals, key=lambda item: item[0]):
        if any(abs(proposal[2] - existing[2]) < np.mean(np.diff(proposal)) * 2.5 for existing in accepted):
            continue
        accepted.append(proposal)
    return sorted(accepted, key=lambda lines: lines[0])


def detect_staffs(binary: np.ndarray, page: int) -> tuple[list[Staff], np.ndarray]:
    mask = horizontal_staff_mask(binary)
    groups = _staff_groups(_candidate_line_centers(mask), binary.shape[0])
    staffs = []
    for index, lines in enumerate(groups):
        spacing = float(np.median(np.diff(lines)))
        rows = []
        for y in lines:
            start, end = max(0, round(y - 1)), min(mask.shape[0], round(y + 2))
            rows.append(np.flatnonzero(np.any(mask[start:end] > 0, axis=0)))
        ink_x = np.concatenate([row for row in rows if len(row)]) if any(len(row) for row in rows) else np.array([])
        if not len(ink_x):
            continue
        x1, x2 = int(np.percentile(ink_x, 1)), int(np.percentile(ink_x, 99))
        if x2 - x1 < binary.shape[1] * 0.18:
            continue
        staffs.append(Staff(page, index, tuple(lines), spacing, x1, x2))
    return staffs, mask


def assign_systems(staffs: list[Staff], instrument: str) -> None:
    if not staffs:
        return
    if instrument == "piano":
        system = 0
        cursor = 0
        while cursor < len(staffs):
            top = staffs[cursor]
            top.system = system
            top.clef = "treble"
            if cursor + 1 < len(staffs):
                lower = staffs[cursor + 1]
                gap = lower.top - top.bottom
                comparable = 0.55 <= lower.spacing / top.spacing <= 1.8
                if comparable and gap < max(top.spacing, lower.spacing) * 14:
                    lower.system = system
                    lower.clef = "bass"
                    cursor += 1
            system += 1
            cursor += 1
        return
    bass_instruments = {"upright_bass", "bass", "cello", "tuba", "bassoon"}
    for system, staff in enumerate(staffs):
        staff.system = system
        staff.clef = "bass" if instrument in bass_instruments else "treble"


def analyze_page(
    gray: np.ndarray,
    text: str,
    page_number: int,
    instrument: str,
    smufl_symbols: list[dict] | None = None,
    source_width: float | None = None,
    source_height: float | None = None,
) -> PageAnalysis:
    corrected, angle = deskew(gray)
    binary = binarize(corrected)
    staffs, mask = detect_staffs(binary, page_number)
    assign_systems(staffs, instrument)
    return PageAnalysis(
        corrected, binary, mask, staffs, angle, text,
        float(source_width or corrected.shape[1]), float(source_height or corrected.shape[0]),
        list(smufl_symbols or []),
    )


def detect_barlines(page: PageAnalysis, staffs: list[Staff]) -> list[float]:
    if not staffs:
        return []
    x1 = max(staff.x1 for staff in staffs)
    x2 = min(staff.x2 for staff in staffs)
    top = max(0, int(min(staff.top for staff in staffs) - staffs[0].spacing))
    bottom = min(page.binary.shape[0], int(max(staff.bottom for staff in staffs) + staffs[-1].spacing))
    crop = page.binary[top:bottom, x1:x2]
    if not crop.size:
        return []
    full_bar_height = max(8, bottom - top)
    vertical = cv2.morphologyEx(
        crop, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(8, int(full_bar_height * 0.68)))),
    )
    score = np.count_nonzero(vertical, axis=0)
    required = max(6, int(full_bar_height * 0.62))
    minimum_thickness = max(3, int(round(np.median([staff.spacing for staff in staffs]) * 0.13)))
    centers = [
        (start + end) / 2 + x1
        for start, end in _runs(np.flatnonzero(score >= required))
        if end - start + 1 >= minimum_thickness
    ]
    left_guard = x1 + np.median([staff.spacing for staff in staffs]) * 5.5
    return [center for center in centers if left_guard < center < x2 - 2]


def _vertical_strength(binary: np.ndarray, x: float, y: float, spacing: float) -> tuple[bool, int, int]:
    h, w = binary.shape
    x1, x2 = max(0, int(x - spacing)), min(w, int(x + spacing) + 1)
    y1, y2 = max(0, int(y - 3.7 * spacing)), min(h, int(y + 3.7 * spacing) + 1)
    roi = binary[y1:y2, x1:x2]
    if not roi.size:
        return False, int(x), 0
    kernel_height = max(5, int(spacing * 1.8))
    vertical = cv2.morphologyEx(roi, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_height)))
    scores = np.count_nonzero(vertical, axis=0)
    best = int(np.argmax(scores))
    strength = int(scores[best])
    return strength >= kernel_height, x1 + best, strength


def _beam_count(binary: np.ndarray, x: float, y: float, stem_x: int, spacing: float) -> int:
    # Count dense horizontal bands near the end of the stem. This recognizes
    # straight and mildly slanted beams while avoiding the five known staff rows.
    direction_up = stem_x >= x
    y1 = int(y - 4.0 * spacing) if direction_up else int(y + 0.8 * spacing)
    y2 = int(y - 0.7 * spacing) if direction_up else int(y + 4.0 * spacing)
    y1, y2 = sorted((max(0, y1), min(binary.shape[0], y2)))
    x1, x2 = max(0, int(stem_x - spacing * 0.5)), min(binary.shape[1], int(stem_x + spacing * 3.0))
    roi = binary[y1:y2, x1:x2]
    if not roi.size:
        return 0
    row_score = np.count_nonzero(roi, axis=1)
    bands = _runs(np.flatnonzero(row_score >= max(3, int(spacing * 0.75))))
    return min(2, sum(1 for start, end in bands if end - start + 1 >= max(1, spacing * 0.12)))


def _is_dotted(binary: np.ndarray, x: float, y: float, spacing: float) -> bool:
    y1, y2 = max(0, int(y - 0.55 * spacing)), min(binary.shape[0], int(y + 0.55 * spacing))
    x1, x2 = max(0, int(x + 0.75 * spacing)), min(binary.shape[1], int(x + 2.1 * spacing))
    roi = binary[y1:y2, x1:x2]
    count, _, stats, _ = cv2.connectedComponentsWithStats((roi > 0).astype(np.uint8), 8)
    for index in range(1, count):
        width, height, area = stats[index, cv2.CC_STAT_WIDTH], stats[index, cv2.CC_STAT_HEIGHT], stats[index, cv2.CC_STAT_AREA]
        if 1 <= width <= spacing * 0.55 and 1 <= height <= spacing * 0.55 and area >= 2:
            return True
    return False


def detect_notes(page: PageAnalysis, staff: Staff, instrument: str) -> tuple[list[NoteMark], int]:
    spacing = staff.spacing
    y1 = max(0, int(staff.top - 4.0 * spacing))
    y2 = min(page.binary.shape[0], int(staff.bottom + 4.0 * spacing))
    x1 = max(0, int(staff.x1 + 4.8 * spacing))
    x2 = min(page.binary.shape[1], int(staff.x2 - 0.5 * spacing))
    crop = page.binary[y1:y2, x1:x2].copy()
    crop_lines = page.horizontal_mask[y1:y2, x1:x2]
    clean = cv2.bitwise_and(crop, cv2.bitwise_not(crop_lines))
    vertical = cv2.morphologyEx(crop, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(3, int(1.7 * spacing)))))
    clean = cv2.bitwise_or(clean, vertical)

    kernel_width = _odd(max(3, int(spacing * 0.48)))
    kernel_height = _odd(max(3, int(spacing * 0.30)))
    heads = cv2.morphologyEx(clean, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_width, kernel_height)))
    heads = cv2.morphologyEx(heads, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    contours, hierarchy = cv2.findContours(heads, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    marks: list[NoteMark] = []
    rejected = 0
    for contour_index, contour in enumerate(contours):
        bx, by, bw, bh = cv2.boundingRect(contour)
        if not (spacing * 0.45 <= bw <= spacing * 2.2 and spacing * 0.32 <= bh <= spacing * 1.45):
            continue
        area = cv2.contourArea(contour)
        fill_ratio = area / max(1.0, bw * bh)
        if fill_ratio < 0.20:
            rejected += 1
            continue
        cx, cy = x1 + bx + bw / 2, y1 + by + bh / 2
        if any(abs(cx - prior.x) < spacing * 0.33 and abs(cy - prior.y) < spacing * 0.33 for prior in marks):
            continue
        original_roi = page.binary[max(0, int(cy - spacing * 0.32)):int(cy + spacing * 0.32) + 1,
                                   max(0, int(cx - spacing * 0.48)):int(cx + spacing * 0.48) + 1]
        center_fill = np.count_nonzero(original_roi) / max(1, original_roi.size)
        filled = center_fill >= 0.42
        stem, stem_x, stem_strength = _vertical_strength(page.binary, cx, cy, spacing)
        beams = _beam_count(page.binary, cx, cy, stem_x, spacing) if stem else 0
        if not stem:
            duration = 4.0 if not filled else 1.0
        elif not filled:
            duration = 2.0
        else:
            duration = 1.0 / (2 ** beams)
        dotted = _is_dotted(page.binary, cx, cy, spacing)
        if dotted:
            duration *= 1.5
        raw_step = (staff.bottom - cy) / (spacing / 2.0)
        staff_step = int(round(raw_step))
        pitch_alignment = max(0.0, 1.0 - abs(raw_step - staff_step) / 0.55)
        midi = staff_step_to_midi(staff.clef, staff_step)
        if not 21 <= midi <= 108:
            rejected += 1
            continue
        shape_score = min(1.0, max(0.0, (fill_ratio - 0.18) / 0.42))
        rhythm_score = 0.90 if stem else (0.72 if not filled else 0.42)
        confidence = 0.42 * shape_score + 0.33 * pitch_alignment + 0.25 * rhythm_score
        if stem_strength > spacing * 2.5:
            confidence = min(1.0, confidence + 0.05)
        if confidence < 0.43:
            rejected += 1
            continue
        marks.append(NoteMark(
            page=staff.page, staff_index=staff.index, system=staff.system,
            x=round(cx, 3), y=round(cy, 3), midi=midi,
            duration_beats=duration, confidence=round(confidence, 4),
            filled=filled, stem=stem, beam_count=beams, dotted=dotted,
            hand="left" if instrument == "piano" and staff.clef == "bass" else "right",
        ))
    return sorted(marks, key=lambda item: (item.x, item.midi)), rejected


NOTEHEAD_DURATIONS = {
    0xE0A0: 8.0,  # noteheadDoubleWhole
    0xE0A1: 8.0,  # noteheadDoubleWholeSquare
    0xE0A2: 4.0,  # noteheadWhole
    0xE0A3: 2.0,  # noteheadHalf
    0xE0A4: 1.0,  # noteheadBlack; beams/flags shorten this below
}
ACCIDENTAL_ALTERS = {
    0xE260: -1, 0xE261: 0, 0xE262: 1, 0xE263: 2, 0xE264: -2,
}
FLAG_BEAMS = {
    0xE240: 1, 0xE241: 1, 0xE242: 2, 0xE243: 2,
    0xE244: 3, 0xE245: 3, 0xE246: 4, 0xE247: 4,
}
AUGMENTATION_DOT = 0xE1E7


def _scaled_symbols(page: PageAnalysis, accepted_codes: set[int]) -> list[dict]:
    scale_x = page.gray.shape[1] / max(1.0, page.source_width)
    scale_y = page.gray.shape[0] / max(1.0, page.source_height)
    return [
        {**symbol, "x": symbol["x"] * scale_x, "y": symbol["y"] * scale_y}
        for symbol in page.smufl_symbols if symbol["codepoint"] in accepted_codes
    ]


def _measure_for_x(boundaries: list[float], x: float) -> int:
    return max(0, min(len(boundaries) - 2, int(np.searchsorted(boundaries, x, side="right") - 1)))


def detect_smufl_notes(
    page: PageAnalysis,
    staff: Staff,
    boundaries: list[float],
    instrument: str,
) -> tuple[list[NoteMark], int, int]:
    """Read semantically encoded music glyphs before considering raster blobs."""
    accepted = set(NOTEHEAD_DURATIONS) | set(ACCIDENTAL_ALTERS) | set(FLAG_BEAMS) | {
        AUGMENTATION_DOT, 0xE050, 0xE062,
    }
    symbols = _scaled_symbols(page, accepted)
    spacing = staff.spacing

    def distance_to_staff(symbol: dict, candidate: Staff) -> float:
        if candidate.top <= symbol["y"] <= candidate.bottom:
            return 0.0
        return min(abs(symbol["y"] - candidate.top), abs(symbol["y"] - candidate.bottom)) / candidate.spacing

    def owned_by_current_staff(symbol: dict) -> bool:
        owner = min(page.staffs, key=lambda candidate: (
            distance_to_staff(symbol, candidate), abs(symbol["y"] - (candidate.top + candidate.bottom) / 2),
        ))
        return owner.index == staff.index

    relevant = [
        symbol for symbol in symbols
        if staff.x1 - spacing <= symbol["x"] <= staff.x2 + spacing
        and staff.top - 5 * spacing <= symbol["y"] <= staff.bottom + 5 * spacing
        and owned_by_current_staff(symbol)
    ]
    clefs = [symbol for symbol in relevant if symbol["codepoint"] in {0xE050, 0xE062}]
    if clefs:
        clef = min(clefs, key=lambda symbol: symbol["x"])
        staff.clef = "bass" if clef["codepoint"] == 0xE062 else "treble"
    heads = sorted(
        [symbol for symbol in relevant if symbol["codepoint"] in NOTEHEAD_DURATIONS],
        key=lambda symbol: (symbol["x"], symbol["y"]),
    )
    if not heads:
        return [], 0, 0
    accidentals = [symbol for symbol in relevant if symbol["codepoint"] in ACCIDENTAL_ALTERS]
    first_head_x = heads[0]["x"]
    signature_symbols = [
        symbol for symbol in accidentals
        if symbol["x"] < first_head_x - spacing * 0.3
    ]
    flat_count = len([symbol for symbol in signature_symbols if symbol["codepoint"] == 0xE260])
    sharp_count = len([symbol for symbol in signature_symbols if symbol["codepoint"] == 0xE262])
    key_fifths = min(7, sharp_count) if sharp_count else -min(7, flat_count)
    inline_accidentals = [symbol for symbol in accidentals if symbol not in signature_symbols]
    flags = [symbol for symbol in relevant if symbol["codepoint"] in FLAG_BEAMS]
    dots = [symbol for symbol in relevant if symbol["codepoint"] == AUGMENTATION_DOT]

    accidental_state: dict[tuple[int, int], tuple[float, int]] = {}
    accidental_rows = []
    for accidental in inline_accidentals:
        raw_step = (staff.bottom - accidental["y"]) / (spacing / 2.0)
        step = int(round(raw_step))
        if abs(raw_step - step) <= 0.62:
            accidental_rows.append((
                _measure_for_x(boundaries, accidental["x"]), step,
                accidental["x"], ACCIDENTAL_ALTERS[accidental["codepoint"]],
            ))

    marks = []
    for head in heads:
        raw_step = (staff.bottom - head["y"]) / (spacing / 2.0)
        staff_step = int(round(raw_step))
        if abs(raw_step - staff_step) > 0.62:
            continue
        measure_index = _measure_for_x(boundaries, head["x"])
        for accidental_measure, accidental_step, accidental_x, alter in accidental_rows:
            if accidental_measure == measure_index and accidental_step == staff_step and accidental_x < head["x"]:
                accidental_state[(measure_index, staff_step)] = (accidental_x, alter)
        explicit = accidental_state.get((measure_index, staff_step))
        midi = staff_step_to_midi(staff.clef, staff_step, key_fifths)
        if explicit:
            midi = staff_step_to_midi(staff.clef, staff_step, 0) + explicit[1]
        if not 21 <= midi <= 108:
            continue
        base_duration = NOTEHEAD_DURATIONS[head["codepoint"]]
        stem, stem_x, _ = _vertical_strength(page.binary, head["x"], head["y"], spacing)
        raster_beams = _beam_count(page.binary, head["x"], head["y"], stem_x, spacing) if stem else 0
        nearby_flags = [
            FLAG_BEAMS[flag["codepoint"]] for flag in flags
            if abs(flag["x"] - stem_x) <= spacing * 1.5
            and abs(flag["y"] - head["y"]) <= spacing * 4.8
        ]
        beam_count = max([raster_beams, *nearby_flags], default=0)
        duration = base_duration
        if head["codepoint"] == 0xE0A4 and beam_count:
            duration = 1.0 / (2 ** beam_count)
        dotted = any(
            0.45 * spacing <= dot["x"] - head["x"] <= 2.0 * spacing
            and abs(dot["y"] - head["y"]) <= 0.7 * spacing
            for dot in dots
        )
        if not dotted:
            dotted = _is_dotted(page.binary, head["x"], head["y"], spacing)
        if dotted:
            duration *= 1.5
        rhythm_confidence = 0.98 if head["codepoint"] != 0xE0A4 or beam_count or dotted else 0.82
        confidence = 0.82 * 0.995 + 0.18 * rhythm_confidence
        marks.append(NoteMark(
            page=staff.page, staff_index=staff.index, system=staff.system,
            x=round(head["x"], 3), y=round(head["y"], 3), midi=midi,
            duration_beats=duration, confidence=round(confidence, 4),
            filled=head["codepoint"] == 0xE0A4, stem=stem,
            beam_count=beam_count, dotted=dotted,
            hand="left" if instrument == "piano" and staff.clef == "bass" else "right",
        ))
    return marks, max(0, len(heads) - len(marks)), key_fifths


def _metadata(page_texts: list[str], filename: str) -> tuple[str, str, float, TimeSignature, list[str]]:
    text = "\n".join(page_texts)
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines() if line.strip()]
    title = Path(filename).stem.replace("_", " ").strip()
    composer = ""
    for line_index, line in enumerate(lines[:12]):
        lower = line.lower()
        if "composer" in lower or "music by" in lower or "composed by" in lower:
            composer = re.sub(r"(?i).*(composer|music by|composed by)\s*[:\-]?\s*", "", line)[:180]
            if line_index + 1 < len(lines):
                continuation = lines[line_index + 1]
                letters = [character for character in continuation if character.isalpha()]
                uppercase_ratio = sum(character.isupper() for character in letters) / max(1, len(letters))
                if uppercase_ratio > 0.75 and len(composer) + len(continuation) < 178:
                    composer = f"{composer} {continuation}".strip()
        elif not title and 3 <= len(line) <= 100:
            title = line
    tempo_match = re.search(r"(?:bpm|[=])\s*(\d{2,3})(?:\b|\s)", text, re.IGNORECASE)
    tempo = float(tempo_match.group(1)) if tempo_match and 20 <= int(tempo_match.group(1)) <= 300 else 120.0
    signature_match = re.search(r"\b(\d{1,2})\s*/\s*(1|2|4|8|16|32)\b", text)
    signature = TimeSignature(int(signature_match.group(1)), int(signature_match.group(2))) if signature_match else TimeSignature()
    warnings = []
    if not tempo_match:
        warnings.append("No machine-readable metronome mark was found; playback uses 120 BPM.")
    if not signature_match:
        warnings.append("No machine-readable time signature was found; measure timing uses 4/4.")
    return title[:180], composer[:180], tempo, signature, warnings


def _smufl_time_signature(page_analyses: list[PageAnalysis]) -> TimeSignature | None:
    candidates = []
    for page in page_analyses:
        scale_x = page.gray.shape[1] / max(1.0, page.source_width)
        scale_y = page.gray.shape[0] / max(1.0, page.source_height)
        symbols = [
            {**symbol, "x": symbol["x"] * scale_x, "y": symbol["y"] * scale_y}
            for symbol in page.smufl_symbols
            if 0xE080 <= symbol["codepoint"] <= 0xE08B
        ]
        for staff in page.staffs:
            nearby = [
                symbol for symbol in symbols
                if staff.x1 <= symbol["x"] <= staff.x1 + staff.spacing * 10
                and staff.top - staff.spacing <= symbol["y"] <= staff.bottom + staff.spacing
            ]
            if any(symbol["codepoint"] == 0xE08A for symbol in nearby):
                candidates.append((4, 4))
                continue
            if any(symbol["codepoint"] == 0xE08B for symbol in nearby):
                candidates.append((2, 2))
                continue
            digit_symbols = [symbol for symbol in nearby if symbol["codepoint"] <= 0xE089]
            if len(digit_symbols) >= 2:
                ordered = sorted(digit_symbols, key=lambda symbol: symbol["y"])
                numerator = ordered[0]["codepoint"] - 0xE080
                denominator = ordered[-1]["codepoint"] - 0xE080
                if 1 <= numerator <= 32 and denominator in {1, 2, 4, 8, 16, 32}:
                    candidates.append((numerator, denominator))
    if not candidates:
        return None
    numerator, denominator = max(set(candidates), key=candidates.count)
    return TimeSignature(numerator, denominator)


def reconstruct(page_analyses: list[PageAnalysis], filename: str, instrument: str) -> dict:
    title, composer, bpm, signature, warnings = _metadata([page.text for page in page_analyses], filename)
    all_marks: list[tuple[NoteMark, list[float], int]] = []
    system_measure_counts: dict[tuple[int, int], int] = {}
    rejected = 0
    system_counter = 0
    staff_diagnostics = []
    semantic_available = any(
        symbol["codepoint"] in NOTEHEAD_DURATIONS
        for page in page_analyses for symbol in page.smufl_symbols
    )
    semantic_signature = _smufl_time_signature(page_analyses) if semantic_available else None
    if semantic_signature:
        signature = semantic_signature
        warnings = [warning for warning in warnings if "time signature" not in warning.lower()]
    detected_key_fifths = []
    repeat_markers: list[tuple[int, int, str]] = []

    for page in page_analyses:
        page_systems = sorted({staff.system for staff in page.staffs})
        for local_system in page_systems:
            staffs = [staff for staff in page.staffs if staff.system == local_system]
            bars = detect_barlines(page, staffs)
            spacing = float(np.median([staff.spacing for staff in staffs]))
            common_x1 = max(staff.x1 for staff in staffs) + 4.8 * spacing
            common_x2 = min(staff.x2 for staff in staffs)
            boundaries = [common_x1] + [bar for bar in bars if common_x1 + spacing < bar < common_x2] + [common_x2]
            boundaries = sorted(boundaries)
            deduplicated = [boundaries[0]]
            for boundary in boundaries[1:]:
                if boundary - deduplicated[-1] >= spacing * 2:
                    deduplicated.append(boundary)
            if len(deduplicated) < 2:
                deduplicated = [common_x1, common_x2]
            if semantic_available:
                scale_x = page.gray.shape[1] / max(1.0, page.source_width)
                scale_y = page.gray.shape[0] / max(1.0, page.source_height)
                repeat_points = [
                    (symbol["x"] * scale_x, symbol["y"] * scale_y)
                    for symbol in page.smufl_symbols if symbol["codepoint"] == 0xE044
                    and min(staff.top for staff in staffs) - spacing <= symbol["y"] * scale_y
                    <= max(staff.bottom for staff in staffs) + spacing
                ]
                repeat_xs = []
                for repeat_x, _ in sorted(repeat_points):
                    if not repeat_xs or abs(repeat_x - repeat_xs[-1]) > spacing * 0.35:
                        repeat_xs.append(repeat_x)
                for repeat_x in repeat_xs:
                    boundary_index = min(
                        range(len(deduplicated)), key=lambda index: abs(deduplicated[index] - repeat_x),
                    )
                    boundary_x = deduplicated[boundary_index]
                    if abs(boundary_x - repeat_x) <= spacing * 2.5:
                        repeat_markers.append((
                            system_counter, boundary_index, "start" if repeat_x > boundary_x else "end",
                        ))
            system_measure_counts[(page.staffs[0].page, system_counter)] = len(deduplicated) - 1
            for staff in staffs:
                if semantic_available:
                    marks, staff_rejected, staff_key_fifths = detect_smufl_notes(
                        page, staff, deduplicated, instrument,
                    )
                    detected_key_fifths.append(staff_key_fifths)
                else:
                    marks, staff_rejected = detect_notes(page, staff, instrument)
                rejected += staff_rejected
                staff_diagnostics.append({
                    **asdict(staff), "detectedNotes": len(marks), "rejectedMarks": staff_rejected,
                })
                for mark in marks:
                    mark.system = system_counter
                    all_marks.append((mark, deduplicated, len(deduplicated) - 1))
            system_counter += 1

    measure_beats = signature.measure_quarter_beats
    system_offsets = {}
    running_beats = 0.0
    for system in range(system_counter):
        counts = [count for (_, sys), count in system_measure_counts.items() if sys == system]
        count = counts[0] if counts else 1
        system_offsets[system] = running_beats
        running_beats += count * measure_beats

    notes = []
    for mark, boundaries, _ in all_marks:
        measure_index = max(0, min(len(boundaries) - 2, int(np.searchsorted(boundaries, mark.x, side="right") - 1)))
        left, right = boundaries[measure_index], boundaries[measure_index + 1]
        normalized_x = min(0.999, max(0.0, (mark.x - left) / max(1.0, right - left)))
        onset_inside = round(normalized_x * measure_beats * 4) / 4
        beat = system_offsets.get(mark.system, 0.0) + measure_index * measure_beats + onset_inside
        duration_beats = min(mark.duration_beats, max(0.0625, measure_beats - onset_inside))
        seconds_per_beat = 60.0 / bpm
        notes.append({
            "note": midi_to_name(mark.midi),
            "time": round(beat * seconds_per_beat, 6),
            "duration": round(max(0.03, duration_beats * seconds_per_beat), 6),
            "velocity": round(0.62 + mark.confidence * 0.24, 4),
            "hand": mark.hand,
            "voice": f"Staff {mark.staff_index + 1}",
            "articulation": "",
            "_confidence": mark.confidence,
        })

    repeat_events = sorted((
        system_offsets.get(system, 0.0) + boundary_index * measure_beats,
        marker_type,
    ) for system, boundary_index, marker_type in repeat_markers)
    repeat_pairs = []
    active_repeat_start = 0.0
    for repeat_beat, marker_type in repeat_events:
        if marker_type == "start":
            active_repeat_start = repeat_beat
        elif repeat_beat > active_repeat_start:
            repeat_pairs.append((active_repeat_start, repeat_beat))
            active_repeat_start = repeat_beat
    if repeat_pairs:
        seconds_per_beat = 60.0 / bpm
        cursor_seconds = 0.0
        accumulated = 0.0
        expanded = []
        for start_beat, end_beat in repeat_pairs:
            start_seconds, end_seconds = start_beat * seconds_per_beat, end_beat * seconds_per_beat
            if end_seconds <= cursor_seconds:
                continue
            expanded.extend({**note, "time": round(note["time"] + accumulated, 6)} for note in notes
                            if cursor_seconds <= note["time"] < end_seconds)
            expanded.extend({**note, "time": round(end_seconds + accumulated + note["time"] - start_seconds, 6)}
                            for note in notes if start_seconds <= note["time"] < end_seconds)
            accumulated += end_seconds - start_seconds
            cursor_seconds = end_seconds
        expanded.extend({**note, "time": round(note["time"] + accumulated, 6)}
                        for note in notes if note["time"] >= cursor_seconds)
        notes = expanded

    # Merge duplicate detections created where staff lines cross a filled head,
    # but preserve real chords and repeated notes at separate quantized onsets.
    unique = {}
    for note in sorted(notes, key=lambda item: (item["time"], item["note"], -item["_confidence"])):
        key = (note["time"], note["note"], note["hand"])
        if key not in unique:
            unique[key] = note
    notes = list(unique.values())
    confidences = [note.pop("_confidence") for note in notes]
    detected_staffs = sum(len(page.staffs) for page in page_analyses)
    if not detected_staffs:
        raise ValueError("No five-line music staffs were detected. Use a straight, high-resolution printed score.")
    if not notes:
        raise ValueError("Staffs were found, but no noteheads passed the accuracy checks.")
    coverage = len(notes) / max(1, len(notes) + rejected)
    staff_quality = min(1.0, detected_staffs / max(1, system_counter * (2 if instrument == "piano" else 1)))
    confidence = float(np.clip(np.mean(confidences) * 0.72 + coverage * 0.18 + staff_quality * 0.10, 0, 1))
    # Semantic glyphs make pitch detection exceptionally strong, but without
    # MusicXML the playback structure still contains inferred rhythm. Keep the
    # public score calibrated instead of presenting false 99% certainty.
    confidence = min(confidence, 0.92 if semantic_available else 0.78)
    if confidence < 0.48:
        raise ValueError(f"Notation confidence was only {confidence:.0%}; the score was rejected rather than guessing notes.")

    if semantic_available:
        warnings.extend([
            "The PDF's SMuFL noteheads, clefs, and standard accidentals were read directly; beamed-note duration is cross-checked visually.",
            "Tuplets, ornaments, repeat jumps, ties, lyrics, dynamics, and pedal marks still require review when no embedded MusicXML is present.",
        ])
    else:
        warnings.extend([
            "Computer-vision mode currently assumes the printed clef is treble, or treble/bass for paired piano staffs.",
            "Key-signature accidentals, inline accidentals, tuplets, ornaments, repeat jumps, ties, lyrics, dynamics, and pedal marks require review unless embedded MusicXML was available.",
            "Every accepted note includes geometric evidence; uncertain marks were omitted rather than invented.",
        ])
    if detected_key_fifths:
        key_fifths = max(set(detected_key_fifths), key=detected_key_fifths.count)
    else:
        key_fifths = 0
    engine = "polymath-smufl-pdf-v1" if semantic_available else "polymath-classical-vision-v1"
    if repeat_pairs:
        warnings.append(f"Expanded {len(repeat_pairs)} standard repeat section(s) into playback order.")
    return {
        "isInstrumentalMusicSheet": True,
        "rejectionReason": "",
        "title": title,
        "composer": composer,
        "instrument": instrument,
        "bpm": bpm,
        "timeSignature": {"numerator": signature.numerator, "denominator": signature.denominator},
        "keySignature": key_name(key_fifths),
        "notes": sorted(notes, key=lambda item: (item["time"], item["note"])),
        "events": [],
        "tabs": [],
        "pedals": [],
        "warnings": warnings,
        "confidence": round(confidence, 4),
        "omrDiagnostics": {
            "engine": engine,
            "pages": len(page_analyses),
            "staffs": detected_staffs,
            "systems": system_counter,
            "measures": sum(system_measure_counts.values()),
            "acceptedNotes": len(notes),
            "rejectedMarks": rejected,
            "deskewDegrees": [round(page.deskew_degrees, 4) for page in page_analyses],
            "staffDetails": staff_diagnostics,
            "limitationsRequireReview": True,
            "semanticGlyphs": semantic_available,
            "repeatSectionsExpanded": len(repeat_pairs),
            "confidenceComponents": {
                "noteheadDetection": 0.995 if semantic_available else round(coverage, 4),
                "pitchMapping": 0.98 if semantic_available else round(float(np.mean(confidences)), 4),
                "rhythmAndPlaybackStructure": 0.82 if semantic_available else 0.55,
            },
        },
    }
