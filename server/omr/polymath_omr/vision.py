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
from pypdfium2 import raw as pdfium_raw

from .music import TimeSignature, key_name, midi_to_name, staff_step_to_midi
from .performance import shape_piano_performance


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
    articulation: str = ""
    stem_direction: str = ""
    duration_source: str = "visual"


@dataclass
class RestMark:
    page: int
    staff_index: int
    system: int
    x: float
    y: float
    duration_beats: float
    dotted: bool = False


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
    text_characters: list[dict]
    vector_curves: list[dict]


def _odd(value: int) -> int:
    value = max(3, int(value))
    return value if value % 2 else value + 1


def _pdfium_page_evidence(page: pdfium.PdfPage) -> tuple[str, list[dict], list[dict], list[dict]]:
    text_page = page.get_textpage()
    page_height = page.get_height()
    symbols = []
    text_characters = []
    character_count = text_page.count_chars()
    for index in range(character_count):
        value = text_page.get_text_range(index, 1)
        if len(value) != 1:
            continue
        text_object = text_page.get_textobj(index)
        if text_object is None:
            continue
        matrix = text_object.get_matrix()
        left, bottom, right, top = text_page.get_charbox(index)
        font = text_object.get_font()
        evidence = {
            "codepoint": ord(value),
            "character": value,
            "x": float(matrix.e),
            # PDF coordinates start at bottom-left; vision starts at top-left.
            "y": float(page_height - matrix.f),
            "bbox": (float(left), float(page_height - top), float(right), float(page_height - bottom)),
            "font": str(font.get_base_name()),
            "size": float(text_object.get_font_size()),
        }
        if 0xE000 <= ord(value) <= 0xF8FF:
            symbols.append(evidence)
        elif value.isprintable() and not value.isspace():
            text_characters.append({
                **evidence,
                "x": float((left + right) / 2),
                "y": float(page_height - (top + bottom) / 2),
            })
    text = text_page.get_text_range()
    text_page.close()
    vector_curves = []
    for page_object in page.get_objects():
        if page_object.type != pdfium_raw.FPDF_PAGEOBJ_PATH:
            continue
        segment_count = pdfium_raw.FPDFPath_CountSegments(page_object.raw)
        segment_types = [
            pdfium_raw.FPDFPathSegment_GetType(
                pdfium_raw.FPDFPath_GetPathSegment(page_object.raw, index)
            )
            for index in range(segment_count)
        ]
        if pdfium_raw.FPDF_SEGMENT_BEZIERTO not in segment_types:
            continue
        left, bottom, right, top = page_object.get_bounds()
        vector_curves.append({
            "bbox": (
                float(left), float(page_height - top),
                float(right), float(page_height - bottom),
            ),
            "bezierSegments": segment_types.count(pdfium_raw.FPDF_SEGMENT_BEZIERTO),
        })
    return text, symbols, text_characters, vector_curves


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
            text, symbols, text_characters, vector_curves = _pdfium_page_evidence(page)
            rendered.append((
                gray.copy(), text, symbols, text_characters, vector_curves,
                float(page_width), float(page_height),
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
    # Also score literal consecutive windows. The first interval of a staff can
    # be shifted by a clef/stem crossing; estimating all targets from only that
    # interval caused an otherwise clean five-line bass staff to disappear.
    for start_index in range(len(candidates) - 4):
        window = tuple(float(value) for value in candidates[start_index:start_index + 5])
        intervals = np.diff(window)
        mean_spacing = float(np.mean(intervals))
        if not min_spacing <= mean_spacing <= max_spacing:
            continue
        consistency = float(np.std(intervals) / max(1.0, mean_spacing))
        if consistency <= 0.22:
            proposals.append((consistency * 5, window))
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
                # Antialiased thick staff lines can shift their detected centre
                # by several pixels, especially on a sparse final page. Keep a
                # wider but still sub-half-space tolerance so the bass staff is
                # not silently dropped.
                if error > max(2.2, spacing * 0.38):
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
    text_characters: list[dict] | None = None,
    vector_curves: list[dict] | None = None,
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
        list(text_characters or []),
        list(vector_curves or []),
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


def _beam_count(
    binary: np.ndarray,
    x: float,
    y: float,
    stem_x: int,
    spacing: float,
    staff_lines: tuple[float, ...] | None = None,
) -> int:
    # Find long, thin components near the stem. A symmetric window handles
    # both up/down stems and beams extending left or right. Rotated-rectangle
    # thickness distinguishes a slanted beam from a notehead or accent.
    y1, y2 = max(0, int(y - 4.5 * spacing)), min(binary.shape[0], int(y + 4.5 * spacing))
    x1, x2 = max(0, int(x - 4.0 * spacing)), min(binary.shape[1], int(x + 4.0 * spacing))
    roi = binary[y1:y2, x1:x2].copy()
    if not roi.size:
        return 0
    if staff_lines:
        for line_y in staff_lines:
            local_y = int(round(line_y - y1))
            guard = max(1, int(round(spacing * 0.14)))
            roi[max(0, local_y - guard):min(len(roi), local_y + guard + 1), :] = 0
    horizontal = cv2.morphologyEx(
        roi, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, int(spacing * 0.9)), 1)),
    )
    contours, _ = cv2.findContours(horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    beam_centers = []
    for contour in contours:
        bx, by, width, height = cv2.boundingRect(contour)
        left, right = x1 + bx, x1 + bx + width
        center_y = y1 + by + height / 2
        if width < spacing * 1.3 or abs(center_y - y) < spacing * 0.75:
            continue
        if not left - spacing * 0.5 <= stem_x <= right + spacing * 0.5:
            continue
        rectangle = cv2.minAreaRect(contour)
        long_side = max(rectangle[1])
        thickness = min(rectangle[1])
        if long_side < spacing * 1.25 or not spacing * 0.08 <= thickness <= spacing * 0.58:
            continue
        beam_centers.append(center_y)
    distinct = []
    for center in sorted(beam_centers):
        if not distinct or center - distinct[-1] > spacing * 0.4:
            distinct.append(center)
    # Raster-only third/fourth beams are too easily confused with ties and
    # ledger fragments. Semantic flag glyphs still preserve 32nd/64th values;
    # the conservative visual path caps at sixteenths.
    return min(2, len(distinct))


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
        beams = _beam_count(page.binary, cx, cy, stem_x, spacing, staff.lines) if stem else 0
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
            articulation="", stem_direction="up" if stem and stem_x >= cx else ("down" if stem else ""),
        ))
    return sorted(marks, key=lambda item: (item.x, item.midi)), rejected


NOTEHEAD_DURATIONS = {
    0xE0A0: 8.0,  # noteheadDoubleWhole
    0xE0A1: 8.0,  # noteheadDoubleWholeSquare
    0xE0A2: 4.0,  # noteheadWhole
    0xE0A3: 2.0,  # noteheadHalf
    0xE0A4: 1.0,  # noteheadBlack; beams/flags shorten this below
}
REST_DURATIONS = {
    0xE4E0: 32.0,  # restMaxima
    0xE4E1: 16.0,  # restLonga
    0xE4E2: 8.0,   # restDoubleWhole
    0xE4E3: 4.0,   # restWhole
    0xE4E4: 2.0,   # restHalf
    0xE4E5: 1.0,   # restQuarter
    0xE4E6: 0.5,   # rest8th
    0xE4E7: 0.25,  # rest16th
    0xE4E8: 0.125,
    0xE4E9: 0.0625,
}
ACCIDENTAL_ALTERS = {
    0xE260: -1, 0xE261: 0, 0xE262: 1, 0xE263: 2, 0xE264: -2,
}
FLAG_BEAMS = {
    0xE240: 1, 0xE241: 1, 0xE242: 2, 0xE243: 2,
    0xE244: 3, 0xE245: 3, 0xE246: 4, 0xE247: 4,
}
AUGMENTATION_DOT = 0xE1E7
ARTICULATION_CODES = {
    0xE4A0: "accent", 0xE4A1: "accent",
    0xE4A2: "staccato", 0xE4A3: "staccato",
    0xE4A4: "tenuto", 0xE4A5: "tenuto",
    0xE4A6: "staccatissimo", 0xE4A7: "staccatissimo",
    0xE4AC: "marcato", 0xE4AD: "marcato",
}
PEDAL_DOWN_CODES = {0xE650, 0xE651, 0xE656, 0xE65C, 0xE65D}
PEDAL_UP_CODES = {0xE655, 0xE657}
DYNAMIC_GLYPHS = {0xE520: "p", 0xE521: "m", 0xE522: "f"}
DYNAMIC_VELOCITIES = {
    "ppp": 0.34, "pp": 0.42, "p": 0.52,
    "mp": 0.64, "mf": 0.76,
    "f": 0.86, "ff": 0.93, "fff": 0.98,
}


def _scaled_symbols(page: PageAnalysis, accepted_codes: set[int]) -> list[dict]:
    scale_x = page.gray.shape[1] / max(1.0, page.source_width)
    scale_y = page.gray.shape[0] / max(1.0, page.source_height)
    return [
        {**symbol, "x": symbol["x"] * scale_x, "y": symbol["y"] * scale_y}
        for symbol in page.smufl_symbols if symbol["codepoint"] in accepted_codes
    ]


def _scaled_curves(page: PageAnalysis) -> list[dict]:
    scale_x = page.gray.shape[1] / max(1.0, page.source_width)
    scale_y = page.gray.shape[0] / max(1.0, page.source_height)
    curves = []
    for curve in page.vector_curves:
        left, top, right, bottom = curve["bbox"]
        curves.append({
            **curve,
            "bbox": (left * scale_x, top * scale_y, right * scale_x, bottom * scale_y),
        })
    return curves


def _vector_tie_pairs(
    page_analyses: list[PageAnalysis],
    all_marks: list[tuple[NoteMark, list[float], int, float]],
    system_layouts: dict[int, dict],
) -> tuple[list[tuple[int, int]], dict]:
    """Associate vector Bezier ties with equal-pitch noteheads.

    A slur may connect different notes; a tie must connect the same written
    pitch. Requiring equal MIDI plus proximity to both curve ends avoids
    turning ordinary legato phrasing into one giant note.
    """
    marks_by_page_system: dict[tuple[int, int], list[NoteMark]] = {}
    for mark, *_ in all_marks:
        marks_by_page_system.setdefault((mark.page, mark.system), []).append(mark)
    pairs = []
    considered = 0
    for page in page_analyses:
        for system, layout in system_layouts.items():
            if layout["page"] is not page:
                continue
            marks = marks_by_page_system.get((page.staffs[0].page if page.staffs else 0, system), [])
            if not marks:
                continue
            spacing = float(layout["spacing"])
            for curve in _scaled_curves(page):
                left, top, right, bottom = curve["bbox"]
                width, height = right - left, bottom - top
                if not spacing * 0.65 <= width <= spacing * 13:
                    continue
                if not spacing * 0.05 <= height <= spacing * 3.2:
                    continue
                if bottom < layout["top"] - spacing * 5 or top > layout["bottom"] + spacing * 5:
                    continue
                considered += 1
                left_candidates = [
                    mark for mark in marks
                    if abs(mark.x - left) <= spacing * 2.3
                    and top - spacing * 2.6 <= mark.y <= bottom + spacing * 2.6
                ]
                right_candidates = [
                    mark for mark in marks
                    if abs(mark.x - right) <= spacing * 2.3
                    and top - spacing * 2.6 <= mark.y <= bottom + spacing * 2.6
                ]
                options = []
                for first in left_candidates:
                    for second in right_candidates:
                        if first.midi != second.midi or second.x <= first.x + spacing * 0.45:
                            continue
                        score = (
                            abs(first.x - left) + abs(second.x - right)
                            + min(abs(first.y - top), abs(first.y - bottom)) * 0.28
                            + min(abs(second.y - top), abs(second.y - bottom)) * 0.28
                        )
                        options.append((score, first, second))
                if options:
                    _, first, second = min(options, key=lambda item: item[0])
                    pairs.append((id(first), id(second)))
    return sorted(set(pairs)), {
        "vectorCurvesConsidered": considered,
        "tieConnections": len(set(pairs)),
    }


def _merge_visual_ties(notes: list[dict], tie_pairs: list[tuple[int, int]], bpm: float) -> tuple[list[dict], int]:
    by_mark = {note.get("_markId"): note for note in notes}
    parent = {}
    beat_seconds = 60.0 / max(20.0, min(300.0, bpm))
    accepted = 0
    for first_id, second_id in tie_pairs:
        first, second = by_mark.get(first_id), by_mark.get(second_id)
        if not first or not second or first["note"] != second["note"]:
            continue
        gap = float(second["time"]) - float(first["time"])
        if gap < -0.001 or gap > float(first["duration"]) + beat_seconds * 1.1:
            continue
        parent[second_id] = first_id
        accepted += 1

    def root(mark_id: int) -> int:
        seen = set()
        while mark_id in parent and mark_id not in seen:
            seen.add(mark_id)
            mark_id = parent[mark_id]
        return mark_id

    groups: dict[int, list[dict]] = {}
    for note in notes:
        groups.setdefault(root(note.get("_markId")), []).append(note)
    merged = []
    for chain in groups.values():
        chain.sort(key=lambda item: item["time"])
        first = chain[0]
        if len(chain) > 1:
            end = max(float(note["time"]) + float(note["duration"]) for note in chain)
            duration = round(end - float(first["time"]), 6)
            first["duration"] = duration
            first["scoreDuration"] = duration
            first["visualDuration"] = duration
            first["articulation"] = "legato tie"
            first["tiedSegments"] = len(chain)
        merged.append(first)
    return merged, accepted


def _measure_for_x(boundaries: list[float], x: float) -> int:
    return max(0, min(len(boundaries) - 2, int(np.searchsorted(boundaries, x, side="right") - 1)))


def detect_smufl_notes(
    page: PageAnalysis,
    staff: Staff,
    boundaries: list[float],
    instrument: str,
) -> tuple[list[NoteMark], int, int]:
    """Read semantically encoded music glyphs before considering raster blobs."""
    accepted = set(NOTEHEAD_DURATIONS) | set(ACCIDENTAL_ALTERS) | set(FLAG_BEAMS) | set(ARTICULATION_CODES) | {
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
    articulations = [symbol for symbol in relevant if symbol["codepoint"] in ARTICULATION_CODES]

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
        raster_beams = _beam_count(
            page.binary, head["x"], head["y"], stem_x, spacing, staff.lines,
        ) if stem else 0
        nearby_flags = [
            FLAG_BEAMS[flag["codepoint"]] for flag in flags
            # Flag glyphs are anchored to the chord's rhythmic x position.
            # Stem detection can accidentally select the neighbouring chord,
            # so head x is the stable semantic coordinate here.
            if abs(flag["x"] - head["x"]) <= spacing * 1.45
            and abs(flag["y"] - head["y"]) <= spacing * 4.8
        ]
        # A semantic flag is authoritative. Raster geometry remains necessary
        # for beamed groups, but must never override an embedded one-beam flag
        # merely because nearby staff ink looked like a second beam.
        beam_count = max(nearby_flags) if nearby_flags else raster_beams
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
        nearby_articulation = min((
            symbol for symbol in articulations
            if abs(symbol["x"] - head["x"]) <= spacing * 1.25
            and abs(symbol["y"] - head["y"]) <= spacing * 3.2
        ), key=lambda symbol: abs(symbol["x"] - head["x"]) + abs(symbol["y"] - head["y"]) * 0.35, default=None)
        articulation = ARTICULATION_CODES[nearby_articulation["codepoint"]] if nearby_articulation else ""
        rhythm_confidence = 0.98 if head["codepoint"] != 0xE0A4 or beam_count or dotted else 0.82
        confidence = 0.82 * 0.995 + 0.18 * rhythm_confidence
        marks.append(NoteMark(
            page=staff.page, staff_index=staff.index, system=staff.system,
            x=round(head["x"], 3), y=round(head["y"], 3), midi=midi,
            duration_beats=duration, confidence=round(confidence, 4),
            filled=head["codepoint"] == 0xE0A4, stem=stem,
            beam_count=beam_count, dotted=dotted,
            hand="left" if instrument == "piano" and staff.clef == "bass" else "right",
            articulation=articulation,
            stem_direction="up" if stem and stem_x >= head["x"] else ("down" if stem else ""),
            duration_source=(
                "semantic-flag" if nearby_flags else
                "explicit-note-value" if head["codepoint"] != 0xE0A4 else
                "augmentation-dot" if dotted else
                "beam-geometry" if raster_beams else
                "black-note"
            ),
        ))
    return marks, max(0, len(heads) - len(marks)), key_fifths


def detect_smufl_rests(
    page: PageAnalysis,
    staff: Staff,
    boundaries: list[float],
) -> list[RestMark]:
    """Return semantically embedded rests owned by one staff.

    Rests are timing evidence, not merely things to omit from playback. A
    leading eighth rest is the difference between an upbeat and a note struck
    on beat one, so the rhythm solver consumes these marks alongside notes.
    """
    spacing = staff.spacing
    accepted = set(REST_DURATIONS) | {AUGMENTATION_DOT}
    symbols = _scaled_symbols(page, accepted)

    def staff_distance(symbol: dict, candidate: Staff) -> float:
        if candidate.top - candidate.spacing <= symbol["y"] <= candidate.bottom + candidate.spacing:
            return 0.0
        return min(abs(symbol["y"] - candidate.top), abs(symbol["y"] - candidate.bottom)) / candidate.spacing

    rests = []
    dots = [symbol for symbol in symbols if symbol["codepoint"] == AUGMENTATION_DOT]
    for symbol in symbols:
        if symbol["codepoint"] not in REST_DURATIONS:
            continue
        if not (staff.x1 - spacing <= symbol["x"] <= staff.x2 + spacing):
            continue
        if not (staff.top - spacing * 2.2 <= symbol["y"] <= staff.bottom + spacing * 2.2):
            continue
        owner = min(page.staffs, key=lambda candidate: (
            staff_distance(symbol, candidate),
            abs(symbol["y"] - (candidate.top + candidate.bottom) / 2),
        ))
        if owner.index != staff.index:
            continue
        dotted = any(
            0.35 * spacing <= dot["x"] - symbol["x"] <= 1.8 * spacing
            and abs(dot["y"] - symbol["y"]) <= spacing
            for dot in dots
        )
        duration = REST_DURATIONS[symbol["codepoint"]] * (1.5 if dotted else 1.0)
        # Ignore multi-measure/maxima rests here; the bar graph already owns
        # measure count and these need separate multi-rest semantics.
        duration = min(duration, 8.0)
        rests.append(RestMark(
            page=staff.page,
            staff_index=staff.index,
            system=staff.system,
            x=round(symbol["x"], 3),
            y=round(symbol["y"], 3),
            duration_beats=duration,
            dotted=dotted,
        ))
    return sorted(rests, key=lambda item: item.x)


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


def _measure_onset_map(
    all_marks: list[tuple[NoteMark, list[float], int, float]],
    measure_beats: float,
) -> dict[int, float]:
    """Align both piano staffs to shared rhythmic columns inside each measure.

    Engravers offset chord heads slightly so they remain readable. Treating each
    x coordinate independently turns those offsets into accidental machine-gun
    attacks. This clusters the grand staff first and quantizes each shared column
    once, at a resolution supported by the printed note values.
    """

    grouped: dict[tuple[int, int], list[tuple[NoteMark, float, float, float]]] = {}
    for mark, boundaries, _, spacing in all_marks:
        measure_index = max(0, min(
            len(boundaries) - 2,
            int(np.searchsorted(boundaries, mark.x, side="right") - 1),
        ))
        grouped.setdefault((mark.system, measure_index), []).append((
            mark, boundaries[measure_index], boundaries[measure_index + 1], spacing,
        ))

    onset_by_mark: dict[int, float] = {}
    for entries in grouped.values():
        left, right = entries[0][1], entries[0][2]
        spacing = float(np.median([entry[3] for entry in entries]))
        # Engravers offset adjacent seconds horizontally so both noteheads stay
        # visible. In these scores the displacement reaches about 1.2 staff
        # spaces; treating it as a new attack split one chord into a rapid
        # stutter. Dense melodic notes remain roughly two spaces apart.
        tolerance = max(1.5, spacing * 1.30)
        clusters: list[list[tuple[NoteMark, float, float, float]]] = []
        for entry in sorted(entries, key=lambda item: item[0].x):
            if not clusters:
                clusters.append([entry])
                continue
            center = float(np.mean([member[0].x for member in clusters[-1]]))
            if entry[0].x - center <= tolerance:
                clusters[-1].append(entry)
            else:
                clusters.append([entry])

        shortest = min((member[0].duration_beats for cluster in clusters for member in cluster), default=0.25)
        grid = 0.125 if shortest <= 0.125 else 0.25
        latest_slot = max(0.0, measure_beats - grid)
        previous_onset = None
        for cluster_index, cluster in enumerate(clusters):
            center = float(np.mean([member[0].x for member in cluster]))
            raw = min(0.999, max(0.0, (center - left) / max(1.0, right - left))) * measure_beats
            onset = round(raw / grid) * grid
            if cluster_index == 0 and raw <= max(0.50, grid * 2):
                onset = 0.0
            onset = min(latest_slot, max(0.0, onset))
            if previous_onset is not None and onset <= previous_onset:
                onset = min(latest_slot, previous_onset + grid)
            for mark, *_ in cluster:
                onset_by_mark[id(mark)] = round(onset, 6)
            previous_onset = onset
    return onset_by_mark


def _semantic_measure_rhythm(
    all_marks: list[tuple[NoteMark, list[float], int, float]],
    all_rests: list[tuple[RestMark, list[float], int, float]],
    measure_beats: float,
) -> tuple[dict[int, float], dict]:
    """Reconstruct measure time from notes *and* printed rests.

    Horizontal engraving is elastic: a note at 25% of a measure's width is not
    necessarily beat two. The previous reader treated it that way. This solver
    instead forms rhythmic columns, reads fixed rest/flag evidence, and chooses
    a legal 48-tick measure whose spacing best matches the engraving. The same
    calculation is shared by every head in a chord.
    """
    grouped: dict[tuple[int, int, int], dict] = {}
    for mark, boundaries, _, spacing in all_marks:
        measure_index = _measure_for_x(boundaries, mark.x)
        bucket = grouped.setdefault((mark.system, mark.staff_index, measure_index), {
            "notes": [], "rests": [], "left": boundaries[measure_index],
            "right": boundaries[measure_index + 1], "spacing": spacing,
        })
        bucket["notes"].append(mark)
    for rest, boundaries, _, spacing in all_rests:
        measure_index = _measure_for_x(boundaries, rest.x)
        bucket = grouped.setdefault((rest.system, rest.staff_index, measure_index), {
            "notes": [], "rests": [], "left": boundaries[measure_index],
            "right": boundaries[measure_index + 1], "spacing": spacing,
        })
        bucket["rests"].append(rest)

    ticks_per_beat = 48
    measure_ticks = max(1, int(round(measure_beats * ticks_per_beat)))
    legal_beats = (0.0625, 0.125, 1 / 6, 0.25, 1 / 3, 0.375, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)
    legal_ticks = sorted({max(1, int(round(value * ticks_per_beat))) for value in legal_beats})
    onset_by_mark: dict[int, float] = {}
    solved_measures = 0
    exact_measures = 0
    rest_anchors = 0
    corrected_durations = 0

    for bucket in grouped.values():
        spacing = float(bucket["spacing"])
        events = [
            {"x": mark.x, "note": mark, "rest": None}
            for mark in bucket["notes"]
        ] + [
            {"x": rest.x, "note": None, "rest": rest}
            for rest in bucket["rests"]
        ]
        if not events:
            continue
        columns: list[list[dict]] = []
        tolerance = max(1.5, spacing * 1.30)
        for event in sorted(events, key=lambda item: item["x"]):
            if not columns:
                columns.append([event])
                continue
            center = float(np.mean([member["x"] for member in columns[-1]]))
            if event["x"] - center <= tolerance:
                columns[-1].append(event)
            else:
                columns.append([event])

        descriptors = []
        for column in columns:
            notes = [event["note"] for event in column if event["note"] is not None]
            rests = [event["rest"] for event in column if event["rest"] is not None]
            center = float(np.mean([event["x"] for event in column]))
            authoritative = [
                note.duration_beats for note in notes
                if note.duration_source in {"semantic-flag", "explicit-note-value", "augmentation-dot"}
            ]
            rest_values = [rest.duration_beats for rest in rests]
            visual_values = [note.duration_beats for note in notes]
            if rest_values:
                preferred = min(rest_values)
                evidence = "rest"
                rest_anchors += 1
            elif authoritative:
                preferred = min(authoritative)
                evidence = "semantic"
            elif visual_values:
                # Chord heads share rhythm. The median is less vulnerable than
                # the old per-head maximum beam count.
                preferred = float(np.median(visual_values))
                sources = {note.duration_source for note in notes}
                evidence = "black" if sources == {"black-note"} else "beam"
            else:
                preferred = 1.0
                evidence = "unknown"
            descriptors.append({
                "x": center, "notes": notes, "rests": rests,
                "preferred": preferred, "evidence": evidence,
            })

        # One beam joins several stems, but raster evidence may be visible only
        # beside some heads. Propagate that rhythmic evidence across a compact
        # uninterrupted run; a printed rest always breaks the run. This turns
        # the last heads of a sixteenth-note beam into sixteenths instead of
        # isolated quarter notes.
        run: list[dict] = []
        beam_runs: list[list[dict]] = []
        for descriptor in descriptors:
            if descriptor["rests"]:
                if run:
                    beam_runs.append(run)
                    run = []
                continue
            if run and descriptor["x"] - run[-1]["x"] > spacing * 2.45:
                beam_runs.append(run)
                run = []
            run.append(descriptor)
        if run:
            beam_runs.append(run)
        for beam_run in beam_runs:
            beam_values = [item["preferred"] for item in beam_run if item["evidence"] == "beam"]
            if len(beam_run) < 2 or not beam_values:
                continue
            propagated = max(beam_values)
            for item in beam_run:
                if item["evidence"] == "black":
                    item["preferred"] = propagated
                    item["evidence"] = "beam-propagated"

        # Dynamic program over inter-column advances, including the final
        # column-to-barline span. Fixed rests are strong constraints; note
        # durations are priors because another voice may move underneath a
        # sustained note.
        left, right = float(bucket["left"]), float(bucket["right"])
        usable_width = max(spacing * 4, right - left - spacing * 2.0)
        pixels_per_beat = usable_width / max(0.25, measure_beats)
        candidate_rows = []
        strict_candidate_rows = []
        for index, descriptor in enumerate(descriptors):
            preferred_ticks = max(1, int(round(descriptor["preferred"] * ticks_per_beat)))
            if descriptor["evidence"] == "rest":
                candidates = [preferred_ticks]
            else:
                candidates = [tick for tick in legal_ticks if tick <= measure_ticks]
                if preferred_ticks not in candidates:
                    candidates.append(preferred_ticks)
            next_x = descriptors[index + 1]["x"] if index + 1 < len(descriptors) else right - spacing
            geometry_beats = max(0.04, (next_x - descriptor["x"]) / max(1.0, pixels_per_beat))
            row = []
            for ticks in sorted(set(candidates)):
                beats = ticks / ticks_per_beat
                ratio_cost = abs(math.log(max(beats, 1 / 48) / max(descriptor["preferred"], 1 / 48)))
                evidence_weight = {
                    "semantic": 1.8,
                    "black": 4.0,
                    "beam": 1.7,
                    "beam-propagated": 2.0,
                }.get(descriptor["evidence"], 0.42)
                geometry_cost = abs(beats - geometry_beats) * 0.22
                # Triplet values are legal but should be selected only when
                # they materially improve a complete measure.
                triplet_penalty = 0.10 if ticks % 3 != 0 else 0.0
                row.append((ticks, ratio_cost * evidence_weight + geometry_cost + triplet_penalty))
            candidate_rows.append(row)
            strict_candidate_rows.append(
                [choice for choice in row if choice[0] == preferred_ticks]
                if descriptor["evidence"] == "black" else row
            )

        def solve(candidate_matrix: list[list[tuple[int, float]]]) -> dict[int, tuple[float, list[int]]]:
            states: dict[int, tuple[float, list[int]]] = {0: (0.0, [])}
            for candidate_row in candidate_matrix:
                next_states: dict[int, tuple[float, list[int]]] = {}
                for total, (cost, path) in states.items():
                    for ticks, choice_cost in candidate_row:
                        new_total = total + ticks
                        if new_total > measure_ticks:
                            continue
                        candidate = (cost + choice_cost, [*path, ticks])
                        if new_total not in next_states or candidate[0] < next_states[new_total][0]:
                            next_states[new_total] = candidate
                states = next_states
                if not states:
                    break
            return states

        strict_states = solve(strict_candidate_rows)
        states = strict_states if measure_ticks in strict_states else solve(candidate_rows)
        if not states:
            # Defensive fallback: semantic pages should virtually always have
            # a legal path, but geometry-only behaviour is safer than failure.
            for descriptor in descriptors:
                for note in descriptor["notes"]:
                    raw = (descriptor["x"] - left) / max(1.0, right - left) * measure_beats
                    onset_by_mark[id(note)] = round(max(0.0, min(measure_beats, raw)), 6)
            continue
        if measure_ticks in states:
            _, path = states[measure_ticks]
            exact_measures += 1
        else:
            best_total = min(states, key=lambda total: (
                abs(measure_ticks - total) * 2.5 + states[total][0], states[total][0],
            ))
            _, path = states[best_total]
        solved_measures += 1

        cursor = 0
        for descriptor, advance_ticks in zip(descriptors, path):
            onset = cursor / ticks_per_beat
            selected_duration = advance_ticks / ticks_per_beat
            for note in descriptor["notes"]:
                onset_by_mark[id(note)] = round(onset, 6)
                if note.duration_source in {"black-note", "beam-geometry"}:
                    if abs(note.duration_beats - selected_duration) > 1e-6:
                        corrected_durations += 1
                    note.duration_beats = selected_duration
                    note.duration_source = "measure-rhythm-solver"
            cursor += advance_ticks

    return onset_by_mark, {
        "measuresSolved": solved_measures,
        "exactMeasureBalances": exact_measures,
        "restAnchors": rest_anchors,
        "correctedAmbiguousDurations": corrected_durations,
        "ticksPerQuarter": ticks_per_beat,
    }


def _semantic_pedal_events(
    system_layouts: dict[int, dict],
    system_offsets: dict[int, float],
    measure_beats: float,
    bpm: float,
) -> list[dict]:
    candidates = []
    accepted = PEDAL_DOWN_CODES | PEDAL_UP_CODES
    for system, layout in system_layouts.items():
        page = layout["page"]
        spacing = layout["spacing"]
        boundaries = layout["boundaries"]
        symbols = _scaled_symbols(page, accepted)
        for symbol in symbols:
            if not boundaries[0] - spacing <= symbol["x"] <= boundaries[-1] + spacing:
                continue
            # Printed pedal instructions normally sit underneath the lower
            # staff. A generous lower band also covers bracket-style exports.
            if not layout["top"] - spacing <= symbol["y"] <= layout["bottom"] + spacing * 7:
                continue
            measure_index = _measure_for_x(boundaries, symbol["x"])
            left, right = boundaries[measure_index], boundaries[measure_index + 1]
            normalized = min(0.999, max(0.0, (symbol["x"] - left) / max(1.0, right - left)))
            onset = round(normalized * measure_beats * 4) / 4
            beat = system_offsets.get(system, 0.0) + measure_index * measure_beats + onset
            candidates.append({
                "beat": beat,
                "down": symbol["codepoint"] in PEDAL_DOWN_CODES,
            })
    seconds_per_beat = 60.0 / bpm
    candidates.sort(key=lambda item: (item["beat"], item["down"]))
    events = []
    state = None
    for index, candidate in enumerate(candidates):
        down = candidate["down"]
        same_beat_has_up = down and any(
            abs(other["beat"] - candidate["beat"]) < 0.0001 and not other["down"]
            for other in candidates
        )
        event_time = candidate["beat"] * seconds_per_beat + (0.045 if same_beat_has_up else 0.0)
        if down == state and not same_beat_has_up:
            continue
        events.append({
            "id": f"smufl-pedal-{index}",
            "time": round(event_time, 6),
            "down": down,
            "value": 127 if down else 0,
            "controller": 64,
            "source": "printed-smufl-pedal",
            "inferred": False,
        })
        state = down
    return events


def _semantic_dynamic_events(
    system_layouts: dict[int, dict],
    system_offsets: dict[int, float],
    measure_beats: float,
) -> list[dict]:
    events = []
    for system, layout in system_layouts.items():
        page = layout["page"]
        spacing = layout["spacing"]
        boundaries = layout["boundaries"]
        symbols = [
            symbol for symbol in _scaled_symbols(page, set(DYNAMIC_GLYPHS))
            if boundaries[0] - spacing <= symbol["x"] <= boundaries[-1] + spacing
            and layout["top"] - spacing * 2 <= symbol["y"] <= layout["bottom"] + spacing * 6
        ]
        clusters: list[list[dict]] = []
        for symbol in sorted(symbols, key=lambda item: item["x"]):
            if not clusters or symbol["x"] - clusters[-1][-1]["x"] > spacing * 1.45:
                clusters.append([symbol])
            else:
                clusters[-1].append(symbol)
        for cluster in clusters:
            mark = "".join(DYNAMIC_GLYPHS[symbol["codepoint"]] for symbol in cluster)
            if mark not in DYNAMIC_VELOCITIES:
                continue
            x = float(np.mean([symbol["x"] for symbol in cluster]))
            measure_index = _measure_for_x(boundaries, x)
            left, right = boundaries[measure_index], boundaries[measure_index + 1]
            normalized = min(0.999, max(0.0, (x - left) / max(1.0, right - left)))
            onset = round(normalized * measure_beats * 4) / 4
            events.append({
                "beat": system_offsets.get(system, 0.0) + measure_index * measure_beats + onset,
                "mark": mark,
                "velocity": DYNAMIC_VELOCITIES[mark],
            })
    return sorted(events, key=lambda item: item["beat"])


def _semantic_navigation(system_layouts: dict[int, dict]) -> dict:
    """Locate Segno, Coda, D.S., and first/second endings by page position."""
    layouts = list(system_layouts.values())

    def scaled_characters(page: PageAnalysis) -> list[dict]:
        scale_x = page.gray.shape[1] / max(1.0, page.source_width)
        scale_y = page.gray.shape[0] / max(1.0, page.source_height)
        return [
            {**item, "x": item["x"] * scale_x, "y": item["y"] * scale_y}
            for item in page.text_characters
        ]

    def nearest_layout(page: PageAnalysis, y: float) -> dict | None:
        candidates = [layout for layout in layouts if layout["page"] is page]
        if not candidates:
            return None
        return min(candidates, key=lambda layout: (
            0.0 if layout["top"] - layout["spacing"] * 4 <= y <= layout["bottom"] + layout["spacing"] * 3
            else min(abs(y - layout["top"]), abs(y - layout["bottom"])),
        ))

    def marker_measure(layout: dict, x: float) -> int:
        boundaries = layout["boundaries"]
        nearest = min(range(len(boundaries)), key=lambda index: abs(boundaries[index] - x))
        if abs(boundaries[nearest] - x) <= layout["spacing"] * 1.45:
            local = min(nearest, len(boundaries) - 2)
        else:
            local = _measure_for_x(boundaries, x)
        return int(layout["measureOffset"] + local)

    directions = []
    for layout in layouts:
        page = layout["page"]
        for symbol in _scaled_symbols(page, {0xE047, 0xE048}):
            owner = nearest_layout(page, symbol["y"])
            if owner is not layout:
                continue
            directions.append({
                "kind": "segno" if symbol["codepoint"] == 0xE047 else "coda",
                "measure": marker_measure(layout, symbol["x"]),
                "page": page.staffs[0].page if page.staffs else 0,
                "y": symbol["y"], "x": symbol["x"],
                "marginTarget": symbol["x"] < layout["boundaries"][0] - layout["spacing"] * 1.5,
            })

    ds_triggers = []
    volta_starts = {"1": [], "2": []}
    seen_pages = []
    seen_page_ids = set()
    for layout in layouts:
        page = layout["page"]
        if id(page) not in seen_page_ids:
            seen_pages.append(page)
            seen_page_ids.add(id(page))
    for page in seen_pages:
        characters = scaled_characters(page)
        for character in characters:
            if character["character"] == "D":
                matching_s = [
                    candidate for candidate in characters
                    if candidate["character"] == "S"
                    and abs(candidate["y"] - character["y"]) <= max(8.0, character["size"] * 0.8)
                    and 0 < candidate["x"] - character["x"] <= max(140.0, character["size"] * 2.8)
                ]
                if matching_s:
                    layout = nearest_layout(page, character["y"])
                    if layout:
                        local = _measure_for_x(layout["boundaries"], character["x"])
                        ds_triggers.append(int(layout["measureOffset"] + local + 1))
            if character["character"] not in volta_starts:
                continue
            if "Bold" not in character["font"]:
                continue
            layout = nearest_layout(page, character["y"])
            if not layout:
                continue
            if character["x"] < layout["boundaries"][0] + layout["spacing"] * 2:
                continue
            if abs(character["y"] - layout["top"]) > layout["spacing"] * 3.2:
                continue
            volta_starts[character["character"]].append(marker_measure(layout, character["x"]))

    segnos = sorted(item["measure"] for item in directions if item["kind"] == "segno")
    coda_entries = list(sorted(
        (entry for entry in directions if entry["kind"] == "coda"),
        key=lambda entry: (entry["page"], entry["y"], entry["x"]),
    ))
    coda_targets = [entry["measure"] for entry in coda_entries if entry["marginTarget"]]
    coda_target = coda_targets[0] if coda_targets else None
    to_coda_candidates = [
        entry["measure"] for entry in coda_entries
        if not entry["marginTarget"]
        and (coda_target is None or entry["measure"] < coda_target)
    ]
    return {
        "segnoMeasure": segnos[0] if segnos else None,
        "toCodaMeasure": to_coda_candidates[0] if to_coda_candidates else None,
        "codaMeasure": coda_target,
        "dsTriggerExclusive": ds_triggers[-1] if ds_triggers else None,
        "firstEndingMeasures": sorted(set(volta_starts["1"])),
        "secondEndingMeasures": sorted(set(volta_starts["2"])),
    }


def _playback_measure_order(
    measure_count: int,
    repeat_pairs: list[tuple[int, int]],
    navigation: dict,
) -> tuple[list[int], dict]:
    """Build the performed measure order without duplicating first endings."""
    ds_trigger = navigation.get("dsTriggerExclusive")
    normal_limit = min(measure_count, ds_trigger) if isinstance(ds_trigger, int) else measure_count
    first_endings = navigation.get("firstEndingMeasures") or []
    second_endings = navigation.get("secondEndingMeasures") or []
    order = []
    cursor = 0
    expanded_repeats = 0
    for start, end in sorted(repeat_pairs):
        start, end = max(cursor, start), min(normal_limit, end)
        if not start < end:
            continue
        order.extend(range(cursor, end))
        first = next((value for value in first_endings if start < value < end), None)
        second = next((value for value in second_endings if value >= end - 1), None)
        repeat_stop = first if first is not None else end
        order.extend(range(start, repeat_stop))
        cursor = second if first is not None and second is not None else end
        expanded_repeats += 1
    order.extend(range(cursor, normal_limit))

    segno = navigation.get("segnoMeasure")
    to_coda = navigation.get("toCodaMeasure")
    coda = navigation.get("codaMeasure")
    ds_expanded = False
    def ds_range(start: int, stop: int) -> list[int]:
        segment = []
        measure = start
        while measure < stop:
            first = next((value for value in first_endings if value == measure), None)
            second = next((value for value in second_endings if value > measure), None)
            if first is not None and second is not None:
                measure = second
                continue
            segment.append(measure)
            measure += 1
        return segment

    if isinstance(ds_trigger, int) and isinstance(segno, int):
        if isinstance(to_coda, int) and isinstance(coda, int) and 0 <= segno < to_coda < coda < measure_count:
            order.extend(ds_range(segno, to_coda))
            order.extend(range(coda, measure_count))
            ds_expanded = True
        elif isinstance(coda, int) and 0 <= segno < measure_count:
            # Some engravers place only the target Coda in the margin and put
            # the other Coda glyph beside the D.S. instruction. Continue the
            # D.S. pass while honouring second endings; the target then occurs
            # naturally in score order.
            order.extend(ds_range(segno, measure_count))
            ds_expanded = True
    elif normal_limit < measure_count:
        order.extend(range(normal_limit, measure_count))
    return order, {
        "repeatSectionsExpanded": expanded_repeats,
        "repeatPairs": [{"startMeasure": start + 1, "endBeforeMeasure": end + 1} for start, end in repeat_pairs],
        "dsAlCodaExpanded": ds_expanded,
        "sourceMeasures": measure_count,
        "playbackMeasures": len(order),
    }


def reconstruct(page_analyses: list[PageAnalysis], filename: str, instrument: str) -> dict:
    title, composer, bpm, signature, warnings = _metadata([page.text for page in page_analyses], filename)
    all_marks: list[tuple[NoteMark, list[float], int, float]] = []
    all_rests: list[tuple[RestMark, list[float], int, float]] = []
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
    system_layouts: dict[int, dict] = {}

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
                # The two strokes of a repeat bar are not two measures.
                if boundary - deduplicated[-1] >= spacing * 4:
                    deduplicated.append(boundary)
            if len(deduplicated) < 2:
                deduplicated = [common_x1, common_x2]
            system_layouts[system_counter] = {
                "page": page,
                "boundaries": deduplicated,
                "spacing": spacing,
                "top": min(staff.top for staff in staffs),
                "bottom": max(staff.bottom for staff in staffs),
            }
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
                    if abs(boundary_x - repeat_x) <= spacing * 3.5:
                        repeat_markers.append((
                            system_counter, boundary_index, "start" if repeat_x > boundary_x else "end",
                        ))
            system_measure_counts[(page.staffs[0].page, system_counter)] = len(deduplicated) - 1
            for staff in staffs:
                if semantic_available:
                    marks, staff_rejected, staff_key_fifths = detect_smufl_notes(
                        page, staff, deduplicated, instrument,
                    )
                    rests = detect_smufl_rests(page, staff, deduplicated)
                    detected_key_fifths.append(staff_key_fifths)
                else:
                    marks, staff_rejected = detect_notes(page, staff, instrument)
                    rests = []
                rejected += staff_rejected
                staff_diagnostics.append({
                    **asdict(staff), "detectedNotes": len(marks), "rejectedMarks": staff_rejected,
                })
                for mark in marks:
                    mark.system = system_counter
                    all_marks.append((mark, deduplicated, len(deduplicated) - 1, spacing))
                for rest in rests:
                    rest.system = system_counter
                    all_rests.append((rest, deduplicated, len(deduplicated) - 1, spacing))
            system_counter += 1

    measure_beats = signature.measure_quarter_beats
    system_offsets = {}
    running_beats = 0.0
    for system in range(system_counter):
        counts = [count for (_, sys), count in system_measure_counts.items() if sys == system]
        count = counts[0] if counts else 1
        system_offsets[system] = running_beats
        if system in system_layouts:
            system_layouts[system]["measureOffset"] = int(round(running_beats / measure_beats))
        running_beats += count * measure_beats

    printed_pedals = _semantic_pedal_events(
        system_layouts, system_offsets, measure_beats, bpm,
    ) if semantic_available and instrument == "piano" else []
    dynamic_events = _semantic_dynamic_events(
        system_layouts, system_offsets, measure_beats,
    ) if semantic_available else []
    tie_pairs, tie_diagnostics = _vector_tie_pairs(
        page_analyses, all_marks, system_layouts,
    ) if semantic_available else ([], {"vectorCurvesConsidered": 0, "tieConnections": 0})

    if semantic_available:
        onset_by_mark, rhythm_diagnostics = _semantic_measure_rhythm(
            all_marks, all_rests, measure_beats,
        )
    else:
        onset_by_mark = _measure_onset_map(all_marks, measure_beats)
        rhythm_diagnostics = {
            "measuresSolved": 0, "exactMeasureBalances": 0,
            "restAnchors": 0, "correctedAmbiguousDurations": 0,
        }
    notes = []
    for mark, boundaries, _, _ in all_marks:
        measure_index = max(0, min(len(boundaries) - 2, int(np.searchsorted(boundaries, mark.x, side="right") - 1)))
        onset_inside = onset_by_mark.get(id(mark), 0.0)
        beat = system_offsets.get(mark.system, 0.0) + measure_index * measure_beats + onset_inside
        duration_beats = min(mark.duration_beats, max(0.0625, measure_beats - onset_inside))
        seconds_per_beat = 60.0 / bpm
        score_duration = max(0.03, duration_beats * seconds_per_beat)
        current_dynamic = next((
            event for event in reversed(dynamic_events) if event["beat"] <= beat + 0.0001
        ), None)
        notes.append({
            "note": midi_to_name(mark.midi),
            "midi": mark.midi,
            "time": round(beat * seconds_per_beat, 6),
            "duration": round(score_duration, 6),
            "scoreDuration": round(score_duration, 6),
            "visualDuration": round(score_duration, 6),
            "velocity": round(current_dynamic["velocity"] if current_dynamic else 0.62 + mark.confidence * 0.24, 4),
            "dynamic": current_dynamic["mark"] if current_dynamic else "",
            "hand": mark.hand,
            "voice": f"{mark.hand} {mark.stem_direction or 'single'} voice",
            "measure": int(beat // measure_beats) + 1,
            "measureBeat": round(beat % measure_beats, 6),
            "articulation": mark.articulation,
            "_confidence": mark.confidence,
            "_markId": id(mark),
        })

    notes, merged_ties = _merge_visual_ties(notes, tie_pairs, bpm)

    repeat_events = sorted((
        int(round(system_offsets.get(system, 0.0) / measure_beats)) + boundary_index,
        marker_type,
    ) for system, boundary_index, marker_type in repeat_markers)
    repeat_pairs = []
    active_repeat_start = 0
    for repeat_measure, marker_type in repeat_events:
        if marker_type == "start":
            active_repeat_start = repeat_measure
        elif repeat_measure > active_repeat_start:
            repeat_pairs.append((active_repeat_start, repeat_measure))
            active_repeat_start = repeat_measure

    navigation = _semantic_navigation(system_layouts) if semantic_available else {}
    measure_count = int(round(running_beats / measure_beats))
    playback_order, playback_diagnostics = _playback_measure_order(
        measure_count, repeat_pairs, navigation,
    )
    if playback_order:
        seconds_per_measure = measure_beats * 60.0 / bpm
        notes_by_measure: dict[int, list[dict]] = {}
        for note in notes:
            source_measure = max(0, min(measure_count - 1, int(note["measure"]) - 1))
            notes_by_measure.setdefault(source_measure, []).append(note)
        expanded = []
        for playback_measure, source_measure in enumerate(playback_order):
            source_start = source_measure * seconds_per_measure
            playback_start = playback_measure * seconds_per_measure
            for note in notes_by_measure.get(source_measure, []):
                expanded.append({
                    **note,
                    "time": round(playback_start + max(0.0, note["time"] - source_start), 6),
                    "measure": playback_measure + 1,
                    "sourceMeasure": source_measure + 1,
                })
        notes = expanded

    # Merge duplicate detections created where staff lines cross a filled head,
    # but preserve real chords and repeated notes at separate quantized onsets.
    unique = {}
    for note in sorted(notes, key=lambda item: (item["time"], item["note"], -item["_confidence"])):
        key = (note["time"], note["note"])
        if key not in unique:
            unique[key] = note
    notes = list(unique.values())
    confidences = [note.pop("_confidence") for note in notes]
    for note in notes:
        note.pop("_markId", None)
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
            "The PDF's semantic noteheads, rests, clefs, flags, dots, and standard accidentals were read directly.",
            "Rhythm was balanced measure-by-measure on a 48-tick grid; uncommon ornaments and ambiguous ties still require review when no embedded MusicXML is present.",
        ])
    else:
        warnings.extend([
            "Computer-vision mode currently assumes the printed clef is treble, or treble/bass for paired piano staffs.",
            "Key-signature accidentals, inline accidentals, tuplets, ornaments, repeat jumps, ties, lyrics, and dynamics require review unless embedded MusicXML was available.",
            "Every accepted note includes geometric evidence; uncertain marks were omitted rather than invented.",
        ])
    if detected_key_fifths:
        key_fifths = max(set(detected_key_fifths), key=detected_key_fifths.count)
    else:
        key_fifths = 0
    engine = "polymath-semantic-score-v3" if semantic_available else "polymath-classical-vision-v2"
    if repeat_pairs:
        warnings.append(f"Expanded {len(repeat_pairs)} standard repeat section(s) into playback order.")
    if playback_diagnostics.get("dsAlCodaExpanded"):
        warnings.append("Expanded the printed D.S. al Coda navigation into performed playback order.")
    result = {
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
        "pedals": printed_pedals,
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
            "semanticRests": len(all_rests),
            "rhythmSolver": rhythm_diagnostics,
            "tieRecognition": {**tie_diagnostics, "mergedTieConnections": merged_ties},
            "repeatSectionsExpanded": playback_diagnostics.get("repeatSectionsExpanded", 0),
            "navigation": navigation,
            "playbackGraph": playback_diagnostics,
            "printedPedalEvents": len(printed_pedals),
            "dynamicChanges": len(dynamic_events),
            "confidenceComponents": {
                "noteheadDetection": 0.995 if semantic_available else round(coverage, 4),
                "pitchMapping": 0.98 if semantic_available else round(float(np.mean(confidences)), 4),
                "rhythmAndPlaybackStructure": 0.82 if semantic_available else 0.55,
            },
        },
    }
    result = shape_piano_performance(result, infer_pedal=instrument == "piano")
    performance = result.get("pianoPerformance", {})
    result["omrDiagnostics"]["pianoPerformance"] = performance
    if performance.get("pedalSource") == "inferred-score-pedaling":
        result["warnings"].append(
            "No printed damper-pedal instructions were encoded; Polymath added conservative, clearly labelled inferred re-pedaling."
        )
    return result
