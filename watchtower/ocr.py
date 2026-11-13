"""Independent plate recognition with EasyOCR.

The EasyOCR model is wrapped behind a tiny ``backend`` callable so that:
  * the "which text on the image is the plate?" logic (``select_plate_text``) is
    pure and unit-testable without downloading a model, and
  * the app can swap OCR engines later without touching the comparison logic.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Callable, Sequence

import cv2
import numpy as np

from .imaging import enhance_for_ocr
from .plate_text import is_plausible_plate, normalize_plate

ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -"

# Words printed on plates/frames that are never part of the registration number.
NON_PLATE_WORDS = frozenset(
    {
        "MICHIGAN", "PURE", "PUREMICHIGAN", "GREAT", "LAKES", "GREATLAKES",
        "STATE", "LAKESTATE", "GREATLAKESSTATE", "USA", "PASSENGER", "TRUCK",
        "TRAILER", "DEALER", "TEMPORARY", "COMMERCIAL", "FICTIONAL", "SAMPLE",
        "DEMO", "VOID", "NOTAREALPLATE",
    }
)

# Below this confidence on the first pass, try again on an enhanced image.
RETRY_CONFIDENCE = 0.6


@dataclass(frozen=True)
class OcrDetection:
    """A single text region as returned by an OCR engine."""

    text: str
    confidence: float
    box: tuple[int, int, int, int]  # x0, y0, x1, y1 in image pixels

    @property
    def height(self) -> int:
        return self.box[3] - self.box[1]

    @property
    def center_y(self) -> float:
        return (self.box[1] + self.box[3]) / 2


@dataclass(frozen=True)
class PlateReading:
    text: str  # normalized (A-Z0-9 only)
    raw_text: str
    confidence: float
    box: tuple[int, int, int, int]
    source: str = "original"  # "original" or "enhanced" image pass
    detections: tuple[OcrDetection, ...] = field(default_factory=tuple)


OcrBackend = Callable[[np.ndarray], list[OcrDetection]]


def _group_into_lines(detections: Sequence[OcrDetection]) -> list[list[OcrDetection]]:
    """Group detections that sit on the same text line with similar heights.

    EasyOCR often splits "ABC 1234" into two boxes; this joins them back.
    """
    lines: list[list[OcrDetection]] = []
    for det in sorted(detections, key=lambda d: d.center_y):
        for line in lines:
            ref_h = statistics.median(d.height for d in line)
            ref_y = statistics.mean(d.center_y for d in line)
            same_row = abs(det.center_y - ref_y) <= 0.5 * max(ref_h, 1)
            similar_size = 0.6 <= det.height / max(ref_h, 1) <= 1.6
            if same_row and similar_size:
                line.append(det)
                break
        else:
            lines.append([det])
    return [sorted(line, key=lambda d: d.box[0]) for line in lines]


def select_plate_text(detections: Sequence[OcrDetection]) -> PlateReading | None:
    """Pick the most plate-like text line from raw OCR detections.

    Heuristic: discard known non-plate words, merge detections on the same line,
    keep lines with a plausible plate length, and prefer the line with the
    tallest characters (registration characters dominate a plate), breaking ties
    by confidence. Returns ``None`` when nothing plausible was found.
    """
    usable = [
        d for d in detections
        if normalize_plate(d.text) and normalize_plate(d.text) not in NON_PLATE_WORDS
    ]
    best: tuple[float, float, PlateReading] | None = None
    for line in _group_into_lines(usable):
        text = "".join(normalize_plate(d.text) for d in line)
        if not is_plausible_plate(text):
            continue
        weights = [len(normalize_plate(d.text)) for d in line]
        confidence = sum(d.confidence * w for d, w in zip(line, weights)) / sum(weights)
        box = (
            min(d.box[0] for d in line), min(d.box[1] for d in line),
            max(d.box[2] for d in line), max(d.box[3] for d in line),
        )
        reading = PlateReading(
            text=text,
            raw_text=" ".join(d.text.strip() for d in line),
            confidence=round(float(confidence), 3),
            box=box,
            detections=tuple(detections),
        )
        height = statistics.median(d.height for d in line)
        key = (height, confidence)
        if best is None or key > best[:2]:
            best = (height, confidence, reading)
    return best[2] if best else None


def _scale_detections(dets: list[OcrDetection], factor: float) -> list[OcrDetection]:
    if factor == 1.0:
        return dets
    return [
        OcrDetection(d.text, d.confidence, tuple(round(v / factor) for v in d.box))  # type: ignore[arg-type]
        for d in dets
    ]


class PlateReader:
    """Reads plate text from an RGB image using an OCR backend (EasyOCR by default)."""

    def __init__(self, backend: OcrBackend | None = None, model_dir: str | None = None):
        self._backend = backend
        self._model_dir = model_dir

    def _get_backend(self) -> OcrBackend:
        if self._backend is None:
            self._backend = make_easyocr_backend(self._model_dir)
        return self._backend

    def warm_up(self) -> None:
        """Load the OCR model now (it is otherwise loaded lazily on first read)."""
        self._get_backend()

    def read(self, rgb: np.ndarray) -> PlateReading | None:
        backend = self._get_backend()
        first = select_plate_text(backend(rgb))
        if first is not None and first.confidence >= RETRY_CONFIDENCE:
            return first

        # Second pass on a contrast-enhanced, upscaled image.
        enhanced = enhance_for_ocr(rgb)
        factor = enhanced.shape[0] / rgb.shape[0]
        second_raw = _scale_detections(backend(enhanced), factor)
        second = select_plate_text(second_raw)
        if second is not None:
            second = PlateReading(**{**second.__dict__, "source": "enhanced"})

        candidates = [r for r in (first, second) if r is not None]
        if not candidates:
            return None
        return max(candidates, key=lambda r: r.confidence)


def make_easyocr_backend(model_dir: str | None = None) -> OcrBackend:
    """Create an EasyOCR-based backend (CPU). Downloads models on first use."""
    try:
        import easyocr  # imported lazily: heavy, and not needed by unit tests
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "EasyOCR is not installed. Run: pip install -r requirements.txt"
        ) from exc

    kwargs = {"gpu": False, "verbose": False}
    if model_dir:
        kwargs["model_storage_directory"] = model_dir
    reader = easyocr.Reader(["en"], **kwargs)

    def backend(image: np.ndarray) -> list[OcrDetection]:
        # EasyOCR expects OpenCV channel order (BGR) for colour input.
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        results = reader.readtext(image, allowlist=ALLOWLIST, paragraph=False)
        detections = []
        for points, text, conf in results:
            xs = [int(p[0]) for p in points]
            ys = [int(p[1]) for p in points]
            detections.append(
                OcrDetection(str(text), float(conf), (min(xs), min(ys), max(xs), max(ys)))
            )
        return detections

    return backend
