"""Watchtower: independent verification of automated license plate reader results.

Prototype for research and education. Decision support only — never an
identification or an enforcement decision.
"""

from .compare import Comparison, DiffSegment, compare_readings
from .imaging import ImageError, QualityReport, assess_quality, load_image
from .ocr import OcrDetection, PlateReader, PlateReading, select_plate_text
from .plate_text import are_confusable, normalize_plate
from .verify import (
    InvalidReadingError,
    Outcome,
    VerificationResult,
    validate_alpr_reading,
    verify,
)

__all__ = [
    "Comparison", "DiffSegment", "compare_readings",
    "ImageError", "QualityReport", "assess_quality", "load_image",
    "OcrDetection", "PlateReader", "PlateReading", "select_plate_text",
    "are_confusable", "normalize_plate",
    "InvalidReadingError", "Outcome", "VerificationResult",
    "validate_alpr_reading", "verify",
]
