"""Plate selection and the verification pipeline, using a fake OCR backend.

These tests never load EasyOCR, so they run in well under a second.
"""

import numpy as np
import pytest

from watchtower.ocr import OcrDetection, PlateReader, select_plate_text
from watchtower.verify import InvalidReadingError, Outcome, validate_alpr_reading, verify

IMAGE = np.full((300, 600, 3), 200, np.uint8)


def det(text, conf=0.95, box=(100, 100, 500, 190)):
    return OcrDetection(text, conf, box)


class FakeBackend:
    """Returns canned detections; records how many times it was called."""

    def __init__(self, *passes):
        self.passes = list(passes)
        self.calls = 0

    def __call__(self, image):
        result = self.passes[min(self.calls, len(self.passes) - 1)]
        self.calls += 1
        return result


# --- select_plate_text -------------------------------------------------------

def test_selects_large_registration_text_and_ignores_plate_furniture():
    detections = [
        det("MICHIGAN", 0.9, (180, 20, 420, 60)),
        det("BXR", 0.99, (60, 100, 250, 190)),
        det("4827", 0.97, (290, 100, 540, 190)),  # split box on the same line
        det("SAMPLE", 1.0, (100, 240, 200, 265)),
        det("NOT A REAL PLATE", 1.0, (210, 240, 500, 265)),
    ]
    reading = select_plate_text(detections)
    assert reading.text == "BXR4827"
    assert reading.box == (60, 100, 540, 190)
    assert 0.97 < reading.confidence < 0.99


def test_prefers_tallest_plausible_line():
    detections = [det("JAN 27", 0.99, (10, 10, 80, 30)), det("ELM 2045", 0.8, (100, 100, 500, 180))]
    assert select_plate_text(detections).text == "ELM2045"


def test_returns_none_when_nothing_plausible():
    assert select_plate_text([]) is None
    assert select_plate_text([det("MICHIGAN"), det("-")]) is None
    assert select_plate_text([det("THISISWAYTOOLONG")]) is None


# --- PlateReader -------------------------------------------------------------

def test_reader_skips_second_pass_when_confident():
    backend = FakeBackend([det("BXR 4827", 0.99)])
    reading = PlateReader(backend).read(IMAGE)
    assert reading.text == "BXR4827" and reading.source == "original"
    assert backend.calls == 1


def test_reader_uses_enhanced_pass_when_it_is_better():
    # Enhanced image is upscaled 400/300; boxes must be mapped back to original pixels.
    backend = FakeBackend([det("BXR 4B27", 0.3)], [det("BXR 4827", 0.9, (133, 133, 667, 253))])
    reading = PlateReader(backend).read(IMAGE)
    assert backend.calls == 2
    assert reading.text == "BXR4827" and reading.source == "enhanced"
    assert reading.box == (100, 100, 500, 190)


def test_reader_returns_none_when_both_passes_fail():
    assert PlateReader(FakeBackend([])).read(IMAGE) is None


# --- verify ------------------------------------------------------------------

def run(alpr, *passes, **kw):
    return verify(IMAGE, alpr, PlateReader(FakeBackend(*passes)), **kw)


def test_agree_still_requires_human_review():
    result = run("bxr-4827", [det("BXR 4827", 0.95)])
    assert result.outcome is Outcome.AGREE
    assert "human verification still required" in result.outcome.title
    record = result.to_record()
    assert record["human_review_required"] is True
    assert record["differences"] == []


def test_disagree_reports_differences():
    result = run("DKT B150", [det("DKT 8150", 0.9)])
    assert result.outcome is Outcome.DISAGREE
    assert result.comparison.all_differences_confusable
    assert result.to_record()["differences"] == [
        {"type": "replace", "ai": "8", "alpr": "B", "confusable_pair": True}
    ]
    assert any("Commonly confused" in r for r in result.reasons)


def test_unreadable_image_is_inconclusive():
    result = run("CPL 6604", [])
    assert result.outcome is Outcome.INCONCLUSIVE
    assert result.comparison is None
    assert any("No plate-like text" in r for r in result.reasons)


def test_low_confidence_is_inconclusive_even_if_text_matches():
    result = run("CPL 6604", [det("CPL 6604", 0.25)], min_confidence=0.4)
    assert result.outcome is Outcome.INCONCLUSIVE
    assert result.ai_reading.text == "CPL6604"  # shown to reviewer, but not trusted
    assert any("below the 40% threshold" in r for r in result.reasons)


@pytest.mark.parametrize("bad", ["", "   ", "--", "A", "ABCDEFGHIJ"])
def test_invalid_alpr_reading_rejected_before_ocr(bad):
    backend = FakeBackend([det("BXR 4827")])
    with pytest.raises(InvalidReadingError):
        verify(IMAGE, bad, PlateReader(backend))
    assert backend.calls == 0


def test_validate_alpr_reading_normalizes():
    assert validate_alpr_reading(" elm 2045 ") == "ELM2045"
