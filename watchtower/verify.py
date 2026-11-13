"""Verification pipeline: image + simulated ALPR reading → one of three outcomes.

Watchtower is decision *support*. No outcome is an identification or an
enforcement decision — every outcome routes to a human reviewer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .compare import Comparison, compare_readings
from .imaging import QualityReport, assess_quality
from .ocr import PlateReader, PlateReading
from .plate_text import MAX_PLATE_LEN, MIN_PLATE_LEN, normalize_plate

DEFAULT_MIN_CONFIDENCE = 0.40


class Outcome(str, Enum):
    AGREE = "agree"
    DISAGREE = "disagree"
    INCONCLUSIVE = "inconclusive"

    @property
    def title(self) -> str:
        return {
            Outcome.AGREE: "Readings agree — human verification still required",
            Outcome.DISAGREE: "Readings disagree — discrepancy detected",
            Outcome.INCONCLUSIVE: "Image unreadable — inconclusive, manual review required",
        }[self]

    @property
    def guidance(self) -> str:
        return {
            Outcome.AGREE: (
                "Two automated systems produced the same characters. That is NOT "
                "proof of identity: both can make the same mistake, and a correct "
                "plate read says nothing about who is driving. A person must compare "
                "the image with the reading and confirm the vehicle details."
            ),
            Outcome.DISAGREE: (
                "The independent reading does not match the ALPR reading. Treat the "
                "ALPR alert as unconfirmed. A person must examine the image before "
                "any action is taken."
            ),
            Outcome.INCONCLUSIVE: (
                "Watchtower could not independently read this plate with enough "
                "confidence. The ALPR reading cannot be corroborated by this tool; "
                "a person must review the image manually."
            ),
        }[self]


class InvalidReadingError(ValueError):
    """The simulated ALPR reading is not a usable plate string."""


def validate_alpr_reading(text: str | None) -> str:
    """Normalize and validate the ALPR reading entered by the user."""
    normalized = normalize_plate(text)
    if not normalized:
        raise InvalidReadingError("Enter the ALPR plate reading (letters and digits).")
    if not MIN_PLATE_LEN <= len(normalized) <= MAX_PLATE_LEN:
        raise InvalidReadingError(
            f"A plate reading must have {MIN_PLATE_LEN}–{MAX_PLATE_LEN} letters/digits "
            f"(got {len(normalized)})."
        )
    return normalized


@dataclass(frozen=True)
class VerificationResult:
    outcome: Outcome
    alpr_input: str
    alpr_normalized: str
    ai_reading: PlateReading | None
    comparison: Comparison | None
    quality: QualityReport
    min_confidence: float
    reasons: tuple[str, ...] = field(default_factory=tuple)

    def to_record(self) -> dict:
        """JSON-serializable review record (contains no image and no personal data)."""
        cmp = self.comparison
        ai = self.ai_reading
        return {
            "tool": "Watchtower (prototype, decision support only)",
            "outcome": self.outcome.value,
            "outcome_title": self.outcome.title,
            "human_review_required": True,
            "alpr_reading": self.alpr_normalized,
            "independent_ai_reading": ai.text if ai else None,
            "ai_confidence": ai.confidence if ai else None,
            "ai_image_pass": ai.source if ai else None,
            "confidence_threshold": self.min_confidence,
            "similarity": cmp.similarity if cmp else None,
            "edit_distance": cmp.edit_distance if cmp else None,
            "differences": [
                {"type": s.op, "ai": s.ai, "alpr": s.alpr, "confusable_pair": s.confusable}
                for s in (cmp.segments if cmp else ())
                if s.op != "equal"
            ],
            "reasons": list(self.reasons),
            "image_quality": {
                "width": self.quality.width,
                "height": self.quality.height,
                "sharpness": self.quality.sharpness,
                "brightness": self.quality.brightness,
                "contrast": self.quality.contrast,
                "warnings": list(self.quality.warnings),
            },
        }


def verify(
    image_rgb: np.ndarray,
    alpr_text: str,
    reader: PlateReader,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> VerificationResult:
    """Independently read the plate in ``image_rgb`` and compare with ``alpr_text``.

    Raises:
        InvalidReadingError: if ``alpr_text`` is not a plausible plate string.
    """
    alpr = validate_alpr_reading(alpr_text)
    quality = assess_quality(image_rgb)
    reading = reader.read(image_rgb)

    def result(outcome: Outcome, comparison: Comparison | None, *reasons: str):
        return VerificationResult(
            outcome=outcome,
            alpr_input=alpr_text,
            alpr_normalized=alpr,
            ai_reading=reading,
            comparison=comparison,
            quality=quality,
            min_confidence=min_confidence,
            reasons=tuple(r for r in reasons if r),
        )

    quality_note = (
        "Image quality issues: " + " ".join(quality.warnings) if quality.warnings else ""
    )

    if reading is None:
        return result(
            Outcome.INCONCLUSIVE, None,
            "No plate-like text was detected in the image.", quality_note,
        )

    if reading.confidence < min_confidence:
        return result(
            Outcome.INCONCLUSIVE, None,
            f"Independent reading '{reading.text}' has confidence "
            f"{reading.confidence:.0%}, below the {min_confidence:.0%} threshold, "
            "so it cannot be used to corroborate or dispute the ALPR reading.",
            quality_note,
        )

    comparison = compare_readings(reading.text, alpr)
    if comparison.match:
        return result(
            Outcome.AGREE, comparison,
            f"Both systems read '{alpr}' (AI confidence {reading.confidence:.0%}).",
            quality_note,
        )
    return result(
        Outcome.DISAGREE, comparison,
        f"AI read '{reading.text}' but ALPR reported '{alpr}' "
        f"({comparison.differing_positions} difference(s), edit distance "
        f"{comparison.edit_distance}).",
        *comparison.notes,
        quality_note,
    )
