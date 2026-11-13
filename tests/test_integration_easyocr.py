"""End-to-end tests with the real EasyOCR model on the staged sample images.

Slow (model load + CPU inference) and needs the model download on first run.
Skipped automatically if EasyOCR is not installed. Run only these with:
    pytest -m integration
"""

from pathlib import Path

import pytest

pytest.importorskip("easyocr")

from watchtower.imaging import load_image  # noqa: E402
from watchtower.ocr import PlateReader  # noqa: E402
from watchtower.verify import Outcome, verify  # noqa: E402

pytestmark = pytest.mark.integration
SAMPLES = Path(__file__).resolve().parent.parent / "samples"


@pytest.fixture(scope="module")
def reader():
    return PlateReader()


@pytest.mark.parametrize(
    "image, alpr, expected_outcome, expected_ai",
    [
        ("plate_clear.png", "BXR 4827", Outcome.AGREE, "BXR4827"),
        ("plate_confusable.png", "DKT B150", Outcome.DISAGREE, "DKT8150"),
        ("plate_mismatch.png", "HWN 3329", Outcome.DISAGREE, "HWN3392"),
        ("plate_unreadable.png", "CPL 6604", Outcome.INCONCLUSIVE, None),
        ("vehicle_scene.png", "ELM 2045", Outcome.AGREE, "ELM2045"),
    ],
)
def test_demo_scenarios(reader, image, alpr, expected_outcome, expected_ai):
    path = SAMPLES / image
    if not path.exists():
        pytest.skip("Sample images missing: run python scripts/make_samples.py")
    result = verify(load_image(path.read_bytes()), alpr, reader)
    assert result.outcome is expected_outcome, result.reasons
    if expected_ai is not None:
        assert result.ai_reading.text == expected_ai
