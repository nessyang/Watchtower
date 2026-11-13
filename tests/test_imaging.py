import io

import numpy as np
import pytest
from PIL import Image

from watchtower import imaging
from watchtower.imaging import ImageError, assess_quality, draw_box, enhance_for_ocr, load_image


def encode(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def checkerboard(w=200, h=100) -> Image.Image:
    arr = np.indices((h, w)).sum(axis=0) // 10 % 2 * 255
    return Image.fromarray(arr.astype(np.uint8)).convert("RGB")


def test_load_png_returns_rgb_array():
    arr = load_image(encode(checkerboard()))
    assert arr.shape == (100, 200, 3)
    assert arr.dtype == np.uint8


def test_load_jpeg_and_rgba_and_grayscale():
    assert load_image(encode(checkerboard(), "JPEG")).shape == (100, 200, 3)
    rgba = Image.new("RGBA", (64, 64), (255, 0, 0, 0))  # fully transparent → white
    assert load_image(encode(rgba))[0, 0].tolist() == [255, 255, 255]
    assert load_image(encode(checkerboard().convert("L"))).shape == (100, 200, 3)


@pytest.mark.parametrize(
    "data, message",
    [
        (b"", "empty"),
        (b"definitely not an image", "not a readable image"),
        (b"\x89PNG\r\n\x1a\n" + b"\x00" * 50, "not a readable image"),  # truncated PNG
    ],
)
def test_rejects_bad_bytes(data, message):
    with pytest.raises(ImageError, match=message):
        load_image(data)


def test_rejects_oversized_upload(monkeypatch):
    monkeypatch.setattr(imaging, "MAX_UPLOAD_BYTES", 100)
    with pytest.raises(ImageError, match="limit"):
        load_image(encode(checkerboard()))


def test_rejects_tiny_image():
    with pytest.raises(ImageError, match="at least"):
        load_image(encode(Image.new("RGB", (20, 20))))


def test_rejects_unsupported_format():
    with pytest.raises(ImageError, match="Unsupported"):
        load_image(encode(checkerboard(), "GIF"))


def test_large_image_is_downscaled():
    arr = load_image(encode(Image.new("RGB", (4000, 1000), "white")))
    assert max(arr.shape[:2]) == imaging.MAX_SIDE


def test_quality_flags_blank_dark_image():
    report = assess_quality(np.zeros((120, 240, 3), np.uint8))
    joined = " ".join(report.warnings)
    assert "blurry" in joined and "dark" in joined and "contrast" in joined


def test_quality_sharp_image_has_no_blur_warning():
    report = assess_quality(np.asarray(checkerboard(400, 200)))
    assert not any("blurry" in w for w in report.warnings)


def test_enhance_upscales_small_images_to_grayscale():
    out = enhance_for_ocr(np.asarray(checkerboard(200, 100)))
    assert out.ndim == 2 and out.shape[0] >= 300


def test_draw_box_does_not_modify_original():
    original = np.full((100, 200, 3), 255, np.uint8)
    boxed = draw_box(original, (10, 10, 100, 60), "ABC123")
    assert (original == 255).all()
    assert not (boxed == 255).all()
