"""Image loading, quality checks and OCR pre-processing (Pillow + OpenCV).

Images are only ever held in memory; nothing is written to disk.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_SIDE = 1600  # larger images are downscaled; OCR gains nothing past this
MIN_SIDE = 32  # anything smaller cannot contain a legible plate
ALLOWED_FORMATS = {"JPEG", "PNG", "BMP", "WEBP"}

# Quality thresholds (empirical; deliberately conservative).
BLUR_THRESHOLD = 60.0  # variance of the Laplacian below this ≈ blurry
DARK_THRESHOLD = 50.0  # mean brightness (0-255)
BRIGHT_THRESHOLD = 220.0
LOW_CONTRAST_THRESHOLD = 25.0  # std-dev of grayscale


class ImageError(ValueError):
    """Raised when an upload cannot be used as evidence."""


@dataclass(frozen=True)
class QualityReport:
    width: int
    height: int
    sharpness: float
    brightness: float
    contrast: float
    warnings: tuple[str, ...] = field(default_factory=tuple)


def load_image(data: bytes) -> np.ndarray:
    """Decode uploaded bytes into an RGB ``uint8`` array, with validation.

    Raises:
        ImageError: empty, too large, not an image, unsupported format or too small.
    """
    if not data:
        raise ImageError("The uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise ImageError(
            f"Image is {len(data) / 1_048_576:.1f} MB; the limit is "
            f"{MAX_UPLOAD_BYTES // 1_048_576} MB."
        )
    try:
        with Image.open(io.BytesIO(data)) as probe:
            probe.verify()  # detects truncated / corrupt files
        img = Image.open(io.BytesIO(data))
        fmt = img.format
        img.load()
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise ImageError("The file is not a readable image.") from exc
    except Image.DecompressionBombError as exc:
        raise ImageError("Image dimensions are too large to process safely.") from exc

    if fmt not in ALLOWED_FORMATS:
        raise ImageError(
            f"Unsupported image format '{fmt}'. Use JPEG, PNG, BMP or WEBP."
        )

    img = ImageOps.exif_transpose(img)  # respect phone-camera orientation
    if img.mode != "RGB":
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGBA")
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.getchannel("A"))
            img = background
        else:
            img = img.convert("RGB")

    if min(img.size) < MIN_SIDE:
        raise ImageError(
            f"Image is only {img.width}×{img.height}px; at least {MIN_SIDE}px "
            "per side is needed."
        )
    if max(img.size) > MAX_SIDE:
        img.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)

    return np.asarray(img, dtype=np.uint8).copy()


def to_gray(rgb: np.ndarray) -> np.ndarray:
    if rgb.ndim == 2:
        return rgb
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def assess_quality(rgb: np.ndarray) -> QualityReport:
    """Cheap, explainable image-quality metrics shown to the reviewer."""
    gray = to_gray(rgb)
    height, width = gray.shape[:2]
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    contrast = float(gray.std())

    warnings: list[str] = []
    if sharpness < BLUR_THRESHOLD:
        warnings.append("Image appears blurry (low edge detail).")
    if brightness < DARK_THRESHOLD:
        warnings.append("Image is very dark.")
    elif brightness > BRIGHT_THRESHOLD:
        warnings.append("Image is very bright / overexposed.")
    if contrast < LOW_CONTRAST_THRESHOLD:
        warnings.append("Image has very low contrast.")
    if min(width, height) < 100:
        warnings.append("Image resolution is low; characters may be illegible.")

    return QualityReport(
        width=width,
        height=height,
        sharpness=round(sharpness, 1),
        brightness=round(brightness, 1),
        contrast=round(contrast, 1),
        warnings=tuple(warnings),
    )


def enhance_for_ocr(rgb: np.ndarray) -> np.ndarray:
    """Return a grayscale, contrast-equalized, upscaled copy for a second OCR pass."""
    gray = to_gray(rgb)
    h, w = gray.shape[:2]
    if h < 400:
        scale = min(3.0, 400 / h)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    return clahe.apply(gray)


def draw_box(rgb: np.ndarray, box: tuple[int, int, int, int], label: str = "") -> np.ndarray:
    """Return a copy of the image with the detected plate region outlined."""
    out = rgb.copy()
    x0, y0, x1, y1 = box
    thickness = max(2, round(max(out.shape[:2]) / 300))
    color = (37, 99, 235)  # blue, RGB
    cv2.rectangle(out, (x0, y0), (x1, y1), color, thickness)
    if label:
        scale = max(0.5, thickness / 3)
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        ty = y0 - 6 if y0 - th - 10 > 0 else y1 + th + 6
        cv2.rectangle(out, (x0, ty - th - 4), (x0 + tw + 6, ty + base), color, -1)
        cv2.putText(out, label, (x0 + 3, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (255, 255, 255), thickness, cv2.LINE_AA)
    return out
