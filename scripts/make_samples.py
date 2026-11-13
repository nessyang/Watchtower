"""Generate fictional, staged license-plate images for demos and tests.

All plates are synthetic, clearly marked "SAMPLE - NOT A REAL PLATE", and
belong to no real vehicle. Run from the project root:

    python scripts/make_samples.py
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT_DIR = Path(__file__).resolve().parent.parent / "samples"

_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "DejaVuSans-Bold.ttf",
]


def _font(size: int) -> ImageFont.FreeTypeFont:
    for path in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)  # Pillow >= 10.1


def _centered(draw: ImageDraw.ImageDraw, y: int, text: str, font, width: int, fill):
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    draw.text(((width - (right - left)) / 2 - left, y - top), text, font=font, fill=fill)


def make_plate(text: str, width: int = 600, height: int = 300) -> Image.Image:
    """A generic white plate with a fictional registration number."""
    plate = Image.new("RGB", (width, height), (248, 248, 244))
    d = ImageDraw.Draw(plate)
    navy = (20, 40, 95)
    d.rounded_rectangle((4, 4, width - 5, height - 5), radius=22, outline=navy, width=8)
    _centered(d, 22, "MICHIGAN", _font(40), width, navy)
    size = 132  # shrink the registration font until it fits inside the border
    while size > 20:
        left, _, right, _ = d.textbbox((0, 0), text, font=_font(size))
        if right - left <= width * 0.84:
            break
        size -= 4
    _centered(d, 150 - size // 2, text, _font(size), width, (15, 15, 20))
    _centered(d, 246, "SAMPLE - NOT A REAL PLATE", _font(24), width, (120, 40, 40))
    return plate


def make_vehicle_scene(text: str) -> Image.Image:
    """A crude, stylized rear view of a car with a perspective-warped plate."""
    w, h = 1200, 800
    scene = Image.new("RGB", (w, h), (70, 78, 88))
    d = ImageDraw.Draw(scene)
    d.rectangle((0, 620, w, h), fill=(45, 45, 48))  # road
    d.rounded_rectangle((150, 180, 1050, 660), radius=60, fill=(140, 22, 30))  # body
    d.rounded_rectangle((250, 120, 950, 330), radius=50, fill=(110, 18, 25))  # roof
    d.rounded_rectangle((290, 150, 910, 310), radius=30, fill=(150, 180, 200))  # window
    d.rounded_rectangle((170, 360, 330, 450), radius=18, fill=(230, 60, 40))  # lights
    d.rounded_rectangle((870, 360, 1030, 450), radius=18, fill=(230, 60, 40))
    d.rectangle((150, 560, 1050, 640), fill=(30, 30, 32))  # bumper
    for x in (210, 990):  # tyres
        d.ellipse((x - 70, 600, x + 70, 720), fill=(20, 20, 20))

    rgb = np.asarray(scene).copy()
    plate = np.asarray(make_plate(text, 600, 300))
    src = np.float32([[0, 0], [600, 0], [600, 300], [0, 300]])
    dst = np.float32([[452, 410], [748, 402], [752, 552], [448, 558]])  # slight skew
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(plate, matrix, (w, h))
    mask = cv2.warpPerspective(np.full((300, 600), 255, np.uint8), matrix, (w, h)) > 0
    rgb[mask] = warped[mask]
    return Image.fromarray(rgb)


def degrade(img: Image.Image, blur: int, noise: float, darken: float, seed: int = 7) -> Image.Image:
    """Simulate a poor capture: motion/defocus blur, sensor noise, low light."""
    arr = np.asarray(img).astype(np.float32)
    if blur:
        arr = cv2.GaussianBlur(arr, (blur, blur), 0)
        kernel = np.zeros((blur, blur), np.float32)
        kernel[blur // 2, :] = 1.0 / blur  # horizontal motion streak
        arr = cv2.filter2D(arr, -1, kernel)
    arr *= darken
    if noise:
        arr += np.random.default_rng(seed).normal(0, noise, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


SAMPLES = {
    "plate_clear.png": lambda: make_plate("BXR 4827"),
    "plate_confusable.png": lambda: make_plate("DKT 8150"),
    "plate_mismatch.png": lambda: make_plate("HWN 3392"),
    "plate_unreadable.png": lambda: degrade(make_plate("CPL 6604"), blur=41, noise=18, darken=0.45),
    "vehicle_scene.png": lambda: make_vehicle_scene("ELM 2045"),
}


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    for name, build in SAMPLES.items():
        path = OUT_DIR / name
        build().save(path)
        print(f"wrote {path.relative_to(OUT_DIR.parent)}")


if __name__ == "__main__":
    main()
