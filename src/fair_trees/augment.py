"""Training augmentations: downscale, flips, rot90 and colour jitter, with boxes kept in step.

Torch-free (numpy and Pillow only) and never imported by serving. Boxes are float32 (N, 4) XYXY in
pixel coordinates. Every function returns new arrays and leaves its inputs untouched. Each draws from
the caller's generator in a fixed order, so one seed per sample reproduces the whole pipeline.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageEnhance

# Open values: pending a check of DeepForest's Downscale config and the unit of "hue +-10" [?].
DOWNSCALE_RANGE = (0.5, 1.0)  # [?]
DOWNSCALE_P = 0.5  # [?]
HUE_JITTER = 10.0  # degrees [?]


def downscale(
    rgb: np.ndarray, rng: np.random.Generator, factor_range: tuple[float, float] = DOWNSCALE_RANGE
) -> np.ndarray:
    """Blur by shrinking with a box filter and scaling back up; the output keeps the input size."""
    h, w = rgb.shape[:2]
    f = rng.uniform(*factor_range)
    small = (max(1, round(w * f)), max(1, round(h * f)))
    if small == (w, h):
        return rgb.copy()
    img = Image.fromarray(rgb)
    img = img.resize(small, Image.Resampling.BOX).resize((w, h), Image.Resampling.BILINEAR)
    return np.asarray(img)


def flip(img: np.ndarray, boxes: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Flip horizontally and/or vertically, each with probability 0.5 (both draws always made)."""
    h, w = img.shape[:2]
    do_h = rng.random() < 0.5
    do_v = rng.random() < 0.5
    out = boxes.astype(np.float32, copy=True)
    if do_h:
        img = img[:, ::-1]
        out[:, [0, 2]] = w - boxes[:, [2, 0]]
    if do_v:
        img = img[::-1]
        out[:, [1, 3]] = h - boxes[:, [3, 1]]
    return np.ascontiguousarray(img), out


def rot90(img: np.ndarray, boxes: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Rotate counter-clockwise by a random multiple of 90 degrees."""
    k = int(rng.integers(0, 4))
    h, w = img.shape[:2]
    out = boxes.astype(np.float32, copy=True)
    for _ in range(k):
        x1, y1, x2, y2 = out.T.copy()
        out = np.stack([y1, w - x2, y2, w - x1], axis=1).astype(np.float32)
        h, w = w, h
    return np.ascontiguousarray(np.rot90(img, k, axes=(0, 1))), out


def colour_jitter(
    rgb: np.ndarray,
    rng: np.random.Generator,
    brightness: float = 0.2,
    contrast: float = 0.2,
    saturation: float = 0.2,
    hue: float = HUE_JITTER,
) -> np.ndarray:
    """Jitter brightness, contrast, saturation (factors) and hue (degrees), in that order."""
    fb = rng.uniform(1 - brightness, 1 + brightness)
    fc = rng.uniform(1 - contrast, 1 + contrast)
    fs = rng.uniform(1 - saturation, 1 + saturation)
    dh = rng.uniform(-hue, hue)
    img = Image.fromarray(rgb)
    for enhancer, factor in (
        (ImageEnhance.Brightness, fb),
        (ImageEnhance.Contrast, fc),
        (ImageEnhance.Color, fs),
    ):
        if factor != 1.0:
            img = enhancer(img).enhance(factor)
    shift = round(dh / 360 * 256) % 256
    if shift:
        hsv = np.asarray(img.convert("HSV")).copy()
        hsv[..., 0] += np.uint8(shift)  # uint8 wraps modulo 256
        img = Image.fromarray(hsv, "HSV").convert("RGB")
    return np.asarray(img)
