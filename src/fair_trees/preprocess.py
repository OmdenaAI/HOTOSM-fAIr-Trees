"""Window preprocessing shared by training and ONNX serving: normalise, resize, pad."""

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image

from fair_trees.params import IMAGE_MEAN, IMAGE_STD, MAX_SIZE, MIN_SIZE, SIZE_DIVISIBLE


@dataclass(frozen=True)
class WindowScale:
    """Resized (unpadded) size and the factors that map window pixels to resized pixels."""

    resized_width: int
    resized_height: int
    scale_x: float
    scale_y: float


def preprocess_window(rgb_hwc: np.ndarray) -> tuple[np.ndarray, WindowScale]:
    """Normalise, resize (short side 800, long side <= 1333) and zero-pad to a multiple of 32.

    Returns a C-contiguous float32 batch of shape (1, 3, Hp, Wp).
    """
    if rgb_hwc.ndim != 3 or rgb_hwc.shape[2] != 3 or rgb_hwc.dtype != np.uint8:
        raise ValueError(f"expected uint8 array of shape (H, W, 3), got {rgb_hwc.shape} {rgb_hwc.dtype}")
    h, w = rgb_hwc.shape[:2]

    mean = np.asarray(IMAGE_MEAN, dtype=np.float32)
    std = np.asarray(IMAGE_STD, dtype=np.float32)
    x = (rgb_hwc.astype(np.float32) / 255.0 - mean) / std

    scale = min(MIN_SIZE / min(h, w), MAX_SIZE / max(h, w))
    rh, rw = math.floor(h * scale), math.floor(w * scale)

    hp = math.ceil(rh / SIZE_DIVISIBLE) * SIZE_DIVISIBLE
    wp = math.ceil(rw / SIZE_DIVISIBLE) * SIZE_DIVISIBLE
    batch = np.zeros((1, 3, hp, wp), np.float32)
    for c in range(3):
        channel = np.ascontiguousarray(x[:, :, c], dtype=np.float32)
        resized = Image.fromarray(channel).resize((rw, rh), Image.Resampling.BILINEAR)
        batch[0, c, :rh, :rw] = np.asarray(resized, dtype=np.float32)
    return batch, WindowScale(rw, rh, rw / w, rh / h)
