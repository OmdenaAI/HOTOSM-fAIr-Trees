import math

import numpy as np
import pytest

from fair_trees.params import IMAGE_MEAN, IMAGE_STD
from fair_trees.preprocess import WindowScale, preprocess_window


def test_resize_floor_and_pad_is_exactly_zero():
    batch, scale = preprocess_window(np.full((256, 300, 3), 200, np.uint8))
    assert scale == WindowScale(resized_width=937, resized_height=800, scale_x=937 / 300, scale_y=800 / 256)
    assert batch.shape == (1, 3, 800, 960) and batch.dtype == np.float32
    assert batch.flags["C_CONTIGUOUS"]
    assert np.all(batch[..., 937:] == 0.0)


def test_values_are_normalised_not_0_255():
    batch, _ = preprocess_window(np.full((400, 400, 3), 255, np.uint8))
    expected = (1.0 - np.array(IMAGE_MEAN)) / np.array(IMAGE_STD)
    np.testing.assert_allclose(batch[0, :, 400, 400], expected, atol=1e-5)


def test_max_size_cap():
    batch, scale = preprocess_window(np.zeros((100, 1000, 3), np.uint8))
    assert (scale.resized_height, scale.resized_width) == (133, 1333)
    assert batch.shape == (1, 3, 160, 1344)


@pytest.mark.parametrize("hw", [(256, 256), (300, 420), (400, 400), (800, 1088), (37, 53), (1600, 2000)])
def test_sizes_follow_torchvision_formula(hw):
    h, w = hw
    batch, scale = preprocess_window(np.zeros((h, w, 3), np.uint8))
    s = min(800 / min(h, w), 1333 / max(h, w))
    assert (scale.resized_height, scale.resized_width) == (math.floor(h * s), math.floor(w * s))
    assert batch.shape[2] % 32 == 0 and batch.shape[3] % 32 == 0
    assert batch.shape[2] - scale.resized_height < 32 and batch.shape[3] - scale.resized_width < 32


@pytest.mark.parametrize(
    "bad", [np.zeros((10, 10), np.uint8), np.zeros((10, 10, 3), np.float32), np.zeros((10, 10, 4), np.uint8)]
)
def test_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        preprocess_window(bad)


def test_downscale_matches_torch_antialias():
    torch = pytest.importorskip("torch")
    yy, xx = np.mgrid[0:1600, 0:2000]
    rgb = np.stack(
        [127 + 100 * np.sin(xx / 37.0), 127 + 100 * np.cos(yy / 23.0), 127 + 60 * np.sin((xx + yy) / 51.0)],
        axis=-1,
    ).astype(np.uint8)
    batch, scale = preprocess_window(rgb)
    assert (scale.resized_height, scale.resized_width) == (800, 1000)
    x = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255)
    x = (x - torch.tensor(IMAGE_MEAN).view(3, 1, 1)) / torch.tensor(IMAGE_STD).view(3, 1, 1)
    ref = torch.nn.functional.interpolate(
        x[None], size=(800, 1000), mode="bilinear", align_corners=False, antialias=True
    )[0].numpy()
    # Stated tolerance [P]: PIL's bilinear downscale and torch's antialiased bilinear agree to this.
    assert np.abs(batch[0, :, :800, :1000] - ref).mean() < 0.02
