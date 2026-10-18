"""Opt-in parity: our training forward pass gives torchvision's training losses on NEON.pt."""

import numpy as np
import pytest

pytestmark = pytest.mark.integration
torch = pytest.importorskip("torch")

from fair_trees.preprocess import preprocess_window  # noqa: E402
from fair_trees.train import collate, forward_loss  # noqa: E402


def test_training_loss_matches_torchvision(neon_model, sample_rgb):
    # a short side of exactly 800 makes torchvision's resize the identity
    rgb = np.ascontiguousarray(np.repeat(np.repeat(sample_rgb, 2, axis=0), 2, axis=1)[:800, :800])
    boxes = np.array([[100, 120, 180, 210], [400, 380, 470, 460], [600, 50, 690, 140]], np.float32)
    batch, scale = preprocess_window(rgb)
    assert (scale.resized_height, scale.resized_width) == (800, 800)
    empty = np.zeros((0, 4), np.float32)
    b = collate([(0, batch[0], (800, 800), boxes), (1, batch[0], (800, 800), empty)])
    image = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255)
    neon_model.train()
    try:
        with torch.no_grad():
            ours = forward_loss(neon_model, b.images, b.image_sizes, b.targets)
            theirs = neon_model([image, image], b.targets)
    finally:
        neon_model.eval()
    for key in ("classification", "bbox_regression"):
        torch.testing.assert_close(ours[key], theirs[key], rtol=1e-4, atol=1e-5)
