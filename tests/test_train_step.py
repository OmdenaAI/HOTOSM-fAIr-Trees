import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from fair_trees.dataset import RegionCache, SampleSpec, load_regions, plan_epoch  # noqa: E402
from fair_trees.params import AugmentParams, TrainParams  # noqa: E402
from fair_trees.train import (  # noqa: E402
    SampleDataset,
    build_model,
    collate,
    forward_loss,
    l2sp_penalty,
    make_loader,
    snapshot,
)


def _item(i, h, w, boxes):
    return (
        i,
        np.zeros((3, h, w), np.float32),
        (h - 16, w - 16),
        np.asarray(boxes, np.float32).reshape(-1, 4),
    )


def test_l2sp_zero_at_init_and_gradient():
    torch.manual_seed(0)
    model = torch.nn.Linear(3, 2)
    ref = snapshot(model)
    assert l2sp_penalty(model, ref).item() == 0.0
    with torch.no_grad():
        model.weight += 0.5
    alpha = 0.1
    (alpha * l2sp_penalty(model, ref)).backward()
    torch.testing.assert_close(model.weight.grad, 2 * alpha * (model.weight - ref["weight"]).detach())
    assert torch.count_nonzero(model.bias.grad) == 0


def test_snapshot_skips_frozen_parameters():
    model = torch.nn.Linear(3, 2)
    model.bias.requires_grad_(False)
    assert set(snapshot(model)) == {"weight"}


def test_collate_pads_and_keeps_resized_sizes():
    b = collate([_item(0, 800, 1216, [[1, 2, 30, 40]]), _item(1, 800, 800, [])])
    assert b.indices == [0, 1]
    assert tuple(b.images.shape) == (2, 3, 800, 1216) and b.images.dtype == torch.float32
    assert b.image_sizes == [(784, 1200), (784, 784)]
    assert b.targets[0]["labels"].tolist() == [0] and b.targets[0]["boxes"].dtype == torch.float32
    assert tuple(b.targets[1]["boxes"].shape) == (0, 4) and b.targets[1]["labels"].dtype == torch.int64


def test_sample_dataset_scales_boxes_and_drops_slivers(write_region):
    tiles = tuple((i, j) for j in range(4) for i in range(4))
    d = write_region("r", tiles=tiles, boxes_px=[(200, 200, 260, 240), (100, 100, 101.2, 140)])
    regions = load_regions(d.parent)
    cfg = TrainParams(augment=AugmentParams(downscale=False))
    gsd = RegionCache(regions).geom("r", 20).gsd_m
    spec = SampleSpec("r", 20, (512.0, 512.0), 1000 * gsd, 0)  # 1000 px crop, resized by 0.8
    index, image, size, boxes = SampleDataset([spec], regions, cfg)[0]
    assert index == 0 and image.shape == (3, 800, 800) and size == (800, 800)
    np.testing.assert_allclose(boxes, [[150.4, 150.4, 198.4, 182.4]], atol=1e-2)  # 0.96 px sliver dropped


def test_workers_yield_the_same_sample_ids(write_region):
    d = write_region("r", tiles=((0, 0), (1, 0)), boxes_px=[(30, 30, 60, 70)])
    regions = load_regions(d.parent)
    cfg = TrainParams(batch_size=2, samples_per_epoch=6)
    specs = plan_epoch(regions, cfg, 0, 0)

    def seq(workers):
        loader = make_loader(specs, regions, cfg, start=2, num_workers=workers)
        return [(b.indices, b.images.sum().item()) for b in loader]

    zero = seq(0)
    assert [i for ids, _ in zero for i in ids] == [2, 3, 4, 5]
    assert seq(2) == zero


def test_one_step_with_an_empty_target_is_finite():
    torch.manual_seed(0)
    model = build_model(None, trainable_layers=0).train()
    b = collate([_item(0, 800, 800, [[100, 100, 180, 200]]), _item(1, 800, 800, [])])
    b.images.normal_()
    losses = forward_loss(model, b.images, b.image_sizes, b.targets)
    assert set(losses) == {"classification", "bbox_regression"}
    total = sum(losses.values())
    assert torch.isfinite(total)
    total.backward()
    assert model.head.classification_head.cls_logits.weight.grad is not None
