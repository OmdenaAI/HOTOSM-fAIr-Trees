import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from fair_trees.dataset import RegionCache, SampleSpec, load_regions, plan_epoch  # noqa: E402
from fair_trees.params import CLASSES, AugmentParams, TrainParams  # noqa: E402
from fair_trees.train import (  # noqa: E402
    SampleDataset,
    build_model,
    collate,
    forward_loss,
    l2sp_penalty,
    make_loader,
    masked_compute_loss,
    snapshot,
)


def _item(i, h, w, boxes, labels=None, unsupervised=None, k=1):
    boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
    return (
        i,
        np.zeros((3, h, w), np.float32),
        (h - 16, w - 16),
        boxes,
        np.zeros(len(boxes), np.int64) if labels is None else np.asarray(labels, np.int64),
        np.zeros((k, h - 16, w - 16), bool) if unsupervised is None else unsupervised,
    )


def _parts(model, b):
    """Head outputs and anchors of a batch, as forward_loss computes them."""
    from torchvision.models.detection.image_list import ImageList

    features = list(model.backbone(b.images).values())
    return model.head(features), model.anchor_generator(ImageList(b.images, b.image_sizes), features)


def _focal_terms(model, b):
    """Per-anchor, per-class focal terms of image 0 over the anchors torchvision scores, with targets."""
    from torchvision.ops import boxes as box_ops
    from torchvision.ops import sigmoid_focal_loss

    head, anchors = _parts(model, b)
    t = b.targets[0]
    matched = model.proposal_matcher(box_ops.box_iou(t["boxes"], anchors[0]))
    fg = matched >= 0
    target = torch.zeros_like(head["cls_logits"][0])
    target[fg, t["labels"][matched[fg]]] = 1.0
    valid = matched != model.head.classification_head.BETWEEN_THRESHOLDS
    terms = sigmoid_focal_loss(head["cls_logits"][0][valid], target[valid], reduction="none")
    return terms, target[valid], int(fg.sum())


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
    assert tuple(b.targets[1]["unsupervised"].shape) == (1, 800, 1216)


def test_collate_pads_unsupervised_with_false():
    unsup = np.ones((2, 784, 784), bool)
    b = collate([_item(0, 800, 800, [[1, 2, 30, 40]], [1], unsup, k=2)])
    u = b.targets[0]["unsupervised"]
    assert u.dtype == torch.bool and tuple(u.shape) == (2, 800, 800)
    assert u[:, :784, :784].all() and not u[:, 784:].any() and not u[:, :, 784:].any()
    assert b.targets[0]["labels"].tolist() == [1]


@pytest.mark.parametrize("k", [1, 2])
def test_masked_loss_equals_torchvision_without_masks(k):
    torch.manual_seed(0)
    model = build_model(None, trainable_layers=0, num_classes=k).train()
    labels = [0, k - 1]
    b = collate([_item(0, 800, 800, [[100, 100, 180, 200], [300, 300, 420, 380]], labels, k=k)])
    b.images.normal_()
    head, anchors = _parts(model, b)
    ours = masked_compute_loss(model, b.targets, head, anchors)
    theirs = model.compute_loss(b.targets, head, anchors)
    for key in theirs:
        torch.testing.assert_close(ours[key], theirs[key], rtol=1e-6, atol=1e-6)


def test_unsupervised_drops_only_negatives():
    torch.manual_seed(0)
    model = build_model(None, trainable_layers=0, num_classes=2).train()
    b = collate([_item(0, 800, 800, [[100, 100, 180, 200]], [0], k=2)])
    b.targets[0]["unsupervised"][1] = True  # the whole padded image, so no anchor keeps a class-1 term
    b.images.normal_()
    terms, target, num_fg = _focal_terms(model, b)
    loss = forward_loss(model, b.images, b.image_sizes, b.targets)["classification"]
    torch.testing.assert_close(loss, terms[:, 0].sum() / num_fg)
    b.targets[0]["unsupervised"][0] = True  # tree unsupervised too: only its positives remain
    loss = forward_loss(model, b.images, b.image_sizes, b.targets)["classification"]
    torch.testing.assert_close(loss, (terms[:, 0] * target[:, 0]).sum() / num_fg)


def test_k1_model_on_group_regions_drops_group_boxes(write_region):
    d = write_region(
        "r", tiles=((0, 0),), boxes_px=[(150, 60, 170, 80)], groups_px=[(40, 40, 120, 100)], classes=CLASSES
    )
    regions = load_regions(d.parent)
    cfg = TrainParams(augment=AugmentParams(downscale=False))
    spec = SampleSpec("r", 20, (128.0, 128.0), 200 * RegionCache(regions).geom("r", 20).gsd_m, 0)
    item = SampleDataset([spec], regions, cfg, num_classes=1)[0]
    _, _, size, boxes, labels, unsup = item
    assert len(boxes) == 1 and labels.tolist() == [0]
    assert unsup.shape == (1, *size) and unsup.any()  # trees unsupervised inside the group
    torch.manual_seed(0)
    model = build_model(None, trainable_layers=0).train()
    b = collate([item])
    assert torch.isfinite(sum(forward_loss(model, b.images, b.image_sizes, b.targets).values()))
    two = SampleDataset([spec], regions, cfg, num_classes=2)[0]
    assert two[4].tolist() == [0, 1] and two[5].shape == (2, *size)


def test_sample_dataset_scales_boxes_and_drops_slivers(write_region):
    tiles = tuple((i, j) for j in range(4) for i in range(4))
    d = write_region("r", tiles=tiles, boxes_px=[(200, 200, 260, 240), (100, 100, 101.2, 140)])
    regions = load_regions(d.parent)
    cfg = TrainParams(augment=AugmentParams(downscale=False))
    gsd = RegionCache(regions).geom("r", 20).gsd_m
    spec = SampleSpec("r", 20, (512.0, 512.0), 1000 * gsd, 0)  # 1000 px crop, resized by 0.8
    index, image, size, boxes, labels, unsup = SampleDataset([spec], regions, cfg)[0]
    assert labels.tolist() == [0] and unsup.shape == (1, 800, 800) and not unsup.any()
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
