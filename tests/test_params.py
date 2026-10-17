import logging

import pytest

from fair_trees.params import (
    INPUT_NAME,
    OUTPUT_NAMES,
    AugmentParams,
    InferenceParams,
    SplitParams,
    TrainParams,
    parse_inference_params,
    parse_split_params,
    parse_train_params,
)


@pytest.mark.parametrize("value", [None, {}])
def test_defaults_on_none_or_empty(value):
    assert parse_inference_params(value) == InferenceParams()
    assert parse_split_params(value) == SplitParams()
    assert parse_train_params(value) == TrainParams()


def test_inference_defaults():
    p = InferenceParams()
    assert (p.window_m, p.window_overlap) == (40.0, 0.5)
    assert (p.confidence_threshold, p.nms_threshold, p.iou_threshold) == (0.1, 0.05, 0.15)
    assert (p.edge_band, p.min_valid_fraction) == (0.05, 0.2)


def test_train_defaults():
    p = TrainParams()
    assert p.trainable_layers == 0 and p.l2sp_alpha == 0.0 and p.weight_decay == 1e-4
    assert p.zoom_weights == {19: 0.25, 20: 0.5, 21: 0.25}
    assert p.augment == AugmentParams()
    assert p.augment.downscale is True
    assert (p.augment.flips, p.augment.rot90, p.augment.colour_jitter) == (False, False, False)


def test_inference_aliases_and_canonical_wins():
    p = parse_inference_params({"score_thresh": 0.3, "nms_thresh": 0.2})
    assert (p.confidence_threshold, p.nms_threshold) == (0.3, 0.2)
    p = parse_inference_params({"confidence_threshold": 0.4, "score_thresh": 0.3})
    assert p.confidence_threshold == 0.4


def test_none_values_fall_back_to_alias_then_default():
    assert parse_inference_params({"confidence_threshold": None}).confidence_threshold == 0.1
    p = parse_inference_params({"confidence_threshold": None, "score_thresh": 0.3})
    assert p.confidence_threshold == 0.3


def test_numeric_strings_accepted():
    assert parse_inference_params({"window_m": "60"}).window_m == 60.0
    assert parse_train_params({"epochs": "3"}).epochs == 3


@pytest.mark.parametrize("key", ["window_m", "confidence_threshold"])
def test_bools_rejected_for_numbers(key):
    with pytest.raises(ValueError, match=key):
        parse_inference_params({key: True})


@pytest.mark.parametrize(
    "hp",
    [
        {"window_overlap": 1.0},
        {"window_overlap": -0.1},
        {"confidence_threshold": 1.5},
        {"window_m": 0},
        {"edge_band": 0.5},
        {"min_valid_fraction": 1.1},
        {"window_m": "abc"},
    ],
)
def test_inference_ranges(hp):
    with pytest.raises(ValueError):
        parse_inference_params(hp)


def test_split_params():
    p = parse_split_params({"val_ratio": 0.3, "z_ref": 19, "block_size": 2})
    assert p == SplitParams(val_ratio=0.3, split_seed=42, block_size=2, z_ref=19)
    for bad in ({"val_ratio": 1.0}, {"val_ratio": 0.0}, {"block_size": 0}, {"z_ref": -1}):
        with pytest.raises(ValueError):
            parse_split_params(bad)


def test_int_fields_reject_fractions():
    assert parse_train_params({"epochs": 3.0}).epochs == 3
    with pytest.raises(ValueError, match="epochs"):
        parse_train_params({"epochs": 2.5})


def test_train_aliases():
    p = parse_train_params({"lr": 3e-4, "freeze_backbone": True})
    assert p.learning_rate == 3e-4 and p.trainable_layers == 0
    assert parse_train_params({"freeze_backbone": False}).trainable_layers == 3
    assert parse_train_params({"freeze_backbone": True, "trainable_layers": 2}).trainable_layers == 2


def test_l2sp_forces_zero_weight_decay():
    p = parse_train_params({"l2sp_alpha": 1e-2, "weight_decay": 1e-4})
    assert p.weight_decay == 0.0 and p.l2sp_alpha == 1e-2
    assert parse_train_params({"weight_decay": 5e-4}).weight_decay == 5e-4


@pytest.mark.parametrize(
    "hp",
    [
        {"trainable_layers": 6},
        {"schedule": "step"},
        {"learning_rate": 0},
        {"patience": 0},
        {"keep_last": 0},
        {"zoom_weights": {20: -1}},
        {"zoom_weights": {20: 0}},
        {"augment": {"crop_m": [80, 30]}},
        {"augment": {"flips": 1}},
        {"amp": "yes"},
    ],
)
def test_train_ranges(hp):
    with pytest.raises(ValueError):
        parse_train_params(hp)


def test_zoom_weight_keys_from_json():
    p = parse_train_params({"zoom_weights": {"20": 1, "19": 0}})
    assert p.zoom_weights == {20: 1.0, 19: 0.0}


def test_augment_nested():
    p = parse_train_params({"augment": {"flips": True, "crop_m": [40, 60]}})
    assert p.augment == AugmentParams(crop_m=(40.0, 60.0), flips=True)


def test_unknown_keys_warn(caplog):
    with caplog.at_level(logging.WARNING, logger="fair_trees.params"):
        parse_inference_params({"windw_m": 50})
    assert "windw_m" in caplog.text


def test_onnx_names():
    assert INPUT_NAME == "images"
    assert OUTPUT_NAMES == ("cls_logits", "bbox_regression")
