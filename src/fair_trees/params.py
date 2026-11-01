"""Shared constants, parameter dataclasses and tolerant parsers (stdlib only)."""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

logger = logging.getLogger(__name__)

MIN_SIZE = 800
MAX_SIZE = 1333
SIZE_DIVISIBLE = 32
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)

MAX_WINDOW_M = 200.0
MAX_MOSAIC_PIXELS = 2**28  # 16384 x 16384 px, ~1 GiB of RGB + valid mask across one call's mosaics

CLASSES = ("tree", "tree_group")  # head order; GeoJSON label and output class number = index + 1
GROUP_MERGE_RULES = ("join", "overlap", "drop_edge")  # see decode.merge_group_detections

INPUT_NAME = "images"
OUTPUT_NAMES = ("cls_logits", "bbox_regression")
THRESHOLDS_KEY = "fair_trees.confidence_thresholds"  # ONNX metadata: JSON {class name: val-picked threshold}


@dataclass(frozen=True)
class SplitParams:
    val_ratio: float = 0.2
    split_seed: int = 42
    block_size: int = 4
    z_ref: int | None = None  # None = lowest zoom in the input


@dataclass(frozen=True)
class InferenceParams:
    window_m: float = 40.0
    window_overlap: float = 0.5
    confidence_threshold: float = 0.1
    nms_threshold: float = 0.05
    iou_threshold: float = 0.15
    edge_band: float = 0.05
    min_valid_fraction: float = 0.2
    group_window_m: float = 120.0  # tree_group window; window_m is the tree window
    # how group boxes cut by a window edge combine (GROUP_MERGE_RULES); drop_edge beat join by +0.08 group F1
    # [0.06, 0.10] on HOT val
    group_merge: str = "drop_edge"
    # per class (CLASSES order), the model's own thresholds; a class not covered uses confidence_threshold.
    # Set from the model's metadata by predict_session, never parsed from the caller's params.
    class_thresholds: tuple[float, ...] = ()

    def threshold(self, class_index: int) -> float:
        """The decode threshold of a class."""
        if class_index < len(self.class_thresholds):
            return self.class_thresholds[class_index]
        return self.confidence_threshold


@dataclass(frozen=True)
class AugmentParams:
    crop_m: tuple[float, float] = (30.0, 80.0)
    downscale: bool = True  # on unless an ablation shows harm
    flips: bool = False
    rot90: bool = False
    colour_jitter: bool = False  # off unless an ablation shows gain


@dataclass(frozen=True)
class TrainParams:
    epochs: int = 10
    learning_rate: float = 1e-3
    momentum: float = 0.9
    weight_decay: float = 1e-4
    l2sp_alpha: float = 0.0  # platform alpha is set after adaptation tuning [?]
    trainable_layers: int = 0  # base 3, platform 0
    batch_size: int = 4
    samples_per_epoch: int | None = None  # None = number of train chips
    schedule: str = "cosine"
    patience: int = 3
    min_delta: float = 0.005
    amp: bool = True
    seed: int = 42
    ckpt_every_min: float = 10.0
    keep_last: int = 4
    zoom_weights: Mapping[int, float] = field(default_factory=lambda: {19: 0.25, 20: 0.5, 21: 0.25})
    source_alpha: float = 0.5
    biome_alpha: float = 0.5
    region_repeat_cap: int | None = None
    empty_crop_cap: float | None = None
    num_classes: int | None = None  # None = the start weights' classes; 2 adds tree_group to a tree model
    augment: AugmentParams = AugmentParams()


_SCHEDULES = ("cosine", "constant")


def _pick(mapping: Mapping, canonical: str, aliases: Sequence[str], default: Any) -> Any:
    """Canonical key first, then the first alias present; `None` counts as absent."""
    for key in (canonical, *aliases):
        if mapping.get(key) is not None:
            return mapping[key]
    return default


def _as_mapping(name: str, value: Mapping | None) -> Mapping:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping, got {type(value).__name__}")
    return value


def _warn_unknown(mapping: Mapping, known: set[str], where: str) -> None:
    unknown = sorted(str(k) for k in mapping if k not in known)
    if unknown:
        logger.warning("Ignoring unknown %s keys: %s", where, ", ".join(unknown))


def _number(name: str, value: Any, kind: type) -> int | float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number, got bool {value!r}")
    if isinstance(value, str):
        try:
            value = float(value.strip())
        except ValueError:
            raise ValueError(f"{name} must be numeric, got {value!r}") from None
    elif not isinstance(value, int | float):
        raise ValueError(f"{name} must be a number, got {type(value).__name__}")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    try:
        if kind is int:
            if value != int(value):
                raise ValueError(f"{name} must be an integer, got {value!r}")
            return int(value)
        return float(value)
    except OverflowError:
        raise ValueError(f"{name} is too large, got {value!r}") from None


def _field(mapping: Mapping, name: str, default: Any, kind: type, aliases: Sequence[str] = ()) -> Any:
    value = _pick(mapping, name, aliases, None)
    return default if value is None else _number(name, value, kind)


def _bool(mapping: Mapping, name: str, default: bool) -> bool:
    value = mapping.get(name)
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool, got {value!r}")
    return value


def _require(name: str, ok: bool, rule: str, value: Any) -> None:
    if not ok:
        raise ValueError(f"{name} must satisfy {rule}, got {value!r}")


def _in_range(name: str, value: Any, lo: float, hi: float, *, lo_open=False, hi_open=False) -> None:
    ok = (lo < value if lo_open else lo <= value) and (value < hi if hi_open else value <= hi)
    rule = f"{lo} {'<' if lo_open else '<='} {name} {'<' if hi_open else '<='} {hi}"
    _require(name, ok, rule, value)


def _at_least(name: str, value: Any, minimum: float, *, strict: bool = False) -> None:
    ok = value > minimum if strict else value >= minimum
    _require(name, ok, f"{'>' if strict else '>='} {minimum}", value)


def parse_split_params(hp: Mapping | None) -> SplitParams:
    hp = _as_mapping("split params", hp)
    _warn_unknown(hp, set(SplitParams.__dataclass_fields__), "split parameter")
    d = SplitParams()
    p = SplitParams(
        val_ratio=_field(hp, "val_ratio", d.val_ratio, float),
        split_seed=_field(hp, "split_seed", d.split_seed, int),
        block_size=_field(hp, "block_size", d.block_size, int),
        z_ref=_field(hp, "z_ref", d.z_ref, int),
    )
    _in_range("val_ratio", p.val_ratio, 0, 1, lo_open=True, hi_open=True)
    _at_least("block_size", p.block_size, 1)
    if p.z_ref is not None:
        _in_range("z_ref", p.z_ref, 0, 30)
    return p


def parse_inference_params(params: Mapping | None) -> InferenceParams:
    params = _as_mapping("inference params", params)
    aliases = {"confidence_threshold": ("score_thresh",), "nms_threshold": ("nms_thresh",)}
    fields = set(InferenceParams.__dataclass_fields__) - {"class_thresholds"}
    _warn_unknown(params, fields | {a for v in aliases.values() for a in v}, "inference parameter")
    d = InferenceParams()
    numbers = sorted(fields - {"group_merge"})
    p = InferenceParams(
        **{name: _field(params, name, getattr(d, name), float, aliases.get(name, ())) for name in numbers},
        group_merge=params.get("group_merge") or d.group_merge,
    )
    _require("group_merge", p.group_merge in GROUP_MERGE_RULES, f"one of {GROUP_MERGE_RULES}", p.group_merge)
    _in_range("window_m", p.window_m, 5, MAX_WINDOW_M)
    _in_range("group_window_m", p.group_window_m, 5, MAX_WINDOW_M)
    _in_range("window_overlap", p.window_overlap, 0, 0.9)
    for name in ("confidence_threshold", "nms_threshold", "iou_threshold"):
        _in_range(name, getattr(p, name), 0, 1)
    _in_range("edge_band", p.edge_band, 0, 0.5, hi_open=True)
    _in_range("min_valid_fraction", p.min_valid_fraction, 0, 1)
    return p


def caller_sets_threshold(params: Mapping | None) -> bool:
    """Whether the caller's inference params name a confidence threshold (canonical or alias)."""
    return any((params or {}).get(k) is not None for k in ("confidence_threshold", "score_thresh"))


def parse_class_thresholds(text: str | None) -> tuple[float, ...]:
    """Per-class thresholds (CLASSES order) from a model's ``THRESHOLDS_KEY`` metadata; ``()`` when absent
    or invalid (warned), so the model serves at ``confidence_threshold``. A class missing from the JSON
    ends the tuple, and every class after it uses ``confidence_threshold`` too."""
    if not text:
        return ()
    try:
        picked = json.loads(text)
        out = []
        for name in CLASSES:
            if name not in picked:
                break
            value = float(picked[name])
            if not 0 <= value <= 1 or isinstance(picked[name], bool):
                raise ValueError(f"{name}: {picked[name]!r} is not in [0, 1]")
            out.append(value)
    except (TypeError, ValueError) as e:  # json.JSONDecodeError is a ValueError
        logger.warning(f"ignoring the model's confidence thresholds {text!r}: {e}")
        return ()
    return tuple(out)


def _parse_crop_m(value: Any, default: tuple[float, float]) -> tuple[float, float]:
    if value is None:
        return default
    if isinstance(value, str | bytes) or not isinstance(value, Sequence) or len(value) != 2:
        raise ValueError(f"crop_m must be a pair [lo, hi], got {value!r}")
    lo, hi = (_number("crop_m", v, float) for v in value)
    _require("crop_m", 0 < lo <= hi, "0 < lo <= hi", value)
    return (lo, hi)


def _parse_augment(value: Mapping | None) -> AugmentParams:
    aug = _as_mapping("augment", value)
    _warn_unknown(aug, set(AugmentParams.__dataclass_fields__), "augment parameter")
    d = AugmentParams()
    return AugmentParams(
        crop_m=_parse_crop_m(aug.get("crop_m"), d.crop_m),
        downscale=_bool(aug, "downscale", d.downscale),
        flips=_bool(aug, "flips", d.flips),
        rot90=_bool(aug, "rot90", d.rot90),
        colour_jitter=_bool(aug, "colour_jitter", d.colour_jitter),
    )


def _parse_zoom_weights(value: Any, default: Mapping[int, float]) -> dict[int, float]:
    if value is None:
        return dict(default)
    if not isinstance(value, Mapping):
        raise ValueError(f"zoom_weights must be a mapping, got {type(value).__name__}")
    weights: dict[int, float] = {}
    for key, weight in value.items():
        zoom = _number(f"zoom_weights key {key!r}", key, int)
        _in_range("zoom_weights key", zoom, 0, 30)
        w = _number(f"zoom_weights[{key!r}]", weight, float)
        _at_least(f"zoom_weights[{key!r}]", w, 0)
        weights[zoom] = w
    _require("zoom_weights", sum(weights.values()) > 0, "a positive sum", dict(value))
    return weights


def _parse_trainable_layers(hp: Mapping, default: int) -> int:
    if hp.get("trainable_layers") is not None:
        return _number("trainable_layers", hp["trainable_layers"], int)
    freeze = hp.get("freeze_backbone")
    if freeze is None:
        return default
    if not isinstance(freeze, bool):
        raise ValueError(f"freeze_backbone must be a bool, got {freeze!r}")
    return 0 if freeze else 3  # 3 is DeepForest's default [P]


_INT_TRAIN_FIELDS = (
    "epochs",
    "batch_size",
    "samples_per_epoch",
    "patience",
    "seed",
    "keep_last",
    "region_repeat_cap",
    "num_classes",
)
_FLOAT_TRAIN_FIELDS = (
    "learning_rate",
    "momentum",
    "weight_decay",
    "l2sp_alpha",
    "min_delta",
    "ckpt_every_min",
    "source_alpha",
    "biome_alpha",
    "empty_crop_cap",
)


def parse_train_params(hp: Mapping | None) -> TrainParams:
    hp = _as_mapping("train params", hp)
    _warn_unknown(hp, set(TrainParams.__dataclass_fields__) | {"lr", "freeze_backbone"}, "train parameter")
    d = TrainParams()
    numbers = {
        name: _field(hp, name, getattr(d, name), kind, ("lr",) if name == "learning_rate" else ())
        for kind, names in ((int, _INT_TRAIN_FIELDS), (float, _FLOAT_TRAIN_FIELDS))
        for name in names
    }
    schedule = _pick(hp, "schedule", (), d.schedule)
    _require("schedule", schedule in _SCHEDULES, f"one of {_SCHEDULES}", schedule)
    p = TrainParams(
        **numbers,
        trainable_layers=_parse_trainable_layers(hp, d.trainable_layers),
        schedule=schedule,
        amp=_bool(hp, "amp", d.amp),
        zoom_weights=_parse_zoom_weights(hp.get("zoom_weights"), d.zoom_weights),
        augment=_parse_augment(hp.get("augment")),
    )
    _at_least("epochs", p.epochs, 1)
    _at_least("learning_rate", p.learning_rate, 0, strict=True)
    _in_range("momentum", p.momentum, 0, 1, hi_open=True)
    _at_least("weight_decay", p.weight_decay, 0)
    _at_least("l2sp_alpha", p.l2sp_alpha, 0)
    _in_range("trainable_layers", p.trainable_layers, 0, 5)
    _at_least("batch_size", p.batch_size, 1)
    _at_least("patience", p.patience, 1)
    _at_least("min_delta", p.min_delta, 0)
    _at_least("ckpt_every_min", p.ckpt_every_min, 0, strict=True)
    _at_least("keep_last", p.keep_last, 1)
    _at_least("source_alpha", p.source_alpha, 0)
    _at_least("biome_alpha", p.biome_alpha, 0)
    if p.samples_per_epoch is not None:
        _at_least("samples_per_epoch", p.samples_per_epoch, 1)
    if p.region_repeat_cap is not None:
        _at_least("region_repeat_cap", p.region_repeat_cap, 1)
    if p.empty_crop_cap is not None:
        _in_range("empty_crop_cap", p.empty_crop_cap, 0, 1)
    if p.num_classes is not None:
        _in_range("num_classes", p.num_classes, 1, len(CLASSES))
    if p.l2sp_alpha > 0:
        logger.info("l2sp_alpha > 0: L2-SP replaces weight decay, so weight_decay is set to 0")
        p = replace(p, weight_decay=0.0)
    return p
