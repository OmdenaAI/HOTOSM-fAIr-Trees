"""NeonTreeEvaluation eval-plot loader and the DeepForest-compatible inference settings."""

import logging
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import rasterio

from fair_trees.evaluate import EvalUnit
from fair_trees.params import InferenceParams
from fair_trees.preprocess import array_mosaic

logger = logging.getLogger(__name__)

# DeepForest's score_thresh 0.1, nms_thresh 0.05, predict_tile iou_threshold 0.15, and patch_size 400 at 10 cm
# (= 40 m windows) [V: confirmed against DeepForest 2.1.0 by the predict_tile parity test].
DF_COMPAT = InferenceParams(
    window_m=40.0,
    window_overlap=0.05,
    confidence_threshold=0.1,
    nms_threshold=0.05,
    iou_threshold=0.15,
    edge_band=0.0,
    min_valid_fraction=0.0,
)

ANNOTATIONS_DIR = Path("annotations")
RGB_DIR = Path("evaluation") / "RGB"
NEON_GSD_M = 0.1


def _read_boxes(xml_path: Path) -> np.ndarray:
    boxes = [
        [float(obj.findtext(f"bndbox/{k}")) for k in ("xmin", "ymin", "xmax", "ymax")]
        for obj in ET.parse(xml_path).getroot().iter("object")
    ]
    return np.asarray(boxes, np.float32).reshape(-1, 4)


def load_eval_plots(root: Path) -> list[EvalUnit]:
    """One EvalUnit per annotated plot with an RGB image; plots without one are skipped with a warning."""
    units = []
    for xml_path in sorted((root / ANNOTATIONS_DIR).glob("*.xml")):
        stem = xml_path.stem
        image = root / RGB_DIR / f"{stem}.tif"
        if not image.exists():
            logger.warning("no RGB image for annotated plot %s, skipping", stem)
            continue
        with rasterio.open(image) as src:
            rgb = np.ascontiguousarray(src.read([1, 2, 3]).transpose(1, 2, 0))
        meta = {"source": "neon", "site": stem.split("_")[0], "zoom": "native"}
        units.append(EvalUnit(stem, array_mosaic(rgb, gsd_m=NEON_GSD_M), _read_boxes(xml_path), None, meta))
    return units
