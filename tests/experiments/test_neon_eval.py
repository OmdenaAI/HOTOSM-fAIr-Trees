import json
import logging

import numpy as np
import pytest
import rasterio

pytest.importorskip("scipy")

from experiments.neon_eval import DF_COMPAT, load_eval_plots  # noqa: E402
from experiments.neon_repro import run_neon_repro  # noqa: E402
from fair_trees.serve import ort_run_fn  # noqa: E402

VOC = """<annotation><filename>{name}.tif</filename>
<size><width>400</width><height>400</height><depth>3</depth></size>
<object><name>Tree</name><bndbox><xmin>10</xmin><ymin>20</ymin><xmax>30</xmax><ymax>40</ymax></bndbox></object>
<object><name>Tree</name><bndbox><xmin>100</xmin><ymin>100</ymin><xmax>150</xmax><ymax>160</ymax></bndbox></object>
</annotation>"""


@pytest.fixture
def neon_root(tmp_path):
    (tmp_path / "annotations").mkdir()
    (tmp_path / "evaluation" / "RGB").mkdir(parents=True)
    for name in ("SITE_001", "OTHER_002"):
        (tmp_path / "annotations" / f"{name}.xml").write_text(VOC.format(name=name))
    data = np.random.default_rng(0).integers(1, 256, (3, 400, 400), dtype=np.uint8)
    with rasterio.open(
        tmp_path / "evaluation" / "RGB" / "SITE_001.tif",
        "w",
        driver="GTiff",
        height=400,
        width=400,
        count=3,
        dtype="uint8",
    ) as dst:
        dst.write(data)
    return tmp_path


def test_df_compat_params():
    assert (DF_COMPAT.window_m, DF_COMPAT.window_overlap, DF_COMPAT.edge_band) == (40.0, 0.05, 0.0)
    assert (DF_COMPAT.confidence_threshold, DF_COMPAT.nms_threshold, DF_COMPAT.iou_threshold) == (
        0.1,
        0.05,
        0.15,
    )


def test_load_eval_plots(neon_root, caplog):
    with caplog.at_level(logging.WARNING):
        units = load_eval_plots(neon_root)
    assert [u.unit_id for u in units] == ["SITE_001"]
    np.testing.assert_array_equal(units[0].gt_boxes, [[10, 20, 30, 40], [100, 100, 150, 160]])
    assert units[0].mosaic.gsd_m == 0.1 and units[0].mosaic.rgb.shape == (400, 400, 3)
    assert units[0].meta == {"source": "neon", "site": "SITE", "zoom": "native"}
    assert "OTHER_002" in caplog.text


def test_run_neon_repro_writes_eval_json(neon_root, fake_session, tmp_path):
    units = load_eval_plots(neon_root)
    out = tmp_path / "res" / "eval.json"
    payload = run_neon_repro(units, ort_run_fn(fake_session()), out, n_boot=50)
    saved = json.loads(out.read_text())
    assert saved == payload
    assert set(saved) >= {"f1", "precision", "recall", "f1_ci95", "published_f1", "pass", "units"}
    assert saved["published_f1"] == 0.719 and saved["pass"] is False  # no detections -> F1 0
