from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
WEIGHTS = REPO / "weights" / "NEON.pt"
SAMPLE = REPO / "tests" / "data_local" / "OSBS_029.tif"


@pytest.fixture(scope="session")
def neon_model():
    pytest.importorskip("torch")
    if not WEIGHTS.exists():
        pytest.fail(f"{WEIGHTS} missing: see experiments/DATA_SOURCES.md")
    from fair_trees.train import build_model

    return build_model(WEIGHTS, trainable_layers=3).eval()


@pytest.fixture(scope="session")
def sample_path():
    return SAMPLE


@pytest.fixture(scope="session")
def sample_rgb():
    if not SAMPLE.exists():
        pytest.fail(f"{SAMPLE} missing: see experiments/DATA_SOURCES.md")
    import rasterio

    with rasterio.open(SAMPLE) as ds:
        return np.ascontiguousarray(np.moveaxis(ds.read([1, 2, 3]), 0, -1))
