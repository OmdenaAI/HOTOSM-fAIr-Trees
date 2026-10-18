import pytest

pytest.importorskip("torch")

from experiments.throughput import measure_loader  # noqa: E402
from fair_trees.dataset import load_regions  # noqa: E402
from fair_trees.params import TrainParams  # noqa: E402


def test_measure_loader(write_region):
    d = write_region("r", tiles=((0, 0), (1, 0)), boxes_px=[(30, 30, 60, 70)])
    rates = measure_loader(load_regions(d.parent), TrainParams(batch_size=2), workers=(0,), n_batches=2)
    assert set(rates) == {0} and rates[0] > 0
