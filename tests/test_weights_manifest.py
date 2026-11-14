"""weights/SHA256SUMS and the DVC receipts under weights/ name the same files."""

import re
from pathlib import Path

WEIGHTS = Path(__file__).resolve().parents[1] / "weights"


def _receipt_paths() -> set[str]:
    paths = set()
    for receipt in WEIGHTS.rglob("*.dvc"):
        (out,) = re.findall(r"^\s*path: (.+)$", receipt.read_text(), flags=re.MULTILINE)
        paths.add((receipt.parent / out.strip()).relative_to(WEIGHTS).as_posix())
    return paths


def _sum_paths() -> set[str]:
    lines = (WEIGHTS / "SHA256SUMS").read_text().splitlines()
    return {line.split(maxsplit=1)[1].lstrip("*") for line in lines if line.strip()}


def test_every_receipt_has_a_sha256_and_back():
    assert _receipt_paths() == _sum_paths()


def test_the_shipped_model_and_base_are_tracked():
    assert {
        "NEON.pt",
        "hot-groups-crop80/best.pt",
        "hot-groups-crop80/model.onnx",
        "hot-groups-crop80/thresholds.json",
        "hot-groups-crop80/windows.json",
    } <= _receipt_paths()
