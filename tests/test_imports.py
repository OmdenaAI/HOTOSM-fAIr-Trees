"""The serving path must never import torch, torchvision or scipy."""

import subprocess
import sys

import pytest

CODE = "import sys, {mod}; print(sorted(m for m in ('torch', 'torchvision', 'scipy') if m in sys.modules))"


@pytest.mark.parametrize("mod", ["fair_trees"])
def test_serving_imports_stay_light(mod):
    out = subprocess.run(
        [sys.executable, "-c", CODE.format(mod=mod)], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "[]"


def test_version():
    import fair_trees

    assert fair_trees.__version__ == "0.1.0.dev0"
