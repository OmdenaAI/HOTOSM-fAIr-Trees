"""Raw-head ONNX export: backbone, FPN and heads only (no anchors, decoding or NMS)."""

import argparse
import io
import warnings
from collections.abc import Sequence
from pathlib import Path

import onnx
import torch
from torch import nn
from torchvision.models.detection import RetinaNet

from fair_trees.params import INPUT_NAME, OUTPUT_NAMES
from fair_trees.train import build_model

OPSET = 17


class _RawHead(nn.Module):
    """Backbone + head returning `(cls_logits, bbox_regression)` tensors."""

    def __init__(self, model: RetinaNet) -> None:
        super().__init__()
        self.backbone = model.backbone
        self.head = model.head

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.head(list(self.backbone(images).values()))
        return out["cls_logits"], out["bbox_regression"]


def export_onnx_bytes(model: RetinaNet, *, opset: int = OPSET) -> bytes:
    """Export the raw network with dynamic H, W and anchor count; returns the ONNX file bytes."""
    wrapper = _RawHead(model).eval().cpu()
    dummy = torch.zeros(1, 3, 800, 800)
    buf = io.BytesIO()
    with warnings.catch_warnings():
        # The legacy TorchScript exporter is chosen on purpose (dynamo=False); silence only its
        # own deprecation notices.
        warnings.filterwarnings("ignore", message="You are using the legacy TorchScript-based ONNX")
        warnings.filterwarnings("ignore", message="The feature will be removed")
        torch.onnx.export(
            wrapper,
            (dummy,),
            buf,
            input_names=[INPUT_NAME],
            output_names=list(OUTPUT_NAMES),
            dynamic_axes={
                INPUT_NAME: {2: "height", 3: "width"},
                "cls_logits": {1: "anchors"},
                "bbox_regression": {1: "anchors"},
            },
            opset_version=opset,
            dynamo=False,
        )
    data = buf.getvalue()
    onnx.checker.check_model(onnx.load_from_string(data))
    return data


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export a frozen-BN RetinaNet checkpoint to ONNX.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args(argv)
    model = build_model(args.checkpoint, trainable_layers=0)
    data = export_onnx_bytes(model)
    args.out.write_bytes(data)
    print(f"{args.out} ({len(data)} bytes)")


if __name__ == "__main__":
    main()
