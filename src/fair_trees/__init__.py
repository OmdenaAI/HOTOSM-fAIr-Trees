"""fair_trees: DeepForest RetinaNet tree-crown detection for HOT fAIr.

The serving path (this module, params, georef, preprocess, decode, postprocess, serve) never imports
torch. Training and export live in fair_trees.train and fair_trees.export and need the `train` extra.
"""

__version__ = "0.1.0"

from fair_trees.serve import predict_session  # noqa: E402

__all__ = ["__version__", "predict_session"]
