"""Streamlined NeurIPS multi-task training package.

Re-exports the public surface from `training.py` so callers can import via
the package name (`from training import MultiTaskConfig`) without colliding
with the `training` package vs. the inner `training.py` module name.
"""
from .training import (
    MultiTaskConfig,
    _build_parser,
    _cfg_from_args,
    _load_training_yaml,
    bridge_vit_defaults,
    evaluate,
    evaluate_export,
    load_checkpoint_for_eval,
    main,
    save_checkpoint,
    train,
)

__all__ = [
    "MultiTaskConfig",
    "_build_parser",
    "_cfg_from_args",
    "_load_training_yaml",
    "bridge_vit_defaults",
    "evaluate",
    "evaluate_export",
    "load_checkpoint_for_eval",
    "main",
    "save_checkpoint",
    "train",
]
