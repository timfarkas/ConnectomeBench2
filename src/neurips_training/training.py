"""training.py — streamlined NeurIPS multi-task training entrypoint.

NO modal infra. Plain argparse + torchrun (or mp.spawn) for DDP. This is the
released code; reviewers reproduce on any GPU box. The internal modal launcher
lives separately under `scripts/model-post-training/`.

Usage:
    # single-GPU
    python scripts/neurips/training/training.py \\
        --blend-config scripts/neurips/training/configs/full_4sp.yaml \\
        --epochs 10 --batch-size 64

    # multi-GPU (preferred — torchrun reads RANK/WORLD_SIZE/LOCAL_RANK env):
    bash scripts/neurips/training/train_ddp.sh \\
        --blend-config configs/full_4sp.yaml --batch-size 128

    # multi-GPU (mp.spawn fallback for envs without torchrun):
    python scripts/neurips/training/training.py \\
        --blend-config configs/full_4sp.yaml --batch-size 128 --world-size 2 --spawn
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

# Sibling-module imports (this package uses flat names; same convention as tests).
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import (  # noqa: E402
    BlendConfig,
    NeurIPSDataset,
    collate,
    load_blend_config,
)
from model import build_model  # noqa: E402
from training_utils import (  # noqa: E402
    build_optimizer,
    build_scheduler,
    compute_multitask_losses,
    dataloader_worker_init_fn,
    ddp_active,
    ddp_all_reduce_metrics,
    ddp_unwrap,
    make_mask_viz_grid,
    optimizer_step_amp,
    run_eval,
    set_seeds,
    setup_ddp,
    teardown_ddp,
)


# ─────────────────────────── Config ──────────────────────────────────────

@dataclass
class MultiTaskConfig:
    """Streamlined multi-task training config.

    Source-of-truth for hyperparams. Saved into checkpoints via `vars(cfg)`.
    """
    # Data
    parquet_root: str | None = None  # parent dir w/ train/, val/, test/ shard dirs
    blend_config_path: str | None = None  # yaml: parquet.path + filter + blend
    image_size: int = 224
    modalities: tuple[str, ...] = ("geom", "em")

    # Model
    model_name: str = "vit_b_16"  # vit_b_16 | vit_l_16 | vit_h_14

    # Splits / sampling
    seed: int = 42
    data_subsample_seed: int | None = None
    find_unused_parameters: bool = False

    # Training
    epochs: int = 10
    batch_size: int = 256
    global_batch_size: int | None = None
    num_workers: int = 16
    prefetch_factor: int = 4
    max_samples: int | None = None
    max_val_samples: int | None = None

    # Optim
    learning_rate: float = 1e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 5
    label_smoothing: float = 0.1
    max_grad_norm: float = 1.0
    drop_path_rate: float = 0.1
    head_dropout: float = 0.1
    backbone_lr_scale: float = 0.1
    warmup_start_factor: float = 0.1
    peak_lr_multiplier: float = 1.0

    # Multi-task losses
    mask_ce_weight: float = 0.5
    loss_scale_endpoint: float = 1.0
    loss_scale_junction: float = 1.0
    loss_scale_mask: float = 1.0
    loss_scale_synapse: float = 0.25
    amp_dtype: str = "bfloat16"  # "bfloat16" | "float16"

    # Class balance
    class_balance: bool = True

    # Init / warmstart
    init_source: str = "imagenet"  # "imagenet" | "fm_checkpoint"
    fm_checkpoint_path: str | None = None
    warmstart_checkpoint_path: str | None = None

    # Logging / ckpt
    use_wandb: bool = False
    wandb_project: str = "neurips-streamlined"
    wandb_run_name: str | None = None
    checkpoint_dir: str = "./checkpoints_neurips"
    checkpoint_name: str = "neurips_run"
    epoch_offset: int = 0

    # DDP
    world_size: int = 1

    # Device override ("auto" | "cuda" | "mps" | "cpu"). "auto" picks cuda > mps > cpu.
    device: str = "auto"


def bridge_vit_defaults(cfg: MultiTaskConfig) -> None:
    """Apply ViT-arch-specific defaults (lr/warmup/drop_path/etc) by model size.

    Mirrors the `ResNetTrainingConfig.apply_model_defaults` bridge from the
    unified script, but inlined w/o the cross-module dependency. The defaults
    track the unified script's `apply_model_defaults` for each backbone.

    When warmstart_checkpoint_path is set, warmup_epochs is treated as user-
    controlled (warmstart recipe wants longer warmup than the default 5ep).
    """
    bs = int(cfg.global_batch_size or cfg.batch_size)
    is_warmstart = cfg.warmstart_checkpoint_path is not None

    if cfg.model_name == "vit_b_16":
        ref_lr, ref_bs = 1e-4, 256
    elif cfg.model_name == "vit_l_16":
        ref_lr, ref_bs = 5e-5, 256
    elif cfg.model_name == "vit_h_14":
        ref_lr, ref_bs = 3e-5, 256
    else:
        ref_lr, ref_bs = 1e-4, 256

    cfg.learning_rate = ref_lr * math.sqrt(bs / ref_bs)
    cfg.weight_decay = 0.05
    cfg.label_smoothing = 0.1
    cfg.max_grad_norm = 1.0
    cfg.drop_path_rate = 0.1
    cfg.head_dropout = 0.1
    if not is_warmstart:
        cfg.warmup_epochs = 5


# ─────────────── Yaml + dataset wiring ───────────────────────────────────

def _load_training_yaml(path: str | Path) -> tuple[Path, BlendConfig]:
    """Parse the streamlined training yaml. Returns (parquet_root, blend_cfg)."""
    raw = yaml.safe_load(Path(path).read_text())
    parquet_root_raw = (raw.get("parquet") or {}).get("path")
    if not parquet_root_raw:
        raise ValueError(f"yaml at {path} must specify `parquet.path`")
    blend_cfg = load_blend_config(path)
    return Path(parquet_root_raw), blend_cfg


def _build_dataloaders(
    cfg: MultiTaskConfig,
    parquet_root: Path,
    blend_cfg: BlendConfig,
) -> tuple[DataLoader, DataLoader, NeurIPSDataset, NeurIPSDataset]:
    # CLI class_balance is authoritative over the yaml's setting.
    blend_cfg.class_balance = bool(cfg.class_balance)
    train_ds = NeurIPSDataset(
        parquet_root=parquet_root,
        split="train",
        blend=blend_cfg,
        modalities=cfg.modalities,
        augment=True,
        max_samples=cfg.max_samples,
        data_subsample_seed=cfg.data_subsample_seed,
        seed=cfg.seed,
    )
    val_ds = NeurIPSDataset(
        parquet_root=parquet_root,
        split="val",
        blend=blend_cfg,
        modalities=cfg.modalities,
        augment=False,
        max_samples=cfg.max_val_samples,
        data_subsample_seed=cfg.data_subsample_seed,
        seed=cfg.seed,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate,
        persistent_workers=cfg.num_workers > 0,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
        worker_init_fn=dataloader_worker_init_fn if cfg.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate,
        persistent_workers=cfg.num_workers > 0,
        worker_init_fn=dataloader_worker_init_fn if cfg.num_workers > 0 else None,
    )
    return train_loader, val_loader, train_ds, val_ds


# ─────────────────── checkpoint save / load ───────────────────────────────

def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    cfg: MultiTaskConfig,
    epoch: int,
    metrics: dict[str, float],
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": ddp_unwrap(model).state_dict(),
            "cfg": vars(cfg),
            "epoch": int(epoch),
            "metrics": dict(metrics),
        },
        str(path),
    )


def load_checkpoint_for_eval(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load `{model, cfg, epoch, metrics}`. Strips `module.` prefix from DDP saves."""
    state = torch.load(str(path), map_location="cpu", weights_only=False)
    sd = state.get("model", state)
    if any(k.startswith("module.") for k in sd):
        sd = {k.removeprefix("module."): v for k, v in sd.items()}
    return sd, state.get("cfg", {})


# ───────────────────────── train ──────────────────────────────────────────

def train(cfg: MultiTaskConfig, rank: int = 0, world_size: int = 1) -> dict[str, float]:
    """Main train loop. Single-rank or per-rank under DDP.

    Behavior preserved 1:1 with unified script's `train()`:
      - bf16 AMP (fp16 opt-in), GradScaler only for fp16
      - clip → step → update → sched.step() per-batch
      - rank-0 wandb every 20 steps with all-reduced losses
      - rank-0 eval (others at barrier); 20% subset eval is NOT implemented for
        IterableDataset (would require rebuilding the loader); we instead
        always eval the (capped) val_ds. Set `--max-val-samples` to control cost.
      - headline = nanmean of [end_corr, jn_id, mask] val metrics
      - save best.pt on improve, last.pt at final epoch
    """
    # Single-GPU path bridges defaults here. DDP path: caller bridged with
    # GLOBAL bs already; train() now sees PER-RANK bs and must NOT re-bridge.
    if world_size == 1:
        bridge_vit_defaults(cfg)

    set_seeds(cfg.seed, rank=rank)
    is_main = rank == 0
    if cfg.device == "auto":
        device = (
            torch.device(f"cuda:{rank}") if torch.cuda.is_available()
            else torch.device("mps") if torch.backends.mps.is_available()
            else torch.device("cpu")
        )
    elif cfg.device == "cuda":
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device(cfg.device)
    if is_main:
        print(f"Device: {device} (world_size={world_size}, rank={rank})", flush=True)

    # Data
    if not cfg.blend_config_path:
        raise ValueError("cfg.blend_config_path is required")
    parquet_root_yaml, blend_cfg = _load_training_yaml(cfg.blend_config_path)
    parquet_root = Path(cfg.parquet_root) if cfg.parquet_root else parquet_root_yaml
    train_loader, val_loader, train_ds, val_ds = _build_dataloaders(cfg, parquet_root, blend_cfg)
    if is_main:
        print(
            f"  train_ds: ~{len(train_ds)} items/epoch, val_ds: ~{len(val_ds)} items, "
            f"shards train/val: {len(train_ds.shards)}/{len(val_ds.shards)}",
            flush=True,
        )

    # Model
    em_enabled = "em" in cfg.modalities
    model: nn.Module = build_model(
        model_name=cfg.model_name,
        image_size=cfg.image_size,
        drop_path_rate=cfg.drop_path_rate,
        head_dropout=cfg.head_dropout,
        init_source=cfg.init_source,
        fm_checkpoint_path=cfg.fm_checkpoint_path,
        em_enabled=em_enabled,
        warmstart_checkpoint_path=cfg.warmstart_checkpoint_path,
    ).to(device)

    # Optim + sched on raw model BEFORE DDP wrap (DDP doesn't change param identity).
    optimizer = build_optimizer(
        model,
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        backbone_lr_scale=cfg.backbone_lr_scale,
        peak_lr_multiplier=cfg.peak_lr_multiplier,
    )
    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * cfg.epochs
    warmup_steps = steps_per_epoch * cfg.warmup_epochs
    scheduler = build_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        warmup_start_factor=cfg.warmup_start_factor,
    )

    # DDP wrap
    if world_size > 1:
        fup = em_enabled or cfg.find_unused_parameters
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=fup)

    # AMP
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if cfg.amp_dtype == "bfloat16" else torch.float16
    use_scaler = use_amp and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    if is_main:
        print(f"  AMP: enabled={use_amp} dtype={cfg.amp_dtype} scaler={use_scaler}", flush=True)

    # wandb (rank 0 only)
    wandb_run = None
    if cfg.use_wandb and is_main:
        import wandb
        wandb_run = wandb.init(project=cfg.wandb_project, name=cfg.wandb_run_name, config=vars(cfg))

    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    best_headline = -float("inf")
    last_metrics: dict[str, float] = {}

    for epoch in range(cfg.epochs):
        train_ds.epoch = epoch  # used by IterableDataset shard-shuffle RNG
        model.train()
        ep_running: dict[str, torch.Tensor] = {
            k: torch.zeros((), device=device)
            for k in ("L_endpoint", "L_junction", "L_mask", "L_synapse", "L_total")
        }
        n_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            x = batch["input"].to(device, non_blocking=True)
            gpu_batch = {
                "cls_label":         batch["cls_label"].to(device, non_blocking=True),
                "task_id":           batch["task_id"].to(device, non_blocking=True),
                "has_mask_gt":       batch["has_mask_gt"].to(device, non_blocking=True),
                "mask_label":        batch["mask_label"].to(device, non_blocking=True),
                "modality":          batch["modality"].to(device, non_blocking=True),
                "has_synapse_label": batch["has_synapse_label"].to(device, non_blocking=True),
                "synapse_label":     batch["synapse_label"].to(device, non_blocking=True),
            }

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                ep_logits, jn_logits, mask_logits, syn_logits = model(x, gpu_batch["modality"])
                losses = compute_multitask_losses(
                    ep_logits, jn_logits, mask_logits, syn_logits, gpu_batch,
                    label_smoothing=cfg.label_smoothing,
                    mask_ce_weight=cfg.mask_ce_weight,
                    loss_scale_endpoint=cfg.loss_scale_endpoint,
                    loss_scale_junction=cfg.loss_scale_junction,
                    loss_scale_mask=cfg.loss_scale_mask,
                    loss_scale_synapse=cfg.loss_scale_synapse,
                )
                loss = losses["L_total"]

            grad_norm = optimizer_step_amp(loss, optimizer, scaler, scheduler, model, cfg.max_grad_norm)

            for k in ep_running:
                ep_running[k] += losses[k].detach()
            n_batches += 1
            global_step += 1

            if cfg.use_wandb and (batch_idx % 20 == 0):
                # All ranks all-reduce; rank 0 logs.
                log_losses = {k: float(losses[k].detach().item()) for k in
                              ("L_total", "L_endpoint", "L_junction", "L_mask", "L_synapse")}
                if ddp_active(world_size):
                    log_losses = ddp_all_reduce_metrics(log_losses, world_size, op="avg")
                if is_main and wandb_run is not None:
                    lrs = [pg["lr"] for pg in optimizer.param_groups]
                    wandb_run.log({
                        "train/total_loss":     log_losses["L_total"],
                        "end_corr/train_loss":  log_losses["L_endpoint"],
                        "jn_id/train_loss":     log_losses["L_junction"],
                        "mask/train_loss":      log_losses["L_mask"],
                        "synapse/train_loss":   log_losses["L_synapse"],
                        "train/lr_top":         lrs[-1],
                        "train/lr_bottom":      lrs[0],
                        "train/lr_backbone":    lrs[0],
                        "train/lr_heads":       lrs[-1],
                        "train/grad_norm":      float(grad_norm),
                        "train/global_step":    global_step,
                        "train/epoch":          epoch + cfg.epoch_offset,
                    })

        # Epoch all-reduce of running losses
        if ddp_active(world_size):
            for v in ep_running.values():
                dist.all_reduce(v, op=dist.ReduceOp.SUM)
        denom = max(1, n_batches) * max(1, world_size)
        avg_losses = {k: v.item() / denom for k, v in ep_running.items()}
        if is_main:
            print(f"[epoch {epoch:3d}] " + " ".join(f"{k}={v:.4f}" for k, v in avg_losses.items()), flush=True)

        # Eval — rank 0 only; others wait at barrier.
        metrics: dict[str, float] = {}
        if is_main:
            val_ds.epoch = epoch
            metrics = run_eval(
                ddp_unwrap(model), val_loader, device,
                label_smoothing=cfg.label_smoothing,
                mask_ce_weight=cfg.mask_ce_weight,
                loss_scale_endpoint=cfg.loss_scale_endpoint,
                loss_scale_junction=cfg.loss_scale_junction,
                loss_scale_mask=cfg.loss_scale_mask,
                loss_scale_synapse=cfg.loss_scale_synapse,
            )
        if ddp_active(world_size):
            dist.barrier()

        # Pretty-print headline + per-task globals (rank 0 only). Mirrors the
        # unified script's val block so stdout carries enough signal without
        # wandb being on.
        if is_main and metrics:
            def _fmt(key: str, label: str | None = None) -> str:
                v = metrics.get(key, float("nan"))
                return "" if (isinstance(v, float) and math.isnan(v)) else f"{label or key.split('/')[-1]}={v:.3f}"
            print("  val:", flush=True)
            for tag in ("end_corr", "jn_id", "synapse"):
                parts = [
                    _fmt(f"{tag}/val_bal_acc_mean_prob", "bal_mp"),
                    _fmt(f"{tag}/val_acc_mean_prob", "acc_mp"),
                    _fmt(f"{tag}/val_loss", "loss"),
                    _fmt(f"{tag}/val_n_mean_prob", "n"),
                ]
                parts = [p for p in parts if p]
                if parts:
                    print(f"    {tag:9s}: " + " ".join(parts), flush=True)
            mask_parts = [
                _fmt("mask/val_mIoU_inv", "mIoU_inv"),
                _fmt("mask/val_IoU_A", "IoU_A"),
                _fmt("mask/val_IoU_B", "IoU_B"),
                _fmt("mask/val_loss", "loss"),
            ]
            mask_parts = [p for p in mask_parts if p]
            if mask_parts:
                print("    mask     : " + " ".join(mask_parts), flush=True)
            total = metrics.get("val/total_loss", float("nan"))
            if isinstance(total, float) and not math.isnan(total):
                print(f"    total_loss={total:.3f}", flush=True)

        last_metrics = metrics

        if is_main and cfg.use_wandb and wandb_run is not None:
            import wandb
            payload: dict[str, Any] = {
                **metrics,
                "train/epoch": epoch + cfg.epoch_offset,
                **{f"train/avg_{k}": v for k, v in avg_losses.items()},
            }
            _viz_model = ddp_unwrap(model)
            val_grid = make_mask_viz_grid(_viz_model, val_loader, device, n_max=8)
            if val_grid is not None:
                payload["mask/val_predictions"] = wandb.Image(
                    val_grid, caption="val (no aug): normals | GT | pred",
                )
            wandb_run.log(payload)

        # Headline = nanmean of 3 cls/mask val metrics
        headline_vals = [
            metrics.get("end_corr/val_bal_acc_mean_prob", float("nan")),
            metrics.get("jn_id/val_bal_acc_mean_prob", float("nan")),
            metrics.get("mask/val_mIoU_inv", float("nan")),
        ]
        headline_vals = [v for v in headline_vals if not math.isnan(v)]
        headline = float(np.mean(headline_vals)) if headline_vals else -float("inf")
        if is_main and headline > best_headline:
            best_headline = headline
            save_checkpoint(ckpt_dir / f"{cfg.checkpoint_name}_best.pt", model, cfg, epoch, metrics)

    if is_main:
        save_checkpoint(ckpt_dir / f"{cfg.checkpoint_name}_last.pt", model, cfg, cfg.epochs - 1, last_metrics)
    if is_main and cfg.use_wandb and wandb_run is not None:
        wandb_run.finish()

    return last_metrics


# ──────────────────────── _train_worker (mp.spawn / torchrun) ────────────

def _train_worker(rank: int, world_size: int, cfg: MultiTaskConfig) -> None:
    """Per-rank entrypoint. Sets up NCCL, divides batch/workers, runs train(), tears down.

    Two paths use this:
      1. mp.spawn(_train_worker, args=(world_size, cfg), nprocs=world_size)
      2. torchrun: each torchrun-spawned process calls _train_worker once,
         reading rank/world_size from env (set by torchrun).
    """
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)

    torch.set_num_threads(1)
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

    if world_size > 1:
        setup_ddp(rank=rank, world_size=world_size)
    cfg.batch_size = max(1, cfg.batch_size // max(1, world_size))
    cfg.num_workers = max(2, cfg.num_workers // max(1, world_size))
    try:
        train(cfg, rank=rank, world_size=world_size)
    finally:
        teardown_ddp()


# ──────────────────────────── evaluate ────────────────────────────────────

def evaluate(cfg: MultiTaskConfig, checkpoint_path: str | Path) -> dict[str, float]:
    """Load a checkpoint, run val eval, return metrics dict.

    Reads parquet path from the same yaml used to train (or from cfg.parquet_root).
    """
    bridge_vit_defaults(cfg)
    set_seeds(cfg.seed, rank=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parquet_root_yaml, blend_cfg = _load_training_yaml(cfg.blend_config_path)
    parquet_root = Path(cfg.parquet_root) if cfg.parquet_root else parquet_root_yaml
    _, val_loader, _, _ = _build_dataloaders(cfg, parquet_root, blend_cfg)

    sd, _ = load_checkpoint_for_eval(checkpoint_path)
    model = build_model(
        model_name=cfg.model_name,
        image_size=cfg.image_size,
        drop_path_rate=cfg.drop_path_rate,
        head_dropout=cfg.head_dropout,
        init_source=cfg.init_source,
        fm_checkpoint_path=None,
        em_enabled="em" in cfg.modalities,
    )
    model.load_state_dict(sd, strict=False)
    model = model.to(device)
    metrics = run_eval(
        model, val_loader, device,
        label_smoothing=cfg.label_smoothing,
        mask_ce_weight=cfg.mask_ce_weight,
        loss_scale_endpoint=cfg.loss_scale_endpoint,
        loss_scale_junction=cfg.loss_scale_junction,
        loss_scale_mask=cfg.loss_scale_mask,
        loss_scale_synapse=cfg.loss_scale_synapse,
    )
    return metrics


def evaluate_export(
    cfg: MultiTaskConfig, checkpoint_path: str | Path, output_path: str | Path,
) -> dict[str, float]:
    """Like `evaluate()` but also dumps raw per-slice predictions to `output_path` (.npz).

    Used for downstream calibration / slicing analysis. Output keys:
      `<head>_<slice>_{labels,probs,ops}` plus the metrics dict.
    """
    bridge_vit_defaults(cfg)
    set_seeds(cfg.seed, rank=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parquet_root_yaml, blend_cfg = _load_training_yaml(cfg.blend_config_path)
    parquet_root = Path(cfg.parquet_root) if cfg.parquet_root else parquet_root_yaml
    _, val_loader, _, _ = _build_dataloaders(cfg, parquet_root, blend_cfg)

    sd, _ = load_checkpoint_for_eval(checkpoint_path)
    model = build_model(
        model_name=cfg.model_name,
        image_size=cfg.image_size,
        drop_path_rate=cfg.drop_path_rate,
        head_dropout=cfg.head_dropout,
        init_source=cfg.init_source,
        fm_checkpoint_path=None,
        em_enabled="em" in cfg.modalities,
    )
    model.load_state_dict(sd, strict=False)
    model = model.to(device)

    metrics, raw = run_eval(
        model, val_loader, device,
        label_smoothing=cfg.label_smoothing,
        mask_ce_weight=cfg.mask_ce_weight,
        loss_scale_endpoint=cfg.loss_scale_endpoint,
        loss_scale_junction=cfg.loss_scale_junction,
        loss_scale_mask=cfg.loss_scale_mask,
        loss_scale_synapse=cfg.loss_scale_synapse,
        return_raw=True,
    )

    flat: dict[str, np.ndarray] = {}
    for head_name, head_dict in raw.items():
        for slice_key, slice_recs in head_dict.items():
            tag = f"{head_name}__{slice_key or 'global'}"
            for k, v in slice_recs.items():
                arr = np.asarray(v)
                if arr.dtype == object:
                    arr = np.array([str(x) for x in v])
                flat[f"{tag}__{k}"] = arr
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(output_path), **flat)
    return metrics


# ──────────────────────────── argparse ────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    # Data
    p.add_argument("--blend-config", required=True, help="path to training yaml (parquet + filter + blend)")
    p.add_argument("--parquet-root", default=None, help="override yaml's parquet.path")
    p.add_argument("--modalities", default="geom,em", help="comma-separated subset of {geom,em}")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--max-val-samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-subsample-seed", type=int, default=None)
    # Model
    p.add_argument("--model", dest="model_name", default="vit_b_16",
                   choices=["vit_b_16", "vit_l_16", "vit_h_14"])
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--init-source", default="imagenet", choices=["imagenet", "fm_checkpoint"])
    p.add_argument("--fm-checkpoint", dest="fm_checkpoint_path", default=None)
    p.add_argument("--warmstart", dest="warmstart_checkpoint_path", default=None)
    # Optim / schedule
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--prefetch-factor", type=int, default=4)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--warmup-start-factor", type=float, default=0.1)
    p.add_argument("--peak-lr-multiplier", type=float, default=1.0)
    p.add_argument("--epoch-offset", type=int, default=0)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--amp-dtype", default="bfloat16", choices=["bfloat16", "float16"])
    # Losses
    p.add_argument("--mask-ce-weight", type=float, default=0.5)
    p.add_argument("--loss-scale-endpoint", type=float, default=1.0)
    p.add_argument("--loss-scale-junction", type=float, default=1.0)
    p.add_argument("--loss-scale-mask",     type=float, default=1.0)
    p.add_argument("--loss-scale-synapse",  type=float, default=0.25)
    p.add_argument("--class-balance", action=argparse.BooleanOptionalAction, default=True)
    # Logging / checkpoint
    p.add_argument("--wandb", dest="use_wandb", action="store_true")
    p.add_argument("--wandb-project", default="neurips-streamlined")
    p.add_argument("--run-name", dest="wandb_run_name", default=None)
    p.add_argument("--checkpoint-dir", default="./checkpoints_neurips")
    p.add_argument("--checkpoint-name", default="neurips_run")
    # DDP
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"],
                   help="device override; 'auto' picks cuda > mps > cpu")
    p.add_argument("--world-size", type=int, default=1, help=">1 enables DDP")
    p.add_argument("--spawn", action="store_true",
                   help="use mp.spawn instead of torchrun-style env vars")
    p.add_argument("--find-unused-parameters", action="store_true")
    # Mode
    p.add_argument("--mode", default="train", choices=["train", "evaluate", "evaluate_export"])
    p.add_argument("--checkpoint", default=None, help="path to ckpt for --mode evaluate*")
    p.add_argument("--export-output", default=None, help="output .npz for --mode evaluate_export")
    return p


def _cfg_from_args(args: argparse.Namespace) -> MultiTaskConfig:
    return MultiTaskConfig(
        parquet_root=args.parquet_root,
        blend_config_path=args.blend_config,
        image_size=args.image_size,
        modalities=tuple(m.strip() for m in args.modalities.split(",") if m.strip()),
        model_name=args.model_name,
        seed=args.seed,
        data_subsample_seed=args.data_subsample_seed,
        find_unused_parameters=args.find_unused_parameters,
        epochs=args.epochs,
        batch_size=args.batch_size,
        global_batch_size=args.batch_size if args.world_size > 1 else None,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        max_samples=args.max_samples,
        max_val_samples=args.max_val_samples,
        warmup_epochs=args.warmup_epochs,
        max_grad_norm=args.max_grad_norm,
        warmup_start_factor=args.warmup_start_factor,
        peak_lr_multiplier=args.peak_lr_multiplier,
        mask_ce_weight=args.mask_ce_weight,
        loss_scale_endpoint=args.loss_scale_endpoint,
        loss_scale_junction=args.loss_scale_junction,
        loss_scale_mask=args.loss_scale_mask,
        loss_scale_synapse=args.loss_scale_synapse,
        amp_dtype=args.amp_dtype,
        class_balance=args.class_balance,
        init_source=args.init_source,
        fm_checkpoint_path=args.fm_checkpoint_path,
        warmstart_checkpoint_path=args.warmstart_checkpoint_path,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_name=args.checkpoint_name,
        epoch_offset=args.epoch_offset,
        world_size=args.world_size,
        device=args.device,
    )


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    cfg = _cfg_from_args(args)

    if args.mode == "evaluate":
        if not args.checkpoint:
            raise SystemExit("--checkpoint required for --mode evaluate")
        metrics = evaluate(cfg, args.checkpoint)
        print(metrics)
        return
    if args.mode == "evaluate_export":
        if not args.checkpoint or not args.export_output:
            raise SystemExit("--checkpoint and --export-output required for evaluate_export")
        metrics = evaluate_export(cfg, args.checkpoint, args.export_output)
        print(metrics)
        return

    # train
    world_size = max(1, args.world_size)

    # Apply CLI class_balance override on top of yaml-loaded blend config.
    # (We don't load blend here — train() does — but we want the override to
    # propagate. Done by having train() respect cfg.class_balance later via
    # blend_cfg.class_balance assignment.)

    # Detect torchrun-style env vars (set by torchrun before our process starts).
    env_world = int(os.environ.get("WORLD_SIZE", "0"))
    env_rank = int(os.environ.get("RANK", "0"))
    if env_world > 1 and not args.spawn:
        # torchrun path: this process IS one rank already. Each rank bridges
        # its OWN cfg with GLOBAL bs, then _train_worker divides per-rank.
        cfg.world_size = env_world
        bridge_vit_defaults(cfg)
        _train_worker(env_rank, env_world, cfg)
        return

    if world_size <= 1:
        train(cfg, rank=0, world_size=1)
        return

    # mp.spawn fallback — bridge once with GLOBAL bs, share cfg across ranks.
    bridge_vit_defaults(cfg)
    mp.spawn(_train_worker, args=(world_size, cfg), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
