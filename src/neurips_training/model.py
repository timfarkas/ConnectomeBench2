"""model.py — MultiTaskViT for ConnectomeBench2 streamlined training.

Single-class architecture (preserved verbatim from
`scripts/model-post-training/neurips_finetune_unified.py` lines 428-576):

  - Shared ViT-{B,L,H} backbone (torchvision) with **7-channel** patch embedding
    (silhouette / depth / nx,ny,nz / mask_a / mask_b). On ImageNet init, RGB
    weights are placed at channels (nx, ny, nz) = (2, 3, 4) — RGB is most
    analogous to a continuous 3-component normal field.
  - Separate **`conv_proj_em`** (3-channel patch embedding for EM) initialised
    from the ORIGINAL ImageNet RGB weights, snapshotted *before* the 7ch patch.
    EM samples come in as a 7ch tensor whose first 3 channels are the RGB PNG
    and channels 3-6 are zero-padded.
  - 4 heads off the shared encoder:
      * cls_endpoint — false-split correction (merge correction)
      * cls_junction — false-merge identification
      * cls_synapse  — auxiliary synapse-pair head
      * mask_head    — `ViTCNNMaskModel.decoder` (frozen ConvTranspose CNN
        upsampling 14×14 → 224×224, 2-class).
  - Per-sample modality routing in `_patchify_mixed`: geom samples go through
    `backbone.conv_proj`, em samples through `conv_proj_em`. Fast path skips
    the boolean scatter + `.any()` GPU↔CPU syncs when `em_enabled=False`.

Vendored dependencies (kept inline so the streamlined dir stands alone):
  - `_create_vit_backbone` — slim ViT-only factory from
    `scripts/model-post-training/resnet/model_factory.py::create_resnet_model`
  - `DropPath`, `_inject_drop_path`, `_make_drop_path_forward` — same source
  - `_extract_patch_tokens`, `ViTCNNMaskModel` — from
    `src/training/split_mask_training.py`
  - `_patch_conv_proj_7ch`, `MultiTaskViT` — from the unified script

NO LLRD param-group helper here (per Tim, 2026-05-07: not needed).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# Constants — these must match data.py exactly.
CH_SILH, CH_DEPTH, CH_NX, CH_NY, CH_NZ, CH_MASK_A, CH_MASK_B = range(7)
MODALITY_GEOM = 0
MODALITY_EM = 1
IMAGE_SIZE = 224
PATCH_GRID = 14   # 224 / 16


# ──────────────────────── DropPath (vendored) ─────────────────────────────

class DropPath(nn.Module):
    """Stochastic depth — randomly drop entire residual branches per-sample.

    Vendored from `scripts/model-post-training/resnet/model_factory.py`.
    """

    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep = 1 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.rand(shape, device=x.device) < keep
        return x * mask / keep


def _make_drop_path_forward(block: nn.Module, drop_path: DropPath):
    """Patch torchvision ViT EncoderBlock.forward to apply DropPath on both
    residual branches. Coupled to torchvision internals (ln_1, self_attention,
    dropout, ln_2, mlp) — fragile vs library version drift but stable across
    the torchvision releases this codebase pins.
    """
    def forward(input: torch.Tensor) -> torch.Tensor:
        torch._assert(
            input.dim() == 3,
            f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}",
        )
        x = block.ln_1(input)
        x, _ = block.self_attention(x, x, x, need_weights=False)
        x = block.dropout(x)
        x = drop_path(x) + input
        y = block.ln_2(x)
        y = block.mlp(y)
        return x + drop_path(y)
    return forward


def _inject_drop_path(model: nn.Module, drop_path_rate: float) -> None:
    """Inject DropPath into all ViT encoder blocks with linearly increasing rates."""
    blocks = list(model.encoder.layers)
    n = len(blocks)
    rates = [drop_path_rate * i / (n - 1) for i in range(n)] if n > 1 else [drop_path_rate]
    for i, block in enumerate(blocks):
        dp = DropPath(rates[i])
        block.add_module("drop_path", dp)
        block.forward = _make_drop_path_forward(block, dp)


# ───────────────── ViT backbone factory (vendored, slim) ──────────────────

_VIT_CONFIGS: dict[str, tuple[Any, str]] = {
    # name → (model_fn, weights_attr); resolved lazily so importing model.py
    # doesn't drag torchvision into the import graph.
    "vit_b_16": ("vit_b_16", "ViT_B_16_Weights.IMAGENET1K_V1"),
    "vit_l_16": ("vit_l_16", "ViT_L_16_Weights.IMAGENET1K_V1"),
    "vit_h_14": ("vit_h_14", "ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1"),
}


def _create_vit_backbone(
    model_name: str,
    pretrained: bool = True,
    image_size: int = IMAGE_SIZE,
    drop_path_rate: float = 0.0,
) -> nn.Module:
    """Create a torchvision ViT backbone with classifier head stripped.

    Slim ViT-only adaptation of `resnet.model_factory.create_resnet_model`.
    Returns a model where `.heads = Identity()` already; caller patches conv_proj.

    Behaviour preserved from the unified script's call site:
        create_resnet_model(model_name, num_classes=2, pretrained=imgnet,
                            freeze_backbone=False, image_size=224,
                            drop_path_rate=0.1, head_dropout=0.0,
                            unfreeze_last_n=0)
    """
    if model_name not in _VIT_CONFIGS:
        raise ValueError(
            f"unsupported backbone {model_name!r} "
            f"(supported: {sorted(_VIT_CONFIGS)})"
        )
    from torchvision import models
    from torchvision.models import (
        ViT_B_16_Weights, ViT_L_16_Weights, ViT_H_14_Weights,
    )

    weight_map = {
        "vit_b_16": ViT_B_16_Weights.IMAGENET1K_V1,
        "vit_l_16": ViT_L_16_Weights.IMAGENET1K_V1,
        "vit_h_14": ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1,
    }
    model_fn = getattr(models, model_name)
    weights = weight_map[model_name]

    # Pretrained ViTs ship for image_size=224. Non-224 input requires position-
    # embedding interpolation (bicubic over a square grid). The unified script
    # always uses 224, so this branch is dormant in production — kept for parity.
    if pretrained and image_size != IMAGE_SIZE:
        pretrained_model = model_fn(weights=weights)
        model = model_fn(weights=None, image_size=image_size)
        pre_state = pretrained_model.state_dict()
        cur_state = model.state_dict()
        for key in pre_state:
            if "pos_embedding" not in key:
                if key in cur_state and pre_state[key].shape == cur_state[key].shape:
                    cur_state[key] = pre_state[key]
        pos_key = "encoder.pos_embedding"
        if pos_key in pre_state:
            old_pos = pre_state[pos_key]
            new_seq_len = cur_state[pos_key].shape[1]
            old_seq_len = old_pos.shape[1]
            if old_seq_len != new_seq_len:
                cls_tok = old_pos[:, :1, :]
                patch_pos = old_pos[:, 1:, :]
                old_grid = int(patch_pos.shape[1] ** 0.5)
                new_grid = int((new_seq_len - 1) ** 0.5)
                patch_pos = patch_pos.reshape(1, old_grid, old_grid, -1).permute(0, 3, 1, 2)
                patch_pos = F.interpolate(
                    patch_pos, size=(new_grid, new_grid),
                    mode="bicubic", align_corners=False,
                )
                patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, new_grid * new_grid, -1)
                cur_state[pos_key] = torch.cat([cls_tok, patch_pos], dim=1)
            else:
                cur_state[pos_key] = old_pos
        model.load_state_dict(cur_state)
        del pretrained_model, pre_state
    elif pretrained:
        model = model_fn(weights=weights)
    else:
        kw = {"image_size": image_size} if image_size != IMAGE_SIZE else {}
        model = model_fn(weights=None, **kw)

    if drop_path_rate > 0:
        _inject_drop_path(model, drop_path_rate)

    # Strip default classifier head — MultiTaskViT owns its own heads.
    if hasattr(model, "heads"):
        model.heads = nn.Identity()
    return model


# ────────────────────── Mask decoder (vendored) ───────────────────────────

def _extract_patch_tokens(backbone: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Patch tokens (no CLS) from torchvision ViT. → (B, N_patch, D).

    Vendored from `src/training/split_mask_training.py`.
    """
    x = backbone._process_input(images)
    n = x.shape[0]
    cls = backbone.class_token.expand(n, -1, -1)
    x = torch.cat([cls, x], dim=1)
    x = backbone.encoder(x)
    return x[:, 1:]


class ViTCNNMaskModel(nn.Module):
    """ConvTranspose CNN decoder over reshaped ViT patch tokens → 2-class mask.

    Vendored from `src/training/split_mask_training.py`. In MultiTaskViT we
    only use `.decoder` (not `.forward`); the optional `view_emb` is kept for
    state-dict compatibility with prior split-mask checkpoints.
    """

    def __init__(self, backbone: nn.Module, num_classes: int = 3, embed_dim: int = 768):
        super().__init__()
        self.backbone = backbone
        self.view_emb = nn.Embedding(3, embed_dim)  # front=0, side=1, top=2
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 256, kernel_size=2, stride=2),  # 14→28
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),        # 28→56
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128,  64, kernel_size=2, stride=2),        # 56→112
            nn.BatchNorm2d(64),  nn.ReLU(inplace=True),
            nn.ConvTranspose2d( 64, num_classes, kernel_size=2, stride=2),  # 112→224
        )

    def forward(
        self, x: torch.Tensor, view_idx: torch.Tensor | None = None
    ) -> torch.Tensor:
        backbone_trainable = any(p.requires_grad for p in self.backbone.parameters())
        ctx = torch.enable_grad() if backbone_trainable else torch.no_grad()
        with ctx:
            tokens = _extract_patch_tokens(self.backbone, x)
        B, N, D = tokens.shape
        h = w = int(N ** 0.5)
        feat = tokens.permute(0, 2, 1).reshape(B, D, h, w)
        if view_idx is not None:
            feat = feat + self.view_emb(view_idx)[:, :, None, None]
        return self.decoder(feat)


# ───────────────────────── 7ch conv_proj patch ────────────────────────────

def _patch_conv_proj_7ch(backbone: nn.Module, init_source: str = "imagenet") -> None:
    """Replace torchvision ViT's 3ch `conv_proj` with a 7ch equivalent.

    ImageNet init: copy RGB weights into channels (CH_NX, CH_NY, CH_NZ) — normals
    are the 3-component continuous signal most analogous to RGB. The other 4
    channels (silhouette, depth, mask_a, mask_b) are kaiming-normal init.

    GOTCHA: a naive first-3 copy would put RGB into (silhouette, depth, nx),
    which is wrong — fight the urge.

    Vendored verbatim from the unified script. MUST be called after
    `_create_vit_backbone` (since we depend on the post-init weights).
    """
    old = backbone.conv_proj
    assert isinstance(old, nn.Conv2d), f"expected Conv2d, got {type(old)}"
    out_ch = old.out_channels
    k = old.kernel_size
    s = old.stride

    new = nn.Conv2d(7, out_ch, kernel_size=k, stride=s)
    with torch.no_grad():
        nn.init.kaiming_normal_(new.weight, mode="fan_out", nonlinearity="linear")
        if new.bias is not None and old.bias is not None:
            new.bias.copy_(old.bias)
        if init_source == "imagenet":
            # old.weight: (out_ch, 3, kH, kW). Map RGB → (nx, ny, nz)
            new.weight[:, CH_NX:CH_NX + 3] = old.weight
    backbone.conv_proj = new


# ───────────────────────────── MultiTaskViT ───────────────────────────────

class MultiTaskViT(nn.Module):
    """Multi-task ViT: shared backbone, dual-modality patchifier, 4 heads.

    Verbatim port of `MultiTaskViT` from `neurips_finetune_unified.py`
    (lines 428–576). Forward signature, return tuple, modality routing,
    and conv_proj_em ImageNet init are all preserved bit-for-bit.

    Args:
        model_name: vit_b_16 | vit_l_16 | vit_h_14
        image_size: input H/W (default 224).
        drop_path_rate: stochastic depth rate (linear schedule across blocks).
        head_dropout: dropout before the linear of each cls head.
        init_source: "imagenet" or "fm_checkpoint". For "fm_checkpoint",
            the caller must invoke `_load_fm_checkpoint` after construction
            (or pass `fm_checkpoint_path` here).
        fm_checkpoint_path: required iff init_source="fm_checkpoint".
        em_enabled: if False, freeze conv_proj_em (DDP-friendly geom-only mode).
    """

    def __init__(
        self,
        model_name: str = "vit_b_16",
        image_size: int = IMAGE_SIZE,
        drop_path_rate: float = 0.1,
        head_dropout: float = 0.1,
        init_source: str = "imagenet",
        fm_checkpoint_path: str | None = None,
        em_enabled: bool = True,
    ) -> None:
        super().__init__()
        backbone = _create_vit_backbone(
            model_name=model_name,
            pretrained=(init_source == "imagenet"),
            image_size=image_size,
            drop_path_rate=drop_path_rate,
        )

        # Snapshot original 3ch conv_proj weights (ImageNet-trained) BEFORE
        # patching to 7ch — we'll reuse them for conv_proj_em init below.
        # ⚠️ CRITICAL ORDERING: this clone MUST happen pre-patch.
        orig_conv_proj = backbone.conv_proj
        orig_weight = orig_conv_proj.weight.detach().clone() if hasattr(orig_conv_proj, "weight") else None
        orig_bias = (
            orig_conv_proj.bias.detach().clone()
            if getattr(orig_conv_proj, "bias", None) is not None
            else None
        )

        # 7ch conv_proj (must happen AFTER torchvision ViT is built at 3ch).
        _patch_conv_proj_7ch(backbone, init_source=init_source)

        # Infer embed dim from CLS token — avoids hardcoding per backbone.
        embed_dim = int(backbone.class_token.shape[-1])

        # EM 3-channel conv_proj, init from original ImageNet weights.
        # Same kernel/stride as the (now-patched) geom conv_proj.
        k = backbone.conv_proj.kernel_size
        s = backbone.conv_proj.stride
        self.conv_proj_em = nn.Conv2d(3, embed_dim, kernel_size=k, stride=s)
        if (
            init_source == "imagenet"
            and orig_weight is not None
            and orig_weight.shape[1] == 3
        ):
            with torch.no_grad():
                self.conv_proj_em.weight.copy_(orig_weight)
                if orig_bias is not None and self.conv_proj_em.bias is not None:
                    self.conv_proj_em.bias.copy_(orig_bias)

        self.backbone = backbone
        self.image_size = image_size
        self.embed_dim = embed_dim
        self.em_enabled = em_enabled
        if not em_enabled:
            # Freeze conv_proj_em params — removes them from DDP's reducer
            # (only requires_grad=True params get grad-sync hooks), unblocking
            # find_unused_parameters=False for geom-only runs.
            for p in self.conv_proj_em.parameters():
                p.requires_grad = False

        self.cls_endpoint = nn.Sequential(nn.Dropout(head_dropout), nn.Linear(embed_dim, 2))
        self.cls_junction = nn.Sequential(nn.Dropout(head_dropout), nn.Linear(embed_dim, 2))
        self.cls_synapse = nn.Sequential(nn.Dropout(head_dropout), nn.Linear(embed_dim, 2))
        self.mask_head = ViTCNNMaskModel(
            backbone=self.backbone, num_classes=2, embed_dim=embed_dim
        )

        if init_source == "fm_checkpoint":
            assert fm_checkpoint_path is not None, (
                "init_source=fm_checkpoint requires fm_checkpoint_path"
            )
            self._load_fm_checkpoint(fm_checkpoint_path)

    def _load_fm_checkpoint(self, path: str) -> None:
        state = torch.load(path, map_location="cpu", weights_only=False)
        sd = state.get("model", state)
        missing, unexpected = self.backbone.load_state_dict(sd, strict=False)
        print(f"  FM checkpoint loaded: missing={len(missing)}, unexpected={len(unexpected)}")

    def _patchify_mixed(self, x: torch.Tensor, modality: torch.Tensor) -> torch.Tensor:
        """Apply correct conv_proj per sample → patch-token sequence (B, N_patch, D).

        x: (B, 7, H, W). For EM samples, channels [0:3] hold the RGB PNG and
           channels [3:7] are zero-padded.
        modality: (B,) long, 0=geom, 1=em.

        Fast path: when em_enabled=False, all samples are geom by data-side
        construction — skip the boolean scatter + `.any()` (each is a CUDA→CPU
        sync that costs us 5-15% step time on H100s).
        """
        if not self.em_enabled:
            patches = self.backbone.conv_proj(x)
            return patches.flatten(2).transpose(1, 2)

        B, _, H, W = x.shape
        is_em = modality == MODALITY_EM
        is_geom = ~is_em

        out_shape = None
        patches_em: torch.Tensor | None = None
        patches_geom: torch.Tensor | None = None

        if is_em.any():
            em_in = x[is_em, :3]
            patches_em = self.conv_proj_em(em_in)              # (B_em, D, H/16, W/16)
            out_shape = patches_em.shape[2:]
        if is_geom.any():
            patches_geom = self.backbone.conv_proj(x[is_geom])  # (B_geom, D, H/16, W/16)
            out_shape = patches_geom.shape[2:]

        assert out_shape is not None, "empty batch"
        Hp, Wp = int(out_shape[0]), int(out_shape[1])
        D = self.embed_dim
        # dtype must match the conv output (may be fp16 under autocast, fp32 otherwise).
        out_dtype = patches_em.dtype if patches_em is not None else patches_geom.dtype

        full = torch.empty(B, D, Hp, Wp, device=x.device, dtype=out_dtype)
        if patches_em is not None:
            full[is_em] = patches_em
        if patches_geom is not None:
            full[is_geom] = patches_geom
        return full.flatten(2).transpose(1, 2)

    def _encode(
        self, x: torch.Tensor, modality: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (cls_token, patch_tokens). Mirrors torchvision ViT._process_input + encoder."""
        patches = self._patchify_mixed(x, modality)
        B = patches.shape[0]
        cls = self.backbone.class_token.expand(B, -1, -1)
        seq = torch.cat([cls, patches], dim=1)
        seq = self.backbone.encoder(seq)
        return seq[:, 0], seq[:, 1:]

    def forward(
        self, x: torch.Tensor, modality: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """x: (B, 7, H, W). modality: (B,) long (0=geom, 1=em).

        Returns:
            (endpoint_logits, junction_logits, mask_logits, synapse_logits)
            shapes (B,2), (B,2), (B,2,h,w), (B,2)
        """
        cls_tok, patch_tok = self._encode(x, modality)
        B, N, D = patch_tok.shape
        h = w = int(math.isqrt(N))
        endpoint_logits = self.cls_endpoint(cls_tok)
        junction_logits = self.cls_junction(cls_tok)
        synapse_logits = self.cls_synapse(cls_tok)
        feat = patch_tok.permute(0, 2, 1).reshape(B, D, h, w)
        mask_logits = self.mask_head.decoder(feat)
        return endpoint_logits, junction_logits, mask_logits, synapse_logits


# ───────────────────────────── Factory ────────────────────────────────────

def build_model(
    model_name: str = "vit_b_16",
    image_size: int = IMAGE_SIZE,
    drop_path_rate: float = 0.1,
    head_dropout: float = 0.1,
    init_source: str = "imagenet",
    fm_checkpoint_path: str | None = None,
    em_enabled: bool = True,
    warmstart_checkpoint_path: str | None = None,
) -> MultiTaskViT:
    """Build a MultiTaskViT and optionally hot-start from a prior best.pt.

    The order of operations matters:
      1. Construct backbone (ImageNet weights or random; FM checkpoint loaded
         into backbone here if init_source="fm_checkpoint").
      2. Patch conv_proj to 7ch.
      3. Construct conv_proj_em from snapshotted ImageNet weights.
      4. Attach 4 heads.
      5. (Optional) Load warmstart checkpoint over the WHOLE model state
         (backbone + conv_proj_em + heads + mask_head).

    Warmstart load is `strict=False` — useful when warmstarting from a
    checkpoint with a slightly different head topology (e.g. an older run
    without cls_synapse).
    """
    model = MultiTaskViT(
        model_name=model_name,
        image_size=image_size,
        drop_path_rate=drop_path_rate,
        head_dropout=head_dropout,
        init_source=init_source,
        fm_checkpoint_path=fm_checkpoint_path,
        em_enabled=em_enabled,
    )
    if warmstart_checkpoint_path is not None:
        load_warmstart_checkpoint(model, warmstart_checkpoint_path)
    return model


def load_warmstart_checkpoint(model: nn.Module, path: str | Path) -> dict[str, Any]:
    """Load a prior best.pt over `model` and return the saved metadata dict.

    Checkpoint format expected (matches unified-script `train()` save):
        {"model": state_dict, "cfg": dict, "epoch": int, "metrics": dict}
    """
    state = torch.load(str(path), map_location="cpu", weights_only=False)
    sd = state.get("model", state)
    # Strip module. prefix from DDP-saved checkpoints.
    if any(k.startswith("module.") for k in sd):
        sd = {k.removeprefix("module."): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(
        f"  warmstart loaded from {path}: "
        f"missing={len(missing)}, unexpected={len(unexpected)}"
    )
    return {
        "epoch": state.get("epoch"),
        "metrics": state.get("metrics", {}),
        "cfg": state.get("cfg", {}),
    }


__all__ = [
    "MultiTaskViT",
    "ViTCNNMaskModel",
    "DropPath",
    "build_model",
    "load_warmstart_checkpoint",
    "_patch_conv_proj_7ch",
    "_create_vit_backbone",
    "_extract_patch_tokens",
    "CH_SILH", "CH_DEPTH", "CH_NX", "CH_NY", "CH_NZ", "CH_MASK_A", "CH_MASK_B",
    "MODALITY_GEOM", "MODALITY_EM",
    "IMAGE_SIZE", "PATCH_GRID",
]
