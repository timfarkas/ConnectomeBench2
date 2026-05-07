"""data.py — NeurIPS streamlined training data layer.

Loads ConnectomeBench2 from HuggingFace-format parquet shards (per
`scripts/neurips/build_parquet_dataset.py` schema, also published as
`jeffbbrown2/connectomebench2` on HF Hub).

Schema (from the HF dataset card):
  - identifier cols: combined_sample_hash, source_archive_sample_hash,
                     source_archive
  - sample identity: sample_type, same_neuron, species, has_single_mask,
                     has_dual_mask, has_em, present_slots, metadata
  - routing/labels:  task_routing, false_split_correction_label,
                     false_merge_identification_label, split
  - image bytes:     geometry, geometry_single (compressed npz; arr_0 →
                     (3, 7, 224, 224) float16), em_xy/em_xz/em_yz/em_best
                     (HF Image struct {bytes, path}; (224, 224, 3) uint8 PNG)

Loading model
-------------
This is an `IterableDataset`. The full dataset is ~280 GB so we never
hold image bytes in memory; instead we iterate shards (`{split}-NNNNN.parquet`)
sequentially. Per epoch:

  1. Shuffle shard order (deterministic on (seed, epoch)).
  2. Each worker takes shards round-robin (`shards[worker_id::n_workers]`).
  3. Each shard is read row-group by row-group sequentially — never seeking
     to specific rows. Optional intra-shard shuffle within each row group.
  4. Per row, a per-cell *blend* weight (computed once at __init__ from the
     cheap metadata-only scan) drives an accept/reject roll. Accepted rows
     yield one item per surviving view (3 geom + up to 4 EM).

Routing semantics (preserves unified script `_route_item` 1:1):
  task_routing                        head        cls_label                          has_mask_gt
  ----------------------------------  --------    -------------------------------    -----------
  false_split_correction              endpoint    false_split_correction_label       False
  false_merge_identification +
    split_mask_generation             junction    false_merge_identification_label   has_dual_mask
  false_merge_identification only     junction    false_merge_identification_label   False

Per-row geometry input selection:
  - junction-routed → geometry_single (single-mesh, leak-free input)
  - endpoint-routed → geometry         (dual-mesh)
  - mask GT (when has_mask_gt) ALWAYS read from geometry ch5,6 (dual-mesh).
  - On overlap (mask_a & mask_b), the GT pixel is 0 (bg) — matches unified
    script convention. The invariant mask loss treats A↔B as swappable.
"""
from __future__ import annotations

import io
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import IterableDataset, get_worker_info

# ───────────────────────────── Constants ──────────────────────────────────

IMAGE_SIZE = 224
NUM_GEOM_VIEWS = 3
NUM_GEOM_CHANNELS = 7

# Geometry channel layout (must match unified script + renderer)
CH_SILH, CH_DEPTH, CH_NX, CH_NY, CH_NZ, CH_MASK_A, CH_MASK_B = range(7)

# EM PNG channel layout (R/G/B)
CH_EM_GREY, CH_EM_MASK_A, CH_EM_MASK_B = 0, 1, 2

# Task IDs
TASK_ENDPOINT = 0
TASK_JUNCTION = 1

# Modality IDs
MODALITY_GEOM = 0
MODALITY_EM = 1

# EM view suffixes (4 views per op when present)
EM_VIEWS = ("em_xy", "em_xz", "em_yz", "em_best")

# Sample type → kind (legacy 3-cell axis: edits / junction / synapse)
SAMPLE_TYPE_TO_KIND: dict[str, str] = {
    "merge_edit":        "edits",
    "split_edit":        "edits",
    "adjacent_control":  "edits",
    "junction_control":  "junction",
    "synapse_control":   "synapse",
}

# HF readme uses "validation" for the val split; pipeline output uses "val".
SPLIT_ALIASES: dict[str, str] = {
    "train":      "train",
    "val":        "val",
    "validation": "val",
    "test":       "test",
}

VALID_BLEND_STRATEGIES = (
    "natural", "uniform_species", "uniform_type", "uniform_cell", "sqrt", "custom",
)

# Columns we read at init time for blend-weight + filter computation. These
# are all small (KB per row) — image bytes (geometry, em_*) are NOT touched
# until iteration. pyarrow column projection skips the heavy bytes columns.
METADATA_COLUMNS: tuple[str, ...] = (
    "combined_sample_hash",
    "source_archive_sample_hash",
    "source_archive",
    "sample_type",
    "same_neuron",
    "species",
    "has_single_mask",
    "has_dual_mask",
    "has_em",
    "task_routing",
    "false_split_correction_label",
    "false_merge_identification_label",
)


# ───────────────────────────── Blend config ───────────────────────────────

@dataclass
class BlendConfig:
    """Sampling blend across (species, kind) cells.

    Strategies:
      - natural          per-item uniform (cell weight = 1/count)
      - uniform_species  equalise mass per species
      - uniform_type     equalise mass per kind
      - uniform_cell     equalise mass per (species, kind) cell
      - sqrt             archive_count^0.5 / total_sqrt — Jeff's default
      - custom           start from natural, expect manual `weights` overrides
    """
    strategy: str = "sqrt"
    cell_axis: str = "kind"  # "kind" (3-cell legacy) | "sample_type" (5-cell)
    weights: dict[tuple[str, str], float] = field(default_factory=dict)
    include_species: list[str] | None = None
    include_sample_types: list[str] | None = None
    class_balance: bool = True


def load_blend_config(yaml_path: str | Path) -> BlendConfig:
    """Parse a streamlined blend yaml. Schema is in
    plans/2026-05-06_streamlined_neurips_training.md §6."""
    import yaml
    raw = yaml.safe_load(Path(yaml_path).read_text())
    blend_raw = raw.get("blend", {}) or {}
    filt_raw = raw.get("filter", {}) or {}

    strategy = blend_raw.get("strategy", "sqrt")
    if strategy not in VALID_BLEND_STRATEGIES:
        raise ValueError(f"unknown blend strategy: {strategy!r}")
    cell_axis = blend_raw.get("cell_axis", "kind")
    if cell_axis not in ("kind", "sample_type"):
        raise ValueError(f"unknown cell_axis: {cell_axis!r}")

    weights: dict[tuple[str, str], float] = {}
    for w in blend_raw.get("weights", []) or []:
        weights[(w["species"], w["cell"])] = float(w["weight"])

    return BlendConfig(
        strategy=strategy,
        cell_axis=cell_axis,
        weights=weights,
        include_species=filt_raw.get("include_species"),
        include_sample_types=filt_raw.get("include_sample_types"),
        class_balance=bool(raw.get("class_balance", True)),
    )


# ───────────────────────────── _Item ──────────────────────────────────────

@dataclass(slots=True)
class _Item:
    row_idx: int          # local row index within the iterating shard's row group
    view_idx: int         # 0..2 for geom views, 0..3 for em_xy/xz/yz/best
    op_id: str            # source_archive_sample_hash (per-op identifier)
    task_id: int          # 0=endpoint, 1=junction
    cls_label: int        # 0/1
    has_mask_gt: bool     # True only when split_mask_generation in routing AND has_dual_mask
    species: str
    sample_type: str
    kind: str             # edits|junction|synapse (derived)
    same_neuron: bool
    modality: int         # 0=geom, 1=em
    em_view_suffix: str   # "" for geom; "em_xy"/"em_xz"/... for em
    has_synapse_label: bool
    synapse_label: int    # 0/1


# ──────────────────────── Routing (column lookups) ────────────────────────

def route_row(row: pd.Series) -> tuple[int, int, bool] | None:
    """Map a parquet row → (task_id, cls_label, has_mask_gt) or None.

    Preserves the dispatch semantics of unified-script `_route_item` exactly:

      - false_merge_identification (split_edit / junction_control)
            → TASK_JUNCTION, label = false_merge_identification_label  (= !same_neuron)
            has_mask_gt = ("split_mask_generation" in routing)
      - false_split_correction (merge_edit / synapse_control / adjacent_control)
            → TASK_ENDPOINT, label = false_split_correction_label  (= same_neuron)
            has_mask_gt = False
      - any row not in either routing → None (no cls supervision; row is dropped)
    """
    routing = list(row["task_routing"])
    if "false_merge_identification" in routing:
        return (
            TASK_JUNCTION,
            int(row["false_merge_identification_label"]),
            "split_mask_generation" in routing,
        )
    if "false_split_correction" in routing:
        return (
            TASK_ENDPOINT,
            int(row["false_split_correction_label"]),
            False,
        )
    return None


def synapse_label_from_row(row: pd.Series) -> tuple[bool, int]:
    """(has_synapse_label, synapse_label) — preserves unified-script `_synapse_label`.

      synapse_control                     → (True, 1)   # positive synapse pair
      merge_edit (same_neuron implied T)  → (True, 0)   # negative: pair collapses to one neuron
      else                                → (False, 0)  # undefined; gated out
    """
    st = row["sample_type"]
    if st == "synapse_control":
        return (True, 1)
    if st == "merge_edit" and bool(row["same_neuron"]):
        return (True, 0)
    return (False, 0)


# ───────────────────────────── Dataset ────────────────────────────────────

class NeurIPSDataset(IterableDataset):
    """Streaming dataset over HF parquet shards. One yielded sample = one (row, view).

    Args:
        parquet_root: directory containing `{split}/{split}-NNNNN.parquet`
            (output of `scripts/neurips/build_parquet_dataset.py`, or a
            `huggingface_hub.snapshot_download(...)` of an HF dataset repo).
        split: "train" | "val" | "validation" | "test". Both "val" and the
            HF-canonical "validation" resolve to the same shard directory.
        blend: `BlendConfig` controlling include filters + per-cell sampling
            weights + class balance fold-in.
        modalities: subset of ("geom", "em").
        augment: enable CPU-side augs in __getitem__.
        max_samples: cap on distinct rows BEFORE view expansion.
        data_subsample_seed: if set, shuffle metadata rows before
            applying max_samples.
        seed: RNG seed for shard / row shuffling.
        shuffle_within_shard: if True, shuffle row order WITHIN each row group
            (cheap; row groups are size-10 by default per build_parquet_dataset).
        infer_n_samples_from_shards: if False (default for tests), use the
            metadata count from current shards verbatim. If True, multiply
            the per-rank sample count by an estimated yield factor.

    The dataset surfaces routing/label metadata that has been pre-computed in
    the parquet — there is NO dynamic routing here. See `route_row()` for the
    1:1 mapping from `task_routing` columns to (task_id, cls_label, has_mask_gt).
    """

    def __init__(
        self,
        parquet_root: str | Path,
        split: str,
        blend: BlendConfig | None = None,
        modalities: tuple[str, ...] = ("geom", "em"),
        augment: bool = False,
        max_samples: int | None = None,
        data_subsample_seed: int | None = None,
        seed: int = 42,
        shuffle_within_shard: bool = True,
    ):
        self.parquet_root = Path(parquet_root)
        canon = SPLIT_ALIASES.get(split, split)
        self.split = canon
        self.blend = blend or BlendConfig()
        self.modalities = tuple(modalities)
        self.augment = bool(augment)
        self.seed = int(seed)
        self.shuffle_within_shard = bool(shuffle_within_shard)
        self.epoch = 0

        # ── discover shards ──
        shard_dir = self.parquet_root / canon
        self.shards: list[Path] = sorted(shard_dir.glob(f"{canon}-*.parquet"))
        if not self.shards:
            raise FileNotFoundError(
                f"no shards at {shard_dir}/{canon}-*.parquet "
                f"(known split aliases: {sorted(SPLIT_ALIASES)})"
            )

        # ── cheap metadata-only scan (no image bytes) ──
        meta_table = pq.read_table(
            [str(s) for s in self.shards],
            columns=list(METADATA_COLUMNS),
        )
        meta_df = meta_table.to_pandas()
        meta_df = self._filter_df(meta_df, self.blend).reset_index(drop=True)

        # ── optional row-level subsample (BEFORE view expansion) ──
        if max_samples is not None and len(meta_df) > max_samples:
            if data_subsample_seed is not None:
                rng = np.random.default_rng(int(data_subsample_seed))
                idx = np.sort(rng.permutation(len(meta_df))[:max_samples])
            else:
                idx = np.arange(max_samples)
            meta_df = meta_df.iloc[idx].reset_index(drop=True)

        # ── compute per-row sampling weight (cell blend + class balance), normalize ──
        self._row_accept_p: dict[str, float] = self._build_accept_probabilities(
            meta_df, self.blend
        )
        self._row_n_items: dict[str, int] = self._build_row_n_items(
            meta_df, self.modalities
        )
        self._surviving_hashes: frozenset[str] = frozenset(
            str(h) for h in meta_df["combined_sample_hash"]
        )

        # Approximate epoch length (sum of per-row accept × per-row n_items).
        self._approx_len = int(round(sum(
            self._row_accept_p.get(h, 0.0) * self._row_n_items.get(h, 0)
            for h in self._surviving_hashes
        )))

        # Cache
        self._meta_df_for_introspection = meta_df

    # ─── filtering / weighting ────────────────────────────────────────────

    @staticmethod
    def _filter_df(df: pd.DataFrame, blend: BlendConfig) -> pd.DataFrame:
        mask = pd.Series(True, index=df.index)
        if blend.include_species is not None:
            mask &= df["species"].isin(blend.include_species)
        if blend.include_sample_types is not None:
            mask &= df["sample_type"].isin(blend.include_sample_types)
        sup_mask = df.apply(lambda r: route_row(r) is not None, axis=1)
        mask &= sup_mask.reindex(df.index, fill_value=False)
        return df.loc[mask]

    @staticmethod
    def _build_accept_probabilities(
        meta_df: pd.DataFrame, blend: BlendConfig
    ) -> dict[str, float]:
        """For each surviving row: accept probability ∈ [0, 1]. Computed by
        running `compute_item_weights` on synthetic 1-item-per-row stand-ins
        and normalising max → 1. Iteration accepts each row with prob = this.
        """
        if len(meta_df) == 0:
            return {}
        items: list[_Item] = []
        hashes: list[str] = []
        for ridx in range(len(meta_df)):
            row = meta_df.iloc[ridx]
            routed = route_row(row)
            if routed is None:
                continue
            task_id, cls_label, _ = routed
            items.append(_Item(
                row_idx=ridx, view_idx=0,
                op_id=str(row["combined_sample_hash"]),
                task_id=task_id, cls_label=cls_label, has_mask_gt=False,
                species=row["species"], sample_type=row["sample_type"],
                kind=SAMPLE_TYPE_TO_KIND[row["sample_type"]],
                same_neuron=bool(row["same_neuron"]),
                modality=MODALITY_GEOM, em_view_suffix="",
                has_synapse_label=False, synapse_label=0,
            ))
            hashes.append(str(row["combined_sample_hash"]))
        if not items:
            return {}
        weights = compute_item_weights(
            items,
            strategy=blend.strategy,
            cell_overrides=blend.weights,
            class_balance=blend.class_balance,
        )
        max_w = float(weights.max()) if len(weights) > 0 else 0.0
        if max_w <= 0:
            return {h: 0.0 for h in hashes}
        return {h: float(weights[i] / max_w) for i, h in enumerate(hashes)}

    @staticmethod
    def _build_row_n_items(
        meta_df: pd.DataFrame, modalities: tuple[str, ...]
    ) -> dict[str, int]:
        """How many per-view items each row would produce if accepted."""
        out: dict[str, int] = {}
        include_geom = "geom" in modalities
        include_em = "em" in modalities
        for ridx in range(len(meta_df)):
            row = meta_df.iloc[ridx]
            routed = route_row(row)
            if routed is None:
                continue
            task_id = routed[0]
            n = 0
            if include_geom:
                if task_id == TASK_JUNCTION and bool(row.get("has_single_mask", False)):
                    n += NUM_GEOM_VIEWS
                elif task_id == TASK_ENDPOINT and bool(row.get("has_dual_mask", False)):
                    n += NUM_GEOM_VIEWS
            # NOTE: we conservatively assume 4 EM views when has_em=True.
            # A row with fewer (rare in practice) will yield fewer items at
            # iter-time; __len__ will be slightly over. Acceptable.
            if include_em and bool(row.get("has_em", False)):
                n += len(EM_VIEWS)
            out[str(row["combined_sample_hash"])] = n
        return out

    # ─── IterableDataset interface ────────────────────────────────────────

    def set_epoch(self, ep: int) -> None:
        """Advance epoch; per-epoch RNG keys re-shuffle shard order + within-shard rows."""
        self.epoch = int(ep)

    def __len__(self) -> int:
        """Approximate # items per epoch (deterministic, but accept rolls add Bernoulli noise)."""
        return self._approx_len

    def __iter__(self) -> Iterator[dict[str, Any]]:
        worker_info = get_worker_info()
        if worker_info is None:
            wid, nworkers = 0, 1
        else:
            wid, nworkers = int(worker_info.id), int(worker_info.num_workers)

        # DDP-rank partitioning: when distributed is initialized, slice shards by
        # rank *before* worker partitioning. Each (rank, worker) pair gets a unique
        # stride class within the global pool of (world_size * nworkers) consumers.
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
        else:
            world_size, rank = 1, 0

        # Shard order shuffle (all consumers see the SAME order so partitioning is clean).
        shard_rng = random.Random(_mix_seed(self.seed, self.epoch, "shards"))
        shuffled_shards = list(self.shards)
        shard_rng.shuffle(shuffled_shards)
        consumer_idx = rank * nworkers + wid
        n_consumers = world_size * nworkers
        my_shards = shuffled_shards[consumer_idx::n_consumers]

        # Per-(rank, worker, epoch) RNG for within-shard shuffling + accept rolls.
        rng = random.Random(_mix_seed(self.seed, self.epoch, rank, wid, "iter"))

        for shard_path in my_shards:
            pf = pq.ParquetFile(str(shard_path))
            n_rg = pf.num_row_groups
            for rg_idx in range(n_rg):
                rg = pf.read_row_group(rg_idx).to_pandas()
                indices = list(range(len(rg)))
                if self.shuffle_within_shard:
                    rng.shuffle(indices)
                for ri in indices:
                    row = rg.iloc[ri]
                    h = str(row["combined_sample_hash"])
                    if h not in self._surviving_hashes:
                        continue
                    p = self._row_accept_p.get(h, 0.0)
                    if p <= 0.0:
                        continue
                    if p < 1.0 and rng.random() >= p:
                        continue
                    yield from self._materialize_row(row)

    # ─── per-row materialisation ──────────────────────────────────────────

    def _materialize_row(self, row: pd.Series) -> Iterator[dict[str, Any]]:
        routed = route_row(row)
        if routed is None:
            return
        task_id, cls_label, has_mask_gt = routed
        has_syn, syn_label = synapse_label_from_row(row)
        sample_type = row["sample_type"]
        kind = SAMPLE_TYPE_TO_KIND[sample_type]
        species = row["species"]
        same_neuron = bool(row["same_neuron"])
        op_id = str(row["source_archive_sample_hash"])

        # ── geom items ──
        if "geom" in self.modalities:
            if task_id == TASK_JUNCTION:
                has_input = bool(row.get("has_single_mask", False))
            else:
                has_input = bool(row.get("has_dual_mask", False))
            if has_input:
                has_dual = bool(row.get("has_dual_mask", False))
                has_mask_gt_eff = has_mask_gt and has_dual

                # Decode source arrays ONCE per row, then iterate views.
                if task_id == TASK_JUNCTION:
                    src = _decode_npz(row["geometry_single"])
                else:
                    src = _decode_npz(row["geometry"])
                dual = _decode_npz(row["geometry"]) if has_mask_gt_eff else None

                for v in range(NUM_GEOM_VIEWS):
                    yield self._build_geom_sample(
                        src, dual, v, task_id, cls_label, has_mask_gt_eff,
                        op_id, species, sample_type, kind, same_neuron,
                        has_syn, syn_label,
                    )

        # ── EM items ──
        if "em" in self.modalities and bool(row.get("has_em", False)):
            for vi, em_suffix in enumerate(EM_VIEWS):
                if not _row_has_em_view(row, em_suffix):
                    continue
                yield self._build_em_sample(
                    row[em_suffix], vi, task_id, cls_label,
                    op_id, species, sample_type, kind, same_neuron,
                    has_syn, syn_label,
                )

    def _build_geom_sample(
        self, src: np.ndarray, dual: np.ndarray | None, view_idx: int,
        task_id: int, cls_label: int, has_mask_gt: bool,
        op_id: str, species: str, sample_type: str, kind: str,
        same_neuron: bool, has_syn: bool, syn_label: int,
    ) -> dict[str, Any]:
        view = src[view_idx].astype(np.float32, copy=True)  # (7, H, W)
        if task_id == TASK_JUNCTION:
            # Defensive: junction input must NOT carry mask channels even if
            # the renderer accidentally populated geometry_single's ch5,6.
            view[CH_MASK_A] = 0.0
            view[CH_MASK_B] = 0.0
        x = torch.from_numpy(view)
        H, W = x.shape[-2:]

        if has_mask_gt and dual is not None:
            dv = dual[view_idx].astype(np.float32, copy=False)
            mask_a = (dv[CH_MASK_A] > 0.5)
            mask_b = (dv[CH_MASK_B] > 0.5)
            # Overlap region → 0 (bg); only exclusive-A and exclusive-B are
            # supervised. Matches unified script main behavior.
            exclusive_a = mask_a & ~mask_b
            exclusive_b = mask_b & ~mask_a
            mask_gt = np.zeros((H, W), dtype=np.int64)
            mask_gt[exclusive_a] = 1
            mask_gt[exclusive_b] = 2
            mask_gt_t = torch.from_numpy(mask_gt)
        else:
            mask_gt_t = torch.zeros((H, W), dtype=torch.int64)

        if self.augment:
            x, mask_gt_t = _train_aug(x, mask_gt_t, MODALITY_GEOM)

        return {
            "input": x,
            "mask_label": mask_gt_t,
            "modality": MODALITY_GEOM,
            "task_id": task_id,
            "cls_label": cls_label,
            "has_mask_gt": has_mask_gt,
            "has_synapse_label": has_syn,
            "synapse_label": syn_label,
            "op_id": op_id,
            "view_idx": view_idx,
            "species": species,
            "kind": kind,
            "sample_type": sample_type,
        }

    def _build_em_sample(
        self, blob: Any, view_idx: int,
        task_id: int, cls_label: int,
        op_id: str, species: str, sample_type: str, kind: str,
        same_neuron: bool, has_syn: bool, syn_label: int,
    ) -> dict[str, Any]:
        png = _hf_image_bytes(blob)
        img = Image.open(io.BytesIO(png)).convert("RGB")
        rgb = np.asarray(img, dtype=np.float32) / 255.0     # (H, W, 3)
        rgb = rgb.transpose(2, 0, 1)                        # (3, H, W)

        if task_id == TASK_JUNCTION:
            or_mask = np.maximum(rgb[CH_EM_MASK_A], rgb[CH_EM_MASK_B])
            rgb[CH_EM_MASK_A] = or_mask
            rgb[CH_EM_MASK_B] = 0.0

        # ImageNet normalization on the 3 populated channels.
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
        rgb = (rgb - mean) / std

        H, W = rgb.shape[-2:]
        x_arr = np.zeros((NUM_GEOM_CHANNELS, H, W), dtype=np.float32)
        x_arr[:3] = rgb
        x = torch.from_numpy(x_arr)
        mask_gt_t = torch.zeros((H, W), dtype=torch.int64)

        if self.augment:
            x, mask_gt_t = _train_aug(x, mask_gt_t, MODALITY_EM)

        return {
            "input": x,
            "mask_label": mask_gt_t,
            "modality": MODALITY_EM,
            "task_id": task_id,
            "cls_label": cls_label,
            "has_mask_gt": False,
            "has_synapse_label": has_syn,
            "synapse_label": syn_label,
            "op_id": op_id,
            "view_idx": view_idx,
            "species": species,
            "kind": kind,
            "sample_type": sample_type,
        }


# ───────────────────────────── Helpers ────────────────────────────────────

def _mix_seed(*parts: Any) -> int:
    """Stable hash → int. `random.Random` accepts ints / bytes; we pass an int.
    Don't use Python's `hash()` on tuples — it's randomised between processes.
    """
    s = "|".join(str(p) for p in parts).encode()
    import hashlib
    return int.from_bytes(hashlib.blake2b(s, digest_size=8).digest(), "big")


def _decode_npz(blob: Any) -> np.ndarray:
    """Decode a `geometry`/`geometry_single` blob → ndarray.

    `build_parquet_dataset.py` writes via `np.savez_compressed(buf, arr)`, so
    the array key is `arr_0`. blob may be raw bytes (pyarrow), or a HF Image
    struct {bytes, path} (rare for npz cols), or a path string.
    """
    if blob is None:
        raise ValueError("npz blob is None")
    if isinstance(blob, dict):
        b = blob.get("bytes")
        if b is not None:
            blob = b
        else:
            blob = blob.get("path")
    if isinstance(blob, (str, Path)):
        return np.load(str(blob))["arr_0"]
    if isinstance(blob, (bytes, bytearray, memoryview)):
        return np.load(io.BytesIO(bytes(blob)))["arr_0"]
    raise TypeError(f"unrecognised npz blob type: {type(blob).__name__}")


def _hf_image_bytes(blob: Any) -> bytes:
    """Extract raw PNG bytes from an HF Image-typed cell.

    HF Image columns serialise as a struct with `bytes` and `path` fields.
    pyarrow surfaces this as a dict; we fall back to reading from path if
    bytes is missing.
    """
    if blob is None:
        raise ValueError("image blob is None")
    if isinstance(blob, (bytes, bytearray, memoryview)):
        return bytes(blob)
    if isinstance(blob, dict):
        b = blob.get("bytes")
        if b is not None:
            return bytes(b)
        p = blob.get("path")
        if p is not None:
            return Path(p).read_bytes()
    raise TypeError(f"unrecognised image blob type: {type(blob).__name__}")


def _row_has_em_view(row: pd.Series, em_suffix: str) -> bool:
    val = row.get(em_suffix)
    if val is None:
        return False
    if isinstance(val, dict):
        return val.get("bytes") is not None or val.get("path") is not None
    return True


# ───────────────────────────── Loaders ────────────────────────────────────

def download_hf_dataset(
    repo_id: str,
    cache_dir: str | Path | None = None,
    revision: str | None = None,
    allow_patterns: list[str] | None = None,
) -> Path:
    """Materialise an HF parquet-shards dataset locally; return the local root.

    Example:
        root = download_hf_dataset("jeffbbrown2/connectomebench2-smoke")
        ds = NeurIPSDataset(root, split="train")
    """
    from huggingface_hub import snapshot_download
    p = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        cache_dir=str(cache_dir) if cache_dir else None,
        revision=revision,
        allow_patterns=allow_patterns,
    )
    return Path(p)


# ────────────────────── Augmentation (CPU path) ───────────────────────────

def _train_aug(
    x: torch.Tensor,
    mask_gt: torch.Tensor,
    modality: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Modality-aware training augs. Spatial augs apply to mask_gt consistently;
    color/blur/erase touch the input only.

    Preserves unified-script `_train_aug` semantics:
      - hflip 0.5
      - rotate ±25°
      - translate ±15%
      - color jitter on intensity-like channels (geom: ch1-4; em: ch0)
      - gaussian blur p=0.4 σ∈[0.1,2.0] same channels
      - random erase p=0.4 box 10-30%
      - channel dropout (geom only, p=0.15 per group: silh / depth / normals)
    """
    import torchvision.transforms.functional as TF

    H, W = x.shape[-2:]
    mask_b = mask_gt.unsqueeze(0).unsqueeze(0).to(torch.float32)  # (1,1,H,W)

    # ── spatial: hflip ──
    if random.random() < 0.5:
        x = TF.hflip(x)
        mask_b = TF.hflip(mask_b)

    # ── spatial: rotate ──
    angle = random.uniform(-25.0, 25.0)
    x = TF.rotate(x, angle)
    mask_b = TF.rotate(mask_b, angle)

    # ── spatial: translate ──
    tx = int(random.uniform(-0.15, 0.15) * W)
    ty = int(random.uniform(-0.15, 0.15) * H)
    x = TF.affine(x, angle=0, translate=[tx, ty], scale=1.0, shear=[0, 0])
    mask_b = TF.affine(mask_b, angle=0, translate=[tx, ty], scale=1.0, shear=[0, 0])

    # ── color / blur / erase channel selection ──
    if modality == MODALITY_GEOM:
        ch_lo, ch_hi = CH_DEPTH, CH_NZ + 1   # ch 1..4 (depth + normals)
    else:
        ch_lo, ch_hi = 0, 1                  # ch 0 (grey) for em

    # ── brightness × contrast on selected channels only ──
    bright = random.uniform(0.6, 1.4)
    contr = random.uniform(0.6, 1.4)
    sub = x[ch_lo:ch_hi].clone() * bright
    sub_mean = sub.mean(dim=(-2, -1), keepdim=True)
    x[ch_lo:ch_hi] = (sub - sub_mean) * contr + sub_mean

    # ── gaussian blur on selected channels ──
    if random.random() < 0.4:
        sigma = random.uniform(0.1, 2.0)
        x[ch_lo:ch_hi] = TF.gaussian_blur(
            x[ch_lo:ch_hi], kernel_size=[5, 5], sigma=[sigma, sigma]
        )

    # ── random erase (input only) ──
    if random.random() < 0.4:
        eh = int(H * random.uniform(0.10, 0.30))
        ew = int(W * random.uniform(0.10, 0.30))
        if eh > 0 and ew > 0:
            y0 = random.randint(0, max(0, H - eh))
            x0 = random.randint(0, max(0, W - ew))
            x[:, y0:y0 + eh, x0:x0 + ew] = 0.0

    # ── channel dropout (geom only) ──
    if modality == MODALITY_GEOM:
        for chans in ([CH_SILH], [CH_DEPTH], [CH_NX, CH_NY, CH_NZ]):
            if random.random() < 0.15:
                for c in chans:
                    x[c] = 0.0

    # mask_gt back to (H,W) long via nearest reconstruction
    mask_out = mask_b.squeeze(0).squeeze(0).round().to(torch.long)
    return x, mask_out


# ─────────────────── Blend weight strategies ──────────────────────────────

def compute_item_weights(
    items: list[_Item],
    strategy: str,
    cell_overrides: dict[tuple[str, str], float] | None = None,
    class_balance: bool = True,
) -> np.ndarray:
    """Per-item sampling weight (1-D float64).

    Returned weights are NOT normalised — `WeightedRandomSampler` only cares
    about ratios. They satisfy `weights.shape == (len(items),)`.

    Cell axis is `(species, kind)`. If you want the finer 5-cell axis
    (sample_type), set `BlendConfig.cell_axis="sample_type"` and we'll branch
    here — currently only the kind axis is implemented.
    """
    if strategy not in VALID_BLEND_STRATEGIES:
        raise ValueError(f"unknown blend strategy: {strategy!r}")
    cell_overrides = cell_overrides or {}

    n = len(items)
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    species = np.array([it.species for it in items])
    kinds = np.array([it.kind for it in items])
    cells = np.array([f"{sp}|{k}" for sp, k in zip(species, kinds)])
    unique_cells, cell_inverse, cell_counts = np.unique(
        cells, return_inverse=True, return_counts=True,
    )

    n_species_total = len(np.unique(species))
    n_kinds_total = len(np.unique(kinds))
    n_cells = len(unique_cells)

    if strategy == "natural":
        cell_w = 1.0 / cell_counts.astype(np.float64)
    elif strategy == "uniform_species":
        sp_count_per_cell = np.array([
            (species == label.split("|", 1)[0]).sum() for label in unique_cells
        ], dtype=np.float64)
        cell_w = 1.0 / (n_species_total * sp_count_per_cell)
    elif strategy == "uniform_type":
        kind_count_per_cell = np.array([
            (kinds == label.split("|", 1)[1]).sum() for label in unique_cells
        ], dtype=np.float64)
        cell_w = 1.0 / (n_kinds_total * kind_count_per_cell)
    elif strategy == "uniform_cell":
        cell_w = 1.0 / (n_cells * cell_counts.astype(np.float64))
    elif strategy == "sqrt":
        sqrt_counts = np.sqrt(cell_counts.astype(np.float64))
        cell_w = sqrt_counts / (sqrt_counts.sum() * cell_counts.astype(np.float64))
    elif strategy == "custom":
        cell_w = 1.0 / cell_counts.astype(np.float64)
    else:
        raise AssertionError("unreachable")

    # Cell-level overrides (multiplicative; applied AFTER the strategy weight)
    for ci, label in enumerate(unique_cells):
        sp, k = label.split("|", 1)
        ov = cell_overrides.get((sp, k))
        if ov is not None:
            cell_w[ci] *= float(ov)

    weights = cell_w[cell_inverse].astype(np.float64)

    # Class balance: ensure each (task_id, cls_label) bucket has equal mass
    # within its bucket (preserves unified-script class_balance fold-in).
    if class_balance:
        task_ids = np.array([it.task_id for it in items])
        cls_labels = np.array([it.cls_label for it in items])
        cb = np.ones(n, dtype=np.float64)
        for tid in (TASK_ENDPOINT, TASK_JUNCTION):
            for cls in (0, 1):
                mask = (task_ids == tid) & (cls_labels == cls)
                count = int(mask.sum())
                if count > 0:
                    cb[mask] = 1.0 / count
        weights = weights * cb

    return weights


# ───────────────────────────── Collate ────────────────────────────────────

def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack tensors; keep python-side fields as lists."""
    keys_tensor = ("input", "mask_label")
    keys_long = ("modality", "task_id", "cls_label", "synapse_label", "view_idx")
    keys_bool = ("has_mask_gt", "has_synapse_label")
    keys_pass = ("op_id", "species", "kind", "sample_type")

    out: dict[str, Any] = {}
    for k in keys_tensor:
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    for k in keys_long:
        out[k] = torch.tensor([b[k] for b in batch], dtype=torch.long)
    for k in keys_bool:
        out[k] = torch.tensor([b[k] for b in batch], dtype=torch.bool)
    for k in keys_pass:
        out[k] = [b[k] for b in batch]
    return out


__all__ = [
    "BlendConfig",
    "NeurIPSDataset",
    "_Item",
    "TASK_ENDPOINT", "TASK_JUNCTION",
    "MODALITY_GEOM", "MODALITY_EM",
    "EM_VIEWS", "SAMPLE_TYPE_TO_KIND", "SPLIT_ALIASES",
    "METADATA_COLUMNS",
    "load_blend_config", "compute_item_weights",
    "download_hf_dataset",
    "route_row", "synapse_label_from_row",
    "collate",
]
