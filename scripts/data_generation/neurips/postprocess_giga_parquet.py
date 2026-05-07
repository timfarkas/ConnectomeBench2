"""Idempotent postprocessing for giga_questions.parquet.

Adds 7 derived columns:
  - has_single_mask, has_dual_mask        (parsed from metadata.image_types)
  - sample_type                            (from source_archive + metadata.strategy)
  - same_neuron                            (derived from sample_type)
  - task_routing                           (list[str] from sample_type + mask presence)
  - false_split_correction_label           (= same_neuron)
  - false_merge_identification_label       (= not same_neuron)

In-place atomic overwrite of giga_questions.parquet. Re-running overwrites
the derived cols (idempotent).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

GIGA_DEFAULT = (
    Path.home() / "VLM_Proofreading" / "neurips_dataset_may06" / "combined.parquet"
)
SPLITS_ROOT_DEFAULT = Path.home() / "ConnectomeEnv" / "configs" / "cube_splits"

SAMPLE_TYPES_TRUE = {"merge_edit", "junction_control"}
SAMPLE_TYPES_FALSE = {"split_edit", "adjacent_control", "synapse_control"}
META_DROP_FIELDS = ("is_merge", "original_is_merge", "inverted_action", "is_correct", "source_type")
ALL_SAMPLE_TYPES = SAMPLE_TYPES_TRUE | SAMPLE_TYPES_FALSE

ROUTE_SPLIT = "false_split_correction"
ROUTE_MERGE_ID = "false_merge_identification"
ROUTE_MASK_GEN = "split_mask_generation"

ARCHIVE_TO_BANK: dict[str, str] = {
    "edits_and_adj_controls": "operation_bank",
    "junction_controls": "operation_bank_junction_controls",
    "synapse_controls": "operation_bank_synapse_controls",
}
SPLIT_NAMES = ("train", "val", "test")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def derive_sample_type(source_archive: str, strategy: Any) -> str | None:
    sa = source_archive or ""
    if "edits_and_adj" in sa:
        if strategy is None or strategy == "merge_correction":
            return "merge_edit"
        s = str(strategy).lower()
        if s in ("inversion", "split_correction"):
            return "split_edit"
        if "adj" in s:
            return "adjacent_control"
        return None
    if "junction" in sa:
        return "junction_control"
    if "synapse" in sa:
        return "synapse_control"
    return None


def derive_same_neuron(sample_type: str) -> bool:
    if sample_type in SAMPLE_TYPES_TRUE:
        return True
    if sample_type in SAMPLE_TYPES_FALSE:
        return False
    raise ValueError(f"unknown sample_type: {sample_type!r}")


def mutate_metadata(meta: dict[str, Any], sample_type: str) -> tuple[dict[str, Any], bool]:
    """Drop legacy fields, swap before/after for split_edits, rewrite strategy.
    Returns (new_meta, swapped)."""
    out = dict(meta)
    for f in META_DROP_FIELDS:
        out.pop(f, None)
    swapped = False
    if sample_type == "split_edit":
        before = out.get("before_root_ids") or []
        after = out.get("after_root_ids") or []
        if isinstance(before, (list, tuple)) and isinstance(after, (list, tuple)) and len(before) > len(after):
            out["before_root_ids"] = list(after)
            out["after_root_ids"] = list(before)
            swapped = True
        out["strategy"] = "split_correction"
    elif sample_type == "merge_edit":
        out["strategy"] = "merge_correction"
    return out, swapped


def derive_task_routing(sample_type: str, has_single: bool, has_dual: bool) -> list[str]:
    routes: list[str] = []
    if sample_type in {"merge_edit", "synapse_control", "adjacent_control"} and has_dual:
        routes.append(ROUTE_SPLIT)
    if sample_type in {"split_edit", "junction_control"} and has_single:
        routes.append(ROUTE_MERGE_ID)
    if sample_type == "split_edit" and has_single:
        routes.append(ROUTE_MASK_GEN)
    return routes


def parse_metadata(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    parsed = df["metadata"].map(lambda v: json.loads(v) if v is not None else {})
    image_types = parsed.map(lambda d: d.get("image_types") or [])
    strategy = parsed.map(lambda d: d.get("strategy"))
    return image_types, strategy, parsed


def split_archive(source_archive: str) -> tuple[str, str] | None:
    for prefix, bank in ARCHIVE_TO_BANK.items():
        if source_archive.startswith(prefix + "_"):
            species = source_archive[len(prefix) + 1:]
            return bank, species
    return None


def hashable(key: Any) -> Any:
    if isinstance(key, list):
        return tuple(hashable(x) for x in key)
    return key


def derive_split_key(bank: str, meta: dict[str, Any]) -> Any | None:
    if bank == "operation_bank":
        op = meta.get("operation_id")
        if op in (None, "", "None"):
            return None
        return int(op)
    if bank == "operation_bank_junction_controls":
        sop = meta.get("source_operation_id")
        if sop in (None, "", "None"):
            return None
        return int(sop)
    if bank == "operation_bank_synapse_controls":
        pre = meta.get("pre_pt_root_id")
        post = meta.get("post_pt_root_id")
        ctr = meta.get("synapse_ctr_pt_nm")
        if pre in (None, "", "None") or post in (None, "", "None") or ctr is None:
            return None
        return (int(pre), int(post), tuple(float(x) for x in ctr))
    return None


def load_split_index(splits_root: Path, log: logging.Logger) -> dict[tuple[str, str, Any], str]:
    """Returns {(species, bank, key) -> split_name}."""
    out: dict[tuple[str, str, Any], str] = {}
    if not splits_root.exists():
        log.warning(f"splits root missing: {splits_root}")
        return out
    n_loaded = 0
    for sp_dir in sorted(splits_root.iterdir()):
        if not sp_dir.is_dir():
            continue
        species = sp_dir.name
        for split_name in SPLIT_NAMES:
            p = sp_dir / f"{split_name}.json"
            if not p.exists():
                continue
            d = json.loads(p.read_text())
            for bank, keys in d.items():
                for k in keys:
                    out[(species, bank, hashable(k))] = split_name
                    n_loaded += 1
    log.info(f"loaded {n_loaded} split keys from {splits_root}")
    return out


def assign_split(
    source_archive: str,
    meta: dict[str, Any],
    split_idx: dict[tuple[str, str, Any], str],
) -> str:
    parts = split_archive(source_archive)
    if parts is None:
        return "unmatched"
    bank, species = parts
    key = derive_split_key(bank, meta)
    if key is None:
        return "unmatched"
    return split_idx.get((species, bank, hashable(key)), "unmatched")


def assert_sample_types(
    df: pd.DataFrame, log: logging.Logger
) -> None:
    missing = df["sample_type"].isna()
    if missing.any():
        n = int(missing.sum())
        log.error(f"{n} rows have null sample_type. examples:")
        hash_col = "source_archive_sample_hash" if "source_archive_sample_hash" in df.columns else "sample_hash"
        cols = ["source_archive", hash_col]
        sample = df.loc[missing, cols].head(10)
        meta_strat = df.loc[missing, "metadata"].map(
            lambda v: json.loads(v).get("strategy") if v else None
        ).head(10)
        for (idx, row), strat in zip(sample.iterrows(), meta_strat):
            log.error(f"  row {idx}: source_archive={row['source_archive']!r}, "
                      f"{hash_col}={row[hash_col]!r}, strategy={strat!r}")
        raise AssertionError(f"{n} rows lack sample_type")


def log_warning_breakdown(
    df: pd.DataFrame, mask: pd.Series, label: str, log: logging.Logger
) -> None:
    n = int(mask.sum())
    if n == 0:
        log.info(f"  [{label}] 0 rows  ✓")
        return
    log.warning(f"  [{label}] {n} rows")
    sub = df.loc[mask]
    by_arch = sub.groupby("source_archive").size().sort_values(ascending=False)
    by_type = sub.groupby("sample_type").size().sort_values(ascending=False)
    for k, v in by_arch.items():
        log.warning(f"      source_archive={k}: {v}")
    for k, v in by_type.items():
        log.warning(f"      sample_type={k}: {v}")


def log_summary(df: pd.DataFrame, log: logging.Logger) -> None:
    log.info("=" * 60)
    log.info("SUMMARY")
    log.info("=" * 60)

    log.info("rows per sample_type:")
    for st, n in df["sample_type"].value_counts().items():
        log.info(f"  {st}: {n}")

    log.info("mask presence by sample_type:")
    grp = df.groupby("sample_type")[["has_single_mask", "has_dual_mask"]].agg(["sum", "count"])
    for st in df["sample_type"].unique():
        s_sum = int(grp.loc[st, ("has_single_mask", "sum")])
        s_cnt = int(grp.loc[st, ("has_single_mask", "count")])
        d_sum = int(grp.loc[st, ("has_dual_mask", "sum")])
        log.info(f"  {st}: single={s_sum}/{s_cnt} ({s_sum/s_cnt:.1%}), "
                 f"dual={d_sum}/{s_cnt} ({d_sum/s_cnt:.1%})")

    log.info("rows per task_routing membership:")
    route_counts: Counter[str] = Counter()
    empty = 0
    for routes in df["task_routing"]:
        if not routes:
            empty += 1
        for r in routes:
            route_counts[r] += 1
    for r, n in route_counts.most_common():
        log.info(f"  {r}: {n}")
    log.info(f"  (empty routing): {empty}")

    log.info("label distributions:")
    for col in ["false_split_correction_label", "false_merge_identification_label"]:
        vc = df[col].value_counts(dropna=False)
        log.info(f"  {col}: " + ", ".join(f"{k}={v}" for k, v in vc.items()))

    log.info("same_neuron distribution by sample_type:")
    sn = df.groupby("sample_type")["same_neuron"].value_counts()
    for (st, val), n in sn.items():
        log.info(f"  {st}: same_neuron={val} -> {n}")


def write_atomic(df: pd.DataFrame, path: Path, log: logging.Logger) -> None:
    table = pa.Table.from_pandas(df, preserve_index=False)
    tmp = path.with_suffix(".parquet.tmp")
    log.info(f"writing parquet -> {path}")
    pq.write_table(table, tmp)
    os.replace(tmp, path)
    log.info(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    setup_logging()
    log = logging.getLogger("postprocess")

    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", type=Path, default=GIGA_DEFAULT,
                    help=f"giga parquet path (default: {GIGA_DEFAULT})")
    ap.add_argument("--dry-run", action="store_true",
                    help="compute + log summary, don't write")
    ap.add_argument("--splits-root", type=Path, default=SPLITS_ROOT_DEFAULT,
                    help=f"cube splits root (default: {SPLITS_ROOT_DEFAULT})")
    args = ap.parse_args()

    pq_path: Path = args.parquet
    if not pq_path.exists():
        log.error(f"not found: {pq_path}")
        sys.exit(1)

    t0 = time.time()
    log.info(f"reading {pq_path}")
    df = pq.read_table(pq_path).to_pandas()
    log.info(f"loaded {len(df)} rows, {len(df.columns)} cols")

    if "sample_hash" in df.columns and "source_archive_sample_hash" not in df.columns:
        df = df.rename(columns={"sample_hash": "source_archive_sample_hash"})
        log.info("renamed sample_hash -> source_archive_sample_hash")
    elif "source_archive_sample_hash" not in df.columns:
        raise RuntimeError("no sample_hash or source_archive_sample_hash column found")

    df["combined_sample_hash"] = [
        hashlib.md5(f"{sa}|{sh}".encode()).hexdigest()
        for sa, sh in zip(df["source_archive"], df["source_archive_sample_hash"])
    ]
    n_unique = df["combined_sample_hash"].nunique()
    if n_unique != len(df):
        log.warning(f"combined_sample_hash NOT fully unique: {n_unique}/{len(df)}")
    else:
        log.info(f"combined_sample_hash: {n_unique}/{len(df)} unique ✓")

    image_types, strategy, parsed_meta = parse_metadata(df)

    df["has_single_mask"] = image_types.map(lambda its: "geometry_single" in its)
    df["has_dual_mask"] = image_types.map(lambda its: "geometry" in its)

    df["sample_type"] = [
        derive_sample_type(sa, st) for sa, st in zip(df["source_archive"], strategy)
    ]
    assert_sample_types(df, log)

    df["same_neuron"] = df["sample_type"].map(derive_same_neuron)
    assert df["same_neuron"].notna().all()

    df["task_routing"] = [
        derive_task_routing(st, hs, hd)
        for st, hs, hd in zip(df["sample_type"], df["has_single_mask"], df["has_dual_mask"])
    ]

    df["false_split_correction_label"] = df["same_neuron"]
    df["false_merge_identification_label"] = ~df["same_neuron"]

    log.info("mutating metadata: drop legacy fields, swap split_edit before/after, rewrite strategy")
    mutated: list[dict[str, Any]] = []
    n_swapped = 0
    n_dropped_total = 0
    for m, st in zip(parsed_meta, df["sample_type"]):
        new_m, swapped = mutate_metadata(m, st)
        n_swapped += int(swapped)
        n_dropped_total += sum(1 for f in META_DROP_FIELDS if f in m and f not in new_m)
        mutated.append(new_m)
    parsed_meta = pd.Series(mutated, index=df.index)
    df["metadata"] = parsed_meta.map(lambda d: json.dumps(d, sort_keys=True))
    log.info(f"  swapped before/after for {n_swapped} split_edit rows")
    log.info(f"  dropped legacy fields total: {n_dropped_total}")
    strat_counts = parsed_meta.map(lambda d: d.get("strategy")).value_counts(dropna=False)
    log.info(f"  strategy distribution: {dict(strat_counts)}")

    legacy_cols = ["source_answer", "question_type", "answer_space", "archive"]
    to_drop = [c for c in legacy_cols if c in df.columns]
    if to_drop:
        df = df.drop(columns=to_drop)
        log.info(f"dropped legacy cols: {to_drop}")

    log.info(f"loading cube splits from {args.splits_root}")
    split_idx = load_split_index(args.splits_root, log)
    df["split"] = [
        assign_split(sa, m, split_idx)
        for sa, m in zip(df["source_archive"], parsed_meta)
    ]
    log.info("split distribution (pre-filter):")
    for k, v in df["split"].value_counts(dropna=False).items():
        log.info(f"  {k}: {v}")
    n_unmatched = int((df["split"] == "unmatched").sum())
    if n_unmatched:
        df = df[df["split"] != "unmatched"].reset_index(drop=True)
        log.info(f"dropped {n_unmatched} unmatched rows; {len(df)} remaining")
    log.info("split x source_archive (top per archive):")
    for arch, sub in df.groupby("source_archive"):
        breakdown = sub["split"].value_counts(dropna=False).to_dict()
        log.info(f"  {arch}: {breakdown}")

    log.info("-" * 60)
    log.info("WARNINGS (rows with mask gaps that block their expected routing)")
    log.info("-" * 60)

    junction_or_inv = df["sample_type"].isin({"junction_control", "split_edit"})
    miss_single = junction_or_inv & ~df["has_single_mask"]
    log_warning_breakdown(df, miss_single, "junction|split_edit MISSING has_single_mask", log)

    needs_dual = df["sample_type"].isin({"merge_edit", "synapse_control", "adjacent_control"})
    miss_dual = needs_dual & ~df["has_dual_mask"]
    log_warning_breakdown(df, miss_dual, "merge_edit|synapse_control|adjacent_control MISSING has_dual_mask", log)

    empty_routing = df["task_routing"].map(lambda rs: len(rs) == 0)
    log_warning_breakdown(df, empty_routing, "EMPTY task_routing (orphan rows)", log)

    log_summary(df, log)

    if args.dry_run:
        log.info(f"[dry-run] not writing. elapsed {time.time()-t0:.1f}s")
        return

    write_atomic(df, pq_path, log)
    log.info(f"done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
