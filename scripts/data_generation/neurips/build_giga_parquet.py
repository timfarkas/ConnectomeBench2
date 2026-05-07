"""Build giga_questions.parquet by union-concatenating all sub-archive parquets.

Reads from `neurips_dataset_may06/` (post-merge layout), writes a single
`giga_questions.parquet` next to the sub-archive dirs. Sub-archive dirs stay
intact so they remain usable in isolation.

Per-source transforms:
  - add `source_archive` = subdir name
  - rename `answer` -> `source_answer` (no cast)
  - stringify `metadata` struct -> JSON string
  - rewrite `images` list paths: "images/x" -> "{source_archive}/images/x"

Cleanup (--cleanup): rm redundant endjunc_binary_4sp_v1_* and single_masks_*
subdirs from out-root only. Safe because they're hardlinks.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

OUT_ROOT_DEFAULT = Path.home() / "VLM_Proofreading" / "neurips_dataset_may06"

INPUT_SUBDIRS: list[str] = [
    "edits_and_adj_controls_fly",
    "edits_and_adj_controls_mouse",
    "edits_and_adj_controls_zebrafish",
    "edits_and_adj_controls_human",
    "junction_controls_fly",
    "junction_controls_mouse",
    "junction_controls_zebrafish",
    "junction_controls_human",
    "synapse_controls_fly",
    "synapse_controls_mouse",
]

REDUNDANT_SUBDIRS: list[str] = [
    "endjunc_binary_4sp_v1_fly",
    "endjunc_binary_4sp_v1_mouse",
    "endjunc_binary_4sp_v1_zebrafish",
    "endjunc_binary_4sp_v1_human",
    "single_masks_fly",
    "single_masks_mouse",
    "single_masks_zebrafish",
]

OUT_NAME = "combined.parquet"


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (bytes, bytearray)):
        return o.decode("utf-8", errors="replace")
    return str(o)


def stringify_metadata(val: Any) -> str | None:
    if val is None:
        return None
    return json.dumps(val, default=_json_default, sort_keys=True)


def rewrite_images(paths: Any, prefix: str) -> list[str] | None:
    if paths is None:
        return None
    return [f"{prefix}/{p}" for p in paths]


def load_one(subdir: Path) -> pd.DataFrame:
    pq_path = subdir / "questions.parquet"
    if not pq_path.exists():
        raise FileNotFoundError(pq_path)
    df = pq.read_table(pq_path).to_pandas()
    archive = subdir.name

    if "answer" in df.columns:
        df = df.rename(columns={"answer": "source_answer"})
        df["source_answer"] = df["source_answer"].map(
            lambda v: None if v is None else str(v)
        )
    df["source_archive"] = archive
    if "metadata" in df.columns:
        df["metadata"] = df["metadata"].map(stringify_metadata)
    if "images" in df.columns:
        df["images"] = df["images"].map(lambda v: rewrite_images(v, archive))
    return df


def union_columns(frames: list[pd.DataFrame]) -> list[str]:
    cols: set[str] = set()
    for f in frames:
        cols.update(f.columns)
    return sorted(cols)


def align(df: pd.DataFrame, all_cols: list[str]) -> pd.DataFrame:
    missing = [c for c in all_cols if c not in df.columns]
    for c in missing:
        df[c] = None
    return df[all_cols]


def write_giga(out_root: Path, frames: list[pd.DataFrame], log: logging.Logger) -> Path:
    all_cols = union_columns(frames)
    log.info(f"unioned columns ({len(all_cols)}): {all_cols}")
    aligned = [align(f, all_cols) for f in frames]
    giga = pd.concat(aligned, ignore_index=True)
    log.info(f"giga shape: {giga.shape}")
    log.info("per-source counts:")
    for archive, n in giga["source_archive"].value_counts().items():
        log.info(f"  {archive}: {n}")

    table = pa.Table.from_pandas(giga, preserve_index=False)
    out_pq = out_root / OUT_NAME
    tmp = out_pq.with_suffix(".parquet.tmp")
    log.info(f"writing parquet -> {out_pq}")
    pq.write_table(table, tmp)
    os.replace(tmp, out_pq)
    log.info(f"wrote {out_pq} ({out_pq.stat().st_size / 1e6:.1f} MB)")
    return out_pq


def cleanup_redundant(out_root: Path, dry_run: bool, log: logging.Logger) -> None:
    for name in REDUNDANT_SUBDIRS:
        d = out_root / name
        if not d.exists():
            log.info(f"[skip] {name} (not present)")
            continue
        if dry_run:
            log.info(f"[dry-run] would rm -rf {d}")
            continue
        log.info(f"rm -rf {d}")
        shutil.rmtree(d)


def main() -> None:
    setup_logging()
    log = logging.getLogger("giga")

    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, default=OUT_ROOT_DEFAULT,
                    help=f"root containing sub-archive dirs (default: {OUT_ROOT_DEFAULT})")
    ap.add_argument("--dry-run", action="store_true",
                    help="don't write parquet or delete dirs, just print summary")
    ap.add_argument("--cleanup", action="store_true",
                    help="after successful write, rm redundant endjunc/single_masks subdirs")
    args = ap.parse_args()

    out_root: Path = args.out_root
    if not out_root.exists():
        raise FileNotFoundError(out_root)

    t0 = time.time()
    frames: list[pd.DataFrame] = []
    for name in INPUT_SUBDIRS:
        sub = out_root / name
        if not sub.exists():
            raise FileNotFoundError(sub)
        df = load_one(sub)
        log.info(f"loaded {name}: {len(df)} rows, {len(df.columns)} cols")
        frames.append(df)

    total_rows = sum(len(f) for f in frames)
    log.info(f"total source rows: {total_rows}")

    if args.dry_run:
        all_cols = union_columns(frames)
        log.info(f"[dry-run] would union {len(all_cols)} cols, {total_rows} rows")
        if args.cleanup:
            cleanup_redundant(out_root, dry_run=True, log=log)
        return

    out_pq = write_giga(out_root, frames, log)

    rt = pq.read_table(out_pq)
    if rt.num_rows != total_rows:
        raise RuntimeError(f"row count mismatch: wrote {rt.num_rows}, expected {total_rows}")
    log.info(f"validated row count: {rt.num_rows}")

    if args.cleanup:
        cleanup_redundant(out_root, dry_run=False, log=log)

    log.info(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
