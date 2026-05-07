"""Pack combined.parquet + image files into HuggingFace-native parquet shards.

Each parquet row = 1 sample. Image bytes are stored as binary columns. Missing
modalities are nulls (no sentinel needed; parquet handles nullable cleanly).

Schema per row:
  - identifier / label / split cols (str / bool / list<str>)
  - metadata (str, JSON)
  - present_slots (list<str>)
  - geometry        : bytes  (compressed npz, nullable)
  - geometry_single : bytes  (compressed npz, nullable)
  - em_xy / em_xz / em_yz / em_best : bytes (PNG, nullable)

Layout:
  out_root/
    train/train-00000.parquet ...
    val/val-00000.parquet ...
    test/test-00000.parquet ...
    metadata/{split}.parquet  (sidecar, no image bytes — for fast filtering)
    shards.csv
    demo.parquet              (stratified mini-shard)

Parquet write params (critical for HF dataset viewer to render):
  - row_group_size = 10  (HF viewer reads first row group; small = under scan limit)
  - write_page_index = True  (random access without loading full row group)

Parallelism:
  - N worker processes (one chunk of shards per worker)
  - K read threads per worker (NFS read concurrency)
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import math
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

GIGA_DEFAULT = (
    Path.home() / "VLM_Proofreading" / "neurips_dataset_may06" / "combined.parquet"
)
OUT_ROOT_DEFAULT = Path("/tmp/parquet_build")
DEFAULT_SHARD_BYTES = int(2.5e8)
DEFAULT_WORKERS = 8
DEFAULT_READ_THREADS = 8
DEFAULT_SAMPLE_FOR_SIZE = 200
DEFAULT_ROW_GROUP_SIZE = 10

KEEP_COLS: tuple[str, ...] = (
    "combined_sample_hash",
    "source_archive_sample_hash",
    "source_archive",
    "sample_type",
    "same_neuron",
    "has_single_mask",
    "has_dual_mask",
    "task_routing",
    "false_split_correction_label",
    "false_merge_identification_label",
    "split",
)

EXT_BY_TYPE: dict[str, str] = {
    "geometry": "npz",
    "geometry_single": "npz",
    "em_xy": "png",
    "em_xz": "png",
    "em_yz": "png",
    "em_best": "png",
}
COMPRESS_TAGS = frozenset({"geometry", "geometry_single"})
IMAGE_TAGS = frozenset({"em_xy", "em_xz", "em_yz", "em_best"})
BYTES_COLUMNS: tuple[str, ...] = tuple(EXT_BY_TYPE.keys())

HF_IMAGE_STRUCT = pa.struct([
    pa.field("bytes", pa.binary()),
    pa.field("path", pa.string()),
])


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


@dataclass
class ShardJob:
    shard_id: int
    split: str
    rows: list[dict[str, Any]]


def py_value(v: Any) -> Any:
    if hasattr(v, "tolist"):
        return v.tolist()
    if hasattr(v, "item"):
        try:
            return v.item()
        except Exception:
            return v
    return v


def row_to_dict(row: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for c in KEEP_COLS:
        out[c] = py_value(row[c])
    images = row["images"]
    if images is not None:
        out["images"] = [str(p) for p in images]
    meta_str = row["metadata"]
    if meta_str is not None:
        out["metadata"] = json.loads(meta_str)
    else:
        out["metadata"] = None
    return out


def type_tag_from_image_types(row: dict[str, Any], image_path: str) -> str | None:
    meta = row.get("metadata") or {}
    its = meta.get("image_types") or []
    images = row.get("images") or []
    try:
        idx = images.index(image_path)
        if idx < len(its):
            return str(its[idx])
    except ValueError:
        pass
    return None


def estimate_avg_sizes(
    rows: list[dict[str, Any]],
    parquet_root: Path,
    n_sample: int,
    seed: int,
    log: logging.Logger,
) -> dict[str, int]:
    import numpy as np
    rng = random.Random(seed)
    sample_rows = rng.sample(rows, min(n_sample, len(rows)))
    ext_sizes: dict[str, list[int]] = {}
    misses = 0
    for r in sample_rows:
        for path in r.get("images") or []:
            tag = type_tag_from_image_types(r, path)
            if tag is None or tag not in EXT_BY_TYPE:
                continue
            full = parquet_root / path
            try:
                if tag in COMPRESS_TAGS:
                    arr = np.load(full)
                    buf = io.BytesIO()
                    np.savez_compressed(buf, arr)
                    ext_sizes.setdefault(tag, []).append(len(buf.getvalue()))
                else:
                    ext_sizes.setdefault(tag, []).append(full.stat().st_size)
            except FileNotFoundError:
                misses += 1
    avgs = {tag: int(sum(s) / len(s)) for tag, s in ext_sizes.items() if s}
    log.info(f"size sample on {len(sample_rows)} rows; misses={misses}")
    for tag, avg in sorted(avgs.items()):
        suffix = " (compressed)" if tag in COMPRESS_TAGS else ""
        log.info(f"  avg {tag}: {avg / 1024:.1f} KB{suffix}")
    return avgs


def estimate_row_bytes(row: dict[str, Any], avgs: dict[str, int]) -> int:
    total = 2048
    for path in row.get("images") or []:
        tag = type_tag_from_image_types(row, path)
        if tag is None or tag not in EXT_BY_TYPE:
            continue
        total += avgs.get(tag, 0)
    return total


def build_shard_jobs(
    rows: list[dict[str, Any]],
    avgs: dict[str, int],
    split: str,
    max_shard_bytes: int,
    log: logging.Logger,
) -> list[ShardJob]:
    total_bytes = sum(estimate_row_bytes(r, avgs) for r in rows)
    n_shards = max(1, math.ceil(total_bytes / max_shard_bytes))
    rows_per_shard = math.ceil(len(rows) / n_shards)
    log.info(
        f"  split={split}: rows={len(rows)} est_total={total_bytes / 1e9:.1f}GB "
        f"n_shards={n_shards} rows/shard~={rows_per_shard}"
    )
    jobs: list[ShardJob] = []
    for shard_id in range(n_shards):
        chunk = rows[shard_id * rows_per_shard:(shard_id + 1) * rows_per_shard]
        if not chunk:
            continue
        jobs.append(ShardJob(shard_id=shard_id, split=split, rows=chunk))
    return jobs


def read_sample(row: dict[str, Any], parquet_root: Path) -> dict[str, Any] | None:
    """Returns row record dict ready for PyArrow table.from_pylist."""
    import numpy as np
    bytes_by_col: dict[str, bytes | None] = {c: None for c in BYTES_COLUMNS}
    present: list[str] = []
    for path in row.get("images") or []:
        tag = type_tag_from_image_types(row, path)
        if tag is None or tag not in EXT_BY_TYPE:
            continue
        full = parquet_root / path
        try:
            if tag in COMPRESS_TAGS:
                arr = np.load(full)
                buf = io.BytesIO()
                np.savez_compressed(buf, arr)
                bytes_by_col[tag] = buf.getvalue()
            else:
                bytes_by_col[tag] = full.read_bytes()
            present.append(tag)
        except FileNotFoundError:
            return None

    meta = row.get("metadata") or {}
    record: dict[str, Any] = {}
    for c in KEEP_COLS:
        record[c] = row.get(c)
    record["species"] = meta.get("species")
    record["has_em"] = any(s.startswith("em_") for s in present)
    record["present_slots"] = sorted(present)
    record["metadata"] = json.dumps(meta, default=str) if meta is not None else None
    for c in BYTES_COLUMNS:
        b = bytes_by_col[c]
        if c in IMAGE_TAGS:
            record[c] = {"bytes": b, "path": None} if b is not None else None
        else:
            record[c] = b
    return record


def _hf_features_metadata() -> dict[bytes, bytes]:
    """Build the parquet schema metadata HF reads to recognize Image features."""
    features: dict[str, Any] = {}
    for tag in IMAGE_TAGS:
        features[tag] = {"_type": "Image"}
    info = {"features": features}
    return {b"huggingface": json.dumps({"info": info}).encode("utf-8")}


def pack_shard(args: tuple[ShardJob, Path, Path, int, int]) -> dict[str, Any]:
    import hashlib
    job, parquet_root, out_root, read_threads, row_group_size = args
    out_dir = out_root / job.split
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{job.split}-{job.shard_id:05d}.parquet"
    tmp_path = out_path.with_suffix(".parquet.tmp")

    t0 = time.time()
    records: list[dict[str, Any]] = []
    n_skip = 0
    with ThreadPoolExecutor(max_workers=read_threads) as ex:
        futures = [ex.submit(read_sample, r, parquet_root) for r in job.rows]
        for fut in futures:
            try:
                rec = fut.result()
            except Exception:
                n_skip += 1
                continue
            if rec is None:
                n_skip += 1
                continue
            records.append(rec)

    table = pa.Table.from_pylist(records)
    new_fields: list[pa.Field] = []
    for f in table.schema:
        if f.name in IMAGE_TAGS:
            new_fields.append(pa.field(f.name, HF_IMAGE_STRUCT, nullable=True))
        else:
            new_fields.append(f)
    target_schema = pa.schema(new_fields, metadata=_hf_features_metadata())
    table = table.cast(target_schema)
    pq.write_table(
        table,
        tmp_path,
        row_group_size=row_group_size,
        write_page_index=True,
        compression="snappy",
    )
    os.replace(tmp_path, out_path)
    size = out_path.stat().st_size
    sha = hashlib.sha256()
    with open(out_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            sha.update(chunk)
    wall = time.time() - t0
    return {
        "shard_id": job.shard_id,
        "split": job.split,
        "path": str(out_path.relative_to(out_root)),
        "size": size,
        "n_ok": len(records),
        "n_skip": n_skip,
        "n_samples": len(records),
        "wall_s": wall,
        "rows_per_s": len(records) / max(wall, 1e-3),
        "mb_per_s": (size / 1e6) / max(wall, 1e-3),
        "sha256": sha.hexdigest(),
    }


def write_shards_csv(
    out_root: Path, shards: list[dict[str, Any]], log: logging.Logger
) -> None:
    rows = sorted(
        [
            {
                "path": s["path"],
                "split": s["split"],
                "shard_id": s["shard_id"],
                "n_samples": s["n_samples"],
                "size_bytes": s["size"],
                "sha256": s["sha256"],
            }
            for s in shards
        ],
        key=lambda r: (r["split"], r["shard_id"]),
    )
    out = out_root / "shards.csv"
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    log.info(f"wrote {out} ({len(rows)} shards)")


def write_metadata_sidecars(
    out_root: Path,
    rows_by_split: dict[str, list[dict[str, Any]]],
    jobs: list[ShardJob],
    log: logging.Logger,
) -> None:
    """Per-split parquet without image bytes — for fast filtering / Croissant."""
    key_to_shard: dict[str, str] = {}
    for j in jobs:
        rel = f"{j.split}/{j.split}-{j.shard_id:05d}.parquet"
        for r in j.rows:
            key_to_shard[r["combined_sample_hash"]] = rel

    meta_dir = out_root / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in rows_by_split.items():
        if not rows:
            continue
        records: list[dict[str, Any]] = []
        for r in rows:
            meta = r.get("metadata") or {}
            records.append({
                "combined_sample_hash": r["combined_sample_hash"],
                "source_archive_sample_hash": r["source_archive_sample_hash"],
                "source_archive": r["source_archive"],
                "sample_type": r["sample_type"],
                "same_neuron": r["same_neuron"],
                "has_single_mask": r["has_single_mask"],
                "has_dual_mask": r["has_dual_mask"],
                "has_em": any(t.startswith("em_") for t in (meta.get("image_types") or [])),
                "task_routing": list(r.get("task_routing") or []),
                "false_split_correction_label": r["false_split_correction_label"],
                "false_merge_identification_label": r["false_merge_identification_label"],
                "split": r["split"],
                "species": meta.get("species"),
                "shard": key_to_shard.get(r["combined_sample_hash"], ""),
            })
        out_pq = meta_dir / f"{split}.parquet"
        pq.write_table(pa.Table.from_pylist(records), out_pq)
        log.info(f"wrote {out_pq} ({len(records)} rows)")


def write_demo(
    rows_by_type: dict[str, list[dict[str, Any]]],
    parquet_root: Path,
    out_root: Path,
    n_per_type: int,
    row_group_size: int,
    log: logging.Logger,
) -> None:
    selected: list[dict[str, Any]] = []
    for st, candidates in rows_by_type.items():
        eligible = [
            r for r in candidates
            if r["split"] == "train"
            and r["has_dual_mask"]
            and (st == "junction_control" or any(t.startswith("em_") for t in (r["metadata"] or {}).get("image_types", [])))
        ]
        if not eligible:
            eligible = [r for r in candidates if r["split"] == "train"]
        rng = random.Random(42 + hash(st))
        selected.extend(rng.sample(eligible, min(n_per_type, len(eligible))))
    log.info(f"demo: writing {len(selected)} samples to demo.parquet")
    out_path = out_root / "demo.parquet"
    tmp = out_path.with_suffix(".parquet.tmp")
    records: list[dict[str, Any]] = []
    for row in selected:
        rec = read_sample(row, parquet_root)
        if rec is None:
            continue
        records.append(rec)
    table = pa.Table.from_pylist(records)
    pq.write_table(
        table,
        tmp,
        row_group_size=row_group_size,
        write_page_index=True,
        compression="snappy",
    )
    os.replace(tmp, out_path)
    log.info(f"demo: wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB, {len(records)} samples)")


def main() -> None:
    setup_logging()
    log = logging.getLogger("parquet")

    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", type=Path, default=GIGA_DEFAULT)
    ap.add_argument("--out-root", type=Path, default=OUT_ROOT_DEFAULT)
    ap.add_argument("--max-shard-bytes", type=int, default=DEFAULT_SHARD_BYTES)
    ap.add_argument("--row-group-size", type=int, default=DEFAULT_ROW_GROUP_SIZE)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--read-threads", type=int, default=DEFAULT_READ_THREADS)
    ap.add_argument("--sample-for-size", type=int, default=DEFAULT_SAMPLE_FOR_SIZE)
    ap.add_argument("--demo-per-type", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    parquet_root: Path = args.parquet.parent
    out_root: Path = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    log.info(f"reading {args.parquet}")
    df = pq.read_table(args.parquet).to_pandas()
    log.info(f"loaded {len(df)} rows")

    log.info("converting rows to dicts")
    rows_all: list[dict[str, Any]] = [row_to_dict(df.iloc[i]) for i in range(len(df))]

    log.info("estimating avg sizes")
    avgs = estimate_avg_sizes(rows_all, parquet_root, args.sample_for_size, args.seed, log)

    rows_by_split: dict[str, list[dict[str, Any]]] = {s: [] for s in splits}
    for r in rows_all:
        sp = r["split"]
        if sp in rows_by_split:
            rows_by_split[sp].append(r)

    rng = random.Random(args.seed)
    for sp in splits:
        rng.shuffle(rows_by_split[sp])
        if args.limit is not None:
            rows_by_split[sp] = rows_by_split[sp][:args.limit]
        log.info(f"  {sp}: {len(rows_by_split[sp])} rows after shuffle/limit")

    all_jobs: list[ShardJob] = []
    for sp in splits:
        all_jobs.extend(build_shard_jobs(
            rows_by_split[sp], avgs, sp, args.max_shard_bytes, log
        ))
    log.info(f"total shards planned: {len(all_jobs)}")

    if args.dry_run:
        log.info("[dry-run] not writing")
        return

    t0 = time.time()
    log.info(f"packing with {args.workers} procs × {args.read_threads} threads, "
             f"row_group_size={args.row_group_size}")
    pool_args = [
        (j, parquet_root, out_root, args.read_threads, args.row_group_size)
        for j in all_jobs
    ]
    n_done = 0
    total_bytes = 0
    n_ok_total = 0
    n_skip_total = 0
    shard_results: list[dict[str, Any]] = []
    with Pool(processes=args.workers) as pool:
        for res in pool.imap_unordered(pack_shard, pool_args):
            n_done += 1
            total_bytes += res["size"]
            n_ok_total += res["n_ok"]
            n_skip_total += res["n_skip"]
            shard_results.append(res)
            log.info(
                f"  [{n_done}/{len(all_jobs)}] {res['path']} "
                f"{res['size'] / 1e6:.1f}MB ok={res['n_ok']} skip={res['n_skip']} "
                f"rows/s={res['rows_per_s']:.0f} ({res['wall_s']:.1f}s)"
            )
    wall = time.time() - t0
    log.info(f"main packing done: {n_ok_total} ok, {n_skip_total} skip, "
             f"{total_bytes / 1e9:.1f}GB across {len(all_jobs)} shards in {wall / 60:.1f}min")

    write_shards_csv(out_root, shard_results, log)
    write_metadata_sidecars(out_root, rows_by_split, all_jobs, log)

    if args.demo_per_type > 0 and "train" in splits:
        rows_by_type: dict[str, list[dict[str, Any]]] = {}
        for r in rows_by_split["train"]:
            rows_by_type.setdefault(r["sample_type"], []).append(r)
        write_demo(rows_by_type, parquet_root, out_root, args.demo_per_type,
                   args.row_group_size, log)

    log.info(f"all done in {(time.time() - t0) / 60:.1f}min total")


if __name__ == "__main__":
    main()
