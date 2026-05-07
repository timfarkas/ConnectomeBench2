"""Operation bank: unified data model for all connectome proofreading operations.

Collects ALL proofreading operations (merges + splits) into one JSONL file per
species, decoupling operation discovery from rendering and control generation.

Key design decisions:
- source/sink coords from get_operation_details are in segmentation mip 0 voxel
  space — we fetch resolution dynamically via segmentation_info and convert to nm.
- interface_point_nm is the midpoint of source/sink centroids (~250nm accurate for
  merges, ~600-750nm for splits). good enough for render centering at 7500nm extent.
- deduplication by operation_id (same op appears in multiple roots' changelogs).
- checkpoint/resume via sidecar .checkpoint.json with atomic writes.
- streaming JSONL for large banks (mouse ~743k ops).
"""

import json
import os
import time
import threading
import numpy as np
from collections import OrderedDict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from queue import Queue, Empty
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

from tqdm import tqdm


@dataclass
class ConnectomeOperation:
    """Single proofreading operation from the chunkedgraph."""

    operation_id: int
    root_id: int  # queried root
    species: str
    is_merge: bool  # True = fixes split error, False = fixes merge error
    timestamp: int  # epoch seconds
    source_coords_nm: List[List[float]]  # proofreader source seeds (nm)
    sink_coords_nm: List[List[float]]  # proofreader sink seeds (nm)
    interface_point_nm: List[float]  # midpoint of source/sink centroids
    before_root_ids: List[int]
    after_root_ids: List[int]  # from op_details.roots (NOT changelog)
    added_edges: Optional[List[List[int]]] = None  # SV pairs connected (merges)
    removed_edges: Optional[List[List[int]]] = None  # SV pairs cut (splits)
    user_id: Optional[str] = None

    @property
    def error_type(self) -> str:
        """The error type this operation corrects."""
        return "split" if self.is_merge else "merge"

    @property
    def correction_type(self) -> str:
        """The correction type performed."""
        return "merge" if self.is_merge else "split"

    def to_json_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # strip None fields to save space
        return {k: v for k, v in d.items() if v is not None}

    @classmethod
    def from_json_dict(cls, d: Dict[str, Any]) -> "ConnectomeOperation":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ControlSample:
    """Generated control (negative) sample via inversion or adjacency."""

    source_operation_id: int
    strategy: str  # 'inversion' | 'adjacent_in_cutout'
    is_correct: bool  # False for controls (proposed action is wrong)
    is_merge: bool  # True = proposed action is merge, False = proposed action is split
    root_id: int
    species: str
    interface_point_nm: List[float]
    before_root_ids: List[int]  # segments before the proposed action
    after_root_ids: List[int]  # segments after the proposed action
    before_timestamp: int  # epoch seconds when before_root_ids are valid
    after_timestamp: int  # epoch seconds when after_root_ids are valid
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["_type"] = "control"
        return d

    @classmethod
    def from_json_dict(cls, d: Dict[str, Any]) -> "ControlSample":
        d = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**d)


# ---------------------------------------------------------------------------
# voxel → nm conversion
# ---------------------------------------------------------------------------


def _get_seg_resolution(client) -> np.ndarray:
    """Fetch base segmentation resolution [x, y, z] in nm from the chunkedgraph.

    Falls back to common defaults if the API call fails.
    """
    try:
        info = client.chunkedgraph.segmentation_info
        return np.array(info["scales"][0]["resolution"], dtype=float)
    except Exception:
        # fallback — these are the known mip 0 resolutions
        return np.array([8.0, 8.0, 40.0])


def _voxels_to_nm(coords_vox: List[List[float]], resolution: np.ndarray) -> List[List[float]]:
    """Convert voxel coordinates to nm."""
    if not coords_vox:
        return []
    arr = np.array(coords_vox, dtype=float) * resolution
    return arr.tolist()


def _compute_interface_point(
    source_coords_nm: List[List[float]],
    sink_coords_nm: List[List[float]],
) -> List[float]:
    """Midpoint of source centroid and sink centroid."""
    if not source_coords_nm and not sink_coords_nm:
        return [0.0, 0.0, 0.0]
    if not source_coords_nm:
        return np.mean(sink_coords_nm, axis=0).tolist()
    if not sink_coords_nm:
        return np.mean(source_coords_nm, axis=0).tolist()

    src_centroid = np.mean(source_coords_nm, axis=0)
    snk_centroid = np.mean(sink_coords_nm, axis=0)
    return ((src_centroid + snk_centroid) / 2).tolist()


# ---------------------------------------------------------------------------
# timestamp normalization
# ---------------------------------------------------------------------------


def _normalize_timestamp(ts) -> int:
    """Convert various timestamp formats to epoch seconds (int)."""
    if isinstance(ts, (int, np.integer)):
        v = int(ts)
        # if it looks like milliseconds (> year 2100 in seconds), convert
        if v > 4_102_444_800:
            return v // 1000
        return v
    if isinstance(ts, float):
        v = int(ts)
        if v > 4_102_444_800:
            return v // 1000
        return v
    if isinstance(ts, datetime):
        return int(ts.timestamp())
    if hasattr(ts, "timestamp"):  # pandas Timestamp etc
        return int(ts.timestamp())
    return 0


# ---------------------------------------------------------------------------
# OperationBankBuilder
# ---------------------------------------------------------------------------


class OperationBankBuilder:
    """Build an operation bank JSONL file from chunkedgraph edit history.

    Uses a background prefetch thread for batched changelogs and processes
    operation details on the main thread. Supports checkpoint/resume.
    """

    def __init__(
        self,
        client,
        species: str,
        output_path: Path,
        *,
        target_count: Optional[int] = None,
        changelog_batch_size: int = 50,
        opdetails_batch_size: int = 100,
        prefetch_queue_size: int = 10,
    ):
        self.client = client
        self.species = species
        self.output_path = Path(output_path)
        self.target_count = target_count
        self.changelog_batch_size = changelog_batch_size
        self.opdetails_batch_size = opdetails_batch_size
        self.prefetch_queue_size = prefetch_queue_size

        self._checkpoint_path = self.output_path.with_suffix(".checkpoint.json")
        self._resolution: Optional[np.ndarray] = None
        self._stop_event = threading.Event()

    def _load_checkpoint(self) -> Dict[str, Any]:
        if self._checkpoint_path.exists():
            return json.loads(self._checkpoint_path.read_text())
        return {
            "processed_roots": [],
            "seen_op_ids": [],
            "total_ops": 0,
            "species": self.species,
        }

    def _save_checkpoint(self, ckpt: Dict[str, Any]):
        tmp = self._checkpoint_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(ckpt))
        tmp.rename(self._checkpoint_path)

    def _prefetch_changelogs(
        self,
        roots: List[int],
        processed_set: Set[int],
        queue: Queue,
    ):
        """Background thread: batch-fetch changelogs and put into queue."""
        batch = []
        for root in roots:
            if self._stop_event.is_set():
                break
            if int(root) in processed_set:
                continue
            batch.append(int(root))
            if len(batch) >= self.changelog_batch_size:
                try:
                    cl = self.client.chunkedgraph.get_tabular_change_log(
                        batch, filtered=True
                    )
                    queue.put((batch, cl))
                except Exception as e:
                    # fall back to individual
                    for rid in batch:
                        try:
                            cl = self.client.chunkedgraph.get_tabular_change_log(
                                [rid], filtered=True
                            )
                            queue.put(([rid], cl))
                        except Exception:
                            queue.put(([rid], {}))
                batch = []

        # flush remaining
        if batch and not self._stop_event.is_set():
            try:
                cl = self.client.chunkedgraph.get_tabular_change_log(
                    batch, filtered=True
                )
                queue.put((batch, cl))
            except Exception:
                for rid in batch:
                    try:
                        cl = self.client.chunkedgraph.get_tabular_change_log(
                            [rid], filtered=True
                        )
                        queue.put(([rid], cl))
                    except Exception:
                        queue.put(([rid], {}))

        queue.put(None)  # sentinel

    def _collect_op_ids_from_changelogs(
        self,
        changelog_dict: Dict,
        seen_op_ids: Set[int],
    ) -> List[Tuple[int, Dict]]:
        """Extract unique (op_id, row_dict) from changelog DataFrames.

        Returns list of (operation_id, {is_merge, timestamp, before_root_ids,
        after_root_ids, ...}) for ops not yet in seen_op_ids.
        """
        new_ops = []
        for rid, df in changelog_dict.items():
            if df is None or len(df) == 0:
                continue
            for _, row in df.iterrows():
                op_id = int(row.get("operation_id", 0))
                if op_id in seen_op_ids:
                    continue
                seen_op_ids.add(op_id)
                new_ops.append(
                    (
                        op_id,
                        {
                            "root_id": int(rid),
                            "is_merge": bool(row.get("is_merge", False)),
                            "timestamp": row.get("timestamp"),
                            "before_root_ids": row.get("before_root_ids", []),
                            "after_root_ids": row.get("after_root_ids", []),
                            "user_id": str(row.get("user_id", ""))
                            if row.get("user_id")
                            else None,
                        },
                    )
                )
        return new_ops

    def _fetch_operation_details_batch(
        self, op_ids: List[int]
    ) -> Dict[int, Dict]:
        """Fetch operation details for a batch of op IDs."""
        result = {}
        for i in range(0, len(op_ids), self.opdetails_batch_size):
            batch = op_ids[i : i + self.opdetails_batch_size]
            try:
                details = self.client.chunkedgraph.get_operation_details(batch)
                # keys come back as strings sometimes
                for k, v in details.items():
                    result[int(k)] = v
            except Exception as e:
                print(f"    op_details batch error: {e}")
        return result

    def _build_operation(
        self,
        op_id: int,
        changelog_info: Dict,
        op_detail: Dict,
        resolution: np.ndarray,
    ) -> ConnectomeOperation:
        """Combine changelog + op_detail into a ConnectomeOperation."""
        # source/sink coords: voxel space → nm
        source_vox = op_detail.get("source_coords", [])
        sink_vox = op_detail.get("sink_coords", [])

        # ensure nested list format
        if source_vox and not isinstance(source_vox[0], (list, tuple)):
            source_vox = [source_vox]
        if sink_vox and not isinstance(sink_vox[0], (list, tuple)):
            sink_vox = [sink_vox]

        source_nm = _voxels_to_nm(source_vox, resolution)
        sink_nm = _voxels_to_nm(sink_vox, resolution)
        interface_nm = _compute_interface_point(source_nm, sink_nm)

        # after_root_ids: prefer op_detail.roots (has all children for splits)
        # over changelog (which may only record 1 child for splits)
        after_roots = op_detail.get("roots", changelog_info.get("after_root_ids", []))
        if isinstance(after_roots, (int, np.integer)):
            after_roots = [int(after_roots)]
        else:
            after_roots = [int(r) for r in after_roots]

        before_roots = changelog_info.get("before_root_ids", [])
        if isinstance(before_roots, (int, np.integer)):
            before_roots = [int(before_roots)]
        else:
            before_roots = [int(r) for r in before_roots]

        added = op_detail.get("added_edges")
        removed = op_detail.get("removed_edges")
        if added is not None:
            added = [[int(x) for x in e] for e in added]
        if removed is not None:
            removed = [[int(x) for x in e] for e in removed]

        return ConnectomeOperation(
            operation_id=op_id,
            root_id=changelog_info["root_id"],
            species=self.species,
            is_merge=changelog_info["is_merge"],
            timestamp=_normalize_timestamp(changelog_info["timestamp"]),
            source_coords_nm=source_nm,
            sink_coords_nm=sink_nm,
            interface_point_nm=interface_nm,
            before_root_ids=before_roots,
            after_root_ids=after_roots,
            added_edges=added,
            removed_edges=removed,
            user_id=changelog_info.get("user_id"),
        )

    def build(self, root_ids: np.ndarray) -> int:
        """Build the operation bank from the given root IDs.

        Returns the total number of operations written.
        """
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # load checkpoint
        ckpt = self._load_checkpoint()
        processed_set = set(ckpt["processed_roots"])
        seen_op_ids = set(ckpt["seen_op_ids"])
        total_written = ckpt["total_ops"]

        # get segmentation resolution
        resolution = _get_seg_resolution(self.client)
        print(f"  segmentation resolution: {resolution.tolist()} nm/voxel")

        remaining = [int(r) for r in root_ids if int(r) not in processed_set]
        print(
            f"  {len(remaining)} roots to process "
            f"({len(processed_set)} already done, {total_written} ops written)"
        )

        if not remaining:
            print("  nothing to do — all roots already processed")
            return total_written

        # open JSONL in append mode
        jsonl_file = open(self.output_path, "a")

        # start prefetch thread
        changelog_queue: Queue = Queue(maxsize=self.prefetch_queue_size)
        prefetch_thread = threading.Thread(
            target=self._prefetch_changelogs,
            args=(remaining, processed_set, changelog_queue),
            daemon=True,
        )
        prefetch_thread.start()

        pbar = tqdm(total=len(remaining), desc="roots", initial=0)
        roots_since_checkpoint = 0
        CHECKPOINT_INTERVAL = 25  # save checkpoint every N roots

        try:
            while True:
                if self.target_count and total_written >= self.target_count:
                    print(f"\n  target count {self.target_count} reached")
                    self._stop_event.set()
                    break

                item = changelog_queue.get()
                if item is None:
                    break  # sentinel — prefetch done

                batch_roots, cl_dict = item

                # collect unique new ops from this batch
                new_ops = self._collect_op_ids_from_changelogs(cl_dict, seen_op_ids)

                if new_ops:
                    # fetch operation details
                    op_id_list = [op_id for op_id, _ in new_ops]
                    op_details = self._fetch_operation_details_batch(op_id_list)

                    # build operations and write to JSONL
                    for op_id, cl_info in new_ops:
                        if self.target_count and total_written >= self.target_count:
                            break

                        detail = op_details.get(op_id, {})
                        if not detail:
                            continue

                        try:
                            op = self._build_operation(op_id, cl_info, detail, resolution)
                            jsonl_file.write(json.dumps(op.to_json_dict()) + "\n")
                            total_written += 1
                        except Exception as e:
                            print(f"    error building op {op_id}: {e}")

                # update processed roots
                for rid in batch_roots:
                    processed_set.add(rid)

                pbar.update(len(batch_roots))
                roots_since_checkpoint += len(batch_roots)

                # periodic checkpoint
                if roots_since_checkpoint >= CHECKPOINT_INTERVAL:
                    jsonl_file.flush()
                    ckpt["processed_roots"] = [int(r) for r in processed_set]
                    ckpt["seen_op_ids"] = [int(x) for x in seen_op_ids]
                    ckpt["total_ops"] = total_written
                    self._save_checkpoint(ckpt)
                    roots_since_checkpoint = 0

        except KeyboardInterrupt:
            print("\n  interrupted — saving checkpoint...")
        finally:
            pbar.close()
            jsonl_file.flush()
            jsonl_file.close()

            # final checkpoint
            ckpt["processed_roots"] = [int(r) for r in processed_set]
            ckpt["seen_op_ids"] = [int(x) for x in seen_op_ids]
            ckpt["total_ops"] = total_written
            self._save_checkpoint(ckpt)
            self._stop_event.set()
            prefetch_thread.join(timeout=5)

        print(f"  done: {total_written} operations written to {self.output_path}")
        return total_written


# ---------------------------------------------------------------------------
# control generation (pass 2: inversion)
# ---------------------------------------------------------------------------


def generate_inversion_controls(
    bank_path: Path,
    output_path: Path,
) -> int:
    """Generate inversion controls from an operation bank.

    Each merge op A+B→C generates control: "split C→A+B" (known wrong).
    Each split op C→A+B generates control: "merge A+B→C" (known wrong).

    Returns count of controls written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with open(output_path, "w") as out_f:
        for op in iter_operation_bank(bank_path):
            # op.timestamp = T (when the original operation happened)
            # before the op: T-1, after the op: T
            if op.is_merge:
                # original: merge A+B→C at T
                # control: propose split C→A+B (WRONG)
                # C exists at T (post-merge), A+B exist at T-1 (pre-merge)
                control = ControlSample(
                    source_operation_id=op.operation_id,
                    strategy="inversion",
                    is_correct=False,
                    is_merge=False,
                    root_id=op.after_root_ids[0] if op.after_root_ids else op.root_id,
                    species=op.species,
                    interface_point_nm=op.interface_point_nm,
                    before_root_ids=op.after_root_ids,
                    after_root_ids=op.before_root_ids,
                    before_timestamp=op.timestamp,
                    after_timestamp=op.timestamp - 1,
                    metadata={
                        "original_is_merge": True,
                        "inverted_action": "split",
                        "cutout_timestamp": op.timestamp - 1,  # pre-merge: when A,B exist
                    },
                )
            else:
                # original: split C→A+B at T
                # control: propose merge A+B→C (WRONG)
                # A+B exist at T+1 (split results materialize 1s after op), C exists at T-1
                control = ControlSample(
                    source_operation_id=op.operation_id,
                    strategy="inversion",
                    is_correct=False,
                    is_merge=True,
                    root_id=op.root_id,
                    species=op.species,
                    interface_point_nm=op.interface_point_nm,
                    before_root_ids=op.after_root_ids,
                    after_root_ids=op.before_root_ids,
                    before_timestamp=op.timestamp + 1,
                    after_timestamp=op.timestamp - 1,
                    metadata={
                        "original_is_merge": False,
                        "inverted_action": "merge",
                        "cutout_timestamp": op.timestamp + 1,  # post-split: when A,B exist
                    },
                )

            out_f.write(json.dumps(control.to_json_dict()) + "\n")
            count += 1

    print(f"  wrote {count} inversion controls to {output_path}")
    return count


# ---------------------------------------------------------------------------
# loaders (streaming)
# ---------------------------------------------------------------------------


def iter_operation_bank(path: Path) -> Iterator[ConnectomeOperation]:
    """Streaming iterator over a JSONL operation bank."""
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("_type") == "control":
                continue
            yield ConnectomeOperation.from_json_dict(d)


def iter_controls(path: Path) -> Iterator[ControlSample]:
    """Streaming iterator over a JSONL controls file."""
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            yield ControlSample.from_json_dict(d)


def load_operation_bank(path: Path) -> List[ConnectomeOperation]:
    """Load full operation bank into memory. Use iter_operation_bank for large banks."""
    return list(iter_operation_bank(path))


def load_bank_mixed(
    bank_path: Path,
    controls_path: Optional[Path] = None,
) -> Tuple[List[ConnectomeOperation], List[ControlSample]]:
    """Load operations + controls into memory."""
    ops = load_operation_bank(bank_path)
    controls = list(iter_controls(controls_path)) if controls_path else []
    return ops, controls


def bank_stats(path: Path) -> Dict[str, Any]:
    """Quick stats for an operation bank without loading everything."""
    n_total = 0
    n_merge = 0
    n_split = 0
    species = None

    for op in iter_operation_bank(path):
        n_total += 1
        if op.is_merge:
            n_merge += 1
        else:
            n_split += 1
        if species is None:
            species = op.species

    return {
        "total": n_total,
        "merges": n_merge,
        "splits": n_split,
        "species": species,
        "path": str(path),
    }


#!/usr/bin/env python3
"""Build an operation bank: unified inventory of all proofreading operations.

Usage:
    # pass 1: collect operations
    pixi run python scripts/build_operation_bank.py build \
        --species mouse --target-count 1000 --seed 42 \
        --output datasets/mouse/operation_bank.jsonl

    # pass 1 with quality filters (human/zebrafish)
    pixi run python scripts/build_operation_bank.py build \
        --species human --target-count full \
        --min-ops-per-mm 10 --min-path-um 500 \
        --output datasets/human/operation_bank.jsonl

    # pass 2: generate inversion controls
    pixi run python scripts/build_operation_bank.py controls \
        --bank-input datasets/mouse/operation_bank.jsonl \
        --controls-output datasets/mouse/controls.jsonl

    # both passes in one command
    pixi run python scripts/build_operation_bank.py run \
        --species mouse --target-count 1000 --seed 42 \
        --output datasets/mouse/operation_bank.jsonl

    # stats
    pixi run python scripts/build_operation_bank.py stats \
        --bank-input datasets/mouse/operation_bank.jsonl
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[0] / ".." / "src"))

from connectome.utils import get_client_for_species, get_latest_proofread_roots
from connectome.proofread_roots import (
    ProofreadRootConfig,
    get_quality_filtered_roots,
)
from connectome.operation_bank import (
    OperationBankBuilder,
    generate_inversion_controls,
    bank_stats,
)


def cmd_build(args):
    species = args.species
    print(f"=== building operation bank for {species} ===")

    client = get_client_for_species(species)

    # get roots (with quality filtering for human/zebrafish)
    if species in ("human", "zebrafish") and (
        args.min_ops_per_mm is not None or args.min_path_um is not None
    ):
        config = ProofreadRootConfig(
            species=species,
            min_ops_per_mm=args.min_ops_per_mm or 20.0,
            min_path_um=args.min_path_um or 100.0,
            max_stale_days=args.max_stale_days,
            seed=args.seed,
        )
        roots = get_quality_filtered_roots(client, species, config, seed=args.seed)
    else:
        roots = get_latest_proofread_roots(client, species, seed=args.seed)

    print(f"  {len(roots)} roots available")

    target = None if args.target_count == "full" else int(args.target_count)

    output = Path(args.output)
    builder = OperationBankBuilder(
        client=client,
        species=species,
        output_path=output,
        target_count=target,
    )

    total = builder.build(roots)
    print(f"\n=== done: {total} operations in {output} ===")


def cmd_controls(args):
    bank_input = Path(args.bank_input)
    controls_output = Path(args.controls_output)

    if not bank_input.exists():
        print(f"error: bank file not found: {bank_input}")
        sys.exit(1)

    print(f"=== generating inversion controls from {bank_input} ===")
    count = generate_inversion_controls(bank_input, controls_output)
    print(f"\n=== done: {count} controls in {controls_output} ===")


def cmd_run(args):
    """Run both passes: build bank → generate inversion controls."""
    # pass 1
    cmd_build(args)

    # pass 2: controls output lives next to the bank
    bank_path = Path(args.output)
    controls_path = bank_path.with_name(
        bank_path.stem + "_controls" + bank_path.suffix
    )

    print(f"\n=== pass 2: generating inversion controls ===")
    count = generate_inversion_controls(bank_path, controls_path)
    print(f"=== done: {count} controls in {controls_path} ===")

    # quick stats
    s = bank_stats(bank_path)
    print(f"\n=== summary ===")
    print(f"  ops:      {s['total']} ({s['merges']} merges, {s['splits']} splits)")
    print(f"  controls: {count}")
    print(f"  bank:     {bank_path}")
    print(f"  controls: {controls_path}")


def cmd_stats(args):
    bank_input = Path(args.bank_input)
    if not bank_input.exists():
        print(f"error: bank file not found: {bank_input}")
        sys.exit(1)

    s = bank_stats(bank_input)
    print(f"=== operation bank stats: {bank_input} ===")
    print(f"  species:  {s['species']}")
    print(f"  total:    {s['total']}")
    print(f"  merges:   {s['merges']} ({100*s['merges']/max(1,s['total']):.0f}%)")
    print(f"  splits:   {s['splits']} ({100*s['splits']/max(1,s['total']):.0f}%)")
    print(f"  ratio:    {s['merges']/max(1,s['splits']):.1f}x merge/split")


def main():
    parser = argparse.ArgumentParser(
        description="build an operation bank from chunkedgraph edit history"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # build subcommand
    build_parser = subparsers.add_parser("build", help="collect operations into JSONL")
    build_parser.add_argument("--species", required=True, choices=["mouse", "fly", "human", "zebrafish"])
    build_parser.add_argument("--target-count", default="full", help="number of ops to collect, or 'full'")
    build_parser.add_argument("--seed", type=int, default=42)
    build_parser.add_argument("--output", required=True, help="output JSONL path")
    build_parser.add_argument("--min-ops-per-mm", type=float, default=None, help="min ops/mm density filter (human/zebrafish)")
    build_parser.add_argument("--min-path-um", type=float, default=None, help="min neurite path length in um (human/zebrafish)")
    build_parser.add_argument("--max-stale-days", type=int, default=None, help="exclude roots with recent edits")
    build_parser.set_defaults(func=cmd_build)

    # run subcommand (build + controls)
    run_parser = subparsers.add_parser("run", help="build bank + generate controls (both passes)")
    run_parser.add_argument("--species", required=True, choices=["mouse", "fly", "human", "zebrafish"])
    run_parser.add_argument("--target-count", default="full", help="number of ops to collect, or 'full'")
    run_parser.add_argument("--seed", type=int, default=42)
    run_parser.add_argument("--output", required=True, help="output JSONL path")
    run_parser.add_argument("--min-ops-per-mm", type=float, default=None, help="min ops/mm density filter (human/zebrafish)")
    run_parser.add_argument("--min-path-um", type=float, default=None, help="min neurite path length in um (human/zebrafish)")
    run_parser.add_argument("--max-stale-days", type=int, default=None, help="exclude roots with recent edits")
    run_parser.set_defaults(func=cmd_run)

    # controls subcommand
    controls_parser = subparsers.add_parser("controls", help="generate inversion controls")
    controls_parser.add_argument("--bank-input", required=True, help="input operation bank JSONL")
    controls_parser.add_argument("--controls-output", required=True, help="output controls JSONL")
    controls_parser.set_defaults(func=cmd_controls)

    # stats subcommand
    stats_parser = subparsers.add_parser("stats", help="show bank statistics")
    stats_parser.add_argument("--bank-input", required=True, help="input operation bank JSONL")
    stats_parser.set_defaults(func=cmd_stats)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
