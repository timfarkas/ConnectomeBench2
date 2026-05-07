"""Spatial cube splits over full opbanks.

Divides each species' volume into fixed-size cubes by `interface_point_nm`,
randomly assigns cubes to train/val/test (default 80/10/10), then tags every
op in every bank by its cube's split.

Output per species at `<root>/<species>/splits/<out_subdir>/`:
- `manifest.json` — bbox, edge, seed, ratios, cube counts, op counts per (bank, split), outliers
- `train.json`, `val.json`, `test.json` — canonical-key lists per bank
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

BANKS: dict[str, str] = {
    "operation_bank": "operation_bank.jsonl",
    "operation_bank_synapse_controls": "operation_bank_synapse_controls.jsonl",
    "operation_bank_junction_controls": "operation_bank_junction_controls.jsonl",
}
OUTLIER_THRESHOLD_NM: float = 1e9


def canonical_key(bank: str, row: dict[str, Any]) -> Any:
    if bank == "operation_bank":
        return int(row["operation_id"])
    if bank == "operation_bank_junction_controls":
        return int(row["source_operation_id"])
    if bank == "operation_bank_synapse_controls":
        meta = row["metadata"]
        ctr = meta["synapse_ctr_pt_nm"]
        return [int(meta["pre_pt_root_id"]), int(meta["post_pt_root_id"]), [float(ctr[0]), float(ctr[1]), float(ctr[2])]]
    raise ValueError(f"unknown bank: {bank}")


def hashable_key(key: Any) -> Any:
    if isinstance(key, list):
        return tuple(hashable_key(x) for x in key)
    return key


def load_bank(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def is_outlier(ip: list[float] | None) -> bool:
    if ip is None:
        return True
    return max(abs(ip[0]), abs(ip[1]), abs(ip[2])) > OUTLIER_THRESHOLD_NM


def build_for_species(
    species: str,
    species_root: Path,
    out_subdir: str,
    edge_nm: float,
    ratios: tuple[float, float, float],
    seed: int,
) -> dict[str, Any]:
    full_dir = species_root / "splits" / "full"
    if not full_dir.is_dir():
        raise FileNotFoundError(f"no splits/full at {full_dir}")

    bank_rows: dict[str, list[dict[str, Any]]] = {}
    bank_outliers: dict[str, int] = {}
    for bank, fname in BANKS.items():
        p = full_dir / fname
        if not p.exists():
            print(f"  [{species}/{bank}] missing, skipping")
            bank_rows[bank] = []
            bank_outliers[bank] = 0
            continue
        rows = load_bank(p)
        kept: list[dict[str, Any]] = []
        n_out = 0
        for r in rows:
            if is_outlier(r.get("interface_point_nm")):
                n_out += 1
                continue
            kept.append(r)
        bank_rows[bank] = kept
        bank_outliers[bank] = n_out
        print(f"  [{species}/{bank}] {len(kept)} rows ({n_out} outliers dropped)")

    all_pts: list[tuple[float, float, float]] = []
    for rows in bank_rows.values():
        for r in rows:
            ip = r["interface_point_nm"]
            all_pts.append((float(ip[0]), float(ip[1]), float(ip[2])))
    if not all_pts:
        raise RuntimeError(f"no points for {species}")

    min_x = min(p[0] for p in all_pts)
    min_y = min(p[1] for p in all_pts)
    min_z = min(p[2] for p in all_pts)
    max_x = max(p[0] for p in all_pts)
    max_y = max(p[1] for p in all_pts)
    max_z = max(p[2] for p in all_pts)

    nx = max(1, math.ceil((max_x - min_x) / edge_nm)) or 1
    ny = max(1, math.ceil((max_y - min_y) / edge_nm)) or 1
    nz = max(1, math.ceil((max_z - min_z) / edge_nm)) or 1

    def cube_of(pt: tuple[float, float, float] | list[float]) -> tuple[int, int, int]:
        return (
            int((pt[0] - min_x) // edge_nm),
            int((pt[1] - min_y) // edge_nm),
            int((pt[2] - min_z) // edge_nm),
        )

    occupied: set[tuple[int, int, int]] = set()
    for rows in bank_rows.values():
        for r in rows:
            occupied.add(cube_of(r["interface_point_nm"]))

    cubes_sorted: list[tuple[int, int, int]] = sorted(occupied)
    rng = random.Random(seed)
    rng.shuffle(cubes_sorted)

    n_total = len(cubes_sorted)
    n_train = int(round(n_total * ratios[0]))
    n_val = int(round(n_total * ratios[1]))
    n_test = n_total - n_train - n_val
    if n_test < 0:
        n_val += n_test
        n_test = 0
    train_cubes = set(cubes_sorted[:n_train])
    val_cubes = set(cubes_sorted[n_train : n_train + n_val])
    test_cubes = set(cubes_sorted[n_train + n_val :])

    assert len(train_cubes) + len(val_cubes) + len(test_cubes) == n_total
    assert not (train_cubes & val_cubes)
    assert not (train_cubes & test_cubes)
    assert not (val_cubes & test_cubes)

    splits: dict[str, dict[str, list[Any]]] = {
        "train": {bank: [] for bank in BANKS},
        "val": {bank: [] for bank in BANKS},
        "test": {bank: [] for bank in BANKS},
    }
    seen_keys: dict[str, set[Any]] = {bank: set() for bank in BANKS}
    duplicate_count: dict[str, int] = {bank: 0 for bank in BANKS}

    for bank, rows in bank_rows.items():
        for r in rows:
            cube = cube_of(r["interface_point_nm"])
            if cube in train_cubes:
                split = "train"
            elif cube in val_cubes:
                split = "val"
            else:
                split = "test"
            key = canonical_key(bank, r)
            hk = hashable_key(key)
            if hk in seen_keys[bank]:
                duplicate_count[bank] += 1
                continue
            seen_keys[bank].add(hk)
            splits[split][bank].append(key)

    out_dir = species_root / "splits" / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    op_counts: dict[str, dict[str, int]] = {
        bank: {split: len(splits[split][bank]) for split in ("train", "val", "test")} for bank in BANKS
    }

    manifest: dict[str, Any] = {
        "species": species,
        "edge_nm": edge_nm,
        "ratios": {"train": ratios[0], "val": ratios[1], "test": ratios[2]},
        "seed": seed,
        "bbox_nm": {
            "min": [min_x, min_y, min_z],
            "max": [max_x, max_y, max_z],
        },
        "grid": {"nx": nx, "ny": ny, "nz": nz, "bbox_cubes": nx * ny * nz},
        "cube_counts": {
            "occupied": n_total,
            "train": len(train_cubes),
            "val": len(val_cubes),
            "test": len(test_cubes),
        },
        "op_counts": op_counts,
        "outliers_dropped": bank_outliers,
        "duplicate_keys_dropped": duplicate_count,
    }

    _atomic_write_json(out_dir / "manifest.json", manifest)
    for split in ("train", "val", "test"):
        _atomic_write_json(out_dir / f"{split}.json", splits[split])

    print(f"  [{species}] wrote {out_dir}")
    print(
        f"  [{species}] cubes: train={len(train_cubes)} val={len(val_cubes)} test={len(test_cubes)} (total={n_total})"
    )
    for bank in BANKS:
        c = op_counts[bank]
        print(f"  [{species}] {bank}: train={c['train']} val={c['val']} test={c['test']}")
        if duplicate_count[bank]:
            print(f"  [{species}] {bank}: dropped {duplicate_count[bank]} duplicate keys")
    return manifest


def _atomic_write_json(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.rename(path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--species", default="mouse,fly,human,zebrafish")
    p.add_argument("--root", default=str(Path.home() / "scratch" / "datasets"))
    p.add_argument("--out-subdir", default="cube_splits")
    p.add_argument("--edge-nm", type=float, default=50_000.0)
    p.add_argument("--ratios", default="0.8,0.1,0.1")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    species_list = [s.strip() for s in args.species.split(",") if s.strip()]
    ratios_parts = [float(x) for x in args.ratios.split(",")]
    if len(ratios_parts) != 3:
        raise SystemExit("--ratios needs 3 floats: train,val,test")
    if not math.isclose(sum(ratios_parts), 1.0, abs_tol=1e-6):
        raise SystemExit(f"--ratios must sum to 1.0, got {sum(ratios_parts)}")
    ratios: tuple[float, float, float] = (ratios_parts[0], ratios_parts[1], ratios_parts[2])

    root = Path(args.root).expanduser()
    if not root.is_dir():
        raise SystemExit(f"root does not exist: {root}")

    print(f"root={root}  edge={args.edge_nm}nm  ratios={ratios}  seed={args.seed}")
    print(f"species={species_list}  out_subdir={args.out_subdir}")

    for sp in species_list:
        print(f"\n== {sp} ==")
        species_root = root / sp
        if not species_root.is_dir():
            print(f"  SKIP: {species_root} missing")
            continue
        build_for_species(sp, species_root, args.out_subdir, args.edge_nm, ratios, args.seed)


if __name__ == "__main__":
    main()
