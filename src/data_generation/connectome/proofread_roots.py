"""Quality-filtered proofread root discovery with ops/mm density filtering.

For mouse/fly, proofreading density is uniformly high so we pass through to
get_latest_proofread_roots(). For human/zebrafish, we do a full scan:
batched changelogs → L2 counts → ops_per_mm filter → cache.

Calibrated um_per_l2 values from 20-root-per-species empirical analysis.
"""

import json
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from connectome.utils import get_client_for_species, get_latest_proofread_roots

# empirical calibration: average neurite length (um) per L2 chunk
SPECIES_L2_CALIBRATION = {
    "mouse": 17.5,
    "fly": 10.0,
    "human": 16.3,
    "zebrafish": 5.86,
}


@dataclass
class ProofreadRootConfig:
    species: str
    min_ops_per_mm: float = 20.0
    min_path_um: float = 100.0
    max_stale_days: Optional[int] = 90
    seed: int = 42

    def cache_key(self) -> str:
        return (
            f"{self.species}_minopsmm{self.min_ops_per_mm}"
            f"_minpath{self.min_path_um}"
            f"_stale{self.max_stale_days}"
            f"_seed{self.seed}"
        )


def _load_cached_scan(cache_dir: Path, config: ProofreadRootConfig):
    """Load cached scan results if config matches."""
    cache_file = cache_dir / f"{config.cache_key()}_filtered_roots.json"
    if not cache_file.exists():
        return None

    data = json.loads(cache_file.read_text())
    stored_config = data.get("config", {})

    # verify config match
    if (
        stored_config.get("min_ops_per_mm") != config.min_ops_per_mm
        or stored_config.get("min_path_um") != config.min_path_um
        or stored_config.get("max_stale_days") != config.max_stale_days
    ):
        return None

    roots = np.array(data["root_ids"], dtype=np.int64)
    print(f"  loaded {len(roots)} cached filtered roots from {cache_file}")
    return roots


def _save_cached_scan(
    cache_dir: Path,
    config: ProofreadRootConfig,
    root_ids: np.ndarray,
    stats: dict,
):
    """Save scan results to cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{config.cache_key()}_filtered_roots.json"

    data = {
        "config": asdict(config),
        "root_ids": [int(r) for r in root_ids],
        "stats": stats,
        "timestamp": datetime.now().isoformat(),
    }
    tmp = cache_file.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    tmp.rename(cache_file)
    print(f"  cached {len(root_ids)} filtered roots to {cache_file}")


def get_quality_filtered_roots(
    client,
    species: str,
    config: Optional[ProofreadRootConfig] = None,
    seed: int = 42,
) -> np.ndarray:
    """Get proofread roots, optionally filtered by ops/mm density.

    For mouse/fly: passthrough to get_latest_proofread_roots() (uniformly high
    proofreading density, no filtering needed).

    For human/zebrafish: full scan pipeline with density filtering.

    Returns:
        np.ndarray of filtered root IDs
    """
    if config is None:
        config = ProofreadRootConfig(species=species, seed=seed)

    # mouse/fly: passthrough, no density filtering needed
    if species in ("mouse", "fly"):
        return get_latest_proofread_roots(client, species, seed=config.seed)

    # human/zebrafish: density-filtered scan
    cache_dir = Path(".cache") / f"{species}_proofread_scan"

    cached = _load_cached_scan(cache_dir, config)
    if cached is not None:
        return cached

    um_per_l2 = SPECIES_L2_CALIBRATION.get(species, 10.0)

    # step 1: get all delta roots
    print(f"  [{species}] fetching all proofread roots...")
    all_roots = get_latest_proofread_roots(client, species, seed=config.seed)
    print(f"  [{species}] {len(all_roots)} total roots")

    # step 2: batched get_tabular_change_log → n_ops per root
    print(f"  [{species}] fetching changelogs (batches of 50)...")
    BATCH_SIZE = 50
    root_ops = {}  # root_id → n_ops
    t0 = time.time()

    for i in range(0, len(all_roots), BATCH_SIZE):
        batch = [int(r) for r in all_roots[i : i + BATCH_SIZE]]
        try:
            cl = client.chunkedgraph.get_tabular_change_log(batch, filtered=True)
            for rid, df in cl.items():
                root_ops[int(rid)] = len(df)
        except Exception as e:
            print(f"    batch {i}-{i+BATCH_SIZE} error: {e}")
            # fall back to individual calls
            for rid in batch:
                try:
                    cl = client.chunkedgraph.get_tabular_change_log(
                        [rid], filtered=True
                    )
                    for r, df in cl.items():
                        root_ops[int(r)] = len(df)
                except Exception:
                    pass

        if (i + BATCH_SIZE) % 500 < BATCH_SIZE:
            elapsed = time.time() - t0
            print(
                f"    {min(i + BATCH_SIZE, len(all_roots))}/{len(all_roots)} "
                f"({elapsed:.0f}s elapsed)"
            )

    roots_with_ops = [r for r, n in root_ops.items() if n > 0]
    print(
        f"  [{species}] {len(roots_with_ops)}/{len(all_roots)} roots have >0 ops "
        f"({time.time() - t0:.0f}s)"
    )

    # step 3: get L2 counts for roots with ops (threaded)
    print(f"  [{species}] fetching L2 counts for {len(roots_with_ops)} roots...")
    root_l2_counts = {}

    def _get_l2_count(rid):
        try:
            leaves = client.chunkedgraph.get_leaves(rid, stop_layer=2)
            return rid, len(leaves)
        except Exception:
            return rid, 0

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(_get_l2_count, r) for r in roots_with_ops]
        for f in tqdm(futures, desc="L2 counts", total=len(futures)):
            rid, count = f.result()
            root_l2_counts[rid] = count

    # step 4: compute ops_per_mm and filter
    filtered_roots = []
    stats = {
        "total_roots": len(all_roots),
        "roots_with_ops": len(roots_with_ops),
        "filtered_count": 0,
        "min_ops_per_mm": config.min_ops_per_mm,
        "min_path_um": config.min_path_um,
    }

    stale_cutoff = None
    if config.max_stale_days is not None:
        stale_cutoff = datetime.now() - timedelta(days=config.max_stale_days)

    for rid in roots_with_ops:
        n_ops = root_ops.get(rid, 0)
        n_l2 = root_l2_counts.get(rid, 0)
        if n_l2 == 0:
            continue

        path_um = n_l2 * um_per_l2
        if path_um < config.min_path_um:
            continue

        ops_per_mm = (n_ops / path_um) * 1000
        if ops_per_mm < config.min_ops_per_mm:
            continue

        filtered_roots.append(rid)

    result = np.array(filtered_roots, dtype=np.int64)
    stats["filtered_count"] = len(result)
    print(
        f"  [{species}] {len(result)} roots pass density filter "
        f"(>= {config.min_ops_per_mm} ops/mm, >= {config.min_path_um} um path)"
    )

    # shuffle deterministically
    rng = np.random.RandomState(config.seed)
    rng.shuffle(result)

    _save_cached_scan(cache_dir, config, result, stats)
    return result
