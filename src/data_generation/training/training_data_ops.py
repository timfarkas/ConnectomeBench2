"""Data operations for the training data renderer pipeline.

Extracted from training_data_renderer.py to keep the renderer as a lean
orchestrator. Contains: mesh fetching, dust filtering, chain correction
filtering, adjacent-in-cutout control generation, job derivation, rendering,
and checkpoint reload.

**Subprocess coupling note**: _render_worker_init / _render_worker_fn use
module-level globals (_rw_viewer, _rw_geom_viewer, _rw_cv_seg, _rw_species,
_rw_lru). These MUST stay in this module because ProcessPoolExecutor pickles
the function references and they need to find the globals in the worker's
copy of this module. (_mp_worker_init / _mp_fetch_mesh moved to
connectome.mesh_prefetch.)
"""

import datetime as dt
import hashlib
import json
import logging
import os
import signal
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from tqdm import tqdm

import cloudvolume
import cloudvolume.mesh as cv_mesh

from connectome.em_data import DATA_PARAMETERS, EMDataFetcher
from utils.profiler import profiled, Profiler
from connectome.meshes import get_full_root_mesh, configure_mesh_cache, configure_full_root_cache
from connectome.utils import get_client_for_species
from connectome.operation_bank import (
    ConnectomeOperation,
    ControlSample,
)
from training.question_dataset import (
    QuestionDataset,
    QuestionType,
    AnswerSpace,
    DatasetQuestion,
)
from rendering.render_pipeline import render_neuron_views
from rendering.render_utils import MeshSpec

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

ALL_TASKS = [
    "endpoint_error_id",
    "endpoint_error_corr",
    "junction_error_id",
    "junction_error_corr",
    "junction_corr_proposal",
]

# question type / answer space mapping per task
TASK_QUESTION_CONFIG = {
    "endpoint_error_id": (QuestionType.ENDPOINT_ERROR_IDENTIFICATION, AnswerSpace.ERROR_OR_CONTROL),
    "endpoint_error_corr": (QuestionType.MERGE_VERIFICATION, AnswerSpace.YES_OR_NO),
    "junction_error_id": (QuestionType.JUNCTION_ERROR_IDENTIFICATION, AnswerSpace.ERROR_OR_CONTROL),
    "junction_error_corr": (QuestionType.SPLIT_VERIFICATION, AnswerSpace.YES_OR_NO),
    "junction_corr_proposal": (QuestionType.SPLIT_PROPOSAL, AnswerSpace.SPLIT_POINTS),
}

# colors for two-segment rendering
COLOR_A = "#1f77b4"  # blue
COLOR_B = "#ff7f0e"  # orange
COLOR_SINGLE = "#1F7788"  # teal

DEFAULT_VIEW_EXTENT_NM = 7500.0
DEFAULT_IMAGE_SIZE = (512, 512)
DEFAULT_DUST_METRIC = "l2"  # "l2" or "sv"
DEFAULT_DUST_THRESHOLD = 10
DEFAULT_CHAIN_MAX_HOPS = 2
DEFAULT_CHAIN_PROXIMITY_NM = 2500.0

DEFAULT_ADJACENT_CONTROLS = True
DEFAULT_ADJACENT_CUTOUT_NM = 200
DEFAULT_MAX_ADJACENT_PER_OP = 1
DEFAULT_DUST_OVERSAMPLE = 4
DEFAULT_MESH_WORKERS = 6
DEFAULT_L2_MESH_CACHE_MB = 2048      # 2 GB
DEFAULT_ROOT_SIZE_CACHE_MB = 256     # 256 MB

# full root mesh mode
DEFAULT_FULL_ROOT_MESH_CACHE_MB = 10240   # 10 GB
DEFAULT_FULL_ROOT_MESH_TIMEOUT_S = 120    # 2 min
DEFAULT_PREFETCH_CACHE_MB = 1024          # 1 GB mesh prefetch budget

# mesh fetch internals
_MESH_FETCH_RETRIES = 3
_MESH_FETCH_BACKOFF = 2.0  # seconds, doubles each retry
DEFAULT_L2_MESH_TIMEOUT_S = 20    # spatial (windowed) fetch
DEFAULT_FULL_MESH_TIMEOUT_S = 60  # full root fetch (no center_nm)

_TIMEOUT = object()  # sentinel: distinguishes "fn returned None" from "timed out"
_ERROR = object()    # sentinel: fn raised an exception (should retry)

# adjacent controls internals
_ADJ_CHECKPOINT_INTERVAL = 50  # save checkpoint every N new controls
_ADJ_PREFETCH_MULTIPLIER = 4   # prefetch buffer per worker


# ---------------------------------------------------------------------------
# render job dataclass
# ---------------------------------------------------------------------------


@dataclass
class RenderJob:
    """Specification for a single rendering task."""

    job_id: str  # unique identifier for caching
    task: str  # which task this serves
    root_ids: List[int]  # roots to fetch meshes for
    center_nm: np.ndarray  # render center
    colors: List[str]  # per-root colors
    answer: Any  # task-specific answer
    question_type: QuestionType
    answer_space: AnswerSpace
    metadata: Dict[str, Any] = field(default_factory=dict)
    extent_nm: float = DEFAULT_VIEW_EXTENT_NM
    # EM slice fields (populated by pipeline when em_cutout is enabled)
    em_enabled: bool = False
    em_center_nm: Optional[np.ndarray] = None  # unjittered center for EM (set before jitter)
    em_window_nm: int = 5000
    em_views: List[str] = field(default_factory=lambda: ["xy", "xz", "yz"])
    em_timestamp: Optional[float] = None  # pre-op timestamp for correct seg state


# ---------------------------------------------------------------------------
# mesh fetching
# ---------------------------------------------------------------------------


def _get_cv_seg(species: str) -> cloudvolume.CloudVolume:
    """Get a CloudVolume segmentation client for a species."""
    seg_path = DATA_PARAMETERS[species]["seg_path"]
    secrets = None
    if species in ("human", "zebrafish"):
        token = os.getenv("CAVE_API_TOKEN")
        if token:
            secrets = {"token": token}
    return cloudvolume.CloudVolume(
        seg_path,
        use_https=True,
        progress=False,
        bounded=False,
        fill_missing=True,
        secrets=secrets,
    )


def _get_seg_resolution(cv_seg: cloudvolume.CloudVolume) -> np.ndarray:
    """Get base segmentation resolution [x, y, z] in nm from CloudVolume."""
    return np.array(cv_seg.scales[0]["resolution"], dtype=float)


def _run_with_hard_timeout(fn, timeout_s: float, label: str = ""):
    """Run fn() with a hard wall-clock timeout via a daemon thread.

    Spawns a fresh daemon thread per call so zombie HTTP threads (blocked
    on network I/O after timeout) can't exhaust a fixed thread pool.

    Returns fn()'s return value (including None for legit empties),
    _TIMEOUT on wall-clock timeout, or _ERROR on exception.
    """
    import threading
    result = [_TIMEOUT]  # mutable container for thread to write into
    exc = [None]

    def _worker():
        try:
            result[0] = fn()
        except Exception as e:
            exc[0] = e

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=timeout_s)

    if t.is_alive():
        # thread still running — abandon it (daemon thread dies at process exit)
        logger.warning(f"{label}: hard timeout after {timeout_s:.0f}s (abandoned)")
        return _TIMEOUT
    if exc[0] is not None:
        logger.debug(f"{label}: {type(exc[0]).__name__}: {exc[0]}")
        return _ERROR
    return result[0]


def fetch_full_root_mesh(
    cv_seg: cloudvolume.CloudVolume,
    root_id: int,
    species: str,
    timeout_s: Optional[float] = None,
) -> Optional[cv_mesh.Mesh]:
    """Fetch a precomputed whole-neuron mesh via the graphene mesh layer.

    Same retry + hard timeout pattern as fetch_mesh_for_root(). Does NOT need
    a chunkedgraph client (no get_leaves call). Delegates to
    get_full_root_mesh() from meshes.py which handles the 2-tier cache.
    """
    if timeout_s is None:
        timeout_s = DEFAULT_FULL_ROOT_MESH_TIMEOUT_S
    t0 = time.monotonic()
    for attempt in range(_MESH_FETCH_RETRIES):
        elapsed = time.monotonic() - t0
        remaining = timeout_s - elapsed
        if remaining <= 0:
            logger.warning(f"full root mesh timed out for {root_id} after {elapsed:.0f}s ({attempt} attempts)")
            return None
        result = _run_with_hard_timeout(
            lambda: get_full_root_mesh(cv_seg, root_id, dataset_id=species),
            timeout_s=remaining,
            label=f"full root mesh root={root_id} attempt={attempt+1}",
        )
        if result is _TIMEOUT or result is _ERROR:
            if attempt < _MESH_FETCH_RETRIES - 1:
                delay = _MESH_FETCH_BACKOFF * (2 ** attempt)
                remaining = timeout_s - (time.monotonic() - t0)
                if remaining < delay:
                    return None
                time.sleep(delay)
            continue
        return result
    return None


# ---------------------------------------------------------------------------
# subprocess mesh fetching (bypass GIL)
# ---------------------------------------------------------------------------

from collections import OrderedDict


_BYTES_PER_VERT = 36  # cloudvolume Mesh: 3×float32 pos + 3×int32 face index avg per-vert


class _MeshLRU:
    """Byte-bounded LRU cache for meshes, keyed by root_id. Not thread-safe."""

    def __init__(self, budget_bytes: int) -> None:
        self._budget: int = max(int(budget_bytes), 0)
        self._data: "OrderedDict[int, cv_mesh.Mesh]" = OrderedDict()
        self._sizes: Dict[int, int] = {}
        self._bytes: int = 0
        self.hits: int = 0
        self.misses: int = 0

    @staticmethod
    def _estimate(mesh: cv_mesh.Mesh) -> int:
        try:
            return int(len(mesh.vertices)) * _BYTES_PER_VERT
        except Exception:
            return 0

    def get(self, rid: int) -> Optional[cv_mesh.Mesh]:
        mesh = self._data.get(rid)
        if mesh is not None:
            self._data.move_to_end(rid)
            self.hits += 1
            return mesh
        self.misses += 1
        return None

    def put(self, rid: int, mesh: cv_mesh.Mesh) -> None:
        if self._budget <= 0:
            return
        if rid in self._data:
            self._data.move_to_end(rid)
            return
        size = self._estimate(mesh)
        while self._data and self._bytes + size > self._budget:
            ev_rid, _ = self._data.popitem(last=False)
            self._bytes -= self._sizes.pop(ev_rid, 0)
        self._data[rid] = mesh
        self._sizes[rid] = size
        self._bytes += size


# per-process state — set by _render_worker_init, used by _render_worker_fn
_rw_viewer = None
_rw_geom_viewer = None
_rw_cv_seg: Optional[cloudvolume.CloudVolume] = None
_rw_species: Optional[str] = None
_rw_lru: Optional[_MeshLRU] = None


def _render_worker_init(
    species: str,
    cache_dir: str,
    l2_cache_bytes: int,
    full_root_cache_bytes: int,
    canvas_size_px: Tuple[int, int],
    render_modes: List[str],
    minimap_config: Optional[Dict[str, Any]],
    uni_color_mode: Optional[str],
    base_extent_nm: float,
    per_worker_mesh_budget_bytes: int,
) -> None:
    """Called once per render worker process at pool startup.

    Each subprocess gets its own octarine Viewer + diskcache handle + mesh LRU.
    """
    import faulthandler
    faulthandler.enable()
    import warnings
    warnings.filterwarnings("ignore", message=".*deduplication.*")

    global _rw_viewer, _rw_geom_viewer, _rw_cv_seg, _rw_species, _rw_lru
    _rw_species = species
    _rw_cv_seg = _get_cv_seg(species)
    configure_mesh_cache(size_limit_bytes=l2_cache_bytes, cache_dir=cache_dir)
    configure_full_root_cache(size_limit_bytes=full_root_cache_bytes, cache_dir=cache_dir)
    _rw_lru = _MeshLRU(per_worker_mesh_budget_bytes)

    from rendering.render_pipeline import create_viewer
    from rendering.render_utils import RendererOptions as _RO
    _rw_viewer = create_viewer(_RO(
        background_color="#ffffff",
        default_ortho_extent_nm=base_extent_nm,
        mesh_transparency=False,
    ))

    try:
        from rendering.geometry_renderer import create_geometry_viewer
        _rw_geom_viewer = create_geometry_viewer()
    except Exception as e:
        logger.warning(f"failed to create geometry viewer (will fall back to per-call): {e}")
        _rw_geom_viewer = None

    logger.info(
        f"render worker pid={os.getpid()} ready "
        f"(RSS={_mp_get_rss_mb():.0f}MB, mesh_lru_budget="
        f"{per_worker_mesh_budget_bytes / 1024**3:.2f}GB)"
    )


def _render_worker_fn(
    job: "RenderJob",
    render_dir_str: str,
    canvas_size_px: Tuple[int, int],
    render_modes_list: List[str],
    minimap_cfg: Optional[Dict[str, Any]],
    uni_color_mode_str: Optional[str],
    geometry_cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Optional[List[str]], Optional[List[str]], Optional[str], float]:
    """Execute render_job in a worker process. Reads meshes from per-worker LRU → disk.

    Returns (job_id, image_paths_or_none, image_types_or_none, error_or_none, render_time_s).
    """
    t0 = time.monotonic()
    try:
        meshes: Dict[int, cv_mesh.Mesh] = {}
        for rid in job.root_ids:
            mesh = _rw_lru.get(rid) if _rw_lru is not None else None
            if mesh is None:
                mesh = get_full_root_mesh(_rw_cv_seg, rid, dataset_id=_rw_species or "",
                                          skip_on_miss=True)
                if mesh is None:
                    return (job.job_id, None, None, f"mesh_cache_miss rid={rid}",
                            time.monotonic() - t0)
                if _rw_lru is not None:
                    _rw_lru.put(rid, mesh)
            meshes[rid] = mesh

        if "geometry_single" in render_modes_list:
            after = job.metadata.get("after_root_ids") or []
            if after:
                extra_rid = int(after[0])
                if extra_rid not in meshes:
                    extra_mesh = _rw_lru.get(extra_rid) if _rw_lru is not None else None
                    if extra_mesh is None:
                        extra_mesh = get_full_root_mesh(
                            _rw_cv_seg, extra_rid, dataset_id=_rw_species or "",
                            skip_on_miss=True,
                        )
                        if extra_mesh is not None and _rw_lru is not None:
                            _rw_lru.put(extra_rid, extra_mesh)
                    if extra_mesh is not None:
                        meshes[extra_rid] = extra_mesh

        result = render_job(
            job, meshes, Path(render_dir_str),
            viewer=_rw_viewer,
            canvas_size_px=tuple(canvas_size_px),
            render_modes=render_modes_list,
            minimap_config=minimap_cfg,
            uni_color_mode=uni_color_mode_str,
            geometry_config=geometry_cfg,
            species=_rw_species or "",
        )
        if result is not None:
            return (job.job_id, result[0], result[1], None, time.monotonic() - t0)
        return (job.job_id, None, None, "render_job returned None", time.monotonic() - t0)
    except Exception as e:
        return (job.job_id, None, None, str(e), time.monotonic() - t0)


# mesh prefetch worker functions live in connectome.mesh_prefetch.
# Re-exported here for backwards compat with main-pipeline code that
# imports them from training_data_ops (used as ProcessPool initializer
# / submit target — pickle resolves by qualname at submit time).
from connectome.mesh_prefetch import (  # noqa: E402
    _mp_get_rss_mb,
    _mp_worker_init,
    _mp_fetch_mesh,
)


# ---------------------------------------------------------------------------
# dust filtering — L2 node count pre-check
# ---------------------------------------------------------------------------


def _get_size_cache(species: str, metric: str, cache_mb: int = DEFAULT_ROOT_SIZE_CACHE_MB):
    """Get or create a diskcache for root size counts, keyed by species_root."""
    import diskcache as dc
    cache_dir = Path(os.getenv("CACHE_DIR", ".cache"))
    return dc.Cache(
        cache_dir / f"root_sizes_{species}_{metric}",
        eviction_policy="least-recently-used",
        size_limit=cache_mb * 1024 * 1024,
    )


def _get_dust_cache(species: str, metric: str, cache_mb: int = 64):
    import diskcache as dc
    cache_dir = Path(os.getenv("CACHE_DIR", ".cache"))
    return dc.Cache(
        cache_dir / f"dust_roots_{species}_{metric}",
        eviction_policy="least-recently-used",
        size_limit=cache_mb * 1024 * 1024,
    )


def get_root_size_lazy(
    root: int,
    client,
    dust_cache,
    size_cache,
    metric: str,
    species: str,
    threshold: int,
) -> int:
    """Lazy root size lookup with two-tier persistent cache.

    dust_cache: diskcache storing root_ids known to be below threshold.
    size_cache: diskcache mapping root_id → size count (non-dust roots).

    Returns 0 for known-dust roots and for roots that fetch as 0.
    Populates caches on miss so subsequent calls are free.
    """
    if root in dust_cache:
        return 0
    cached = size_cache.get(root)
    if cached is not None:
        return cached
    size = fetch_root_sizes(client, {root}, metric=metric, species=species).get(root, 0)
    if size < threshold:
        dust_cache.add(root, True)
        return 0
    size_cache.set(root, size)
    return size


def _extract_candidates_from_cutout(
    seg_acc,
    base_roots: List[int],
    exclude: set,
    get_size_fn,
    dust_threshold: int,
    rng: np.random.Generator,
    n_random_roots: int,
    batch_warm_fn=None,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Extract adjacent candidate pairs from an already-fetched segmentation cutout.

    Returns two lists of (base_root, adj_root) tuples:
      adj_to_base_root: adjacencies of each op base root (anchored to the op).
      rand_adj_in_cutout: adjacencies of randomly sampled other roots in the cutout.

    batch_warm_fn: optional callable(roots: List[int]) that pre-warms the size cache
      for a batch of roots in one API call before per-root get_size_fn filtering.
    """
    adj_to_base_root: List[Tuple[int, int]] = []
    for base_root in base_roots:
        adj = seg_acc.get_adjacent_roots(of_ids=np.array([base_root], dtype=np.int64))
        for r in adj:
            r_int = int(r)
            if r_int not in exclude:
                adj_to_base_root.append((base_root, r_int))

    rand_adj_in_cutout: List[Tuple[int, int]] = []
    if n_random_roots > 0:
        unique = seg_acc.unique_roots
        candidates = [int(r) for r in unique if int(r) not in exclude]
        if batch_warm_fn is not None and candidates:
            batch_warm_fn(candidates)
        pool = [r for r in candidates if get_size_fn(r) >= dust_threshold]
        if pool:
            n_pick = min(n_random_roots, len(pool))
            chosen = rng.choice(pool, size=n_pick, replace=False)
            for rand_root in chosen:
                rand_root = int(rand_root)
                adj = seg_acc.get_adjacent_roots(of_ids=np.array([rand_root], dtype=np.int64))
                for r in adj:
                    r_int = int(r)
                    if r_int not in exclude:
                        rand_adj_in_cutout.append((rand_root, r_int))

    return adj_to_base_root, rand_adj_in_cutout


def fetch_root_sizes(
    client,
    root_ids: set,
    metric: str = "l2",
    workers: int = 8,
    species: str = "unknown",
    cache_mb: int = DEFAULT_ROOT_SIZE_CACHE_MB,
) -> Dict[int, int]:
    """Fetch size counts for root_ids in parallel, with diskcache.

    Cache key is species + root_id (root sizes are immutable — a root_id
    is a fixed point in chunkedgraph history).

    Args:
        metric: "l2" for L2 node count, "sv" for supervoxel count.
        cache_mb: size limit in MB for the diskcache.

    Returns dict mapping root_id → count.
    """
    counts: Dict[int, int] = {}
    if not root_ids:
        return counts

    cache = _get_size_cache(species, metric, cache_mb=cache_mb)

    # check cache first
    uncached: set = set()
    for rid in root_ids:
        cached_val = cache.get(rid)
        if cached_val is not None:
            counts[rid] = cached_val
        else:
            uncached.add(rid)

    if uncached:
        logger.info(f"L2 size cache: {len(counts)} hits, {len(uncached)} misses")
    else:
        logger.info(f"L2 size cache: {len(counts)} hits (all cached)")
        return counts

    stop_layer = 2 if metric == "l2" else None

    def _count_one(rid: int) -> Tuple[int, int]:
        try:
            leaves = client.chunkedgraph.get_leaves(rid, stop_layer=stop_layer)
            return rid, len(leaves)
        except Exception as e:
            logger.warning(f"get_leaves failed for {rid}: {e}")
            return rid, 0

    label = "L2 counts" if metric == "l2" else "SV counts"
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_count_one, rid): rid for rid in uncached}
        for future in tqdm(
            as_completed(futures), total=len(futures),
            desc=label, unit="root",
        ):
            rid, count = future.result()
            counts[rid] = count
            cache.set(rid, count)

    return counts


def compute_major_roots(
    before_root_ids: List[int],
    after_root_ids: List[int],
    size_counts: Dict[int, int],
    dust_threshold: int,
    dust_threshold_big: Optional[int] = None,
) -> Optional[Tuple[List[int], List[int]]]:
    """Rank roots by size count, take top-2, check dust threshold.

    Returns (major_before, major_after) or None if any major root is dust.

    Supports paired thresholds: when dust_threshold_big is set, pairs of roots
    are filtered as (smaller >= dust_threshold, bigger >= dust_threshold_big).
    Single roots are checked against dust_threshold only.
    """
    major_before = sorted(
        before_root_ids, key=lambda r: size_counts.get(r, 0), reverse=True
    )[:2]
    major_after = sorted(
        after_root_ids, key=lambda r: size_counts.get(r, 0), reverse=True
    )[:2]

    thresh_big = dust_threshold_big if dust_threshold_big is not None else dust_threshold

    for roots in [major_before, major_after]:
        if len(roots) == 2:
            sizes = sorted([size_counts.get(r, 0) for r in roots])
            if sizes[0] < dust_threshold or sizes[1] < thresh_big:
                return None
        else:
            for rid in roots:
                if size_counts.get(rid, 0) < dust_threshold:
                    return None

    return major_before, major_after


# ---------------------------------------------------------------------------
# chain correction filtering
# ---------------------------------------------------------------------------


def filter_chain_corrections(
    ops: List[ConnectomeOperation],
    controls: List[ControlSample],
    *,
    chain_max_hops: int = DEFAULT_CHAIN_MAX_HOPS,
    chain_proximity_nm: float = DEFAULT_CHAIN_PROXIMITY_NM,
) -> Tuple[List[ConnectomeOperation], List[ControlSample]]:
    """Remove operations superseded by a later inverse operation nearby.

    Chain corrections like split→merge (swap segment C for D) produce
    intermediate states that aren't valid ground truth for either errors or
    controls. This walks the operation graph forward from each op: if a later
    op of the opposite type (merge↔split) exists within chain_proximity_nm,
    the earlier op is marked superseded and excluded.

    Args:
        chain_max_hops: max forward hops through the operation graph.
        chain_proximity_nm: max distance (nm) between interface points
            to consider two operations part of the same chain correction.

    Returns:
        (filtered_ops, filtered_controls) with superseded entries removed.
    """
    if not ops or chain_max_hops <= 0:
        return ops, controls

    # index: root_id → ops that consume it (have it in before_root_ids)
    op_by_id: Dict[int, ConnectomeOperation] = {op.operation_id: op for op in ops}
    consumers: Dict[int, List[int]] = {}
    for op in ops:
        for rid in op.before_root_ids:
            consumers.setdefault(rid, []).append(op.operation_id)

    superseded: set = set()

    for op in ops:
        if op.operation_id in superseded:
            continue

        center = np.array(op.interface_point_nm)
        frontier_roots = set(op.after_root_ids)
        found = False

        for _hop in range(chain_max_hops):
            if not frontier_roots:
                break

            next_frontier: set = set()
            for rid in frontier_roots:
                for cid in consumers.get(rid, []):
                    if cid == op.operation_id:
                        continue
                    consumer = op_by_id.get(cid)
                    if consumer is None:
                        continue

                    # inverse type check
                    if consumer.is_merge != op.is_merge:
                        dist = np.linalg.norm(
                            np.array(consumer.interface_point_nm) - center
                        )
                        if dist <= chain_proximity_nm:
                            superseded.add(op.operation_id)
                            found = True
                            break

                    next_frontier.update(consumer.after_root_ids)

                if found:
                    break

            if found:
                break
            frontier_roots = next_frontier

    filtered_ops = [op for op in ops if op.operation_id not in superseded]
    filtered_ctrls = [
        c for c in controls if c.source_operation_id not in superseded
    ]

    n_removed = len(ops) - len(filtered_ops)
    if n_removed > 0:
        logger.info(
            f"chain filter: {n_removed}/{len(ops)} ops superseded "
            f"(max_hops={chain_max_hops}, proximity={chain_proximity_nm}nm)"
        )
    else:
        logger.info("chain filter: no chain corrections detected")

    return filtered_ops, filtered_ctrls


# ---------------------------------------------------------------------------
# adjacent-in-cutout control generation
# ---------------------------------------------------------------------------


def _adj_fetch_cutout(op, species, client, cutout_nm, cutout_ts: int, quiet_logger, cv_em, cv_seg):
    """Fetch segmentation cutout for one op (used by thread pool)."""
    try:
        ts = dt.datetime.fromtimestamp(cutout_ts, tz=dt.timezone.utc)
        fetcher = EMDataFetcher(
            species, timestamp=ts, client=client, logger=quiet_logger,
            quiet=True, cv_em=cv_em, cv_seg=cv_seg,
        )
        _em_acc, seg_acc = fetcher.fetch_cutout(
            position_nm=op.interface_point_nm, window_size_nm=cutout_nm,
            with_roots=True,
        )
        return op, seg_acc, cutout_ts
    except Exception:
        return op, None, cutout_ts


def _load_adj_checkpoint(
    cache_path: Path,
    species: str,
    cutout_nm: int,
) -> Optional[Tuple[List[dict], set]]:
    """Load adjacent candidates checkpoint (v3 format).

    Returns (candidates, processed_op_ids) if settings match, else None.
    v2 caches are silently rejected and regenerated.
    Only species/cutout_nm must match — dust_threshold and max_per_op are
    runtime parameters applied after loading.
    """
    if not cache_path.exists():
        return None
    try:
        lines = cache_path.read_text().splitlines()
        if not lines:
            return None
        header = json.loads(lines[0])
        if header.get("version") != 4:
            return None  # old format (v3 had wrong cutout_timestamp), regenerate
        if (header.get("species") != species
                or header.get("cutout_nm") != cutout_nm):
            return None
        candidates = [json.loads(l) for l in lines[1:] if l.strip()]
        processed_op_ids = set(header.get("processed_op_ids", []))
        return candidates, processed_op_ids
    except Exception:
        return None


def _save_adj_checkpoint(
    cache_path: Path,
    species: str,
    cutout_nm: int,
    candidates: List[dict],
    processed_op_ids: set,
):
    """Save adjacent candidates checkpoint (atomic write, v4 format)."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    header = {
        "version": 4,
        "species": species,
        "cutout_nm": cutout_nm,
        "processed_op_ids": sorted(processed_op_ids),
        "n_candidates": len(candidates),
    }
    tmp = cache_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        f.write(json.dumps(header) + "\n")
        for cand in candidates:
            f.write(json.dumps(cand) + "\n")
    tmp.rename(cache_path)


def _collect_adj_candidates(
    ops: List[ConnectomeOperation],
    species: str,
    client,
    *,
    cutout_nm: int = DEFAULT_ADJACENT_CUTOUT_NM,
    cache_path: Optional[Path] = None,
    setup_workers: int = 4,
    raw_target: Optional[int] = None,
    n_random_roots: int = 1,
    dust_threshold: int = 0,
    dust_metric: str = "l2",
    root_size_cache_mb: int = DEFAULT_ROOT_SIZE_CACHE_MB,
    seed: Optional[int] = None,
) -> List[dict]:
    """Shared threading+checkpointing engine for adjacent candidate collection.

    Runs over ALL ops (merge + split). Per op:
      merge ops: base_roots = before_root_ids[:2], cutout_ts = timestamp - 1
      split ops: base_roots = after_root_ids[:2],  cutout_ts = timestamp + 1

    Candidate dict format (v4):
      {op_id, root_id, species, interface_point_nm, before_root_ids,
       after_root_ids, timestamp, is_merge,
       adj_to_base_root: [[base, adj], ...],
       rand_adj_in_cutout: [[rand, adj], ...]}

    Both endpoint and junction orchestrators use the same cache_path so cutout
    fetches are shared — one fetch per op total.
    """
    if not ops:
        return []

    candidates: List[dict] = []
    processed_op_ids: set = set()

    if cache_path:
        cached = _load_adj_checkpoint(cache_path, species, cutout_nm)
        if cached is not None:
            candidates, processed_op_ids = cached
            logger.info(
                f"adj candidates: loaded checkpoint "
                f"({len(candidates)} candidates, "
                f"{len(processed_op_ids)} ops processed)"
            )

    if raw_target and len(candidates) >= raw_target:
        logger.info(
            f"adj candidates: {len(candidates)} cached >= "
            f"{raw_target} target, skipping collection"
        )
        return candidates

    shuffle_seed = seed if seed is not None else np.random.randint(0, 1000000)
    shuffle_rng = np.random.default_rng(shuffle_seed)
    extract_rng = np.random.default_rng(shuffle_seed + 1)

    remaining_ops = [op for op in ops if op.operation_id not in processed_op_ids]
    if not remaining_ops:
        if cache_path:
            _save_adj_checkpoint(cache_path, species, cutout_nm, candidates, processed_op_ids)
        return candidates

    indices = shuffle_rng.permutation(len(remaining_ops))
    remaining_ops = [remaining_ops[i] for i in indices]

    need_str = str(raw_target - len(candidates)) if raw_target else "all"
    logger.info(
        f"adj candidates phase 1: {len(remaining_ops)} remaining ops "
        f"(cutout={cutout_nm}nm, need ~{need_str}, workers={setup_workers})"
    )

    quiet_logger = logging.getLogger("adjacent_ctrl")
    quiet_logger.setLevel(logging.WARNING)

    params = DATA_PARAMETERS[species]
    secrets = None
    if species in ("human", "zebrafish"):
        token = os.getenv("CAVE_API_TOKEN")
        if token:
            secrets = {"token": token}
    shared_cv_em = cloudvolume.CloudVolume(
        params["em_path"], use_https=True, mip=params["em_mip"],
        cache=False, lru_bytes=500 * 1024 * 1024, secrets=secrets,
    )
    shared_cv_seg = cloudvolume.CloudVolume(
        params["seg_path"], use_https=True, fill_missing=True,
        mip=params["seg_mip"], cache=True, lru_bytes=500 * 1024 * 1024,
        secrets=secrets,
    )

    size_cache = _get_size_cache(species, dust_metric, cache_mb=root_size_cache_mb)
    dust_cache = _get_dust_cache(species, dust_metric)

    def get_size_fn(root: int) -> int:
        return get_root_size_lazy(
            root, client, dust_cache, size_cache,
            metric=dust_metric, species=species, threshold=dust_threshold,
        )

    def batch_warm_fn(roots: List[int]) -> None:
        """Pre-warm size cache for a batch of roots in one API call."""
        unknown = [r for r in roots if r not in dust_cache and size_cache.get(r) is None]
        if not unknown:
            return
        sizes = fetch_root_sizes(client, set(unknown), metric=dust_metric, species=species)
        for r in unknown:
            sz = sizes.get(r, 0)
            if sz < dust_threshold:
                dust_cache.add(r, True)
            else:
                size_cache.set(r, sz)

    _noop_size_fn = lambda r: 1  # noqa: E731

    n_failed = 0
    last_ckpt_n = len(candidates)
    target_str = str(raw_target) if raw_target else "all"
    pbar = tqdm(total=len(remaining_ops), desc="adj candidates", unit="op")
    op_iter = iter(remaining_ops)

    with ThreadPoolExecutor(max_workers=setup_workers) as executor:
        active: set = set()

        def _submit_batch(n: int) -> None:
            for _ in range(n):
                try:
                    op = next(op_iter)
                    cutout_ts = op.timestamp - 1 if op.is_merge else op.timestamp + 1
                    active.add(executor.submit(
                        _adj_fetch_cutout, op, species, client,
                        cutout_nm, cutout_ts, quiet_logger,
                        shared_cv_em, shared_cv_seg,
                    ))
                except StopIteration:
                    return

        _submit_batch(setup_workers * _ADJ_PREFETCH_MULTIPLIER)

        while active:
            if raw_target and len(candidates) >= raw_target:
                for f in active:
                    f.cancel()
                break

            done, active = wait(active, return_when=FIRST_COMPLETED)

            for fut in done:
                pbar.update(1)
                pbar.set_description(f"adj candidates {len(candidates)}/{target_str}")
                op_result, seg_acc, cutout_ts = fut.result()
                processed_op_ids.add(op_result.operation_id)

                if seg_acc is None:
                    n_failed += 1
                    continue

                base_roots = (
                    op_result.before_root_ids[:2]
                    if op_result.is_merge
                    else op_result.after_root_ids[:2]
                )
                exclude = (
                    set(op_result.before_root_ids)
                    | set(op_result.after_root_ids)
                    | {0}
                )
                size_fn = get_size_fn if dust_threshold > 0 else _noop_size_fn
                warm_fn = batch_warm_fn if dust_threshold > 0 else None
                adj_to_base, rand_adj = _extract_candidates_from_cutout(
                    seg_acc, base_roots, exclude,
                    size_fn, dust_threshold, extract_rng, n_random_roots,
                    batch_warm_fn=warm_fn,
                )

                if adj_to_base or rand_adj:
                    candidates.append({
                        "op_id": op_result.operation_id,
                        "root_id": op_result.root_id,
                        "species": op_result.species,
                        "interface_point_nm": op_result.interface_point_nm,
                        "before_root_ids": op_result.before_root_ids,
                        "after_root_ids": op_result.after_root_ids,
                        "timestamp": op_result.timestamp,
                        "cutout_timestamp": cutout_ts,  # ts-1 for merge, ts+1 for split
                        "is_merge": op_result.is_merge,
                        "adj_to_base_root": [list(p) for p in adj_to_base],
                        "rand_adj_in_cutout": [list(p) for p in rand_adj],
                    })

            _submit_batch(len(done))

            if (cache_path
                    and len(candidates) - last_ckpt_n >= _ADJ_CHECKPOINT_INTERVAL):
                _save_adj_checkpoint(
                    cache_path, species, cutout_nm, candidates, processed_op_ids,
                )
                last_ckpt_n = len(candidates)
                logger.info(f"  checkpoint: {len(candidates)} candidates")

    pbar.close()
    logger.info(
        f"adj candidates phase 1: {len(candidates)} candidates ({n_failed} failed)"
    )

    if cache_path:
        _save_adj_checkpoint(cache_path, species, cutout_nm, candidates, processed_op_ids)

    return candidates


def generate_adjacent_endpoint_samples(
    ops: List[ConnectomeOperation],
    species: str,
    client,
    *,
    cutout_nm: int = DEFAULT_ADJACENT_CUTOUT_NM,
    max_per_op: int = DEFAULT_MAX_ADJACENT_PER_OP,
    max_controls: Optional[int] = None,
    seed: Optional[int] = None,
    cache_path: Optional[Path] = None,
    setup_workers: int = 4,
    dust_threshold: int = 0,
    dust_threshold_big: Optional[int] = None,
    dust_metric: str = "l2",
    root_size_cache_mb: int = DEFAULT_ROOT_SIZE_CACHE_MB,
    n_random_roots: int = 1,
) -> List[ControlSample]:
    """Generate 'adjacent_in_cutout' False controls for endpoint_error_corr.

    Calls _collect_adj_candidates() with all ops (shared cache with junction),
    then filters to merge-op candidates for phase 2+3.
    """
    merge_ops = [op for op in ops if op.is_merge and len(op.before_root_ids) >= 2]
    if not merge_ops:
        return []

    raw_target = max((max_controls or 0) * 3, 500) if max_controls else None
    candidates = _collect_adj_candidates(
        ops, species, client,
        cutout_nm=cutout_nm,
        cache_path=cache_path,
        setup_workers=setup_workers,
        raw_target=raw_target,
        n_random_roots=n_random_roots,
        dust_threshold=dust_threshold,
        dust_metric=dust_metric,
        root_size_cache_mb=root_size_cache_mb,
        seed=seed,
    )

    candidates = [c for c in candidates if c.get("is_merge", True)]

    # ── phase 2: batch dust filter on adj_roots + op's before_root_ids ──
    if dust_threshold > 0 and candidates:
        all_roots_to_check: set = set()
        for cand in candidates:
            for p in cand["adj_to_base_root"]:
                all_roots_to_check.add(p[1])
            for p in cand["rand_adj_in_cutout"]:
                all_roots_to_check.add(p[1])
            all_roots_to_check.update(cand["before_root_ids"])

        thresh_big = dust_threshold_big if dust_threshold_big is not None else dust_threshold
        logger.info(
            f"endpoint adj phase 2: fetching {dust_metric} sizes for "
            f"{len(all_roots_to_check)} unique roots (threshold={dust_threshold}/{thresh_big})"
        )
        size_counts = fetch_root_sizes(
            client, all_roots_to_check, metric=dust_metric,
            species=species, workers=setup_workers, cache_mb=root_size_cache_mb,
        )

        n_before = len(candidates)
        filtered = []
        for cand in candidates:
            # paired filter on the op's own before_root_ids
            before_sizes = sorted([size_counts.get(r, 0) for r in cand["before_root_ids"]])
            if len(before_sizes) >= 2:
                if before_sizes[0] < dust_threshold or before_sizes[-1] < thresh_big:
                    continue
            elif any(s < dust_threshold for s in before_sizes):
                continue
            cand = dict(cand)
            cand["adj_to_base_root"] = [
                p for p in cand["adj_to_base_root"]
                if size_counts.get(p[1], 0) >= dust_threshold
            ]
            cand["rand_adj_in_cutout"] = [
                p for p in cand["rand_adj_in_cutout"]
                if size_counts.get(p[1], 0) >= dust_threshold
            ]
            if cand["adj_to_base_root"] or cand["rand_adj_in_cutout"]:
                filtered.append(cand)
        candidates = filtered
        logger.info(
            f"endpoint adj phase 2: {len(candidates)}/{n_before} ops "
            f"have viable candidates after dust filter"
        )

    # ── phase 3: sample max_per_op, build ControlSamples ─────────────
    # Prefer adj_to_base_root (harder negatives — directly adjacent to the
    # op's own roots), backfill remaining slots with rand_adj_in_cutout.
    if seed is None:
        seed = np.random.randint(0, 1000000)
    rng = np.random.default_rng(seed)

    controls: List[ControlSample] = []
    for cand in candidates:
        adj_pool = [(p, "adj_to_base_root") for p in cand["adj_to_base_root"]]
        rand_pool = [(p, "rand_adj_in_cutout") for p in cand["rand_adj_in_cutout"]]
        if not adj_pool and not rand_pool:
            continue
        n_pick = min(max_per_op, len(adj_pool) + len(rand_pool))
        n_adj = min(n_pick, len(adj_pool))
        chosen_adj = list(rng.choice(len(adj_pool), size=n_adj, replace=False)) if n_adj else []
        n_rand = min(n_pick - n_adj, len(rand_pool))
        chosen_rand = list(rng.choice(len(rand_pool), size=n_rand, replace=False)) if n_rand else []
        chosen = [adj_pool[i] for i in chosen_adj] + [rand_pool[i] for i in chosen_rand]
        for idx, ((base_root, adj_root), pair_source) in enumerate(chosen):
            strategy = "adjacent_in_cutout" if pair_source == "adj_to_base_root" else "rand_adj_in_cutout"
            op_id = cand["op_id"]
            if pair_source == "rand_adj_in_cutout":
                op_id = f"{op_id}_rand_adj_{idx}"
            controls.append(ControlSample(
                source_operation_id=op_id,
                strategy=strategy,
                is_correct=False,
                is_merge=True,
                root_id=cand["root_id"],
                species=cand["species"],
                interface_point_nm=cand["interface_point_nm"],
                before_root_ids=[base_root, adj_root],
                after_root_ids=[],
                before_timestamp=cand["timestamp"] - 1,
                after_timestamp=cand["timestamp"] + 1,
                metadata={"adjacent_root": adj_root, "base_root": base_root, "pair_source": pair_source, "cutout_timestamp": cand["cutout_timestamp"]},
            ))

    if max_controls and len(controls) > max_controls:
        idx = rng.permutation(len(controls))[:max_controls]
        controls = [controls[i] for i in sorted(idx)]

    logger.info(
        f"endpoint adj phase 3: {len(controls)} controls "
        f"(max_per_op={max_per_op}, cap={max_controls or 'none'})"
    )
    return controls


def generate_adjacent_junction_samples(
    ops: List[ConnectomeOperation],
    species: str,
    client,
    *,
    cutout_nm: int = DEFAULT_ADJACENT_CUTOUT_NM,
    max_per_op: int = DEFAULT_MAX_ADJACENT_PER_OP,
    max_controls: Optional[int] = None,
    seed: Optional[int] = None,
    cache_path: Optional[Path] = None,
    setup_workers: int = 4,
    dust_threshold: int = 0,
    dust_threshold_big: Optional[int] = None,
    dust_metric: str = "l2",
    root_size_cache_mb: int = DEFAULT_ROOT_SIZE_CACHE_MB,
    n_random_roots: int = 1,
) -> List[ControlSample]:
    """Generate 'adjacent_in_cutout_junction' True samples for junction_error_corr.

    Uses all ops (merge + split). Each (base_root, adj_root) pair shows two
    genuinely-separate neurons near a proofreading interface — hard True positive
    for split verification (these segments should be separate).
    """
    if not ops:
        return []

    raw_target = max((max_controls or 0) * 3, 500) if max_controls else None
    candidates = _collect_adj_candidates(
        ops, species, client,
        cutout_nm=cutout_nm,
        cache_path=cache_path,
        setup_workers=setup_workers,
        raw_target=raw_target,
        n_random_roots=n_random_roots,
        dust_threshold=dust_threshold,
        dust_metric=dust_metric,
        root_size_cache_mb=root_size_cache_mb,
        seed=seed,
    )

    # ── phase 2: batch dust filter on adj_roots ──────────────────────
    if dust_threshold > 0 and candidates:
        all_roots_to_check: set = set()
        for cand in candidates:
            for p in cand["adj_to_base_root"]:
                all_roots_to_check.add(p[1])
            for p in cand["rand_adj_in_cutout"]:
                all_roots_to_check.add(p[1])

        logger.info(
            f"junction adj phase 2: fetching {dust_metric} sizes for "
            f"{len(all_roots_to_check)} unique roots (threshold={dust_threshold})"
        )
        size_counts = fetch_root_sizes(
            client, all_roots_to_check, metric=dust_metric,
            species=species, workers=setup_workers, cache_mb=root_size_cache_mb,
        )

        n_before = len(candidates)
        filtered = []
        for cand in candidates:
            cand = dict(cand)
            cand["adj_to_base_root"] = [
                p for p in cand["adj_to_base_root"]
                if size_counts.get(p[1], 0) >= dust_threshold
            ]
            cand["rand_adj_in_cutout"] = [
                p for p in cand["rand_adj_in_cutout"]
                if size_counts.get(p[1], 0) >= dust_threshold
            ]
            if cand["adj_to_base_root"] or cand["rand_adj_in_cutout"]:
                filtered.append(cand)
        candidates = filtered
        logger.info(
            f"junction adj phase 2: {len(candidates)}/{n_before} ops "
            f"have viable candidates after dust filter"
        )

    # ── phase 3: sample max_per_op, build ControlSamples ─────────────
    if seed is None:
        seed = np.random.randint(0, 1000000)
    rng = np.random.default_rng(seed)

    controls: List[ControlSample] = []
    for cand in candidates:
        tagged_pool: List[Tuple[Tuple[int, int], str]] = (
            [(p, "adj_to_base_root") for p in cand["adj_to_base_root"]]
            + [(p, "rand_adj_in_cutout") for p in cand["rand_adj_in_cutout"]]
        )
        if not tagged_pool:
            continue
        n_pick = min(max_per_op, len(tagged_pool))
        chosen_idx = rng.choice(len(tagged_pool), size=n_pick, replace=False)
        for i in chosen_idx:
            (base_root, adj_root), pair_source = tagged_pool[i]
            strategy = "adjacent_in_cutout_junction" if pair_source == "adj_to_base_root" else "rand_adj_in_cutout_junction"
            controls.append(ControlSample(
                source_operation_id=cand["op_id"],
                strategy=strategy,
                is_correct=True,
                is_merge=False,
                root_id=cand["root_id"],
                species=cand["species"],
                interface_point_nm=cand["interface_point_nm"],
                before_root_ids=[base_root, adj_root],
                after_root_ids=[base_root, adj_root],
                before_timestamp=cand["timestamp"] - 1,
                after_timestamp=cand["timestamp"] + 1,
                metadata={
                    "adjacent_root": adj_root,
                    "base_root": base_root,
                    "source_is_merge": cand["is_merge"],
                    "pair_source": pair_source,
                    "cutout_timestamp": cand["cutout_timestamp"],
                },
            ))

    if max_controls and len(controls) > max_controls:
        idx = rng.permutation(len(controls))[:max_controls]
        controls = [controls[i] for i in sorted(idx)]

    logger.info(
        f"junction adj phase 3: {len(controls)} controls "
        f"(max_per_op={max_per_op}, cap={max_controls or 'none'})"
    )
    return controls


# ---------------------------------------------------------------------------
# job derivation from operations / controls
# ---------------------------------------------------------------------------


def _op_hash(op_id: int, task: str, suffix: str = "") -> str:
    """Deterministic hash for a render job."""
    s = f"{op_id}:{task}:{suffix}"
    return hashlib.md5(s.encode()).hexdigest()[:12]


def derive_jobs_from_operation(
    op: ConnectomeOperation,
    tasks: List[str],
    *,
    both_endpoints: bool = True,
    extent_nm: float = DEFAULT_VIEW_EXTENT_NM,
    major_before: Optional[List[int]] = None,
    major_after: Optional[List[int]] = None,
    em_tasks: Optional[List[str]] = None,
    em_config: Optional[Dict[str, Any]] = None,
) -> List[RenderJob]:
    """Derive render jobs from a single ConnectomeOperation.

    If major_before / major_after are provided, they are the ranked (by L2
    count, descending) roots to use for rendering. Otherwise falls back to
    the raw before/after_root_ids.
    """
    jobs = []
    center = np.array(op.interface_point_nm)

    before = major_before if major_before is not None else op.before_root_ids
    after = major_after if major_after is not None else op.after_root_ids

    # latest_root_id: largest after-root (ranked first when major_after given)
    latest_root_id = after[0] if after else op.root_id
    base_meta = {
        "operation_id": str(op.operation_id),
        "root_id": op.root_id,
        "latest_root_id": latest_root_id,
        "species": op.species,
        "is_merge": op.is_merge,
        "timestamp": op.timestamp,
        "source_type": "operation",
        "interface_point_nm": op.interface_point_nm,
        "before_root_ids": op.before_root_ids,   # original (provenance)
        "after_root_ids": op.after_root_ids,      # original (provenance)
        "view_extent_nm": extent_nm,
    }

    # EM fields for jobs that qualify
    em_tasks = em_tasks or []
    em_cfg = em_config or {}

    def _em_fields(task: str) -> dict:
        if task not in em_tasks:
            return {}
        # merge tasks: op.timestamp (both roots resolvable right before merge materializes)
        # split tasks: op.timestamp (untested — kept as-is for now)
        if op.is_merge:
            em_ts = op.timestamp
        else:
            em_ts = op.timestamp
        return {
            "em_enabled": True,
            "em_window_nm": em_cfg.get("window_nm", 5000),
            "em_views": em_cfg.get("views", ["xy", "xz", "yz"]),
            "em_timestamp": em_ts,
        }

    if op.is_merge:
        # merge op corrects split error → endpoint tasks
        if "endpoint_error_id" in tasks and len(before) >= 1:
            qt, ans = TASK_QUESTION_CONFIG["endpoint_error_id"]
            roots_to_render = before if both_endpoints else before[:1]
            for i, rid in enumerate(roots_to_render):
                jobs.append(RenderJob(
                    job_id=_op_hash(op.operation_id, "endpoint_error_id", str(rid)),
                    task="endpoint_error_id",
                    root_ids=[rid],
                    center_nm=center,
                    colors=[COLOR_SINGLE],
                    answer="error",
                    question_type=qt,
                    answer_space=ans,
                    metadata={**base_meta, "rendered_root_id": rid, "segment_index": i},
                    extent_nm=extent_nm,
                    **_em_fields("endpoint_error_id"),
                ))

        if "endpoint_error_corr" in tasks and len(before) >= 2:
            qt, ans = TASK_QUESTION_CONFIG["endpoint_error_corr"]
            jobs.append(RenderJob(
                job_id=_op_hash(op.operation_id, "endpoint_error_corr"),
                task="endpoint_error_corr",
                root_ids=before[:2],
                center_nm=center,
                colors=[COLOR_A, COLOR_B],
                answer=True,  # should merge
                question_type=qt,
                answer_space=ans,
                metadata={**base_meta, "segment1_id": before[0], "segment2_id": before[1], "same_neuron": True},
                extent_nm=extent_nm,
                **_em_fields("endpoint_error_corr"),
            ))

    else:
        # split op corrects merge error → junction tasks
        if "junction_error_id" in tasks and len(before) >= 1:
            qt, ans = TASK_QUESTION_CONFIG["junction_error_id"]
            jobs.append(RenderJob(
                job_id=_op_hash(op.operation_id, "junction_error_id"),
                task="junction_error_id",
                root_ids=[before[0]],
                center_nm=center,
                colors=[COLOR_SINGLE],
                answer="error",
                question_type=qt,
                answer_space=ans,
                metadata=base_meta,
                extent_nm=extent_nm,
                **_em_fields("junction_error_id"),
            ))

        if "junction_error_corr" in tasks and len(after) >= 2:
            qt, ans = TASK_QUESTION_CONFIG["junction_error_corr"]
            jobs.append(RenderJob(
                job_id=_op_hash(op.operation_id, "junction_error_corr"),
                task="junction_error_corr",
                root_ids=after[:2],
                center_nm=center,
                colors=[COLOR_A, COLOR_B],
                answer=True,  # split is correct
                question_type=qt,
                answer_space=ans,
                metadata={
                    **base_meta,
                    "sources": op.source_coords_nm,
                    "sinks": op.sink_coords_nm,
                },
                extent_nm=extent_nm,
                **_em_fields("junction_error_corr"),
            ))

        if "junction_corr_proposal" in tasks and len(after) >= 2:
            qt, ans = TASK_QUESTION_CONFIG["junction_corr_proposal"]
            sources = op.source_coords_nm
            sinks = op.sink_coords_nm
            jobs.append(RenderJob(
                job_id=_op_hash(op.operation_id, "junction_corr_proposal"),
                task="junction_corr_proposal",
                root_ids=after[:2],
                center_nm=center,
                colors=[COLOR_A, COLOR_B],
                answer=(sources, sinks),
                question_type=qt,
                answer_space=ans,
                metadata={**base_meta, "sources": sources, "sinks": sinks},
                extent_nm=extent_nm,
                **_em_fields("junction_corr_proposal"),
            ))

    return jobs


def derive_jobs_from_control(
    ctrl: ControlSample,
    tasks: List[str],
    *,
    both_endpoints: bool = True,
    extent_nm: float = DEFAULT_VIEW_EXTENT_NM,
    major_before: Optional[List[int]] = None,
    major_after: Optional[List[int]] = None,
    em_tasks: Optional[List[str]] = None,
    em_config: Optional[Dict[str, Any]] = None,
) -> List[RenderJob]:
    """Derive render jobs from a single ControlSample.

    If major_before / major_after are provided, they are the ranked (by L2
    count, descending) roots to use for rendering.
    """
    jobs = []
    center = np.array(ctrl.interface_point_nm)

    before = major_before if major_before is not None else ctrl.before_root_ids
    after = major_after if major_after is not None else ctrl.after_root_ids

    # latest_root_id: the primary rendered root (before[0] is the "current" state
    # for an inverted control — it's the root that actually gets rendered)
    latest_root_id = before[0] if before else ctrl.root_id
    base_meta = {
        "operation_id": str(ctrl.source_operation_id),   # consistent with op metadata
        "source_operation_id": str(ctrl.source_operation_id),
        "root_id": ctrl.root_id,
        "latest_root_id": latest_root_id,
        "species": ctrl.species,
        "is_merge": ctrl.is_merge,
        "strategy": ctrl.strategy,
        "is_correct": ctrl.is_correct,
        "source_type": "control",
        "interface_point_nm": ctrl.interface_point_nm,
        "before_root_ids": ctrl.before_root_ids,   # original (provenance)
        "after_root_ids": ctrl.after_root_ids,      # original (provenance)
        "view_extent_nm": extent_nm,
        **(ctrl.metadata or {}),  # propagate ControlSample-specific fields (pair_source, etc.)
    }

    # EM fields for jobs that qualify
    em_tasks = em_tasks or []
    em_cfg = em_config or {}

    def _em_fields(task: str) -> dict:
        if task not in em_tasks:
            return {}
        meta = ctrl.metadata or {}
        if "cutout_timestamp" not in meta:
            raise ValueError(
                f"control {ctrl.source_operation_id} strategy={ctrl.strategy} "
                f"is missing cutout_timestamp in metadata — cannot determine EM seg timestamp"
            )
        return {
            "em_enabled": True,
            "em_window_nm": em_cfg.get("window_nm", 5000),
            "em_views": em_cfg.get("views", ["xy", "xz", "yz"]),
            "em_timestamp": meta["cutout_timestamp"],
        }

    if ctrl.is_merge:
        # proposed action is merge (WRONG — original was a split)
        # → endpoint tasks with "control" / False answers
        if "endpoint_error_id" in tasks and len(before) >= 1:
            qt, ans = TASK_QUESTION_CONFIG["endpoint_error_id"]
            roots_to_render = before if both_endpoints else before[:1]
            for i, rid in enumerate(roots_to_render):
                jobs.append(RenderJob(
                    job_id=_op_hash(ctrl.source_operation_id, "endpoint_error_id", f"ctrl_{rid}"),
                    task="endpoint_error_id",
                    root_ids=[rid],
                    center_nm=center,
                    colors=[COLOR_SINGLE],
                    answer="control",
                    question_type=qt,
                    answer_space=ans,
                    metadata={**base_meta, "latest_root_id": rid, "rendered_root_id": rid, "segment_index": i},
                    extent_nm=extent_nm,
                    **_em_fields("endpoint_error_id"),
                ))

        if "endpoint_error_corr" in tasks and len(before) >= 2:
            qt, ans = TASK_QUESTION_CONFIG["endpoint_error_corr"]
            ctrl_suffix = f"ctrl_{before[0]}_{before[1]}"
            jobs.append(RenderJob(
                job_id=_op_hash(ctrl.source_operation_id, "endpoint_error_corr", ctrl_suffix),
                task="endpoint_error_corr",
                root_ids=before[:2],
                center_nm=center,
                colors=[COLOR_A, COLOR_B],
                answer=False,  # should NOT merge
                question_type=qt,
                answer_space=ans,
                metadata={**base_meta, "segment1_id": before[0], "segment2_id": before[1], "same_neuron": False},
                extent_nm=extent_nm,
                **_em_fields("endpoint_error_corr"),
            ))

    else:
        if not ctrl.is_correct:
            # proposed action is split (WRONG — original was a merge)
            # → junction tasks with "control" / False answers
            if "junction_error_id" in tasks and len(before) >= 1:
                qt, ans = TASK_QUESTION_CONFIG["junction_error_id"]
                jobs.append(RenderJob(
                    job_id=_op_hash(ctrl.source_operation_id, "junction_error_id", "ctrl"),
                    task="junction_error_id",
                    root_ids=[before[0]],
                    center_nm=center,
                    colors=[COLOR_SINGLE],
                    answer="control",
                    question_type=qt,
                    answer_space=ans,
                    metadata=base_meta,
                    extent_nm=extent_nm,
                    **_em_fields("junction_error_id"),
                ))

            if "junction_error_corr" in tasks and len(after) >= 2:
                qt, ans = TASK_QUESTION_CONFIG["junction_error_corr"]
                jobs.append(RenderJob(
                    job_id=_op_hash(ctrl.source_operation_id, "junction_error_corr", "ctrl"),
                    task="junction_error_corr",
                    root_ids=after[:2],   # show both pieces of the (wrong) proposed split
                    center_nm=center,
                    colors=[COLOR_A, COLOR_B],
                    answer=False,  # split is NOT correct
                    question_type=qt,
                    answer_space=ans,
                    metadata=base_meta,
                    extent_nm=extent_nm,
                    **_em_fields("junction_error_corr"),
                ))

        else:
            # adjacent_in_cutout_junction: two genuinely-separate neurons, split IS correct
            if "junction_error_id" in tasks and len(before) >= 2:
                qt, ans = TASK_QUESTION_CONFIG["junction_error_id"]
                jobs.append(RenderJob(
                    job_id=_op_hash(ctrl.source_operation_id, "junction_error_id", f"adj_{before[0]}_{before[1]}"),
                    task="junction_error_id",
                    root_ids=before[:2],
                    center_nm=center,
                    colors=[COLOR_SINGLE, COLOR_SINGLE],  # render as if merged — mimics the visual of a junction error
                    answer="error",  # yes, junction error here
                    question_type=qt,
                    answer_space=ans,
                    metadata={
                        **base_meta,
                        "segment1_id": before[0],
                        "segment2_id": before[1],
                    },
                    extent_nm=extent_nm,
                    **_em_fields("junction_error_id"),
                ))

            if "junction_error_corr" in tasks and len(before) >= 2:
                qt, ans = TASK_QUESTION_CONFIG["junction_error_corr"]
                jobs.append(RenderJob(
                    job_id=_op_hash(ctrl.source_operation_id, "junction_error_corr", f"adj_{before[0]}_{before[1]}"),
                    task="junction_error_corr",
                    root_ids=before[:2],
                    center_nm=center,
                    colors=[COLOR_A, COLOR_B],
                    answer=True,  # split IS correct
                    question_type=qt,
                    answer_space=ans,
                    metadata={
                        **base_meta,
                        "segment1_id": before[0],
                        "segment2_id": before[1],
                    },
                    extent_nm=extent_nm,
                    **_em_fields("junction_error_corr"),
                ))

    return jobs


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def random_high_contrast_pair(rng: np.random.Generator) -> List[str]:
    """Generate two hex colors with guaranteed high contrast (hue separation 120-240°)."""
    import colorsys
    h1 = rng.uniform(0, 1.0)
    h2 = (h1 + rng.uniform(1 / 3, 2 / 3)) % 1.0
    s, v = 0.85, 0.85
    rgb1 = colorsys.hsv_to_rgb(h1, s, v)
    rgb2 = colorsys.hsv_to_rgb(h2, s, v)
    return [
        f"#{int(rgb1[0]*255):02x}{int(rgb1[1]*255):02x}{int(rgb1[2]*255):02x}",
        f"#{int(rgb2[0]*255):02x}{int(rgb2[1]*255):02x}{int(rgb2[2]*255):02x}",
    ]


def _render_pass(
    mesh_specs: List[MeshSpec],
    job: RenderJob,
    output_dir: Path,
    extent_nm: float,
    prefix: str,
    canvas_size_px: Tuple[int, int],
    viewer: Any = None,
    material_mode: Optional[str] = None,
    background_color: Optional[str] = None,
    show_projection_legend: bool = True,
) -> Optional[Dict[str, Path]]:
    """Execute one render pass (detail or minimap). Returns {view: path} or None."""
    image_paths, _ = render_neuron_views(
        root_id=job.root_ids[0],
        meshes=mesh_specs,
        neuron_graph=None,
        center_coord=job.center_nm,
        center_node=None,
        extent_nm=extent_nm,
        output_dir=output_dir,
        base_name_prefix=prefix,
        show_projection_legend=show_projection_legend,
        include_graph=False,
        canvas_size_px=canvas_size_px,
        mesh_crop_enabled=True,
        mesh_transparency=False,
        viewer=viewer,
        material_mode=material_mode,
        background_color=background_color,
    )
    return image_paths


def _compute_mesh_extent(meshes: Dict[int, cv_mesh.Mesh], root_ids: List[int]) -> float:
    """Compute the maximum half-extent (nm) across all meshes for minimap zoom."""
    all_verts = []
    for rid in root_ids:
        m = meshes.get(rid)
        if m is not None and hasattr(m, "vertices") and len(m.vertices) > 0:
            all_verts.append(m.vertices)
    if not all_verts:
        return 50000.0
    combined = np.concatenate(all_verts, axis=0)
    half_dims = (combined.max(axis=0) - combined.min(axis=0)) / 2.0
    return float(half_dims.max())


def render_job(
    job: RenderJob,
    meshes: Dict[int, cv_mesh.Mesh],
    render_dir: Path,
    viewer: Any = None,
    canvas_size_px: Tuple[int, int] = (512, 512),
    minimap_config: Optional[Dict[str, Any]] = None,
    render_modes: Optional[List[str]] = None,
    uni_color_mode: Optional[str] = None,
    geometry_config: Optional[Dict[str, Any]] = None,
    species: str = "",
) -> Optional[Tuple[List[str], List[str]]]:
    """Render a single job. Returns (image_paths, image_types) or None on failure.

    Must be called from the main thread (GPU access).
    If viewer is provided, reuses it (scene cleared internally by render_neuron_scene).
    render_modes: list of "colored", "geometry", "geometry_single", "normals",
        "depth", and/or "mask". Default: ["colored"]. NOTE: "mask", "depth", and
        "normals" are DEPRECATED — use "geometry" for foundation-aligned 7-channel
        output. "geometry_single" renders a single-mesh foundation tensor of
        meta["after_root_ids"][0] (silent no-op if empty).
    uni_color_mode: None | "add" | "replace".  If "add", emit extra uni_colored_* views
        alongside the normal colored views.  If "replace", emit only uni_colored_* views
        (suppresses the standard colored pass).  All roots rendered in a single neutral color.
    geometry_config: dict with optional keys {resolution:int, angles:tuple[str,...]}.
        Extent is always job.extent_nm (inherits view_extent_nm/center_jitter_nm).
    species: species tag written into the MeshScene metadata.
    """
    modes = render_modes or ["colored"]
    mesh_specs = []
    for rid, color in zip(job.root_ids, job.colors):
        m = meshes.get(rid)
        if m is None:
            logger.warning(f"missing mesh for root {rid} in job {job.job_id}, skipping")
            return None
        mesh_specs.append(MeshSpec(root_id=rid, mesh=m, color=color, opacity=1.0))

    output_dir = render_dir / job.task / job.job_id
    output_dir.mkdir(parents=True, exist_ok=True)

    ordered_paths: List[str] = []
    ordered_types: List[str] = []

    # --- detail passes (one per render mode) ---
    for mode in modes:
        if mode in ("mask", "depth"):
            continue  # handled separately below (per-root isolation)
        if mode == "colored" and uni_color_mode == "replace":
            continue  # suppressed — uni_colored pass emitted instead below
        if mode == "geometry":
            continue  # handled separately below (foundation MeshScene)
        if mode == "geometry_single":
            continue  # handled separately below (single-mesh MeshScene)
        if mode == "normals":
            logger.warning(
                "render_mode 'normals' is DEPRECATED — use 'geometry' for foundation-aligned 7ch output"
            )
            material = "normals"
            suffix = "_normal"
            type_prefix = "normal"
        elif mode == "depth":
            material = "depth"
            suffix = "_depth"
            type_prefix = "depth"
        else:
            material = None
            suffix = ""
            type_prefix = "colored"
        prefix = f"{job.job_id}{suffix}"

        try:
            paths = _render_pass(
                mesh_specs, job, output_dir, job.extent_nm,
                prefix, canvas_size_px, viewer, material_mode=material,
            )
        except Exception as e:
            logger.error(f"render failed for job {job.job_id} mode={mode}: {e}")
            if mode == "colored":
                return None
            continue

        if paths is None:
            if mode == "colored":
                return None
            continue
        for view in ("front", "side", "top"):
            p = paths.get(view)
            if p is None:
                if mode == "colored":
                    logger.warning(f"missing {view} view for job {job.job_id}")
                    return None
                continue
            ordered_paths.append(str(p))
            ordered_types.append(f"{type_prefix}_{view}")

    # --- uni-colored pass (all roots in one neutral color) ---
    if uni_color_mode in ("add", "replace"):
        uni_specs = [
            MeshSpec(root_id=spec.root_id, mesh=spec.mesh, color=COLOR_SINGLE, opacity=1.0)
            for spec in mesh_specs
        ]
        uni_prefix = f"{job.job_id}_uni"
        try:
            uni_paths = _render_pass(
                uni_specs, job, output_dir, job.extent_nm,
                uni_prefix, canvas_size_px, viewer,
            )
        except Exception as e:
            logger.warning(f"uni_colored render failed for job {job.job_id}: {e}")
            uni_paths = None
        if uni_paths is not None:
            for view in ("front", "side", "top"):
                p = uni_paths.get(view)
                if p is not None:
                    ordered_paths.append(str(p))
                    ordered_types.append(f"uni_colored_{view}")

    # --- mask passes (render each segment solo on black → binary mask) ---
    if "mask" in modes:
        logger.warning(
            "render_mode 'mask' is DEPRECATED — use 'geometry' for foundation-aligned 7ch output"
        )
        from PIL import Image
        for seg_idx in range(len(job.root_ids)):
            label = "A" if seg_idx == 0 else "B"
            # Only show this segment, white on black
            mask_specs = [
                MeshSpec(
                    root_id=spec.root_id,
                    mesh=spec.mesh,
                    color="#FFFFFF",
                    opacity=1.0,
                    visible=(i == seg_idx),
                )
                for i, spec in enumerate(mesh_specs)
            ]
            mask_prefix = f"{job.job_id}_mask_{label}"
            try:
                mask_paths = _render_pass(
                    mask_specs, job, output_dir, job.extent_nm,
                    mask_prefix, canvas_size_px, viewer,
                    background_color="#000000",
                    show_projection_legend=False,
                )
            except Exception as e:
                logger.warning(f"mask render failed for job {job.job_id} seg={label}: {e}")
                mask_paths = None

            if mask_paths is not None:
                # flush async saves so mask PNGs are on disk before we re-read them
                from rendering.render_pipeline import flush_saves, _async_save
                flush_saves()
                for view in ("front", "side", "top"):
                    p = mask_paths.get(view)
                    if p is None:
                        continue
                    # Threshold to true binary
                    img = Image.open(p).convert("L")
                    binary = img.point(lambda x: 255 if x > 10 else 0)
                    _async_save(binary, p)
                    ordered_paths.append(str(p))
                    ordered_types.append(f"mask_{label}_{view}")

        # Single-segment: emit empty mask_B
        if len(job.root_ids) == 1:
            from rendering.render_pipeline import _async_save
            for view in ("front", "side", "top"):
                b_path = output_dir / f"{job.job_id}_mask_B_{int(job.extent_nm)}nm_{view}.png"
                empty = Image.new("L", canvas_size_px, 0)
                _async_save(empty, b_path)
                ordered_paths.append(str(b_path))
                ordered_types.append(f"mask_B_{view}")


    # --- per-root depth passes (render each segment solo → grayscale depth) ---
    if "depth" in modes:
        logger.warning(
            "render_mode 'depth' is DEPRECATED — use 'geometry' for foundation-aligned 7ch output"
        )
        for seg_idx in range(len(job.root_ids)):
            label = "A" if seg_idx == 0 else "B"
            depth_specs = [
                MeshSpec(
                    root_id=spec.root_id,
                    mesh=spec.mesh,
                    color="#FFFFFF",
                    opacity=1.0,
                    visible=(i == seg_idx),
                )
                for i, spec in enumerate(mesh_specs)
            ]
            depth_prefix = f"{job.job_id}_depth_{label}"
            try:
                depth_paths = _render_pass(
                    depth_specs, job, output_dir, job.extent_nm,
                    depth_prefix, canvas_size_px, viewer,
                    material_mode="depth",
                    background_color="#000000",
                    show_projection_legend=False,
                )
            except Exception as e:
                logger.warning(f"depth render failed for job {job.job_id} seg={label}: {e}")
                depth_paths = None

            if depth_paths is not None:
                for view in ("front", "side", "top"):
                    p = depth_paths.get(view)
                    if p is not None:
                        ordered_paths.append(str(p))
                        ordered_types.append(f"depth_{label}_{view}")

        # Single-segment: emit empty depth_B
        if len(job.root_ids) == 1:
            from rendering.render_pipeline import _async_save
            for view in ("front", "side", "top"):
                b_path = output_dir / f"{job.job_id}_depth_B_{int(job.extent_nm)}nm_{view}.png"
                empty = Image.new("L", canvas_size_px, 0)
                _async_save(empty, b_path)
                ordered_paths.append(str(b_path))
                ordered_types.append(f"depth_B_{view}")

    # --- geometry pass (foundation 7-channel tensor, saved as one .npy) ---
    if "geometry" in modes:
        try:
            from foundation.scenes.mesh_scene import MeshScene, ViewConfig

            geo_cfg = geometry_config or {}
            geo_resolution = int(geo_cfg.get("resolution", 224))
            geo_angles = tuple(geo_cfg.get("angles", ("front", "side", "top")))
            view_config = ViewConfig(
                angles=geo_angles,
                extents_nm=(float(job.extent_nm),),
                resolution=geo_resolution,
            )

            split_mask_arr: Optional[np.ndarray] = None
            if len(job.root_ids) == 2:
                from foundation.scenes.merge_pair_scene import MergePairScene
                from foundation.scenes.fused_merge_pair_scene import (
                    FusedMergePairScene,
                    split_mask_from_pair_views,
                )

                mesh_a = meshes[job.root_ids[0]]
                mesh_b = meshes[job.root_ids[1]]
                pair_scene = MergePairScene.from_meshes(
                    mesh_a=mesh_a,
                    mesh_b=mesh_b,
                    anchor_nm=job.center_nm,
                    segment_id_a=int(job.root_ids[0]),
                    segment_id_b=int(job.root_ids[1]),
                    species=species,
                    view_config=view_config,
                    skip_contact=True,
                    viewer=_rw_geom_viewer,
                )
                split_mask_arr = split_mask_from_pair_views(pair_scene.views)
                fused = FusedMergePairScene.from_pair_scene(pair_scene)
                scene = MeshScene(
                    views=fused.views,
                    anchor_nm=fused.anchor_nm,
                    segment_id=int(job.root_ids[0]),
                    species=species,
                    view_config=view_config,
                )
            else:
                mesh_a = meshes[job.root_ids[0]]
                scene = MeshScene.from_mesh(
                    mesh=mesh_a,
                    anchor_nm=job.center_nm,
                    segment_id=int(job.root_ids[0]),
                    species=species,
                    view_config=view_config,
                    viewer=_rw_geom_viewer,
                )

            geo_path = output_dir / f"{job.job_id}_geometry.npy"
            np.save(str(geo_path), scene.views)
            ordered_paths.append(str(geo_path))
            ordered_types.append("geometry")

            if split_mask_arr is not None:
                from rendering.render_pipeline import _async_save
                from PIL import Image as _PILImage
                view_angles = list(view_config.view_angles)
                for v_idx, angle in enumerate(view_angles):
                    sm = split_mask_arr[v_idx]
                    sm_img = _PILImage.fromarray(sm, mode="L")
                    sm_path = (
                        output_dir
                        / f"{job.job_id}_split_mask_{int(job.extent_nm)}nm_{angle}.png"
                    )
                    _async_save(sm_img, sm_path)
                    ordered_paths.append(str(sm_path))
                    ordered_types.append(f"split_mask_{angle}")
        except Exception as e:
            logger.warning(f"geometry render failed for job {job.job_id}: {e}")

    # --- geometry_single pass (single post-merge mesh, foundation 7ch tensor) ---
    if "geometry_single" in modes:
        after = job.metadata.get("after_root_ids") or []
        if after:
            single_rid = int(after[0])
            mesh_single = meshes.get(single_rid)
            if mesh_single is not None:
                try:
                    from foundation.scenes.mesh_scene import MeshScene, ViewConfig

                    geo_cfg = geometry_config or {}
                    geo_resolution = int(geo_cfg.get("resolution", 224))
                    geo_angles = tuple(geo_cfg.get("angles", ("front", "side", "top")))
                    view_config_single = ViewConfig(
                        angles=geo_angles,
                        extents_nm=(float(job.extent_nm),),
                        resolution=geo_resolution,
                    )
                    scene_single = MeshScene.from_mesh(
                        mesh=mesh_single,
                        anchor_nm=job.center_nm,
                        segment_id=single_rid,
                        species=species,
                        view_config=view_config_single,
                        viewer=_rw_geom_viewer,
                    )
                    geo_single_path = output_dir / f"{job.job_id}_geometry_single.npy"
                    np.save(str(geo_single_path), scene_single.views)
                    ordered_paths.append(str(geo_single_path))
                    ordered_types.append("geometry_single")
                except Exception as e:
                    logger.warning(f"geometry_single render failed for job {job.job_id}: {e}")

    # --- minimap pass (colored only — normals not useful at macro scale) ---
    mm_cfg = minimap_config or {}
    if mm_cfg.get("enabled", False):
        max_extent = mm_cfg.get("max_extent_nm", 100000)
        mesh_half_extent = _compute_mesh_extent(meshes, job.root_ids)
        minimap_extent = min(max_extent, mesh_half_extent * 1.1)
        minimap_extent = max(minimap_extent, job.extent_nm * 3)

        minimap_prefix = f"{job.job_id}_minimap"
        try:
            mm_paths = _render_pass(
                mesh_specs, job, output_dir, minimap_extent,
                minimap_prefix, canvas_size_px, viewer, material_mode=None,
            )
        except Exception as e:
            logger.warning(f"minimap render failed for job {job.job_id}: {e}")
            mm_paths = None

        if mm_paths is not None:
            for view in ("front", "side", "top"):
                p = mm_paths.get(view)
                if p is not None:
                    ordered_paths.append(str(p))
                    ordered_types.append(f"minimap_{view}")

    # ensure all async PNG saves are flushed before returning paths
    from rendering.render_pipeline import flush_saves
    flush_saves()

    return ordered_paths, ordered_types


# ---------------------------------------------------------------------------
# checkpoint reload (reconstruct questions from rendered images)
# ---------------------------------------------------------------------------


def reload_questions_from_renders(
    all_jobs: List[RenderJob],
    job_ids: set,
    render_dir: Path,
) -> List[DatasetQuestion]:
    """Reload questions for previously rendered jobs (checkpoint resume).

    Detects image types from filenames:
    - detail views: *_{view}.png (not minimap/em) → colored_{view}
    - normal views: *_normal_{view}.png → normal_{view}
    - minimap views: *_minimap_{view}.png → minimap_{view}
    - EM views: *_em_{plane}.png → em_{plane}
    """
    if not job_ids:
        return []

    job_map = {j.job_id: j for j in all_jobs}
    questions = []

    for jid in job_ids:
        job = job_map.get(jid)
        if job is None:
            continue

        output_dir = render_dir / job.task / jid
        try:
            entries = sorted(os.listdir(output_dir))
        except (FileNotFoundError, NotADirectoryError):
            continue

        image_paths: List[str] = []
        image_types: List[str] = []

        def _pick(predicate) -> Optional[str]:
            for name in entries:
                if predicate(name):
                    return str(output_dir / name)
            return None

        # detail colored views (*_{view}.png excluding tagged variants)
        excl = ("_minimap_", "_normal_", "_em_", "_uni_", "_depth_", "_mask_")
        for view in ("front", "side", "top"):
            suffix = f"_{view}.png"
            hit = _pick(lambda n, s=suffix: n.endswith(s) and not any(t in n for t in excl))
            if hit is None:
                break
            image_paths.append(hit)
            image_types.append(f"colored_{view}")

        # uni-colored views
        for view in ("front", "side", "top"):
            suffix = f"_{view}.png"
            hit = _pick(lambda n, s=suffix: "_uni_" in n and n.endswith(s))
            if hit:
                image_paths.append(hit)
                image_types.append(f"uni_colored_{view}")

        # detail normal views
        for view in ("front", "side", "top"):
            suffix = f"{view}.png"
            hit = _pick(lambda n, s=suffix: "_normal_" in n and "_minimap_" not in n and n.endswith(s))
            if hit:
                image_paths.append(hit)
                image_types.append(f"normal_{view}")

        # per-root depth views
        for label in ("A", "B"):
            tag = f"_depth_{label}_"
            for view in ("front", "side", "top"):
                suffix = f"_{view}.png"
                hit = _pick(lambda n, t=tag, s=suffix: t in n and n.endswith(s))
                if hit:
                    image_paths.append(hit)
                    image_types.append(f"depth_{label}_{view}")

        # minimap colored views (with normal-tagged exclusion)
        for view in ("front", "side", "top"):
            suffix_long = f"_{view}.png"
            suffix_short = f"_minimap_{view}.png"
            hit = _pick(lambda n, s=suffix_long: "_minimap_" in n and "_normal_" not in n and n.endswith(s))
            if hit is None:
                hit = _pick(lambda n, s=suffix_short: n.endswith(s) and "_normal_" not in n)
            if hit:
                image_paths.append(hit)
                image_types.append(f"minimap_{view}")

        # mask views
        for label in ("A", "B", "BG"):
            tag = f"_mask_{label}_"
            for view in ("front", "side", "top"):
                suffix = f"_{view}.png"
                hit = _pick(lambda n, t=tag, s=suffix: t in n and n.endswith(s))
                if hit:
                    image_paths.append(hit)
                    image_types.append(f"mask_{label}_{view}")

        # geometry tensor
        geo_name = f"{jid}_geometry.npy"
        if geo_name in entries:
            image_paths.append(str(output_dir / geo_name))
            image_types.append("geometry")

        # geometry_single tensor
        geo_single_name = f"{jid}_geometry_single.npy"
        if geo_single_name in entries:
            image_paths.append(str(output_dir / geo_single_name))
            image_types.append("geometry_single")

        # EM views
        for em_view in ("xy", "xz", "yz", "best"):
            suffix = f"_em_{em_view}.png"
            hit = _pick(lambda n, s=suffix: n.endswith(s))
            if hit:
                image_paths.append(hit)
                image_types.append(f"em_{em_view}")

        # need at least colored, mask, or geometry views to be valid
        has_colored = any(t.startswith("colored_") for t in image_types)
        has_masks = any(t.startswith("mask_") for t in image_types)
        has_geometry = "geometry" in image_types
        if not has_colored and not has_masks and not has_geometry:
            continue

        meta = dict(job.metadata)
        meta["image_types"] = image_types

        questions.append(DatasetQuestion(
            question_type=job.question_type,
            answer_space=job.answer_space,
            answer=job.answer,
            images=image_paths,
            metadata=meta,
            sample_hash=jid,
        ))

    return questions


# ---------------------------------------------------------------------------
# EM slice rendering (numpy/PIL, no plotly/kaleido)
# ---------------------------------------------------------------------------


@dataclass
class EMCutoutResult:
    """Result of fetching + resampling an EM cutout for overlay rendering."""
    em_volume: np.ndarray          # 3D uint8 grayscale (isotropic)
    root_masks: Dict[int, np.ndarray]  # root_id → 3D bool mask (isotropic)
    root_ids: List[int]            # which roots are present
    resolution_nm: np.ndarray      # [x, y, z] after isotropic resampling


def resample_to_isotropic(
    volume: np.ndarray,
    resolution_nm: Tuple[float, float, float],
    order: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Resample volume to isotropic voxels (finest axis resolution).

    Args:
        volume: 3D array (x, y, z).
        resolution_nm: (rx, ry, rz) in nm.
        order: interpolation order (1=linear for EM, 0=nearest for labels).

    Returns:
        (resampled_volume, new_resolution_nm).
    """
    from scipy.ndimage import zoom

    res = np.array(resolution_nm, dtype=float)
    target = res.min()
    zoom_factors = res / target

    if np.allclose(zoom_factors, 1.0):
        return volume, np.full(3, target)

    resampled = zoom(volume, zoom_factors, order=order)
    return resampled, np.full(3, target)


INTERFACE_PAD_FACTOR = 1.5  # fetch 50% larger volume to allow recentering


@profiled("fetch_em_cutout")
def fetch_em_cutout(
    species: str,
    center_nm: np.ndarray,
    root_ids: List[int],
    window_nm: int = 5000,
    timestamp: Optional[float] = None,
    client=None,
    cv_em=None,
    cv_seg=None,
    skip_recenter: bool = False,
) -> Optional[EMCutoutResult]:
    """Fetch EM + seg volumes, recenter on segment interface, build per-root masks.

    Fetches a padded volume (1.3×), uses distance transforms to find the
    closest-approach point between the two root segments, then crops back
    to the requested window size centered on the interface.

    Args:
        species: species identifier.
        center_nm: [x, y, z] center position in nm.
        root_ids: roots to build masks for.
        window_nm: cutout half-extent in nm (isotropic).
        timestamp: UNIX timestamp for segmentation state. None = latest.
        client: CAVEclient (reuse across calls).
        cv_em: shared EM CloudVolume.
        cv_seg: shared seg CloudVolume.

    Returns:
        EMCutoutResult with isotropic volumes, or None on failure.
    """
    import datetime as _dt

    if client is None:
        client = get_client_for_species(species)

    ts_dt = _dt.datetime.fromtimestamp(timestamp, tz=_dt.timezone.utc) if timestamp else None
    quiet_logger = logging.getLogger("em_cutout")
    quiet_logger.setLevel(logging.WARNING)

    padded_window = int(window_nm * INTERFACE_PAD_FACTOR)

    try:
        fetcher = EMDataFetcher(
            species,
            timestamp=ts_dt if ts_dt else "detect",
            client=client,
            logger=quiet_logger,
            quiet=True,
            cv_em=cv_em,
            cv_seg=cv_seg,
        )
        em_acc, seg_acc = fetcher.fetch_cutout(
            position_nm=center_nm.tolist(),
            window_size_nm=padded_window,
            with_roots=True,
        )
    except Exception as e:
        logger.warning(f"EM cutout fetch failed at {center_nm}: {e}")
        return None

    # find interface point and recenter
    t_iface = time.monotonic()
    interface_center = center_nm
    if len(root_ids) >= 2 and not skip_recenter:
        iface_pt = seg_acc.find_interface_point(
            root_ids[0], root_ids[1], inner_frac=0.3,
        )
        if iface_pt is not None:
            interface_center = iface_pt
            logger.debug(
                f"interface recentered: {center_nm} → {iface_pt} "
                f"(Δ={np.linalg.norm(iface_pt - center_nm):.0f}nm)"
            )
    Profiler.record("fetch_em_cutout.find_interface", time.monotonic() - t_iface)

    # crop: use min(window_nm, max_margin_to_edge) so we stay centered on
    # the interface. with 1.3× pad + inner 30% search, worst case ≥ 0.91×window_nm.
    t_crop = time.monotonic()
    try:
        iface_arr = np.asarray(interface_center)
        padded_min = np.array(seg_acc.min_nm)
        padded_max = np.array(seg_acc.max_nm)
        max_margin = float(np.min(np.minimum(iface_arr - padded_min, padded_max - iface_arr)))
        crop_extent = min(window_nm, max_margin)
        em_acc = em_acc.crop(interface_center, crop_extent)
        seg_acc = seg_acc.crop(interface_center, crop_extent)
    except Exception as e:
        logger.warning(f"crop failed, using padded volume as-is: {e}")
    Profiler.record("fetch_em_cutout.crop", time.monotonic() - t_crop)

    # raw volumes (anisotropic)
    em_vol = np.array(em_acc.cutout, dtype=np.uint8)
    raw_resolution = np.array(fetcher.resolution, dtype=float)

    # build per-root masks from the batch-resolved roots volume
    t_masks = time.monotonic()
    roots_cutout = seg_acc.roots_cutout
    unique_roots = np.unique(roots_cutout[roots_cutout != 0])
    root_masks_raw: Dict[int, np.ndarray] = {}
    for rid in root_ids:
        mask = roots_cutout == rid
        root_masks_raw[rid] = mask
        if not mask.any():
            logger.warning(f"root {rid}: not found in cutout roots (ts={timestamp}, unique_roots_in_cutout={len(unique_roots)}, requested={root_ids})")
    Profiler.record("fetch_em_cutout.build_masks", time.monotonic() - t_masks)

    both_found = all(root_masks_raw[rid].any() for rid in root_ids)
    Profiler.count("fetch_em_cutout.both_roots_present", int(both_found))
    Profiler.count("fetch_em_cutout.interface_recentered", int(interface_center is not center_nm))

    # resample to isotropic
    t_resample = time.monotonic()
    em_iso, iso_res = resample_to_isotropic(em_vol, tuple(raw_resolution), order=1)
    root_masks_iso: Dict[int, np.ndarray] = {}
    for rid, mask in root_masks_raw.items():
        mask_iso, _ = resample_to_isotropic(
            mask.astype(np.uint8), tuple(raw_resolution), order=0,
        )
        root_masks_iso[rid] = mask_iso > 0
    Profiler.record("fetch_em_cutout.resample_isotropic", time.monotonic() - t_resample)

    return EMCutoutResult(
        em_volume=em_iso,
        root_masks=root_masks_iso,
        root_ids=list(root_ids),
        resolution_nm=iso_res,
    )


# ---------------------------------------------------------------------------
# Oblique plane search & slicing
# ---------------------------------------------------------------------------

def _rot_matrix(axis: int, angle_rad: float) -> np.ndarray:
    """Rotation matrix around axis 0/1/2."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    if axis == 0:
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    elif axis == 1:
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    else:
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def generate_candidate_normals(step_deg: float = 10) -> List[np.ndarray]:
    """Generate ~51 unique plane normals by rotating 3 cardinal axes in 10° steps."""
    n_per_axis = int(180 / step_deg) + 1
    normals = []
    half = (n_per_axis - 1) // 2
    angles_deg = np.linspace(-half * step_deg, half * step_deg, n_per_axis)
    rot_axes = [1, 2, 0]
    for card_ax, rot_ax in enumerate(rot_axes):
        base = np.zeros(3)
        base[card_ax] = 1.0
        for a in angles_deg:
            n = _rot_matrix(rot_ax, np.radians(a)) @ base
            idx = np.argmax(np.abs(n))
            if n[idx] < 0:
                n = -n
            normals.append(n / np.linalg.norm(n))
    unique: List[np.ndarray] = []
    for n in normals:
        if not any(np.dot(n, u) > 0.99 for u in unique):
            unique.append(n)
    return unique


def _plane_basis(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Orthonormal (u, v) tangent vectors for a plane with given normal."""
    n = normal / np.linalg.norm(normal)
    ref = np.zeros(3)
    ref[np.argmin(np.abs(n))] = 1.0
    u = np.cross(n, ref)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    v /= np.linalg.norm(v)
    return u, v


def _plane_bases_batch(normals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Batch-compute (u, v) plane basis vectors for all normals at once."""
    N = normals.shape[0]
    abs_n = np.abs(normals)
    ref_idx = np.argmin(abs_n, axis=1)
    ref = np.zeros((N, 3))
    ref[np.arange(N), ref_idx] = 1.0
    us = np.cross(normals, ref)
    us /= np.linalg.norm(us, axis=1, keepdims=True)
    vs = np.cross(normals, us)
    vs /= np.linalg.norm(vs, axis=1, keepdims=True)
    return us, vs


def slice_volume_oblique(
    volume: np.ndarray,
    normal: np.ndarray,
    center: np.ndarray,
    output_size: int = 256,
    slab_thickness: int = 5,
) -> np.ndarray:
    """Extract mean-pooled oblique slab from 3D volume via map_coordinates."""
    from scipy.ndimage import map_coordinates

    n = normal / np.linalg.norm(normal)
    u, v = _plane_basis(n)
    center = np.asarray(center, dtype=np.float64)
    extent = min(volume.shape) / 2.0
    grid_1d = np.linspace(-extent, extent, output_size)
    uu, vv = np.meshgrid(grid_1d, grid_1d, indexing="xy")
    half = slab_thickness // 2
    slabs = []
    for off in range(-half, -half + slab_thickness):
        coords_3d = (
            center[None, None, :] + off * n[None, None, :]
            + uu[..., None] * u[None, None, :] + vv[..., None] * v[None, None, :]
        )
        sliced = map_coordinates(
            volume.astype(np.float64),
            [coords_3d[..., 0], coords_3d[..., 1], coords_3d[..., 2]],
            order=1, mode="nearest",
        )
        slabs.append(sliced)
    return np.mean(slabs, axis=0)


def slice_mask_oblique(
    mask: np.ndarray,
    normal: np.ndarray,
    center: np.ndarray,
    output_size: int = 256,
    slab_thickness: int = 5,
) -> np.ndarray:
    """Extract any-pooled oblique slab from 3D boolean mask."""
    from scipy.ndimage import map_coordinates

    n = normal / np.linalg.norm(normal)
    u, v = _plane_basis(n)
    center = np.asarray(center, dtype=np.float64)
    extent = min(mask.shape) / 2.0
    grid_1d = np.linspace(-extent, extent, output_size)
    uu, vv = np.meshgrid(grid_1d, grid_1d, indexing="xy")
    half = slab_thickness // 2
    any_mask = np.zeros((output_size, output_size), dtype=bool)
    for off in range(-half, -half + slab_thickness):
        coords_3d = (
            center[None, None, :] + off * n[None, None, :]
            + uu[..., None] * u[None, None, :] + vv[..., None] * v[None, None, :]
        )
        sliced = map_coordinates(
            mask.astype(np.float64),
            [coords_3d[..., 0], coords_3d[..., 1], coords_3d[..., 2]],
            order=0, mode="constant", cval=0,
        )
        any_mask |= sliced > 0.5
    return any_mask


def _inner_mask(shape_2d: Tuple[int, int], frac: float = 0.5) -> np.ndarray:
    """Boolean mask for the central frac of each side."""
    h, w = shape_2d
    margin_h = int(h * (1 - frac) / 2)
    margin_w = int(w * (1 - frac) / 2)
    mask = np.zeros((h, w), dtype=bool)
    mask[margin_h:h - margin_h, margin_w:w - margin_w] = True
    return mask


def _score_area_geomean(masks_2d: Dict[int, np.ndarray], inner_frac: float = 0.5) -> float:
    """Geometric mean of segment areas in the inner region."""
    if not masks_2d:
        return 0.0
    shape = next(iter(masks_2d.values())).shape
    inner = _inner_mask(shape, frac=inner_frac)
    areas = [float((m & inner).sum()) for m in masks_2d.values()]
    product = 1.0
    for a in areas:
        product *= max(a, 1.0)
    return product ** (1.0 / len(areas))


def find_best_plane(
    root_masks: Dict[int, np.ndarray],
    volume_shape: Tuple[int, ...],
    search_slab: int = 1,
    search_size: int = 256,
    angle_step: float = 10,
    inner_frac: float = 0.5,
) -> np.ndarray:
    """Search ~51 candidate normals, return the one maximizing inner-area geomean."""
    center = np.array(volume_shape, dtype=float) / 2.0
    normals = generate_candidate_normals(angle_step)
    best_score = -1.0
    best_normal = normals[0]
    for normal in normals:
        masks_2d = {}
        for rid, m3d in root_masks.items():
            masks_2d[rid] = slice_mask_oblique(m3d, normal, center, search_size, search_slab)
        s = _score_area_geomean(masks_2d, inner_frac=inner_frac)
        if s > best_score:
            best_score = s
            best_normal = normal.copy()
    return best_normal


def find_best_plane_vectorized(
    root_masks: Dict[int, np.ndarray],
    volume_shape: Tuple[int, ...],
    search_slab: int = 1,
    search_size: int = 64,
    angle_step: float = 10,
    inner_frac: float = 0.5,
) -> np.ndarray:
    """Vectorized best-plane search — batches all candidate normals into one
    map_coordinates call per root mask. ~30x faster than find_best_plane."""
    from scipy.ndimage import map_coordinates

    center = np.array(volume_shape, dtype=np.float64) / 2.0
    normals = np.array(generate_candidate_normals(angle_step), dtype=np.float64)
    N = normals.shape[0]
    S = search_size

    us, vs = _plane_bases_batch(normals)
    extent = min(volume_shape) / 2.0
    grid_1d = np.linspace(-extent, extent, S, dtype=np.float64)
    uu, vv = np.meshgrid(grid_1d, grid_1d, indexing="xy")

    base_coords = (
        center[None, None, None, :]
        + uu[None, :, :, None] * us[:, None, None, :]
        + vv[None, :, :, None] * vs[:, None, None, :]
    )

    h, w = S, S
    margin_h = int(h * (1 - inner_frac) / 2)
    margin_w = int(w * (1 - inner_frac) / 2)
    inner = np.zeros((S, S), dtype=bool)
    inner[margin_h:h - margin_h, margin_w:w - margin_w] = True

    half = search_slab // 2
    slab_offsets = np.arange(-half, -half + search_slab, dtype=np.float64)

    log_scores = np.zeros(N, dtype=np.float64)
    n_roots = 0
    for _rid, m3d in root_masks.items():
        mask_f64 = m3d.astype(np.float64)
        any_hit = np.zeros((N, S, S), dtype=bool)
        for off in slab_offsets:
            coords = base_coords + off * normals[:, None, None, :]
            flat = coords.reshape(-1, 3)
            sampled = map_coordinates(
                mask_f64,
                [flat[:, 0], flat[:, 1], flat[:, 2]],
                order=0, mode="constant", cval=0.0,
            )
            any_hit |= sampled.reshape(N, S, S) > 0.5
        areas = (any_hit & inner[None, :, :]).sum(axis=(1, 2)).astype(np.float64)
        np.maximum(areas, 1.0, out=areas)
        log_scores += np.log(areas)
        n_roots += 1

    if n_roots > 0:
        log_scores /= n_roots
    return normals[int(np.argmax(log_scores))].copy()


def render_em_slice(
    em_volume: np.ndarray,
    root_masks: Dict[int, np.ndarray],
    colors: Dict[int, str],
    slice_axis: str,
    focal_root_ids: Optional[List[int]] = None,
    slice_idx: Optional[int] = None,
    output_path: Optional[Path] = None,
    output_size: Tuple[int, int] = (512, 512),
    slab_thickness: int = 5,
    normalize: str = "percentile",
) -> Optional[np.ndarray]:
    """Render EM slice as 3ch RGB: R=grayscale EM, G=mask_A, B=mask_B.

    Args:
        em_volume: 3D uint8 grayscale (isotropic).
        root_masks: root_id → 3D bool mask.
        colors: unused (kept for signature stability).
        slice_axis: "xy", "xz", or "yz".
        focal_root_ids: first root → G channel, second → B. Others ignored.
        slice_idx: index along the slicing axis (None = central).
        output_path: if provided, saves PNG here.
        output_size: (width, height) for output.
        normalize: "percentile" (1st/99th) or "minmax".

    Returns:
        RGB numpy array (H, W, 3) uint8, or None on failure.
    """
    from PIL import Image

    axis_map = {"xy": 2, "xz": 1, "yz": 0}
    if slice_axis not in axis_map:
        logger.error(f"invalid slice_axis: {slice_axis}")
        return None

    ax = axis_map[slice_axis]
    n_slices = em_volume.shape[ax]
    if slice_idx is None:
        slice_idx = n_slices // 2
    slice_idx = max(0, min(slice_idx, n_slices - 1))

    # extract mean-pooled slab centered on slice_idx
    half = slab_thickness // 2
    slab_start = max(0, slice_idx - half)
    slab_end = min(n_slices, slab_start + slab_thickness)
    slab_start = max(0, slab_end - slab_thickness)

    slab = np.take(em_volume, range(slab_start, slab_end), axis=ax)
    em_2d = slab.mean(axis=ax).astype(np.float32)

    # normalize
    if normalize == "percentile":
        p1, p99 = np.percentile(em_2d, [1, 99])
        if p99 > p1:
            em_2d = np.clip((em_2d - p1) / (p99 - p1), 0, 1)
        else:
            em_2d = np.zeros_like(em_2d)
    else:  # minmax
        vmin, vmax = em_2d.min(), em_2d.max()
        if vmax > vmin:
            em_2d = (em_2d - vmin) / (vmax - vmin)
        else:
            em_2d = np.zeros_like(em_2d)

    # RGB: R=EM grayscale, G=mask_A (first focal root), B=mask_B (second focal root)
    em_uint8 = (em_2d * 255).astype(np.uint8)
    em_rgb = np.zeros((*em_uint8.shape, 3), dtype=np.uint8)
    em_rgb[..., 0] = em_uint8

    if focal_root_ids is None:
        focal_root_ids = list(root_masks.keys())
    for idx, rid in enumerate(focal_root_ids[:2]):
        mask_3d = root_masks.get(rid)
        if mask_3d is None:
            continue
        mask_slab = np.take(mask_3d, range(slab_start, slab_end), axis=ax)
        mask_2d = mask_slab.any(axis=ax)
        if not mask_2d.any():
            continue
        em_rgb[mask_2d, 1 + idx] = 255  # G for root 0, B for root 1

    # resize + save
    img = Image.fromarray(em_rgb)
    img = img.resize(output_size, Image.LANCZOS)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(output_path))

    return np.array(img)


def render_em_slice_oblique(
    em_volume: np.ndarray,
    root_masks: Dict[int, np.ndarray],
    colors: Dict[int, str],
    normal: np.ndarray,
    focal_root_ids: Optional[List[int]] = None,
    output_path: Optional[Path] = None,
    output_size: Tuple[int, int] = (512, 512),
    slab_thickness: int = 9,
    normalize: str = "percentile",
) -> Optional[np.ndarray]:
    """Oblique version of render_em_slice (3ch RGB: R=EM, G=mask_A, B=mask_B)."""
    from PIL import Image

    center = np.array(em_volume.shape, dtype=float) / 2.0
    tile = output_size[0]  # assume square

    em_2d = slice_volume_oblique(em_volume, normal, center, tile, slab_thickness).astype(np.float32)

    if normalize == "percentile":
        p1, p99 = np.percentile(em_2d, [1, 99])
        if p99 > p1:
            em_2d = np.clip((em_2d - p1) / (p99 - p1), 0, 1)
        else:
            em_2d = np.zeros_like(em_2d)
    else:
        vmin, vmax = em_2d.min(), em_2d.max()
        if vmax > vmin:
            em_2d = (em_2d - vmin) / (vmax - vmin)
        else:
            em_2d = np.zeros_like(em_2d)

    # RGB: R=EM grayscale, G=mask_A, B=mask_B
    em_uint8 = (em_2d * 255).astype(np.uint8)
    em_rgb = np.zeros((*em_uint8.shape, 3), dtype=np.uint8)
    em_rgb[..., 0] = em_uint8

    if focal_root_ids is None:
        focal_root_ids = list(root_masks.keys())
    for idx, rid in enumerate(focal_root_ids[:2]):
        mask_3d = root_masks.get(rid)
        if mask_3d is None:
            continue
        mask_2d = slice_mask_oblique(mask_3d, normal, center, tile, slab_thickness)
        if not mask_2d.any():
            continue
        em_rgb[mask_2d, 1 + idx] = 255

    img = Image.fromarray(em_rgb)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(output_path))

    return np.array(img)


@profiled("render_em_views")
def render_em_views(
    em_cutout: EMCutoutResult,
    colors: Dict[int, str],
    output_dir: Path,
    prefix: str,
    focal_root_ids: Optional[List[int]] = None,
    views: Optional[List[str]] = None,
    best_plane_mode: Optional[str] = None,
    best_plane_kwargs: Optional[Dict] = None,
    **kwargs,
) -> Dict[str, Path]:
    """Render EM slices for multiple planes. Returns {view: path}.

    Args:
        em_cutout: result from fetch_em_cutout.
        colors: root_id → hex color.
        output_dir: where to save PNGs.
        prefix: filename prefix (e.g. job_id).
        focal_root_ids: roots to highlight (others muted gray).
        views: list of planes to render (default: ["xy", "xz", "yz"]).
        best_plane_mode: None (cardinal only), "replace" (oblique only), "add" (both).
        best_plane_kwargs: search params for find_best_plane.
        **kwargs: forwarded to render_em_slice.

    Returns:
        dict mapping view name → output Path.
    """
    if views is None:
        views = ["xy", "xz", "yz"]

    output_dir.mkdir(parents=True, exist_ok=True)
    result = {}

    # cardinal views (skip if replace mode)
    if best_plane_mode != "replace":
        for view in views:
            out_path = output_dir / f"{prefix}_em_{view}.png"
            render_em_slice(
                em_volume=em_cutout.em_volume,
                root_masks=em_cutout.root_masks,
                colors=colors,
                slice_axis=view,
                focal_root_ids=focal_root_ids,
                output_path=out_path,
                **kwargs,
            )
            result[view] = out_path

    # best oblique plane (if replace or add)
    if best_plane_mode in ("replace", "add"):
        bp = best_plane_kwargs or {}
        best_normal = find_best_plane_vectorized(
            em_cutout.root_masks, em_cutout.em_volume.shape,
            search_slab=bp.get("search_slab", 1),
            search_size=bp.get("search_size", 64),
            angle_step=bp.get("angle_step", 10),
            inner_frac=bp.get("inner_frac", 0.5),
        )
        out_path = output_dir / f"{prefix}_em_best.png"
        render_em_slice_oblique(
            em_volume=em_cutout.em_volume,
            root_masks=em_cutout.root_masks,
            colors=colors, normal=best_normal,
            focal_root_ids=focal_root_ids,
            output_path=out_path,
            output_size=kwargs.get("output_size", (512, 512)),
            slab_thickness=bp.get("render_slab", 9),
            normalize=kwargs.get("normalize", "percentile"),
        )
        result["best"] = out_path

    return result
