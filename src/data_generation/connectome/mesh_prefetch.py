"""Streaming mesh prefetcher.

Encapsulates the ProcessPool-based full-root mesh fetch path so callers can
overlap network/decompression with downstream work (rendering, scene build).

Design:
- Workers populate the full-root diskcache; meshes are NEVER returned across
  the process boundary. Caller re-reads via `get_full_root_mesh()` from the
  main thread (diskcache hit, ~free).
- Backpressure window measured in rids-in-flight, sized off a running average
  of mesh vertex counts × `_BYTES_PER_VERT`.
- Per-rid dedup within a single `stream()` call: shared rids → single fetch.
- BrokenProcessPool recovery: in-flight rids marked failed; pool recreated up
  to `max_pool_recreates` times (no resubmit in v1).

The worker init / fetch functions live here (not in training_data_ops) so this
module is the single source of truth for the mesh-fetch subprocess contract.
"""

from __future__ import annotations

import atexit
import logging
import multiprocessing as mp
import os
import signal
import sys
import time
from concurrent.futures import (
    BrokenExecutor,
    FIRST_COMPLETED,
    Future,
    ProcessPoolExecutor,
    wait,
)
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    TypeVar,
)

import cloudvolume

logger = logging.getLogger(__name__)

T = TypeVar("T")

_BYTES_PER_VERT = 36

DEFAULT_MESH_WORKERS = 6
DEFAULT_PREFETCH_CACHE_MB = 1024
DEFAULT_L2_MESH_CACHE_MB = 2048
DEFAULT_FULL_ROOT_MESH_CACHE_MB = 10240
DEFAULT_FETCH_TIMEOUT_S = 120.0
DEFAULT_MAX_POOL_RECREATES = 5


# ---------------------------------------------------------------------------
# subprocess worker globals + fns
# ---------------------------------------------------------------------------

_mp_cv_seg: Optional[cloudvolume.CloudVolume] = None
_mp_species: Optional[str] = None


def _mp_get_rss_mb() -> float:
    try:
        import resource

        raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return raw / 1024 if sys.platform == "linux" else raw / (1024 * 1024)
    except Exception:
        return -1.0


def _mp_worker_init(
    species: str,
    cache_dir: str,
    l2_cache_bytes: int,
    full_root_cache_bytes: int,
) -> None:
    """Called once per fetch worker at pool startup.

    Each subprocess gets its own CloudVolume + diskcache handle so there's no
    GIL contention with the main process.
    """
    import faulthandler
    import warnings

    faulthandler.enable()
    warnings.filterwarnings("ignore", message=".*deduplication.*")

    # late imports avoid a circular import at module load time
    from connectome.meshes import configure_full_root_cache, configure_mesh_cache
    from training.training_data_ops import _get_cv_seg

    global _mp_cv_seg, _mp_species
    _mp_species = species
    _mp_cv_seg = _get_cv_seg(species)
    configure_mesh_cache(size_limit_bytes=l2_cache_bytes, cache_dir=cache_dir)
    configure_full_root_cache(size_limit_bytes=full_root_cache_bytes, cache_dir=cache_dir)
    logger.info(
        f"mesh worker pid={os.getpid()} ready (RSS={_mp_get_rss_mb():.0f}MB)"
    )


def _mp_fetch_mesh(
    rid: int,
    timeout_s: float,
    max_mesh_vertices: Optional[int],
):
    """Fetch full-root mesh in subprocess. Returns (rid, n_verts_or_none, fetch_seconds).

    Meshes are NOT returned across the process boundary — they're written to
    diskcache by the write-through in `get_full_root_mesh()`.
    """
    from training.training_data_ops import fetch_full_root_mesh

    t0 = time.monotonic()
    try:
        mesh = fetch_full_root_mesh(
            _mp_cv_seg, rid, _mp_species or "", timeout_s=timeout_s,
        )
        nv = (
            len(mesh.vertices)
            if mesh is not None and hasattr(mesh, "vertices")
            else None
        )

        if mesh is not None and max_mesh_vertices is not None and nv is not None:
            if nv > max_mesh_vertices:
                logger.warning(
                    f"root {rid}: {nv:,} verts > {max_mesh_vertices:,}, skip "
                    f"(RSS={_mp_get_rss_mb():.0f}MB)"
                )
                return rid, None, time.monotonic() - t0

        if nv is not None and nv > 1_000_000:
            logger.debug(
                f"root {rid}: {nv:,} verts, RSS={_mp_get_rss_mb():.0f}MB"
            )

        return rid, nv, time.monotonic() - t0
    except Exception as e:
        logger.warning(
            f"subprocess fetch failed root {rid}: {e} (RSS={_mp_get_rss_mb():.0f}MB)"
        )
        return rid, None, time.monotonic() - t0


# ---------------------------------------------------------------------------
# public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PrefetchJob(Generic[T]):
    """One unit of work: caller-defined payload + the rids it needs ready."""

    job_id: str
    rids: List[int]
    payload: T


@dataclass
class PrefetchResult(Generic[T]):
    """Yielded once all rids in a job are diskcache-ready (or definitively failed)."""

    job: PrefetchJob[T]
    failed_rids: List[int] = field(default_factory=list)
    fetch_seconds: float = 0.0


# ---------------------------------------------------------------------------
# MeshPrefetcher
# ---------------------------------------------------------------------------


class MeshPrefetcher(Generic[T]):
    """Streaming full-root mesh prefetcher.

    Usage:
        with MeshPrefetcher(species, mesh_workers=6, prefetch_cache_mb=1024) as pf:
            for result in pf.stream(jobs):
                for rid in result.job.rids:
                    if rid in result.failed_rids:
                        continue
                    mesh = get_full_root_mesh(cv_seg, rid, dataset_id=species)
                    ...

    Jobs are yielded in *completion* order, not submission order.
    """

    def __init__(
        self,
        species: str,
        *,
        mesh_workers: int = DEFAULT_MESH_WORKERS,
        prefetch_cache_mb: int = DEFAULT_PREFETCH_CACHE_MB,
        cache_dir: Optional[str] = None,
        l2_cache_mb: int = DEFAULT_L2_MESH_CACHE_MB,
        full_root_cache_mb: int = DEFAULT_FULL_ROOT_MESH_CACHE_MB,
        fetch_timeout_s: float = DEFAULT_FETCH_TIMEOUT_S,
        max_mesh_vertices: Optional[int] = None,
        max_pool_recreates: int = DEFAULT_MAX_POOL_RECREATES,
    ) -> None:
        self.species = species
        self.mesh_workers = mesh_workers
        self.prefetch_cache_mb = prefetch_cache_mb
        self.cache_dir = cache_dir or os.environ.get("CACHE_DIR", ".cache")
        self.l2_cache_bytes = l2_cache_mb * 1024**2
        self.full_root_cache_bytes = full_root_cache_mb * 1024**2
        self.fetch_timeout_s = fetch_timeout_s
        self.max_mesh_vertices = max_mesh_vertices
        self.max_pool_recreates = max_pool_recreates

        self._executor: Optional[ProcessPoolExecutor] = None
        self._spawn_ctx = mp.get_context("spawn")
        self._pool_recreate_count = 0
        self._worker_pids: Set[int] = set()
        self._atexit_registered = False

    # ------------------------------------------------------------------
    # context management
    # ------------------------------------------------------------------

    def __enter__(self) -> "MeshPrefetcher[T]":
        self._executor = self._create_pool()
        if not self._atexit_registered:
            atexit.register(self._kill_workers)
            self._atexit_registered = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown()

    def shutdown(self) -> None:
        if self._executor is not None:
            try:
                self._executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
            self._executor = None
        self._kill_workers()

    def _kill_workers(self) -> None:
        for pid in list(self._worker_pids):
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
        self._worker_pids.clear()

    # ------------------------------------------------------------------
    # pool lifecycle
    # ------------------------------------------------------------------

    def _create_pool(self) -> ProcessPoolExecutor:
        pool = ProcessPoolExecutor(
            max_workers=self.mesh_workers,
            mp_context=self._spawn_ctx,
            initializer=_mp_worker_init,
            initargs=(
                self.species,
                self.cache_dir,
                self.l2_cache_bytes,
                self.full_root_cache_bytes,
            ),
        )
        # force workers to start so we can track PIDs for cleanup
        pool.submit(bool, 0).result()
        self._worker_pids.update(pool._processes.keys())  # type: ignore[attr-defined]
        return pool

    def _recover_broken_pool(
        self,
        active_fetches: Dict[Future, int],
        rid_to_jobs: Dict[int, Set[int]],
    ) -> Set[int]:
        """Recreate pool after BrokenProcessPool. Marks all in-flight rids as failed.

        Returns the set of rids that crashed (caller marks them failed in its
        per-job bookkeeping).
        """
        self._pool_recreate_count += 1
        if self._pool_recreate_count > self.max_pool_recreates:
            raise RuntimeError(
                f"mesh prefetch pool died {self._pool_recreate_count} times, giving up"
            )
        cause = getattr(self._executor, "_broken", "unknown") if self._executor else "unknown"
        logger.error(
            f"MESH PREFETCH POOL CRASH #{self._pool_recreate_count}: {cause}\n"
            f"  in-flight fetches: {len(active_fetches)}\n"
            f"  hint: if SIGKILL (signal 9), check `dmesg -T | grep -i oom`"
        )
        try:
            if self._executor is not None:
                for pid in self._executor._processes:  # type: ignore[attr-defined]
                    try:
                        os.kill(pid, signal.SIGTERM)
                    except OSError:
                        pass
                self._executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        crashed_rids: Set[int] = set(active_fetches.values())
        active_fetches.clear()
        for rid in crashed_rids:
            rid_to_jobs.pop(rid, None)
        logger.info(f"  {len(crashed_rids)} roots marked failed after pool recovery")
        self._executor = self._create_pool()
        return crashed_rids

    # ------------------------------------------------------------------
    # streaming
    # ------------------------------------------------------------------

    def stream(self, jobs: Sequence[PrefetchJob[T]]) -> Iterator[PrefetchResult[T]]:
        if self._executor is None:
            raise RuntimeError("MeshPrefetcher must be used as a context manager")

        n_jobs = len(jobs)
        if n_jobs == 0:
            return

        active_fetches: Dict[Future, int] = {}
        rid_to_jobs: Dict[int, Set[int]] = {}
        job_pending_rids: List[Set[int]] = [set() for _ in range(n_jobs)]
        job_failed_rids: List[Set[int]] = [set() for _ in range(n_jobs)]
        job_fetch_seconds: List[float] = [0.0 for _ in range(n_jobs)]
        ready_jobs: List[int] = []
        yielded: Set[int] = set()
        rid_ready: Set[int] = set()
        rid_failed: Set[int] = set()

        prefetch_budget_bytes = self.prefetch_cache_mb * 1024 * 1024
        avg_nv_prior = 10 * 1024 * 1024 // _BYTES_PER_VERT
        avg_nv_sum = avg_nv_prior * 5
        avg_nv_count = 5

        def _prefetch_limit() -> int:
            avg_nv = avg_nv_sum / max(avg_nv_count, 1)
            avg_bytes = max(avg_nv * _BYTES_PER_VERT, 1)
            limit = int(prefetch_budget_bytes / avg_bytes)
            return max(limit, self.mesh_workers * 2)

        def _submit_job(ji: int) -> None:
            job = jobs[ji]
            pending: Set[int] = set()
            for rid in job.rids:
                if rid in rid_failed:
                    job_failed_rids[ji].add(rid)
                    continue
                if rid in rid_ready:
                    continue
                pending.add(rid)
                if rid not in rid_to_jobs:
                    rid_to_jobs[rid] = set()
                    assert self._executor is not None
                    fut = self._executor.submit(
                        _mp_fetch_mesh,
                        rid,
                        self.fetch_timeout_s,
                        self.max_mesh_vertices,
                    )
                    active_fetches[fut] = rid
                rid_to_jobs[rid].add(ji)
            job_pending_rids[ji] = pending
            if not pending:
                ready_jobs.append(ji)

        def _mark_rid_done(rid: int, nv: Optional[int], fetch_time: float) -> None:
            nonlocal avg_nv_sum, avg_nv_count
            if nv is not None:
                avg_nv_sum += nv
                avg_nv_count += 1
                rid_ready.add(rid)
            else:
                rid_failed.add(rid)
            for ji in rid_to_jobs.pop(rid, set()):
                job_pending_rids[ji].discard(rid)
                job_fetch_seconds[ji] += fetch_time
                if nv is None:
                    job_failed_rids[ji].add(rid)
                if not job_pending_rids[ji]:
                    ready_jobs.append(ji)

        next_to_submit = 0
        while next_to_submit < n_jobs and next_to_submit < _prefetch_limit():
            try:
                _submit_job(next_to_submit)
            except BrokenExecutor:
                crashed = self._recover_broken_pool(active_fetches, rid_to_jobs)
                for rid in crashed:
                    _mark_rid_done(rid, None, 0.0)
            next_to_submit += 1

        n_yielded = 0
        while n_yielded < n_jobs:
            while ready_jobs:
                ji = ready_jobs.pop(0)
                if ji in yielded:
                    continue
                yielded.add(ji)
                n_yielded += 1
                yield PrefetchResult(
                    job=jobs[ji],
                    failed_rids=sorted(job_failed_rids[ji]),
                    fetch_seconds=job_fetch_seconds[ji],
                )

                # refill submission window
                pf_limit = _prefetch_limit()
                while (
                    next_to_submit < n_jobs
                    and (next_to_submit - n_yielded) < pf_limit
                ):
                    try:
                        _submit_job(next_to_submit)
                    except BrokenExecutor:
                        crashed = self._recover_broken_pool(active_fetches, rid_to_jobs)
                        for rid in crashed:
                            _mark_rid_done(rid, None, 0.0)
                    next_to_submit += 1

            if n_yielded >= n_jobs:
                break

            if not active_fetches:
                # nothing in flight but jobs remaining → submit more aggressively
                if next_to_submit < n_jobs:
                    try:
                        _submit_job(next_to_submit)
                    except BrokenExecutor:
                        crashed = self._recover_broken_pool(active_fetches, rid_to_jobs)
                        for rid in crashed:
                            _mark_rid_done(rid, None, 0.0)
                    next_to_submit += 1
                    continue
                # everything submitted, nothing pending, but jobs unyielded —
                # shouldn't happen, but break to avoid spin
                logger.warning(
                    f"prefetch loop stalled: {n_yielded}/{n_jobs} yielded with no active fetches"
                )
                break

            try:
                done, _pending = wait(
                    active_fetches, timeout=5, return_when=FIRST_COMPLETED
                )
            except BrokenExecutor:
                crashed = self._recover_broken_pool(active_fetches, rid_to_jobs)
                for rid in crashed:
                    _mark_rid_done(rid, None, 0.0)
                continue

            for fut in done:
                rid = active_fetches.pop(fut)
                try:
                    rid_out, nv, fetch_time = fut.result(timeout=0)
                except BrokenExecutor:
                    crashed = self._recover_broken_pool(active_fetches, rid_to_jobs)
                    for crid in crashed:
                        _mark_rid_done(crid, None, 0.0)
                    _mark_rid_done(rid, None, 0.0)
                    break
                except Exception as e:
                    logger.warning(f"mesh fetch failed for root {rid}: {e}")
                    _mark_rid_done(rid, None, 0.0)
                    continue
                _mark_rid_done(rid, nv, fetch_time)
