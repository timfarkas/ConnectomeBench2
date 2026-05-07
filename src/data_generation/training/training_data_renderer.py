#!/usr/bin/env python3
"""Render training data from an operation bank into per-task parquet datasets.

Takes ConnectomeOperation + ControlSample JSONL files (produced by
build_operation_bank.py) and renders 3-view mesh images, then packages
them as QuestionDataset parquets.

Always reads from a YAML config (default: configs/renderer_config.yaml).
CLI flags override config values.

Usage:
    # run all species from config
    pixi run python src/training/training_data_renderer.py

    # single species, limited samples
    pixi run python src/training/training_data_renderer.py \
        --species mouse --max-samples 50

    # specific tasks, force regeneration
    pixi run python src/training/training_data_renderer.py \
        --species mouse --tasks endpoint_error_id,junction_error_corr --force

    # enable EM cutout slices
    pixi run python src/training/training_data_renderer.py \
        --species mouse --em-cutout

    # add EM to existing renders (no mesh re-fetch)
    pixi run python src/training/training_data_renderer.py \
        --species mouse --em-enrich

    # custom config path
    pixi run python src/training/training_data_renderer.py \
        --config configs/my_config.yaml
"""

import argparse
import atexit
import hashlib
import json
import logging
import multiprocessing as mp
import os
import signal
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, as_completed, wait
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from connectome.meshes import configure_mesh_cache, configure_full_root_cache
from connectome.mesh_prefetch import MeshPrefetcher, PrefetchJob
from connectome.utils import get_client_for_species
from connectome.operation_bank import (
    ConnectomeOperation,
    ControlSample,
    iter_operation_bank,
    iter_controls,
)
from training.question_dataset import (
    QuestionDataset,
    DatasetQuestion,
)

# import all data ops from the ops module
import training.training_data_ops as ops
from utils.profiler import Profiler
from training.training_data_ops import (
    ALL_TASKS,
    TASK_QUESTION_CONFIG,
    RenderJob,
    _render_worker_init,
    _render_worker_fn,
    _get_cv_seg as _get_cv_seg_for_species,
    # constants
    COLOR_A,
    COLOR_B,
    COLOR_SINGLE,
    DEFAULT_VIEW_EXTENT_NM,
    DEFAULT_DUST_METRIC,
    DEFAULT_DUST_THRESHOLD,
    DEFAULT_CHAIN_MAX_HOPS,
    DEFAULT_CHAIN_PROXIMITY_NM,
    DEFAULT_ADJACENT_CONTROLS,
    DEFAULT_ADJACENT_CUTOUT_NM,
    DEFAULT_MAX_ADJACENT_PER_OP,
    DEFAULT_DUST_OVERSAMPLE,
    DEFAULT_MESH_WORKERS,
    DEFAULT_L2_MESH_CACHE_MB,
    DEFAULT_ROOT_SIZE_CACHE_MB,
    DEFAULT_FULL_ROOT_MESH_CACHE_MB,
    DEFAULT_FULL_ROOT_MESH_TIMEOUT_S,
    DEFAULT_PREFETCH_CACHE_MB,
    # functions
    fetch_root_sizes,
    compute_major_roots,
    filter_chain_corrections,
    generate_adjacent_endpoint_samples,
    generate_adjacent_junction_samples,
    derive_jobs_from_operation,
    derive_jobs_from_control,
    render_job,
    reload_questions_from_renders,
    # subprocess functions (needed for ProcessPoolExecutor initargs)
    _mp_worker_init,
    _mp_fetch_mesh,
    # EM functions
    fetch_em_cutout,
    render_em_views,
    EMCutoutResult,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# suppress noisy urllib3 connection pool warnings from parallel mesh fetching
logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)


# ---------------------------------------------------------------------------
# cache key logic (mirrors generate_dataset.py)
# ---------------------------------------------------------------------------


def compute_cache_key(
    bank_path: Path,
    controls_path: Optional[Path],
    task: str,
    config: Dict[str, Any],
) -> str:
    """sha256 of bank mtime + controls mtime + task + config + source files."""
    h = hashlib.sha256()
    h.update(task.encode())
    h.update(json.dumps(config, sort_keys=True).encode())
    # include bank/controls file identity
    if bank_path.exists():
        stat = bank_path.stat()
        h.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
    if controls_path and controls_path.exists():
        stat = controls_path.stat()
        h.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
    # include BOTH source files to detect changes in either
    for src_file in [Path(__file__), Path(ops.__file__)]:
        h.update(src_file.read_bytes())
    return h.hexdigest()


def check_cache(task_dir: Path, expected_key: str) -> bool:
    cache_file = task_dir / ".cache_key"
    parquet = task_dir / "parquet" / "questions.parquet"
    if not cache_file.exists() or not parquet.exists():
        return False
    return cache_file.read_text().strip() == expected_key


def write_cache_key(task_dir: Path, key: str):
    (task_dir / ".cache_key").write_text(key)


# ---------------------------------------------------------------------------
# consistent sampling — priority is deterministic per (seed, key), independent
# of pool size.  top-10K is always a prefix of top-20K.
# ---------------------------------------------------------------------------


def _sample_priority(seed: int, key: int) -> int:
    """Deterministic priority for an item, independent of total sample count."""
    h = hashlib.md5(f"{seed}:{key}".encode()).digest()
    return int.from_bytes(h[:8], "little")


def consistent_sample(items: list, seed: int, n: int, key_fn) -> list:
    """Return top-n items by deterministic hash priority (stable across n)."""
    scored = sorted(items, key=lambda x: _sample_priority(seed, key_fn(x)))
    return scored[:n]


# ---------------------------------------------------------------------------
# checkpoint logic (per-task, tracks rendered job_ids)
# ---------------------------------------------------------------------------


def load_checkpoint(task_dir: Path) -> Dict[str, Any]:
    ckpt_path = task_dir / ".render_checkpoint.json"
    if ckpt_path.exists():
        try:
            return json.loads(ckpt_path.read_text())
        except Exception:
            pass
    return {"rendered_job_ids": [], "n_rendered": 0}


def save_checkpoint(task_dir: Path, ckpt: Dict[str, Any]):
    ckpt_path = task_dir / ".render_checkpoint.json"
    tmp = ckpt_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(ckpt))
    tmp.rename(ckpt_path)


# ---------------------------------------------------------------------------
# main pipeline
# ---------------------------------------------------------------------------


def run_pipeline(
    bank_path: Path,
    controls_path: Optional[Path],
    output_dir: Path,
    species: str,
    tasks: List[str],
    *,
    extra_controls_paths: Optional[List[Path]] = None,
    setup_workers: int = 4,
    view_extent_nm: Union[float, List[float]] = DEFAULT_VIEW_EXTENT_NM,
    both_endpoints: bool = True,
    dust_metric: str = DEFAULT_DUST_METRIC,
    dust_threshold: int = DEFAULT_DUST_THRESHOLD,
    dust_threshold_big: int = None,
    chain_max_hops: int = DEFAULT_CHAIN_MAX_HOPS,
    chain_proximity_nm: float = DEFAULT_CHAIN_PROXIMITY_NM,
    adjacent_controls: bool = DEFAULT_ADJACENT_CONTROLS,
    adjacent_junction_samples: bool = True,
    adjacent_cutout_nm: int = DEFAULT_ADJACENT_CUTOUT_NM,
    max_adjacent_per_op: int = DEFAULT_MAX_ADJACENT_PER_OP,
    max_adjacent_junction_per_op: int = 1,
    max_adjacent_junction_samples: Optional[int] = None,
    n_random_roots: int = 1,
    dust_oversample: int = DEFAULT_DUST_OVERSAMPLE,
    mesh_workers: int = DEFAULT_MESH_WORKERS,
    l2_mesh_cache_mb: int = DEFAULT_L2_MESH_CACHE_MB,
    root_size_cache_mb: int = DEFAULT_ROOT_SIZE_CACHE_MB,
    full_root_mesh_cache_mb: int = DEFAULT_FULL_ROOT_MESH_CACHE_MB,
    full_root_mesh_timeout_s: float = DEFAULT_FULL_ROOT_MESH_TIMEOUT_S,
    max_mesh_vertices: Optional[int] = None,
    max_root_l2_count: Optional[int] = None,
    prefetch_cache_mb: int = DEFAULT_PREFETCH_CACHE_MB,
    total_mesh_memory_gb: float = 5.0,
    max_samples: Optional[int] = None,
    max_ops: Optional[int] = None,
    stop_early: Optional[int] = None,
    seed: int = 42,
    force: bool = False,
    force_all: bool = False,
    dry_run: bool = False,
    finalize_only: bool = False,
    em_enrich: bool = False,
    em_cutout_config: Optional[Dict[str, Any]] = None,
    image_size: Tuple[int, int] = (512, 512),
    minimap_config: Optional[Dict[str, Any]] = None,
    center_jitter_nm: float = 0.0,
    color_randomize: bool = False,
    uni_color_meshes: bool = False,
    render_modes: Optional[List[str]] = None,
    uni_color_mode: Optional[str] = None,
    geometry_config: Optional[Dict[str, Any]] = None,
    render_workers: int = 2,
    split_mask_enrich: bool = False,
    geometry_single_enrich: bool = False,
):
    """Main rendering pipeline."""
    # tag all log lines with species for parallel runs
    global logger
    logger = logging.getLogger(f"renderer.{species}")
    logger.setLevel(logging.INFO)

    # --force-all implies --force
    if force_all:
        force = True
    # validate tasks
    for t in tasks:
        if t not in ALL_TASKS:
            raise ValueError(f"unknown task: {t}. valid: {ALL_TASKS}")
    if dust_metric not in ("l2", "sv"):
        raise ValueError(f"dust_metric must be 'l2' or 'sv', got: {dust_metric}")

    # configure caches before any fetches
    configure_mesh_cache(size_limit_bytes=l2_mesh_cache_mb * 1024 * 1024)
    configure_full_root_cache(size_limit_bytes=full_root_mesh_cache_mb * 1024 * 1024)

    if render_workers < 1:
        raise ValueError(f"render_workers must be >= 1, got: {render_workers}")
    if total_mesh_memory_gb <= 0:
        raise ValueError(f"total_mesh_memory_gb must be > 0, got: {total_mesh_memory_gb}")
    per_worker_mesh_budget_bytes = int(total_mesh_memory_gb * (1024 ** 3)) // render_workers
    logger.info(
        f"render workers: {render_workers} (per-worker mesh LRU budget: "
        f"{per_worker_mesh_budget_bytes / 1024**3:.2f}GB, total: "
        f"{total_mesh_memory_gb:.2f}GB)"
    )

    # parse view_extent_nm: scalar → fixed, list → [min, max] range
    if isinstance(view_extent_nm, (list, tuple)):
        extent_range = (float(view_extent_nm[0]), float(view_extent_nm[1]))
        base_extent_nm = (extent_range[0] + extent_range[1]) / 2.0
    else:
        extent_range = None
        base_extent_nm = float(view_extent_nm)

    # EM cutout configuration
    em_cfg = em_cutout_config or {}
    em_enabled_global = em_cfg.get("enabled", False)
    em_tasks = em_cfg.get("tasks", ["endpoint_error_id", "endpoint_error_corr"]) if em_enabled_global else []
    em_skip_on_failure = em_cfg.get("skip_on_failure", True)
    bp_cfg = em_cfg.get("best_plane", {})
    best_plane_mode = bp_cfg.get("mode", None)
    best_plane_kwargs = {
        "search_slab": bp_cfg.get("search_slab", 1),
        "render_slab": bp_cfg.get("render_slab", 9),
        "angle_step": bp_cfg.get("angle_step", 10),
        "inner_frac": bp_cfg.get("inner_frac", 0.5),
    }
    em_mip_override = em_cfg.get("em_mip", None)  # None = use species default
    seg_mip_override = em_cfg.get("seg_mip", None)  # None = use species default
    if em_enabled_global:
        logger.info(f"EM cutout: enabled for tasks {em_tasks}")
        Profiler.enable()
        logger.info("Profiler enabled (em_data + fetch_em_cutout instrumentation)")
        if em_mip_override is not None:
            logger.info(f"EM mip override: {em_mip_override}")
        if seg_mip_override is not None:
            logger.info(f"Seg mip override: {seg_mip_override}")
        if best_plane_mode:
            logger.info(f"EM best_plane: mode={best_plane_mode}")

    config = {
        "species": species,
        "tasks": tasks,
        "view_extent_nm": view_extent_nm,
        "both_endpoints": both_endpoints,
        "dust_metric": dust_metric,
        "dust_threshold": dust_threshold,
        "dust_threshold_big": dust_threshold_big,
        "chain_max_hops": chain_max_hops,
        "chain_proximity_nm": chain_proximity_nm,
        "adjacent_controls": adjacent_controls,
        "adjacent_junction_samples": adjacent_junction_samples,
        "adjacent_cutout_nm": adjacent_cutout_nm,
        "max_adjacent_per_op": max_adjacent_per_op,
        "max_adjacent_junction_per_op": max_adjacent_junction_per_op,
        "dust_oversample": dust_oversample,
        "mesh_workers": mesh_workers,
        "max_samples": max_samples,
        "seed": seed,
        "render_modes": render_modes or ["colored"],
        "uni_color_mode": uni_color_mode,
        "geometry_config": geometry_config,
        "color_randomize": color_randomize,
    }
    # include EM config in cache key only when enabled (avoids busting old caches)
    if em_enabled_global:
        config["em_cutout"] = em_cfg

    # --- EM-enrich mode: add EM images to existing renders, skip mesh work ---
    # Scans the EXISTING parquet (not the op bank) so it enriches all samples
    # regardless of seed/config changes. Early return — skips all op bank work.
    if em_enrich:
        if not em_enabled_global:
            raise ValueError(
                "--em-enrich requires EM cutout to be enabled in config "
                "(em_cutout.enabled: true)"
            )

        render_dir = output_dir / "_renders"
        if not render_dir.exists():
            logger.error("no _renders directory found — run normal pipeline first")
            return

        from connectome.em_data import DATA_PARAMETERS
        params = DATA_PARAMETERS[species]
        _em_mip = em_mip_override if em_mip_override is not None else params["em_mip"]
        _seg_mip = seg_mip_override if seg_mip_override is not None else params["seg_mip"]
        if em_mip_override is not None:
            params["em_mip"] = _em_mip
        if seg_mip_override is not None:
            params["seg_mip"] = _seg_mip
        secrets = None
        if species in ("human", "zebrafish"):
            token = os.getenv("CAVE_API_TOKEN")
            if token:
                secrets = {"token": token}
        import cloudvolume as _cv
        shared_cv_em = _cv.CloudVolume(
            params["em_path"], use_https=True, mip=_em_mip,
            cache=False, lru_bytes=500 * 1024 * 1024, secrets=secrets,
        )
        shared_cv_seg_em = _cv.CloudVolume(
            params["seg_path"], use_https=True, fill_missing=True,
            mip=_seg_mip, cache=False, lru_bytes=500 * 1024 * 1024,
            secrets=secrets,
        )
        client = get_client_for_species(species)
        _em_workers = em_cfg.get("workers", 4)
        em_pool = ThreadPoolExecutor(max_workers=_em_workers, thread_name_prefix="em_enrich")
        logger.info(f"EM enrich: thread pool ready ({_em_workers} workers)")

        em_window_nm = em_cfg.get("window_nm", 5000)
        em_views = em_cfg.get("views", ["xy", "xz", "yz"])
        em_skip_recenter = em_cfg.get("skip_recenter", False)
        is_junction_task = lambda t: "junction" in t

        # build operation_id → timestamp lookup from operation bank
        _op_ts_lookup: Dict[int, int] = {}
        if bank_path.exists():
            import json as _json
            with open(bank_path) as _f:
                for _line in _f:
                    _op = _json.loads(_line)
                    _oid = _op.get("operation_id")
                    _ts = _op.get("timestamp")
                    if _oid is not None and _ts is not None:
                        _op_ts_lookup[int(_oid)] = int(_ts)
            logger.info(f"EM enrich: loaded {len(_op_ts_lookup)} operation timestamps from bank")

        try:
            for task_name in tasks:
                task_dir = output_dir / task_name
                parquet_dir = task_dir / "parquet"
                parquet_path = parquet_dir / "questions.parquet"
                if not parquet_path.exists():
                    logger.warning(f"{task_name}: no parquet found, skipping")
                    continue

                ds = QuestionDataset.from_parquet(str(parquet_dir))
                logger.info(f"{task_name}: loaded {len(ds.questions)} questions from parquet")

                em_todo = []
                em_already = 0
                no_mesh = 0
                no_meta = 0
                for q in ds.questions:
                    meta = q.metadata or {}
                    job_id = q.sample_hash
                    job_dir = render_dir / task_name / job_id
                    if not job_dir.exists():
                        no_mesh += 1
                        continue
                    has_mesh = all(
                        list(job_dir.glob(f"*_{v}.png"))
                        for v in ("front", "side", "top")
                    )
                    if not has_mesh:
                        no_mesh += 1
                        continue

                    expected_em = list(em_views)
                    if best_plane_mode in ("replace", "add"):
                        expected_em.append("best")
                    has_em = all(
                        list(job_dir.glob(f"*_em_{v}.png"))
                        for v in expected_em
                    )
                    if has_em:
                        em_already += 1
                        continue

                    # before_root_ids for endpoint tasks, after_root_ids for junction
                    if is_junction_task(task_name):
                        raw_rids = meta.get("after_root_ids")
                    else:
                        raw_rids = meta.get("before_root_ids")
                    if raw_rids is None or len(raw_rids) == 0:
                        no_meta += 1
                        continue
                    root_ids = [int(r) for r in raw_rids]

                    center = meta.get("interface_point_nm")
                    if center is None:
                        center = meta.get("coordinate_nm")
                    if center is None:
                        no_meta += 1
                        continue
                    center_nm = np.array([float(x) for x in center])

                    # always derive timestamp from op bank (parquet timestamps unreliable)
                    timestamp = None
                    _oid = meta.get("operation_id") or meta.get("source_operation_id")
                    if _oid is not None:
                        _bank_ts = _op_ts_lookup.get(int(_oid))
                        if _bank_ts is not None:
                            _is_inversion = meta.get("strategy") == "inversion"
                            if is_junction_task(task_name):
                                # split ops: untested, keep existing heuristic
                                if _is_inversion:
                                    timestamp = _bank_ts - 1
                                else:
                                    timestamp = _bank_ts + 1
                            else:
                                # endpoint (merge) tasks
                                if _is_inversion:
                                    # inversion of split: results materialize at T+1
                                    timestamp = _bank_ts + 1
                                else:
                                    # merge ops + adjacent controls: both roots at T
                                    timestamp = _bank_ts
                    if timestamp is None:
                        # last resort: use parquet timestamp
                        timestamp = meta.get("timestamp")
                    colors = [COLOR_A, COLOR_B] if len(root_ids) >= 2 else [COLOR_SINGLE]

                    em_todo.append((job_id, root_ids, center_nm, timestamp, colors))

                if max_samples and len(em_todo) > max_samples:
                    em_todo = consistent_sample(
                        em_todo, seed, max_samples,
                        key_fn=lambda t: t[0],  # job_id string — must match main path
                    )

                logger.info(
                    f"{task_name}: {len(em_todo)} jobs need EM enrichment "
                    f"({em_already} already have EM, {no_mesh} missing mesh renders, "
                    f"{no_meta} missing metadata)"
                )

                if em_todo:
                    futures = {}
                    for job_id, root_ids, center_nm_, timestamp, colors in em_todo:
                        fut = em_pool.submit(
                            fetch_em_cutout,
                            species, center_nm_, root_ids,
                            window_nm=em_window_nm,
                            timestamp=timestamp,
                            client=client,
                            cv_em=shared_cv_em,
                            cv_seg=shared_cv_seg_em,
                            skip_recenter=em_skip_recenter,
                        )
                        futures[fut] = (job_id, root_ids, colors)

                    n_success = 0
                    n_fail = 0
                    pbar = tqdm(total=len(futures), desc=f"{task_name} EM", unit="job")
                    for fut in as_completed(futures):
                        job_id, root_ids, colors = futures[fut]
                        pbar.update(1)
                        try:
                            em_result = fut.result(timeout=120)
                        except Exception as e:
                            logger.warning(f"EM fetch failed for {job_id}: {e}")
                            n_fail += 1
                            continue
                        if em_result is None:
                            n_fail += 1
                            continue

                        job_dir = render_dir / task_name / job_id
                        color_map = dict(zip(root_ids, colors))
                        try:
                            render_em_views(
                                em_result, color_map, job_dir, job_id,
                                focal_root_ids=root_ids,
                                views=em_views,
                                best_plane_mode=best_plane_mode,
                                best_plane_kwargs=best_plane_kwargs,
                                output_size=image_size,
                                normalize=em_cfg.get("normalize", "percentile"),
                                slab_thickness=em_cfg.get("cardinal_slab", 5),
                            )
                            n_success += 1
                        except Exception as e:
                            logger.warning(f"EM render failed for {job_id}: {e}")
                            n_fail += 1
                    pbar.close()
                    logger.info(
                        f"{task_name}: EM enrichment done "
                        f"({n_success} success, {n_fail} failed)"
                    )

                # rebuild parquet: re-scan render dirs for updated image lists
                updated_questions = []
                for q in ds.questions:
                    job_id = q.sample_hash
                    job_dir = render_dir / task_name / job_id
                    if not job_dir.exists():
                        updated_questions.append(q)
                        continue

                    # gather all images: colored views + minimap + normal + EM
                    image_paths = []
                    image_types = []
                    for view in ("front", "side", "top"):
                        matches = sorted(job_dir.glob(f"*_{view}.png"))
                        matches = [m for m in matches if "_minimap_" not in m.name and "_normal_" not in m.name and "_em_" not in m.name]
                        if matches:
                            image_paths.append(str(matches[0]))
                            image_types.append(f"colored_{view}")
                    for view in ("front", "side", "top"):
                        matches = sorted(job_dir.glob(f"*_normal_{view}.png"))
                        matches = [m for m in matches if "_minimap_" not in m.name]
                        if matches:
                            image_paths.append(str(matches[0]))
                            image_types.append(f"normal_{view}")
                    for view in ("front", "side", "top"):
                        matches = sorted(job_dir.glob(f"*_minimap_*_{view}.png"))
                        matches = [m for m in matches if "_normal_" not in m.name]
                        if not matches:
                            matches = sorted(job_dir.glob(f"*_minimap_{view}.png"))
                            matches = [m for m in matches if "_normal_" not in m.name]
                        if matches:
                            image_paths.append(str(matches[0]))
                            image_types.append(f"minimap_{view}")
                    for em_view in ("xy", "xz", "yz", "best"):
                        matches = sorted(job_dir.glob(f"*_em_{em_view}.png"))
                        if matches:
                            image_paths.append(str(matches[0]))
                            image_types.append(f"em_{em_view}")

                    if len([t for t in image_types if t.startswith("colored_")]) == 3:
                        meta = dict(q.metadata) if q.metadata else {}
                        meta["image_types"] = image_types
                        updated_questions.append(DatasetQuestion(
                            question_type=q.question_type,
                            answer_space=q.answer_space,
                            answer=q.answer,
                            images=image_paths,
                            metadata=meta,
                            sample_hash=q.sample_hash,
                        ))
                    else:
                        updated_questions.append(q)

                if updated_questions:
                    ds_out = QuestionDataset(questions=updated_questions)
                    parquet_dir.mkdir(parents=True, exist_ok=True)
                    ds_out.to_parquet(str(parquet_dir / "questions.parquet"), move_images=False)
                    logger.info(
                        f"{task_name}: rebuilt parquet with {len(updated_questions)} questions"
                    )
        finally:
            em_pool.shutdown(wait=False)
        return

    # --- split-mask enrich: add GT split-mask PNGs to existing pair renders ---
    # Mirrors em_enrich: scans existing parquet, skips op-bank work, only
    # touches jobs with ≥2 root_ids that are missing split_mask sidecars.
    if split_mask_enrich:
        from connectome.meshes import get_full_root_mesh
        from foundation.scenes.mesh_scene import ViewConfig
        from foundation.scenes.merge_pair_scene import MergePairScene
        from foundation.scenes.fused_merge_pair_scene import (
            split_mask_from_pair_views,
        )
        from rendering.geometry_renderer import create_geometry_viewer
        from PIL import Image as _Image

        render_dir = output_dir / "_renders"
        if not render_dir.exists():
            logger.error("no _renders directory found — run normal pipeline first")
            return

        cv_seg = _get_cv_seg_for_species(species)
        try:
            geom_viewer = create_geometry_viewer()
        except Exception as e:
            logger.warning(f"failed to create geometry viewer: {e}")
            geom_viewer = None

        geo_cfg = (geometry_config or {})
        geo_resolution = int(geo_cfg.get("resolution", 224))
        geo_angles = tuple(geo_cfg.get("angles", ("front", "side", "top")))

        for task_name in tasks:
            task_dir = output_dir / task_name
            parquet_dir = task_dir / "parquet"
            parquet_path = parquet_dir / "questions.parquet"
            if not parquet_path.exists():
                logger.warning(f"{task_name}: no parquet found, skipping")
                continue

            ds = QuestionDataset.from_parquet(str(parquet_dir))
            logger.info(f"{task_name}: loaded {len(ds.questions)} questions from parquet")

            sm_todo: List[Tuple[str, List[int], np.ndarray, float]] = []
            sm_already = 0
            sm_single = 0
            sm_no_meta = 0
            for q in ds.questions:
                meta = q.metadata or {}
                job_id = q.sample_hash
                job_dir = render_dir / task_name / job_id
                if not job_dir.exists():
                    sm_no_meta += 1
                    continue

                is_junction = "junction" in task_name
                raw_rids = meta.get("after_root_ids") if is_junction else meta.get("before_root_ids")
                if raw_rids is None or len(raw_rids) < 2:
                    sm_single += 1
                    continue
                root_ids = [int(r) for r in raw_rids[:2]]

                center = meta.get("render_center_nm")
                if center is None:
                    center = meta.get("interface_point_nm")
                if center is None:
                    center = meta.get("coordinate_nm")
                if center is None:
                    sm_no_meta += 1
                    continue
                center_nm = np.array([float(x) for x in center])

                extent_nm_meta = meta.get("view_extent_nm")
                if extent_nm_meta is None:
                    sm_no_meta += 1
                    continue
                extent_nm_val = float(extent_nm_meta)

                expected = [
                    job_dir / f"{job_id}_split_mask_{int(extent_nm_val)}nm_{angle}.png"
                    for angle in geo_angles
                ]
                if all(p.exists() for p in expected):
                    sm_already += 1
                    continue

                sm_todo.append((job_id, root_ids, center_nm, extent_nm_val))

            if max_samples and len(sm_todo) > max_samples:
                sm_todo = consistent_sample(
                    sm_todo, seed, max_samples,
                    key_fn=lambda t: t[0],  # job_id string — must match main path
                )

            logger.info(
                f"{task_name}: {len(sm_todo)} jobs need split-mask "
                f"(already={sm_already}, single-segment={sm_single}, no-meta={sm_no_meta})"
            )

            sm_pf_jobs: List[PrefetchJob[Tuple[np.ndarray, float]]] = [
                PrefetchJob(
                    job_id=job_id,
                    rids=list(root_ids),
                    payload=(center_nm, extent_nm_val),
                )
                for job_id, root_ids, center_nm, extent_nm_val in sm_todo
            ]

            n_success = 0
            n_fail = 0
            pbar = tqdm(total=len(sm_pf_jobs), desc=f"{task_name} split-mask", unit="job")
            with MeshPrefetcher[Tuple[np.ndarray, float]](
                species,
                mesh_workers=mesh_workers,
                prefetch_cache_mb=prefetch_cache_mb,
                l2_cache_mb=l2_mesh_cache_mb,
                full_root_cache_mb=full_root_mesh_cache_mb,
                fetch_timeout_s=full_root_mesh_timeout_s,
                max_mesh_vertices=max_mesh_vertices,
            ) as pf:
                for result in pf.stream(sm_pf_jobs):
                    pbar.update(1)
                    job = result.job
                    job_id = job.job_id
                    root_ids = job.rids
                    center_nm, extent_nm_val = job.payload
                    if result.failed_rids:
                        logger.warning(
                            f"split-mask skip {job_id}: failed rids "
                            f"{result.failed_rids}"
                        )
                        n_fail += 1
                        continue
                    try:
                        meshes_pair = []
                        ok = True
                        for rid in root_ids:
                            m = get_full_root_mesh(cv_seg, rid, dataset_id=species)
                            if m is None:
                                logger.warning(
                                    f"split-mask: diskcache miss for rid={rid} "
                                    f"in {job_id}"
                                )
                                ok = False
                                break
                            meshes_pair.append(m)
                        if not ok:
                            n_fail += 1
                            continue

                        view_config = ViewConfig(
                            angles=geo_angles,
                            extents_nm=(extent_nm_val,),
                            resolution=geo_resolution,
                        )
                        pair_scene = MergePairScene.from_meshes(
                            mesh_a=meshes_pair[0],
                            mesh_b=meshes_pair[1],
                            anchor_nm=center_nm,
                            segment_id_a=root_ids[0],
                            segment_id_b=root_ids[1],
                            species=species,
                            view_config=view_config,
                            skip_contact=True,
                            viewer=geom_viewer,
                        )
                        split_mask_arr = split_mask_from_pair_views(pair_scene.views)

                        job_dir = render_dir / task_name / job_id
                        for v_idx, angle in enumerate(geo_angles):
                            sm_img = _Image.fromarray(split_mask_arr[v_idx], mode="L")
                            sm_path = (
                                job_dir
                                / f"{job_id}_split_mask_{int(extent_nm_val)}nm_{angle}.png"
                            )
                            sm_img.save(sm_path)
                        n_success += 1
                    except Exception as e:
                        logger.warning(f"split-mask render failed for {job_id}: {e}")
                        n_fail += 1
            pbar.close()
            logger.info(
                f"{task_name}: split-mask done ({n_success} success, {n_fail} failed)"
            )

            updated_questions = []
            for q in ds.questions:
                meta = q.metadata or {}
                job_id = q.sample_hash
                job_dir = render_dir / task_name / job_id
                if not job_dir.exists():
                    updated_questions.append(q)
                    continue

                existing_types = list(q.metadata.get("image_types", [])) if q.metadata else []
                existing_paths = list(q.images)
                added_any = False
                for angle in geo_angles:
                    itype = f"split_mask_{angle}"
                    if itype in existing_types:
                        continue
                    matches = sorted(job_dir.glob(f"*_split_mask_*_{angle}.png"))
                    if not matches:
                        continue
                    existing_paths.append(str(matches[0]))
                    existing_types.append(itype)
                    added_any = True

                if added_any:
                    new_meta = dict(meta)
                    new_meta["image_types"] = existing_types
                    updated_questions.append(DatasetQuestion(
                        question_type=q.question_type,
                        answer_space=q.answer_space,
                        answer=q.answer,
                        images=existing_paths,
                        metadata=new_meta,
                        sample_hash=q.sample_hash,
                    ))
                else:
                    updated_questions.append(q)

            if updated_questions:
                ds_out = QuestionDataset(questions=updated_questions)
                parquet_dir.mkdir(parents=True, exist_ok=True)
                ds_out.to_parquet(str(parquet_dir / "questions.parquet"), move_images=False)
                logger.info(
                    f"{task_name}: rebuilt parquet with {len(updated_questions)} questions"
                )
        return

    # --- geometry-single enrich: render single post-merge mesh ---
    # Endpoint_error_corr (unified mode): all rows have is_merge=True; positives
    # are real merge ops, negatives are inversion controls (after_root_ids
    # already flipped to the pre-split merged segment). Both have a resolvable
    # after_root_ids[0]. Adjacent_in_cutout controls have empty after_root_ids
    # (no segment ever existed) — silently skipped.
    # Fresh single-mesh render (not derivation from fused tensor) to avoid
    # A↔B seam leak in the silhouette.
    if geometry_single_enrich:
        from connectome.meshes import get_full_root_mesh
        from foundation.scenes.mesh_scene import MeshScene, ViewConfig
        from rendering.geometry_renderer import create_geometry_viewer

        render_dir = output_dir / "_renders"
        if not render_dir.exists():
            logger.error("no _renders directory found — run normal pipeline first")
            return

        cv_seg = _get_cv_seg_for_species(species)
        try:
            geom_viewer = create_geometry_viewer()
        except Exception as e:
            logger.warning(f"failed to create geometry viewer: {e}")
            geom_viewer = None

        geo_cfg = (geometry_config or {})
        geo_resolution = int(geo_cfg.get("resolution", 224))
        geo_angles = tuple(geo_cfg.get("angles", ("front", "side", "top")))

        for task_name in tasks:
            task_dir = output_dir / task_name
            parquet_dir = task_dir / "parquet"
            parquet_path = parquet_dir / "questions.parquet"
            if not parquet_path.exists():
                logger.warning(f"{task_name}: no parquet found, skipping")
                continue

            ds = QuestionDataset.from_parquet(str(parquet_dir))
            logger.info(f"{task_name}: loaded {len(ds.questions)} questions from parquet")

            gs_todo: List[Tuple[str, int, np.ndarray, float]] = []
            gs_already = 0
            gs_no_meta = 0
            gs_adj_skipped = 0
            for q in ds.questions:
                meta = q.metadata or {}
                job_id = q.sample_hash
                job_dir = render_dir / task_name / job_id
                if not job_dir.exists():
                    gs_no_meta += 1
                    continue

                out_path = job_dir / f"{job_id}_geometry_single.npy"
                if out_path.exists():
                    gs_already += 1
                    continue

                rids = meta.get("after_root_ids")
                if rids is None or len(rids) == 0:
                    # adjacent_in_cutout controls have empty after_root_ids
                    gs_adj_skipped += 1
                    continue
                single_rid = int(rids[0])

                center = meta.get("render_center_nm")
                if center is None:
                    center = meta.get("interface_point_nm")
                if center is None:
                    center = meta.get("coordinate_nm")
                if center is None:
                    gs_no_meta += 1
                    continue
                center_nm = np.array([float(x) for x in center])

                extent_nm_meta = meta.get("view_extent_nm")
                if extent_nm_meta is None:
                    gs_no_meta += 1
                    continue
                extent_nm_val = float(extent_nm_meta)

                gs_todo.append((job_id, single_rid, center_nm, extent_nm_val))

            if max_samples and len(gs_todo) > max_samples:
                gs_todo = consistent_sample(
                    gs_todo, seed, max_samples,
                    key_fn=lambda t: t[0],  # job_id string — must match main path
                )

            logger.info(
                f"{task_name}: {len(gs_todo)} jobs need geometry_single "
                f"(already={gs_already}, no-meta={gs_no_meta}, adj_skipped={gs_adj_skipped})"
            )

            gs_pf_jobs: List[PrefetchJob[Tuple[np.ndarray, float]]] = [
                PrefetchJob(
                    job_id=job_id,
                    rids=[single_rid],
                    payload=(center_nm, extent_nm_val),
                )
                for job_id, single_rid, center_nm, extent_nm_val in gs_todo
            ]

            n_success = 0
            n_fail = 0
            pbar = tqdm(total=len(gs_pf_jobs), desc=f"{task_name} geometry_single", unit="job")
            with MeshPrefetcher[Tuple[np.ndarray, float]](
                species,
                mesh_workers=mesh_workers,
                prefetch_cache_mb=prefetch_cache_mb,
                l2_cache_mb=l2_mesh_cache_mb,
                full_root_cache_mb=full_root_mesh_cache_mb,
                fetch_timeout_s=full_root_mesh_timeout_s,
                max_mesh_vertices=max_mesh_vertices,
            ) as pf:
                for result in pf.stream(gs_pf_jobs):
                    pbar.update(1)
                    job = result.job
                    job_id = job.job_id
                    single_rid = job.rids[0]
                    center_nm, extent_nm_val = job.payload
                    if result.failed_rids:
                        logger.warning(
                            f"geometry_single skip {job_id}: failed rid "
                            f"{result.failed_rids}"
                        )
                        n_fail += 1
                        continue
                    try:
                        mesh = get_full_root_mesh(cv_seg, single_rid, dataset_id=species)
                        if mesh is None:
                            logger.warning(
                                f"geometry_single: diskcache miss for rid={single_rid} "
                                f"in {job_id}"
                            )
                            n_fail += 1
                            continue

                        view_config = ViewConfig(
                            angles=geo_angles,
                            extents_nm=(extent_nm_val,),
                            resolution=geo_resolution,
                        )
                        scene = MeshScene.from_mesh(
                            mesh=mesh,
                            anchor_nm=center_nm,
                            segment_id=single_rid,
                            species=species,
                            view_config=view_config,
                            viewer=geom_viewer,
                        )

                        job_dir = render_dir / task_name / job_id
                        out_path = job_dir / f"{job_id}_geometry_single.npy"
                        np.save(str(out_path), scene.views)
                        n_success += 1
                    except Exception as e:
                        logger.warning(f"geometry_single render failed for {job_id}: {e}")
                        n_fail += 1
            pbar.close()
            logger.info(
                f"{task_name}: geometry_single done ({n_success} success, {n_fail} failed)"
            )

            updated_questions = []
            for q in ds.questions:
                meta = q.metadata or {}
                job_id = q.sample_hash
                job_dir = render_dir / task_name / job_id
                if not job_dir.exists():
                    updated_questions.append(q)
                    continue

                existing_types = list(q.metadata.get("image_types", [])) if q.metadata else []
                existing_paths = list(q.images)
                out_path = job_dir / f"{job_id}_geometry_single.npy"
                if "geometry_single" in existing_types or not out_path.exists():
                    updated_questions.append(q)
                    continue

                existing_paths.append(str(out_path))
                existing_types.append("geometry_single")
                new_meta = dict(meta)
                new_meta["image_types"] = existing_types
                updated_questions.append(DatasetQuestion(
                    question_type=q.question_type,
                    answer_space=q.answer_space,
                    answer=q.answer,
                    images=existing_paths,
                    metadata=new_meta,
                    sample_hash=q.sample_hash,
                ))

            if updated_questions:
                ds_out = QuestionDataset(questions=updated_questions)
                parquet_dir.mkdir(parents=True, exist_ok=True)
                ds_out.to_parquet(str(parquet_dir / "questions.parquet"), move_images=False)
                logger.info(
                    f"{task_name}: rebuilt parquet with {len(updated_questions)} questions"
                )
        return

    # --- cache invalidation ---
    adj_cache_path = output_dir / ".adjacent_candidates_cache.jsonl"
    if force and adj_cache_path.exists():
        adj_cache_path.unlink()
        logger.info("force: cleared adjacent candidates checkpoint")
    if force_all:
        import diskcache as dc
        for suffix in ("l2", "sv"):
            cache_dir = Path(os.getenv("CACHE_DIR", ".cache")) / f"root_sizes_{species}_{suffix}"
            if cache_dir.exists():
                dc.Cache(str(cache_dir)).clear()
                logger.info(f"force-all: cleared L2 size cache ({cache_dir.name})")

    # --- load operations + controls ---
    logger.info(f"loading operation bank from {bank_path}")
    all_ops = list(iter_operation_bank(bank_path))
    logger.info(f"  {len(all_ops)} operations loaded")

    controls = []
    if controls_path and controls_path.exists():
        logger.info(f"loading controls from {controls_path}")
        controls = list(iter_controls(controls_path))
        logger.info(f"  {len(controls)} controls loaded")

    if extra_controls_paths:
        for p in extra_controls_paths:
            p = Path(p)
            if not p.exists():
                logger.warning(f"extra controls path not found: {p}")
                continue
            logger.info(f"loading extra controls from {p}")
            extra = list(iter_controls(p))
            logger.info(f"  {len(extra)} extra controls loaded from {p}")
            controls.extend(extra)

    # --- restrict to first N ops ---
    if max_ops is not None:
        all_ops = all_ops[:max_ops]
        controls = controls[:max_ops]
        logger.info(f"--max-ops {max_ops}: using first {len(all_ops)} ops, {len(controls)} controls")

    # --- chain correction filtering (before dust filter) ---
    if chain_max_hops > 0:
        all_ops, controls = filter_chain_corrections(
            all_ops, controls,
            chain_max_hops=chain_max_hops,
            chain_proximity_nm=chain_proximity_nm,
        )

    # --- pre-sample ops + controls when max_samples is set ---
    # uses consistent sampling so top-10K is always a prefix of top-20K
    if max_samples:
        budget = max_samples * dust_oversample
        if len(all_ops) > budget:
            all_ops = consistent_sample(
                all_ops, seed, budget,
                key_fn=lambda op: op.operation_id,
            )
            logger.info(f"pre-sample: trimmed ops to {len(all_ops)} (budget={budget})")
        if len(controls) > budget:
            controls = consistent_sample(
                controls, seed, budget,
                key_fn=lambda c: c.source_operation_id,
            )
            logger.info(f"pre-sample: trimmed controls to {len(controls)} (budget={budget})")

    # --- dust filtering: fetch root sizes and compute major roots ---
    size_counts: Dict[int, int] = {}
    # maps (op_id_or_ctrl_src_id, source_type) → (major_before, major_after)
    major_roots_map: Dict[Tuple, Tuple[List[int], List[int]]] = {}

    if dust_threshold > 0:
        # collect all unique root_ids
        all_roots: set = set()
        for op in all_ops:
            all_roots.update(op.before_root_ids)
            all_roots.update(op.after_root_ids)
        for ctrl in controls:
            all_roots.update(ctrl.before_root_ids)
            all_roots.update(ctrl.after_root_ids)

        _thresh_str = f"{dust_threshold}" + (f"/{dust_threshold_big}" if dust_threshold_big is not None else "")
        logger.info(
            f"dust filter: fetching {dust_metric} counts for "
            f"{len(all_roots)} unique roots (threshold={_thresh_str})"
        )
        client_for_counts = get_client_for_species(species)
        size_counts = fetch_root_sizes(
            client_for_counts, all_roots, metric=dust_metric, workers=setup_workers,
            species=species, cache_mb=root_size_cache_mb,
        )

        n_ops_filtered = 0
        n_ctrl_filtered = 0

        for op in all_ops:
            result = compute_major_roots(
                op.before_root_ids, op.after_root_ids, size_counts, dust_threshold,
                dust_threshold_big=dust_threshold_big,
            )
            if result is None:
                n_ops_filtered += 1
            else:
                major_roots_map[(op.operation_id, "op")] = result

        for i, ctrl in enumerate(controls):
            result = compute_major_roots(
                ctrl.before_root_ids, ctrl.after_root_ids, size_counts, dust_threshold,
                dust_threshold_big=dust_threshold_big,
            )
            if result is None:
                n_ctrl_filtered += 1
            else:
                major_roots_map[("ctrl", i)] = result

        logger.info(
            f"dust filter: {n_ops_filtered}/{len(all_ops)} ops filtered, "
            f"{n_ctrl_filtered}/{len(controls)} controls filtered"
        )

    # --- adjacent-in-cutout controls ---
    if dust_threshold > 0:
        dust_surviving_ops = [
            op for op in all_ops
            if (op.operation_id, "op") in major_roots_map
        ]
    else:
        dust_surviving_ops = all_ops
    client_for_adj = get_client_for_species(species)

    adj_ctrls: List[ControlSample] = []
    if adjacent_controls and "endpoint_error_corr" in tasks:
        adj_cap = (max_samples * max_adjacent_per_op) if max_samples else None
        adj_ctrls = generate_adjacent_endpoint_samples(
            dust_surviving_ops, species, client_for_adj,
            cutout_nm=adjacent_cutout_nm,
            max_per_op=max_adjacent_per_op,
            max_controls=adj_cap,
            seed=seed,
            cache_path=adj_cache_path,
            dust_threshold=dust_threshold,
            dust_threshold_big=dust_threshold_big,
            dust_metric=dust_metric,
            root_size_cache_mb=root_size_cache_mb,
            n_random_roots=n_random_roots,
        )

    adj_junction_ctrls: List[ControlSample] = []
    junction_adj_tasks = [t for t in ("junction_error_corr", "junction_error_id") if t in tasks]
    if adjacent_junction_samples and junction_adj_tasks:
        adj_junction_cap = max_adjacent_junction_samples or (
            (max_samples * max_adjacent_junction_per_op) if max_samples else None
        )
        adj_junction_ctrls = generate_adjacent_junction_samples(
            dust_surviving_ops, species, client_for_adj,
            cutout_nm=adjacent_cutout_nm,
            max_per_op=max_adjacent_junction_per_op,
            max_controls=adj_junction_cap,
            seed=seed,
            cache_path=adj_cache_path,
            dust_threshold=dust_threshold,
            dust_threshold_big=dust_threshold_big,
            dust_metric=dust_metric,
            root_size_cache_mb=root_size_cache_mb,
            n_random_roots=n_random_roots,
        )

    # --- derive render jobs per task ---
    jobs_by_task: Dict[str, List[RenderJob]] = {t: [] for t in tasks}

    for op in all_ops:
        key = (op.operation_id, "op")
        if dust_threshold > 0 and key not in major_roots_map:
            continue  # dust-filtered
        mb, ma = major_roots_map.get(key, (None, None))
        for job in derive_jobs_from_operation(
            op, tasks, both_endpoints=both_endpoints, extent_nm=base_extent_nm,
            major_before=mb, major_after=ma,
            em_tasks=em_tasks, em_config=em_cfg,
        ):
            jobs_by_task[job.task].append(job)

    for i, ctrl in enumerate(controls):
        key = ("ctrl", i)
        if dust_threshold > 0 and key not in major_roots_map:
            continue  # dust-filtered
        mb, ma = major_roots_map.get(key, (None, None))
        for job in derive_jobs_from_control(
            ctrl, tasks, both_endpoints=both_endpoints, extent_nm=base_extent_nm,
            major_before=mb, major_after=ma,
            em_tasks=em_tasks, em_config=em_cfg,
        ):
            jobs_by_task[job.task].append(job)

    # adj endpoint controls (merge verification, answer=False)
    if "endpoint_error_corr" in tasks:
        for adj_ctrl in adj_ctrls:
            for job in derive_jobs_from_control(
                adj_ctrl, ["endpoint_error_corr"], both_endpoints=both_endpoints,
                extent_nm=base_extent_nm,
                em_tasks=em_tasks, em_config=em_cfg,
            ):
                jobs_by_task[job.task].append(job)

    # adj junction samples: TRUE positives for both junction tasks
    # junction_error_corr: "is this split correct?" → yes (answer=True)
    # junction_error_id:   "is there a junction/merge error here?" → yes (answer=True)
    if junction_adj_tasks:
        for adj_junc_ctrl in adj_junction_ctrls:
            for job in derive_jobs_from_control(
                adj_junc_ctrl, junction_adj_tasks, both_endpoints=both_endpoints,
                extent_nm=base_extent_nm,
                em_tasks=em_tasks, em_config=em_cfg,
            ):
                jobs_by_task[job.task].append(job)

    # apply max_samples limit per task (consistent sampling for expandability)
    for t in tasks:
        if max_samples and len(jobs_by_task[t]) > max_samples:
            jobs_by_task[t] = consistent_sample(
                jobs_by_task[t], seed, max_samples,
                key_fn=lambda j: j.job_id,
            )

    # --- render-time augmentation (mutates jobs in-place) ---
    aug_rng = np.random.default_rng(seed + 7)
    n_augmented = 0
    for t in tasks:
        for job in jobs_by_task[t]:
            if extent_range is not None:
                job.extent_nm = float(aug_rng.uniform(extent_range[0], extent_range[1]))
            if center_jitter_nm > 0:
                job.em_center_nm = job.center_nm.copy()  # preserve unjittered center for EM
                job.center_nm = job.center_nm + aug_rng.normal(0, center_jitter_nm, size=3)
            if color_randomize:
                if uni_color_meshes:
                    c = ops.random_high_contrast_pair(aug_rng)[0]
                    job.colors = [c] * len(job.colors)
                elif len(job.colors) >= 2:
                    job.colors = ops.random_high_contrast_pair(aug_rng)
            # Save actual render center (post-jitter) to metadata
            job.metadata["render_center_nm"] = job.center_nm.tolist()
            n_augmented += 1
    if uni_color_meshes:
        logger.warning("WARNING: uni_color_meshes=True: all meshes rendered in same color — may be undesirable depending on task")
    if extent_range or center_jitter_nm > 0 or color_randomize:
        logger.info(
            f"augmentation: {n_augmented} jobs "
            f"(extent={'range' if extent_range else 'fixed'}, "
            f"jitter={center_jitter_nm}nm, color_rand={color_randomize})"
        )

    # --- report ---
    logger.info("render job counts:")
    for t in tasks:
        logger.info(f"  {t}: {len(jobs_by_task[t])} jobs")

    total_jobs = sum(len(jobs_by_task[t]) for t in tasks)
    if total_jobs == 0:
        logger.info("no jobs to render")
        return

    if dry_run:
        logger.info("dry run — exiting")
        return

    # --- check caches ---
    tasks_to_run = []
    for t in tasks:
        task_dir = output_dir / t
        cache_key = compute_cache_key(bank_path, controls_path, t, config)
        if not force and check_cache(task_dir, cache_key):
            logger.info(f"  {t}: CACHED (skipping)")
        else:
            tasks_to_run.append(t)

    if not tasks_to_run:
        logger.info("all tasks cached, nothing to do")
        return

    # --- setup ---
    client = get_client_for_species(species)

    render_dir = output_dir / "_renders"
    render_dir.mkdir(parents=True, exist_ok=True)

    # --- EM thread pool setup (separate from mesh ProcessPoolExecutor) ---
    em_pool = None
    shared_cv_em = None
    shared_cv_seg_em = None
    em_skip_recenter = em_cfg.get("skip_recenter", None) if em_cfg else None
    if em_enabled_global:
        from connectome.em_data import DATA_PARAMETERS
        _em_workers = em_cfg.get("workers", 4)
        em_pool = ThreadPoolExecutor(max_workers=_em_workers, thread_name_prefix="em_fetch")
        params = DATA_PARAMETERS[species]
        _em_mip = em_mip_override if em_mip_override is not None else params["em_mip"]
        _seg_mip = seg_mip_override if seg_mip_override is not None else params["seg_mip"]
        if em_mip_override is not None:
            params["em_mip"] = _em_mip
        if seg_mip_override is not None:
            params["seg_mip"] = _seg_mip
        secrets = None
        if species in ("human", "zebrafish"):
            token = os.getenv("CAVE_API_TOKEN")
            if token:
                secrets = {"token": token}
        import cloudvolume as _cv
        shared_cv_em = _cv.CloudVolume(
            params["em_path"], use_https=True, mip=_em_mip,
            cache=False, lru_bytes=500 * 1024 * 1024, secrets=secrets,
        )
        shared_cv_seg_em = _cv.CloudVolume(
            params["seg_path"], use_https=True, fill_missing=True,
            mip=_seg_mip, cache=False, lru_bytes=500 * 1024 * 1024,
            secrets=secrets,
        )
        logger.info("EM fetch thread pool ready (4 workers)")

    # --- process each task ---
    for task_name in tasks_to_run:
        task_jobs = jobs_by_task[task_name]
        if not task_jobs:
            logger.info(f"{task_name}: no jobs, skipping")
            continue

        task_dir = output_dir / task_name
        task_dir.mkdir(parents=True, exist_ok=True)

        # load checkpoint for resume (--force clears it)
        if force:
            ckpt = {"rendered_job_ids": [], "n_rendered": 0, "failed_job_ids": []}
            # also wipe stale render artifacts for this task
            stale_render_dir = render_dir / task_name
            if stale_render_dir.exists():
                import shutil
                shutil.rmtree(stale_render_dir)
                stale_render_dir.mkdir(parents=True, exist_ok=True)
            save_checkpoint(task_dir, ckpt)
        else:
            ckpt = load_checkpoint(task_dir)
        rendered_ids = set(ckpt["rendered_job_ids"])
        failed_job_ids = set(ckpt.get("failed_job_ids", []))
        skip_ids = rendered_ids | failed_job_ids
        pending_jobs = [j for j in task_jobs if j.job_id not in skip_ids]

        if not pending_jobs:
            logger.info(f"{task_name}: all {len(task_jobs)} jobs done (checkpoint)")
        else:
            n_matched = len(task_jobs) - len(pending_jobs)
            logger.info(
                f"{task_name}: {len(pending_jobs)} pending "
                f"({len(rendered_ids)} rendered, {len(failed_job_ids)} permanently failed, "
                f"{n_matched}/{len(task_jobs)} skipped from checkpoint)"
            )

        # --finalize-only: reconstruct parquet from already-rendered jobs, skip rendering
        if finalize_only:
            logger.info(f"{task_name}: --finalize-only, reconstructing {len(rendered_ids)} rendered jobs")
            all_questions = reload_questions_from_renders(task_jobs, rendered_ids, render_dir)
            if not all_questions:
                logger.warning(f"{task_name}: no rendered jobs found, skipping parquet")
                continue
            ds = QuestionDataset(questions=all_questions)
            parquet_dir = task_dir / "parquet"
            parquet_dir.mkdir(parents=True, exist_ok=True)
            parquet_path = parquet_dir / "questions.parquet"
            ds.to_parquet(str(parquet_path), move_images=False)
            logger.info(f"{task_name}: wrote {len(all_questions)} questions to {parquet_path}")
            continue

        questions: List[DatasetQuestion] = []
        n_failed = 0
        checkpoint_interval = 25

        pbar = tqdm(total=len(pending_jobs), desc=task_name, unit="job")

        # producer-consumer: fetch meshes in background, render on main thread
        MESH_TIMEOUT = 30
        PREFETCH_AHEAD = len(pending_jobs)
        GC_INTERVAL = 20

        logger.info(f"  mesh_workers: {mesh_workers} subprocess workers")
        if max_root_l2_count is not None:
            logger.info(f"  max_root_l2_count: {max_root_l2_count:,} (pre-fetch blacklist)")
        if max_mesh_vertices is not None:
            logger.info(f"  max_mesh_vertices: {max_mesh_vertices:,} (post-fetch filter)")

        import gc
        cache_dir = os.environ.get("CACHE_DIR", ".cache")

        # spawn N persistent render workers with affinity routing. Each worker is
        # a 1-process ProcessPoolExecutor so `_render_pools[i].submit(...)` pins to
        # worker i. Routing by hash(op_id) % N ensures jobs from the same op land
        # on the same worker, so its local mesh LRU gets hot reuse.
        _render_spawn_ctx = mp.get_context("spawn")
        _render_pool_initargs = (
            species, cache_dir,
            l2_mesh_cache_mb * 1024**2, full_root_mesh_cache_mb * 1024**2,
            image_size, render_modes or ["colored"], minimap_config, uni_color_mode,
            base_extent_nm, per_worker_mesh_budget_bytes,
        )
        _render_pools: List[ProcessPoolExecutor] = [
            ProcessPoolExecutor(
                max_workers=1,
                mp_context=_render_spawn_ctx,
                initializer=_render_worker_init,
                initargs=_render_pool_initargs,
            )
            for _ in range(render_workers)
        ]
        for p in _render_pools:
            p.submit(bool, 0).result()  # warm up worker
        _render_futures: Dict[Any, int] = {}  # future → job_index
        logger.info(f"  render pool: {render_workers} affinity-routed workers")

        # profiling accumulators
        _prof_wait = 0.0
        _prof_render = 0.0
        _prof_other = 0.0
        _prof_fetch_times: list = []

        # track per-job mesh readiness (set of rids populated to diskcache)
        mesh_ready: set = set()
        mesh_failed: set = set()
        _FETCH_TIMEOUT_S = full_root_mesh_timeout_s

        # EM data cache (per-job, keyed by job_id)
        em_data_cache: Dict[str, Optional[EMCutoutResult]] = {}
        em_futures: Dict[str, Any] = {}  # job_id → Future

        # subprocess pool — each worker has its own GIL, CV, HTTP pool
        _spawn_ctx = mp.get_context("spawn")
        _pool_initargs = (species, cache_dir, l2_mesh_cache_mb * 1024**2, full_root_mesh_cache_mb * 1024**2)
        _pool_recreate_count = 0
        _MAX_POOL_RECREATES = 5

        # track all pool worker PIDs for cleanup on exit
        _all_worker_pids: set = set()

        def _kill_all_workers():
            """atexit: SIGKILL any surviving pool workers so they don't leak."""
            for pid in _all_worker_pids:
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass

        atexit.register(_kill_all_workers)

        def _create_pool():
            pool = ProcessPoolExecutor(
                max_workers=mesh_workers,
                mp_context=_spawn_ctx,
                initializer=_mp_worker_init,
                initargs=_pool_initargs,
            )
            # force workers to start so we can track their PIDs
            pool.submit(bool, 0).result()  # picklable no-op (bool is a builtin)
            _all_worker_pids.update(pool._processes.keys())
            return pool

        def _recover_broken_pool():
            """Recreate pool after BrokenProcessPool. Marks all in-flight fetches as failed."""
            nonlocal executor, _pool_recreate_count
            _pool_recreate_count += 1
            if _pool_recreate_count > _MAX_POOL_RECREATES:
                raise RuntimeError(f"subprocess pool died {_pool_recreate_count} times, giving up")
            cause = getattr(executor, '_broken', 'unknown')
            logger.error(
                f"SUBPROCESS POOL CRASH #{_pool_recreate_count}: {cause}\n"
                f"  in-flight fetches: {len(active_fetches)}\n"
                f"  hint: if cause is SIGKILL (signal 9), check `dmesg -T | grep -i oom` for OOM killer"
            )
            try:
                for pid in executor._processes:
                    try:
                        os.kill(pid, signal.SIGTERM)
                    except OSError:
                        pass
                executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
            crashed_rids = set(active_fetches.values())
            active_fetches.clear()
            for rid in crashed_rids:
                rid_to_jobs.pop(rid, None)
            logger.info(f"  {len(crashed_rids)} roots eligible for retry on fresh pool")
            executor = _create_pool()
            _resubmit_count = 0
            for ji2 in range(len(pending_jobs)):
                pends = job_pending_rids.get(ji2)
                if pends and (pends & crashed_rids):
                    try:
                        _submit_job_fetches(ji2)
                        _resubmit_count += 1
                    except BrokenProcessPool:
                        break
            if _resubmit_count:
                logger.info(f"  re-submitted {_resubmit_count} jobs after pool recovery")

        executor = _create_pool()

        def _route_worker_idx(job: RenderJob) -> int:
            """Affinity routing: hash(source_operation_id or operation_id) % n_workers.

            Clusters all jobs from the same op on the same render worker so its
            local mesh LRU hits across adjacency samples.
            """
            meta = job.metadata
            key = meta.get("source_operation_id") or meta.get("operation_id") or str(job.root_ids[0])
            digest = hashlib.blake2b(str(key).encode(), digest_size=8).digest()
            return int.from_bytes(digest, "big") % len(_render_pools)

        try:
            active_fetches: Dict = {}
            rid_to_jobs: Dict[int, set] = {}
            job_pending_rids: Dict[int, set] = {}
            ready_queue: list = []

            # prefetch backpressure: memory-bounded with adaptive mesh size estimate
            _BYTES_PER_VERT = 36
            _prefetch_budget_bytes = prefetch_cache_mb * 1024 * 1024
            _avg_nv_prior = 10 * 1024 * 1024 // _BYTES_PER_VERT
            _avg_nv_sum = _avg_nv_prior * 5
            _avg_nv_count = 5
            next_to_submit = 0

            _wants_geom_single = "geometry_single" in (render_modes or [])

            def _submit_job_fetches(job_idx):
                """Submit mesh fetches for a job, deduping against cache/active."""
                job = pending_jobs[job_idx]
                extra_rids: List[int] = []
                if _wants_geom_single:
                    after = job.metadata.get("after_root_ids") or []
                    if after:
                        extra_rids = [int(after[0])]
                all_rids = list(job.root_ids) + [r for r in extra_rids if r not in job.root_ids]
                # L2 count pre-filter
                if max_root_l2_count is not None and size_counts:
                    for rid in all_rids:
                        l2c = size_counts.get(rid, 0)
                        if l2c > max_root_l2_count:
                            logger.warning(
                                f"root {rid}: {l2c:,} L2 nodes exceeds "
                                f"max_root_l2_count={max_root_l2_count:,}, blacklisting"
                            )
                            mesh_failed.add(rid)
                pending = set()
                for rid in all_rids:
                    if rid in mesh_ready or rid in mesh_failed:
                        continue
                    if rid not in rid_to_jobs:
                        rid_to_jobs[rid] = set()
                        fut = executor.submit(
                            _mp_fetch_mesh, rid, _FETCH_TIMEOUT_S, max_mesh_vertices,
                        )
                        active_fetches[fut] = rid
                    rid_to_jobs[rid].add(job_idx)
                    pending.add(rid)
                job_pending_rids[job_idx] = pending

                # submit EM fetch if enabled for this job
                if job.em_enabled and em_pool is not None and job.job_id not in em_futures:
                    em_center = job.em_center_nm if job.em_center_nm is not None else job.center_nm
                    if em_skip_recenter is None:
                        _job_skip_recenter = (job.metadata.get("strategy") == "synapse_pair")
                    else:
                        _job_skip_recenter = em_skip_recenter
                    em_fut = em_pool.submit(
                        fetch_em_cutout,
                        species, em_center, job.root_ids,
                        window_nm=job.em_window_nm,
                        timestamp=job.em_timestamp,
                        client=client,
                        cv_em=shared_cv_em,
                        cv_seg=shared_cv_seg_em,
                        skip_recenter=_job_skip_recenter,
                    )
                    em_futures[job.job_id] = em_fut

                if not pending:
                    ready_queue.append(job_idx)

            def _prefetch_limit():
                avg_nv = _avg_nv_sum / _avg_nv_count
                avg_bytes = avg_nv * _BYTES_PER_VERT
                limit = int(_prefetch_budget_bytes / max(avg_bytes, 1))
                return max(limit, mesh_workers * 2)

            # initial batch
            while next_to_submit < len(pending_jobs) and next_to_submit < _prefetch_limit():
                try:
                    _submit_job_fetches(next_to_submit)
                except BrokenProcessPool:
                    _recover_broken_pool()
                next_to_submit += 1

            n_rendered = 0

            # --- helper: process a completed render result ---
            def _process_result(ji, image_paths, image_types, has_failed_root, render_time_s):
                nonlocal n_rendered, n_failed, _prof_render, _prof_other, next_to_submit
                _prof_render += render_time_s
                t_other = time.monotonic()

                job = pending_jobs[ji]

                # EM slice rendering: always release em_futures/em_data_cache refs so
                # mesh-render failures don't orphan EMCutoutResult bytes in main. only
                # await fut.result() when we actually need the data (image_paths valid).
                em_result = None
                if job.em_enabled:
                    if job.job_id in em_futures:
                        fut = em_futures.pop(job.job_id)
                        if image_paths is not None:
                            try:
                                em_result = fut.result(timeout=120)
                            except Exception as e:
                                logger.warning(f"EM fetch failed for job {job.job_id}: {e}")
                        else:
                            # mesh failed; no need to wait. cancel if not yet started.
                            # if already running, the fetch completes in bg then GCs
                            # once the pool releases its ref (we already released ours).
                            fut.cancel()
                    elif job.job_id in em_data_cache:
                        em_result = em_data_cache.pop(job.job_id)

                if image_paths is not None and job.em_enabled:
                    if em_result is not None:
                        output_em_dir = render_dir / job.task / job.job_id
                        color_map = {rid: c for rid, c in zip(job.root_ids, job.colors)}
                        try:
                            em_paths = render_em_views(
                                em_result, color_map, output_em_dir, job.job_id,
                                focal_root_ids=job.root_ids,
                                views=job.em_views,
                                best_plane_mode=best_plane_mode,
                                best_plane_kwargs=best_plane_kwargs,
                                output_size=image_size,
                                normalize=em_cfg.get("normalize", "percentile"),
                                slab_thickness=em_cfg.get("cardinal_slab", 5),
                            )
                            for view in job.em_views:
                                if view in em_paths:
                                    image_paths.append(str(em_paths[view]))
                                    image_types.append(f"em_{view}")
                            if "best" in em_paths:
                                image_paths.append(str(em_paths["best"]))
                                image_types.append("em_best")
                        except Exception as e:
                            logger.warning(f"EM render failed for job {job.job_id}: {e}")
                    elif em_skip_on_failure:
                        image_paths = None

                if image_paths is None:
                    n_failed += 1
                    if has_failed_root:
                        failed_job_ids.add(job.job_id)
                else:
                    meta = dict(job.metadata)
                    meta["image_types"] = image_types

                    q = DatasetQuestion(
                        question_type=job.question_type,
                        answer_space=job.answer_space,
                        answer=job.answer,
                        images=image_paths,
                        metadata=meta,
                        sample_hash=job.job_id,
                    )
                    questions.append(q)
                    rendered_ids.add(job.job_id)
                pbar.update(1)
                n_rendered += 1

                if (len(rendered_ids) + len(failed_job_ids)) % checkpoint_interval == 0:
                    ckpt["rendered_job_ids"] = list(rendered_ids)
                    ckpt["failed_job_ids"] = list(failed_job_ids)
                    ckpt["n_rendered"] = len(rendered_ids)
                    save_checkpoint(task_dir, ckpt)
                _prof_other += time.monotonic() - t_other

                # release this job's refcount on its rids (mesh_ready stays populated;
                # the render-worker LRUs handle RAM eviction, diskcache handles disk)
                for rid in job.root_ids:
                    if rid in rid_to_jobs:
                        rid_to_jobs[rid].discard(ji)
                        if not rid_to_jobs[rid]:
                            del rid_to_jobs[rid]

                # periodic gc to free GPU buffers from cleared scenes
                if n_rendered % GC_INTERVAL == 0:
                    gc.collect()

                # refill prefetch window
                _pf_limit = _prefetch_limit()
                while next_to_submit < len(pending_jobs) and (next_to_submit - n_rendered) < _pf_limit:
                    try:
                        _submit_job_fetches(next_to_submit)
                    except BrokenProcessPool:
                        _recover_broken_pool()
                    next_to_submit += 1

            # --- helper: collect completed render pool results (non-blocking) ---
            def _collect_render_results():
                if not _render_futures:
                    return
                done_futs = [f for f in _render_futures if f.done()]
                for fut in done_futs:
                    ji = _render_futures.pop(fut)
                    try:
                        job_id, img_paths, img_types, error, rtime = fut.result(timeout=0)
                        if error:
                            logger.warning(f"render worker error for job {job_id}: {error}")
                        has_failed = any(rid in mesh_failed for rid in pending_jobs[ji].root_ids)
                        _process_result(ji, img_paths, img_types or [], has_failed, rtime)
                    except Exception as e:
                        logger.warning(f"render worker exception for job index {ji}: {e}")
                        _process_result(ji, None, [], True, 0.0)

            while n_rendered < len(pending_jobs):
                # --- dispatch ready jobs (affinity-routed) ---
                while ready_queue:
                    ji = ready_queue.pop(0)
                    job = pending_jobs[ji]

                    worker_idx = _route_worker_idx(job)
                    fut = _render_pools[worker_idx].submit(
                        _render_worker_fn,
                        job, str(render_dir), image_size,
                        render_modes or ["colored"], minimap_config, uni_color_mode,
                        geometry_config,
                    )
                    _render_futures[fut] = ji

                # --- collect render pool results ---
                _collect_render_results()

                # --stop-early
                if stop_early is not None and n_rendered >= stop_early:
                    logger.info(f"--stop-early {stop_early}: processed {n_rendered} jobs ({len(questions)} successful), stopping")
                    break

                # wait for next mesh fetch or render result
                _has_pending = active_fetches or _render_futures
                if n_rendered < len(pending_jobs) and _has_pending:
                    t_wait = time.monotonic()

                    if active_fetches:
                        try:
                            done, _ = wait(active_fetches, timeout=5, return_when=FIRST_COMPLETED)
                        except BrokenProcessPool:
                            _recover_broken_pool()
                            done = set()
                    else:
                        time.sleep(0.1)
                        done = set()

                    _prof_wait += time.monotonic() - t_wait

                    for fut in done:
                        rid = active_fetches.pop(fut)
                        try:
                            rid_out, nv, fetch_time = fut.result(timeout=0)
                            _prof_fetch_times.append(fetch_time)
                            if nv is not None:
                                _avg_nv_sum += nv
                                _avg_nv_count += 1
                                mesh_ready.add(rid)
                            else:
                                mesh_failed.add(rid)
                            for ji2 in rid_to_jobs.get(rid, set()):
                                job_pending_rids[ji2].discard(rid)
                                if not job_pending_rids[ji2]:
                                    ready_queue.append(ji2)
                        except BrokenProcessPool:
                            _recover_broken_pool()
                            break
                        except Exception as e:
                            logger.warning(f"mesh fetch failed for root {rid}: {e}")
                            mesh_failed.add(rid)
                            for ji2 in rid_to_jobs.get(rid, set()):
                                job_pending_rids[ji2].discard(rid)
                                if not job_pending_rids[ji2]:
                                    ready_queue.append(ji2)

                    _collect_render_results()

                    pbar.set_postfix(
                        ready=len(ready_queue),
                        fetching=len(active_fetches),
                        ready_rids=len(mesh_ready),
                        rendering=len(_render_futures),
                    )

                elif n_rendered < len(pending_jobs) and not active_fetches and not _render_futures:
                    for ji2 in range(len(pending_jobs)):
                        if pending_jobs[ji2].job_id not in rendered_ids and ji2 not in ready_queue:
                            ready_queue.append(ji2)

            # drain remaining render pool results before cleanup
            if _render_futures:
                for fut in as_completed(list(_render_futures)):
                    ji = _render_futures.pop(fut)
                    try:
                        job_id, img_paths, img_types, error, rtime = fut.result(timeout=30)
                        if error:
                            logger.warning(f"render worker error for job {job_id}: {error}")
                        has_failed = any(rid in mesh_failed for rid in pending_jobs[ji].root_ids)
                        _process_result(ji, img_paths, img_types or [], has_failed, rtime)
                    except Exception as e:
                        logger.warning(f"render worker exception for job index {ji}: {e}")
                        _process_result(ji, None, [], True, 0.0)

            # cancel any in-flight fetches
            for fut in list(active_fetches):
                fut.cancel()
            active_fetches.clear()
        finally:
            try:
                for pid in executor._processes:
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except OSError:
                        pass
            except Exception:
                pass
            executor.shutdown(wait=False, cancel_futures=True)
            _all_worker_pids.clear()
            for p in _render_pools:
                p.shutdown(wait=True, cancel_futures=True)

        pbar.close()
        gc.collect()

        # profiling summary
        _prof_total = _prof_wait + _prof_render + _prof_other
        if _prof_total > 0:
            fetch_sum = sum(_prof_fetch_times)
            fetch_mean = fetch_sum / len(_prof_fetch_times) if _prof_fetch_times else 0
            fetch_max = max(_prof_fetch_times) if _prof_fetch_times else 0
            logger.info(
                f"  profile: {_prof_total:.1f}s total | "
                f"render {_prof_render:.1f}s ({100*_prof_render/_prof_total:.0f}%) | "
                f"wait {_prof_wait:.1f}s ({100*_prof_wait/_prof_total:.0f}%) | "
                f"other {_prof_other:.1f}s ({100*_prof_other/_prof_total:.0f}%)"
            )
            serial_fetch = fetch_sum / max(mesh_workers, 1)
            no_overlap = serial_fetch + _prof_render + _prof_other
            saved = no_overlap - _prof_total
            logger.info(
                f"  mesh fetches: {len(_prof_fetch_times)} total, "
                f"{fetch_mean:.2f}s avg, {fetch_max:.2f}s max, "
                f"{fetch_sum:.1f}s cumulative (across {mesh_workers} subprocesses)"
            )
            if saved > 0:
                logger.info(
                    f"  overlap saved ~{saved:.1f}s vs sequential "
                    f"({no_overlap:.0f}s → {_prof_total:.0f}s)"
                )

        # EM cutout profiler summary (fetch_em_cutout + em_data internals)
        if Profiler.is_enabled():
            Profiler.summary(group_by_step=False)

        # final checkpoint
        ckpt["rendered_job_ids"] = list(rendered_ids)
        ckpt["failed_job_ids"] = list(failed_job_ids)
        ckpt["n_rendered"] = len(rendered_ids)
        save_checkpoint(task_dir, ckpt)

        # also load questions from previously checkpointed jobs
        prev_questions = reload_questions_from_renders(
            task_jobs, rendered_ids - {j.job_id for j in pending_jobs}, render_dir
        )
        all_questions = prev_questions + questions

        if not all_questions:
            logger.warning(f"{task_name}: no successful renders, skipping parquet")
            continue

        logger.info(
            f"{task_name}: {len(all_questions)} questions "
            f"({n_failed} failed, {len(prev_questions)} from checkpoint)"
        )

        # build parquet
        ds = QuestionDataset(questions=all_questions)
        parquet_dir = task_dir / "parquet"
        parquet_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = parquet_dir / "questions.parquet"
        ds.to_parquet(str(parquet_path), move_images=False)

        # write cache key
        cache_key = compute_cache_key(bank_path, controls_path, task_name, config)
        write_cache_key(task_dir, cache_key)

        logger.info(f"{task_name}: wrote {len(all_questions)} questions to {parquet_path}")

    # clean up EM pool
    if em_pool is not None:
        em_pool.shutdown(wait=False)


# ---------------------------------------------------------------------------
# config loading (mirrors generate_dataset.py pattern)
# ---------------------------------------------------------------------------


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML config with defaults → species overrides."""
    try:
        import yaml
    except ImportError:
        logger.error("pyyaml required for --config. pip install pyyaml")
        sys.exit(1)

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    defaults = raw.get("defaults", {})
    out = raw.get("output_base", "datasets/")
    config = {
        "seed": raw.get("seed", 42),
        "input_base": raw.get("input_base", out),
        "output_base": out,
        "defaults": defaults,
        "species": {},
    }

    for species, sp_overrides in raw.get("species", {}).items():
        merged = dict(defaults)
        merged.update(sp_overrides or {})
        config["species"][species] = merged

    return config


def run_from_config(
    config: Dict[str, Any],
    *,
    filter_species: Optional[str] = None,
    max_samples: Optional[int] = None,
    max_ops: Optional[int] = None,
    stop_early: Optional[int] = None,
    force: bool = False,
    force_all: bool = False,
    dry_run: bool = False,
    finalize_only: bool = False,
    em_enrich: bool = False,
    em_cutout: bool = False,
    split_mask_enrich: bool = False,
    geometry_single_enrich: bool = False,
    cli_overrides: Optional[Dict[str, Any]] = None,
):
    """Run the renderer for all species in config."""
    input_base = Path(config["input_base"])
    output_base = Path(config["output_base"])
    seed = config["seed"]
    ovr = cli_overrides or {}
    defaults_cfg = config.get("defaults", {})

    def _get(key, default):
        """CLI override > per-species config > YAML defaults > hardcoded default."""
        v = ovr.get(key)
        if v is not None:
            return v
        return sp_cfg.get(key, defaults_cfg.get(key, default))

    for species, sp_cfg in config["species"].items():
        if filter_species and species != filter_species:
            continue

        split = ovr.get("split") or config.get("split", "train")
        input_dir = input_base / species / "splits" / split
        output_dir = output_base / species
        bank_path = input_dir / "operation_bank.jsonl"
        controls_path = input_dir / "operation_bank_controls.jsonl"

        if not bank_path.exists():
            logger.warning(f"{species}: no operation bank at {bank_path}, skipping")
            continue

        # CLI --tasks overrides per-species config
        tasks_ovr = ovr.get("tasks")
        if tasks_ovr is not None:
            tasks = tasks_ovr
        else:
            tasks = sp_cfg.get("tasks", ALL_TASKS)
        if isinstance(tasks, str):
            tasks = [t.strip() for t in tasks.split(",")]

        # resolve EM cutout config: species-level > defaults > disabled
        em_cfg_defaults = defaults_cfg.get("em_cutout", {})
        em_cfg_species = sp_cfg.get("em_cutout", {})
        em_cutout_config = {**em_cfg_defaults, **em_cfg_species}
        if em_cutout:
            em_cutout_config["enabled"] = True
        em_workers_ovr = ovr.get("em_workers")
        if em_workers_ovr is not None:
            em_cutout_config["workers"] = em_workers_ovr

        logger.info(f"\n{'='*60}")
        logger.info(f"species: {species} | input: {input_dir} | output: {output_dir}")
        logger.info(f"{'='*60}")

        _image_size_raw = _get("image_size", [512, 512])
        _image_size = tuple(_image_size_raw) if isinstance(_image_size_raw, list) else _image_size_raw

        extra_controls_raw = _get("extra_controls_banks", None)
        if extra_controls_raw is None:
            extra_controls_paths: Optional[List[Path]] = None
        else:
            if isinstance(extra_controls_raw, str):
                extra_controls_raw = [extra_controls_raw]
            extra_controls_paths = [Path(p) for p in extra_controls_raw]

        run_pipeline(
            bank_path=bank_path,
            controls_path=controls_path if controls_path.exists() else None,
            output_dir=output_dir,
            extra_controls_paths=extra_controls_paths,
            species=species,
            tasks=tasks,
            setup_workers=_get("setup_workers", 4),
            view_extent_nm=_get("view_extent_nm", DEFAULT_VIEW_EXTENT_NM),
            both_endpoints=_get("both_endpoints", True),
            dust_metric=_get("dust_metric", DEFAULT_DUST_METRIC),
            dust_threshold=_get("dust_threshold", DEFAULT_DUST_THRESHOLD),
            dust_threshold_big=_get("dust_threshold_big", None),
            chain_max_hops=_get("chain_max_hops", DEFAULT_CHAIN_MAX_HOPS),
            chain_proximity_nm=_get("chain_proximity_nm", DEFAULT_CHAIN_PROXIMITY_NM),
            adjacent_controls=_get("adjacent_controls", DEFAULT_ADJACENT_CONTROLS),
            adjacent_junction_samples=_get("adjacent_junction_samples", True),
            adjacent_cutout_nm=_get("adjacent_cutout_nm", DEFAULT_ADJACENT_CUTOUT_NM),
            max_adjacent_per_op=_get("max_adjacent_per_op", DEFAULT_MAX_ADJACENT_PER_OP),
            max_adjacent_junction_per_op=_get("max_adjacent_junction_per_op", 1),
            n_random_roots=_get("n_random_roots", 1),
            max_adjacent_junction_samples=_get("max_adjacent_junction_samples", None),
            dust_oversample=_get("dust_oversample", DEFAULT_DUST_OVERSAMPLE),
            mesh_workers=_get("mesh_workers", DEFAULT_MESH_WORKERS),
            l2_mesh_cache_mb=_get("l2_mesh_cache_mb", DEFAULT_L2_MESH_CACHE_MB),
            root_size_cache_mb=_get("root_size_cache_mb", DEFAULT_ROOT_SIZE_CACHE_MB),
            full_root_mesh_cache_mb=_get("full_root_mesh_cache_mb", DEFAULT_FULL_ROOT_MESH_CACHE_MB),
            full_root_mesh_timeout_s=_get("full_root_mesh_timeout_s", DEFAULT_FULL_ROOT_MESH_TIMEOUT_S),
            max_mesh_vertices=_get("max_mesh_vertices", None),
            max_root_l2_count=_get("max_root_l2_count", None),
            prefetch_cache_mb=_get("prefetch_cache_mb", DEFAULT_PREFETCH_CACHE_MB),
            total_mesh_memory_gb=_get("total_mesh_memory_gb", 5.0),
            max_samples=max_samples,
            max_ops=max_ops or _get("max_ops", None),
            stop_early=stop_early,
            seed=seed,
            force=force,
            force_all=force_all,
            dry_run=dry_run,
            finalize_only=finalize_only,
            em_enrich=em_enrich,
            em_cutout_config=em_cutout_config,
            split_mask_enrich=split_mask_enrich,
            geometry_single_enrich=geometry_single_enrich,
            image_size=_image_size,
            minimap_config=sp_cfg.get("minimap", defaults_cfg.get("minimap")),
            center_jitter_nm=float(_get("center_jitter_nm", 0)),
            color_randomize=bool(_get("color_randomize", False)),
            uni_color_meshes=bool(_get("uni_color_meshes", False)),
            render_modes=_get("render_modes", ["colored"]),
            uni_color_mode=_get("uni_color_mode", None),
            geometry_config=sp_cfg.get("geometry", defaults_cfg.get("geometry")),
            render_workers=int(_get("render_workers", 2)),
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    DEFAULT_CONFIG = "configs/renderer_config.yaml"

    parser = argparse.ArgumentParser(
        description="render training data from operation bank → per-task parquets"
    )

    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help=f"path to YAML config (default: {DEFAULT_CONFIG})")
    parser.add_argument("--input", default=None,
                        help="override input_base from config")
    parser.add_argument("--output", default=None,
                        help="override output_base from config")
    parser.add_argument("--species", default=None, choices=["mouse", "fly", "human", "zebrafish"],
                        help="run only this species (default: all in config)")
    parser.add_argument("--split", default=None, choices=["full", "train", "test"],
                        help="which lineage split to read from datasets/{species}/splits/{split}/ (default: train)")
    parser.add_argument(
        "--tasks", "--task",
        default=None,
        help=f"comma-separated tasks to generate (default: from config). choices: {ALL_TASKS}",
    )
    parser.add_argument("--setup-workers", type=int, default=None,
                        help="threads for L2 size counts + adjacent controls")
    parser.add_argument("--mesh-workers", type=int, default=None,
                        help=f"subprocess workers for mesh fetching (default: {DEFAULT_MESH_WORKERS})")
    parser.add_argument("--render-workers", type=int, default=None,
                        help="parallel render worker processes (default: 2)")
    parser.add_argument("--total-mesh-memory-gb", type=float, default=None,
                        help="total RAM budget (GB) for mesh LRU caches, divided equally among render workers (default: 5.0)")
    parser.add_argument("--view-extent-nm", type=float, default=None)
    parser.add_argument("--single-endpoint", action="store_true", default=False,
                        help="render only one before_root_id for endpoint_error_id")
    parser.add_argument("--dust-metric", default=None, choices=["l2", "sv"],
                        help=f"metric for dust filtering (default: {DEFAULT_DUST_METRIC})")
    parser.add_argument("--dust-threshold", type=int, default=None,
                        help=f"min size for smaller root in pair (default: {DEFAULT_DUST_THRESHOLD}, 0=disabled)")
    parser.add_argument("--dust-threshold-big", type=int, default=None,
                        help="min size for bigger root in pair (default: same as --dust-threshold)")
    parser.add_argument("--chain-max-hops", type=int, default=None,
                        help=f"max forward hops to detect chain corrections (default: {DEFAULT_CHAIN_MAX_HOPS}, 0=disabled)")
    parser.add_argument("--chain-proximity-nm", type=float, default=None,
                        help=f"max distance (nm) for chain correction detection (default: {DEFAULT_CHAIN_PROXIMITY_NM})")
    parser.add_argument("--no-adjacent-controls", action="store_true", default=False,
                        help="disable adjacent_in_cutout controls")
    parser.add_argument("--no-adjacent-junction-samples", action="store_true", default=False,
                        help="disable adjacent_in_cutout_junction True samples for junction_error_corr")
    parser.add_argument("--adjacent-cutout-nm", type=int, default=None,
                        help=f"cutout size (nm) for adjacent control sampling (default: {DEFAULT_ADJACENT_CUTOUT_NM})")
    parser.add_argument("--max-adjacent-per-op", type=int, default=None,
                        help=f"max adjacent controls per merge op (default: {DEFAULT_MAX_ADJACENT_PER_OP})")
    parser.add_argument("--dust-oversample", type=int, default=None,
                        help=f"pre-sample budget multiplier (default: {DEFAULT_DUST_OVERSAMPLE})")
    parser.add_argument("--full-root-mesh-cache-mb", type=int, default=None,
                        help=f"disk cache size (MB) for full root meshes (default: {DEFAULT_FULL_ROOT_MESH_CACHE_MB})")
    parser.add_argument("--full-root-mesh-timeout-s", type=float, default=None,
                        help=f"per-fetch timeout for full root meshes (default: {DEFAULT_FULL_ROOT_MESH_TIMEOUT_S}s)")
    parser.add_argument("--max-mesh-vertices", type=int, default=None,
                        help="skip meshes with more vertices than this")
    parser.add_argument("--max-root-l2-count", type=int, default=None,
                        help="blacklist roots with more L2 nodes than this")
    parser.add_argument("--prefetch-cache-mb", type=int, default=None,
                        help=f"mesh prefetch memory budget in MB (default: {DEFAULT_PREFETCH_CACHE_MB})")
    parser.add_argument("--max-samples", type=int, default=None, help="max samples per task")
    parser.add_argument("--max-ops", type=int, default=None,
                        help="use only the first N operations from the bank")
    parser.add_argument("--stop-early", type=int, default=None,
                        help="render at most N new samples then materialize parquet")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="ignore cache, regenerate")
    parser.add_argument("--force-all", action="store_true", help="--force + clear L2 size cache")
    parser.add_argument("--dry-run", action="store_true", help="show job counts without rendering")
    parser.add_argument("--finalize-only", action="store_true", help="write parquet from already-rendered jobs (checkpoint), skip rendering")
    parser.add_argument("--em-workers", type=int, default=None,
                        help="EM fetch thread pool workers (default: 4)")
    parser.add_argument("--em-cutout", action="store_true",
                        help="enable EM cutout slices alongside mesh renders (overrides config)")
    parser.add_argument("--em-enrich", action="store_true",
                        help="add EM cutout images to existing renders without re-fetching meshes")
    parser.add_argument("--split-mask-enrich", action="store_true",
                        help="add GT split-mask sidecar PNGs to existing pair renders (skips op-bank work)")
    parser.add_argument("--geometry-single-enrich", action="store_true",
                        help="DEPRECATED: post-hoc enrich pass for single-segment geometry sidecar. "
                             "Prefer adding 'geometry_single' to render_modes for new datasets; "
                             "use this flag only to backfill datasets already rendered without it.")

    args = parser.parse_args()

    # --em-enrich implies --em-cutout
    if args.em_enrich:
        args.em_cutout = True

    # load config
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"config not found: {config_path}")
        sys.exit(1)
    config = load_config(config_path)

    # apply CLI overrides to top-level config
    if args.seed is not None:
        config["seed"] = args.seed
    if args.input is not None:
        config["input_base"] = args.input
    if args.output is not None:
        config["output_base"] = args.output

    # build per-species overrides dict (None values = use config default)
    cli_overrides = {
        "setup_workers": args.setup_workers,
        "view_extent_nm": args.view_extent_nm,
        "dust_metric": args.dust_metric,
        "dust_threshold": args.dust_threshold,
        "dust_threshold_big": args.dust_threshold_big,
        "chain_max_hops": args.chain_max_hops,
        "chain_proximity_nm": args.chain_proximity_nm,
        "adjacent_cutout_nm": args.adjacent_cutout_nm,
        "max_adjacent_per_op": args.max_adjacent_per_op,
        "dust_oversample": args.dust_oversample,
        "mesh_workers": args.mesh_workers,
        "render_workers": args.render_workers,
        "total_mesh_memory_gb": args.total_mesh_memory_gb,
        "full_root_mesh_cache_mb": args.full_root_mesh_cache_mb,
        "full_root_mesh_timeout_s": args.full_root_mesh_timeout_s,
        "max_mesh_vertices": args.max_mesh_vertices,
        "max_root_l2_count": args.max_root_l2_count,
        "prefetch_cache_mb": args.prefetch_cache_mb,
        "em_workers": args.em_workers,
        "tasks": [t.strip() for t in args.tasks.split(",")] if args.tasks else None,
        "split": args.split,
    }
    if args.no_adjacent_controls:
        cli_overrides["adjacent_controls"] = False
    if args.no_adjacent_junction_samples:
        cli_overrides["adjacent_junction_samples"] = False
    if args.single_endpoint:
        cli_overrides["both_endpoints"] = False

    # print resolved run config
    active_overrides = {k: v for k, v in cli_overrides.items() if v is not None}
    logger.info(f"config: {config_path}")
    logger.info(f"  input_base: {config.get('input_base', 'datasets/')}")
    logger.info(f"  output_base: {config.get('output_base', 'datasets/')}")
    logger.info(f"  seed: {config.get('seed', 42)}")
    logger.info(f"  species: {args.species or ', '.join(config.get('species', {}).keys())}")
    if active_overrides:
        logger.info(f"  cli overrides: {active_overrides}")
    if args.em_cutout:
        logger.info(f"  em_cutout: enabled (cli)")
    if args.em_enrich:
        logger.info(f"  em_enrich: enabled")
    if args.split_mask_enrich:
        logger.info("  split_mask_enrich: enabled")
    if args.geometry_single_enrich:
        logger.info("  geometry_single_enrich: enabled")
    if args.max_samples:
        logger.info(f"  max_samples: {args.max_samples}")
    if args.max_ops:
        logger.info(f"  max_ops: {args.max_ops}")
    if args.force or args.force_all:
        logger.info(f"  force: {args.force_all and 'all' or 'yes'}")

    run_from_config(
        config,
        filter_species=args.species,
        max_samples=args.max_samples,
        max_ops=args.max_ops,
        stop_early=args.stop_early,
        force=args.force or args.force_all,
        force_all=args.force_all,
        dry_run=args.dry_run,
        finalize_only=args.finalize_only,
        em_enrich=args.em_enrich,
        em_cutout=args.em_cutout,
        split_mask_enrich=args.split_mask_enrich,
        geometry_single_enrich=args.geometry_single_enrich,
        cli_overrides=cli_overrides,
    )


if __name__ == "__main__":
    main()
    # force-exit to avoid blocking on zombie mesh-fetch threads
    os._exit(0)
