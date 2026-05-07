from collections import OrderedDict
from collections.abc import Set
import hashlib
import os
from pathlib import Path
import cloudvolume
import diskcache as dc
import pickle
import cloudvolume.mesh as cv_mesh
from utils.profiler import profiled

from typing import Optional

_L2_MESH_MEMO = OrderedDict()
_L2_MESH_MEMO_LIMIT = 1000

# defaults — overridable via configure_mesh_cache()
_L2_DISK_CACHE_SIZE_LIMIT = 2 * 1024 * 1024 * 1024  # 2 GB
_L2_DISK: Optional[dc.Cache] = None

# full root mesh cache — parallel to L2, for precomputed whole-neuron meshes.
# Disk-only (2-tier: disk → cloud). RAM tier removed — under ProcessPoolExecutor
# fetch workers, the per-worker memo never hits (main orchestrator dedupes rids).
_FULL_ROOT_DISK_CACHE_SIZE_LIMIT = 10 * 1024 * 1024 * 1024  # 10 GB
_FULL_ROOT_DISK: Optional[dc.Cache] = None


def configure_mesh_cache(
    size_limit_bytes: Optional[int] = None,
    cache_dir: Optional[str] = None,
    ram_limit: Optional[int] = None,
):
    """Configure mesh disk cache. Call before any mesh fetches.

    If never called, defaults are used on first access.
    """
    global _L2_DISK, _L2_DISK_CACHE_SIZE_LIMIT, _L2_MESH_MEMO_LIMIT
    if size_limit_bytes is not None:
        _L2_DISK_CACHE_SIZE_LIMIT = size_limit_bytes
    if ram_limit is not None:
        _L2_MESH_MEMO_LIMIT = ram_limit
    path = Path(cache_dir or os.environ.get("CACHE_DIR", ".cache")) / "l2_meshes"
    path.mkdir(parents=True, exist_ok=True)
    _L2_DISK = dc.Cache(path, eviction_policy="least-recently-used", size_limit=_L2_DISK_CACHE_SIZE_LIMIT)
    print(f"L2 mesh cache: {path} ({_L2_DISK_CACHE_SIZE_LIMIT / 1024**3:.1f} GB limit)")


def _get_disk_cache() -> dc.Cache:
    """Lazy init — configure_mesh_cache() overrides if called first."""
    global _L2_DISK
    if _L2_DISK is None:
        configure_mesh_cache()
    return _L2_DISK


def configure_full_root_cache(
    size_limit_bytes: Optional[int] = None,
    cache_dir: Optional[str] = None,
):
    """Configure full root mesh disk cache. Call before any mesh fetches."""
    global _FULL_ROOT_DISK, _FULL_ROOT_DISK_CACHE_SIZE_LIMIT
    if size_limit_bytes is not None:
        _FULL_ROOT_DISK_CACHE_SIZE_LIMIT = size_limit_bytes
    path = Path(cache_dir or os.environ.get("CACHE_DIR", ".cache")) / "full_root_meshes"
    path.mkdir(parents=True, exist_ok=True)
    _FULL_ROOT_DISK = dc.Cache(path, eviction_policy="least-recently-used", size_limit=_FULL_ROOT_DISK_CACHE_SIZE_LIMIT)
    print(f"Full root mesh cache: {path} ({_FULL_ROOT_DISK_CACHE_SIZE_LIMIT / 1024**3:.1f} GB limit)")


def _get_full_root_disk_cache() -> dc.Cache:
    """Lazy init — configure_full_root_cache() overrides if called first."""
    global _FULL_ROOT_DISK
    if _FULL_ROOT_DISK is None:
        configure_full_root_cache()
    return _FULL_ROOT_DISK


def get_full_root_mesh(cv_seg, root_id: int, *, dataset_id: str,
                       skip_on_miss: bool = False) -> Optional[cv_mesh.Mesh]:
    """Fetch a precomputed whole-neuron mesh via cv_seg.mesh.get().

    2-tier lookup: disk → cloud (write-through on miss).
    Cache key uses 'root|' prefix to avoid collision with L2 keys.

    If skip_on_miss=True, returns None on disk miss instead of falling through
    to cloud — caller is responsible for retry semantics.
    """
    # 1) disk hit
    disk_key = _mesh_key(dataset_id, f"root|{root_id}")
    b = _get_full_root_disk_cache().get(disk_key, default=None)
    if b is not None:
        try:
            return _bytes_to_mesh(b)
        except Exception:
            _get_full_root_disk_cache().delete(disk_key)

    if skip_on_miss:
        return None

    # 2) fetch from cloud (let exceptions propagate for caller retry logic)
    # cv_seg.mesh.get() returns a dict {id: Mesh} even for a single id
    result = cv_seg.mesh.get(root_id, allow_missing=True)
    if isinstance(result, dict):
        mesh = result.get(root_id)
    else:
        mesh = result
    if mesh is None or (hasattr(mesh, 'empty') and mesh.empty()):
        return None

    # write-through to disk
    _get_full_root_disk_cache().set(disk_key, _mesh_to_bytes(mesh))
    return mesh


def _mesh_key(dataset_id: str, seg_id) -> str:
    # include anything that changes mesh bytes: dataset, lod, maybe cv path/version
    # seg_id can be int (L2 node) or str (e.g. "root|12345" for full root meshes)
    s = f"{dataset_id}|{seg_id}"
    # hashing keeps keys short + friendly for sqlite index
    return hashlib.blake2b(s.encode(), digest_size=16).hexdigest()

def _put_ram(seg_id, mesh):
    _L2_MESH_MEMO[seg_id] = mesh
    _L2_MESH_MEMO.move_to_end(seg_id)
    if len(_L2_MESH_MEMO) > _L2_MESH_MEMO_LIMIT:
        _L2_MESH_MEMO.popitem(last=False)

def _mesh_to_bytes(mesh) -> bytes:
    # simplest: pickle. if mesh objects are big, consider lz4/zstd around it.
    return pickle.dumps(mesh, protocol=pickle.HIGHEST_PROTOCOL)

def _bytes_to_mesh(b: bytes):
    return pickle.loads(b)

def get_l2_meshes(cv_seg, ids, *, dataset_id: str):
    out = {}

    # 1) ram hits
    missing = []
    for seg_id in ids:
        m = _L2_MESH_MEMO.get(seg_id)
        if m is not None:
            _L2_MESH_MEMO.move_to_end(seg_id)
            out[seg_id] = m
        else:
            missing.append(seg_id)

    if not missing:
        return out

    # 2) disk hits (bulk)
    disk_hits = {}
    still_missing = []
    for seg_id in missing:
        k = _mesh_key(dataset_id, seg_id)
        b = _get_disk_cache().get(k, default=None)
        if b is not None:
            try:
                mesh = _bytes_to_mesh(b)
                disk_hits[seg_id] = mesh
            except Exception:
                # corrupted/old format -> treat as miss
                _get_disk_cache().delete(k)
                still_missing.append(seg_id)
        else:
            still_missing.append(seg_id)

    for seg_id, mesh in disk_hits.items():
        _put_ram(seg_id, mesh)
        out[seg_id] = mesh

    if not still_missing:
        return out

    # 3) fetch from source (bulk)
    fetched = cv_seg.mesh.get(still_missing, allow_missing=True) or {}
    if isinstance(fetched, dict):
        for seg_id, mesh in fetched.items():
            if mesh is None or mesh.empty():
                continue
            # write-through: disk then ram
            k = _mesh_key(dataset_id, seg_id)
            _get_disk_cache().set(k, _mesh_to_bytes(mesh))
            _put_ram(seg_id, mesh)
            out[seg_id] = mesh

    return out

@profiled
def fetch_and_concatenate_meshes(l2_node_ids: Set[int], root_id: int, component_name: str, batch_size: int, cv_seg: cloudvolume.CloudVolume, dataset_id: str) -> Optional[cv_mesh.Mesh]:
    """Fetch meshes for L2 nodes and concatenate them."""
    if not l2_node_ids:
        return None
    
    l2_list = list(l2_node_ids)
    meshes = []
    failed_nodes = []
    
    print(f"  Fetching meshes for {len(l2_list)} L2 nodes ({component_name})... (batch size: {batch_size})")
    
    # Fetch meshes in batches
    for i in range(0, len(l2_list), batch_size):
        batch = l2_list[i:i+batch_size]
        try:
            batch_meshes_dict = get_l2_meshes(cv_seg, batch, dataset_id=dataset_id)
            # cv_seg.mesh.get returns a dict mapping node_id -> Mesh
            # Handle both dict and list return types
            if isinstance(batch_meshes_dict, dict):
                for node_id in batch:
                    if node_id in batch_meshes_dict:
                        mesh = batch_meshes_dict[node_id]
                        if mesh is not None and not mesh.empty():
                            meshes.append(mesh)
                    else:
                        failed_nodes.append(node_id)
            else:
                # If it returns a list or single mesh, handle accordingly
                if isinstance(batch_meshes_dict, list):
                    for mesh in batch_meshes_dict:
                        if mesh is not None and not mesh.empty():
                            meshes.append(mesh)
                elif batch_meshes_dict is not None and not batch_meshes_dict.empty():
                    meshes.append(batch_meshes_dict)
        except Exception as e:
            print(f"    Warning: Error fetching batch {i//batch_size}: {e}")
            failed_nodes.extend(batch)
    
    if failed_nodes:
        print(f"    Warning: Failed to fetch {len(failed_nodes)} L2 node meshes")
    
    if not meshes:
        print(f"    No meshes found for {component_name}")
        return None
    
    # Concatenate meshes using CloudVolume's built-in method
    print(f"    Concatenating {len(meshes)} meshes...")
    concatenated = cv_mesh.Mesh.concatenate(*meshes, segid=root_id)
    
    print(f"    Concatenated mesh: {len(concatenated.vertices):,} vertices, {len(concatenated.faces):,} faces")
    return concatenated