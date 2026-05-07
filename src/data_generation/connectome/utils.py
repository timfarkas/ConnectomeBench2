from glob import glob
import os
from caveclient import CAVEclient
from datetime import datetime, timedelta
from typing import Optional, List, Tuple
from pathlib import Path
import numpy as np
import random
from tqdm import tqdm
import time
import pickle
from math import ceil
from dotenv import load_dotenv

from connectome.em_data import CutoutAccessor, EMDataFetcher, DATA_PARAMETERS

load_dotenv()


def get_most_similar_descendant(
    client: CAVEclient,
    species: str,
    root_id: int,
    center_nm: Tuple[float, float, float],
    view_extent_nm: float,
) -> int:
    """
    Given a (possibly outdated) root_id and a point of interest, return the
    latest descendant root that best covers the same region.

    1. Call get_latest_roots to find all current descendants.
    2. If exactly one, return it.
    3. If multiple, fetch seg cutouts at the original and current timestamps,
       and return the descendant whose voxel mask overlaps most with the
       original root's mask.

    Args:
        client: CAVEclient instance.
        species: Species string (for EMDataFetcher).
        root_id: The (possibly stale) root ID.
        center_nm: (x, y, z) center of the region of interest in nm.
        view_extent_nm: Full view extent; cutout uses half this as window.

    Returns:
        The best-matching latest root ID.
    """
    descendants = client.chunkedgraph.get_latest_roots(root_id)
    descendants = np.unique(descendants)

    if len(descendants) == 1:
        return int(descendants[0])

    if len(descendants) == 0:
        return int(root_id)

    # multiple descendants — disambiguate via seg overlap
    window_nm = view_extent_nm / 2
    og_timestamp = get_root_timestamp(client, root_id)

    # fetch roots cutout at original timestamp
    og_fetcher = EMDataFetcher(
        species=species,
        roots_filter=[root_id],
        timestamp=og_timestamp,
        client=client,
    )
    _, og_seg = og_fetcher.fetch_cutout(center_nm, window_nm, with_roots=True)
    og_roots = og_seg.roots_cutout
    og_mask = (og_roots == root_id)

    if og_mask.sum() == 0:
        # original root not visible in cutout — fall back to first descendant
        return int(descendants[0])

    # fetch roots cutout at current timestamp
    now_fetcher = EMDataFetcher(
        species=species,
        timestamp=datetime.now(),
        client=client,
    )
    _, now_seg = now_fetcher.fetch_cutout(center_nm, window_nm, with_roots=True)
    now_roots = now_seg.roots_cutout

    # compute overlap for each descendant
    best_id = int(descendants[0])
    best_overlap = 0
    for desc in descendants:
        overlap = np.sum(og_mask & (now_roots == desc))
        if overlap > best_overlap:
            best_overlap = overlap
            best_id = int(desc)

    return best_id


def get_latest_local_root(species: str,
position_nm : Tuple[float, float, float], 
tolerance_nm: float = 500,
client: Optional[CAVEclient] = None,
root_id: Optional[int] = None,
cutout_accessor: Optional[CutoutAccessor] = None) -> Optional[int]:
    """
    Return the latest root ID for the supervoxel nearest to ``position``.
    """
    if client is None:
        client = get_client_for_species(species)
    if cutout_accessor is None:
        filter = [root_id] if root_id is not None else None
        fetcher = EMDataFetcher(species=species, 
                                roots_filter=filter,
                                client=client)
        _, cutout_accessor = fetcher.fetch_cutout(position_nm, tolerance_nm*2, with_roots=False)
    position_vox = cutout_accessor.global_nm_to_cutout_voxels(position_nm)
    sv_id = cutout_accessor.find_sv_at_coords(position_vox, tolerance_nm)
    return int(client.chunkedgraph.get_root_id(sv_id, timestamp=datetime.now()))


def get_root_timestamp(
    client: CAVEclient,
    root_id: int,
    fallback_timestamp: Optional[datetime] = None,
) -> Optional[datetime]:
    """
    Find the timestamp when a root_id was valid.
    
    Args:
        client: CAVEclient instance
        root_id: Root ID to find timestamp for
        fallback_timestamp: Optional timestamp to use if root is still valid
        
    Returns:
        datetime object representing when the root was valid.
    """
    return client.chunkedgraph.get_root_timestamps([root_id], latest=False)[0]

def get_client_for_species(species: str) -> CAVEclient:
    if species == "fly":
        client = CAVEclient("flywire_fafb_public")
        daf_token = os.getenv("CAVE_API_TOKEN_DAF")
        if daf_token:
            try:
                client.auth.save_token(token=daf_token, overwrite=True)
            except Exception:
                pass
            client = CAVEclient("flywire_fafb_public")
        return client
    elif species == "mouse":
        client = CAVEclient("minnie65_public")
        daf_token = os.getenv("CAVE_API_TOKEN_DAF")
        if daf_token:
            try:
                client.auth.save_token(token=daf_token, overwrite=True)
            except Exception:
                pass
            client = CAVEclient("minnie65_public")
        return client
    elif species in ["human", "zebrafish"]:
        token = os.getenv("CAVE_API_TOKEN")
        if not token:
            raise ValueError(f"{species} dataset requires CAVE_API_TOKEN env var")
        server = "https://global.brain-wire-test.org/"
        datastack = DATA_PARAMETERS[species]["datastack_name"]
        return CAVEclient(datastack, server_address=server, auth_token=token)
    else:
        raise ValueError(f"Invalid species: {species}")


def get_time_cutoff() -> datetime:
    """Resolve as-of timestamp for proofread-root queries.

    Reads TIME_CUTOFF env var (ISO 8601, e.g. '2026-03-30' or
    '2026-03-30T00:00:00+00:00'). Falls back to datetime.now() if unset.
    Blow up loudly on malformed values — silent drift to "now" would mask
    contamination during lineage-split rebuilds.
    """
    raw = os.getenv("TIME_CUTOFF")
    if not raw:
        return datetime.now()
    return datetime.fromisoformat(raw)


def get_latest_proofread_roots(client : CAVEclient, species : str, delta_days : int = 720, batch_size : int = 5000, seed : int = 42, cutoff : Optional[datetime] = None):
    """
    Get root IDs for sampling. For mouse/fly uses proofreading tables.
    For human/zebrafish uses delta_roots (roots changed in last delta_days).

    Uses batched get_latest_roots calls to avoid hanging on large datasets.
    Results are cached per species for fast reloading on relaunch.

    `cutoff` (or env TIME_CUTOFF) freezes the as-of time for live_query,
    get_latest_roots, and get_delta_roots, so queries match a lineage-split
    snapshot rather than drifting with CAVE state.
    """
    if cutoff is None:
        cutoff = get_time_cutoff()

    cache_dir = Path(os.getenv("CACHE_DIR", ".cache")) / "proofread_roots"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cutoff_tag = cutoff.strftime("%Y%m%dT%H%M%S")
    cache_file = cache_dir / f"{species}_latest_roots_seed{seed}_at{cutoff_tag}.npy"

    # Return cached result if available
    if cache_file.exists():
        print(f"  Loading cached roots from {cache_file}...", flush=True)
        return np.load(cache_file)

    def batched_get_latest_roots(roots, batch_size=5000):
        """Get latest roots in batches to avoid API timeout."""
        all_latest = []
        for i in range(0, len(roots), batch_size):
            batch = roots[i:i+batch_size]
            latest = client.chunkedgraph.get_latest_roots(batch, timestamp=cutoff)
            all_latest.extend(latest)
            if i % 5000 == 0:
                print(f"  get_latest_roots: {i}/{len(roots)} processed...", flush=True)
        return np.asarray(all_latest)

    print(f"  [{species}] using cutoff={cutoff.isoformat()}", flush=True)

    if species == "mouse":
        print("  Querying proofreading_status_and_strategy table...", flush=True)
        roots = np.asarray(client.materialize.live_query("proofreading_status_and_strategy", cutoff)["pt_root_id"])
        latest = batched_get_latest_roots(roots, batch_size)
        result = np.unique(latest)
    elif species == "fly":
        print("  Querying proofread_neurons table...", flush=True)
        roots = np.asarray(client.materialize.live_query("proofread_neurons", cutoff)["pt_root_id"])
        print(f"  Found {len(roots)} proofread neurons, getting latest roots...", flush=True)
        # Subsample to 20k for fly (API is slow)
        MAX_ROOTS = 20000
        if len(roots) > MAX_ROOTS:
            print(f"  Subsampling to {MAX_ROOTS} roots (seed={seed})...", flush=True)
            np.random.seed(seed)
            np.random.shuffle(roots)
            roots = roots[:MAX_ROOTS]
        # Use smaller batch size for fly (500)
        latest = batched_get_latest_roots(roots, batch_size=500)
        result = np.unique(latest)
    elif species in ["human", "zebrafish"]:
        # No proofreading table - use delta_roots to get recently changed roots
        print("  Querying delta_roots table...", flush=True)
        end_time = cutoff
        start_time = end_time - timedelta(days=delta_days)
        old_roots, new_roots = client.chunkedgraph.get_delta_roots(start_time, end_time)
        print(f"  Found {len(new_roots)} delta roots, getting latest roots...", flush=True)
        latest = batched_get_latest_roots(new_roots, batch_size)
        result = np.unique(latest)
    else:
        raise ValueError(f"Species {species} not supported")

    # Cache the result
    print(f"  Caching {len(result)} roots to {cache_file}...", flush=True)
    np.save(cache_file, result)
    return result

def get_n_ancestors(client : CAVEclient,  root : int, n : int):
    """
    Get the first n (or maximum amount of) ancestors of a root, cached if available.
    Uses a monolithic cache file to avoid file count quota issues.
    Backward compatible with old per-root directory cache.

    Args:
        client: CAVEclient instance
        root: Root ID to get ancestors for
        n: Number of ancestors to get

    Returns:
        List of ancestor IDs
    """
    import fcntl

    cache_base = Path(os.getenv("CACHE_DIR", ".cache")) / "root_ancestors"
    cache_file = cache_base / "ancestors_cache.pkl"
    lock_file = cache_base / "ancestors_cache.lock"

    # Ensure cache dir exists
    cache_base.mkdir(parents=True, exist_ok=True)

    root_key = str(root)

    # 1. Check old per-root directory cache (backward compatibility)
    old_cache_dir = cache_base / root_key
    old_cache_files = glob(f"{old_cache_dir}/*.pkl") if old_cache_dir.exists() else []

    if old_cache_files:
        # first: look for a *_max.pkl file
        max_files = [f for f in old_cache_files if f.endswith("_max.pkl")]
        if max_files:
            max_file = max_files[0]
            ancestors = pickle.load(open(max_file, "rb"))
            if len(ancestors) >= n:
                return ancestors[:n]
            return ancestors

        # otherwise: try to find a file with at least n ancestors
        try:
            counts_and_files = []
            for f in old_cache_files:
                base = os.path.basename(f)
                stem = base[:-4]
                count = int(stem)
                counts_and_files.append((count, f))

            counts_and_files.sort(key=lambda x: x[0])

            for count, f in counts_and_files:
                if count >= n:
                    ancestors = pickle.load(open(f, "rb"))
                    return ancestors[:n]
        except Exception as e:
            pass  # Fall through to new cache or API

    # 2. Check new monolithic cache
    cache = {}
    if cache_file.exists():
        try:
            with open(lock_file, 'w') as lf:
                fcntl.flock(lf, fcntl.LOCK_SH)
                try:
                    with open(cache_file, 'rb') as f:
                        cache = pickle.load(f)
                finally:
                    fcntl.flock(lf, fcntl.LOCK_UN)
        except Exception as e:
            cache = {}

    if root_key in cache:
        entry = cache[root_key]
        ancestors = entry["ancestors"]
        max_reached = entry.get("max_reached", False)

        if len(ancestors) >= n or max_reached:
            return ancestors[:n]

    # 3. Fetch ancestors from API
    intervals = 270
    root_timestamp = client.chunkedgraph.get_root_timestamps([root], latest=False)[0]

    time_ranges = [timedelta(days=intervals*i) for i in range(10)]

    max_range_reached = False
    sources = []
    for time_range in time_ranges:
        graph_edges = client.chunkedgraph.get_lineage_graph(root, timestamp_past=root_timestamp-time_range)["links"]
        sources = [edge["source"] for edge in graph_edges]
        if len(sources) > n:
            break
        if time_range == time_ranges[-1]:
            max_range_reached = True

    # 4. Save to monolithic cache with file locking
    try:
        with open(lock_file, 'w') as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
            try:
                # Re-read cache in case it was updated
                if cache_file.exists():
                    try:
                        with open(cache_file, 'rb') as f:
                            cache = pickle.load(f)
                    except:
                        cache = {}

                cache[root_key] = {
                    "ancestors": sources,
                    "max_reached": max_range_reached,
                }

                # Write atomically
                tmp_file = cache_file.with_suffix('.tmp')
                with open(tmp_file, 'wb') as f:
                    pickle.dump(cache, f)
                tmp_file.rename(cache_file)
            finally:
                fcntl.flock(lf, fcntl.LOCK_UN)
    except Exception as e:
        print(f"Warning: Could not save to cache: {e}")

    return sources


def get_n_ancestors_for_roots(client : CAVEclient,  roots : List[int], target_count : int, oversample_factor : float = 3.0, seed : Optional[int] = None, ancestors_per_root : Optional[int] = None):
    ancestors = []
    n_per_root = ancestors_per_root if ancestors_per_root is not None else max(1, target_count // len(roots))
    print(f"Fetching {n_per_root} ancestors per root.")

    print(f"Fetching {len(roots)} lineages, this may take a while...")
    
    for i in tqdm(range(len(roots)), desc="Fetching ancestors"):
        root_ancestors = list(set(get_n_ancestors(client, roots[i], ceil(n_per_root * oversample_factor))))
        if seed is not None:
            random.seed(seed)
        random.shuffle(root_ancestors)
        ancestors.extend(root_ancestors[:n_per_root])
        if len(set(ancestors)) >= target_count * oversample_factor:
            break
    random.seed(seed) if seed is not None else random.seed()
    random.shuffle(ancestors)
    ancestors = list(set(ancestors))[:target_count]
    return ancestors

        

def get_random_ancestors(client : CAVEclient,  roots : List[int], target_count : int, months_back : float = 1, batch_size : int = 10, seed : Optional[int] = None):
    roots_sources = {}
    running_total = 0

    root_timestamps = client.chunkedgraph.get_root_timestamps(roots, latest=False)
    assert len(roots) == len(root_timestamps)

        

    n_per_lineage = max(1, target_count // len(roots_sources))
    print(f"Sampling {n_per_lineage} random ancestors per lineage.")


    random_ancestors = []
    for root in roots_sources:
        ancestors = roots_sources[root]
        root_ancestors = random.sample(ancestors, n_per_lineage) if len(ancestors) > n_per_lineage else ancestors
        random_ancestors.extend(root_ancestors)
    
    random_ancestors = list(set(random_ancestors))[:target_count]
    print("Sampled",len(random_ancestors),"unique random ancestors.")

    return random_ancestors


if __name__ == "__main__":
    client = get_client_for_species("mouse")
    latest_roots = get_latest_proofread_roots(client, "mouse")
    random_ancestors = get_random_ancestors(client, latest_roots, target_count=1000, months_back=1, batch_size=10, seed=42)
    
    print(random_ancestors)