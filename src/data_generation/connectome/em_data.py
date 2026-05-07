from __future__ import annotations
import time
import cloudvolume
import math
import datetime
from typing import Tuple, Sequence
import logging
import numpy as np
from tqdm import tqdm
import diskcache as dc
import os
from pathlib import Path
from scipy.ndimage import distance_transform_edt, zoom
from caveclient.chunkedgraph import ChunkedGraphClient
from caveclient import CAVEclient
from dotenv import load_dotenv
from utils.profiler import profiled

load_dotenv()


DATA_PARAMETERS = {
    "mouse": {
        "em_path": "precomputed://https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/minnie65/em",
        "seg_path": "graphene://https://minnie.microns-daf.com/segmentation/table/minnie65_public",
        "datastack_name": "minnie65_public",
        "em_mip": 2,
        "seg_mip": 2
    },
    "fly": {
        "em_path": "precomputed://https://bossdb-open-data.s3.amazonaws.com/flywire/fafbv14",
        "seg_path": "graphene://https://prod.flywire-daf.com/segmentation/1.0/flywire_public",
        "datastack_name": "flywire_fafb_public",
        "em_mip": 2,
        "seg_mip": 0
    },
    "human": {
        "em_path": "precomputed://gs://h01-release/data/20210601/4nm_raw",
        "seg_path": "graphene://https://local.brain-wire-test.org/segmentation/table/h01_full0_v2",
        "datastack_name": "h01_c3_flat",
        "em_mip": 1,
        "seg_mip": 0
    },
    "zebrafish": {
        "em_path": "precomputed://gs://fish1-public/clahe_231218",
        "seg_path": "graphene://https://pcgv3local.brain-wire-test.org/segmentation/table/fish1_v250915",
        "datastack_name": "fish1_full",
        "em_mip": 1,
        "seg_mip": 0
    }
}

CACHE_DIR = Path(os.getenv("CACHE_DIR", ".cache"))
CACHE_SIZE_LIMIT = 1024 * 1024 * 1024  # 1 GB

def _get_client_for_species(species: str) -> CAVEclient:
    if species == "fly":
        return CAVEclient("flywire_fafb_public")
    elif species == "mouse":
        return CAVEclient("minnie65_public")
    elif species in ["human", "zebrafish"]:
        token = os.getenv("CAVE_API_TOKEN")
        if not token:
            raise ValueError(f"{species} dataset requires CAVE_API_TOKEN env var")
        server = "https://global.brain-wire-test.org/"
        datastack = DATA_PARAMETERS[species]["datastack_name"]
        return CAVEclient(datastack, server_address=server, auth_token=token)
    else:
        raise ValueError(f"Invalid species: {species}")

def nm_to_voxel_1d(nm: int | float, resolution: int):
    nm = float(nm)
    resolution = int(resolution)
    return int(math.floor(nm / resolution))

def nm_to_voxel_3d(
    nm: Tuple[int, int, int] | list[int, int, int],
    resolution: Tuple[int, int, int] | list[int, int, int]
):
    nm = tuple(nm)
    resolution = tuple(resolution)
    if len(nm) != 3 or len(resolution) != 3:
        raise ValueError("nm and resolution must have 3 elements each")
    return tuple(nm_to_voxel_1d(n, r) for n, r in zip(nm, resolution))

def voxel_to_nm_1d(voxel: int, resolution: int):
    voxel = int(voxel)
    resolution = int(resolution)
    return voxel * resolution

def voxel_to_nm_3d(
    voxel: Tuple[int, int, int] | list[int, int, int],
    resolution: Tuple[int, int, int] | list[int, int, int]
):
    voxel = tuple(voxel)
    resolution = tuple(resolution)
    if len(voxel) != 3 or len(resolution) != 3:
        raise ValueError("voxel and resolution must have 3 elements each")
    return tuple(voxel_to_nm_1d(v, r) for v, r in zip(voxel, resolution))


class EMDataFetcher:
    """
    - A class for efficiently fetching EM and segmentation data from the data sources.
        - fetch_cutout() : Fetches the EM and segmentation data around a given position.
        - get_cutouts() : Gets the fetched (and optionally filtered), most up-to-date cutouts for the EM and segmentation data.
    - Supports parsing of and basic filtering for root IDs (at fetch time, or afterwards):
        - roots_filter (list[int]) : Used at fetch time to filter the data for the given roots.
        - filter_cutout_for_roots(roots_filter: list[int]) : Filters (previously fetched, unfiltered) data for the given roots.
        - get_cutout_roots() : Gets the roots for all supervoxel IDs in the cutout, at the provided timestamp (detected automatically, or provided in __init__).
            - If preserve_structure is True, returns a cutout of the roots.
            - If preserve_structure is False, returns a flattened array of unique roots.
        - unique_roots_in_cutout (property) : Same as get_cutout_roots(preserve_structure=False), cached.
    - This class maintains state for the most recently fetched/filtered cutout; methods like get_cutout_roots and filter_cutout_for_roots operate on that cutout.
        - If you want to filter the cutout for different roots, you must fetch a new cutout.
    """
    def __init__(self,
                species : str,
                roots_filter : list[int] = None,
                timestamp : datetime.datetime | str= "detect",
                data_parameters : dict = DATA_PARAMETERS,
                client : CAVEclient = None,
                logger : logging.Logger = None,
                quiet : bool = False,
                cv_em: cloudvolume.CloudVolume = None,
                cv_seg: cloudvolume.CloudVolume = None):
        """
        - Initialize the EMDataFetcher.

        - Args:
            - species (str): The species to fetch data for.
            - roots_filter (list[int]): The roots to filter the data by.
            - timestamp (datetime.datetime | str) : The timestamp to use for the data.
                - If "detect", the timestamp will be detected automatically from the first root in roots_filter.
                    - If timestamp="detect" and no roots_filter, latest timestamp is used by chunkedgraph.
                - If a datetime object, the timestamp will be used directly.

            - data_parameters (dict): The data parameters to use for the data.
            - client (CAVEclient): The client to use for the chunkedgraph. If not provided, a temporary client will be created.
            - logger (logging.Logger): The logger to use for logging.
            - cv_em (cloudvolume.CloudVolume): Pre-connected EM CloudVolume to reuse across fetcher instances.
            - cv_seg (cloudvolume.CloudVolume): Pre-connected segmentation CloudVolume to reuse across fetcher instances.


        """
        self.logger = logger if logger is not None else logging.getLogger()
        self.quiet = quiet
        self.logger.info(f"Initializing EMDataFetcher with timestamp: {timestamp} and roots_filter: {roots_filter}")
        self.species = species
        self.client = client

        if roots_filter is None:
            self.roots_filter = []
        else:
            self.roots_filter = roots_filter


        if isinstance(timestamp, str) and timestamp == "detect":
            self._detect_timestamp()
            self._is_detected_timestamp = True
        elif isinstance(timestamp, datetime.datetime):
            self.timestamp = timestamp
            self._is_detected_timestamp = False
        else:
            raise ValueError(f"Invalid timestamp type: {type(timestamp)}")

        self.position_nm = None
        self.window_size_nm = None

        self.data_parameters = data_parameters[species]
        self.em_mip = self.data_parameters["em_mip"]
        self.seg_mip = self.data_parameters["seg_mip"]

        self.em_path = self.data_parameters["em_path"]
        self.seg_path = self.data_parameters["seg_path"]

        # reuse pre-connected CloudVolumes if provided
        self._external_cv = cv_em is not None and cv_seg is not None
        if self._external_cv:
            self.cv_em = cv_em
            self.cv_seg = cv_seg
            self._em_resolution = cv_em.resolution
            self._seg_resolution = cv_seg.resolution
            self.em_bounds = cv_em.bounds
            self.seg_bounds = cv_seg.bounds
        else:
            self._connect_to_data_sources()

        self._em_resolution_arr = np.array(self._em_resolution, dtype=float)
        self._seg_resolution_arr = np.array(self._seg_resolution, dtype=float)
        self.resolution = tuple(int(r) for r in np.minimum(self._em_resolution_arr, self._seg_resolution_arr))
        self.logger.info(f"EMDataFetcher initialized with resolution: {self.resolution}")
        if not self._external_cv:
            del self.cv_em, self.cv_seg
        self._roots_cutout = None
        self._unique_roots_in_cutout = None
        
    def _connect_to_data_sources(self):
        """Connect to the EM and segmentation data sources."""

        try:
            # measure time start
            start_time = time.time()
            # 500MB LRU cache (500 * 1024 * 1024 bytes)
            lru_cache_bytes = 500 * 1024 * 1024

            # Get secrets for authenticated species (human/zebrafish)
            secrets = None
            if self.species in ["human", "zebrafish"]:
                token = os.getenv("CAVE_API_TOKEN")
                if token:
                    secrets = {"token": token}

            self.cv_em = cloudvolume.CloudVolume(self.em_path, use_https=True, mip=self.em_mip, timestamp=self.timestamp, cache=False, lru_bytes=lru_cache_bytes, secrets=secrets)
            self._em_resolution = self.cv_em.resolution
            self.em_bounds = self.cv_em.bounds

            self.cv_seg = cloudvolume.CloudVolume(self.seg_path, use_https=True, fill_missing=True, mip=self.seg_mip, timestamp=self.timestamp, cache=False, lru_bytes=lru_cache_bytes, secrets=secrets)
            self._seg_resolution = self.cv_seg.resolution
            self.seg_bounds = self.cv_seg.bounds

            self.logger.info(f"[EMDataFetcher] Successfully connected to data sources in {time.time() - start_time} seconds.")
        except Exception as e:
            self.logger.error(f"Error connecting to data sources: {e}")
            raise e

    def _detect_timestamp(self):
        """Detect the timestamp to use for the data."""
        if self.roots_filter is None or len(self.roots_filter) == 0:
            self.timestamp = None
        else:
            from connectome.utils import get_root_timestamp
            timestamps = []
            roots = self.roots_filter if isinstance(self.roots_filter, list) else [self.roots_filter]
            
            is_client_provided = self.client is not None
            client = _get_client_for_species(self.species) if self.client is None else self.client
            
            for root in roots:
                timestamps.append(get_root_timestamp(client, root))
            
            if not is_client_provided:
                del client
            
            self.logger.debug(f"Detected timestamp: {timestamps}")            
            self.timestamp = max(timestamps)

    def set_roots_filter(self, roots_filter : list[int]):
        """
        Set the roots filter for the EMDataFetcher.
        If the timestamp is set to "detect", it will be detected again.
        """
        self.roots_filter = roots_filter
        if self._is_detected_timestamp:
            self._detect_timestamp()

    @staticmethod
    @profiled("em_data.fetch_and_pad_volume")
    def _fetch_and_pad_volume(cv, resolution, position_nm, window_size_nm, bounds, logger):
        """Fetch a volume from CloudVolume, clipping to bounds and zero-padding."""
        window_vox = nm_to_voxel_3d(window_size_nm, resolution)
        pos_vox = nm_to_voxel_3d(position_nm, resolution)

        min_v = np.array([p - w // 2 for p, w in zip(pos_vox, window_vox)])
        max_v = np.array([m + w for m, w in zip(min_v, window_vox)])

        b_min = np.array(bounds.minpt if hasattr(bounds, 'minpt') else bounds[0])
        b_max = np.array(bounds.maxpt if hasattr(bounds, 'maxpt') else bounds[1])
        clip_lo = np.maximum(min_v, b_min)
        clip_hi = np.minimum(max_v, b_max)

        raw = cv[clip_lo[0]:clip_hi[0], clip_lo[1]:clip_hi[1], clip_lo[2]:clip_hi[2]][:,:,:,0]

        needs_pad = not (np.array_equal(clip_lo, min_v) and np.array_equal(clip_hi, max_v))
        if needs_pad:
            pad_widths = list(zip((clip_lo - min_v).astype(int), (max_v - clip_hi).astype(int)))
            raw = np.pad(np.asarray(raw), pad_widths, mode='constant', constant_values=0)
            logger.info(f"volume clipped to bounds (pad {pad_widths})")

        return np.asarray(raw), needs_pad

    @profiled("em_data.fetch_cutout")
    def fetch_cutout(
        self,
        position_nm : Tuple[float, float, float] | list[float, float, float],
        window_size_nm : int | Tuple[int, int, int] | list[int, int, int],
        with_roots : bool = False,
    ) -> Tuple["CutoutAccessor", "CutoutAccessor"]:
        """
        Fetch a cutout of the EM and segmentation data around a given position.

        Args:
            position_nm (Tuple[int, int, int] | list[int, int, int]): The position to fetch the cutout around (in nanometers).
            window_size_nm (int | Tuple[int, int, int] | list[int, int, int]): The window size to fetch the cutout in nanometers.
            with_roots (bool): If True (or if ``roots_filter`` is set on the fetcher) this resolves a ``roots_cutout``
                and attaches it to the returned segmentation accessor so filtered views can be produced locally.

        Returns:
            CutoutAccessor: The EM cutout accessor.
            CutoutAccessor: The segmentation cutout accessor.
        Notes:
            * When `roots_filter` is configured, the segmentation accessor returned here is already zeroed (filtered)
              for that root list, but its ``roots_cutout`` still reflects the full per-voxel assignments.
            * Subsequent calls to ``filter_supervoxels_for_roots`` or ``filtered_view`` on that accessor will
              mask the current (possibly already filtered) cutout with any new root list; repeated filtering can
              therefore only clear additional voxels, not resurrect ones filtered out earlier.
        """
        if isinstance(window_size_nm, (int, float)):
            window_size_nm = (window_size_nm, window_size_nm, window_size_nm)
        elif isinstance(window_size_nm, tuple) or isinstance(window_size_nm, list):
            assert len(window_size_nm) == 3, "Window size must be a tuple of 3 integers"
            window_size_nm = window_size_nm

        window_size_voxels = nm_to_voxel_3d(window_size_nm, self.resolution)
        position_voxels = nm_to_voxel_3d(position_nm, self.resolution)
        self.logger.info(f"Fetching cutout around position {position_nm} (nm) {position_voxels} (voxels) with window size {window_size_nm} (nm) {window_size_voxels} (voxels), for timestamp {self.timestamp}. (with_roots: {with_roots})")

        if not self._external_cv:
            self._connect_to_data_sources()

        # Fetch EM and seg independently at their native resolutions
        em_res = tuple(int(r) for r in self._em_resolution)
        seg_res = tuple(int(r) for r in self._seg_resolution)

        self.em_cutout, _ = self._fetch_and_pad_volume(
            self.cv_em, em_res, position_nm, window_size_nm, self.em_bounds, self.logger,
        )
        self.seg_cutout, _ = self._fetch_and_pad_volume(
            self.cv_seg, seg_res, position_nm, window_size_nm, self.seg_bounds, self.logger,
        )

        if not self._external_cv:
            del self.cv_em, self.cv_seg

        # Resample coarser volume to match the finer one's grid
        em_shape = np.array(self.em_cutout.shape, dtype=float)
        seg_shape = np.array(self.seg_cutout.shape, dtype=float)
        if not np.array_equal(em_shape, seg_shape):
            if np.prod(em_shape) < np.prod(seg_shape):
                # seg is finer → upsample EM (linear interp for continuous data)
                self.em_cutout = zoom(self.em_cutout, seg_shape / em_shape, order=1).astype(self.em_cutout.dtype)
                self.logger.info(f"resampled EM {tuple(em_shape.astype(int))} → {self.em_cutout.shape} to match seg")
            else:
                # em is finer → upsample seg (nearest-neighbor for discrete labels)
                self.seg_cutout = zoom(self.seg_cutout, em_shape / seg_shape, order=0)
                self.logger.info(f"resampled seg {tuple(seg_shape.astype(int))} → {self.seg_cutout.shape} to match em")

        assert self.em_cutout.shape == self.seg_cutout.shape, (
            f"shape mismatch after resample: em {self.em_cutout.shape} vs seg {self.seg_cutout.shape}"
        )

        if not self.quiet:
            if self.em_cutout.mean() <= 0:
                self.logger.warning("EM cutout has a mean less than or equal to 0")
            if self.seg_cutout.mean() <= 0:
                self.logger.warning("Segmentation cutout has a mean less than or equal to 0")

        self.em_cutout = np.asarray(self.em_cutout)
        self.seg_cutout = np.asarray(self.seg_cutout)

        self.position_nm = position_nm
        self.window_size_nm = window_size_nm
        self._unique_roots_in_cutout = None

        should_resolve_roots = with_roots or (self.roots_filter is not None and len(self.roots_filter) > 0)
        roots_cutout = None
        if should_resolve_roots:
            roots_cutout = self.get_cutout_roots(batch_size=32768, preserve_structure=True)
            assert roots_cutout is not None, "Unexpectedly received None from get_cutout_roots"
        self._roots_cutout = roots_cutout

        em_accessor = CutoutAccessor(self.em_cutout, position_nm, window_size_nm, self.resolution, type="em")
        seg_accessor = CutoutAccessor(
            self.seg_cutout,
            position_nm,
            window_size_nm,
            self.resolution,
            type="seg",
            roots_cutout=roots_cutout,
        )

        self.em_data = em_accessor
        self.seg_data = seg_accessor

        # if roots_filter is set, filter the segmentation accessor for the roots
        if self.roots_filter and roots_cutout is not None:
            filtered_seg = seg_accessor.filter_supervoxels_for_roots(
                roots_cutout,
                self.roots_filter,
                preserve_structure=True,
            )
            
            seg_accessor = CutoutAccessor(
                filtered_seg,
                position_nm,
                window_size_nm,
                self.resolution,
                type="seg",
                roots_cutout=roots_cutout,
            )
        else:
            self.logger.debug("No roots filter provided, returning unfiltered segmentation accessor")

        return (em_accessor, seg_accessor)

    def get_cutouts(self) -> Tuple["CutoutAccessor", "CutoutAccessor"]:
        """
        Get the fetched (and optionally filtered), most up-to-date cutouts for the EM and segmentation data.
        Returns:
            Tuple[CutoutAccessor, CutoutAccessor]: The EM and segmentation cutouts.
        Note:
            The segmentation cutout returned here is always the raw volume and ignores `roots_filter`;
            use the accessor returned by `fetch_cutout` when you need the auto-filtered view.
        """
        # return fresh accessor objects with current cutout data (note: segmentation cutout is always raw/unfiltered here)
        if hasattr(self, "em_data") and hasattr(self, "seg_data"):
            return (
                CutoutAccessor(self.em_data, self.position_nm, self.window_size_nm, self.resolution, type="em"),
                CutoutAccessor(self.seg_data, self.position_nm, self.window_size_nm, self.resolution, type="seg", roots_cutout=self._roots_cutout),
            )
        else:
            raise ValueError("EM and segmentation data not fetched yet")


    def _load_batched_roots_for_sv_ids(self, ids: np.ndarray, chunkedgraph : ChunkedGraphClient, timestamp: datetime.datetime = None, batch_size: int = 32768) -> np.ndarray:
        self.logger.debug(f"Loading roots for {len(ids)} SV IDs with timestamp: {timestamp}")

        if timestamp is None:
            self.logger.info("No timestamp provided, fetching roots for current timestamp.")
            timestamp = datetime.datetime.now()
        out = []
        total = len(ids)
        for start in tqdm(range(0, total, batch_size), desc=f"Fetching roots ({timestamp if timestamp else datetime.datetime.now()})", disable=self.quiet):
            chunk = ids[start : start + batch_size]
            if timestamp is None:
                roots = chunkedgraph.get_roots(chunk)
            else:
                roots = chunkedgraph.get_roots(chunk, timestamp=timestamp)
            out.append(np.asarray(roots, dtype=np.int64))
        
        return_value = np.concatenate(out) if out else np.empty(0, dtype=np.int64)
        return return_value
         
    @property
    def unique_roots_in_cutout(self) -> np.ndarray:
        if getattr(self, "_unique_roots_in_cutout", None) is not None:
            return self._unique_roots_in_cutout
        self._unique_roots_in_cutout = self.get_cutout_roots(batch_size=32768, preserve_structure=False)
        return self._unique_roots_in_cutout

    @profiled("em_data.get_cutout_roots")
    def get_cutout_roots(
        self,
        batch_size: int = 32768,
        preserve_structure: bool = True,
    ) -> np.ndarray:
        """
        Cached getter for the roots for all supervoxel IDs in the most recently fetched segmentation cutout.
        Args:
            batch_size (int): The batch size to use for the roots, if not cached.
            preserve_structure (bool): Whether to preserve the structure of the roots. True returns a cutout, False returns a flattened array of unique roots.
        Returns:
            np.ndarray: The roots for the supervoxel IDs.
        """
        if getattr(self, "seg_cutout", None) is None:
            raise ValueError("Segmentation cutout not fetched yet")

        position_voxels = nm_to_voxel_3d(self.position_nm, self.resolution)
        window_size_voxels = nm_to_voxel_3d(self.window_size_nm, self.resolution)

        cache = CutoutDiskCache(
            self.species,
            position_voxels,
            window_size_voxels,
            self.resolution,
            "cutout" if preserve_structure else "flat",
            self.timestamp,
        )
        if not cache.is_cache_hit:
            self._roots_cutout = self.load_cutout_roots(
                    supervoxels=self.seg_cutout,
                    batch_size=batch_size,
                    preserve_structure=preserve_structure,
                )
            cache.set(
                self._roots_cutout,
            )
        else:
            self.logger.debug(f"Cache hit for roots cutout at position {self.position_nm} with window size {self.window_size_nm}, returning cached values.")
            self._roots_cutout = cache.get()
        return self._roots_cutout
        
    @profiled("em_data.load_cutout_roots")
    def load_cutout_roots(
        self,
        supervoxels: np.ndarray,
        batch_size: int = 32768,
        preserve_structure: bool = True,
    ) -> np.ndarray:
        """
        Loads the roots for all supervoxel IDs in the provided cutout (or the most recently fetched segmentation).
        Args:
            supervoxels (np.ndarray | None): Array of supervoxel IDs. Defaults to the cached segmentation cutout.
            batch_size (int): The batch size to use for the roots.
            preserve_structure (bool): Whether to preserve the structure of the roots. True returns a cutout, False returns a flattened array of unique roots.
        Returns:
            np.ndarray: The roots for the supervoxel IDs.
        """
        original_shape = supervoxels.shape
        if supervoxels.size == 0:
            return np.zeros_like(supervoxels) if preserve_structure else np.empty(0, dtype=np.int64)

        flat_svs = supervoxels.reshape(-1)

        is_client_provided = self.client is not None
        client = self.client if is_client_provided else _get_client_for_species(self.species)

        if client is None or not hasattr(client, "chunkedgraph"):
            raise ValueError("Client with a chunkedgraph attribute is required")

        unique_ids, inverse_idx = np.unique(flat_svs, return_inverse=True)
        nonzero_mask = unique_ids != 0
        roots_map = np.zeros_like(unique_ids, dtype=np.int64)

        if np.any(nonzero_mask):
            nonzero_ids = unique_ids[nonzero_mask]
            roots_nonzero = self._load_batched_roots_for_sv_ids(
                ids=nonzero_ids,
                chunkedgraph=client.chunkedgraph,
                timestamp=self.timestamp,
                batch_size=batch_size,
            )
            if roots_nonzero.size != nonzero_ids.size:
                raise ValueError("get_roots returned an unexpected number of assignments")
            roots_map[nonzero_mask] = roots_nonzero

        if not is_client_provided:
            del client

        unique_roots_nonzero = np.unique(roots_map[nonzero_mask]) if np.any(nonzero_mask) else np.empty(0, dtype=np.int64)

        if not preserve_structure:
            if supervoxels is None:
                self._unique_roots_in_cutout = unique_roots_nonzero
            return unique_roots_nonzero

        roots_flat = roots_map[inverse_idx]
        roots_cutout = roots_flat.reshape(original_shape)

        if supervoxels is None:
            self._roots_cutout = roots_cutout
            self._unique_roots_in_cutout = unique_roots_nonzero
        return roots_cutout

class OutOfBoundsError(ValueError):
    """
    Raised when a global coordinate is out of bounds for a cutout.
    """
    pass

class CutoutAccessor:
    def __init__(self, cutout : np.ndarray, center_nm : Tuple[float, float, float], window_size_nm : Tuple[float, float, float], resolution : Tuple[int, int, int], type : str = None, roots_cutout: np.ndarray | None = None):
        """
        Initialize the CutoutAccessor.
        Args:
            cutout (np.ndarray): The cutout data.
            center_nm (Tuple[float, float, float]): The center of the cutout in global (world) coordinates.
            window_size_nm (Tuple[float, float, float]): The window size of the cutout in global (world) coordinates.
            resolution (Tuple[int, int, int]): The resolution of the cutout.
        """
        self.cutout = cutout
        self.center_nm = center_nm
        self.window_size_nm = window_size_nm
        self.resolution = resolution
        self.type = type
        self._roots_cutout = None
        if roots_cutout is not None:
            self._attach_roots_cutout(roots_cutout)

        # compute shape in world space (voxels, and nm), compare to shape of cutout, and set min and max voxels
        self.shape_vox = cutout.shape
        self.center_vox = nm_to_voxel_3d(center_nm, resolution)

        half_shape_vox = tuple(s // 2 for s in self.shape_vox)
        self.min_vox = tuple(c - h for c, h in zip(self.center_vox, half_shape_vox))
        self.max_vox = tuple(m + s for m, s in zip(self.min_vox, self.shape_vox))

        # print(f"min_vox: {self.min_vox}, max_vox: {self.max_vox}")

        self.min_nm = voxel_to_nm_3d(self.min_vox, resolution)
        self.max_nm = voxel_to_nm_3d(self.max_vox, resolution)

    
        vox_range = tuple(max_v - min_v for max_v, min_v in zip(self.max_vox, self.min_vox)) # ensure int
        assert vox_range == self.shape_vox, f"window size range {vox_range} (voxels), {self.window_size_nm} (nm) must match cutout shape {self.shape_vox} (voxels)"

        # if self.type == "seg":
        #     print(f"=== CUTOUT ACCESSOR coord debug ===")
        #     print(f"min_nm: {self.min_nm}, max_nm: {self.max_nm}")
        #     print(f"min_vox: {self.min_vox}, max_vox: {self.max_vox}")
        #     print(f"shape_vox: {self.shape_vox}")
        #     print(f"window_size_nm: {self.window_size_nm}")
        #     print(f"resolution: {self.resolution}")
        #     print(f"center_nm: {self.center_nm}")
        #     print(f"center_vox: {self.center_vox}")
        #     print(f"cutout.shape: {self.cutout.shape}")

    def mean(self) -> float:
        """
        Compute the mean of the cutout.
        Returns:
            float: The mean of the cutout.
        """
        return np.mean(self.cutout)

    def global_nm_to_cutout_voxels(self, global_nm : Tuple[float, float, float]) -> Tuple[int, int, int]:
        """
        Convert global (world) coordinates to local (cutout) coordinates.
        Args:
            global_nm (Tuple[float, float, float]): The global (world) coordinates to convert to local (cutout) coordinates.
        Returns:
            Tuple[int, int, int]: The local (cutout) coordinates.
        """
        in_bounds = all(
            (mn <= g < mx) 
            for g, mn, mx in zip(global_nm, self.min_nm, self.max_nm)
        )
        if not in_bounds:
            raise OutOfBoundsError(f"Global coordinates {global_nm} are out of bounds: min_bound {self.min_nm} <= coords {global_nm} < max_bound{self.max_nm}. Try increasing the window size {self.window_size_nm} or moving the cutout center {self.center_nm}.")

        global_voxels = nm_to_voxel_3d(global_nm, self.resolution)
        local_voxels = np.asarray(global_voxels) - np.asarray(self.min_vox)
        if not np.all(local_voxels >= 0) or not np.all(local_voxels < self.shape_vox):
            raise AssertionError(f"Voxel coordinates out of bounds: min_bound {self.min_vox} <= coords {local_voxels+self.min_vox} < max_bound{self.max_vox}")        
        return tuple(local_voxels.astype(int))

    def cutout_voxels_to_global_nm(self, cutout_voxels : Tuple[int, int, int]) -> Tuple[float, float, float]:
        """
        Convert local (cutout) voxel coordinates to global (world) nm coordinates.
        Args:
            cutout_voxels (Tuple[int, int, int]): The local (cutout) voxel coordinates to convert to global (world) nm coordinates.
        Returns:
            Tuple[float, float, float]: The global (world) coordinates.
        """
        global_voxels = np.asarray(cutout_voxels) + np.asarray(self.min_vox) # move up by min_vox to get global coordinates
        return voxel_to_nm_3d(global_voxels, self.resolution) # convert to global nm coordinates

    def find_coords_of_sv(self, sv_id : int) -> Tuple[float, float, float]:
        """
        Find the mean coordinates of a supervoxel in the cutout.
        Args:
            sv_id (int): The supervoxel ID to find.
        Returns:
            Tuple[float, float, float]: The coordinates of the supervoxel.
        """
        assert self.type is not None and self.type == "seg", "CutoutAccessor must be of type 'seg' to find supervoxel coordinates"
        coords = np.mean(np.argwhere(self.cutout == sv_id), axis=0, dtype=float)
        if coords.size == 0 or np.isnan(coords).any():
            raise ValueError(f"Supervoxel ID {sv_id} not found in cutout or mean coordinate is NaN.")
        return coords

    def _valid_local_sv_coords(self, coords : Tuple[int, int, int]) -> bool:
        """
        Check if a given set of local, voxel coordinates is within the cutout.
        Args:
            coords (Tuple[int, int, int]): The coordinates to check.
        Returns:
            bool: True if the coordinates are within the cutout, False otherwise.
        """
        assert len(coords) == 3, "Coordinates must be a tuple of 3 values."
        return all(0 <= c < s for c, s in zip(coords, self.shape_vox))

    @property
    def sv_coords(self) -> dict:
        """
        Get a dict of all supervoxel IDs in the cutout and their mean (local, voxel) coordinates.
        Returns:
            dict: A dict of all supervoxel IDs in the cutout and their coordinates. Key is the supervoxel ID, value is a 3d array of their mean (local, voxel) coordinates.
        """
        if hasattr(self, "_sv_coords"):
            return self._sv_coords
        else:
            self._sv_coords = self._compute_sv_coords()
            return self._sv_coords

    @property
    def dist_transform(self) -> np.ndarray:
        """
        Tuple of (dist_vox, idx_x, idx_y, idx_z):
            dist_vox: distance transform in voxels
            idx_x: x indices of the distance transform
            idx_y: y indices of the distance transform
            idx_z: z indices of the distance transform
        """
        ## compute distance transform
        assert self.type is not None and self.type == "seg", "CutoutAccessor must be of type 'seg' to compute distance transform"

        mask = self.cutout == 0    # background
        if mask.size == 0 or not np.any(mask):
            return None

        # dist_vox, idx_x, idx_y, idx_z
        dist_vox, indices =distance_transform_edt(
            mask,
            return_indices=True,
        )
        return dist_vox, indices[0], indices[1], indices[2]

    def _compute_sv_coords(self) -> dict:
        """
        Get a dict of all supervoxel IDs in the cutout and their coordinates.
        Returns:
            dict: A dict of all supervoxel IDs in the cutout and their coordinates. Key is the supervoxel ID, value is a 3d array of their mean (local, voxel) coordinates.
        """
        vol = self.cutout
        mask = vol > 0
        if vol.size == 0 or not np.any(mask):
            return {}

        sv_ids = vol[mask]  # shape (N_nonzero,)

        # get indices of all nonzero voxels
        x_idx, y_idx, z_idx = np.nonzero(mask)

        # reindex labels to [0..n_labels-1] to keep bincount small
        unique_ids, inv = np.unique(sv_ids, return_inverse=True)
        # inv has same shape as sv_ids, values in [0, len(unique_ids)-1]

        counts = np.bincount(inv).astype(float)

        sum_x = np.bincount(inv, weights=x_idx.astype(float))
        sum_y = np.bincount(inv, weights=y_idx.astype(float))
        sum_z = np.bincount(inv, weights=z_idx.astype(float))

        mean_x = sum_x / counts
        mean_y = sum_y / counts
        mean_z = sum_z / counts

        centers = np.stack([mean_x, mean_y, mean_z], axis=1)  # (n_labels, 3)

        # build dict: sv_id -> np.array([x, y, z])
        return {int(sv_id): centers[i] for i, sv_id in enumerate(unique_ids)}

    def find_sv_at_coords(self, coords : Tuple[float, float, float], tolerance_nm : float = 300.0) -> int:
        """
        Find the closest supervoxel ID for a given set of local, voxel coordinates, within a given tolerance.
        Args:
            coords (Tuple[float, float, float]): The coordinates to find the supervoxel ID of.
            tolerance_nm (float): The tolerance in nanometers.
        Returns:
            int: The closest supervoxel ID.
        """
        assert self.type is not None and self.type == "seg", "CutoutAccessor must be of type 'seg' to find supervoxel ID."
        shape_bounds = ", ".join(str(s) for s in self.shape_vox)
        assert self._valid_local_sv_coords(coords), f"Local coordinates {coords} are out of bounds for cutout shape {self.shape_vox}."
        target_vox = np.asarray(coords, dtype=int)
        tolerance_vox = nm_to_voxel_3d((tolerance_nm, tolerance_nm, tolerance_nm), self.resolution)
        # first, check if the target voxel is directly on a supervoxel
        immediate_sv_id = self.cutout[target_vox[0], target_vox[1], target_vox[2]]
        if immediate_sv_id != 0:
            return immediate_sv_id

        # if not, find the closest supervoxel ID within the tolerance         
        # get distance transform

        dist_transform = self.dist_transform

        if dist_transform is None:
            print(f"[CutoutAccessor] Warning: No distance transform found for coordinates {coords}")
            return None
        
        dist_vox, idx_x, idx_y, idx_z = dist_transform

        # determine indices of closest voxel to target voxel
        x, y, z = target_vox
        closest_x = idx_x[x, y, z]
        closest_y = idx_y[x, y, z]
        closest_z = idx_z[x, y, z]

        # get supervoxel ID of closest voxel
        sv_id = self.cutout[closest_x, closest_y, closest_z]

        # check if closest voxel is within tolerance
        within_tolerance = all(dist_vox[x,y,z] <= tol_i for tol_i in tolerance_vox)
        
        if within_tolerance and sv_id != 0:
            return int(sv_id)
        else:
            print(f"[CutoutAccessor] Warning: No SV within tolerance {tolerance_nm} nm ({tolerance_vox} voxels) was found for coordinates {coords}. Closest SV is {dist_vox[x, y, z]} voxels away")
            return None # or return 0 / None

    def _attach_roots_cutout(self, roots_cutout: np.ndarray) -> None:
        roots_arr = np.asarray(roots_cutout, dtype=np.int64)
        if roots_arr.shape != tuple(self.cutout.shape):
            raise ValueError(f"roots_cutout shape {roots_arr.shape} must match cutout shape {self.cutout.shape}")
        self._roots_cutout = roots_arr

    @property
    def roots_cutout(self) -> np.ndarray | None:
        if self._roots_cutout is None:
            raise ValueError("Roots_cutout is not set for this cutout. Did you fetch the cutout with with_roots=True?")
        return self._roots_cutout

    @property
    def unique_roots(self) -> np.ndarray:
        if self._roots_cutout is None:
            raise ValueError("roots_cutout is not set for this accessor")
        filtered = self._roots_cutout[self._roots_cutout != 0]
        return np.unique(filtered) if filtered.size > 0 else np.empty(0, dtype=np.int64)

    def filter_supervoxels_for_roots(
        self,
        roots_cutout: np.ndarray | None = None,
        roots_filter: Sequence[int] | np.ndarray | int | None = None,
        preserve_structure: bool = True,
    ) -> np.ndarray:
        """
        Filter this segmentation cutout by matching roots.
        Args:
            roots_cutout (np.ndarray | None): Broadcasted roots volume that matches ``self.cutout``. Defaults to the attached roots_cutout.
            roots_filter (Sequence[int] | np.ndarray | None): Roots to keep within the cutout. ``None`` returns the original cutout.
            preserve_structure (bool): Return a cutout with background zeros or a flattened array of unique matching SV IDs.
        Notes:
            * ``self.cutout`` may already have been zeroed by a previous filter, so re-filtering can only remove
              additional voxels (the underlying ``roots_cutout`` remains complete, but the cutout itself loses data).
        """
        assert self.type == "seg", "Root filtering is only supported for segmentation cutouts"

        roots_array = roots_cutout if roots_cutout is not None else self._roots_cutout
        if roots_array is None:
            raise ValueError("roots_cutout is not available; fetch it from the EMDataFetcher first")

        roots_array = np.asarray(roots_array, dtype=np.int64)
        if roots_array.shape != tuple(self.cutout.shape):
            raise ValueError(f"roots_cutout shape {roots_array.shape} must match cutout shape {self.cutout.shape}")

        roots_filter_arr: np.ndarray | None = None
        if roots_filter is not None:
            if isinstance(roots_filter, (int, np.integer)):
                roots_filter_arr = np.asarray([int(roots_filter)], dtype=np.int64)
            else:
                arr = np.asarray(roots_filter, dtype=np.int64)
                if arr.ndim == 0:
                    arr = arr.reshape(1)
                roots_filter_arr = arr

        if roots_filter_arr is None or roots_filter_arr.size == 0:
            if preserve_structure:
                return np.asarray(self.cutout, dtype=np.int64)
            return np.unique(np.asarray(self.cutout, dtype=np.int64))

        roots_filter_arr = np.unique(roots_filter_arr)
        mask = np.isin(roots_array, roots_filter_arr)

        if preserve_structure:
            filtered = np.zeros_like(self.cutout, dtype=np.int64)
            filtered[mask] = self.cutout[mask]
            return filtered

        matching = self.cutout[mask]
        if matching.size == 0:
            return np.empty(0, dtype=np.int64)
        return np.unique(matching)

    def filtered_view(
        self,
        roots_filter: Sequence[int] | np.ndarray | int | None = None,
        preserve_structure: bool = True,
        as_accessor: bool = False,
    ) -> np.ndarray | CutoutAccessor:
        """
        Get a filtered view of the cutout.
        Args:
            roots_filter (Sequence[int] | np.ndarray | int | None): Roots to keep within the cutout. ``None`` returns the original cutout.
            preserve_structure (bool): Return a cutout with background zeros or a flattened array of unique matching SV IDs.
            as_accessor (bool): Return a CutoutAccessor object instead of a numpy array.
        Returns:
            np.ndarray | CutoutAccessor: The filtered cutout.
        """
        filtered = self.filter_supervoxels_for_roots(
            roots_filter=roots_filter,
            preserve_structure=preserve_structure,
        )
        if not as_accessor:
            return filtered
        return CutoutAccessor(
            filtered,
            self.center_nm,
            self.window_size_nm,
            self.resolution,
            type=self.type,
            roots_cutout=self._roots_cutout,
        )

    @staticmethod
    def _normalize_filter_ids(filter_ids):
        if filter_ids is None:
            return None
        arr = np.asarray(filter_ids, dtype=np.int64)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        return arr.ravel()

    def _filter_adjacency_edges(self, edges: np.ndarray, filter_ids: Sequence[int] | np.ndarray | None) -> np.ndarray:
        filter_arr = self._normalize_filter_ids(filter_ids)
        if filter_arr is None or filter_arr.size == 0:
            return edges
        mask = np.isin(edges[:, 0], filter_arr) | np.isin(edges[:, 1], filter_arr)
        return edges[mask]

    def get_adjacency_edges(self, filter_ids: Sequence[int] | np.ndarray | int | None = None) -> np.ndarray:
        """
        Compute 6-connectivity edges between root IDs (non-zero IDs only).
        Returns a (n, 2) array of unique sorted ID pairs that touch along one face.
        """
        edges = []
        for ax in range(3):
            slicer1 = [slice(None), slice(None), slice(None)]
            slicer2 = [slice(None), slice(None), slice(None)]
            slicer1[ax] = slice(0, -1)
            slicer2[ax] = slice(1, None)
            A = self.roots_cutout[tuple(slicer1)]
            B = self.roots_cutout[tuple(slicer2)]

            mask = (A != B) & (A != 0) & (B != 0)
            if not np.any(mask):
                continue

            A_faces = A[mask]
            B_faces = B[mask]
            pairs = np.stack([A_faces, B_faces], axis=1)
            pairs.sort(axis=1)
            edges.append(pairs)

        if not edges:
            return np.empty((0, 2), dtype=np.int64)
        all_edges = np.concatenate(edges, axis=0)
        unique_edges = np.unique(all_edges, axis=0)
        return self._filter_adjacency_edges(unique_edges, filter_ids)

    def get_adjacent_roots(self, of_ids: Sequence[int] | np.ndarray) -> np.ndarray:
        """
        Return unique IDs that neighbor at least one of the provided IDs. The provided IDs themselves are excluded.
        """
        if of_ids is None:
            raise ValueError("of_ids must be provided")
        edges = self.get_adjacency_edges(filter_ids=of_ids)
        unique_ids = np.unique(edges)
        filter_arr = self._normalize_filter_ids(of_ids)
        if filter_arr is None or filter_arr.size == 0:
            return unique_ids
        return unique_ids[~np.isin(unique_ids, filter_arr)]

    @staticmethod
    def _filter_cutout_for_masks(cutout, masks):
        result = np.zeros_like(cutout)
        combined_mask = np.logical_and.reduce(masks)
        result[combined_mask] = cutout[combined_mask]
        return result

    def find_interface_point(
        self,
        id_a: int,
        id_b: int,
        mode: str = "root",
        inner_frac: float = 0.5,
    ) -> np.ndarray | None:
        """Find the closest-approach point between two segments via distance transforms.

        Args:
            id_a: first segment ID.
            id_b: second segment ID.
            mode: 'root' uses roots_cutout, 'sv' uses raw seg cutout.
            inner_frac: restrict search to the central fraction of the volume
                (0.5 = inner 50%) to avoid edge artifacts.

        Returns:
            Global nm coordinates of the interface point, or None if either
            segment is missing from the volume.
        """
        vol = self.roots_cutout if mode == "root" else self.cutout
        mask_a = vol == id_a
        mask_b = vol == id_b
        if not mask_a.any() or not mask_b.any():
            return None

        dist_a = distance_transform_edt(~mask_a)
        dist_b = distance_transform_edt(~mask_b)
        combined = dist_a + dist_b

        # restrict to inner fraction
        if 0.0 < inner_frac < 1.0:
            margin = tuple(int(s * (1 - inner_frac) / 2) for s in vol.shape)
            inner_mask = np.zeros(vol.shape, dtype=bool)
            inner_mask[
                margin[0]:vol.shape[0] - margin[0],
                margin[1]:vol.shape[1] - margin[1],
                margin[2]:vol.shape[2] - margin[2],
            ] = True
            combined[~inner_mask] = np.inf

        idx = np.unravel_index(np.argmin(combined), combined.shape)
        return np.array(self.cutout_voxels_to_global_nm(idx), dtype=float)

    def crop(self, center_nm: Tuple[float, float, float], extent_nm: float) -> "CutoutAccessor":
        """Return a new CutoutAccessor cropped around a new center.

        Args:
            center_nm: new center in global nm coordinates.
            extent_nm: half-extent in nm (isotropic).

        Returns:
            New CutoutAccessor with cropped volumes and updated bounds.
        """
        half_vox = np.array([extent_nm / r for r in self.resolution], dtype=float)
        new_center_vox = np.array(nm_to_voxel_3d(center_nm, self.resolution), dtype=int)
        local_center = new_center_vox - np.array(self.min_vox, dtype=int)

        lo = np.maximum(local_center - (half_vox).astype(int), 0)
        hi = np.minimum(local_center + (half_vox).astype(int), np.array(self.shape_vox))
        slices = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))

        new_cutout = self.cutout[slices]
        new_roots = self._roots_cutout[slices] if self._roots_cutout is not None else None
        new_shape_nm = tuple(float((b - a) * r) for a, b, r in zip(lo, hi, self.resolution))

        return CutoutAccessor(
            new_cutout,
            center_nm=tuple(center_nm),
            window_size_nm=new_shape_nm,
            resolution=self.resolution,
            type=self.type,
            roots_cutout=new_roots,
        )

    def get_roots_in_n_neighborhood_of_root(self, root_id: int, n_nm: float | int, preserve_shape: bool = False) -> np.ndarray:
        """
        Get a list of unique root ids in the n-neighborhood of the provided root id. Zero and the root itself are excluded.
        Args:
            root_id (int): The root id to get the neighborhood of.
            n_nm (float | int): The extent of the neighborhood in nanometers (from edge of root to edge of neighborhood).
            preserve_shape (bool): Whether to return the neighborhood as a cutout with background zeros or a flattened array of unique matching SV IDs.
        Returns:
            np.ndarray: A list of unique root ids in the n-neighborhood of the provided root id.
        """
        sv_cutout = self.cutout
        root_cutout = self.roots_cutout
        
        assert sv_cutout.ndim == 3 
        assert sv_cutout.shape == root_cutout.shape
        assert np.any(root_cutout == root_id)

        if n_nm <= 0 + 1e-6:
            return np.array([])

        root_mask = root_cutout != root_id # 0 where root is, 1 otherwise

        assert len(np.unique(sv_cutout[root_mask != 0])) > 1, "Supervoxels of neighboring roots must be present. Cutout must be unfiltered."
        
        # get the distance transform of the root mask
        dist_transform = distance_transform_edt(root_mask, sampling=self.resolution)

        # threshold the distance transform to get the n-neighborhood mask
        neighborhood_mask = dist_transform <= n_nm
        non_root_mask = root_cutout != root_id
        # apply the masks to the supervoxel cutout
        result = self._filter_cutout_for_masks(root_cutout, [neighborhood_mask, non_root_mask])
        if preserve_shape:
            return result
        else:
            return np.unique(result[result != 0])

class CutoutDiskCache:
    def __init__(
        self,
        species,
        position_voxels: Tuple[int, int, int],
        window_size_voxels: Tuple[int, int, int],
        resolution: Tuple[int, int, int],
        cutout_shape: str,
        timestamp: datetime,
    ):
        """
        Cache for roots cutout.
        Args:
            species (str): The species of the data.
            position_voxels (Tuple[int, int, int]): The position of the cutout in voxels.
            window_size_voxels (Tuple[int, int, int]): The window size of the cutout in voxels.
            resolution (Tuple[int, int, int]): The resolution of the data.
            cutout_shape (str): The shape of the cutout ('flat' for flattened array, 'cutout' for cutout).
            timestamp (datetime.datetime): Timestamp of the cutout.
        """
        self.cache = dc.Cache(CACHE_DIR / "roots_cutout_cache", eviction_policy="least-recently-used", size_limit=CACHE_SIZE_LIMIT)
        self.is_cache_hit = False

        self.cutout_shape_str = cutout_shape
        self.shape = window_size_voxels if cutout_shape == 'cutout' else None

        if isinstance(timestamp, datetime.datetime):
            self.timestamp = int(timestamp.timestamp())
        else:
            self.timestamp = int(timestamp) if timestamp is not None else None
        self.key = f"{species}:{position_voxels}:{window_size_voxels}:{resolution}:{cutout_shape}:{self.timestamp if self.timestamp is not None else 'latest'}"

        if self.key in self.cache:
            self.is_cache_hit = True
    
    def get(self):
        if not self.is_cache_hit:
            return None
        return self.cache.get(self.key)

    def set(self, value: np.ndarray):
        assert self.cutout_shape_str in ['flat', 'cutout'], f"Invalid cutout shape: {self.cutout_shape_str}, must be 'flat' or 'cutout'"
        if self.cutout_shape_str == 'flat':
            assert value.ndim == 1, f"Flat cutout must be a 1D array, got {value.ndim}D array"
        elif self.cutout_shape_str == 'cutout':
            assert value.ndim == 3 and value.shape == self.shape, f"Cutout must be a 3D array of shape {self.window_size_voxels}, got {value.shape}"
        self.cache.set(self.key, value)
        self.is_cache_hit = True
    
if __name__ == "__main__":
    node_position = (926288.0, 390952.0, 819600.0) 
    fetcher = EMDataFetcher("mouse", roots_filter=[864691135572735469])
    fetcher.fetch_cutout(node_position, 5000)
    cutout = fetcher.seg_cutout
