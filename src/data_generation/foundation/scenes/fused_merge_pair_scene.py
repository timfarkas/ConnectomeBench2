"""
FusedMergePairScene: compact pair representation for merge tasks.

Per-view channel layout [V, 7, H, W]:
    0: silhouette   (union of visible A/B pixels)
    1: depth        (front-most visible surface depth)
    2: normal_x     (front-most visible surface normal x-component)
    3: normal_y     (front-most visible surface normal y-component)
    4: normal_z     (front-most visible surface normal z-component)
    5: a_mask       (binary: segment A visible at pixel)
    6: b_mask       (binary: segment B visible at pixel)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from foundation.scenes.merge_pair_scene import MergePairScene
from foundation.scenes.mesh_scene import ViewConfig

FUSED_MERGE_CHANNEL_NAMES = ("silhouette", "depth", "nx", "ny", "nz", "a_mask", "b_mask")
N_CHANNELS_FUSED_MERGE = 7


def split_mask_from_pair_views(views: np.ndarray) -> np.ndarray:
    """Derive a depth-resolved split-mask label from MergePairScene views.

    Input:  [V, 11, H, W] (per-segment silhouette+depth+normals + contact).
    Output: [V, H, W] uint8 with {0=A, 128=B, 255=BG}.
    Overlap (both visible) is resolved by which segment is closer in depth.
    """
    if views.shape[1] != 11:
        raise ValueError(f"expected [V,11,H,W], got {views.shape}")

    a_sil = np.clip(views[:, 0].astype(np.float32), 0.0, 1.0)
    a_depth = views[:, 1].astype(np.float32)
    b_sil = np.clip(views[:, 5].astype(np.float32), 0.0, 1.0)
    b_depth = views[:, 6].astype(np.float32)

    a_only = (a_sil > 0.5) & (b_sil <= 0.5)
    b_only = (b_sil > 0.5) & (a_sil <= 0.5)
    both = (a_sil > 0.5) & (b_sil > 0.5)
    a_front = both & (a_depth <= b_depth)
    b_front = both & ~a_front

    label = np.full(a_sil.shape, 255, dtype=np.uint8)
    label[a_only | a_front] = 0
    label[b_only | b_front] = 128
    return label


def fuse_merge_pair_views(views: np.ndarray) -> np.ndarray:
    """Fuse MergePairScene views [V, 11, H, W] into [V, 7, H, W]."""
    if views.shape[1] != 11:
        raise ValueError(f"expected [V,11,H,W], got {views.shape}")

    a_sil = np.clip(views[:, 0], 0.0, 1.0)
    a_depth = views[:, 1]
    a_normals = views[:, 2:5]

    b_sil = np.clip(views[:, 5], 0.0, 1.0)
    b_depth = views[:, 6]
    b_normals = views[:, 7:10]

    # Choose front-most segment normals where both project to the same pixel.
    a_only = (a_sil > 0.5) & (b_sil <= 0.5)
    b_only = (b_sil > 0.5) & (a_sil <= 0.5)
    both = (a_sil > 0.5) & (b_sil > 0.5)
    a_front = both & (a_depth <= b_depth)
    b_front = both & ~a_front

    a_visible = a_only | a_front
    b_visible = b_only | b_front

    silhouette = np.maximum(a_sil, b_sil).astype(np.float32)
    depth = np.zeros_like(a_depth, dtype=np.float32)
    normals = np.zeros_like(a_normals, dtype=np.float32)
    normals[:, 0] = np.where(a_visible, a_normals[:, 0], normals[:, 0])
    normals[:, 1] = np.where(a_visible, a_normals[:, 1], normals[:, 1])
    normals[:, 2] = np.where(a_visible, a_normals[:, 2], normals[:, 2])
    depth = np.where(a_visible, a_depth, depth)
    normals[:, 0] = np.where(b_visible, b_normals[:, 0], normals[:, 0])
    normals[:, 1] = np.where(b_visible, b_normals[:, 1], normals[:, 1])
    normals[:, 2] = np.where(b_visible, b_normals[:, 2], normals[:, 2])
    depth = np.where(b_visible, b_depth, depth)

    fused = np.concatenate(
        [
            silhouette[:, np.newaxis].astype(np.float16),
            depth[:, np.newaxis].astype(np.float16),
            normals.astype(np.float16),
            a_sil[:, np.newaxis].astype(np.float16),
            b_sil[:, np.newaxis].astype(np.float16),
        ],
        axis=1,
    )
    return fused


@dataclass
class FusedMergePairScene:
    views: np.ndarray       # [V, 7, H, W] float16
    anchor_nm: np.ndarray   # [3] float64
    segment_id_a: int
    segment_id_b: int
    species: str = ""
    view_config: ViewConfig = field(default_factory=ViewConfig)

    CHANNEL_NAMES = FUSED_MERGE_CHANNEL_NAMES
    N_CHANNELS = N_CHANNELS_FUSED_MERGE

    def __post_init__(self):
        self.anchor_nm = np.asarray(self.anchor_nm, dtype=np.float64)
        if self.views.dtype != np.float16:
            self.views = self.views.astype(np.float16)

    def validate(self) -> None:
        expected = (
            self.view_config.n_views,
            N_CHANNELS_FUSED_MERGE,
            self.view_config.resolution,
            self.view_config.resolution,
        )
        if self.views.shape != expected:
            raise ValueError(f"views shape {self.views.shape} != expected {expected}")
        if self.anchor_nm.shape != (3,):
            raise ValueError(f"anchor_nm shape {self.anchor_nm.shape} != expected (3,)")

    @classmethod
    def from_pair_scene(cls, pair_scene: MergePairScene) -> "FusedMergePairScene":
        return cls(
            views=fuse_merge_pair_views(pair_scene.views),
            anchor_nm=pair_scene.anchor_nm,
            segment_id_a=pair_scene.segment_id_a,
            segment_id_b=pair_scene.segment_id_b,
            species=pair_scene.species,
            view_config=pair_scene.view_config,
        )

    @classmethod
    def from_meshes(
        cls,
        mesh_a,
        mesh_b,
        anchor_nm,
        segment_id_a: int,
        segment_id_b: int,
        species: str = "",
        view_config: Optional[ViewConfig] = None,
        contact_threshold_nm: float = 256.0,
        contact_point_radius_px: float = 1.5,
        skip_contact: bool = True,
        viewer=None,
    ) -> "FusedMergePairScene":
        pair_scene = MergePairScene.from_meshes(
            mesh_a=mesh_a,
            mesh_b=mesh_b,
            anchor_nm=anchor_nm,
            segment_id_a=segment_id_a,
            segment_id_b=segment_id_b,
            species=species,
            view_config=view_config,
            contact_threshold_nm=contact_threshold_nm,
            contact_point_radius_px=contact_point_radius_px,
            skip_contact=skip_contact,
            viewer=viewer,
        )
        return cls.from_pair_scene(pair_scene)
