"""
MeshScene: canonical data object for foundation model mesh inputs.

A MeshScene holds the multi-view, multi-scale geometry tensor for a single
segment at a single anchor, plus the metadata needed to reconstruct how it
was rendered and store it in HDF5.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# Defaults matching the foundation model plan (Section 1.2)
DEFAULT_ANGLES = ("front", "side", "top")
DEFAULT_EXTENTS_NM = (4_000.0, 16_000.0, 64_000.0)
DEFAULT_RESOLUTION = 224
CHANNEL_NAMES = ("silhouette", "depth", "nx", "ny", "nz", "a_mask", "b_mask")
N_CHANNELS = 7


def _augment_single_segment_views(views: np.ndarray) -> np.ndarray:
    """Extend [V,5,H,W] geometry views with explicit segment identity masks."""
    if views.shape[1] != 5:
        raise ValueError(f"expected [V,5,H,W], got {views.shape}")
    a_mask = np.clip(views[:, 0:1], 0.0, 1.0).astype(np.float16)
    b_mask = np.zeros_like(a_mask, dtype=np.float16)
    return np.concatenate([views.astype(np.float16), a_mask, b_mask], axis=1)


@dataclass
class ViewConfig:
    """Describes the view grid that produced a MeshScene's tensor."""

    angles: tuple[str, ...] = DEFAULT_ANGLES
    extents_nm: tuple[float, ...] = DEFAULT_EXTENTS_NM
    resolution: int = DEFAULT_RESOLUTION

    @property
    def n_views(self) -> int:
        return len(self.angles) * len(self.extents_nm)

    @property
    def view_names(self) -> list[str]:
        """Ordered list of view names matching tensor row ordering.

        Ordering: group by angle, vary extent within each angle.
        e.g. front_4um, front_16um, front_64um, side_4um, ...
        """
        return [
            f"{angle}_{int(extent / 1_000)}um"
            for angle in self.angles
            for extent in self.extents_nm
        ]

    @property
    def view_angles(self) -> list[str]:
        """Angle label per view, same order as view_names."""
        return [
            angle
            for angle in self.angles
            for _extent in self.extents_nm
        ]

    @property
    def view_extents_um(self) -> list[float]:
        """Extent in micrometers per view, same order as view_names."""
        return [
            extent / 1_000.0
            for _angle in self.angles
            for extent in self.extents_nm
        ]


@dataclass
class MeshScene:
    """
    One training example for the mesh modality.

    Attributes:
        views: [V, C, H, W] float16 geometry tensor.
               V = n_views, C = 7 (silhouette, depth, nx, ny, nz, a_mask, b_mask).
        anchor_nm: [3] float64 anchor coordinate in nanometers.
        segment_id: Integer segment / root ID.
        species: Species identifier string (e.g. "mouse", "fly").
        view_config: ViewConfig describing angle × extent grid.
    """

    views: np.ndarray  # [V, C, H, W] float16
    anchor_nm: np.ndarray  # [3] float64
    segment_id: int
    species: str = ""
    view_config: ViewConfig = field(default_factory=ViewConfig)

    def __post_init__(self):
        self.anchor_nm = np.asarray(self.anchor_nm, dtype=np.float64)
        if self.views.dtype != np.float16:
            self.views = self.views.astype(np.float16)

    @property
    def n_views(self) -> int:
        return self.views.shape[0]

    @property
    def resolution(self) -> int:
        return self.views.shape[-1]

    def validate(self) -> None:
        """Raise ValueError if tensor shapes don't match view_config."""
        vc = self.view_config
        expected = (vc.n_views, N_CHANNELS, vc.resolution, vc.resolution)
        if self.views.shape != expected:
            raise ValueError(
                f"views shape {self.views.shape} != expected {expected}"
            )
        if self.anchor_nm.shape != (3,):
            raise ValueError(
                f"anchor_nm shape {self.anchor_nm.shape} != expected (3,)"
            )

    @classmethod
    def from_mesh(
        cls,
        mesh,
        anchor_nm,
        segment_id: int,
        species: str = "",
        view_config: Optional[ViewConfig] = None,
        viewer=None,
    ) -> MeshScene:
        """
        Render a MeshScene from a cloudvolume Mesh object.

        Args:
            mesh: cloudvolume.mesh.Mesh with vertices and faces.
            anchor_nm: [3] anchor coordinate in nanometers.
            segment_id: Segment / root ID.
            species: Species identifier string.
            view_config: ViewConfig for angles/extents/resolution.
                         Uses defaults (3 angles × 3 zooms, 224px) if None.
            viewer: Optional reusable offscreen viewer (from
                    rendering.geometry_renderer.create_geometry_viewer).
        """
        from rendering.geometry_renderer import (
            generate_geometry_view_specs,
            render_geometry_views,
        )
        from cloudvolume import Bbox

        if view_config is None:
            view_config = ViewConfig()

        anchor = np.asarray(anchor_nm, dtype=np.float64)

        max_extent = max(view_config.extents_nm)
        # [dataImprovement deviation from em-foundation-multimodal, 2026-04-20]
        # Anchor-centered synthetic bbox (rather than mesh-derived via
        # prepare_mesh_for_rendering) so camera framing is purely
        # (anchor, extent)-driven and deterministic across A/B pairs.
        # Needed for LICONN eval consistency (FusedMergePairScene renders A
        # and B separately — both must share identical framing). Orthographic
        # projection → camera distance doesn't affect the image, only near/far
        # clipping, so half = 2 * max_extent gives plenty of z-headroom.
        # KEEP this when merging back into em-foundation-multimodal.
        half = max_extent * 2.0
        bbox = Bbox(anchor - half, anchor + half)

        view_specs = generate_geometry_view_specs(
            anchor_nm=anchor,
            bbox=bbox,
            angles=view_config.angles,
            extents_nm=view_config.extents_nm,
        )

        views = render_geometry_views(
            mesh=mesh,
            anchor_nm=anchor,
            view_specs=view_specs,
            resolution=view_config.resolution,
            crop_padding_nm=max_extent * 1.2,
            viewer=viewer,
        )
        views = _augment_single_segment_views(views)

        return cls(
            views=views,
            anchor_nm=anchor,
            segment_id=segment_id,
            species=species,
            view_config=view_config,
        )

    @classmethod
    def from_merge_meshes(
        cls,
        mesh_a,
        mesh_b,
        anchor_nm,
        segment_id_a: int,
        segment_id_b: int,
        species: str = "",
        view_config: Optional[ViewConfig] = None,
        viewer=None,
    ) -> MeshScene:
        """Render a fused pair scene into the canonical 7-channel mesh format."""
        from foundation.scenes.fused_merge_pair_scene import FusedMergePairScene

        fused = FusedMergePairScene.from_meshes(
            mesh_a=mesh_a,
            mesh_b=mesh_b,
            anchor_nm=anchor_nm,
            segment_id_a=segment_id_a,
            segment_id_b=segment_id_b,
            species=species,
            view_config=view_config,
            viewer=viewer,
        )
        return cls(
            views=fused.views,
            anchor_nm=fused.anchor_nm,
            segment_id=segment_id_a,
            species=fused.species,
            view_config=fused.view_config,
        )
