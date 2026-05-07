"""
MergePairScene: data object for segment-pair merge/contact tasks.

Extends the single-segment MeshScene with a second segment and a contact
channel. Used for merge correction (scoring candidate cut planes between
two segments).

Channel layout per view [V, 11, H, W]:
    0-4:  Segment A geometry  (silhouette, depth, nx, ny, nz)
    5-9:  Segment B geometry  (silhouette, depth, nx, ny, nz)
    10:   Contact mask        (binary: 1 where A and B are close)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from foundation.scenes.mesh_scene import (
    DEFAULT_ANGLES,
    DEFAULT_EXTENTS_NM,
    DEFAULT_RESOLUTION,
    ViewConfig,
)

logger = logging.getLogger(__name__)

N_CHANNELS_MERGE = 11
MERGE_CHANNEL_NAMES = (
    "a_silhouette", "a_depth", "a_nx", "a_ny", "a_nz",
    "b_silhouette", "b_depth", "b_nx", "b_ny", "b_nz",
    "contact",
)


@dataclass
class MergePairScene:
    """
    One training example for the merge-pair modality.

    Attributes:
        views: [V, 11, H, W] float16 geometry tensor.
               Channels 0-4: segment A, 5-9: segment B, 10: contact.
        anchor_nm: [3] float64 interface point between A and B in nanometers.
        segment_id_a: Integer segment / root ID for segment A.
        segment_id_b: Integer segment / root ID for segment B.
        species: Species identifier string (e.g. "mouse", "fly").
        view_config: ViewConfig describing angle x extent grid.
    """

    views: np.ndarray       # [V, 11, H, W] float16
    anchor_nm: np.ndarray   # [3] float64
    segment_id_a: int
    segment_id_b: int
    species: str = ""
    view_config: ViewConfig = field(default_factory=ViewConfig)

    N_CHANNELS = N_CHANNELS_MERGE
    CHANNEL_NAMES = MERGE_CHANNEL_NAMES

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
        expected = (vc.n_views, N_CHANNELS_MERGE, vc.resolution, vc.resolution)
        if self.views.shape != expected:
            raise ValueError(
                f"views shape {self.views.shape} != expected {expected}"
            )
        if self.anchor_nm.shape != (3,):
            raise ValueError(
                f"anchor_nm shape {self.anchor_nm.shape} != expected (3,)"
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
        skip_contact: bool = False,
        viewer=None,
    ) -> MergePairScene:
        """
        Render a MergePairScene from two cloudvolume Mesh objects.

        Args:
            mesh_a: cloudvolume.mesh.Mesh for segment A.
            mesh_b: cloudvolume.mesh.Mesh for segment B.
            anchor_nm: [3] interface point in nanometers (center of view).
            segment_id_a: Segment / root ID for A.
            segment_id_b: Segment / root ID for B.
            species: Species identifier string.
            view_config: ViewConfig for angles/extents/resolution.
                         Uses defaults (3 angles x 3 zooms, 224px) if None.
            contact_threshold_nm: Max distance in nm for contact detection.
            contact_point_radius_px: Radius of contact point markers in pixels.
        """
        from rendering.geometry_renderer import (
            compute_contact_vertices,
            generate_geometry_view_specs,
            render_contact_mask,
            render_geometry_views,
        )
        from cloudvolume import Bbox

        if view_config is None:
            view_config = ViewConfig()

        anchor = np.asarray(anchor_nm, dtype=np.float64)
        max_extent = max(view_config.extents_nm)
        crop_padding_nm = max_extent * 1.2

        # [dataImprovement deviation from em-foundation-multimodal, 2026-04-20]
        # Anchor-centered synthetic bbox (rather than mesh_a-derived via
        # prepare_mesh_for_rendering). Ensures segments A and B are framed
        # identically across every view — critical for LICONN eval since the
        # fuse step assumes shared framing. Orthographic projection means
        # camera distance doesn't affect the rendered image, only near/far
        # clipping, so half = 2 * max_extent gives ample z-headroom.
        # KEEP this when merging back into em-foundation-multimodal.
        half = max_extent * 2.0
        bbox = Bbox(anchor - half, anchor + half)

        # Generate camera specs for all angle x extent combinations
        view_specs = generate_geometry_view_specs(
            anchor_nm=anchor,
            bbox=bbox,
            angles=view_config.angles,
            extents_nm=view_config.extents_nm,
        )

        # 1. Render segment A geometry → [V, 5, H, W]
        logger.debug("Rendering segment A geometry (%d views)", len(view_specs))
        views_a = render_geometry_views(
            mesh=mesh_a,
            anchor_nm=anchor,
            view_specs=view_specs,
            resolution=view_config.resolution,
            crop_padding_nm=crop_padding_nm,
            viewer=viewer,
        )

        # 2. Render segment B geometry → [V, 5, H, W]
        logger.debug("Rendering segment B geometry (%d views)", len(view_specs))
        views_b = render_geometry_views(
            mesh=mesh_b,
            anchor_nm=anchor,
            view_specs=view_specs,
            resolution=view_config.resolution,
            crop_padding_nm=crop_padding_nm,
            viewer=viewer,
        )

        # 3+4. Contact channel — skipped when only the fused 7ch output is
        # needed (contact is discarded in FusedMergePairScene.from_pair_scene).
        if skip_contact:
            contact_ch = np.zeros(
                (len(view_specs), 1, view_config.resolution, view_config.resolution),
                dtype=views_a.dtype,
            )
        else:
            contact_verts = compute_contact_vertices(
                mesh_a, mesh_b,
                threshold_nm=contact_threshold_nm,
                bbox=bbox,
            )
            logger.debug(
                "Rendering contact mask (%d contact points)", len(contact_verts)
            )
            contact_masks = render_contact_mask(
                contact_vertices=contact_verts,
                view_specs=view_specs,
                resolution=view_config.resolution,
                point_radius_px=contact_point_radius_px,
            )
            contact_ch = contact_masks[:, np.newaxis, :, :]  # [V, 1, H, W]

        # 5. Concatenate → [V, 11, H, W]
        views = np.concatenate([views_a, views_b, contact_ch], axis=1)

        return cls(
            views=views,
            anchor_nm=anchor,
            segment_id_a=segment_id_a,
            segment_id_b=segment_id_b,
            species=species,
            view_config=view_config,
        )
