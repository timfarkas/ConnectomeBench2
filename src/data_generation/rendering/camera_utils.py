"""
Camera positioning and view generation utilities.

Functions for creating camera specifications and configuring viewer cameras
for different orthographic and perspective projections.
"""

from typing import Any, Optional, Tuple

import numpy as np
import pygfx as gfx
from cloudvolume import Bbox

from rendering.render_utils import CameraViewSpec



def _is_perspective_camera(camera: Any) -> bool:
    """Check if camera is a perspective camera (pygfx or Kaolin)."""
    class_name = type(camera).__name__
    return class_name in ('PerspectiveCamera', 'KaolinPerspectiveCamera')


def _is_orthographic_camera(camera: Any) -> bool:
    """Check if camera is an orthographic camera (pygfx or Kaolin)."""
    class_name = type(camera).__name__
    return class_name in ('OrthographicCamera', 'KaolinOrthographicCamera')


def _create_perspective_camera(viewer: Any) -> Any:
    """Create appropriate perspective camera for viewer type."""
    viewer_type = type(viewer).__name__
    if viewer_type == 'KaolinViewer':
        from rendering.kaolin_backend import KaolinPerspectiveCamera
        return KaolinPerspectiveCamera()
    else:
        return gfx.PerspectiveCamera()


def _create_orthographic_camera(viewer: Any) -> Any:
    """Create appropriate orthographic camera for viewer type."""
    viewer_type = type(viewer).__name__
    if viewer_type == 'KaolinViewer':
        from rendering.kaolin_backend import KaolinOrthographicCamera
        return KaolinOrthographicCamera()
    else:
        return gfx.OrthographicCamera()


def generate_orthographic_views(
    bbox: Bbox,
    center_override: Optional[np.ndarray] = None,
    margin: float = 0.15,
) -> Tuple[CameraViewSpec, CameraViewSpec, CameraViewSpec]:
    """
    Generate front, side, and top orthographic camera views for a bounding box.

    Args:
        bbox: Bounding box defining the scene extent
        center_override: Optional center point (overrides bbox center)
        margin: Fraction of bbox size to add as padding (default 0.15 = 15%)

    Returns:
        Tuple of (front_view, side_view, top_view) camera specifications

    Notes:
        - Front view: camera looks along +X axis
        - Side view: camera looks along +Y axis
        - Top view: camera looks down -Z axis with Y as up
        - Camera positions are placed outside bbox with margin
    """
    minpt = np.asarray(bbox.minpt, dtype=float)
    maxpt = np.asarray(bbox.maxpt, dtype=float)
    center = (minpt + maxpt) / 2.0

    if center_override is not None:
        center = np.asarray(center_override, dtype=float)

    sizes = np.maximum(maxpt - minpt, 1.0)
    half_sizes = sizes / 2.0
    extents = half_sizes * (1.0 + margin)

    # Position cameras outside the bounding box
    front_pos = center.copy()
    front_pos[0] -= extents[0]

    side_pos = center.copy()
    side_pos[1] -= extents[1]

    top_pos = center.copy()
    top_pos[2] += extents[2]

    front_view = CameraViewSpec(
        name="front",
        projection="orthographic",
        position_nm=tuple(front_pos),
        target_nm=tuple(center),
    )

    side_view = CameraViewSpec(
        name="side",
        projection="orthographic",
        position_nm=tuple(side_pos),
        target_nm=tuple(center),
    )

    top_view = CameraViewSpec(
        name="top",
        projection="orthographic",
        position_nm=tuple(top_pos),
        target_nm=tuple(center),
        up=(0.0, 1.0, 0.0),  # Y-axis as up for top view
    )

    return front_view, side_view, top_view


def configure_camera_for_view(
    viewer: Any,
    view: CameraViewSpec,
    default_ortho_extent_nm: Optional[float] = None,
    default_zoom: Optional[float] = None,
) -> None:
    """
    Configure viewer camera based on camera view specification.

    Args:
        viewer: Octarine Viewer instance with camera to configure
        view: Camera view specification with projection and parameters
        default_ortho_extent_nm: Default orthographic extent if not in view
        default_zoom: Default zoom level if not in view

    Notes:
        - Switches camera type between Orthographic and Perspective as needed
        - For orthographic: sets width/height from ortho_extent_nm or zoom/scale
        - For perspective: sets field of view
        - Applies clip range if specified
        - Positions camera and sets look-at target
    """
    if view.projection == "perspective":
        # Configure perspective camera
        if not _is_perspective_camera(viewer.camera):
            viewer.camera = _create_perspective_camera(viewer)

        if view.fov_degrees is not None:
            viewer.camera.fov = view.fov_degrees
        elif viewer.camera.fov == 0:
            viewer.camera.fov = 45.0

    else:
        # Configure orthographic camera
        if not _is_orthographic_camera(viewer.camera):
            viewer.camera = _create_orthographic_camera(viewer)

        viewer.camera.fov = 0.0

        # Set orthographic extent (view width/height)
        if view.ortho_extent_nm is not None and hasattr(viewer.camera, "width"):
            extent = float(view.ortho_extent_nm)
            viewer.camera.width = extent * 2.0
            viewer.camera.height = extent * 2.0
        elif default_ortho_extent_nm is not None and hasattr(viewer.camera, "width"):
            extent = float(default_ortho_extent_nm)
            viewer.camera.width = extent * 2.0
            viewer.camera.height = extent * 2.0
        elif view.zoom is not None and hasattr(viewer.camera, "scale"):
            viewer.camera.scale = view.zoom
        elif default_zoom is not None and hasattr(viewer.camera, "scale"):
            viewer.camera.scale = default_zoom

    # Set near/far clipping planes if specified
    if view.clip_range_nm:
        near, far = view.clip_range_nm
        viewer.camera.near = near
        viewer.camera.far = far

    # Position camera and set target
    viewer.camera.local.position = view.position_nm

    try:
        viewer.camera.world.reference_up = np.asarray(view.up, dtype=float)
    except Exception:
        pass

    viewer.camera.look_at(view.target_nm)


def compute_world_dimensions(
    viewer: Any,
    view: CameraViewSpec,
    canvas_width_px: int,
    default_ortho_extent_nm: Optional[float] = None,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Compute world space dimensions for an orthographic camera view.

    Args:
        viewer: Octarine Viewer instance
        view: Camera view specification
        canvas_width_px: Canvas width in pixels
        default_ortho_extent_nm: Fallback extent if not in camera or view

    Returns:
        Tuple of (world_width_nm, nm_per_px) or (None, None) if not orthographic

    Notes:
        - Only applicable for orthographic cameras
        - Used for scale bar rendering and spatial measurements
        - Tries to get width from camera.width, then view, then default
    """
    if not _is_orthographic_camera(viewer.camera):
        return None, None

    width_nm_value = getattr(viewer.camera, "width", None)

    if width_nm_value is None:
        if view.ortho_extent_nm is not None:
            width_nm_value = float(view.ortho_extent_nm) * 2.0
        elif default_ortho_extent_nm is not None:
            width_nm_value = float(default_ortho_extent_nm) * 2.0

    if width_nm_value:
        world_width_nm = float(width_nm_value)
        nm_per_px = world_width_nm / float(canvas_width_px)
        return world_width_nm, nm_per_px

    return None, None
