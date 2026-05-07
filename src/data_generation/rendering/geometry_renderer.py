"""
Geometry channel renderer for foundation model training data.

Renders mesh scenes as [V, 5, H, W] float16 tensors with channels:
  0: silhouette  (binary foreground mask)
  1: depth       (normalized 0-1 over foreground)
  2: normal_x    (surface normal x-component, -1 to 1)
  3: normal_y    (surface normal y-component, -1 to 1)
  4: normal_z    (surface normal z-component, -1 to 1)

Uses a single-pass approach per view: renders with MeshNormalMaterial to get
normals in the color buffer and reads depth from the GPU depth buffer.
Silhouette is derived from depth thresholding.
"""

import logging
import os
import tempfile
from typing import Optional, Sequence

import numpy as np
import pygfx as gfx
from PIL import Image

import cloudvolume.mesh

from rendering.camera_utils import configure_camera_for_view, generate_orthographic_views
from rendering.scene_builder import prepare_mesh_for_rendering
from rendering.render_utils import CameraViewSpec

logger = logging.getLogger(__name__)

# Standard viewing angles: (name, look-axis description)
STANDARD_ANGLES = ("front", "side", "top")

# Channel indices
CH_SILHOUETTE = 0
CH_DEPTH = 1
CH_NORMAL_X = 2
CH_NORMAL_Y = 3
CH_NORMAL_Z = 4
N_CHANNELS = 5


def create_geometry_viewer():
    """Create a reusable offscreen viewer for geometry rendering.

    Stash this per worker process and pass into render_geometry_views to avoid
    re-creating pygfx/wgpu resources per sample (leaks GPU buffers).
    """
    import octarine as oc
    viewer = oc.Viewer(offscreen=True)
    viewer.set_bgcolor((0, 0, 0))
    return viewer


def render_geometry_views(
    mesh: cloudvolume.mesh.Mesh,
    anchor_nm: np.ndarray,
    view_specs: Sequence[CameraViewSpec],
    resolution: int = 224,
    crop_padding_nm: Optional[float] = None,
    viewer=None,
) -> np.ndarray:
    """
    Render multi-channel geometry tensors for a mesh from multiple viewpoints.

    Uses a single offscreen viewer with MeshNormalMaterial. For each view,
    configures the camera, renders, and reads both the color buffer (normals
    encoded as RGB) and the depth buffer from the GPU.

    Args:
        mesh: CloudVolume Mesh object with vertices and faces.
        anchor_nm: Center coordinate [x, y, z] in nanometers for cropping.
        view_specs: Camera view specifications. Each should have ortho_extent_nm
                    set for per-view zoom control.
        resolution: Square output resolution in pixels (default 224).
        crop_padding_nm: Padding around anchor for mesh cropping. If None,
                         uses 1.2x the max ortho_extent_nm across views.

    Returns:
        np.ndarray of shape [V, 5, H, W] float16 where V = len(view_specs).
        Channels: [silhouette, depth, normal_x, normal_y, normal_z].
    """
    import octarine as oc

    anchor = np.asarray(anchor_nm, dtype=float)
    n_views = len(view_specs)

    # Determine crop padding from view extents if not provided
    if crop_padding_nm is None:
        max_extent = max(
            (v.ortho_extent_nm or 0.0 for v in view_specs), default=0.0
        )
        crop_padding_nm = max_extent * 1.2 if max_extent > 0 else 50_000.0

    # Crop mesh to region of interest
    cropped_mesh, bbox = prepare_mesh_for_rendering(
        mesh, anchor, crop_padding_nm, crop=True
    )

    # Handle empty mesh after cropping
    if hasattr(cropped_mesh, "empty") and cropped_mesh.empty():
        logger.warning("Mesh is empty after cropping, returning zero tensor")
        return np.zeros(
            (n_views, N_CHANNELS, resolution, resolution), dtype=np.float16
        )
    if hasattr(cropped_mesh, "vertices") and len(cropped_mesh.vertices) == 0:
        logger.warning("Mesh has no vertices after cropping, returning zero tensor")
        return np.zeros(
            (n_views, N_CHANNELS, resolution, resolution), dtype=np.float16
        )

    owns_viewer = viewer is None
    if owns_viewer:
        viewer = oc.Viewer(offscreen=True)
        viewer.set_bgcolor((0, 0, 0))
    else:
        # reuse path: clear previous scene contents before adding this mesh
        for child in list(viewer.scene.children):
            viewer.scene.remove(child)

    # Add mesh to scene (octarine handles cloudvolume Mesh → pygfx conversion)
    viewer.add(cropped_mesh, color="white")

    # Swap all mesh materials to MeshNormalMaterial for normal-encoded rendering
    _apply_normal_material(viewer)

    # Render each view
    all_channels = []
    for view_spec in view_specs:
        configure_camera_for_view(
            viewer,
            view_spec,
            default_ortho_extent_nm=view_spec.ortho_extent_nm,
        )
        channels = _capture_geometry_channels(viewer, resolution)
        all_channels.append(channels)

    if owns_viewer:
        viewer.close()
    else:
        # leave viewer alive for next call; drop mesh refs now to release GPU buffers
        for child in list(viewer.scene.children):
            viewer.scene.remove(child)

    return np.stack(all_channels).astype(np.float16)


def generate_geometry_view_specs(
    anchor_nm: np.ndarray,
    bbox,
    angles: Sequence[str] = STANDARD_ANGLES,
    extents_nm: Sequence[float] = (10_000,),
) -> list:
    """
    Generate CameraViewSpec objects for geometry rendering.

    Creates specs for all combinations of viewing angles and ortho extents,
    producing len(angles) * len(extents_nm) total views.

    Args:
        anchor_nm: Center coordinate [x, y, z] in nanometers.
        bbox: cloudvolume Bbox defining the scene extent (used to position cameras).
        angles: Which standard angles to include ("front", "side", "top").
        extents_nm: Orthographic extents (half-widths) in nm. Multiple values
                    produce multi-scale views.

    Returns:
        List of CameraViewSpec objects with ortho_extent_nm set.
    """
    base_views = generate_orthographic_views(bbox, center_override=np.asarray(anchor_nm, dtype=float))
    view_map = {v.name: v for v in base_views}

    specs = []
    for angle_name in angles:
        base = view_map.get(angle_name)
        if base is None:
            logger.warning("Unknown angle %r, skipping (available: %s)", angle_name, list(view_map))
            continue
        for extent in extents_nm:
            spec = CameraViewSpec(
                name=f"{angle_name}_{int(extent / 1_000)}um",
                projection=base.projection,
                position_nm=base.position_nm,
                target_nm=base.target_nm,
                up=base.up,
                ortho_extent_nm=float(extent),
            )
            specs.append(spec)

    return specs


def _apply_normal_material(viewer) -> None:
    """Replace all mesh materials with MeshNormalMaterial for normal encoding."""
    for child in viewer.scene.children:
        if isinstance(child, gfx.Mesh):
            child.material = gfx.MeshNormalMaterial(side="both")


def _capture_geometry_channels(viewer, resolution: int) -> np.ndarray:
    """
    Capture [5, H, W] geometry channels from a single render pass.

    Triggers a render via screenshot, reads the color buffer (normals encoded
    as RGB by MeshNormalMaterial) and the depth buffer from the GPU.
    """
    size = (resolution, resolution)

    # Trigger render and get color buffer (normals encoded as RGB)
    rgba = _render_color_buffer(viewer, size)  # [H, W, 4] uint8

    # Read depth buffer from GPU
    depth_raw = None
    try:
        depth_raw = _read_depth_buffer(viewer)
    except Exception as e:
        logger.warning(
            "Failed to read depth buffer from GPU (%s). "
            "Falling back to alpha-channel silhouette with synthetic depth.",
            e,
        )

    h, w = rgba.shape[0], rgba.shape[1]

    # Determine foreground mask and depth
    if depth_raw is not None:
        # Resize depth to match color buffer if needed (pixel_ratio mismatch)
        if depth_raw.shape[0] != h or depth_raw.shape[1] != w:
            depth_raw = _resize_array_2d(depth_raw, w, h)
        # pygfx uses 1.0 for background (far plane)
        foreground = depth_raw < 1.0
    else:
        # Fallback: use alpha channel for silhouette
        foreground = rgba[:, :, 3] > 0
        # Synthesize flat depth for foreground (no real depth info)
        depth_raw = np.where(foreground, 0.5, 1.0).astype(np.float32)

    # --- Silhouette ---
    silhouette = foreground.astype(np.float32)

    # --- Depth normalization ---
    # Normalize to [0, 1] over foreground pixels only
    depth_norm = np.zeros((h, w), dtype=np.float32)
    if foreground.any():
        d_min = depth_raw[foreground].min()
        d_max = depth_raw[foreground].max()
        depth_norm = np.where(
            foreground,
            (depth_raw - d_min) / (d_max - d_min + 1e-8),
            0.0,
        )

    # --- Normals ---
    # MeshNormalMaterial encodes: rgb = normal * 0.5 + 0.5
    # Decode: normal = rgb * 2.0 - 1.0
    rgb = rgba[:, :, :3].astype(np.float32) / 255.0
    normals = rgb * 2.0 - 1.0  # [H, W, 3] in [-1, 1]
    # Zero out background normals (background is black → decodes to (-1,-1,-1))
    normals[~foreground] = 0.0

    # Stack channels: [silhouette, depth, nx, ny, nz]
    channels = np.stack(
        [
            silhouette,
            depth_norm,
            normals[:, :, 0],
            normals[:, :, 1],
            normals[:, :, 2],
        ],
        axis=0,
    )  # [5, H, W]

    # Resize to target resolution if render produced a different size
    if h != resolution or w != resolution:
        channels = _resize_channels(channels, resolution)

    return channels


def _render_color_buffer(viewer, size: tuple) -> np.ndarray:
    """
    Trigger a render and return the color buffer as [H, W, 4] uint8 RGBA.

    Uses viewer.screenshot() to a temp file, which triggers the full render
    pipeline (including blender state for subsequent depth buffer reads).
    """
    fd, tmp_path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    try:
        viewer.screenshot(tmp_path, alpha=True, size=size)
        with Image.open(tmp_path) as im:
            rgba = np.array(im.convert("RGBA"))
    finally:
        os.unlink(tmp_path)
    return rgba


def _read_depth_buffer(viewer) -> np.ndarray:
    """
    Read the depth buffer from the pygfx renderer's internal blender.

    Returns [H, W] float32 array with values in [0, 1].
    Background pixels have depth = 1.0 (far plane).

    Uses private pygfx API (renderer._blender, renderer._device) which
    mirrors how pygfx reads GPU textures internally in its snapshot() method.
    """
    renderer = viewer.renderer
    blender = renderer._blender
    depth_texture = blender.get_texture("depth")
    tex_size = depth_texture.size  # (width, height, depth_or_layers)
    device = renderer._device

    width, height = tex_size[0], tex_size[1]
    bytes_per_pixel = 4  # depth32float = 4 bytes per pixel
    bytes_per_row = width * bytes_per_pixel

    data = device.queue.read_texture(
        {
            "texture": depth_texture,
            "mip_level": 0,
            "origin": (0, 0, 0),
        },
        {
            "offset": 0,
            "bytes_per_row": bytes_per_row,
            "rows_per_image": height,
        },
        tex_size,
    )

    # Parse depth data, handling potential row padding from GPU alignment
    expected_size = width * height * bytes_per_pixel
    actual_size = len(data)

    if actual_size == expected_size:
        depth = np.frombuffer(data, dtype=np.float32).reshape(height, width)
    else:
        # Row padding present — wgpu may align rows to 256 bytes
        aligned_bpr = actual_size // height
        row_floats = aligned_bpr // bytes_per_pixel
        raw = np.frombuffer(data, dtype=np.float32).reshape(height, row_floats)
        depth = raw[:, :width].copy()

    return depth


def _resize_array_2d(arr: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Resize a 2D float array using nearest-neighbor interpolation."""
    img = Image.fromarray(arr.astype(np.float32), mode="F")
    img = img.resize((target_w, target_h), Image.NEAREST)
    return np.array(img)


def _resize_channels(channels: np.ndarray, resolution: int) -> np.ndarray:
    """Resize [C, H, W] channels to [C, resolution, resolution]."""
    c = channels.shape[0]
    out = np.empty((c, resolution, resolution), dtype=channels.dtype)
    for i in range(c):
        out[i] = _resize_array_2d(channels[i], resolution, resolution)
    return out


# ---------------------------------------------------------------------------
# Contact vertex computation and rendering for merge-pair scenes
# ---------------------------------------------------------------------------


def compute_contact_vertices(
    mesh_a: cloudvolume.mesh.Mesh,
    mesh_b: cloudvolume.mesh.Mesh,
    threshold_nm: float = 256.0,
    bbox=None,
) -> np.ndarray:
    """
    Find contact points between two meshes using vertex proximity.

    Builds a KD-tree on mesh B vertices and queries nearest-neighbor distances
    for each mesh A vertex. Returns midpoints of vertex pairs within threshold.

    Args:
        mesh_a: First mesh (CloudVolume Mesh).
        mesh_b: Second mesh (CloudVolume Mesh).
        threshold_nm: Maximum distance in nm to count as contact.
        bbox: Optional cloudvolume Bbox to crop vertices before computing.
              Only vertices inside the bbox are considered.

    Returns:
        np.ndarray of shape [N, 3] — midpoints of contacting vertex pairs.
        Returns empty [0, 3] array if no contacts found.
    """
    from scipy.spatial import cKDTree

    verts_a = np.asarray(mesh_a.vertices, dtype=np.float64)
    verts_b = np.asarray(mesh_b.vertices, dtype=np.float64)

    # Crop to bounding box if provided
    if bbox is not None:
        bbox_min = np.asarray(bbox.minpt, dtype=np.float64)
        bbox_max = np.asarray(bbox.maxpt, dtype=np.float64)
        mask_a = np.all((verts_a >= bbox_min) & (verts_a <= bbox_max), axis=1)
        mask_b = np.all((verts_b >= bbox_min) & (verts_b <= bbox_max), axis=1)
        verts_a = verts_a[mask_a]
        verts_b = verts_b[mask_b]

    if len(verts_a) == 0 or len(verts_b) == 0:
        return np.empty((0, 3), dtype=np.float64)

    tree_b = cKDTree(verts_b)
    dists, indices = tree_b.query(verts_a)

    contact_mask = dists <= threshold_nm
    if not np.any(contact_mask):
        return np.empty((0, 3), dtype=np.float64)

    contact_a = verts_a[contact_mask]
    contact_b = verts_b[indices[contact_mask]]
    contact_points = (contact_a + contact_b) / 2.0

    logger.info(
        "Found %d contact points (threshold=%.0f nm)", len(contact_points), threshold_nm
    )
    return contact_points


def render_contact_mask(
    contact_vertices: np.ndarray,
    view_specs: Sequence[CameraViewSpec],
    resolution: int = 224,
    point_radius_px: float = 1.5,
) -> np.ndarray:
    """
    Render contact points as a binary mask for each view via projection.

    Projects contact vertices into 2D using the orthographic camera parameters
    from each view spec, then rasterizes them as filled circles. No GPU render
    pass needed — pure matrix math.

    Args:
        contact_vertices: [N, 3] array of contact point positions in nm.
        view_specs: Camera view specifications (same as geometry rendering).
        resolution: Square output resolution in pixels.
        point_radius_px: Radius of each contact point in pixels.

    Returns:
        np.ndarray of shape [V, H, W] float32 binary mask (0 or 1).
    """
    n_views = len(view_specs)

    if len(contact_vertices) == 0:
        return np.zeros((n_views, resolution, resolution), dtype=np.float32)

    pts = np.asarray(contact_vertices, dtype=np.float64)

    masks = []
    for view_spec in view_specs:
        mask = _project_points_to_mask(
            pts, view_spec, resolution, point_radius_px
        )
        masks.append(mask)

    return np.stack(masks)


def _project_points_to_mask(
    points: np.ndarray,
    view: CameraViewSpec,
    resolution: int,
    radius_px: float,
) -> np.ndarray:
    """
    Project 3D points onto a 2D mask using orthographic camera parameters.

    For an orthographic camera:
      1. Translate points so camera target is at origin
      2. Rotate into camera frame (look direction = -Z, up = Y, right = X)
      3. Project by dropping the Z component
      4. Scale from world nm to pixel coordinates using ortho_extent_nm
      5. Rasterize each point as a filled circle

    Returns [H, W] float32 binary mask.
    """
    cam_pos = np.asarray(view.position_nm, dtype=np.float64)
    cam_target = np.asarray(view.target_nm, dtype=np.float64)
    cam_up = np.asarray(view.up, dtype=np.float64)

    # Camera basis vectors
    forward = cam_target - cam_pos
    forward = forward / (np.linalg.norm(forward) + 1e-12)
    right = np.cross(forward, cam_up)
    right = right / (np.linalg.norm(right) + 1e-12)
    up = np.cross(right, forward)
    up = up / (np.linalg.norm(up) + 1e-12)

    # Project: translate to target-centered, then dot with right/up
    centered = points - cam_target  # [N, 3]
    x_world = centered @ right      # [N] along screen-X axis
    y_world = centered @ up          # [N] along screen-Y axis

    # World → pixel: ortho_extent_nm is half-width of the view
    extent = view.ortho_extent_nm or 10_000.0
    half_res = resolution / 2.0
    scale = half_res / extent  # pixels per nm

    x_px = x_world * scale + half_res
    y_px = -y_world * scale + half_res  # flip Y (image coords)

    # Rasterize: round to nearest pixel and scatter into mask
    mask = np.zeros((resolution, resolution), dtype=np.float32)

    # Integer pixel coordinates (nearest-neighbor)
    xi = np.round(x_px).astype(np.intp)
    yi = np.round(y_px).astype(np.intp)

    # Filter to points within image bounds
    in_bounds = (xi >= 0) & (xi < resolution) & (yi >= 0) & (yi < resolution)
    xi = xi[in_bounds]
    yi = yi[in_bounds]

    if len(xi) == 0:
        return mask

    # Splat single-pixel points (vectorized)
    mask[yi, xi] = 1.0

    # Dilate if radius > 1 to produce thicker contact marks
    if radius_px > 1.0:
        from scipy.ndimage import binary_dilation

        r = int(np.ceil(radius_px))
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
        kernel = (xx ** 2 + yy ** 2 <= radius_px ** 2).astype(np.uint8)
        mask = binary_dilation(mask > 0, structure=kernel).astype(np.float32)

    return mask
