"""
Scene construction functions for adding meshes, graphs, and highlights to the viewer.

These functions handle the low-level details of adding visual elements to an
octarine Viewer, including material properties, transparency, and depth testing.
"""

from typing import Any, List, MutableSequence, Optional, Sequence

import numpy as np
import pygfx as gfx
from cloudvolume import Bbox

import cloudvolume 
from cloudvolume.mesh import Mesh, Bbox
import copy 

try:
    import networkx as nx
except ImportError:
    nx = None

from rendering.render_utils import (
    MeshSpec,
    NeuronGraphSpec,
    HighlightsSpec,
    RendererOptions,
    MarkerStyle,
    FONT_FAMILY,
)
from octarine.visuals import text2gfx

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.profiler import profiled


def _world_to_ndc(points: np.ndarray, camera: gfx.Camera) -> np.ndarray:
    """Project world coordinates into normalized device coordinates."""
    pts = np.asarray(points, dtype=float)
    if pts.ndim == 1:
        pts = pts.reshape(1, 3)
    if pts.shape[1] != 3:
        raise ValueError("points must have shape (..., 3)")

    ones = np.ones((pts.shape[0], 1), dtype=float)
    hom = np.concatenate([pts, ones], axis=1)
    clip = hom @ camera.camera_matrix.T
    w = clip[:, 3:4]

    ndc = np.full((pts.shape[0], 3), np.nan, dtype=float)
    mask = np.abs(w) > np.finfo(float).eps
    if np.any(mask):
        ndc[mask[:, 0]] = clip[mask[:, 0], :3] / w[mask[:, 0]]
    return ndc


@profiled("add_meshes")
def add_meshes_to_viewer(
    viewer: Any,
    mesh_specs: Sequence[MeshSpec],
    options: RendererOptions,
) -> None:
    """
    Add mesh objects to the viewer with specified colors and opacity.

    Args:
        viewer: Octarine Viewer instance
        mesh_specs: Sequence of mesh specifications to add
        options: Rendering options for mesh transparency settings

    Notes:
        - Meshes can be cropped to a bounding box if spec.bbox is set
        - Transparency and depth_write are configured based on opacity
        - Invisible meshes (visible=False) are skipped
    """
    import os as _os_m, time as _time_m
    _mv = _os_m.environ.get("RENDER_VERBOSE", "0") == "1"
    _m0 = _time_m.monotonic()
    def _mlog(msg):
        if _mv:
            print(f"      [add_mesh +{_time_m.monotonic() - _m0:.3f}s] {msg}", flush=True)

    for si, spec in enumerate(mesh_specs):
        if not spec.visible or spec.mesh is None:
            continue

        mesh_obj = spec.mesh
        mesh_to_add = mesh_obj

        # Crop mesh to bounding box if specified
        if spec.bbox is not None and hasattr(mesh_obj, "crop"):
            _mlog(f"spec {si}: cropping (bbox set)")
            try:
                cropped = mesh_obj.crop(spec.bbox)
                if cropped is not None:
                    mesh_to_add = cropped
            except Exception:
                # If cropping fails, use original mesh
                mesh_to_add = mesh_obj
            _mlog(f"spec {si}: crop done")

        # Skip empty meshes - octarine crashes when trying to add them
        if hasattr(mesh_to_add, 'empty') and mesh_to_add.empty():
            continue
        if hasattr(mesh_to_add, 'vertices') and len(mesh_to_add.vertices) == 0:
            continue

        nv = len(mesh_to_add.vertices) if hasattr(mesh_to_add, 'vertices') else '?'
        _mlog(f"spec {si}: viewer.add ({nv} verts)")
        # Add mesh and configure material properties
        before_children = len(viewer.scene.children)
        viewer.add(mesh_to_add, color=spec.color)
        _mlog(f"spec {si}: viewer.add done")
        new_children = viewer.scene.children[before_children:]

        for child in new_children:
            if isinstance(child, gfx.Mesh):
                mat = child.material
                try:
                    current_color = mat.color
                    mat.color = (current_color.r, current_color.g, current_color.b, spec.opacity)
                except AttributeError:
                    pass
                mat.opacity = spec.opacity
                mat.transparent = spec.opacity < 1.0 or options.mesh_transparency
                # Only disable depth_write for actually transparent meshes
                # This ensures proper depth sorting for opaque meshes
                mat.depth_write = not mat.transparent
                assert (spec.opacity == 1 and not options.mesh_transparency) or (spec.opacity < 1.0 and options.mesh_transparency), f"Opacity {spec.opacity} and mesh_transparency {options.mesh_transparency} are not consistent"
        _mlog(f"spec {si}: material setup done")


@profiled("add_graph")
def add_graph_to_viewer(
    viewer: Any,
    graph_spec: Optional[NeuronGraphSpec],
    scene_center: np.ndarray,
    options: RendererOptions,
    bbox: Optional[Bbox] = None,
) -> None:
    """
    Add skeleton graph overlay as line segments to the viewer.

    Args:
        viewer: Octarine Viewer instance
        graph_spec: Graph specification with networkx graph and rendering params
        scene_center: Center point of scene for distance-based pruning
        options: Rendering options for line depth testing
        bbox: Optional bounding box to crop graph segments (should match mesh crop)

    Notes:
        - Edges can have 'polyline' data for detailed geometry
        - Falls back to straight lines between node coords if no polyline
        - Segments beyond prune_radius_nm from scene_center are excluded
        - Segments outside bbox are cropped to match mesh boundaries
        - Line opacity and depth testing are configurable
    """
    if graph_spec is None or graph_spec.graph is None or nx is None:
        return

    segments: MutableSequence[np.ndarray] = []
    graph = graph_spec.graph
    radius_sq = None

    if graph_spec.prune_radius_nm is not None:
        radius_sq = float(graph_spec.prune_radius_nm) ** 2

    # Get bbox bounds for cropping
    bbox_min = None
    bbox_max = None
    if bbox is not None:
        bbox_min = np.asarray(bbox.minpt, dtype=float)
        bbox_max = np.asarray(bbox.maxpt, dtype=float)

    # Extract polyline segments from graph edges
    for u, v, data in graph.edges(data=True):
        poly = data.get("polyline")

        if poly is None:
            # No polyline data - use straight line between nodes
            coords = []
            for node_id in (u, v):
                coord = graph.nodes.get(node_id, {}).get("coord")
                if coord is None:
                    break
                coords.append(np.asarray(coord, dtype=float))
            if len(coords) != 2:
                continue
            poly_arr = np.vstack(coords)
        else:
            poly_arr = np.asarray(poly, dtype=float)

        if poly_arr.ndim != 2 or poly_arr.shape[1] != 3:
            continue

        # Prune segments that are too far from scene center
        if radius_sq is not None:
            deltas = poly_arr - scene_center
            if np.all(np.einsum("ij,ij->i", deltas, deltas) > radius_sq):
                continue

        # Crop segments to bounding box if specified
        if bbox_min is not None and bbox_max is not None:
            # Check if any part of the segment is within the bbox
            in_bbox = np.all((poly_arr >= bbox_min) & (poly_arr <= bbox_max), axis=1)
            if not np.any(in_bbox):
                # Entire segment is outside bbox, skip it
                continue

            # If segment has points both inside and outside, we still include it
            # This ensures continuity at bbox boundaries
            # More sophisticated clipping could be added here if needed

        segments.append(poly_arr)

    # Add line segments to viewer
    if segments:
        before_children = len(viewer.scene.children)
        viewer.add_lines(
            segments,
            color=graph_spec.color,
            linewidth=graph_spec.linewidth,
            linewidth_space="screen",
            center=False,
        )

        # Configure line material properties
        line_obj = None
        for child in viewer.scene.children[before_children:]:
            if isinstance(child, gfx.Line):
                line_obj = child
                break

        if line_obj is not None:
            material = line_obj.material
            if hasattr(material, "opacity"):
                material.opacity = graph_spec.opacity
            if not options.line_depth_test:
                material.depth_test = False
            material.depth_write = False

def _is_point_in_bbox(position_nm: np.ndarray, bbox: Bbox) -> bool:
    position = np.asarray(position_nm, dtype=float)
    return np.all((position >= bbox.minpt) & (position <= bbox.maxpt))

@profiled("add_highlights")
def add_highlights_to_viewer(
    viewer: Any,
    highlights_spec: Optional[HighlightsSpec],
    marker_styles: dict[str, MarkerStyle],
    options: RendererOptions,
    bbox: Optional[Bbox] = None,
) -> None:
    """
    Add highlight markers and text labels to the viewer.

    Args:
        viewer: Octarine Viewer instance
        highlights_spec: Specification of highlight points with positions and labels
        marker_styles: Dictionary mapping highlight kinds to marker styles
        options: Rendering options for marker depth testing

    Notes:
        - Each highlight has a position, kind, and optional label
        - Marker style is determined by the highlight kind
        - Labels can have outlines for better visibility
        - Labels are positioned with an offset in the Z direction
    """
    if highlights_spec is None:
        return

    # Add highlight points
    if highlights_spec.points:
        # crop highlights to bounding box if specified
        if bbox is not None:
            highlights_spec.points = {key: entry for key, entry in highlights_spec.points.items() if _is_point_in_bbox(entry.position_nm, bbox)}

        for key, entry in highlights_spec.points.items():
            style = marker_styles.get(entry.kind, marker_styles.get("default"))
            if style is None:
                continue

            position = np.asarray(entry.position_nm, dtype=float)

            # Track children before adding point
            before_children = len(viewer.scene.children)

            # Add marker point
            viewer.add_points(
                np.asarray([position]),
                color=style.color,
                marker=style.marker,
                size=style.size,
                size_space=style.size_space,
                center=False,
                name=f"highlight:{key}",
            )

            # Set material opacity on the newly added point(s)
            for child in viewer.scene.children[before_children:]:
                if hasattr(child, 'material') and hasattr(child.material, 'opacity'):
                    child.material.opacity = style.opacity

            label_text = entry.label
            if label_text:
                offset_world = np.array([0.0, 0.0, style.label_offset_nm], dtype=float)
                label_world = position + offset_world
                ndc_label = _world_to_ndc(label_world, viewer.camera)[0]
                if np.isnan(ndc_label).any():
                    label_world = position
                    ndc_label = _world_to_ndc(label_world, viewer.camera)[0]
                if np.isnan(ndc_label).any():
                    continue

                def _ndc_delta_from_world(delta_world: np.ndarray) -> np.ndarray:
                    target_ndc = _world_to_ndc(label_world + delta_world, viewer.camera)[0]
                    if np.isnan(target_ndc).any():
                        return np.zeros(3, dtype=float)
                    return target_ndc - ndc_label

                def _add_text(delta_world: np.ndarray, color: str, *, is_outline: bool = False) -> gfx.Text:
                    delta_ndc = _ndc_delta_from_world(delta_world)
                    visual = text2gfx(
                        label_text,
                        position=tuple((ndc_label + delta_ndc).astype(float)),
                        color=color,
                        font_size=style.label_font_size,
                        screen_space=True,
                        anchor=style.label_anchor,
                    )
                    material = visual.material
                    if hasattr(material, "font_name"):
                        material.font_name = FONT_FAMILY
                    material.depth_test = False
                    material.depth_write = False
                    visual.render_order = 1_000_000
                    visual.user_data = {
                        "highlight_label": True,
                        "world_base": label_world.astype(float),
                        "world_delta": delta_world.astype(float),
                        "is_outline": is_outline,
                    }
                    viewer.overlay_scene.add(visual)
                    return visual

                # Add outline/shadow for label if configured
                if style.label_outline_color:
                    d = float(style.label_outline_nm)
                    _add_text(np.array([d, -d, 0.0], dtype=float), style.label_outline_color, is_outline=True)

                # Add main label
                _add_text(np.zeros(3, dtype=float), style.label_color or style.color)
    
    # Add highlight edges
    if highlights_spec.edges:
        edge_segments = []
        edge_colors = []
        edge_linewidths = []
        edge_opacities = []
        
        for key, edge in highlights_spec.edges.items():
            start = np.asarray(edge.start_nm, dtype=float)
            end = np.asarray(edge.end_nm, dtype=float)
            segment = np.array([start, end])
            edge_segments.append(segment)
            
            # Determine color
            if edge.color is not None:
                color = edge.color
            else:
                # Use default color for kind if available
                style = marker_styles.get(edge.kind, marker_styles.get("default"))
                color = style.color if style else "#ff0000"
            
            edge_colors.append(color)
            edge_linewidths.append(edge.linewidth)
            edge_opacities.append(edge.opacity)
        
        if edge_segments:
            before_children = len(viewer.scene.children)
            viewer.add_lines(
                edge_segments,
                color=edge_colors[0] if len(set(edge_colors)) == 1 else edge_colors,
                linewidth=edge_linewidths[0] if len(set(edge_linewidths)) == 1 else edge_linewidths,
                linewidth_space="screen",
                center=False,
            )
            
            # Configure line material properties
            for i, child in enumerate(viewer.scene.children[before_children:]):
                if isinstance(child, gfx.Line):
                    material = child.material
                    if hasattr(material, "opacity"):
                        material.opacity = edge_opacities[i] if i < len(edge_opacities) else 1.0
                    if not options.line_depth_test:
                        material.depth_test = False
                    material.depth_write = False


def update_overlay_label_positions(viewer: Any) -> None:
    """Recompute screen-space positions for highlight labels based on current camera."""
    if not hasattr(viewer, "overlay_scene"):
        return

    for child in getattr(viewer.overlay_scene, "children", []):
        userdata = getattr(child, "user_data", None)
        if not isinstance(userdata, dict):
            continue
        if not userdata.get("highlight_label"):
            continue
        base = userdata.get("world_base")
        delta = userdata.get("world_delta", np.zeros(3, dtype=float))
        if base is None:
            continue
        base = np.asarray(base, dtype=float)
        delta = np.asarray(delta, dtype=float)
        target_ndc = _world_to_ndc(base + delta, viewer.camera)[0]
        if np.isnan(target_ndc).any():
            child.visible = False
        else:
            child.visible = True
            child.local.position = tuple(target_ndc.astype(float))



def crop_mesh(mesh : Mesh, bbox: Bbox):
    """
    Create a cropped version of the mesh.
    Vectorized version: no python-level loops over verts/faces.
    """
    if mesh.empty():
        return Mesh([], [], normals=None)

    verts = mesh.vertices
    faces = mesh.faces

    # bbox.minpt / bbox.maxpt are the two corners
    bbox_min = np.asarray(bbox.minpt, dtype=verts.dtype)
    bbox_max = np.asarray(bbox.maxpt, dtype=verts.dtype)

    # 1) mask vertices inside bbox
    #    (broadcast comparison: (N,3) vs (3,))
    vmask = np.all((verts >= bbox_min) & (verts <= bbox_max), axis=1)

    # if nothing survives, return empty mesh
    if not np.any(vmask):
        return Mesh([], [], normals=None)

    # 2) build index map: old_index -> new_index (or -1 if dropped)
    idx_map = -np.ones(len(verts), dtype=np.int64)
    idx_map[vmask] = np.arange(vmask.sum(), dtype=np.int64)

    # 3) keep only faces whose three vertices all survive
    keep_faces = np.all(vmask[faces], axis=1)
    cropped_faces = faces[keep_faces].copy()  # copy so we can mutate safely

    # 4) remap to new vertex indices in a single vectorized op
    cropped_faces = idx_map[cropped_faces]

    # 5) subset vertices (and normals if you have them)
    cropped_verts = verts[vmask]
    cropped_normals = None
    if mesh.normals is not None and len(mesh.normals):
        # original code treated normals as per-face, so keep the same convention
        cropped_normals = mesh.normals[keep_faces]

    return Mesh(
        cropped_verts,
        cropped_faces,
        cropped_normals,
        segid=mesh.segid,
        encoding_type=copy.deepcopy(mesh.encoding_type),
        encoding_options=copy.deepcopy(mesh.encoding_options),
    )
  



@profiled("prepare_mesh")
def prepare_mesh_for_rendering(
    mesh: cloudvolume.mesh.Mesh,
    center_coord: np.ndarray,
    padding_nm: float,
    crop: bool = True,
) -> tuple[Any, Bbox]:
    """
    Prepare a mesh for rendering by optionally cropping to a bounding box.

    Args:
        mesh: Mesh object with vertices
        center_coord: Center coordinate for bounding box
        padding_nm: Padding distance in nanometers around center
        crop: Whether to actually crop the mesh (if False, just returns bbox)

    Returns:
        Tuple of (cropped_mesh, bounding_box)

    Notes:
        - Bounding box is created as center ± padding in all dimensions
        - If crop=False, original mesh is returned with computed bbox
    """
    bbox_min = np.asarray(center_coord, dtype=float) - padding_nm
    bbox_max = np.asarray(center_coord, dtype=float) + padding_nm
    bbox = Bbox(tuple(bbox_min), tuple(bbox_max), unit="nm")

    if crop:
        cropped = crop_mesh(mesh, bbox) # 50x faster than mesh.crop due to proper vectorization
        if cropped is not None:
            return cropped, bbox
        

    return mesh, bbox