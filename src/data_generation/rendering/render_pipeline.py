"""
Main rendering pipeline for orchestrating neuron scene rendering.

This module provides the high-level functions that coordinate scene construction,
camera configuration, rendering, and post-processing to produce final images.
"""

from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import tempfile
import time
import os

from cloudvolume import Bbox

import logging
import numpy as np
import octarine as oc
from PIL import Image

# Enable timing debug output with RENDER_TIMING=1
RENDER_TIMING = os.environ.get("RENDER_TIMING", "0") == "1"

# ---------------------------------------------------------------------------
# Async PNG save — fire-and-forget saves on background threads.
# PIL's PNG encoder releases the GIL, so this overlaps with the next render.
# ---------------------------------------------------------------------------
from concurrent.futures import ThreadPoolExecutor as _ThreadPool

_save_pool = _ThreadPool(max_workers=2, thread_name_prefix="png_save")
_save_futures: list = []


def _async_save(image: Image.Image, path) -> None:
    """Queue an image save on a background thread."""
    _save_futures.append(_save_pool.submit(image.save, str(path)))


def flush_saves() -> None:
    """Block until all pending saves complete. Call before returning image paths."""
    for f in _save_futures:
        f.result()
    _save_futures.clear()


from rendering.render_utils import (
    CameraViewSpec,
    HighlightsSpec,
    MeshSpec,
    NeuronGraphSpec,
    OutputConfig,
    RendererOptions,
    ProjectionLegendConfig,
    MarkerStyle,
)
from rendering.scene_builder import (
    add_meshes_to_viewer,
    add_graph_to_viewer,
    add_highlights_to_viewer,
    update_overlay_label_positions,
    prepare_mesh_for_rendering,
)
from rendering.camera_utils import (
    configure_camera_for_view,
    compute_world_dimensions,
    generate_orthographic_views,
)
from rendering.image_processing import (
    draw_projection_legend,
    stack_images,
    convert_rgba_to_rgb,
)
from rendering.highlight_utils import ensure_marker_styles_registry

import pygfx as gfx

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.profiler import profiled

logger = logging.getLogger(__name__)


def _apply_normal_material(viewer: Any) -> List[Any]:
    """Replace all mesh materials with MeshNormalMaterial. Returns saved originals."""
    saved = []
    for child in viewer.scene.children:
        if isinstance(child, gfx.Mesh):
            saved.append((child, child.material))
            child.material = gfx.MeshNormalMaterial(side="both")
    return saved


def _restore_materials(saved: List[Any]) -> None:
    """Restore original materials from _apply_normal_material() output."""
    for child, original_material in saved:
        child.material = original_material


def _read_depth_buffer(viewer: Any) -> np.ndarray:
    """Read the GPU depth buffer after a render pass.

    Returns [H, W] float32 array with values in [0, 1].
    Background pixels have depth = 1.0 (far plane).
    """
    renderer = viewer.renderer
    blender = renderer._blender
    depth_texture = blender.get_texture("depth")
    tex_size = depth_texture.size  # (width, height, depth_or_layers)
    device = renderer._device

    width, height = tex_size[0], tex_size[1]
    bytes_per_pixel = 4  # depth32float
    bytes_per_row = width * bytes_per_pixel

    data = device.queue.read_texture(
        {"texture": depth_texture, "mip_level": 0, "origin": (0, 0, 0)},
        {"offset": 0, "bytes_per_row": bytes_per_row, "rows_per_image": height},
        tex_size,
    )

    expected_size = width * height * bytes_per_pixel
    actual_size = len(data)

    if actual_size == expected_size:
        depth = np.frombuffer(data, dtype=np.float32).reshape(height, width)
    else:
        # Row padding — wgpu may align rows to 256 bytes
        aligned_bpr = actual_size // height
        row_floats = aligned_bpr // bytes_per_pixel
        raw = np.frombuffer(data, dtype=np.float32).reshape(height, row_floats)
        depth = raw[:, :width].copy()

    return depth


def _capture_depth_image(
    viewer: Any,
    output_path: Path,
    canvas_size_px: Tuple[int, int],
    dpi_scale: float,
    view: "CameraViewSpec",
    default_ortho_extent_nm: Optional[float],
) -> Tuple[Path, Image.Image]:
    """Render the current scene and save a normalized depth grayscale image.

    Foreground depth is mapped to [1, 255] (nearest → farthest), background = 0.
    """
    _t = time.time if RENDER_TIMING else None
    if _t: _t0 = _t()

    configure_camera_for_view(viewer, view, default_ortho_extent_nm)
    if _t: _t_cam = _t()

    width = int(canvas_size_px[0] * dpi_scale)
    height = int(canvas_size_px[1] * dpi_scale)

    # Trigger GPU render via screenshot (we discard the color image)
    fd, tmp_path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    try:
        viewer.screenshot(tmp_path, alpha=True, size=(width, height))
    finally:
        os.unlink(tmp_path)
    if _t: _t_gpu = _t()

    # Read depth buffer from GPU
    depth_raw = _read_depth_buffer(viewer)

    # Resize if needed (pixel_ratio mismatch)
    if depth_raw.shape[0] != height or depth_raw.shape[1] != width:
        depth_img = Image.fromarray(depth_raw, mode="F")
        depth_img = depth_img.resize((width, height), Image.NEAREST)
        depth_raw = np.array(depth_img)

    foreground = depth_raw < 1.0

    # Normalize foreground to [1, 255], background = 0
    out = np.zeros(depth_raw.shape, dtype=np.uint8)
    if foreground.any():
        d_min = depth_raw[foreground].min()
        d_max = depth_raw[foreground].max()
        normalized = (depth_raw - d_min) / (d_max - d_min + 1e-8)
        out[foreground] = (normalized[foreground] * 254 + 1).clip(1, 255).astype(np.uint8)
    if _t: _t_read = _t()

    depth_image = Image.fromarray(out, mode="L")
    _async_save(depth_image.copy(), output_path)
    if _t:
        _t_done = _t()
        print(f"[RENDER_TIMING] depth_view({view.name}): cam={_t_cam-_t0:.3f}s gpu={_t_gpu-_t_cam:.3f}s readback+norm={_t_read-_t_gpu:.3f}s save=async total={_t_done-_t0:.3f}s")

    return output_path, depth_image


@profiled("create_viewer")
def create_viewer(options: RendererOptions, backend: str = "octarine") -> Any:
    """
    Create and configure an offscreen viewer.

    Args:
        options: Rendering options with background color and display settings
        backend: Rendering backend - "octarine" (pygfx/wgpu) or "kaolin" (CUDA)

    Returns:
        Configured Viewer instance (octarine or Kaolin)
    """
    if backend == "kaolin":
        from rendering.kaolin_backend import KaolinViewer
        viewer = KaolinViewer(offscreen=True)
    else:
        viewer = oc.Viewer(offscreen=True)

    viewer.set_bgcolor(options.background_color)
    viewer.show_bounds = options.show_bounds
    return viewer


@profiled("render_single_view")
def render_single_view(
    viewer: Any,
    view: CameraViewSpec,
    output_path: Path,
    canvas_size_px: Tuple[int, int],
    dpi_scale: float,
    legend_config: ProjectionLegendConfig,
    default_ortho_extent_nm: Optional[float] = None,
    overwrite: bool = True,
    background_color: str = "#ffffff",
) -> Tuple[Path, Image.Image]:
    """
    Render a single camera view to an image file.

    Args:
        viewer: Octarine Viewer with scene already populated
        view: Camera view specification
        output_path: Path where image should be saved
        canvas_size_px: (width, height) in pixels before DPI scaling
        dpi_scale: Scale factor for high-DPI rendering
        legend_config: Configuration for projection legend overlay
        default_ortho_extent_nm: Default orthographic extent if not in view
        overwrite: Whether to overwrite existing files

    Returns:
        Tuple of (output_path, rendered_image)

    Raises:
        FileExistsError: If file exists and overwrite=False

    Notes:
        - Configures camera for the specified view
        - Renders to file with alpha channel
        - Adds projection legend with axes and scale bar
        - Converts RGBA to RGB before final save
    """
    _t = time.time if RENDER_TIMING else None
    if _t: _t0 = _t()

    # Check if file exists
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} exists and overwrite=False")

    # Configure camera for this view
    configure_camera_for_view(viewer, view, default_ortho_extent_nm)
    update_overlay_label_positions(viewer)
    if _t: _t_cam = _t()

    # Compute canvas dimensions with DPI scaling
    width = int(canvas_size_px[0] * dpi_scale)
    height = int(canvas_size_px[1] * dpi_scale)

    # Render to file with alpha channel
    viewer.screenshot(str(output_path), alpha=True, size=(width, height))
    if _t: _t_gpu = _t()

    # Load and post-process image
    with Image.open(output_path) as im:
        rgba = im.convert("RGBA")

    # Compute world dimensions for scale bar using ACTUAL image dimensions
    world_width_nm, nm_per_px = compute_world_dimensions(
        viewer, view, rgba.width, default_ortho_extent_nm
    )

    # Add projection legend
    annotated = draw_projection_legend(
        rgba, view, legend_config, nm_per_px, world_width_nm
    )

    # Convert to RGB
    bg = tuple(int(background_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
    final = convert_rgba_to_rgb(annotated, background_color=bg)
    if _t: _t_post = _t()

    # Save final image (async — continues to next view while PNG encodes)
    _async_save(final.copy(), output_path)
    if _t:
        _t_done = _t()
        print(f"[RENDER_TIMING] color_view({view.name}): cam={_t_cam-_t0:.3f}s gpu={_t_gpu-_t_cam:.3f}s post={_t_post-_t_gpu:.3f}s save=async total={_t_done-_t0:.3f}s")

    return output_path, final


@profiled("render_neuron_scene")
def render_neuron_scene(
    meshes: Sequence[MeshSpec],
    neuron_graph: Optional[NeuronGraphSpec],
    highlights: Optional[HighlightsSpec],
    cameras: Sequence[CameraViewSpec],
    center_nm: Tuple[float, float, float],
    output: OutputConfig,
    renderer_opts: RendererOptions,
    viewer: Any = None,
    material_mode: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Render complete neuron scene from multiple camera views.

    Args:
        meshes: Mesh specifications to render
        neuron_graph: Optional skeleton graph overlay
        highlights: Optional highlight markers and labels
        cameras: Sequence of camera view specifications
        center_nm: Center coordinate for the scene
        output: Output configuration (directory, naming, stacking)
        renderer_opts: Rendering options (canvas size, colors, styles)
        viewer: Optional pre-created octarine Viewer to reuse. If None, a new
                viewer is created. When reusing, the scene is cleared first.

    Returns:
        Dictionary mapping view names to output paths (or (path, image) tuples
        if output.return_images is True). Includes "stacked" entry if stacking enabled.

    Notes:
        - Reuses viewer if provided (clears scene first), otherwise creates new one
        - Renders each camera view sequentially
        - Optionally stacks views into combined image
        - Scene center is used for graph pruning if specified
    """

    import os as _os2, time as _time2
    _rv = _os2.environ.get("RENDER_VERBOSE", "0") == "1"
    _s0 = _time2.monotonic()
    def _slog(msg):
        if _rv:
            print(f"    [scene +{_time2.monotonic() - _s0:.3f}s] {msg}", flush=True)

    # Ensure output directory exists
    output.directory.mkdir(parents=True, exist_ok=True)

    t0 = time.time() if RENDER_TIMING else 0

    # Ensure marker styles registry is complete
    marker_registry = ensure_marker_styles_registry(
        renderer_opts.marker_styles,
        renderer_opts.default_marker_style,
    )

    # Reuse or create viewer
    _slog("viewer setup")
    if viewer is not None:
        # clear previous scene objects, keep lights/background
        viewer.clear()
        # overlay_scene holds text labels from highlights — not cleared by clear()
        if hasattr(viewer, "overlay_scene"):
            viewer.overlay_scene.clear()
        # update background color (may differ between render passes)
        viewer.set_bgcolor(renderer_opts.background_color)
    else:
        viewer = create_viewer(renderer_opts)
    _slog("viewer ready")

    # Compute scene center for graph pruning
    scene_center = np.asarray(center_nm, dtype=float)

    # Extract bbox from mesh specs if available (for graph cropping)
    scene_bbox = None
    for mesh_spec in meshes:
        if mesh_spec.bbox is not None:
            scene_bbox = mesh_spec.bbox
            break  # Use first bbox found

    # Add scene elements
    _slog(f"add_meshes_to_viewer ({len(meshes)} meshes)")
    add_meshes_to_viewer(viewer, meshes, renderer_opts)
    _slog("add_meshes done")
    add_graph_to_viewer(viewer, neuron_graph, scene_center, renderer_opts, bbox=scene_bbox)
    _slog("add_graph done")
    add_highlights_to_viewer(viewer, highlights, marker_registry, renderer_opts, bbox=scene_bbox)
    _slog("add_highlights done")

    # Apply material mode (normals override)
    _saved_materials = None
    if material_mode == "normals":
        _saved_materials = _apply_normal_material(viewer)
        _slog("applied normal material")

    if RENDER_TIMING:
        print(f"[RENDER_TIMING] add_highlights: {time.time() - t0:.3f}s")
        t0 = time.time()

    # Render each view
    outputs: Dict[str, Any] = {}
    captured_images: List[Image.Image] = []

    for view in cameras:
        _slog(f"render view '{view.name}'")
        filename = f"{output.base_name}_{view.name}.png"
        path = output.directory / filename

        if material_mode == "depth":
            output_path, rendered_image = _capture_depth_image(
                viewer=viewer,
                output_path=path,
                canvas_size_px=renderer_opts.canvas_size_px,
                dpi_scale=renderer_opts.dpi_scale,
                view=view,
                default_ortho_extent_nm=renderer_opts.default_ortho_extent_nm,
            )
        else:
            output_path, rendered_image = render_single_view(
                viewer=viewer,
                view=view,
                output_path=path,
                canvas_size_px=renderer_opts.canvas_size_px,
                dpi_scale=renderer_opts.dpi_scale,
                legend_config=renderer_opts.projection_legend,
                default_ortho_extent_nm=renderer_opts.default_ortho_extent_nm,
                overwrite=output.overwrite,
                background_color=renderer_opts.background_color,
            )
        _slog(f"render view '{view.name}' done")

        captured_images.append(rendered_image.copy())

        if output.return_images:
            outputs[view.name] = (output_path, rendered_image.copy())
        else:
            outputs[view.name] = output_path

    # Stack images if requested
    if output.stack:
        _slog("stacking images")
        stacked = stack_images(captured_images, output.stack)
        stack_path = output.directory / f"{output.base_name}_{output.stack_name}.png"
        stacked.save(stack_path)
        outputs["stacked"] = (
            (stack_path, stacked) if output.return_images else stack_path
        )

    if _saved_materials is not None:
        _restore_materials(_saved_materials)
        _slog("restored original materials")

    _slog("render_neuron_scene total done")
    return outputs


@profiled("render_neuron_views")
def render_neuron_views(
    root_id: int,
    meshes: Union[
        MeshSpec,
        Sequence[MeshSpec],
        Any,
        Sequence[Any],
    ],
    neuron_graph: Optional[Any],
    center_coord: np.ndarray,
    center_node: Optional[int],
    extent_nm: float,
    output_dir: Path,
    base_name_prefix: Optional[str] = None,
    padding_nm: float = None,
    show_projection_legend: bool = True,
    include_graph: bool = True,
    node_count_limit: int = 20,
    extra_highlights: Optional[HighlightsSpec] = None,
    center_marker: str = "diamond",
    neighbor_marker: str = "circle",
    center_color: str = "#ff4f4f",
    neighbor_color: str = "#ffa500",
    center_opacity: float = 1.0,
    neighbor_opacity: float = 1.0,
    base_marker_size_nm: float = 200.0,
    max_marker_size_nm: float = 20_000.0,
    canvas_size_px: Tuple[int, int] = (1024, 1024),
    mesh_crop_enabled: bool = True,
    logger : logging.Logger = None,
    backend: str = "octarine",
    stack_views: bool = None,
    mesh_transparency: bool = True,
    viewer: Any = None,
    material_mode: Optional[str] = None,
    background_color: Optional[str] = None,
) -> Tuple[Dict[str, Path], Dict[str, Any]]:
    """
    Render orthographic neuron views with optional graph overlay and highlights.

    Args:
        root_id: Neuron root ID
        meshes: Single mesh object, `MeshSpec`, or a sequence of meshes/specs to render
        neuron_graph: NetworkX graph with node coordinates
        center_coord: Center coordinate for views
        center_node: Center node ID (if using graph)
        extent_nm: Orthographic extent (half-width of view) to render
        output_dir: Directory for output images
        base_name_prefix: Prefix for output filenames
        padding_nm: Padding around center for mesh cropping
        show_projection_legend: Whether to show axis legend
        include_graph: Whether to include graph overlay
        extra_highlights: Additional highlights beyond graph nodes
        center_marker: Marker shape for center node
        neighbor_marker: Marker shape for neighbor nodes
        center_color: Color for center marker
        neighbor_color: Color for neighbor markers
        base_marker_size_nm: Base size for markers
        max_marker_size_nm: Maximum marker size
        canvas_size_px: Canvas dimensions
        mesh_crop_enabled: Whether to crop mesh to bounding box (True) or show full mesh (False)
        backend: Rendering backend - "octarine" (pygfx/wgpu) or "kaolin" (CUDA)
        mesh_transparency: Whether to render meshes with transparency (True) or not (False)
    Returns:
        Tuple of (image_paths, metadata) where:
        - image_paths: Dict mapping view names ("front", "side", "top") to file paths
        - metadata: Dict with "neighbors" list

    Notes:
        - Finds nearby graph nodes if graph overlay enabled
        - Creates appropriately scaled markers and labels
        - Optionally crops each mesh to a shared bounding box around the center
        - Renders from front, side, and top orthographic views
    """

    from rendering.highlight_utils import (
        create_highlight_entries_from_graph,
        compute_marker_styles,
        merge_highlight_entries,
    )
    verbose = False
    if logger is not None:
        verbose = True

    import os as _os, time as _time
    _render_verbose = _os.environ.get("RENDER_VERBOSE", "0") == "1"
    _t0 = _time.monotonic()
    def _vlog(msg):
        if _render_verbose:
            elapsed = _time.monotonic() - _t0
            print(f"  [render_verbose +{elapsed:.3f}s] {msg}", flush=True)

    _vlog("render_neuron_views START")

    if padding_nm is None:
        padding_nm = extent_nm * 1.2

    output_dir.mkdir(parents=True, exist_ok=True)
    # Create highlights from graph if enabled
    highlight_entries: Dict[str, Any] = {}
    neighbor_meta: List[Dict[str, Any]] = []

    _vlog("creating highlights from graph")

    if include_graph and neuron_graph is not None:
        highlight_entries, neighbor_meta = (
            create_highlight_entries_from_graph(
                neuron_graph, center_coord, center_node, extent_nm, count_limit=node_count_limit
            )
        )
    _vlog("done highlights")

    # Merge with extra highlights if provided
    if extra_highlights is not None:
        highlight_entries = merge_highlight_entries(
            highlight_entries, extra_highlights
        )

    # Compute marker styles for this zoom level
    marker_styles = compute_marker_styles(
        extent_nm,
        center_marker=center_marker,
        neighbor_marker=neighbor_marker,
        center_color=center_color,
        neighbor_color=neighbor_color,
        base_marker_size_nm=base_marker_size_nm,
        max_marker_size_nm=max_marker_size_nm,
        extra_highlights=extra_highlights,
        center_opacity=center_opacity,
        neighbor_opacity=neighbor_opacity,
    )

    # Early validation: ensure meshes parameter is valid
    assert meshes is not None, (
        f"render_neuron_views: meshes parameter cannot be None. "
        f"Pass a MeshSpec, a sequence of MeshSpec/mesh objects, or a single mesh object."
    )
    

    # Normalize meshes input into MeshSpec objects
    def _iter_mesh_specs(
        value: Union[MeshSpec, Sequence[MeshSpec], Any, Sequence[Any]]
    ) -> Iterable[MeshSpec]:
        if isinstance(value, MeshSpec):
            yield value
            return
        if isinstance(value, (list, tuple)):
            for item in value:  # type: ignore[arg-type]
                if isinstance(item, MeshSpec):
                    yield item
                else:
                    # Create MeshSpec with lower opacity for better transparency
                    yield MeshSpec(root_id=root_id, mesh=item, opacity=1)
            return
        # Single mesh object (not a sequence) - also set lower opacity
        yield MeshSpec(root_id=root_id, mesh=value, opacity=1)

    normalized_specs = list(_iter_mesh_specs(meshes))
    if not normalized_specs:
        raise ValueError(
            f"render_neuron_views requires at least one mesh to render. "
            f"Received meshes parameter of type {type(meshes)}. "
            f"If passing a single mesh object, ensure it is not None. "
            f"If passing a sequence, ensure it is not empty."
        )

    if verbose:
        logger.debug("Preparing meshes for rendering...")

    extent_padding = max(padding_nm, extent_nm)
    cropped_specs: List[MeshSpec] = []
    combined_bbox: Optional[Bbox] = None
    _vlog(f"prepare_mesh_for_rendering: {len(normalized_specs)} specs, crop={mesh_crop_enabled}")
    for i, spec in enumerate(normalized_specs):
        if spec.mesh is None:
            continue
        verts_before = len(spec.mesh.vertices) if hasattr(spec.mesh, 'vertices') else '?'
        _vlog(f"  crop mesh {i}: {verts_before} verts before crop")
        cropped_mesh, bbox = prepare_mesh_for_rendering(
            spec.mesh, center_coord, extent_padding, crop=mesh_crop_enabled
        )
        verts_after = len(cropped_mesh.vertices) if hasattr(cropped_mesh, 'vertices') else '?'
        _vlog(f"  crop mesh {i}: {verts_after} verts after crop")
        combined_bbox = _union_bbox(combined_bbox, bbox)
        # bbox=None: mesh is already cropped, skip redundant crop in add_meshes_to_viewer
        cropped_specs.append(replace(spec, mesh=cropped_mesh, bbox=None))
    _vlog(f"prepare_mesh_for_rendering DONE")

    if not cropped_specs:
        raise ValueError("render_neuron_views could not prepare any meshes for rendering.")
    if combined_bbox is None:
        raise ValueError("render_neuron_views failed to compute a bounding box for rendering.")

    if verbose:
        logger.debug("Prepared meshes for rendering...")


    # Generate camera views
    _vlog("generate_orthographic_views START")
    if verbose:
        logger.debug("Generating orthographic views...")
    view_specs = generate_orthographic_views(combined_bbox, center_override=center_coord)
    _vlog("generate_orthographic_views DONE")
    if verbose:
        logger.debug("Done generating views.")

    # Configure output
    base_name = (
        f"{base_name_prefix}_{int(extent_nm)}nm"
        if base_name_prefix
        else f"{root_id}_{int(extent_nm)}nm"
    )

    output_cfg = OutputConfig(
        directory=output_dir,
        base_name=base_name,
        stack="horizontal" if stack_views else None,
        overwrite=True,
    )

    # Configure renderer
    renderer_opts = RendererOptions(
        background_color=background_color or "#ffffff",
        default_ortho_extent_nm=extent_nm,
        marker_styles=marker_styles,
        default_marker_style=MarkerStyle(size_space="world"),
        canvas_size_px=canvas_size_px,
        projection_legend=ProjectionLegendConfig(enabled=show_projection_legend),
        mesh_transparency=mesh_transparency,
    )

    # Create specs for rendering
    graph_spec = (
        NeuronGraphSpec(graph=neuron_graph)
        if include_graph and neuron_graph is not None
        else None
    )
    
    # Merge highlight entries with extra_highlights
    merged_points = {}
    merged_edges = {}
    
    # Add points from highlight_entries (graph-based highlights)
    if highlight_entries:
        merged_points.update(highlight_entries)
    
    # Add points and edges from extra_highlights
    if extra_highlights is not None:
        if extra_highlights.points:
            merged_points.update(extra_highlights.points)
        if extra_highlights.edges:
            merged_edges.update(extra_highlights.edges)
    
    highlights_spec = (
        HighlightsSpec(points=merged_points, edges=merged_edges)
        if (merged_points or merged_edges) else None
    )

    # Render the scene
    _vlog("render_neuron_scene START")
    image_paths = render_neuron_scene(
        meshes=cropped_specs,
        neuron_graph=graph_spec,
        highlights=highlights_spec,
        cameras=view_specs,
        center_nm=tuple(float(v) for v in center_coord),
        output=output_cfg,
        renderer_opts=renderer_opts,
        material_mode=material_mode,
        viewer=viewer,
    )
    _vlog("render_neuron_scene DONE")
    # Convert paths to Path objects
    image_paths_out = {
        view: Path(path)
        for view, path in image_paths.items()
        if isinstance(path, (str, Path))
    }

    # Build metadata
    metadata = {
        "neighbors": neighbor_meta,
    }
    _vlog(f"render_neuron_views TOTAL DONE")
    return image_paths_out, metadata


def _union_bbox(existing: Optional[Bbox], new_bbox: Bbox) -> Bbox:
    """
    Combine two bounding boxes, returning a new box that spans both.
    """
    if existing is None:
        return new_bbox
    min_pt = tuple(min(a, b) for a, b in zip(existing.minpt, new_bbox.minpt))
    max_pt = tuple(max(a, b) for a, b in zip(existing.maxpt, new_bbox.maxpt))
    unit = existing.unit or new_bbox.unit
    return Bbox(min_pt, max_pt, unit=unit)
