"""
Neuron rendering pipeline for 3D visualization of connectome data.

This module provides a functional pipeline for rendering neurons with meshes,
skeleton graphs, and highlight markers from multiple camera views.
"""

from rendering.render_pipeline import (
    render_neuron_scene,
    render_neuron_views,
)
from rendering.scene_builder import (
    add_meshes_to_viewer,
    add_graph_to_viewer,
    add_highlights_to_viewer,
)
from rendering.camera_utils import (
    generate_orthographic_views,
    configure_camera_for_view,
)
from rendering.highlight_utils import (
    create_highlight_entries_from_graph,
    compute_marker_styles,
)
from rendering.image_processing import (
    draw_projection_legend,
    stack_images,
)
from rendering.geometry_renderer import (
    render_geometry_views,
    generate_geometry_view_specs,
    STANDARD_ANGLES,
)

__all__ = [
    "render_neuron_scene",
    "render_neuron_views",
    "add_meshes_to_viewer",
    "add_graph_to_viewer",
    "add_highlights_to_viewer",
    "generate_orthographic_views",
    "configure_camera_for_view",
    "create_highlight_entries_from_graph",
    "compute_marker_styles",
    "draw_projection_legend",
    "stack_images",
    "render_geometry_views",
    "generate_geometry_view_specs",
    "STANDARD_ANGLES",
]
