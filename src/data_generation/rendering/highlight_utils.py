"""
Utilities for creating and styling highlight markers and labels.

Functions for generating highlight specifications from graph data and
computing marker styles based on zoom levels and marker types.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from rendering.render_utils import HighlightEntry, HighlightsSpec, MarkerStyle


def create_highlight_entries_from_graph(
    graph: Any,
    center_coord: np.ndarray,
    center_node: int,
    extent_nm: float,
    count_limit: int = 20,
) -> Tuple[Dict[str, HighlightEntry], List[Dict[str, Any]], Optional[Tuple[int, int]]]:
    """
    Create highlight entries for nodes near a center point in the graph.

    Args:
        graph: NetworkX graph with node coordinates
        center_coord: Center point coordinates (x, y, z) in nm
        center_node: ID of the center node
        extent_nm: Search radius in nanometers

    Returns:
        Tuple of:
        - highlight_entries: Dict mapping labels to HighlightEntry objects
        - neighbor_metadata: List of dicts with node info (label, id, coord, distance, edges)

    Notes:
        - Center node gets label "0" and kind "center"
        - Nearby nodes get sequential numeric labels and kind "neighbor"
        - Uses fetch_available_nodes to find neighbors within extent
        - Returns adjacent edge information for each node
    """
    from connectome.neuron_graph import fetch_available_nodes, get_nearest_node

    if center_node is None:
        center_node, _ = get_nearest_node(graph, center_coord)
        print(f"No explicit center node provided, finding nearest node to {center_coord}")
        
    neighbor_data = fetch_available_nodes(graph, center_coord, extent_nm, count_limit=count_limit)
    available = neighbor_data.get("available", {})
    nodes_data = available.get("nodes", [])

    highlight_entries: Dict[str, HighlightEntry] = {}
    neighbor_meta: List[Dict[str, Any]] = [] 

    # Add center node highlight
    highlight_entries["0"] = HighlightEntry(
        position_nm=list(center_coord),
        kind="center",
        label="0",
    )
    neighbor_meta.append(
        {
            "label": "0",
            "node_id": center_node,
            "coord_nm": list(center_coord),
            "distance_nm": 0.0,
        }
    )

    # Add neighbor node highlights
    idx_counter = 1
    for entry in nodes_data:
        node_id = int(entry.get("node_id"))
        coord = np.asarray(entry.get("coord_nm"), dtype=float)
        label = str(idx_counter)

        if node_id == center_node:
            continue

        highlight_entries[label] = HighlightEntry(
            position_nm=tuple(float(v) for v in coord),
            kind="neighbor",
            label=label,
        )
        neighbor_meta.append(
            {
                "label": label,
                "node_id": node_id,
                "coord_nm": coord.tolist(),
                "distance_nm": float(entry.get("distance_nm", np.nan)),
            }
        )
        idx_counter += 1

    return highlight_entries, neighbor_meta


def compute_marker_styles(
    extent_nm: float,
    center_marker: str = "diamond",
    neighbor_marker: str = "circle",
    center_color: str = "#ff4f4f",
    neighbor_color: str = "#ffa500",
    base_marker_size_nm: float = 200.0,
    max_marker_size_nm: float = 20_000.0,
    extra_highlights: Optional[HighlightsSpec] = None,
    center_opacity: float = 1.0,
    neighbor_opacity: float = 1.0,
) -> Dict[str, MarkerStyle]:
    """
    Compute marker styles scaled appropriately for the zoom level.

    Args:
        extent_nm: Current orthographic extent (half-width of view) in nm
        center_marker: Marker shape for center node
        neighbor_marker: Marker shape for neighbor nodes
        center_color: Color for center marker
        neighbor_color: Color for neighbor markers
        base_marker_size_nm: Base size for markers in nm
        max_marker_size_nm: Maximum marker size in nm
        extra_highlights: Optional extra highlights that need marker styles

    Returns:
        Dictionary mapping marker kinds to MarkerStyle objects

    Notes:
        - Marker sizes scale with zoom level (larger at wider extents)
        - Label offsets and font sizes also scale with extent
        - Extra highlights get default styling based on their kind
    """
    # Compute scaled marker size
    size_scale = max(1.0, extent_nm / 5_000.0)
    scaled_size = max(
        base_marker_size_nm,
        min(base_marker_size_nm * size_scale, max_marker_size_nm),
    )

    # Create base marker styles
    marker_styles = {
        "center": MarkerStyle(
            color=center_color,
            marker=center_marker,
            size=scaled_size,
            opacity=center_opacity,
            size_space="world",
            label_font_size=28.0,
            label_offset_nm=max(200.0, 1_800.0 * (extent_nm / 20_000.0)),
        ),
        "neighbor": MarkerStyle(
            color=neighbor_color,
            marker=neighbor_marker,
            size=scaled_size,
            opacity=neighbor_opacity,
            size_space="world",
            label_font_size=24.0,
            label_offset_nm=max(200.0, 1_500.0 * (extent_nm / 20_000.0)),
        ),
    }

    # Add styles for extra highlight kinds
    if extra_highlights is not None and extra_highlights.points:
        for entry in extra_highlights.points.values():
            kind = entry.kind or "extra"
            if kind in marker_styles:
                continue

            if kind == "merge_error":
                color = "#ff0000"
                marker = "+"
                label_color = "#ff0000"
            elif kind == "split_error":
                color = "#ff0000"
                marker = "x"
                label_color = "#ff0000"
            elif kind == "agent":
                color = "#1f77ff"
                marker = "square"
                label_color = "#1f77ff"
            elif kind == "you":
                color = "#1f77ff"
                marker = "circle"
                label_color = "#1f77ff"
            elif kind == "source":
                color = "#00ff00"  # Green
                marker = "diamond"
                label_color = "#00ff00"
            elif kind == "sink":
                color = "#0000ff"  # Blue
                marker = "circle"
                label_color = "#0000ff"
            elif kind == "interface":
                color = "#00ff00"  # Green - interface point for split error
                marker = "diamond"
                label_color = "#00ff00"
            elif kind == "ground_truth":
                color = "#00ff00"  # Green
                marker = "diamond"
                label_color = "#00ff00"
            elif kind == "predicted":
                color = "#ff0000"  # Red
                marker = "circle"
                label_color = "#ff0000"
            else:
                color = "#000000"
                marker = "x"
                label_color = "#000000"

            marker_styles[kind] = MarkerStyle(
                color=color,
                marker=marker,
                size=scaled_size,
                opacity=1.0,
                size_space="world",
                label_font_size=24.0,
                label_offset_nm=max(200.0, 1_500.0 * (extent_nm / 20_000.0)),
                label_color=label_color,
            )

    return marker_styles


def merge_highlight_entries(
    base_entries: Dict[str, HighlightEntry],
    extra_highlights: Optional[HighlightsSpec],
) -> Dict[str, HighlightEntry]:
    """
    Merge base highlight entries with additional highlights.

    Args:
        base_entries: Base highlight entries (from graph nodes)
        extra_highlights: Optional additional highlights to include

    Returns:
        Combined dictionary of highlight entries

    Notes:
        - Extra highlights override base entries with same key
        - Preserves all base entries not overridden
    """
    merged = dict(base_entries)

    if extra_highlights is not None and extra_highlights.points:
        merged.update(extra_highlights.points)

    return merged


def ensure_marker_styles_registry(
    marker_styles: Dict[str, MarkerStyle],
    default_style: MarkerStyle,
) -> Dict[str, MarkerStyle]:
    """
    Ensure marker styles registry has a default entry.

    Args:
        marker_styles: Dictionary of marker styles by kind
        default_style: Default marker style to use if "default" not present

    Returns:
        Marker styles dict with guaranteed "default" entry
    """
    registry = dict(marker_styles)
    if "default" not in registry:
        registry["default"] = default_style

    return registry
