from dataclasses import dataclass, field
from typing import Any, Optional, Tuple, Dict, Mapping
from cloudvolume import Bbox
from pathlib import Path
import pygfx as gfx
from pygfx import font_manager
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FONT_FAMILY = "Atkinson Hyperlegible"
FONT_PATH = REPO_ROOT / "assets" / "fonts" / "AtkinsonHyperlegible-Regular.ttf"

if FONT_PATH.exists():
    try:
        font_manager.add_font_file(str(FONT_PATH))
    except Exception:
        pass


@dataclass
class MeshSpec:
    root_id: Optional[int] = None
    mesh: Optional[Any] = None
    color: str = "#1F7788"
    opacity: float = 0.9
    bbox: Optional[Bbox] = None
    visible: bool = True


@dataclass
class NeuronGraphSpec:
    graph: Any  # Expected networkx.Graph with 3D coords
    color: str = "#ffff00"
    linewidth: float = 0.1
    opacity: float = 0.9
    prune_radius_nm: Optional[float] = None


@dataclass
class HighlightEntry:
    position_nm: Tuple[float, float, float]
    kind: str = "default"
    label: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HighlightEdge:
    """Represents a single highlight edge with start and end coordinates."""
    start_nm: Tuple[float, float, float]
    end_nm: Tuple[float, float, float]
    kind: str = "default"
    color: Optional[str] = None  # If None, uses default for kind
    linewidth: float = 2.0
    opacity: float = 1.0


@dataclass
class HighlightsSpec:
    points: Mapping[str, HighlightEntry] = field(default_factory=dict)
    edges: Mapping[str, HighlightEdge] = field(default_factory=dict)


@dataclass
class MarkerStyle:
    color: str = "#2ca02c"
    marker: str = "circle"
    size: float = 20.0
    opacity: float = 1.0
    size_space: str = "screen"
    label_color: Optional[str] = "#ffffff"
    label_font_size: float = 32.0
    label_offset_nm: float = 1200.0
    label_anchor: str = "top-center"
    label_outline_color: Optional[str] = "#000000"
    label_outline_nm: float = 60.0


@dataclass
class CameraViewSpec:
    name: str
    projection: str = "orthographic"  # or "perspective"
    position_nm: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    target_nm: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    up: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    fov_degrees: Optional[float] = None
    clip_range_nm: Optional[Tuple[float, float]] = None
    zoom: Optional[float] = None
    ortho_extent_nm: Optional[float] = None


@dataclass
class ProjectionLegendConfig:
    enabled: bool = True
    margin_px: int = 40
    arrow_length_px: int = 120
    arrow_width_px: int = 6
    arrow_head_px: int = 18
    font_size_px: int = 28
    label_offset_px: int = 10
    color: str = "#000000"
    scale_bar: bool = True
    scale_bar_max_fraction: float = 0.25
    scale_bar_height_px: int = 6
    scale_bar_label_offset_px: int = 14
    show_axes: bool = True  # Show axis arrows and labels (X/Y/Z)


@dataclass
class RendererOptions:
    background_color: str = "#ffffff"
    canvas_size_px: Tuple[int, int] = (1024, 1024)
    dpi_scale: float = 1.0
    mesh_transparency: bool = True
    line_depth_test: bool = False
    marker_depth_test: bool = False
    show_bounds: bool = False
    marker_styles: Dict[str, MarkerStyle] = field(default_factory=dict)
    default_marker_style: MarkerStyle = field(default_factory=MarkerStyle)
    scene_bbox: Optional[Bbox] = None
    ensure_label_visibility: bool = True
    default_ortho_extent_nm: Optional[float] = None
    default_zoom: Optional[float] = None
    projection_legend: ProjectionLegendConfig = field(default_factory=ProjectionLegendConfig)


@dataclass
class OutputConfig:
    directory: Path = Path("plots/mesh_highlights")
    base_name: str = "neuron_view"
    stack: Optional[str] = None  # "vertical" | "horizontal"
    stack_name: str = "stacked"
    return_images: bool = False
    overwrite: bool = True