"""
Kaolin CUDA rendering backend.

Drop-in replacement for octarine/pygfx rendering that uses Kaolin's CUDA
rasterizer instead of wgpu/OpenGL. This enables GPU-accelerated rendering
on compute-only environments like Modal where graphics drivers aren't available.

Usage:
    from scripts.rendering.kaolin_backend import KaolinViewer

    viewer = KaolinViewer(offscreen=True)
    viewer.set_bgcolor("#ffffff")
    viewer.add(mesh, color="#1F7788")
    viewer.screenshot("output.png", size=(512, 512), alpha=True)

Note: torch and kaolin are imported lazily to allow this module to be
imported even when those dependencies aren't available (e.g., for testing
on systems without CUDA).
"""

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, TYPE_CHECKING

import numpy as np
from PIL import Image

# Lazy imports for torch and kaolin - only load when actually needed
_torch = None
_kal = None

def _get_torch():
    """Lazy import torch."""
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch

def _get_kaolin():
    """Lazy import kaolin."""
    global _kal
    if _kal is None:
        import kaolin as kal
        _kal = kal
    return _kal

def _check_dependencies():
    """Check if torch and kaolin are available."""
    try:
        _get_torch()
        _get_kaolin()
        return True
    except ImportError:
        return False

# Type hints only
if TYPE_CHECKING:
    import torch
    import kaolin as kal
    from kaolin.render.camera import Camera, CameraExtrinsics


# =============================================================================
# Helper Functions
# =============================================================================

def _hex_to_rgba(hex_color: str, alpha: float = 1.0) -> Tuple[float, float, float, float]:
    """Convert hex color string to RGBA tuple (0-1 range)."""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 6:
        r, g, b = (int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))
        return (r, g, b, alpha)
    elif len(hex_color) == 8:
        r, g, b, a = (int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4, 6))
        return (r, g, b, a * alpha)
    else:
        raise ValueError(f"Invalid hex color: {hex_color}")


def _parse_color(color: Union[str, Tuple, List], alpha: float = 1.0) -> Tuple[float, float, float, float]:
    """Parse color from various formats to RGBA tuple."""
    if isinstance(color, str):
        if color.startswith('#'):
            return _hex_to_rgba(color, alpha)
        # Named colors - basic support
        named_colors = {
            'red': (1, 0, 0), 'green': (0, 1, 0), 'blue': (0, 0, 1),
            'white': (1, 1, 1), 'black': (0, 0, 0), 'yellow': (1, 1, 0),
            'cyan': (0, 1, 1), 'magenta': (1, 0, 1), 'orange': (1, 0.5, 0),
            'r': (1, 0, 0), 'g': (0, 1, 0), 'b': (0, 0, 1),
            'w': (1, 1, 1), 'k': (0, 0, 0), 'y': (1, 1, 0),
        }
        if color.lower() in named_colors:
            r, g, b = named_colors[color.lower()]
            return (r, g, b, alpha)
        raise ValueError(f"Unknown color name: {color}")
    elif isinstance(color, (tuple, list)):
        if len(color) == 3:
            return (*color, alpha)
        elif len(color) == 4:
            return tuple(color)
    raise ValueError(f"Cannot parse color: {color}")


# =============================================================================
# Proxy Objects - Mimic pygfx interface
# =============================================================================

class KaolinMaterial:
    """Proxy for pygfx Material - stores rendering properties."""

    def __init__(self, color: Tuple[float, float, float, float] = (1, 1, 1, 1)):
        self._color = color
        self.opacity = color[3] if len(color) > 3 else 1.0
        self.transparent = self.opacity < 1.0
        self.depth_write = True
        self.depth_test = True

    @property
    def color(self):
        return _ColorProxy(self._color)

    @color.setter
    def color(self, value):
        if isinstance(value, tuple) and len(value) >= 3:
            self._color = value if len(value) == 4 else (*value, 1.0)
            self.opacity = self._color[3]


class _ColorProxy:
    """Proxy for color with r, g, b, a attributes."""
    def __init__(self, rgba):
        self.r, self.g, self.b = rgba[:3]
        self.a = rgba[3] if len(rgba) > 3 else 1.0
        self.rgba = rgba


class KaolinMesh:
    """Proxy for pygfx Mesh - stores mesh data and material."""

    def __init__(self, vertices: np.ndarray, faces: np.ndarray, color: Tuple):
        self.vertices = np.asarray(vertices, dtype=np.float32)
        self.faces = np.asarray(faces, dtype=np.int64)
        self.material = KaolinMaterial(color)
        self._object_id = None

    @property
    def color(self) -> Tuple[float, float, float, float]:
        return self.material._color


class KaolinLine:
    """Proxy for pygfx Line - stores line segment data."""

    def __init__(self, segments: List[np.ndarray], color: Tuple, linewidth: float):
        self.segments = [np.asarray(s, dtype=np.float32) for s in segments]
        self.material = KaolinMaterial(color)
        self.linewidth = linewidth
        self._object_id = None


class KaolinPoints:
    """Proxy for pygfx Points - stores point data."""

    def __init__(self, positions: np.ndarray, color: Tuple, size: float, marker: str):
        self.positions = np.asarray(positions, dtype=np.float32)
        self.material = KaolinMaterial(color)
        self.size = size
        self.marker = marker
        self._object_id = None


class KaolinTransform:
    """Proxy for pygfx local/world transform."""

    def __init__(self):
        self._position = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self._reference_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    @property
    def position(self):
        return tuple(self._position)

    @position.setter
    def position(self, value):
        self._position = np.asarray(value, dtype=np.float64)

    @property
    def reference_up(self):
        return self._reference_up

    @reference_up.setter
    def reference_up(self, value):
        self._reference_up = np.asarray(value, dtype=np.float64)


class KaolinCameraBase:
    """
    Base class for Kaolin cameras.
    Proxy for pygfx PerspectiveCamera/OrthographicCamera.
    Stores camera parameters and converts to Kaolin Camera at render time.
    """

    def __init__(self, projection: str = "perspective"):
        self._projection = projection  # "perspective" or "orthographic"
        self._fov = 50.0  # degrees
        self._width = 100.0  # ortho width
        self._height = 100.0  # ortho height
        self._near = 0.1
        self._far = 10000.0
        self._scale = 1.0

        self.local = KaolinTransform()
        self.world = KaolinTransform()

        self._target = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    @property
    def fov(self) -> float:
        return self._fov

    @fov.setter
    def fov(self, value: float):
        self._fov = value
        # fov=0 means orthographic
        if value == 0:
            self._projection = "orthographic"
        else:
            self._projection = "perspective"

    @property
    def width(self) -> float:
        return self._width

    @width.setter
    def width(self, value: float):
        self._width = value

    @property
    def height(self) -> float:
        return self._height

    @height.setter
    def height(self, value: float):
        self._height = value

    @property
    def near(self) -> float:
        return self._near

    @near.setter
    def near(self, value: float):
        self._near = value

    @property
    def far(self) -> float:
        return self._far

    @far.setter
    def far(self, value: float):
        self._far = value

    @property
    def scale(self) -> float:
        return self._scale

    @scale.setter
    def scale(self, value: float):
        self._scale = value

    def look_at(self, target):
        """Set camera to look at target point."""
        self._target = np.asarray(target, dtype=np.float64)

    @property
    def camera_matrix(self) -> np.ndarray:
        """
        Compute camera matrix (world to NDC) for coordinate transforms.
        Used by _world_to_ndc in scene_builder.py.
        """
        # Build view matrix
        eye = np.asarray(self.local.position, dtype=np.float64)
        target = self._target
        up = self.world.reference_up

        # Forward, right, up vectors
        forward = target - eye
        forward_len = np.linalg.norm(forward)
        if forward_len > 0:
            forward = forward / forward_len
        else:
            forward = np.array([0, 0, -1], dtype=np.float64)

        right = np.cross(forward, up)
        right_len = np.linalg.norm(right)
        if right_len > 0:
            right = right / right_len
        else:
            right = np.array([1, 0, 0], dtype=np.float64)

        true_up = np.cross(right, forward)

        # View matrix (4x4)
        view = np.eye(4, dtype=np.float64)
        view[0, :3] = right
        view[1, :3] = true_up
        view[2, :3] = -forward
        view[0, 3] = -np.dot(right, eye)
        view[1, 3] = -np.dot(true_up, eye)
        view[2, 3] = np.dot(forward, eye)

        # Projection matrix
        if self._projection == "orthographic":
            # Orthographic projection
            half_w = self._width / 2.0
            half_h = self._height / 2.0
            proj = np.array([
                [1/half_w, 0, 0, 0],
                [0, 1/half_h, 0, 0],
                [0, 0, -2/(self._far - self._near), -(self._far + self._near)/(self._far - self._near)],
                [0, 0, 0, 1]
            ], dtype=np.float64)
        else:
            # Perspective projection
            fov_rad = self._fov * math.pi / 180.0
            f = 1.0 / math.tan(fov_rad / 2.0)
            aspect = 1.0  # Assume square for simplicity
            proj = np.array([
                [f/aspect, 0, 0, 0],
                [0, f, 0, 0],
                [0, 0, (self._far + self._near)/(self._near - self._far), (2*self._far*self._near)/(self._near - self._far)],
                [0, 0, -1, 0]
            ], dtype=np.float64)

        return proj @ view

    def to_kaolin_camera(self, width: int, height: int, device: str = 'cuda'):
        """Convert to Kaolin Camera for rendering."""
        torch = _get_torch()
        kal = _get_kaolin()
        from kaolin.render.camera import Camera, CameraExtrinsics

        eye = torch.tensor([list(self.local.position)], dtype=torch.float32, device=device)
        at = torch.tensor([list(self._target)], dtype=torch.float32, device=device)
        up = torch.tensor([list(self.world.reference_up)], dtype=torch.float32, device=device)

        extrinsics = CameraExtrinsics.from_lookat(eye, at, up)

        if self._projection == "orthographic":
            # fov_distance controls orthographic zoom - use the larger of width/height
            # to ensure the full scene fits in view
            fov_distance = max(self._width, self._height) / 2.0
            intrinsics = kal.render.camera.OrthographicIntrinsics.from_frustum(
                width=width, height=height,
                fov_distance=fov_distance,
                near=self._near, far=self._far,
                device=device
            )
            # Store fov_distance for manual projection later
            intrinsics._fov_distance = fov_distance
        else:
            fov_rad = self._fov * math.pi / 180.0
            intrinsics = kal.render.camera.PinholeIntrinsics.from_fov(
                width=width, height=height, fov=fov_rad
            )

        return Camera(extrinsics, intrinsics)


class KaolinPerspectiveCamera(KaolinCameraBase):
    """Kaolin perspective camera - for isinstance() checks."""

    def __init__(self, fov: float = 50.0, **kwargs):
        super().__init__(projection="perspective")
        self._fov = fov


class KaolinOrthographicCamera(KaolinCameraBase):
    """Kaolin orthographic camera - for isinstance() checks."""

    def __init__(self, width: float = 100.0, height: float = 100.0, **kwargs):
        super().__init__(projection="orthographic")
        self._fov = 0.0  # Orthographic cameras have no FOV
        self._width = width
        self._height = height


# Alias for backward compatibility
KaolinCamera = KaolinCameraBase


class KaolinScene:
    """Proxy for pygfx Scene - container for visual objects."""

    def __init__(self):
        self._children: List[Any] = []

    @property
    def children(self) -> List[Any]:
        return self._children

    def add(self, visual):
        self._children.append(visual)

    def remove(self, *visuals):
        for v in visuals:
            if v in self._children:
                self._children.remove(v)


class KaolinOverlay:
    """
    Proxy for overlay scene (text labels).

    Note: Text rendering is complex in Kaolin, so we store label data
    and render them with PIL post-processing instead.
    """

    def __init__(self):
        self._children: List[Any] = []

    @property
    def children(self) -> List[Any]:
        return self._children

    def add(self, visual):
        self._children.append(visual)


# =============================================================================
# Main Viewer Class
# =============================================================================

class KaolinViewer:
    """
    Drop-in replacement for octarine.Viewer using Kaolin CUDA rendering.

    Implements the same interface as octarine.Viewer so existing code
    can use it without modification.
    """

    def __init__(self, offscreen: bool = True, camera: str = "ortho"):
        """
        Initialize the Kaolin viewer.

        Parameters
        ----------
        offscreen : bool
            Must be True (only offscreen rendering supported).
        camera : str
            Camera type: "ortho" or "perspective"
        """
        if not offscreen:
            raise ValueError("KaolinViewer only supports offscreen rendering")

        self._bgcolor = (1.0, 1.0, 1.0, 1.0)  # White background

        # Determine device - lazy check for torch
        try:
            torch = _get_torch()
            self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        except ImportError:
            self._device = 'cpu'  # Will fail at render time if torch not available

        # Initialize camera
        projection = "orthographic" if camera == "ortho" else "perspective"
        if projection == "orthographic":
            self._camera = KaolinOrthographicCamera()
        else:
            self._camera = KaolinPerspectiveCamera()

        # Initialize scenes
        self.scene = KaolinScene()
        self.overlay_scene = KaolinOverlay()

        # Settings
        self.show_bounds = False

        # Color cycle for auto-coloring
        self._color_cycle = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]
        self._color_index = 0
        self._label_counters: Dict[str, int] = {}

    @property
    def camera(self) -> KaolinCameraBase:
        """Get the camera object."""
        return self._camera

    @camera.setter
    def camera(self, value):
        """
        Set camera - handles both Kaolin and pygfx camera types.

        If a pygfx camera is assigned, we convert it to the equivalent Kaolin camera
        while preserving any properties that were already set.
        """
        # Check camera type by class name (avoids import path issues)
        class_name = type(value).__name__

        if class_name in ('PerspectiveCamera', 'KaolinPerspectiveCamera'):
            # Create new perspective camera, preserving old settings if possible
            new_cam = KaolinPerspectiveCamera()
            if hasattr(value, 'fov'):
                new_cam._fov = value.fov
            # Copy transform from old camera if it exists
            if self._camera is not None:
                new_cam.local._position = self._camera.local._position.copy()
                new_cam.world._reference_up = self._camera.world._reference_up.copy()
                new_cam._target = self._camera._target.copy()
                new_cam._near = self._camera._near
                new_cam._far = self._camera._far
            self._camera = new_cam

        elif class_name in ('OrthographicCamera', 'KaolinOrthographicCamera'):
            # Create new orthographic camera, preserving old settings if possible
            new_cam = KaolinOrthographicCamera()
            if hasattr(value, 'width'):
                new_cam._width = value.width
            if hasattr(value, 'height'):
                new_cam._height = value.height
            # Copy transform from old camera if it exists
            if self._camera is not None:
                new_cam.local._position = self._camera.local._position.copy()
                new_cam.world._reference_up = self._camera.world._reference_up.copy()
                new_cam._target = self._camera._target.copy()
                new_cam._near = self._camera._near
                new_cam._far = self._camera._far
            self._camera = new_cam

        elif 'KaolinCamera' in class_name or class_name == 'KaolinCameraBase':
            # Direct Kaolin camera assignment
            self._camera = value

        else:
            raise TypeError(f"Cannot assign camera of type {type(value)}")

    def _next_color(self) -> str:
        """Get next color from cycle."""
        color = self._color_cycle[self._color_index % len(self._color_cycle)]
        self._color_index += 1
        return color

    def _next_label(self, prefix: str) -> str:
        """Generate next label with prefix."""
        count = self._label_counters.get(prefix, 0)
        self._label_counters[prefix] = count + 1
        return f"{prefix}_{count}"

    def set_bgcolor(self, color: Union[str, Tuple]):
        """Set background color."""
        self._bgcolor = _parse_color(color)

    def add(self, obj, color: Optional[str] = None, name: Optional[str] = None,
            center: bool = True, **kwargs):
        """
        Add object to scene.

        Supports mesh objects with .vertices and .faces attributes.
        """
        if color is None:
            color = self._next_color()

        rgba = _parse_color(color)

        # Check if it's a mesh-like object
        vertices = getattr(obj, 'vertices', None)
        faces = getattr(obj, 'faces', None)

        if vertices is not None and faces is not None:
            mesh = KaolinMesh(vertices, faces, rgba)
            mesh._object_id = name or self._next_label("Mesh")
            self.scene.add(mesh)
        else:
            raise TypeError(f"Cannot add object of type {type(obj)}. "
                          f"Expected mesh with .vertices and .faces attributes.")

    def add_mesh(self, vertices: np.ndarray, faces: np.ndarray,
                 color: Optional[str] = None, name: Optional[str] = None,
                 center: bool = True, **kwargs):
        """Add mesh directly from vertices and faces."""
        if color is None:
            color = self._next_color()

        rgba = _parse_color(color)
        mesh = KaolinMesh(vertices, faces, rgba)
        mesh._object_id = name or self._next_label("Mesh")
        self.scene.add(mesh)

    def add_lines(self, lines: Union[np.ndarray, List[np.ndarray]],
                  name: Optional[str] = None,
                  color: Optional[str] = None,
                  linewidth: float = 1.0,
                  linewidth_space: str = "screen",
                  linestyle: str = "solid",
                  center: bool = True):
        """Add line segments to scene."""
        if color is None:
            color = self._next_color()

        rgba = _parse_color(color)

        # Normalize to list of segments
        if isinstance(lines, np.ndarray):
            segments = [lines]
        else:
            segments = list(lines)

        line_obj = KaolinLine(segments, rgba, linewidth)
        line_obj._object_id = name or self._next_label("Lines")
        self.scene.add(line_obj)

    def add_points(self, points: np.ndarray,
                   name: Optional[str] = None,
                   color: Optional[str] = None,
                   marker: Optional[str] = None,
                   size: float = 2.0,
                   size_space: str = "screen",
                   center: bool = True):
        """Add point markers to scene."""
        if color is None:
            color = self._next_color()

        rgba = _parse_color(color)

        points_obj = KaolinPoints(points, rgba, size, marker or "circle")
        points_obj._object_id = name or self._next_label("Points")
        self.scene.add(points_obj)

    def screenshot(self, filename: Optional[str] = "screenshot.png",
                   size: Optional[Tuple[int, int]] = None,
                   pixel_ratio: Optional[int] = None,
                   alpha: bool = True) -> Optional[np.ndarray]:
        """
        Render scene and save to file.

        Parameters
        ----------
        filename : str, optional
            Output filename. If None, returns numpy array.
        size : tuple, optional
            (width, height) of output image.
        pixel_ratio : int, optional
            Scale factor (ignored, for API compatibility).
        alpha : bool
            If True, include alpha channel.

        Returns
        -------
        numpy array if filename is None, otherwise None.
        """
        width, height = size or (512, 512)

        # Render the scene
        image = self._render(width, height, alpha=alpha)

        if filename:
            filename = Path(filename)
            if filename.suffix.lower() not in ['.png', '.jpg', '.jpeg']:
                filename = filename.with_suffix('.png')

            # Convert to PIL and save
            if alpha:
                pil_image = Image.fromarray(image, mode='RGBA')
            else:
                pil_image = Image.fromarray(image[:, :, :3], mode='RGB')

            pil_image.save(str(filename))
            return None
        else:
            return image

    def _render(self, width: int, height: int, alpha: bool = True) -> np.ndarray:
        """
        Render scene using Kaolin CUDA rasterizer.

        Returns RGBA numpy array (H, W, 4) with values 0-255.
        """
        # Start with background
        if alpha:
            # Transparent background
            image = np.zeros((height, width, 4), dtype=np.uint8)
        else:
            # Solid background
            bg = np.array(self._bgcolor[:3]) * 255
            image = np.ones((height, width, 4), dtype=np.uint8)
            image[:, :, :3] = bg.astype(np.uint8)
            image[:, :, 3] = 255

        # Collect meshes
        meshes = [obj for obj in self.scene.children if isinstance(obj, KaolinMesh)]

        if not meshes:
            return image

        # Render meshes with Kaolin
        mesh_image = self._render_meshes(meshes, width, height)

        # Composite mesh image onto background
        if mesh_image is not None:
            # Alpha blending
            mesh_alpha = mesh_image[:, :, 3:4].astype(np.float32) / 255.0
            bg_alpha = 1.0 - mesh_alpha

            image[:, :, :3] = (
                mesh_image[:, :, :3].astype(np.float32) * mesh_alpha +
                image[:, :, :3].astype(np.float32) * bg_alpha
            ).astype(np.uint8)

            image[:, :, 3] = np.clip(
                mesh_image[:, :, 3].astype(np.float32) + image[:, :, 3].astype(np.float32) * (1 - mesh_alpha[:, :, 0]),
                0, 255
            ).astype(np.uint8)

        # TODO: Render lines and points
        # For now, lines/points would need PIL post-processing or mesh conversion

        return image

    def _render_meshes(self, meshes: List[KaolinMesh], width: int, height: int) -> Optional[np.ndarray]:
        """Render meshes using Kaolin CUDA rasterizer with Blinn-Phong lighting."""
        if not meshes:
            return None

        torch = _get_torch()
        kal = _get_kaolin()

        device = self._device

        # Concatenate all meshes
        all_verts = []
        all_faces = []
        all_colors = []
        vertex_offset = 0

        for mesh in meshes:
            verts = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)
            faces = torch.tensor(mesh.faces, dtype=torch.long, device=device)

            # Get color with opacity
            r, g, b, a = mesh.material._color
            a = mesh.material.opacity  # Use material opacity

            # Per-face colors (expand to 3 vertices per face)
            n_faces = faces.shape[0]
            face_colors = torch.tensor([[r, g, b]] * n_faces, dtype=torch.float32, device=device)
            face_alphas = torch.full((n_faces,), a, dtype=torch.float32, device=device)

            all_verts.append(verts)
            all_faces.append(faces + vertex_offset)
            all_colors.append((face_colors, face_alphas))
            vertex_offset += verts.shape[0]

        # Stack everything
        vertices = torch.cat(all_verts, dim=0).unsqueeze(0)  # (1, V, 3)
        faces = torch.cat(all_faces, dim=0)  # (F, 3)
        face_colors = torch.cat([c[0] for c in all_colors], dim=0)  # (F, 3) - base colors
        face_alphas = torch.cat([c[1] for c in all_colors], dim=0)  # (F,)

        # Get Kaolin camera
        kaolin_camera = self.camera.to_kaolin_camera(width, height, device)

        # Index vertices by faces for world-space positions
        face_vertices = kal.ops.mesh.index_vertices_by_faces(vertices, faces)  # (1, F, 3, 3)

        # =====================================================================
        # Blinn-Phong Lighting (matching octarine/pygfx defaults)
        # =====================================================================

        # Compute face normals from world-space vertices
        v0 = face_vertices[0, :, 0, :]  # (F, 3)
        v1 = face_vertices[0, :, 1, :]  # (F, 3)
        v2 = face_vertices[0, :, 2, :]  # (F, 3)

        edge1 = v1 - v0
        edge2 = v2 - v0
        face_normals = torch.cross(edge1, edge2, dim=-1)  # (F, 3)
        face_normals = torch.nn.functional.normalize(face_normals, dim=-1)

        # Face centroids for lighting calculations
        face_centroids = face_vertices[0].mean(dim=1)  # (F, 3)

        # Camera position (eye) in world space
        camera_pos = torch.tensor(self._camera.local.position, dtype=torch.float32, device=device)

        # View direction per face (from face to camera)
        view_dirs = camera_pos.unsqueeze(0) - face_centroids  # (F, 3)
        view_dirs = torch.nn.functional.normalize(view_dirs, dim=-1)

        # Flip normals facing away from camera (back-face lighting)
        dot_nv = (face_normals * view_dirs).sum(dim=-1, keepdim=True)  # (F, 1)
        face_normals = torch.where(dot_nv < 0, -face_normals, face_normals)

        # Convert base colors to linear space (sRGB -> linear)
        base_colors_linear = torch.pow(face_colors, 2.2)

        # Material properties (MeshPhongMaterial defaults)
        shininess = 30.0
        specular_color = torch.tensor([0.287, 0.287, 0.287], device=device)  # #494949 in linear
        specular_strength = 1.0

        # Octarine default lights:
        # 1. AmbientLight (intensity 0.5)
        # 2. PointLight (intensity 4) at (-1M, -1M, -1M) - front/top/left
        # 3. PointLight (intensity 1) at (+1M, +1M, +1M) - back/bottom/right

        ambient_intensity = 0.5
        ambient_color = torch.ones(3, device=device)

        lights = [
            {'pos': torch.tensor([-1e6, -1e6, -1e6], device=device), 'intensity': 4.0, 'color': torch.ones(3, device=device)},
            {'pos': torch.tensor([1e6, 1e6, 1e6], device=device), 'intensity': 1.0, 'color': torch.ones(3, device=device)},
        ]

        # Initialize accumulated light
        total_diffuse = torch.zeros_like(base_colors_linear)  # (F, 3)
        total_specular = torch.zeros_like(base_colors_linear)  # (F, 3)

        # Process each point light
        for light in lights:
            light_pos = light['pos']
            light_intensity = light['intensity']
            light_color = light['color']

            # Light direction (from face to light)
            light_dirs = light_pos.unsqueeze(0) - face_centroids  # (F, 3)
            light_distance = torch.norm(light_dirs, dim=-1, keepdim=True)  # (F, 1)
            light_dirs = light_dirs / (light_distance + 1e-8)  # Normalize

            # Distance attenuation (inverse square, clamped)
            # For very distant lights like octarine's, attenuation is nearly constant
            decay = 2.0
            attenuation = 1.0 / torch.clamp(torch.pow(light_distance / 1e6, decay), min=0.01)

            # Lambertian diffuse: (1/π) * base_color * max(N·L, 0)
            dot_nl = torch.clamp((face_normals * light_dirs).sum(dim=-1, keepdim=True), min=0.0)  # (F, 1)
            irradiance = dot_nl * light_color.unsqueeze(0) * light_intensity * attenuation  # (F, 3)
            diffuse = (1.0 / math.pi) * base_colors_linear * irradiance

            # Blinn-Phong specular
            half_dirs = torch.nn.functional.normalize(light_dirs + view_dirs, dim=-1)  # (F, 3)
            dot_nh = torch.clamp((face_normals * half_dirs).sum(dim=-1, keepdim=True), min=0.0)  # (F, 1)

            # Microfacet distribution: D = (shininess/2 + 1) / π * (N·H)^shininess
            D = (1.0 / math.pi) * (shininess * 0.5 + 1.0) * torch.pow(dot_nh, shininess)

            # Simplified Fresnel (F0 = specular_color)
            dot_vh = torch.clamp((view_dirs * half_dirs).sum(dim=-1, keepdim=True), min=0.0)
            fresnel = specular_color.unsqueeze(0) + (1.0 - specular_color.unsqueeze(0)) * torch.pow(1.0 - dot_vh, 5.0)

            # Geometric term (simplified)
            G = 0.25

            specular = fresnel * G * D * irradiance * specular_strength

            total_diffuse += diffuse
            total_specular += specular

        # Ambient contribution
        ambient = ambient_intensity * ambient_color.unsqueeze(0) * base_colors_linear * (1.0 / math.pi)

        # Final color (linear space)
        final_color_linear = ambient + total_diffuse + total_specular

        # Convert back to sRGB
        final_color_srgb = torch.pow(torch.clamp(final_color_linear, 0.0, 1.0), 1.0 / 2.2)
        final_color_srgb = torch.clamp(final_color_srgb, 0.0, 1.0)

        # =====================================================================
        # Rasterization
        # =====================================================================

        # Transform to camera space
        vertices_camera = kaolin_camera.extrinsics.transform(vertices)
        face_vertices_camera = kal.ops.mesh.index_vertices_by_faces(vertices_camera, faces)

        # Get depth values - negate because camera looks along -Z, but rasterizer expects positive depth
        face_vertices_z = -face_vertices_camera[..., 2].contiguous()  # (1, F, 3)

        # Project to image space - Kaolin rasterize expects normalized [-1, 1] coords
        face_verts_2d = face_vertices_camera[..., :2]  # (1, F, 3, 2)

        intrinsics = kaolin_camera.intrinsics
        intrinsics_class = type(intrinsics).__name__

        if intrinsics_class == 'OrthographicIntrinsics':
            # Orthographic projection: scale x,y to normalized [-1, 1] coords
            fov_distance = getattr(intrinsics, '_fov_distance', max(self._camera._width, self._camera._height) / 2.0)
            x = face_verts_2d[..., 0] / fov_distance
            y = face_verts_2d[..., 1] / fov_distance
            face_vertices_image = torch.stack([x, y], dim=-1)
        else:
            # PinholeIntrinsics - use transform for NDC coordinates (includes perspective division)
            flat_verts = face_vertices_camera.reshape(1, -1, 3)
            flat_ndc = intrinsics.transform(flat_verts)  # (1, F*3, 3) - x,y are NDC, z is depth
            face_vertices_image = flat_ndc[..., :2].reshape(1, -1, 3, 2)  # (1, F, 3, 2)

        # Prepare face features with shaded colors - need (1, F, 3, C) where C is feature dim
        # Use the same shaded color for all 3 vertices of each face (flat shading)
        shaded_colors = final_color_srgb.unsqueeze(0).unsqueeze(2).expand(-1, -1, 3, -1)  # (1, F, 3, 3)

        # Add alpha channel
        face_alphas_expanded = face_alphas.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 3, 1)  # (1, F, 3, 1)
        face_features = torch.cat([shaded_colors, face_alphas_expanded], dim=-1)  # (1, F, 3, 4)

        try:
            # Rasterize with CUDA backend
            rendered, _ = kal.render.mesh.rasterize(
                height, width,
                face_vertices_z,
                face_vertices_image,
                face_features,
                backend='cuda'
            )

            # rendered is (1, H, W, 4) - RGBA
            rendered = rendered.squeeze(0)  # (H, W, 4)

            # Clamp and convert to uint8
            rendered = torch.clamp(rendered, 0.0, 1.0)
            rendered = (rendered * 255).to(torch.uint8)

            return rendered.cpu().numpy()

        except Exception as e:
            print(f"Kaolin rendering failed: {e}")
            # Return transparent image on failure
            return np.zeros((height, width, 4), dtype=np.uint8)

    @property
    def visuals(self) -> List[Any]:
        """Return all visual objects in scene."""
        return [c for c in self.scene.children if hasattr(c, '_object_id')]

    def clear(self):
        """Remove all objects from scene."""
        self.scene._children.clear()

    def close(self):
        """Close the viewer (no-op for offscreen)."""
        pass


# =============================================================================
# Factory function for drop-in replacement
# =============================================================================

def create_viewer(offscreen: bool = True, camera: str = "ortho") -> KaolinViewer:
    """
    Create a Kaolin-based viewer.

    Drop-in replacement for octarine.Viewer.
    """
    return KaolinViewer(offscreen=offscreen, camera=camera)
