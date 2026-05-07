"""
Image post-processing functions for adding legends, scale bars, and stacking images.

These functions operate on PIL Images after rendering to add annotations
and combine multiple views.
"""

import math
from typing import Optional, Sequence, Tuple, Dict

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from rendering.render_utils import CameraViewSpec, ProjectionLegendConfig


# Axis basis vectors for determining projection directions
_AXIS_BASIS: Dict[str, np.ndarray] = {
    "X": np.array([1.0, 0.0, 0.0], dtype=float),
    "Y": np.array([0.0, 1.0, 0.0], dtype=float),
    "Z": np.array([0.0, 0.0, 1.0], dtype=float),
}


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    """Normalize a vector to unit length."""
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector
    return vector / norm


def _dominant_axes_for_view(view: CameraViewSpec) -> Tuple[Tuple[str, float], Tuple[str, float]]:
    """
    Determine dominant X/Y/Z axes for horizontal and vertical screen directions.

    Args:
        view: Camera view specification with position, target, and up vector

    Returns:
        Tuple of ((horiz_axis_name, horiz_sign), (vert_axis_name, vert_sign))
        where signs indicate positive (+1) or negative (-1) direction

    Notes:
        - Computes camera right and up vectors from view geometry
        - Projects these onto world X/Y/Z axes to find dominant alignments
        - Used for labeling projection legend axes
    """
    direction = np.asarray(view.target_nm, dtype=float) - np.asarray(view.position_nm, dtype=float)
    if not np.any(direction):
        direction = np.array([0.0, 0.0, -1.0], dtype=float)
    forward = _normalize_vector(direction)

    up = np.asarray(view.up, dtype=float)
    if not np.any(up):
        up = np.array([0.0, 0.0, 1.0], dtype=float)

    right = np.cross(up, forward)
    if float(np.linalg.norm(right)) == 0.0:
        fallback = np.array([1.0, 0.0, 0.0], dtype=float)
        if np.allclose(fallback, forward):
            fallback = np.array([0.0, 1.0, 0.0], dtype=float)
        right = np.cross(fallback, forward)
    right = _normalize_vector(right)

    true_up = np.cross(forward, right)
    if float(np.linalg.norm(true_up)) == 0.0:
        true_up = np.array([0.0, 0.0, 1.0], dtype=float)
    true_up = _normalize_vector(true_up)

    horiz_name, horiz_vec = max(_AXIS_BASIS.items(), key=lambda item: abs(float(np.dot(item[1], right))))
    vert_name, vert_vec = max(_AXIS_BASIS.items(), key=lambda item: abs(float(np.dot(item[1], true_up))))

    horiz_sign = float(np.dot(horiz_vec, right))
    vert_sign = float(np.dot(vert_vec, true_up))

    return (horiz_name, horiz_sign), (vert_name, vert_sign)


def _load_font(size: int) -> ImageFont.ImageFont:
    """Load a TrueType font with fallbacks to system defaults."""
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except (OSError, IOError):
        try:
            return ImageFont.truetype("Arial.ttf", size)
        except (OSError, IOError):
            return ImageFont.load_default()


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
    """Get the bounding box size for text rendering."""
    bbox = draw.textbbox((0, 0), text, font=font)
    width = int(bbox[2] - bbox[0])
    height = int(bbox[3] - bbox[1])
    return width, height


def _draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: Tuple[float, float],
    end: Tuple[float, float],
    *,
    color: str,
    width: int,
    head: int,
) -> None:
    """
    Draw an arrow from start to end point with an arrowhead.

    Args:
        draw: PIL ImageDraw object
        start: (x, y) starting point
        end: (x, y) ending point
        color: Arrow color
        width: Line width in pixels
        head: Arrowhead size in pixels
    """
    start_pt = (int(round(start[0])), int(round(start[1])))
    end_pt = (int(round(end[0])), int(round(end[1])))
    draw.line([start_pt, end_pt], fill=color, width=width)

    angle = math.atan2(end_pt[1] - start_pt[1], end_pt[0] - start_pt[0])
    left = (
        end_pt[0] - int(round(head * math.cos(angle - math.pi / 6))),
        end_pt[1] - int(round(head * math.sin(angle - math.pi / 6))),
    )
    right = (
        end_pt[0] - int(round(head * math.cos(angle + math.pi / 6))),
        end_pt[1] - int(round(head * math.sin(angle + math.pi / 6))),
    )
    draw.line([end_pt, left], fill=color, width=width)
    draw.line([end_pt, right], fill=color, width=width)


def _nice_length_near(target_nm: float) -> Optional[float]:
    """
    Find a "nice" round number near the target length for scale bars.

    Args:
        target_nm: Target length in nanometers

    Returns:
        Nice round number close to target (e.g., 1000, 2000, 2500, 5000)
        Returns None if target is invalid (<= 0)

    Notes:
        - Uses mantissas of 1.0, 2.0, 2.5, 5.0 with powers of 10
        - Searches nearby orders of magnitude for best match
    """
    if target_nm <= 0.0:
        return None

    exponent = int(math.floor(math.log10(target_nm)))
    best = None
    best_diff = None
    mantissas = (1.0, 2.0, 2.5, 5.0)

    for exp in range(exponent - 1, exponent + 2):
        scale = 10.0 ** exp
        for mantissa in mantissas:
            candidate = mantissa * scale
            if candidate <= 0.0:
                continue
            diff = abs(candidate - target_nm)
            if best is None or diff < best_diff:
                best = candidate
                best_diff = diff

    return best


def _format_scale_label(length_nm: float) -> str:
    """
    Format a length in nanometers as a human-readable string.

    Args:
        length_nm: Length in nanometers

    Returns:
        Formatted string like "10 um", "2.5 um", "500 nm", etc.
    """
    if length_nm >= 1_000.0:
        value_um = length_nm / 1_000.0
        if abs(value_um - round(value_um)) < 1e-6:
            return f"{int(round(value_um))} um"
        return f"{value_um:.1f} um"
    if length_nm >= 100.0:
        return f"{int(round(length_nm))} nm"
    if length_nm >= 10.0:
        return f"{length_nm:.1f} nm"
    return f"{length_nm:.2f} nm"


def draw_projection_legend(
    image: Image.Image,
    view: CameraViewSpec,
    legend_config: ProjectionLegendConfig,
    nm_per_px: Optional[float],
    world_width_nm: Optional[float],
) -> Image.Image:
    """
    Draw projection legend with axis arrows and scale bar on rendered image.

    Args:
        image: PIL Image to annotate
        view: Camera view specification for determining axis orientations
        legend_config: Configuration for legend appearance
        nm_per_px: Nanometers per pixel for scale bar calculation
        world_width_nm: Total world width in view for scale bar sizing

    Returns:
        Annotated image with legend drawn

    Notes:
        - Draws orthogonal arrows showing horizontal and vertical world axes
        - Labels arrows with X/Y/Z based on camera orientation
        - Adds scale bar if nm_per_px and world_width_nm provided
        - Scale bar is sized to fit within max_fraction of image width
        - All legend elements positioned with configurable margins
    """
    if not legend_config.enabled:
        return image

    draw = ImageDraw.Draw(image)
    font = _load_font(legend_config.font_size_px)

    # Draw axis arrows and labels if enabled
    if legend_config.show_axes:
        horizontal, vertical = _dominant_axes_for_view(view)
        horiz_axis, horiz_sign = horizontal
        vert_axis, vert_sign = vertical

        horiz_dir = 1 if horiz_sign >= 0.0 else -1
        vert_dir = -1 if vert_sign >= 0.0 else 1  # negative draws upward in screen coords

        # Position base of arrows with margin
        base_x = legend_config.margin_px + (legend_config.arrow_length_px if horiz_dir == -1 else 0)
        base_y = legend_config.margin_px + (legend_config.arrow_length_px if vert_dir == -1 else 0)
        base = (base_x, base_y)

        horizontal_end = (base_x + legend_config.arrow_length_px * horiz_dir, base_y)
        vertical_end = (base_x, base_y + legend_config.arrow_length_px * vert_dir)

        # Draw axis arrows
        _draw_arrow(
            draw,
            base,
            horizontal_end,
            color=legend_config.color,
            width=legend_config.arrow_width_px,
            head=legend_config.arrow_head_px,
        )
        _draw_arrow(
            draw,
            base,
            vertical_end,
            color=legend_config.color,
            width=legend_config.arrow_width_px,
            head=legend_config.arrow_head_px,
        )

        # Draw axis labels
        horiz_label = horiz_axis
        vert_label = vert_axis

        horiz_text_w, horiz_text_h = _text_size(draw, horiz_label, font)
        if horiz_dir >= 0:
            horiz_text_x = horizontal_end[0] + legend_config.label_offset_px
        else:
            horiz_text_x = horizontal_end[0] - legend_config.label_offset_px - horiz_text_w
        horiz_text_y = horizontal_end[1] - horiz_text_h / 2.0

        vert_text_w, vert_text_h = _text_size(draw, vert_label, font)
        if vert_dir == -1:
            vert_text_y = vertical_end[1] - legend_config.label_offset_px - vert_text_h
        else:
            vert_text_y = vertical_end[1] + legend_config.label_offset_px
        vert_text_x = vertical_end[0] - vert_text_w / 2.0

        draw.text((horiz_text_x, horiz_text_y), horiz_label, fill=legend_config.color, font=font)
        draw.text((vert_text_x, vert_text_y), vert_label, fill=legend_config.color, font=font)

    # Draw scale bar if configured
    if nm_per_px is None or world_width_nm is None or not legend_config.scale_bar:
        return image

    # Compute nice scale bar length
    target_nm = world_width_nm / 5.0
    scale_nm = _nice_length_near(target_nm)
    if scale_nm is None or scale_nm <= 0.0:
        return image

    # Limit scale bar to max fraction of image width
    max_bar_px = float(image.width) * legend_config.scale_bar_max_fraction
    scale_px = scale_nm / nm_per_px
    if scale_px > max_bar_px:
        limited_nm = max_bar_px * nm_per_px
        limited_nice = _nice_length_near(limited_nm)
        if limited_nice is not None and limited_nice > 0.0:
            scale_nm = min(limited_nice, world_width_nm)
            scale_px = scale_nm / nm_per_px
        else:
            scale_px = max_bar_px
            scale_nm = scale_px * nm_per_px

    if scale_px < 10.0:
        return image

    # Position scale bar - always start at margin for consistent positioning
    bar_start_x = legend_config.margin_px
    if bar_start_x + scale_px > image.width - legend_config.margin_px:
        bar_start_x = image.width - legend_config.margin_px - scale_px
    bar_start_x = max(bar_start_x, legend_config.margin_px)

    # Position below the arrow origin if axes shown, otherwise just at margin
    if legend_config.show_axes:
        bar_y = legend_config.margin_px + legend_config.arrow_length_px + legend_config.scale_bar_label_offset_px
    else:
        bar_y = legend_config.margin_px
    max_bar_y = image.height - legend_config.margin_px - legend_config.scale_bar_height_px
    bar_y = min(bar_y, max_bar_y)

    # Draw scale bar rectangle
    bar_box = [
        (int(round(bar_start_x)), int(round(bar_y))),
        (int(round(bar_start_x + scale_px)), int(round(bar_y + legend_config.scale_bar_height_px))),
    ]
    draw.rectangle(bar_box, fill=legend_config.color)

    # Draw scale bar label
    scale_label = _format_scale_label(scale_nm)
    label_w, label_h = _text_size(draw, scale_label, font)
    label_x = bar_start_x + scale_px / 2.0 - label_w / 2.0
    label_y = bar_y + legend_config.scale_bar_height_px + legend_config.label_offset_px
    if label_y + label_h > image.height - legend_config.margin_px:
        label_y = bar_y - legend_config.label_offset_px - label_h
    draw.text((label_x, label_y), scale_label, fill=legend_config.color, font=font)

    return image


def stack_images(images: Sequence[Image.Image], orientation: str = "horizontal") -> Image.Image:
    """
    Stack multiple images vertically or horizontally into a single image.

    Args:
        images: Sequence of PIL Images to stack
        orientation: "vertical" or "horizontal"

    Returns:
        Combined image with all inputs stacked

    Raises:
        ValueError: If orientation is invalid or images sequence is empty

    Notes:
        - Vertical: images stacked top to bottom, width = max width
        - Horizontal: images stacked left to right, height = max height
        - Output has transparent background (RGBA mode)
    """
    if orientation not in {"vertical", "horizontal"}:
        raise ValueError("orientation must be 'vertical' or 'horizontal'")
    if not images:
        raise ValueError("no images to stack")

    if orientation == "vertical":
        total_w = max(im.width for im in images)
        total_h = sum(im.height for im in images)
        canvas = Image.new("RGBA", (total_w, total_h), (0, 0, 0, 0))
        y_offset = 0
        for im in images:
            # Ensure image is RGBA for alpha compositing
            if im.mode in ("RGBA", "LA"):
                im_rgba = im.convert("RGBA")
                alpha = im_rgba.getchannel("A")
                canvas.paste(im_rgba, (0, y_offset), mask=alpha)
            else:
                canvas.paste(im.convert("RGBA"), (0, y_offset))
            y_offset += im.height
    else:
        total_w = sum(im.width for im in images)
        max_h = max(im.height for im in images)
        canvas = Image.new("RGBA", (total_w, max_h), (0, 0, 0, 0))
        x_offset = 0
        for im in images:
            # Ensure image is RGBA for alpha compositing
            if im.mode in ("RGBA", "LA"):
                im_rgba = im.convert("RGBA")
                alpha = im_rgba.getchannel("A")
                canvas.paste(im_rgba, (x_offset, 0), mask=alpha)
            else:
                canvas.paste(im.convert("RGBA"), (x_offset, 0))
            x_offset += im.width

    return canvas


def create_image_grid(
    images: Sequence[Image.Image],
    rows: int,
    cols: int,
    background_color: tuple = (255, 255, 255),
) -> Image.Image:
    """
    Arrange images in a grid layout (rows x cols).
    
    Args:
        images: Sequence of PIL Images to arrange. Length must equal rows * cols.
        rows: Number of rows in the grid
        cols: Number of columns in the grid
        background_color: RGB tuple for background (default white)
    
    Returns:
        Combined image with all inputs arranged in grid
        
    Raises:
        ValueError: If number of images doesn't match rows * cols
        
    Notes:
        - Images are placed left-to-right, top-to-bottom
        - All images are resized to match the size of the first image
        - Output is in RGB mode
        
    Example:
        >>> # Create 2x3 grid (2 rows, 3 columns)
        >>> mesh_views = [front, side, top]  # 3 mesh images
        >>> em_views = [em_front, em_side, em_top]  # 3 EM images
        >>> combined = create_image_grid(mesh_views + em_views, rows=2, cols=3)
    """
    expected_count = rows * cols
    if len(images) != expected_count:
        raise ValueError(
            f"Expected {expected_count} images for {rows}x{cols} grid, got {len(images)}"
        )
    
    if not images:
        raise ValueError("no images to arrange in grid")
    
    # Use first image size as reference
    ref_width, ref_height = images[0].size
    
    # Create canvas
    canvas_width = ref_width * cols
    canvas_height = ref_height * rows
    canvas = Image.new("RGB", (canvas_width, canvas_height), background_color)
    
    # Place images in grid
    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        
        # Resize if needed to match reference size
        if img.size != (ref_width, ref_height):
            img = img.resize((ref_width, ref_height), Image.Resampling.LANCZOS)
        
        # Convert to RGB if needed
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        x_offset = col * ref_width
        y_offset = row * ref_height
        canvas.paste(img, (x_offset, y_offset))
    
    return canvas


def convert_rgba_to_rgb(image: Image.Image, background_color: tuple = (255, 255, 255)) -> Image.Image:
    """
    Convert RGBA image to RGB by compositing onto a solid background.

    Args:
        image: RGBA PIL Image
        background_color: RGB tuple for background (default white)

    Returns:
        RGB PIL Image with alpha channel flattened

    Notes:
        - Preserves original if already in RGB mode
        - Uses alpha channel for proper blending
    """
    if image.mode == "RGB":
        return image

    if image.mode == "RGBA":
        opaque = Image.new("RGB", image.size, background_color)
        opaque.paste(image, mask=image.split()[-1])
        return opaque

    return image.convert("RGB")
