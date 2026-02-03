"""Image export utilities for rendered images.

This module provides functions for saving rendered images to files
with support for tone mapping and gamma correction.

Supported formats:
    - PNG (8-bit sRGB via Pillow)

Future formats:
    - EXR (32-bit HDR via imageio/OpenEXR)

Example:
    >>> from src.python.preview.export import save_png
    >>> from src.python.core.progressive import ProgressiveRenderer
    >>>
    >>> renderer = ProgressiveRenderer(512, 512)
    >>> renderer.render(100)
    >>> save_png(renderer, "output.png", tone_map="reinhard")
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from PIL import Image as PILImage

from src.python.preview.display import ToneMapMethod, process_image_for_display

if TYPE_CHECKING:
    from src.python.core.progressive import ProgressiveRenderer


def save_png(
    renderer: ProgressiveRenderer,
    filepath: str,
    *,
    tone_map: ToneMapMethod = "none",
    gamma: float = 2.2,
    exposure: float = 1.0,
) -> None:
    """Save the rendered image as a PNG file.

    Applies tone mapping and gamma correction before saving.
    The output is an 8-bit sRGB PNG.

    Args:
        renderer: The ProgressiveRenderer instance to save.
        filepath: Output file path (should end in .png).
        tone_map: Tone mapping method ("none", "reinhard", or "exposure").
        gamma: Gamma correction value (default 2.2 for sRGB).
        exposure: Exposure value for exposure tone mapping (default 1.0).

    Example:
        >>> renderer = ProgressiveRenderer(512, 512)
        >>> renderer.render(100)
        >>> save_png(renderer, "output.png", tone_map="reinhard", gamma=2.2)
    """
    # Get linear image from renderer
    image = renderer.get_image_numpy(gamma=1.0)

    # Process for output
    processed = process_image_for_display(
        image,
        tone_map=tone_map,
        gamma=gamma,
        exposure=exposure,
    )

    # Convert to 8-bit
    image_uint8 = (processed * 255).astype(np.uint8)

    # Save using Pillow
    pil_image = PILImage.fromarray(image_uint8, mode="RGB")
    pil_image.save(filepath)


def save_png_from_array(
    image: npt.NDArray[np.float32],
    filepath: str,
    *,
    tone_map: ToneMapMethod = "none",
    gamma: float = 2.2,
    exposure: float = 1.0,
) -> None:
    """Save a NumPy array as a PNG file.

    Applies tone mapping and gamma correction before saving.
    The output is an 8-bit sRGB PNG.

    Args:
        image: Linear HDR image array of shape (H, W, 3).
        filepath: Output file path (should end in .png).
        tone_map: Tone mapping method ("none", "reinhard", or "exposure").
        gamma: Gamma correction value (default 2.2 for sRGB).
        exposure: Exposure value for exposure tone mapping (default 1.0).
    """
    # Process for output
    processed = process_image_for_display(
        image,
        tone_map=tone_map,
        gamma=gamma,
        exposure=exposure,
    )

    # Convert to 8-bit
    image_uint8 = (processed * 255).astype(np.uint8)

    # Save using Pillow
    pil_image = PILImage.fromarray(image_uint8, mode="RGB")
    pil_image.save(filepath)


def image_to_uint8(
    image: npt.NDArray[np.float32],
    *,
    tone_map: ToneMapMethod = "none",
    gamma: float = 2.2,
    exposure: float = 1.0,
) -> npt.NDArray[np.uint8]:
    """Convert a linear float32 image to uint8 for display/export.

    Applies tone mapping and gamma correction.

    Args:
        image: Linear HDR image array of shape (H, W, 3).
        tone_map: Tone mapping method ("none", "reinhard", or "exposure").
        gamma: Gamma correction value (default 2.2 for sRGB).
        exposure: Exposure value for exposure tone mapping (default 1.0).

    Returns:
        8-bit image array of shape (H, W, 3) with dtype uint8.
    """
    # Process for display/output
    processed = process_image_for_display(
        image,
        tone_map=tone_map,
        gamma=gamma,
        exposure=exposure,
    )

    # Convert to 8-bit
    return (processed * 255).astype(np.uint8)


def compute_rmse(
    image_a: npt.NDArray[np.floating[npt.NBitBase]],
    image_b: npt.NDArray[np.floating[npt.NBitBase]],
) -> float:
    """Compute root mean squared error between two images.

    Args:
        image_a: First image array.
        image_b: Second image array (must have same shape as image_a).

    Returns:
        RMSE value (lower is more similar).

    Raises:
        ValueError: If image shapes don't match.
    """
    if image_a.shape != image_b.shape:
        raise ValueError(
            f"Image shapes must match: {image_a.shape} vs {image_b.shape}"
        )

    diff = image_a.astype(np.float64) - image_b.astype(np.float64)
    return float(np.sqrt(np.mean(diff**2)))
