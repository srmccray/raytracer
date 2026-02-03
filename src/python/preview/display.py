"""Matplotlib-based preview display for rendered images.

This module provides functions for displaying rendered images using Matplotlib,
with support for tone mapping and gamma correction.

Features:
    - Interactive preview window
    - Tone mapping (Reinhard, exposure-based)
    - Gamma correction (sRGB 2.2)
    - Sample count display

Example:
    >>> from src.python.preview.display import show_preview
    >>> from src.python.core.progressive import ProgressiveRenderer
    >>>
    >>> renderer = ProgressiveRenderer(512, 512)
    >>> renderer.render(100)
    >>> show_preview(renderer, tone_map="reinhard")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from src.python.core.progressive import ProgressiveRenderer


# Type alias for tone mapping options
ToneMapMethod = Literal["none", "reinhard", "exposure"]


def tone_map_reinhard(
    image: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Apply Reinhard tone mapping: L / (1 + L).

    Simple global tone mapping operator that compresses HDR values
    into the displayable [0, 1] range.

    Args:
        image: Linear HDR image array of shape (H, W, 3).

    Returns:
        Tone mapped image in [0, 1] range.
    """
    # Ensure non-negative values
    image = np.maximum(image, 0.0)

    # Reinhard: c / (1 + c)
    result = image / (1.0 + image)

    return result.astype(np.float32)


def tone_map_exposure(
    image: npt.NDArray[np.float32],
    exposure: float = 1.0,
) -> npt.NDArray[np.float32]:
    """Apply exposure-based tone mapping: 1 - exp(-c * exposure).

    Provides control over image brightness through the exposure parameter.

    Args:
        image: Linear HDR image array of shape (H, W, 3).
        exposure: Exposure value (default 1.0). Higher values brighten the image.

    Returns:
        Tone mapped image in [0, 1] range.
    """
    # Ensure non-negative values
    image = np.maximum(image, 0.0)

    # Exposure: 1 - exp(-c * exposure)
    result = 1.0 - np.exp(-image * exposure)

    return result.astype(np.float32)


def apply_gamma(
    image: npt.NDArray[np.float32],
    gamma: float = 2.2,
) -> npt.NDArray[np.float32]:
    """Apply gamma correction for display.

    Converts linear values to sRGB gamma space for correct display
    on standard monitors.

    Args:
        image: Linear image array of shape (H, W, 3) in [0, 1] range.
        gamma: Gamma value (default 2.2 for sRGB).

    Returns:
        Gamma corrected image.
    """
    if gamma == 1.0:
        return image

    # Clamp to [0, 1] before gamma to avoid NaN from negative values
    image = np.clip(image, 0.0, 1.0)

    # Apply gamma encoding: out = in^(1/gamma)
    result = np.power(image, 1.0 / gamma)

    return result.astype(np.float32)


def process_image_for_display(
    image: npt.NDArray[np.float32],
    tone_map: ToneMapMethod = "none",
    gamma: float = 2.2,
    exposure: float = 1.0,
) -> npt.NDArray[np.float32]:
    """Process an image for display with tone mapping and gamma correction.

    Applies the full display pipeline:
    1. Tone mapping (optional, for HDR content)
    2. Gamma correction (for sRGB display)
    3. Clamping to [0, 1]

    Args:
        image: Linear HDR image array of shape (H, W, 3).
        tone_map: Tone mapping method ("none", "reinhard", or "exposure").
        gamma: Gamma correction value (default 2.2 for sRGB).
        exposure: Exposure value for exposure tone mapping (default 1.0).

    Returns:
        Processed image ready for display, in [0, 1] range.
    """
    result = image.copy()

    # Apply tone mapping
    if tone_map == "reinhard":
        result = tone_map_reinhard(result)
    elif tone_map == "exposure":
        result = tone_map_exposure(result, exposure)
    elif tone_map != "none":
        raise ValueError(f"Unknown tone mapping method: {tone_map}")

    # Apply gamma correction
    result = apply_gamma(result, gamma)

    # Final clamp
    result = np.clip(result, 0.0, 1.0)

    return result.astype(np.float32)


def show_preview(
    renderer: ProgressiveRenderer,
    *,
    tone_map: ToneMapMethod = "none",
    gamma: float = 2.2,
    exposure: float = 1.0,
    title: str | None = None,
    figsize: tuple[float, float] = (8, 8),
    block: bool = True,
) -> None:
    """Display the current render as a Matplotlib figure.

    Shows the rendered image with optional tone mapping and gamma correction.
    The sample count is displayed in the title.

    Args:
        renderer: The ProgressiveRenderer instance to display.
        tone_map: Tone mapping method ("none", "reinhard", or "exposure").
        gamma: Gamma correction value (default 2.2 for sRGB).
        exposure: Exposure value for exposure tone mapping (default 1.0).
        title: Custom title (default shows sample count).
        figsize: Figure size in inches (width, height).
        block: Whether to block execution until figure is closed.

    Example:
        >>> renderer = ProgressiveRenderer(512, 512)
        >>> renderer.render(100)
        >>> show_preview(renderer, tone_map="reinhard")
    """
    import matplotlib.pyplot as plt

    # Get linear image from renderer (gamma=1.0 for linear)
    image = renderer.get_image_numpy(gamma=1.0)

    # Process for display
    display_image = process_image_for_display(
        image,
        tone_map=tone_map,
        gamma=gamma,
        exposure=exposure,
    )

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Display image
    ax.imshow(display_image)
    ax.axis("off")

    # Set title
    if title is None:
        spp = renderer.sample_count
        title_text = f"Render Preview - {spp} SPP"
        if tone_map != "none":
            title_text += f" ({tone_map})"
    else:
        title_text = title

    ax.set_title(title_text)

    plt.tight_layout()
    plt.show(block=block)


def show_comparison(
    image_a: npt.NDArray[np.float32],
    image_b: npt.NDArray[np.float32],
    *,
    labels: tuple[str, str] = ("A", "B"),
    tone_map: ToneMapMethod = "none",
    gamma: float = 2.2,
    diff_scale: float = 10.0,
    figsize: tuple[float, float] = (16, 6),
    block: bool = True,
) -> float:
    """Display side-by-side comparison of two images with difference view.

    Shows two images and their amplified difference, along with RMSE metric.

    Args:
        image_a: First image array (H, W, 3) in linear space.
        image_b: Second image array (H, W, 3) in linear space.
        labels: Labels for the two images.
        tone_map: Tone mapping method to apply.
        gamma: Gamma correction value.
        diff_scale: Scale factor for difference amplification.
        figsize: Figure size in inches.
        block: Whether to block execution until figure is closed.

    Returns:
        RMSE (root mean squared error) between the two images.
    """
    import matplotlib.pyplot as plt

    # Process images for display
    display_a = process_image_for_display(image_a, tone_map=tone_map, gamma=gamma)
    display_b = process_image_for_display(image_b, tone_map=tone_map, gamma=gamma)

    # Compute RMSE in display space
    diff = display_a.astype(np.float64) - display_b.astype(np.float64)
    rmse = float(np.sqrt(np.mean(diff**2)))

    # Create amplified difference visualization
    diff_abs = np.abs(diff)
    diff_amplified = np.clip(diff_abs * diff_scale, 0.0, 1.0)

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    axes[0].imshow(display_a)
    axes[0].set_title(labels[0])
    axes[0].axis("off")

    axes[1].imshow(display_b)
    axes[1].set_title(labels[1])
    axes[1].axis("off")

    axes[2].imshow(diff_amplified)
    axes[2].set_title(f"Difference ({diff_scale}x) - RMSE: {rmse:.6f}")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show(block=block)

    return rmse
