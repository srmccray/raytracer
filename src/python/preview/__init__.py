"""Preview module for output and visualization.

This module handles rendering output and interactive preview:

Components:
    display: Matplotlib-based interactive preview window
    export: PNG/EXR image export utilities

Features:
    - Real-time progressive rendering preview
    - Tonemapping for HDR output (Reinhard, exposure-based)
    - Gamma-correct PNG export (sRGB)
    - Side-by-side comparison visualization

The preview window shows the accumulated render buffer and
updates progressively as samples are added. Tone mapping
allows HDR content to be displayed on standard monitors.

Example:
    >>> from src.python.preview import show_preview, save_png
    >>> from src.python.core.progressive import ProgressiveRenderer
    >>>
    >>> renderer = ProgressiveRenderer(512, 512)
    >>> renderer.render(100)
    >>> show_preview(renderer, tone_map="reinhard")
    >>> save_png(renderer, "output.png", gamma=2.2)
"""

from src.python.preview.display import (
    ToneMapMethod,
    apply_gamma,
    process_image_for_display,
    show_comparison,
    show_preview,
    tone_map_exposure,
    tone_map_reinhard,
)
from src.python.preview.export import (
    compute_rmse,
    image_to_uint8,
    save_png,
    save_png_from_array,
)

__all__ = [
    # Display functions
    "show_preview",
    "show_comparison",
    # Tone mapping
    "tone_map_reinhard",
    "tone_map_exposure",
    "apply_gamma",
    "process_image_for_display",
    "ToneMapMethod",
    # Export functions
    "save_png",
    "save_png_from_array",
    "image_to_uint8",
    "compute_rmse",
]
