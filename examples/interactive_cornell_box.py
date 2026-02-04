#!/usr/bin/env python3
"""Interactive Cornell box renderer with real-time parameter controls.

This script launches an interactive preview window for the Cornell box scene
with GPU-accelerated progressive rendering and real-time parameter adjustment.

Usage:
    python -m examples.interactive_cornell_box

Features:
    - Real-time progressive rendering with sample accumulation
    - Interactive sliders for light intensity and color
    - Wall color controls (left, right, back walls)
    - PNG export with timestamp
    - Automatic scene rebuild on parameter changes

Controls:
    - Light Intensity: Adjust brightness (0-50)
    - Light Color R/G/B: Adjust light color components
    - Wall Colors: Adjust left (green), right (red), back (white) walls
    - Export PNG: Save current render with timestamp

The renderer uses 1 sample per frame for responsive UI interaction.
Sample count accumulates continuously until parameters change.
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path

# Ensure the project root is in the Python path for direct execution
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import taichi as ti  # noqa: E402


def initialize_taichi() -> str:
    """Initialize Taichi with the best available backend.

    On macOS, prefers Metal. Falls back to CPU if GPU is unavailable.

    Returns:
        Name of the backend being used.
    """
    system = platform.system()

    if system == "Darwin":
        # macOS: prefer Metal
        try:
            ti.init(arch=ti.metal)
            return "Metal (GPU)"
        except Exception:
            pass

    # Try generic GPU (CUDA on Linux/Windows, Vulkan as fallback)
    try:
        ti.init(arch=ti.gpu)
        return "GPU"
    except Exception:
        pass

    # Fall back to CPU
    ti.init(arch=ti.cpu)
    return "CPU"


def main() -> int:
    """Main entry point for the interactive Cornell box renderer.

    Returns:
        Exit code (0 for success, non-zero for error).
    """
    # Initialize Taichi first (before importing modules that use ti.kernel)
    backend = initialize_taichi()
    print(f"Taichi backend: {backend}")

    # Import after Taichi initialization
    from src.python.preview.interactive import InteractivePreview
    from src.python.scene.cornell_box import CornellBoxParams

    # Check if display is available
    if not InteractivePreview.is_display_available():
        print("Error: No display available. Cannot run interactive preview.")
        print("This script requires a graphical display environment.")
        return 1

    # Create the interactive preview window
    print("Creating interactive preview window (512x512)...")
    preview = InteractivePreview(512, 512)

    # Set initial scene parameters
    params = CornellBoxParams(
        light_intensity=15.0,
        light_color=(1.0, 1.0, 1.0),
    )
    preview.set_params(params)

    # Run the reactive rendering loop
    print("Starting interactive rendering...")
    print("  - Adjust sliders to modify scene parameters")
    print("  - Click 'Export PNG' to save current render")
    print("  - Close window to exit")
    print()

    try:
        preview.run_reactive()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        preview.close()
        print("Preview window closed.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
