#!/usr/bin/env python3
"""Render the Cornell box scene.

This script demonstrates end-to-end rendering of the classic Cornell box scene
using the Beads raytracer. It creates the scene, sets up the camera, configures
lighting, and renders with progressive refinement.

Usage:
    python -m examples.render_cornell_box [options]

Options:
    --width WIDTH       Image width in pixels (default: 512)
    --height HEIGHT     Image height in pixels (default: 512)
    --samples SAMPLES   Number of samples per pixel (default: 100)
    --output OUTPUT     Output file path (default: cornell_box.png)
    --batch-size SIZE   Samples per progress update (default: 10)
    --quiet             Suppress progress output

Example:
    python -m examples.render_cornell_box --width 256 --height 256 --samples 50
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import taichi as ti


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Render the Cornell box scene.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Image width in pixels (default: 512)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Image height in pixels (default: 512)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Number of samples per pixel (default: 100)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="cornell_box.png",
        help="Output file path (default: cornell_box.png)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Samples per progress update (default: 10)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    return parser.parse_args()


def render_cornell_box(
    width: int = 512,
    height: int = 512,
    num_samples: int = 100,
    output_path: str = "cornell_box.png",
    batch_size: int = 10,
    quiet: bool = False,
) -> Path:
    """Render the Cornell box scene and save to file.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        num_samples: Number of samples per pixel.
        output_path: Output file path (PNG).
        batch_size: Number of samples to render between progress updates.
        quiet: If True, suppress progress output.

    Returns:
        Path to the saved image file.
    """
    # Lazy imports to allow Taichi initialization first
    from src.python.camera.pinhole import setup_camera
    from src.python.core.integrator import setup_light
    from src.python.core.progressive import ProgressiveRenderer
    from src.python.preview.export import save_png
    from src.python.scene.cornell_box import create_cornell_box_scene, get_light_quad_info

    # Create the Cornell box scene
    if not quiet:
        print(f"Creating Cornell box scene ({width}x{height})...")

    scene, camera = create_cornell_box_scene()

    # Setup camera
    setup_camera(camera)

    # Configure the area light
    light_info = get_light_quad_info()
    # Material ID 3 is the light material in Cornell box
    # (after red=0, green=1, white=2, light=3)
    setup_light(
        corner=light_info["corner"],
        edge_u=light_info["edge_u"],
        edge_v=light_info["edge_v"],
        emission=(15.0, 15.0, 15.0),
        material_id=3,
    )

    # Create progressive renderer
    renderer = ProgressiveRenderer(width, height)

    # Render with progress callback
    if not quiet:
        print(f"Rendering {num_samples} samples per pixel...")

    start_time = time.time()

    def progress_callback(current: int, target: int) -> None:
        if not quiet:
            elapsed = time.time() - start_time
            progress_pct = (current / target) * 100 if target > 0 else 0
            samples_per_sec = current / elapsed if elapsed > 0 else 0
            print(
                f"\r  Progress: {current}/{target} samples "
                f"({progress_pct:.1f}%) - {samples_per_sec:.1f} spp/s",
                end="",
                flush=True,
            )

    renderer.render(
        num_samples=num_samples,
        batch_size=batch_size,
        callback=progress_callback,
    )

    if not quiet:
        print()  # Newline after progress

    # Save the image
    output_file = Path(output_path)
    save_png(
        renderer,
        str(output_file),
        tone_map="reinhard",
        gamma=2.2,
        exposure=1.0,
    )

    total_time = time.time() - start_time
    if not quiet:
        print(f"Saved to: {output_file.absolute()}")
        print(f"Total time: {total_time:.2f}s")

    return output_file


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Initialize Taichi
    # Use GPU if available, fall back to CPU
    try:
        ti.init(arch=ti.gpu)
        if not args.quiet:
            print("Using GPU backend")
    except Exception:
        ti.init(arch=ti.cpu)
        if not args.quiet:
            print("Using CPU backend")

    try:
        render_cornell_box(
            width=args.width,
            height=args.height,
            num_samples=args.samples,
            output_path=args.output,
            batch_size=args.batch_size,
            quiet=args.quiet,
        )
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
