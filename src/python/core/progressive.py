"""Progressive renderer for iterative sample accumulation.

This module provides a convenient wrapper around the core integrator that supports:
- Progressive rendering that refines over time
- Batch rendering (multiple SPP in one call)
- Progress callbacks for UI updates
- Easy reset and re-render functionality

The ProgressiveRenderer class encapsulates the render target state and provides
a clean interface for interactive rendering workflows.

Example:
    >>> import taichi as ti
    >>> ti.init(arch=ti.gpu)
    >>> from src.python.core.progressive import ProgressiveRenderer
    >>> from src.python.scene.cornell_box import create_cornell_box_scene
    >>> from src.python.camera.pinhole import setup_camera
    >>>
    >>> scene, camera = create_cornell_box_scene()
    >>> setup_camera(camera)
    >>>
    >>> renderer = ProgressiveRenderer(512, 512)
    >>> renderer.render(100)  # Render 100 SPP
    >>> image = renderer.get_image_numpy()
"""

from collections.abc import Callable, Generator
from typing import Any

import numpy as np
import numpy.typing as npt

from src.python.core.integrator import (
    clear_render_target,
    get_image,
    get_normalized_image_numpy,
    get_total_samples,
    render_image,
    setup_render_target,
)

# Type alias for progress callback
# Callback receives (current_samples, total_target_samples)
ProgressCallback = Callable[[int, int], None]


class ProgressiveRenderer:
    """A progressive renderer that accumulates samples over time.

    This class wraps the core integrator functions to provide a convenient
    interface for progressive rendering with support for:
    - Incremental sample accumulation
    - Batch rendering (multiple SPP per call)
    - Progress callbacks
    - Reset functionality

    The renderer maintains its own state for width/height and delegates
    to the global integrator buffers (which are Taichi fields).

    Attributes:
        width: Image width in pixels.
        height: Image height in pixels.
    """

    def __init__(self, width: int, height: int) -> None:
        """Initialize the progressive renderer.

        Args:
            width: Image width in pixels (max 2048).
            height: Image height in pixels (max 2048).

        Raises:
            ValueError: If dimensions exceed maximum supported size.
        """
        self._width = width
        self._height = height
        setup_render_target(width, height)

    @property
    def width(self) -> int:
        """Get the image width."""
        return self._width

    @property
    def height(self) -> int:
        """Get the image height."""
        return self._height

    @property
    def sample_count(self) -> int:
        """Get the current number of accumulated samples per pixel."""
        return get_total_samples()

    def reset(self) -> None:
        """Reset the accumulator for a new render.

        Clears the color buffer and sample count, allowing a fresh render
        without changing the image dimensions.
        """
        clear_render_target()

    def resize(self, width: int, height: int) -> None:
        """Resize the render target and reset accumulator.

        Args:
            width: New image width in pixels.
            height: New image height in pixels.

        Raises:
            ValueError: If dimensions exceed maximum supported size.
        """
        self._width = width
        self._height = height
        setup_render_target(width, height)

    def render(
        self,
        num_samples: int = 1,
        batch_size: int = 1,
        callback: ProgressCallback | None = None,
    ) -> None:
        """Render samples progressively with optional progress callback.

        Accumulates the specified number of samples into the existing buffer.
        Can be called multiple times to continue refining the image.

        Args:
            num_samples: Total number of samples to add.
            batch_size: Number of samples to render before each callback.
                A larger batch size reduces callback overhead but provides
                less frequent updates.
            callback: Optional callback function called after each batch.
                Receives (current_total_samples, target_total_samples).

        Example:
            >>> def progress(current, target):
            ...     print(f"Progress: {current}/{target} samples")
            >>> renderer.render(100, batch_size=10, callback=progress)
        """
        if num_samples <= 0:
            return

        # Calculate target sample count
        start_samples = self.sample_count
        target_samples = start_samples + num_samples

        # Render in batches
        remaining = num_samples
        while remaining > 0:
            batch = min(batch_size, remaining)
            render_image(batch)
            remaining -= batch

            if callback is not None:
                callback(self.sample_count, target_samples)

    def render_progressive(
        self,
        num_samples: int = 1,
        batch_size: int = 1,
    ) -> Generator[tuple[int, int], None, None]:
        """Render samples progressively, yielding progress after each batch.

        This is a generator-based alternative to render() with callbacks,
        useful for integration with asyncio or iterative processing.

        Args:
            num_samples: Total number of samples to add.
            batch_size: Number of samples to render before each yield.

        Yields:
            Tuple of (current_total_samples, target_total_samples).

        Example:
            >>> for current, target in renderer.render_progressive(100, batch_size=10):
            ...     print(f"Progress: {current}/{target} samples")
            ...     # Could update UI, check for cancellation, etc.
        """
        if num_samples <= 0:
            return

        start_samples = self.sample_count
        target_samples = start_samples + num_samples

        remaining = num_samples
        while remaining > 0:
            batch = min(batch_size, remaining)
            render_image(batch)
            remaining -= batch
            yield (self.sample_count, target_samples)

    def get_image(self) -> Any:
        """Get the raw Taichi color buffer field.

        Note: This returns the full preallocated buffer. Use width/height
        properties to determine the active region.

        Returns:
            Taichi VectorField containing the accumulated RGB image.
        """
        return get_image()

    def get_image_numpy(self, gamma: float = 1.0) -> npt.NDArray[np.float32]:
        """Get the rendered image as a NumPy array.

        Returns the color buffer with values clamped to [0, 1] and optionally
        gamma corrected. The array shape is (height, width, 3).

        Args:
            gamma: Gamma correction value. Default 1.0 (linear).
                Use 2.2 for sRGB display.

        Returns:
            NumPy array of shape (height, width, 3) with dtype float32.
        """
        image = get_normalized_image_numpy()

        if gamma != 1.0:
            image = np.power(image, 1.0 / gamma)

        return image

    def get_image_uint8(self, gamma: float = 2.2) -> npt.NDArray[np.uint8]:
        """Get the rendered image as an 8-bit NumPy array.

        Applies gamma correction and converts to uint8 format suitable
        for display or saving.

        Args:
            gamma: Gamma correction value. Default 2.2 for sRGB.

        Returns:
            NumPy array of shape (height, width, 3) with dtype uint8.
        """
        image = self.get_image_numpy(gamma=gamma)
        return (image * 255).astype(np.uint8)

    def save_image(self, filepath: str, gamma: float = 2.2) -> None:
        """Save the rendered image to a file.

        Args:
            filepath: Path to save the image (e.g., "output.png").
            gamma: Gamma correction value. Default 2.2 for sRGB.
        """
        from PIL import Image as PILImage

        image_uint8 = self.get_image_uint8(gamma=gamma)
        pil_image = PILImage.fromarray(image_uint8, mode="RGB")
        pil_image.save(filepath)

    def __repr__(self) -> str:
        """Return a string representation of the renderer state."""
        return (
            f"ProgressiveRenderer(width={self.width}, height={self.height}, "
            f"samples={self.sample_count})"
        )
