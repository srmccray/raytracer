"""Interactive preview window using Taichi GGUI.

This module provides an interactive preview window for real-time rendering
using Taichi's ti.ui.Window and canvas system.

Features:
    - Real-time progressive rendering display
    - Taichi GGUI-based window (GPU-accelerated)
    - Support for updating display from numpy arrays
    - Basic window event loop handling
    - Reactive rendering with parameter change detection
    - Continuous rendering until window closed

Example:
    >>> import numpy as np
    >>> from src.python.preview.interactive import InteractivePreview
    >>>
    >>> preview = InteractivePreview(512, 512)
    >>> image = np.zeros((512, 512, 3), dtype=np.float32)
    >>> preview.update_image(image)
    >>> preview.run()

Reactive Rendering Example:
    >>> from src.python.preview.interactive import InteractivePreview
    >>> from src.python.scene.cornell_box import CornellBoxParams
    >>>
    >>> preview = InteractivePreview(512, 512)
    >>> params = CornellBoxParams(light_intensity=15.0)
    >>> preview.set_params(params)
    >>> preview.run_reactive()  # Renders continuously until window closed
"""

from __future__ import annotations

import copy
import os
from datetime import datetime
from typing import TYPE_CHECKING, Any

import numpy as np
import taichi as ti

if TYPE_CHECKING:
    import numpy.typing as npt

    from src.python.scene.cornell_box import CornellBoxParams


# Lazy kernel holder - kernel is created on first use after Taichi is initialized
_copy_field_kernel: Any = None


def _get_copy_field_kernel() -> Any:
    """Get or create the field copy kernel.

    The kernel is created lazily to ensure Taichi is initialized first.
    """
    global _copy_field_kernel
    if _copy_field_kernel is None:

        @ti.kernel
        def _kernel(src: ti.template(), dst: ti.template()) -> None:
            for i, j in src:
                dst[i, j] = src[i, j]

        _copy_field_kernel = _kernel
    return _copy_field_kernel


class InteractivePreview:
    """Interactive preview window using Taichi GGUI.

    This class wraps ti.ui.Window to provide a simple interface for
    displaying rendered images in real-time. It manages the window,
    canvas, and display buffer.

    Attributes:
        width: Window width in pixels.
        height: Window height in pixels.
        window: The Taichi GGUI window instance.
        canvas: The canvas for rendering.
        display_image: Taichi field storing the display image (RGB float).

    Example:
        >>> preview = InteractivePreview(800, 600)
        >>> preview.update_image(my_numpy_array)
        >>> preview.run()
    """

    def __init__(
        self,
        width: int,
        height: int,
        *,
        title: str = "Cornell Box - Interactive Preview",
    ) -> None:
        """Initialize the interactive preview window.

        Args:
            width: Window width in pixels.
            height: Window height in pixels.
            title: Window title (default: "Cornell Box - Interactive Preview").

        Note:
            This will initialize Taichi if not already initialized.
            The window is created but not shown until run() is called.
        """
        self.width = width
        self.height = height
        self._title = title
        self._is_initialized = False

        # Defer window creation until run() to support headless checks
        self._window: ti.ui.Window | None = None
        self._canvas: ti.ui.Canvas | None = None

        # Create the display image field
        # Shape is (width, height) for Taichi field, RGB values stored as vec3
        self.display_image: ti.MatrixField = ti.Vector.field(
            3, dtype=ti.f32, shape=(width, height)
        )

    def _initialize_window(self) -> None:
        """Initialize the Taichi GGUI window and canvas.

        This is called lazily to allow headless operation checks.
        """
        if self._is_initialized:
            return

        self._window = ti.ui.Window(
            name=self._title,
            res=(self.width, self.height),
            vsync=True,
        )
        self._canvas = self._window.get_canvas()
        self._is_initialized = True

    @property
    def window(self) -> ti.ui.Window:
        """Get the Taichi GGUI window, initializing if needed."""
        if self._window is None:
            self._initialize_window()
        assert self._window is not None
        return self._window

    @property
    def canvas(self) -> ti.ui.Canvas:
        """Get the canvas for rendering."""
        if self._canvas is None:
            self._initialize_window()
        assert self._canvas is not None
        return self._canvas

    def update_image(self, image: npt.NDArray[np.float32]) -> None:
        """Update the display image from a numpy array.

        The image should be in linear RGB space with values in [0, 1].
        Tone mapping and gamma correction should be applied before calling
        this method if needed.

        Args:
            image: NumPy array of shape (height, width, 3) with dtype float32.
                   Values should be in [0, 1] range for proper display.

        Raises:
            ValueError: If image shape doesn't match (height, width, 3).
        """
        expected_shape = (self.height, self.width, 3)
        if image.shape != expected_shape:
            raise ValueError(
                f"Image shape {image.shape} doesn't match expected {expected_shape}"
            )

        # Taichi fields use (x, y) indexing which corresponds to (width, height)
        # NumPy images are (height, width, channels), so we need to transpose
        # Also flip Y axis as Taichi has origin at bottom-left
        image_transposed = np.ascontiguousarray(
            np.transpose(np.flipud(image), (1, 0, 2))
        )
        self.display_image.from_numpy(image_transposed)

    def update_image_from_field(self, field: ti.MatrixField) -> None:
        """Update the display image from a Taichi field.

        This is more efficient than update_image() as it avoids
        CPU-GPU data transfer when the source is already on GPU.

        Args:
            field: Taichi Vector.field of shape (width, height) with 3 components.
        """
        # Use lazy-initialized kernel for the copy
        kernel = _get_copy_field_kernel()
        kernel(field, self.display_image)

    def is_running(self) -> bool:
        """Check if the window is still open.

        Returns:
            True if the window is running, False if it should close.
        """
        return self.window.running

    def show_frame(self) -> None:
        """Display a single frame.

        This updates the canvas with the current display image and
        presents the frame. Call this in a loop for continuous updates.
        """
        self.canvas.set_image(self.display_image)
        self.window.show()

    def run(self) -> None:
        """Run the main window event loop.

        This blocks until the window is closed. The display image
        is shown on each frame.

        Use this for simple preview without external rendering loop.
        For integration with a renderer, use is_running() and
        show_frame() directly in your own loop.
        """
        self._initialize_window()

        while self.is_running():
            self.show_frame()

    def close(self) -> None:
        """Close the preview window.

        After calling this, the window cannot be reopened.
        """
        if self._window is not None:
            # Taichi windows close automatically when the reference is dropped
            # Set running to false to exit any active loop
            self._window.running = False

    @staticmethod
    def is_display_available() -> bool:
        """Check if a display is available for GUI rendering.

        Returns:
            True if a display is available, False for headless environments.
        """
        # Check common environment variables for headless detection
        display = os.environ.get("DISPLAY")
        wayland = os.environ.get("WAYLAND_DISPLAY")

        # On macOS, display is always available if not in SSH
        if os.uname().sysname == "Darwin":
            # Check if we're in an SSH session without X forwarding
            ssh_connection = os.environ.get("SSH_CONNECTION")
            if ssh_connection and not display:
                return False
            return True

        # On Linux, check for X11 or Wayland
        if display or wayland:
            return True

        # Windows generally always has display
        if os.name == "nt":
            return True

        return False

    # =========================================================================
    # Reactive Rendering Support
    # =========================================================================

    def set_params(self, params: CornellBoxParams) -> None:
        """Set the scene parameters for reactive rendering.

        When parameters change (detected by comparing to previous params),
        the scene will be rebuilt and the accumulator reset on the next
        render frame.

        Args:
            params: The CornellBoxParams to use for scene configuration.

        Example:
            >>> preview = InteractivePreview(512, 512)
            >>> params = CornellBoxParams(light_intensity=20.0)
            >>> preview.set_params(params)
        """
        # Store a deep copy to prevent external mutation
        self._pending_params: CornellBoxParams = copy.deepcopy(params)

    def _params_changed(self) -> bool:
        """Check if parameters have changed since last scene build.

        Returns:
            True if params have changed or if this is the first build.
        """
        if not hasattr(self, "_pending_params"):
            return False

        if not hasattr(self, "_current_params"):
            return True

        # Compare all fields of the dataclass
        return (
            self._pending_params.light_intensity != self._current_params.light_intensity
            or self._pending_params.light_color != self._current_params.light_color
            or self._pending_params.left_wall_color != self._current_params.left_wall_color
            or self._pending_params.right_wall_color != self._current_params.right_wall_color
            or self._pending_params.back_wall_color != self._current_params.back_wall_color
        )

    def _rebuild_scene(self) -> None:
        """Rebuild the scene from current parameters.

        This creates a new scene using create_cornell_box_scene with the
        current parameters, sets up the camera and light, and resets the
        progressive renderer accumulator.
        """
        # Import here to avoid circular imports
        from src.python.camera.pinhole import setup_camera
        from src.python.core.integrator import setup_light
        from src.python.scene.cornell_box import (
            CornellBoxParams,
            create_cornell_box_scene,
            get_light_quad_info,
        )

        # Use pending params or default
        if hasattr(self, "_pending_params"):
            params = self._pending_params
        else:
            params = CornellBoxParams()

        # Create new scene with params
        scene, camera, light_mat_id = create_cornell_box_scene(params=params)

        # Setup camera for ray generation
        setup_camera(camera)

        # Setup light for emission
        light_info = get_light_quad_info()
        emission = (
            params.light_color[0] * params.light_intensity,
            params.light_color[1] * params.light_intensity,
            params.light_color[2] * params.light_intensity,
        )
        setup_light(
            corner=light_info["corner"],
            edge_u=light_info["edge_u"],
            edge_v=light_info["edge_v"],
            emission=emission,
            material_id=light_mat_id,
        )

        # Store references
        self._scene = scene
        self._camera = camera
        self._light_mat_id = light_mat_id

        # Update current params (deep copy)
        self._current_params = copy.deepcopy(params)

        # Reset the progressive renderer if it exists
        if hasattr(self, "_renderer"):
            self._renderer.reset()

    def _ensure_renderer(self) -> None:
        """Ensure the progressive renderer is initialized."""
        from src.python.core.progressive import ProgressiveRenderer

        if not hasattr(self, "_renderer"):
            self._renderer = ProgressiveRenderer(self.width, self.height)

    def run_reactive(self) -> None:
        """Run the reactive rendering loop.

        This is the main entry point for interactive rendering. It:
        1. Initializes the window and renderer
        2. On each frame:
           - Reads slider values from GUI controls
           - Checks for parameter changes (including from sliders)
           - Rebuilds scene if needed (resets accumulator)
           - Renders 1 sample per pixel
           - Updates the display with tone-mapped result
        3. Continues until the window is closed

        The loop renders 1 SPP per frame for responsive UI. Sample count
        is tracked and can be displayed.

        GUI Controls:
            - Light Intensity slider (0.0 to 50.0)
            - Light Color R/G/B sliders (0.0 to 1.0 each)
            - Export PNG button

        Example:
            >>> preview = InteractivePreview(512, 512)
            >>> params = CornellBoxParams(light_intensity=15.0)
            >>> preview.set_params(params)
            >>> preview.run_reactive()  # Blocks until window closed
        """
        from src.python.scene.cornell_box import CornellBoxParams

        self._initialize_window()
        self._ensure_renderer()

        # Set default params if none provided
        if not hasattr(self, "_pending_params"):
            self._pending_params = CornellBoxParams()

        # Initialize slider values from current params
        self._slider_intensity: float = self._pending_params.light_intensity
        self._slider_color_r: float = self._pending_params.light_color[0]
        self._slider_color_g: float = self._pending_params.light_color[1]
        self._slider_color_b: float = self._pending_params.light_color[2]

        # Initialize wall color slider values from current params
        self._slider_left_wall_r: float = self._pending_params.left_wall_color[0]
        self._slider_left_wall_g: float = self._pending_params.left_wall_color[1]
        self._slider_left_wall_b: float = self._pending_params.left_wall_color[2]
        self._slider_right_wall_r: float = self._pending_params.right_wall_color[0]
        self._slider_right_wall_g: float = self._pending_params.right_wall_color[1]
        self._slider_right_wall_b: float = self._pending_params.right_wall_color[2]
        self._slider_back_wall_r: float = self._pending_params.back_wall_color[0]
        self._slider_back_wall_g: float = self._pending_params.back_wall_color[1]
        self._slider_back_wall_b: float = self._pending_params.back_wall_color[2]

        # Initial scene build
        self._rebuild_scene()

        while self.is_running():
            # Check for parameter changes (includes slider changes)
            if self._params_changed():
                self._rebuild_scene()

            # Render 1 sample for responsive UI
            self._renderer.render(num_samples=1)

            # Get the rendered image and update display
            image = self._renderer.get_image_numpy(gamma=2.2)
            self.update_image(image)

            # Update window title with sample count
            sample_count = self._renderer.sample_count
            self._update_title(sample_count)

            # Draw GUI panel with light controls and export button
            self._draw_gui_panel()

            # Show the frame
            self.show_frame()

    def _update_title(self, sample_count: int) -> None:
        """Update the window title with sample count.

        Args:
            sample_count: Current samples per pixel.
        """
        # Taichi GGUI doesn't support dynamic title updates after creation,
        # but we can track the sample count for other purposes (e.g., HUD overlay)
        self._sample_count = sample_count

    def get_sample_count(self) -> int:
        """Get the current sample count.

        Returns:
            The number of samples per pixel rendered so far.
        """
        if hasattr(self, "_sample_count"):
            return self._sample_count
        return 0

    def get_renderer(self) -> Any:
        """Get the underlying progressive renderer.

        Returns:
            The ProgressiveRenderer instance, or None if not initialized.
        """
        if hasattr(self, "_renderer"):
            return self._renderer
        return None

    def _draw_gui_panel(self) -> None:
        """Draw the GUI panel with light controls and export options.

        This creates a GUI panel using Taichi's GGUI system with:
        - Light Intensity slider (0.0 to 50.0)
        - Light Color R/G/B sliders (0.0 to 1.0 each)
        - Export PNG button
        """
        # Light Controls panel
        with self.window.GUI.sub_window(
            "Light Controls", 0.02, 0.02, 0.25, 0.22
        ) as gui:
            # Light intensity slider (0-50, default 15)
            new_intensity = gui.slider_float(
                "Intensity", self._slider_intensity, minimum=0.0, maximum=50.0
            )

            # Light color RGB sliders (0-1 each, default 1.0)
            new_r = gui.slider_float(
                "Color R", self._slider_color_r, minimum=0.0, maximum=1.0
            )
            new_g = gui.slider_float(
                "Color G", self._slider_color_g, minimum=0.0, maximum=1.0
            )
            new_b = gui.slider_float(
                "Color B", self._slider_color_b, minimum=0.0, maximum=1.0
            )

        # Check if any slider values changed
        intensity_changed = abs(new_intensity - self._slider_intensity) > 1e-6
        color_changed = (
            abs(new_r - self._slider_color_r) > 1e-6
            or abs(new_g - self._slider_color_g) > 1e-6
            or abs(new_b - self._slider_color_b) > 1e-6
        )

        if intensity_changed or color_changed:
            # Update stored slider values
            self._slider_intensity = new_intensity
            self._slider_color_r = new_r
            self._slider_color_g = new_g
            self._slider_color_b = new_b

            # Update pending params to trigger scene rebuild
            from src.python.scene.cornell_box import CornellBoxParams

            # Preserve other params from current pending params
            self._pending_params = CornellBoxParams(
                light_intensity=new_intensity,
                light_color=(new_r, new_g, new_b),
                left_wall_color=self._pending_params.left_wall_color,
                right_wall_color=self._pending_params.right_wall_color,
                back_wall_color=self._pending_params.back_wall_color,
            )

        # Export panel (positioned below light controls)
        with self.window.GUI.sub_window("Export", 0.02, 0.25, 0.25, 0.08) as gui:
            if gui.button("Export PNG"):
                self._export_png()

        # Wall Colors panel (positioned below Export)
        with self.window.GUI.sub_window(
            "Wall Colors", 0.02, 0.34, 0.25, 0.32
        ) as gui:
            # Left wall RGB sliders (default: 0.12, 0.45, 0.15 - green)
            new_left_r = gui.slider_float(
                "Left R", self._slider_left_wall_r, minimum=0.0, maximum=1.0
            )
            new_left_g = gui.slider_float(
                "Left G", self._slider_left_wall_g, minimum=0.0, maximum=1.0
            )
            new_left_b = gui.slider_float(
                "Left B", self._slider_left_wall_b, minimum=0.0, maximum=1.0
            )

            # Right wall RGB sliders (default: 0.65, 0.05, 0.05 - red)
            new_right_r = gui.slider_float(
                "Right R", self._slider_right_wall_r, minimum=0.0, maximum=1.0
            )
            new_right_g = gui.slider_float(
                "Right G", self._slider_right_wall_g, minimum=0.0, maximum=1.0
            )
            new_right_b = gui.slider_float(
                "Right B", self._slider_right_wall_b, minimum=0.0, maximum=1.0
            )

            # Back wall RGB sliders (default: 0.73, 0.73, 0.73 - white)
            new_back_r = gui.slider_float(
                "Back R", self._slider_back_wall_r, minimum=0.0, maximum=1.0
            )
            new_back_g = gui.slider_float(
                "Back G", self._slider_back_wall_g, minimum=0.0, maximum=1.0
            )
            new_back_b = gui.slider_float(
                "Back B", self._slider_back_wall_b, minimum=0.0, maximum=1.0
            )

        # Check if any wall color slider values changed
        left_wall_changed = (
            abs(new_left_r - self._slider_left_wall_r) > 1e-6
            or abs(new_left_g - self._slider_left_wall_g) > 1e-6
            or abs(new_left_b - self._slider_left_wall_b) > 1e-6
        )
        right_wall_changed = (
            abs(new_right_r - self._slider_right_wall_r) > 1e-6
            or abs(new_right_g - self._slider_right_wall_g) > 1e-6
            or abs(new_right_b - self._slider_right_wall_b) > 1e-6
        )
        back_wall_changed = (
            abs(new_back_r - self._slider_back_wall_r) > 1e-6
            or abs(new_back_g - self._slider_back_wall_g) > 1e-6
            or abs(new_back_b - self._slider_back_wall_b) > 1e-6
        )

        if left_wall_changed or right_wall_changed or back_wall_changed:
            # Update stored slider values
            self._slider_left_wall_r = new_left_r
            self._slider_left_wall_g = new_left_g
            self._slider_left_wall_b = new_left_b
            self._slider_right_wall_r = new_right_r
            self._slider_right_wall_g = new_right_g
            self._slider_right_wall_b = new_right_b
            self._slider_back_wall_r = new_back_r
            self._slider_back_wall_g = new_back_g
            self._slider_back_wall_b = new_back_b

            # Update pending params to trigger scene rebuild
            from src.python.scene.cornell_box import CornellBoxParams

            # Preserve light params from current pending params
            self._pending_params = CornellBoxParams(
                light_intensity=self._pending_params.light_intensity,
                light_color=self._pending_params.light_color,
                left_wall_color=(new_left_r, new_left_g, new_left_b),
                right_wall_color=(new_right_r, new_right_g, new_right_b),
                back_wall_color=(new_back_r, new_back_g, new_back_b),
            )

    def _export_png(self) -> None:
        """Export the current rendered image to a timestamped PNG file.

        Generates a filename in the format cornell_box_YYYYMMDD_HHMMSS.png
        and saves using the save_png utility from the export module.
        """
        from src.python.preview.export import save_png

        # Generate timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"cornell_box_{timestamp}.png"

        # Get the renderer and save
        renderer = self.get_renderer()
        if renderer is not None:
            save_png(renderer, filename, tone_map="reinhard", gamma=2.2)
            sample_count = self.get_sample_count()
            print(f"Exported: {filename} ({sample_count} SPP)")
        else:
            print("Error: No renderer available for export")
