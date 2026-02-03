"""Tests for the progressive renderer.

This module tests the ProgressiveRenderer class including:
- Initialization and setup
- Progressive sample accumulation
- Batch rendering
- Progress callbacks and generators
- Reset functionality
- Image output in various formats

Note: Imports are done inside test methods to avoid Taichi initialization issues.
The conftest.py fixture initializes Taichi before tests run.
"""

import numpy as np
import pytest


class TestProgressiveRendererInit:
    """Test ProgressiveRenderer initialization."""

    def test_init_creates_render_target(self):
        """Test that initialization creates a render target with correct dimensions."""
        from src.python.core.progressive import ProgressiveRenderer

        renderer = ProgressiveRenderer(128, 96)

        assert renderer.width == 128
        assert renderer.height == 96
        assert renderer.sample_count == 0

    def test_init_with_different_dimensions(self):
        """Test initialization with various dimensions."""
        from src.python.core.progressive import ProgressiveRenderer

        for width, height in [(64, 64), (100, 50), (512, 256)]:
            renderer = ProgressiveRenderer(width, height)
            assert renderer.width == width
            assert renderer.height == height

    def test_init_rejects_oversized_dimensions(self):
        """Test that initialization rejects dimensions exceeding max size."""
        from src.python.core.progressive import ProgressiveRenderer

        with pytest.raises(ValueError, match="exceed maximum"):
            ProgressiveRenderer(4096, 100)

        with pytest.raises(ValueError, match="exceed maximum"):
            ProgressiveRenderer(100, 4096)


class TestProgressiveRendererReset:
    """Test reset functionality."""

    def test_reset_clears_sample_count(self):
        """Test that reset clears the sample count."""
        from src.python.camera.pinhole import PinholeCamera, setup_camera
        from src.python.core.progressive import ProgressiveRenderer
        from src.python.scene.manager import SceneManager

        # Setup scene
        scene = SceneManager()
        scene.add_lambertian_material((0.5, 0.5, 0.5))
        scene.add_sphere((0, 0, -2), 1.0, 0)
        camera = PinholeCamera(
            lookfrom=(0, 0, 0),
            lookat=(0, 0, -1),
            vup=(0, 1, 0),
            vfov=90.0,
            aspect_ratio=1.0,
        )
        setup_camera(camera)

        renderer = ProgressiveRenderer(16, 16)
        renderer.render(5)
        assert renderer.sample_count == 5

        renderer.reset()
        assert renderer.sample_count == 0

    def test_reset_clears_color_buffer(self):
        """Test that reset clears the accumulated color."""
        from src.python.camera.pinhole import PinholeCamera, setup_camera
        from src.python.core.integrator import setup_light
        from src.python.core.progressive import ProgressiveRenderer
        from src.python.scene.manager import SceneManager

        # Setup scene with bright light
        scene = SceneManager()
        mat_id = scene.add_lambertian_material((0.8, 0.8, 0.8))
        scene.add_sphere((0, 0, -2), 1.0, mat_id)
        scene.add_quad((0, 3, -2), (1, 0, 0), (0, 0, 1), mat_id)

        setup_light(
            corner=(0, 3, -2),
            edge_u=(1, 0, 0),
            edge_v=(0, 0, 1),
            emission=(15.0, 15.0, 15.0),
            material_id=mat_id,
        )

        camera = PinholeCamera(
            lookfrom=(0, 0, 0),
            lookat=(0, 0, -1),
            vup=(0, 1, 0),
            vfov=90.0,
            aspect_ratio=1.0,
        )
        setup_camera(camera)

        renderer = ProgressiveRenderer(16, 16)
        renderer.render(5)

        # Get some non-zero value from rendered image (verify it was rendered)
        _ = renderer.get_image_numpy()

        renderer.reset()

        # After reset, buffer should be zero
        image_after = renderer.get_image_numpy()
        assert np.allclose(image_after, 0.0)


class TestProgressiveRendererResize:
    """Test resize functionality."""

    def test_resize_changes_dimensions(self):
        """Test that resize changes the dimensions."""
        from src.python.core.progressive import ProgressiveRenderer

        renderer = ProgressiveRenderer(64, 64)
        assert renderer.width == 64
        assert renderer.height == 64

        renderer.resize(128, 96)
        assert renderer.width == 128
        assert renderer.height == 96

    def test_resize_resets_samples(self):
        """Test that resize resets sample count."""
        from src.python.camera.pinhole import PinholeCamera, setup_camera
        from src.python.core.progressive import ProgressiveRenderer
        from src.python.scene.manager import SceneManager

        scene = SceneManager()
        scene.add_lambertian_material((0.5, 0.5, 0.5))
        scene.add_sphere((0, 0, -2), 1.0, 0)
        camera = PinholeCamera(
            lookfrom=(0, 0, 0),
            lookat=(0, 0, -1),
            vup=(0, 1, 0),
            vfov=90.0,
            aspect_ratio=1.0,
        )
        setup_camera(camera)

        renderer = ProgressiveRenderer(32, 32)
        renderer.render(5)
        assert renderer.sample_count == 5

        renderer.resize(64, 64)
        assert renderer.sample_count == 0


class TestProgressiveRendererRender:
    """Test render functionality."""

    def _setup_simple_scene(self):
        """Helper to set up a simple test scene."""
        from src.python.camera.pinhole import PinholeCamera, setup_camera
        from src.python.core.integrator import disable_light
        from src.python.scene.manager import SceneManager

        scene = SceneManager()
        mat_id = scene.add_lambertian_material((0.5, 0.5, 0.5))
        scene.add_sphere((0, 0, -2), 1.0, mat_id)

        camera = PinholeCamera(
            lookfrom=(0, 0, 0),
            lookat=(0, 0, -1),
            vup=(0, 1, 0),
            vfov=90.0,
            aspect_ratio=1.0,
        )
        setup_camera(camera)
        disable_light()

    def test_render_accumulates_samples(self):
        """Test that render accumulates samples correctly."""
        from src.python.core.progressive import ProgressiveRenderer

        self._setup_simple_scene()

        renderer = ProgressiveRenderer(16, 16)
        assert renderer.sample_count == 0

        renderer.render(5)
        assert renderer.sample_count == 5

        renderer.render(10)
        assert renderer.sample_count == 15

    def test_render_with_zero_samples_does_nothing(self):
        """Test that rendering zero samples has no effect."""
        from src.python.core.progressive import ProgressiveRenderer

        self._setup_simple_scene()

        renderer = ProgressiveRenderer(16, 16)
        renderer.render(5)
        count_before = renderer.sample_count

        renderer.render(0)
        assert renderer.sample_count == count_before

    def test_render_with_negative_samples_does_nothing(self):
        """Test that rendering negative samples has no effect."""
        from src.python.core.progressive import ProgressiveRenderer

        self._setup_simple_scene()

        renderer = ProgressiveRenderer(16, 16)
        renderer.render(5)
        count_before = renderer.sample_count

        renderer.render(-10)
        assert renderer.sample_count == count_before

    def test_render_with_batch_size(self):
        """Test rendering with different batch sizes."""
        from src.python.core.progressive import ProgressiveRenderer

        self._setup_simple_scene()

        renderer = ProgressiveRenderer(16, 16)

        # Render with batch size 3 (10 samples = 3 + 3 + 3 + 1)
        renderer.render(10, batch_size=3)
        assert renderer.sample_count == 10


class TestProgressiveRendererCallbacks:
    """Test progress callback functionality."""

    def _setup_simple_scene(self):
        """Helper to set up a simple test scene."""
        from src.python.camera.pinhole import PinholeCamera, setup_camera
        from src.python.core.integrator import disable_light
        from src.python.scene.manager import SceneManager

        scene = SceneManager()
        mat_id = scene.add_lambertian_material((0.5, 0.5, 0.5))
        scene.add_sphere((0, 0, -2), 1.0, mat_id)

        camera = PinholeCamera(
            lookfrom=(0, 0, 0),
            lookat=(0, 0, -1),
            vup=(0, 1, 0),
            vfov=90.0,
            aspect_ratio=1.0,
        )
        setup_camera(camera)
        disable_light()

    def test_callback_receives_progress(self):
        """Test that callback receives correct progress values."""
        from src.python.core.progressive import ProgressiveRenderer

        self._setup_simple_scene()

        renderer = ProgressiveRenderer(16, 16)

        progress_values = []

        def callback(current, target):
            progress_values.append((current, target))

        renderer.render(10, batch_size=2, callback=callback)

        # Should have 5 callbacks (10 / 2 = 5 batches)
        assert len(progress_values) == 5

        # Each callback should report correct target
        for current, target in progress_values:
            assert target == 10

        # Current should increase
        currents = [p[0] for p in progress_values]
        assert currents == [2, 4, 6, 8, 10]

    def test_callback_with_existing_samples(self):
        """Test that callback accounts for pre-existing samples."""
        from src.python.core.progressive import ProgressiveRenderer

        self._setup_simple_scene()

        renderer = ProgressiveRenderer(16, 16)

        # Render some samples first
        renderer.render(5)

        progress_values = []

        def callback(current, target):
            progress_values.append((current, target))

        renderer.render(10, batch_size=5, callback=callback)

        # Should have 2 callbacks
        assert len(progress_values) == 2

        # Target should be 5 (existing) + 10 (new) = 15
        for current, target in progress_values:
            assert target == 15

        # Current should be 10 and 15
        currents = [p[0] for p in progress_values]
        assert currents == [10, 15]

    def test_render_without_callback(self):
        """Test that render works without callback."""
        from src.python.core.progressive import ProgressiveRenderer

        self._setup_simple_scene()

        renderer = ProgressiveRenderer(16, 16)
        renderer.render(10, batch_size=2, callback=None)
        assert renderer.sample_count == 10


class TestProgressiveRendererGenerator:
    """Test generator-based progress reporting."""

    def _setup_simple_scene(self):
        """Helper to set up a simple test scene."""
        from src.python.camera.pinhole import PinholeCamera, setup_camera
        from src.python.core.integrator import disable_light
        from src.python.scene.manager import SceneManager

        scene = SceneManager()
        mat_id = scene.add_lambertian_material((0.5, 0.5, 0.5))
        scene.add_sphere((0, 0, -2), 1.0, mat_id)

        camera = PinholeCamera(
            lookfrom=(0, 0, 0),
            lookat=(0, 0, -1),
            vup=(0, 1, 0),
            vfov=90.0,
            aspect_ratio=1.0,
        )
        setup_camera(camera)
        disable_light()

    def test_render_progressive_yields_progress(self):
        """Test that render_progressive yields correct progress."""
        from src.python.core.progressive import ProgressiveRenderer

        self._setup_simple_scene()

        renderer = ProgressiveRenderer(16, 16)

        progress_values = list(renderer.render_progressive(10, batch_size=2))

        # Should have 5 yields
        assert len(progress_values) == 5

        # Check values
        for current, target in progress_values:
            assert target == 10

        currents = [p[0] for p in progress_values]
        assert currents == [2, 4, 6, 8, 10]

    def test_render_progressive_with_zero_samples(self):
        """Test that render_progressive with zero samples yields nothing."""
        from src.python.core.progressive import ProgressiveRenderer

        self._setup_simple_scene()

        renderer = ProgressiveRenderer(16, 16)

        progress_values = list(renderer.render_progressive(0))
        assert progress_values == []

    def test_render_progressive_interruptible(self):
        """Test that render_progressive can be interrupted mid-render."""
        from src.python.core.progressive import ProgressiveRenderer

        self._setup_simple_scene()

        renderer = ProgressiveRenderer(16, 16)

        # Only consume first 3 yields
        gen = renderer.render_progressive(10, batch_size=2)
        for _ in range(3):
            next(gen)

        # Should have rendered 6 samples
        assert renderer.sample_count == 6


class TestProgressiveRendererImageOutput:
    """Test image output functionality."""

    def _setup_simple_scene(self):
        """Helper to set up a simple test scene."""
        from src.python.camera.pinhole import PinholeCamera, setup_camera
        from src.python.core.integrator import disable_light
        from src.python.scene.manager import SceneManager

        scene = SceneManager()
        mat_id = scene.add_lambertian_material((0.5, 0.5, 0.5))
        scene.add_sphere((0, 0, -2), 1.0, mat_id)

        camera = PinholeCamera(
            lookfrom=(0, 0, 0),
            lookat=(0, 0, -1),
            vup=(0, 1, 0),
            vfov=90.0,
            aspect_ratio=1.0,
        )
        setup_camera(camera)
        disable_light()

    def test_get_image_numpy_returns_correct_shape(self):
        """Test that get_image_numpy returns correct shape."""
        from src.python.core.progressive import ProgressiveRenderer

        self._setup_simple_scene()

        renderer = ProgressiveRenderer(64, 48)
        renderer.render(1)

        image = renderer.get_image_numpy()

        assert image.shape == (48, 64, 3)  # (height, width, channels)
        assert image.dtype == np.float32

    def test_get_image_numpy_with_gamma(self):
        """Test gamma correction in get_image_numpy."""
        from src.python.core.progressive import ProgressiveRenderer

        self._setup_simple_scene()

        renderer = ProgressiveRenderer(16, 16)
        renderer.render(1)

        # Linear image
        linear = renderer.get_image_numpy(gamma=1.0)

        # Gamma corrected image
        gamma_corrected = renderer.get_image_numpy(gamma=2.2)

        # Where linear > 0, gamma corrected should be brighter (higher values)
        mask = linear > 0.01
        if np.any(mask):
            assert np.mean(gamma_corrected[mask]) >= np.mean(linear[mask])

    def test_get_image_uint8_returns_correct_type(self):
        """Test that get_image_uint8 returns uint8 array."""
        from src.python.core.progressive import ProgressiveRenderer

        self._setup_simple_scene()

        renderer = ProgressiveRenderer(32, 32)
        renderer.render(1)

        image = renderer.get_image_uint8()

        assert image.dtype == np.uint8
        assert image.shape == (32, 32, 3)

    def test_get_image_uint8_values_in_range(self):
        """Test that uint8 image values are in valid range."""
        from src.python.core.progressive import ProgressiveRenderer

        self._setup_simple_scene()

        renderer = ProgressiveRenderer(32, 32)
        renderer.render(1)

        image = renderer.get_image_uint8()

        assert np.all(image >= 0)
        assert np.all(image <= 255)

    def test_get_image_returns_taichi_field(self):
        """Test that get_image returns the Taichi field."""
        from src.python.core.progressive import ProgressiveRenderer

        self._setup_simple_scene()

        renderer = ProgressiveRenderer(32, 32)
        renderer.render(1)

        field = renderer.get_image()

        # Should be a Taichi field (can call to_numpy)
        assert hasattr(field, "to_numpy")


class TestProgressiveRendererRepr:
    """Test string representation."""

    def test_repr_shows_state(self):
        """Test that __repr__ shows renderer state."""
        from src.python.camera.pinhole import PinholeCamera, setup_camera
        from src.python.core.integrator import disable_light
        from src.python.core.progressive import ProgressiveRenderer
        from src.python.scene.manager import SceneManager

        scene = SceneManager()
        scene.add_lambertian_material((0.5, 0.5, 0.5))
        scene.add_sphere((0, 0, -2), 1.0, 0)
        camera = PinholeCamera(
            lookfrom=(0, 0, 0),
            lookat=(0, 0, -1),
            vup=(0, 1, 0),
            vfov=90.0,
            aspect_ratio=1.0,
        )
        setup_camera(camera)
        disable_light()

        renderer = ProgressiveRenderer(64, 48)

        repr_str = repr(renderer)

        assert "width=64" in repr_str
        assert "height=48" in repr_str
        assert "samples=0" in repr_str

        renderer.render(5)
        repr_str = repr(renderer)
        assert "samples=5" in repr_str


class TestProgressiveRendererNumericalStability:
    """Test numerical stability of progressive accumulation."""

    def _setup_simple_scene(self):
        """Helper to set up a simple test scene."""
        from src.python.camera.pinhole import PinholeCamera, setup_camera
        from src.python.core.integrator import disable_light
        from src.python.scene.manager import SceneManager

        scene = SceneManager()
        mat_id = scene.add_lambertian_material((0.5, 0.5, 0.5))
        scene.add_sphere((0, 0, -2), 1.0, mat_id)

        camera = PinholeCamera(
            lookfrom=(0, 0, 0),
            lookat=(0, 0, -1),
            vup=(0, 1, 0),
            vfov=90.0,
            aspect_ratio=1.0,
        )
        setup_camera(camera)
        disable_light()

    def test_no_nan_after_many_samples(self):
        """Test that image has no NaN values after many samples."""
        from src.python.core.progressive import ProgressiveRenderer

        self._setup_simple_scene()

        renderer = ProgressiveRenderer(32, 32)
        renderer.render(50)  # Render many samples

        image = renderer.get_image_numpy()

        assert not np.any(np.isnan(image))

    def test_no_inf_after_many_samples(self):
        """Test that image has no Inf values after many samples."""
        from src.python.core.progressive import ProgressiveRenderer

        self._setup_simple_scene()

        renderer = ProgressiveRenderer(32, 32)
        renderer.render(50)

        image = renderer.get_image_numpy()

        assert not np.any(np.isinf(image))

    def test_values_in_expected_range(self):
        """Test that image values are in expected range after accumulation."""
        from src.python.core.progressive import ProgressiveRenderer

        self._setup_simple_scene()

        renderer = ProgressiveRenderer(32, 32)
        renderer.render(20)

        image = renderer.get_image_numpy()

        # Values should be non-negative
        assert np.all(image >= 0.0)

        # After normalization, values should be clamped to [0, 1]
        assert np.all(image <= 1.0)


class TestProgressiveRendererContinuousRefinement:
    """Test continuous refinement behavior."""

    def _setup_simple_scene(self):
        """Helper to set up a simple test scene."""
        from src.python.camera.pinhole import PinholeCamera, setup_camera
        from src.python.core.integrator import disable_light
        from src.python.scene.manager import SceneManager

        scene = SceneManager()
        mat_id = scene.add_lambertian_material((0.5, 0.5, 0.5))
        scene.add_sphere((0, 0, -2), 1.0, mat_id)

        camera = PinholeCamera(
            lookfrom=(0, 0, 0),
            lookat=(0, 0, -1),
            vup=(0, 1, 0),
            vfov=90.0,
            aspect_ratio=1.0,
        )
        setup_camera(camera)
        disable_light()

    def test_multiple_render_calls_accumulate(self):
        """Test that multiple render calls properly accumulate."""
        from src.python.core.progressive import ProgressiveRenderer

        self._setup_simple_scene()

        renderer = ProgressiveRenderer(16, 16)

        # Render in multiple small batches
        for _ in range(10):
            renderer.render(1)

        assert renderer.sample_count == 10

        # Should be equivalent to rendering 10 at once
        renderer2 = ProgressiveRenderer(16, 16)
        renderer2.render(10)

        assert renderer2.sample_count == 10

    def test_can_render_additional_samples_after_pause(self):
        """Test that rendering can be resumed after getting intermediate results."""
        from src.python.core.progressive import ProgressiveRenderer

        self._setup_simple_scene()

        renderer = ProgressiveRenderer(16, 16)

        # Render some samples
        renderer.render(5)

        # Get intermediate result (simulating user viewing)
        _ = renderer.get_image_uint8()

        # Continue rendering
        renderer.render(5)
        image_10 = renderer.get_image_numpy()

        # Should have more samples
        assert renderer.sample_count == 10

        # Image should still be valid
        assert not np.any(np.isnan(image_10))
        assert not np.any(np.isinf(image_10))
