"""Tests for the preview module.

This module tests the preview/display and preview/export functionality including:
- Tone mapping functions (Reinhard, exposure)
- Gamma correction
- PNG export
- RMSE computation

Note: Tests avoid displaying actual windows by not calling show_preview
in automated tests. The processing functions are tested directly.
"""

import os
import tempfile

import numpy as np
import pytest
from PIL import Image as PILImage


class TestToneMapReinhard:
    """Test Reinhard tone mapping."""

    def test_reinhard_preserves_black(self):
        """Test that Reinhard preserves black (0 -> 0)."""
        from src.python.preview.display import tone_map_reinhard

        image = np.zeros((10, 10, 3), dtype=np.float32)
        result = tone_map_reinhard(image)

        assert np.allclose(result, 0.0)

    def test_reinhard_compresses_bright_values(self):
        """Test that Reinhard compresses bright HDR values."""
        from src.python.preview.display import tone_map_reinhard

        # Create image with HDR values
        image = np.full((10, 10, 3), 10.0, dtype=np.float32)
        result = tone_map_reinhard(image)

        # 10 / (1 + 10) = 10/11 ~ 0.909
        expected = 10.0 / 11.0
        assert np.allclose(result, expected, atol=1e-5)

    def test_reinhard_output_in_01_range(self):
        """Test that Reinhard output is always in [0, 1] range."""
        from src.python.preview.display import tone_map_reinhard

        # Test with very bright values
        image = np.full((10, 10, 3), 1000.0, dtype=np.float32)
        result = tone_map_reinhard(image)

        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_reinhard_handles_negative_input(self):
        """Test that Reinhard clamps negative values to zero."""
        from src.python.preview.display import tone_map_reinhard

        image = np.full((10, 10, 3), -1.0, dtype=np.float32)
        result = tone_map_reinhard(image)

        assert np.all(result >= 0.0)

    def test_reinhard_formula(self):
        """Test Reinhard formula: L / (1 + L)."""
        from src.python.preview.display import tone_map_reinhard

        values = [0.0, 0.5, 1.0, 2.0, 5.0]
        for val in values:
            image = np.full((2, 2, 3), val, dtype=np.float32)
            result = tone_map_reinhard(image)
            expected = val / (1.0 + val) if val >= 0 else 0.0
            assert np.allclose(result, expected, atol=1e-6)


class TestToneMapExposure:
    """Test exposure-based tone mapping."""

    def test_exposure_preserves_black(self):
        """Test that exposure tone mapping preserves black."""
        from src.python.preview.display import tone_map_exposure

        image = np.zeros((10, 10, 3), dtype=np.float32)
        result = tone_map_exposure(image, exposure=1.0)

        assert np.allclose(result, 0.0)

    def test_exposure_higher_value_brighter(self):
        """Test that higher exposure values produce brighter results."""
        from src.python.preview.display import tone_map_exposure

        image = np.full((10, 10, 3), 0.5, dtype=np.float32)

        result_low = tone_map_exposure(image, exposure=0.5)
        result_high = tone_map_exposure(image, exposure=2.0)

        assert np.mean(result_high) > np.mean(result_low)

    def test_exposure_output_in_01_range(self):
        """Test that exposure output is in [0, 1] range."""
        from src.python.preview.display import tone_map_exposure

        image = np.full((10, 10, 3), 100.0, dtype=np.float32)
        result = tone_map_exposure(image, exposure=10.0)

        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_exposure_formula(self):
        """Test exposure formula: 1 - exp(-c * exposure)."""
        from src.python.preview.display import tone_map_exposure

        val = 1.0
        exposure = 2.0
        image = np.full((2, 2, 3), val, dtype=np.float32)
        result = tone_map_exposure(image, exposure=exposure)
        expected = 1.0 - np.exp(-val * exposure)
        assert np.allclose(result, expected, atol=1e-6)


class TestApplyGamma:
    """Test gamma correction."""

    def test_gamma_1_no_change(self):
        """Test that gamma=1.0 produces no change."""
        from src.python.preview.display import apply_gamma

        image = np.random.rand(10, 10, 3).astype(np.float32)
        result = apply_gamma(image, gamma=1.0)

        assert np.allclose(result, image)

    def test_gamma_brightens_midtones(self):
        """Test that gamma correction brightens midtones."""
        from src.python.preview.display import apply_gamma

        # 0.5 in linear should become brighter with gamma 2.2
        image = np.full((10, 10, 3), 0.5, dtype=np.float32)
        result = apply_gamma(image, gamma=2.2)

        # 0.5^(1/2.2) ~ 0.73
        assert np.all(result > 0.5)

    def test_gamma_preserves_black_and_white(self):
        """Test that gamma preserves 0 and 1 values."""
        from src.python.preview.display import apply_gamma

        image = np.array([[[0.0, 1.0, 0.5]]], dtype=np.float32)
        result = apply_gamma(image, gamma=2.2)

        assert np.isclose(result[0, 0, 0], 0.0)
        assert np.isclose(result[0, 0, 1], 1.0)

    def test_gamma_clamps_negative(self):
        """Test that gamma clamps negative values."""
        from src.python.preview.display import apply_gamma

        image = np.full((2, 2, 3), -0.5, dtype=np.float32)
        result = apply_gamma(image, gamma=2.2)

        assert np.all(result >= 0.0)


class TestProcessImageForDisplay:
    """Test the full image processing pipeline."""

    def test_process_with_no_tone_map(self):
        """Test processing with no tone mapping."""
        from src.python.preview.display import process_image_for_display

        image = np.full((10, 10, 3), 0.5, dtype=np.float32)
        result = process_image_for_display(image, tone_map="none", gamma=1.0)

        assert np.allclose(result, 0.5)

    def test_process_with_reinhard(self):
        """Test processing with Reinhard tone mapping."""
        from src.python.preview.display import process_image_for_display

        image = np.full((10, 10, 3), 1.0, dtype=np.float32)
        result = process_image_for_display(image, tone_map="reinhard", gamma=1.0)

        # 1 / (1 + 1) = 0.5
        assert np.allclose(result, 0.5)

    def test_process_with_exposure(self):
        """Test processing with exposure tone mapping."""
        from src.python.preview.display import process_image_for_display

        image = np.full((10, 10, 3), 1.0, dtype=np.float32)
        result = process_image_for_display(
            image, tone_map="exposure", gamma=1.0, exposure=1.0
        )

        # 1 - exp(-1) ~ 0.632
        expected = 1.0 - np.exp(-1.0)
        assert np.allclose(result, expected, atol=1e-5)

    def test_process_output_always_valid(self):
        """Test that processed output is always in valid display range."""
        from src.python.preview.display import process_image_for_display

        # Test with various inputs
        for tone_map in ["none", "reinhard", "exposure"]:
            image = np.random.rand(10, 10, 3).astype(np.float32) * 10  # HDR range
            result = process_image_for_display(image, tone_map=tone_map, gamma=2.2)

            assert np.all(result >= 0.0)
            assert np.all(result <= 1.0)
            assert not np.any(np.isnan(result))
            assert not np.any(np.isinf(result))

    def test_process_invalid_tone_map_raises(self):
        """Test that invalid tone map method raises ValueError."""
        from src.python.preview.display import process_image_for_display

        image = np.zeros((10, 10, 3), dtype=np.float32)

        with pytest.raises(ValueError, match="Unknown tone mapping"):
            process_image_for_display(image, tone_map="invalid")


class TestSavePng:
    """Test PNG export functionality."""

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

    def test_save_png_creates_file(self):
        """Test that save_png creates a valid PNG file."""
        from src.python.core.progressive import ProgressiveRenderer
        from src.python.preview.export import save_png

        self._setup_simple_scene()

        renderer = ProgressiveRenderer(32, 32)
        renderer.render(1)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            filepath = f.name

        try:
            save_png(renderer, filepath, gamma=2.2)

            # Verify file exists and is valid
            assert os.path.exists(filepath)

            # Load and verify with PIL
            img = PILImage.open(filepath)
            assert img.size == (32, 32)
            assert img.mode == "RGB"
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)

    def test_save_png_with_tone_mapping(self):
        """Test PNG export with tone mapping."""
        from src.python.core.progressive import ProgressiveRenderer
        from src.python.preview.export import save_png

        self._setup_simple_scene()

        renderer = ProgressiveRenderer(16, 16)
        renderer.render(1)

        for tone_map in ["none", "reinhard", "exposure"]:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                filepath = f.name

            try:
                save_png(renderer, filepath, tone_map=tone_map, gamma=2.2)
                assert os.path.exists(filepath)

                img = PILImage.open(filepath)
                assert img.size == (16, 16)
            finally:
                if os.path.exists(filepath):
                    os.remove(filepath)


class TestSavePngFromArray:
    """Test PNG export from NumPy array."""

    def test_save_png_from_array(self):
        """Test saving a NumPy array as PNG."""
        from src.python.preview.export import save_png_from_array

        # Create a gradient image
        image = np.zeros((32, 64, 3), dtype=np.float32)
        image[:, :, 0] = np.linspace(0, 1, 64)  # Red gradient

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            filepath = f.name

        try:
            save_png_from_array(image, filepath, gamma=2.2)

            assert os.path.exists(filepath)

            img = PILImage.open(filepath)
            assert img.size == (64, 32)  # PIL size is (width, height)
            assert img.mode == "RGB"
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)


class TestImageToUint8:
    """Test conversion to uint8."""

    def test_image_to_uint8_output_type(self):
        """Test that output is uint8."""
        from src.python.preview.export import image_to_uint8

        image = np.random.rand(10, 10, 3).astype(np.float32)
        result = image_to_uint8(image, gamma=2.2)

        assert result.dtype == np.uint8

    def test_image_to_uint8_range(self):
        """Test that uint8 values are in valid range."""
        from src.python.preview.export import image_to_uint8

        image = np.random.rand(10, 10, 3).astype(np.float32) * 10  # HDR values
        result = image_to_uint8(image, tone_map="reinhard", gamma=2.2)

        assert np.all(result >= 0)
        assert np.all(result <= 255)

    def test_image_to_uint8_black_and_white(self):
        """Test uint8 conversion of black and white."""
        from src.python.preview.export import image_to_uint8

        image = np.array([[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]], dtype=np.float32)
        result = image_to_uint8(image, gamma=1.0, tone_map="none")

        assert np.all(result[0, 0] == 0)
        assert np.all(result[0, 1] == 255)


class TestComputeRmse:
    """Test RMSE computation."""

    def test_rmse_identical_images(self):
        """Test RMSE of identical images is zero."""
        from src.python.preview.export import compute_rmse

        image = np.random.rand(10, 10, 3).astype(np.float32)
        rmse = compute_rmse(image, image)

        assert np.isclose(rmse, 0.0)

    def test_rmse_different_images(self):
        """Test RMSE of different images is positive."""
        from src.python.preview.export import compute_rmse

        image_a = np.zeros((10, 10, 3), dtype=np.float32)
        image_b = np.ones((10, 10, 3), dtype=np.float32)
        rmse = compute_rmse(image_a, image_b)

        assert rmse > 0.0
        # RMSE should be 1.0 for all zeros vs all ones
        assert np.isclose(rmse, 1.0)

    def test_rmse_shape_mismatch_raises(self):
        """Test that RMSE raises for shape mismatch."""
        from src.python.preview.export import compute_rmse

        image_a = np.zeros((10, 10, 3), dtype=np.float32)
        image_b = np.zeros((20, 20, 3), dtype=np.float32)

        with pytest.raises(ValueError, match="shapes must match"):
            compute_rmse(image_a, image_b)


class TestModuleExports:
    """Test that all expected symbols are exported from the module."""

    def test_display_exports(self):
        """Test that display functions are exported."""
        from src.python.preview import (
            apply_gamma,
            process_image_for_display,
            show_comparison,
            show_preview,
            tone_map_exposure,
            tone_map_reinhard,
        )

        # Just verify they're callable
        assert callable(show_preview)
        assert callable(show_comparison)
        assert callable(tone_map_reinhard)
        assert callable(tone_map_exposure)
        assert callable(apply_gamma)
        assert callable(process_image_for_display)

    def test_export_exports(self):
        """Test that export functions are exported."""
        from src.python.preview import (
            compute_rmse,
            image_to_uint8,
            save_png,
            save_png_from_array,
        )

        assert callable(save_png)
        assert callable(save_png_from_array)
        assert callable(image_to_uint8)
        assert callable(compute_rmse)

    def test_type_exports(self):
        """Test that types are exported."""
        from src.python.preview import ToneMapMethod

        # ToneMapMethod should be a type alias, check it's defined
        assert ToneMapMethod is not None

    def test_interactive_preview_export(self):
        """Test that InteractivePreview is exported."""
        from src.python.preview import InteractivePreview

        assert InteractivePreview is not None
        assert callable(InteractivePreview)


class TestInteractivePreview:
    """Tests for the InteractivePreview class.

    Note: These tests avoid creating actual GUI windows by testing
    the initialization and data handling logic only.
    """

    def test_init_creates_display_field(self):
        """Test that initialization creates the display image field."""
        from src.python.preview.interactive import InteractivePreview

        preview = InteractivePreview(64, 48)

        assert preview.width == 64
        assert preview.height == 48
        assert preview.display_image is not None
        # Shape should be (width, height) for Taichi field
        assert preview.display_image.shape == (64, 48)

    def test_init_defers_window_creation(self):
        """Test that window creation is deferred until run/show."""
        from src.python.preview.interactive import InteractivePreview

        preview = InteractivePreview(32, 32)

        # Window should not be created yet
        assert preview._window is None
        assert preview._canvas is None
        assert preview._is_initialized is False

    def test_update_image_validates_shape(self):
        """Test that update_image validates the input shape."""
        from src.python.preview.interactive import InteractivePreview

        preview = InteractivePreview(32, 24)

        # Wrong shape should raise ValueError
        wrong_shape = np.zeros((10, 10, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="doesn't match expected"):
            preview.update_image(wrong_shape)

    def test_update_image_accepts_correct_shape(self):
        """Test that update_image accepts correctly shaped input."""
        from src.python.preview.interactive import InteractivePreview

        preview = InteractivePreview(32, 24)

        # Correct shape: (height, width, 3)
        correct_shape = np.zeros((24, 32, 3), dtype=np.float32)
        # Should not raise
        preview.update_image(correct_shape)

    def test_update_image_transfers_data(self):
        """Test that update_image transfers data to the Taichi field."""
        from src.python.preview.interactive import InteractivePreview

        preview = InteractivePreview(4, 4)

        # Create a test pattern
        image = np.ones((4, 4, 3), dtype=np.float32) * 0.5
        preview.update_image(image)

        # Read back from Taichi field
        result = preview.display_image.to_numpy()
        # After transposition, check values are preserved
        assert np.allclose(result, 0.5, atol=1e-5)

    def test_is_display_available_returns_bool(self):
        """Test that is_display_available returns a boolean."""
        from src.python.preview.interactive import InteractivePreview

        result = InteractivePreview.is_display_available()
        assert isinstance(result, bool)

    def test_custom_title(self):
        """Test that custom title is stored."""
        from src.python.preview.interactive import InteractivePreview

        preview = InteractivePreview(32, 32, title="Custom Title")
        assert preview._title == "Custom Title"

    def test_default_title(self):
        """Test the default title."""
        from src.python.preview.interactive import InteractivePreview

        preview = InteractivePreview(32, 32)
        assert preview._title == "Cornell Box - Interactive Preview"

    def test_get_sample_count_returns_zero_initially(self):
        """Test that get_sample_count returns 0 before any rendering."""
        from src.python.preview.interactive import InteractivePreview

        preview = InteractivePreview(32, 32)
        assert preview.get_sample_count() == 0

    def test_get_renderer_returns_none_initially(self):
        """Test that get_renderer returns None before initialization."""
        from src.python.preview.interactive import InteractivePreview

        preview = InteractivePreview(32, 32)
        assert preview.get_renderer() is None


class TestInteractivePreviewParams:
    """Tests for InteractivePreview parameter handling."""

    def test_set_params_stores_params(self):
        """Test that set_params stores the params."""
        from src.python.preview.interactive import InteractivePreview
        from src.python.scene.cornell_box import CornellBoxParams

        preview = InteractivePreview(32, 32)
        params = CornellBoxParams(light_intensity=20.0)

        preview.set_params(params)

        assert hasattr(preview, "_pending_params")
        assert preview._pending_params.light_intensity == 20.0

    def test_set_params_deep_copies(self):
        """Test that set_params creates a deep copy of params."""
        from src.python.preview.interactive import InteractivePreview
        from src.python.scene.cornell_box import CornellBoxParams

        preview = InteractivePreview(32, 32)
        params = CornellBoxParams(light_intensity=20.0)

        preview.set_params(params)

        # Verify it's a copy, not the same object
        assert preview._pending_params is not params
        assert preview._pending_params == params

    def test_params_changed_returns_false_without_set_params(self):
        """Test _params_changed returns False when no params have been set."""
        from src.python.preview.interactive import InteractivePreview

        preview = InteractivePreview(32, 32)

        # No params set yet
        assert preview._params_changed() is False

    def test_params_changed_returns_true_on_first_build(self):
        """Test _params_changed returns True on first build (no current params)."""
        from src.python.preview.interactive import InteractivePreview
        from src.python.scene.cornell_box import CornellBoxParams

        preview = InteractivePreview(32, 32)
        params = CornellBoxParams()
        preview.set_params(params)

        # Should return True since _current_params doesn't exist
        assert preview._params_changed() is True

    def test_params_changed_returns_false_when_unchanged(self):
        """Test _params_changed returns False when params haven't changed."""
        from src.python.preview.interactive import InteractivePreview
        from src.python.scene.cornell_box import CornellBoxParams

        preview = InteractivePreview(32, 32)
        params = CornellBoxParams(light_intensity=15.0)
        preview.set_params(params)

        # Simulate a scene build by setting _current_params
        import copy

        preview._current_params = copy.deepcopy(params)

        # Params are the same, should return False
        assert preview._params_changed() is False

    def test_params_changed_detects_light_intensity_change(self):
        """Test _params_changed detects light intensity changes."""
        from src.python.preview.interactive import InteractivePreview
        from src.python.scene.cornell_box import CornellBoxParams

        preview = InteractivePreview(32, 32)

        # Set initial params
        preview._pending_params = CornellBoxParams(light_intensity=15.0)
        preview._current_params = CornellBoxParams(light_intensity=15.0)

        # Change intensity
        preview._pending_params = CornellBoxParams(light_intensity=25.0)

        assert preview._params_changed() is True

    def test_params_changed_detects_light_color_change(self):
        """Test _params_changed detects light color changes."""
        from src.python.preview.interactive import InteractivePreview
        from src.python.scene.cornell_box import CornellBoxParams

        preview = InteractivePreview(32, 32)

        # Set initial params
        preview._pending_params = CornellBoxParams(light_color=(1.0, 1.0, 1.0))
        preview._current_params = CornellBoxParams(light_color=(1.0, 1.0, 1.0))

        # Change color
        preview._pending_params = CornellBoxParams(light_color=(1.0, 0.9, 0.8))

        assert preview._params_changed() is True

    def test_params_changed_detects_wall_color_changes(self):
        """Test _params_changed detects wall color changes."""
        from src.python.preview.interactive import InteractivePreview
        from src.python.scene.cornell_box import CornellBoxParams

        preview = InteractivePreview(32, 32)

        # Test left wall change
        preview._pending_params = CornellBoxParams(left_wall_color=(0.12, 0.45, 0.15))
        preview._current_params = CornellBoxParams(left_wall_color=(0.12, 0.45, 0.15))
        assert preview._params_changed() is False

        preview._pending_params = CornellBoxParams(left_wall_color=(0.2, 0.5, 0.2))
        assert preview._params_changed() is True

        # Reset and test right wall change
        preview._pending_params = CornellBoxParams(right_wall_color=(0.65, 0.05, 0.05))
        preview._current_params = CornellBoxParams(right_wall_color=(0.65, 0.05, 0.05))
        assert preview._params_changed() is False

        preview._pending_params = CornellBoxParams(right_wall_color=(0.8, 0.1, 0.1))
        assert preview._params_changed() is True

        # Reset and test back wall change
        preview._pending_params = CornellBoxParams(back_wall_color=(0.73, 0.73, 0.73))
        preview._current_params = CornellBoxParams(back_wall_color=(0.73, 0.73, 0.73))
        assert preview._params_changed() is False

        preview._pending_params = CornellBoxParams(back_wall_color=(0.9, 0.9, 0.9))
        assert preview._params_changed() is True


class TestInteractivePreviewSliderState:
    """Tests for InteractivePreview slider state initialization.

    These tests verify slider state without needing a display.
    Slider states are initialized in run_reactive(), but we can test
    the initial conditions and state management patterns.
    """

    def test_slider_attributes_not_present_before_run_reactive(self):
        """Test slider attributes don't exist before run_reactive is called."""
        from src.python.preview.interactive import InteractivePreview

        preview = InteractivePreview(32, 32)

        # Slider attributes should not exist until run_reactive initializes them
        assert not hasattr(preview, "_slider_intensity")
        assert not hasattr(preview, "_slider_color_r")
        assert not hasattr(preview, "_slider_color_g")
        assert not hasattr(preview, "_slider_color_b")
        assert not hasattr(preview, "_slider_left_wall_r")

    @pytest.mark.skipif(
        not __import__(
            "src.python.preview.interactive", fromlist=["InteractivePreview"]
        ).InteractivePreview.is_display_available(),
        reason="No display available",
    )
    def test_slider_initialization_values(self):
        """Test that slider values would be initialized from params.

        This test manually simulates the slider initialization that happens
        in run_reactive() without actually creating a window.
        """
        from src.python.preview.interactive import InteractivePreview
        from src.python.scene.cornell_box import CornellBoxParams

        preview = InteractivePreview(32, 32)
        params = CornellBoxParams(
            light_intensity=20.0,
            light_color=(0.9, 0.8, 0.7),
            left_wall_color=(0.1, 0.2, 0.3),
            right_wall_color=(0.4, 0.5, 0.6),
            back_wall_color=(0.7, 0.8, 0.9),
        )
        preview.set_params(params)

        # Manually initialize sliders as run_reactive would do
        preview._slider_intensity = preview._pending_params.light_intensity
        preview._slider_color_r = preview._pending_params.light_color[0]
        preview._slider_color_g = preview._pending_params.light_color[1]
        preview._slider_color_b = preview._pending_params.light_color[2]
        preview._slider_left_wall_r = preview._pending_params.left_wall_color[0]
        preview._slider_left_wall_g = preview._pending_params.left_wall_color[1]
        preview._slider_left_wall_b = preview._pending_params.left_wall_color[2]
        preview._slider_right_wall_r = preview._pending_params.right_wall_color[0]
        preview._slider_right_wall_g = preview._pending_params.right_wall_color[1]
        preview._slider_right_wall_b = preview._pending_params.right_wall_color[2]
        preview._slider_back_wall_r = preview._pending_params.back_wall_color[0]
        preview._slider_back_wall_g = preview._pending_params.back_wall_color[1]
        preview._slider_back_wall_b = preview._pending_params.back_wall_color[2]

        # Verify light sliders
        assert preview._slider_intensity == 20.0
        assert preview._slider_color_r == 0.9
        assert preview._slider_color_g == 0.8
        assert preview._slider_color_b == 0.7

        # Verify wall color sliders
        assert preview._slider_left_wall_r == 0.1
        assert preview._slider_left_wall_g == 0.2
        assert preview._slider_left_wall_b == 0.3
        assert preview._slider_right_wall_r == 0.4
        assert preview._slider_right_wall_g == 0.5
        assert preview._slider_right_wall_b == 0.6
        assert preview._slider_back_wall_r == 0.7
        assert preview._slider_back_wall_g == 0.8
        assert preview._slider_back_wall_b == 0.9

    def test_slider_default_values_from_default_params(self):
        """Test slider defaults match CornellBoxParams defaults."""
        from src.python.preview.interactive import InteractivePreview
        from src.python.scene.cornell_box import CornellBoxParams

        preview = InteractivePreview(32, 32)

        # Set default params
        params = CornellBoxParams()
        preview.set_params(params)

        # Manually initialize sliders with defaults
        preview._slider_intensity = preview._pending_params.light_intensity
        preview._slider_color_r = preview._pending_params.light_color[0]
        preview._slider_color_g = preview._pending_params.light_color[1]
        preview._slider_color_b = preview._pending_params.light_color[2]

        # Should match CornellBoxParams defaults
        assert preview._slider_intensity == 15.0
        assert preview._slider_color_r == 1.0
        assert preview._slider_color_g == 1.0
        assert preview._slider_color_b == 1.0
