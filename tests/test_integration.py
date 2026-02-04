"""Integration tests for end-to-end rendering pipeline.

This module tests the complete rendering pipeline from scene creation through
final image output. It verifies that all components work together correctly
and that the output meets basic quality criteria.

Tests are designed to be fast (low resolution, few samples) while still
exercising the full pipeline.

Note: Imports are done inside test methods to avoid Taichi initialization issues.
The conftest.py fixture initializes Taichi before tests run.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    import numpy.typing as npt


# Reference image directory
REFERENCE_DIR = Path(__file__).parent / "reference"


class TestCornellBoxIntegration:
    """Integration tests for Cornell box rendering."""

    def test_cornell_box_end_to_end_renders_successfully(self) -> None:
        """Test that complete Cornell box pipeline runs without errors."""
        from src.python.camera.pinhole import setup_camera
        from src.python.core.integrator import setup_light
        from src.python.core.progressive import ProgressiveRenderer
        from src.python.scene.cornell_box import create_cornell_box_scene, get_light_quad_info

        # Create scene
        scene, camera, _ = create_cornell_box_scene()
        setup_camera(camera)

        # Configure light
        light_info = get_light_quad_info()
        setup_light(
            corner=light_info["corner"],
            edge_u=light_info["edge_u"],
            edge_v=light_info["edge_v"],
            emission=(15.0, 15.0, 15.0),
            material_id=3,
        )

        # Create renderer at low resolution
        renderer = ProgressiveRenderer(64, 64)

        # Render a few samples
        renderer.render(num_samples=4)

        # Verify render completed
        assert renderer.sample_count == 4
        assert renderer.width == 64
        assert renderer.height == 64

    def test_cornell_box_output_contains_no_nan(self) -> None:
        """Test that rendered Cornell box image contains no NaN values."""
        from src.python.camera.pinhole import setup_camera
        from src.python.core.integrator import setup_light
        from src.python.core.progressive import ProgressiveRenderer
        from src.python.scene.cornell_box import create_cornell_box_scene, get_light_quad_info

        scene, camera, _ = create_cornell_box_scene()
        setup_camera(camera)

        light_info = get_light_quad_info()
        setup_light(
            corner=light_info["corner"],
            edge_u=light_info["edge_u"],
            edge_v=light_info["edge_v"],
            emission=(15.0, 15.0, 15.0),
            material_id=3,
        )

        renderer = ProgressiveRenderer(64, 64)
        renderer.render(num_samples=4)

        image = renderer.get_image_numpy()
        assert not np.any(np.isnan(image)), "Image contains NaN values"

    def test_cornell_box_output_contains_no_inf(self) -> None:
        """Test that rendered Cornell box image contains no Inf values."""
        from src.python.camera.pinhole import setup_camera
        from src.python.core.integrator import setup_light
        from src.python.core.progressive import ProgressiveRenderer
        from src.python.scene.cornell_box import create_cornell_box_scene, get_light_quad_info

        scene, camera, _ = create_cornell_box_scene()
        setup_camera(camera)

        light_info = get_light_quad_info()
        setup_light(
            corner=light_info["corner"],
            edge_u=light_info["edge_u"],
            edge_v=light_info["edge_v"],
            emission=(15.0, 15.0, 15.0),
            material_id=3,
        )

        renderer = ProgressiveRenderer(64, 64)
        renderer.render(num_samples=4)

        image = renderer.get_image_numpy()
        assert not np.any(np.isinf(image)), "Image contains Inf values"

    def test_cornell_box_output_non_negative(self) -> None:
        """Test that rendered Cornell box image has no negative values."""
        from src.python.camera.pinhole import setup_camera
        from src.python.core.integrator import setup_light
        from src.python.core.progressive import ProgressiveRenderer
        from src.python.scene.cornell_box import create_cornell_box_scene, get_light_quad_info

        scene, camera, _ = create_cornell_box_scene()
        setup_camera(camera)

        light_info = get_light_quad_info()
        setup_light(
            corner=light_info["corner"],
            edge_u=light_info["edge_u"],
            edge_v=light_info["edge_v"],
            emission=(15.0, 15.0, 15.0),
            material_id=3,
        )

        renderer = ProgressiveRenderer(64, 64)
        renderer.render(num_samples=4)

        image = renderer.get_image_numpy()
        assert np.all(image >= 0.0), "Image contains negative values"

    def test_cornell_box_has_nonzero_illumination(self) -> None:
        """Test that Cornell box render has some illumination (not all black)."""
        from src.python.camera.pinhole import setup_camera
        from src.python.core.integrator import setup_light
        from src.python.core.progressive import ProgressiveRenderer
        from src.python.scene.cornell_box import create_cornell_box_scene, get_light_quad_info

        scene, camera, _ = create_cornell_box_scene()
        setup_camera(camera)

        light_info = get_light_quad_info()
        setup_light(
            corner=light_info["corner"],
            edge_u=light_info["edge_u"],
            edge_v=light_info["edge_v"],
            emission=(15.0, 15.0, 15.0),
            material_id=3,
        )

        # Use more samples to ensure some light hits
        renderer = ProgressiveRenderer(64, 64)
        renderer.render(num_samples=16)

        image = renderer.get_image_numpy()

        # Check that the image has some energy (not all black)
        total_energy = np.sum(image)
        assert total_energy > 0.0, "Image is completely black (no illumination)"

    def test_cornell_box_image_in_valid_range(self) -> None:
        """Test that rendered image values are in expected range after clamping."""
        from src.python.camera.pinhole import setup_camera
        from src.python.core.integrator import setup_light
        from src.python.core.progressive import ProgressiveRenderer
        from src.python.scene.cornell_box import create_cornell_box_scene, get_light_quad_info

        scene, camera, _ = create_cornell_box_scene()
        setup_camera(camera)

        light_info = get_light_quad_info()
        setup_light(
            corner=light_info["corner"],
            edge_u=light_info["edge_u"],
            edge_v=light_info["edge_v"],
            emission=(15.0, 15.0, 15.0),
            material_id=3,
        )

        renderer = ProgressiveRenderer(64, 64)
        renderer.render(num_samples=8)

        # get_image_numpy() returns clamped values in [0, 1]
        image = renderer.get_image_numpy()
        assert np.all(image >= 0.0), "Image has values below 0"
        assert np.all(image <= 1.0), "Image has values above 1"

    def test_cornell_box_save_png(self, tmp_path: Path) -> None:
        """Test that Cornell box render can be saved as PNG."""
        from src.python.camera.pinhole import setup_camera
        from src.python.core.integrator import setup_light
        from src.python.core.progressive import ProgressiveRenderer
        from src.python.preview.export import save_png
        from src.python.scene.cornell_box import create_cornell_box_scene, get_light_quad_info

        scene, camera, _ = create_cornell_box_scene()
        setup_camera(camera)

        light_info = get_light_quad_info()
        setup_light(
            corner=light_info["corner"],
            edge_u=light_info["edge_u"],
            edge_v=light_info["edge_v"],
            emission=(15.0, 15.0, 15.0),
            material_id=3,
        )

        renderer = ProgressiveRenderer(64, 64)
        renderer.render(num_samples=4)

        output_path = tmp_path / "test_cornell_box.png"
        save_png(renderer, str(output_path), tone_map="reinhard", gamma=2.2)

        assert output_path.exists(), "PNG file was not created"
        assert output_path.stat().st_size > 0, "PNG file is empty"

    def test_cornell_box_shape_consistency(self) -> None:
        """Test that rendering produces consistent image shapes across runs.

        Note: Due to Taichi's random number generation, we cannot guarantee
        pixel-perfect reproducibility. Instead, we verify structural consistency.
        """
        from src.python.camera.pinhole import setup_camera
        from src.python.core.integrator import clear_render_target, setup_light
        from src.python.core.progressive import ProgressiveRenderer
        from src.python.scene.cornell_box import create_cornell_box_scene, get_light_quad_info

        def render_scene() -> npt.NDArray[np.float32]:
            scene, camera, _ = create_cornell_box_scene()
            setup_camera(camera)

            light_info = get_light_quad_info()
            setup_light(
                corner=light_info["corner"],
                edge_u=light_info["edge_u"],
                edge_v=light_info["edge_v"],
                emission=(15.0, 15.0, 15.0),
                material_id=3,
            )

            renderer = ProgressiveRenderer(32, 32)
            renderer.render(num_samples=2)
            return renderer.get_image_numpy().copy()

        # Render twice with clearing between
        image1 = render_scene()
        clear_render_target()
        image2 = render_scene()

        # Images should have the same shape and valid values
        assert image1.shape == image2.shape
        assert image1.shape == (32, 32, 3)
        # Both images should have valid values (no NaN/Inf)
        assert not np.any(np.isnan(image1))
        assert not np.any(np.isnan(image2))
        assert not np.any(np.isinf(image1))
        assert not np.any(np.isinf(image2))


class TestProgressiveRefinement:
    """Test progressive rendering refinement."""

    def test_progressive_rendering_accumulates_samples(self) -> None:
        """Test that samples accumulate correctly with multiple render calls."""
        from src.python.camera.pinhole import setup_camera
        from src.python.core.integrator import setup_light
        from src.python.core.progressive import ProgressiveRenderer
        from src.python.scene.cornell_box import create_cornell_box_scene, get_light_quad_info

        scene, camera, _ = create_cornell_box_scene()
        setup_camera(camera)

        light_info = get_light_quad_info()
        setup_light(
            corner=light_info["corner"],
            edge_u=light_info["edge_u"],
            edge_v=light_info["edge_v"],
            emission=(15.0, 15.0, 15.0),
            material_id=3,
        )

        renderer = ProgressiveRenderer(32, 32)

        renderer.render(num_samples=5)
        assert renderer.sample_count == 5

        renderer.render(num_samples=5)
        assert renderer.sample_count == 10

    def test_progressive_rendering_callback_called(self) -> None:
        """Test that progress callback is called during rendering."""
        from src.python.camera.pinhole import setup_camera
        from src.python.core.integrator import setup_light
        from src.python.core.progressive import ProgressiveRenderer
        from src.python.scene.cornell_box import create_cornell_box_scene, get_light_quad_info

        scene, camera, _ = create_cornell_box_scene()
        setup_camera(camera)

        light_info = get_light_quad_info()
        setup_light(
            corner=light_info["corner"],
            edge_u=light_info["edge_u"],
            edge_v=light_info["edge_v"],
            emission=(15.0, 15.0, 15.0),
            material_id=3,
        )

        renderer = ProgressiveRenderer(32, 32)

        callback_counts: list[tuple[int, int]] = []

        def callback(current: int, target: int) -> None:
            callback_counts.append((current, target))

        renderer.render(num_samples=10, batch_size=2, callback=callback)

        # Should have 5 callbacks (10 samples / batch_size of 2)
        assert len(callback_counts) == 5
        # Final callback should show all samples rendered
        assert callback_counts[-1][0] == 10

    def test_progressive_rendering_generator(self) -> None:
        """Test the generator-based progressive rendering API."""
        from src.python.camera.pinhole import setup_camera
        from src.python.core.integrator import setup_light
        from src.python.core.progressive import ProgressiveRenderer
        from src.python.scene.cornell_box import create_cornell_box_scene, get_light_quad_info

        scene, camera, _ = create_cornell_box_scene()
        setup_camera(camera)

        light_info = get_light_quad_info()
        setup_light(
            corner=light_info["corner"],
            edge_u=light_info["edge_u"],
            edge_v=light_info["edge_v"],
            emission=(15.0, 15.0, 15.0),
            material_id=3,
        )

        renderer = ProgressiveRenderer(32, 32)

        progress_updates = list(renderer.render_progressive(num_samples=6, batch_size=2))

        assert len(progress_updates) == 3
        assert progress_updates[-1][0] == 6  # Final sample count


class TestIntegrationWithDifferentMaterials:
    """Test rendering with various material combinations."""

    def test_all_material_types_in_scene(self) -> None:
        """Test scene with Lambertian, Metal, and Dielectric materials."""
        from src.python.camera.pinhole import setup_camera
        from src.python.core.integrator import disable_light
        from src.python.core.progressive import ProgressiveRenderer
        from src.python.scene.cornell_box import create_cornell_box_scene

        # Cornell box already has all three material types
        scene, camera, _ = create_cornell_box_scene()
        setup_camera(camera)
        disable_light()  # Render without light to test material interactions

        renderer = ProgressiveRenderer(32, 32)
        renderer.render(num_samples=2)

        image = renderer.get_image_numpy()
        assert not np.any(np.isnan(image)), "NaN with mixed materials"
        assert not np.any(np.isinf(image)), "Inf with mixed materials"


class TestImageQuality:
    """Test image quality metrics."""

    def test_image_has_expected_dimensions(self) -> None:
        """Test that output image has correct dimensions."""
        from src.python.camera.pinhole import setup_camera
        from src.python.core.integrator import setup_light
        from src.python.core.progressive import ProgressiveRenderer
        from src.python.scene.cornell_box import create_cornell_box_scene, get_light_quad_info

        scene, camera, _ = create_cornell_box_scene()
        setup_camera(camera)

        light_info = get_light_quad_info()
        setup_light(
            corner=light_info["corner"],
            edge_u=light_info["edge_u"],
            edge_v=light_info["edge_v"],
            emission=(15.0, 15.0, 15.0),
            material_id=3,
        )

        width, height = 128, 96
        renderer = ProgressiveRenderer(width, height)
        renderer.render(num_samples=2)

        image = renderer.get_image_numpy()
        assert image.shape == (height, width, 3)

    def test_uint8_conversion_valid(self) -> None:
        """Test that uint8 conversion produces valid values."""
        from src.python.camera.pinhole import setup_camera
        from src.python.core.integrator import setup_light
        from src.python.core.progressive import ProgressiveRenderer
        from src.python.scene.cornell_box import create_cornell_box_scene, get_light_quad_info

        scene, camera, _ = create_cornell_box_scene()
        setup_camera(camera)

        light_info = get_light_quad_info()
        setup_light(
            corner=light_info["corner"],
            edge_u=light_info["edge_u"],
            edge_v=light_info["edge_v"],
            emission=(15.0, 15.0, 15.0),
            material_id=3,
        )

        renderer = ProgressiveRenderer(64, 64)
        renderer.render(num_samples=4)

        image_uint8 = renderer.get_image_uint8(gamma=2.2)

        assert image_uint8.dtype == np.uint8
        assert np.all(image_uint8 >= 0)
        assert np.all(image_uint8 <= 255)


@pytest.mark.slow
class TestHighQualityRender:
    """Higher quality render tests (marked slow for optional execution)."""

    def test_cornell_box_reference_render(self, tmp_path: Path) -> None:
        """Render a reference Cornell box image for visual inspection.

        This test renders at higher quality and saves to the reference directory.
        Run with: pytest -m slow tests/test_integration.py
        """
        from src.python.camera.pinhole import setup_camera
        from src.python.core.integrator import setup_light
        from src.python.core.progressive import ProgressiveRenderer
        from src.python.preview.export import save_png
        from src.python.scene.cornell_box import create_cornell_box_scene, get_light_quad_info

        scene, camera, _ = create_cornell_box_scene()
        setup_camera(camera)

        light_info = get_light_quad_info()
        setup_light(
            corner=light_info["corner"],
            edge_u=light_info["edge_u"],
            edge_v=light_info["edge_v"],
            emission=(15.0, 15.0, 15.0),
            material_id=3,
        )

        # Higher quality render (still fast enough for CI)
        renderer = ProgressiveRenderer(128, 128)
        renderer.render(num_samples=32)

        # Save to reference directory (create if needed)
        REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
        output_path = REFERENCE_DIR / "cornell_box_reference.png"
        save_png(renderer, str(output_path), tone_map="reinhard", gamma=2.2)

        # Verify file was created
        assert output_path.exists()

        # Basic quality checks on the reference render
        image = renderer.get_image_numpy()
        assert not np.any(np.isnan(image))
        assert not np.any(np.isinf(image))
        assert np.sum(image) > 0  # Has some illumination
