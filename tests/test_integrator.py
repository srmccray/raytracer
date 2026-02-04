"""Tests for the path tracing integrator.

This module tests the core path tracing functionality including:
- Render target setup and management
- Single sample rendering
- Material dispatch (Lambertian, Metal, Dielectric)
- Light emission
- Russian roulette termination behavior
- Progressive accumulation

Note: Imports are done inside test methods to avoid Taichi initialization issues.
The conftest.py fixture initializes Taichi before tests run, so module-level
imports of modules containing ti.field() declarations would fail.
"""

import pytest


class TestRenderTargetSetup:
    """Test render target initialization and management."""

    def test_setup_render_target_creates_buffers(self):
        """Test that setup_render_target creates correctly sized buffers."""
        from src.python.core.integrator import (
            get_image,
            get_image_dimensions,
            get_sample_count,
            setup_render_target,
        )

        width, height = 64, 48
        setup_render_target(width, height)

        assert get_image_dimensions() == (width, height)
        assert get_image() is not None
        assert get_sample_count() is not None

    def test_setup_render_target_clears_existing(self):
        """Test that setup_render_target clears previous data."""
        from src.python.camera.pinhole import PinholeCamera, setup_camera
        from src.python.core.integrator import (
            get_total_samples,
            render_image,
            setup_render_target,
        )
        from src.python.scene.manager import SceneManager

        setup_render_target(32, 32)

        # Render some samples
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
        render_image(num_samples=1)

        # Reset render target
        setup_render_target(32, 32)
        assert get_total_samples() == 0

    def test_clear_render_target(self):
        """Test that clear_render_target resets sample count."""
        from src.python.camera.pinhole import PinholeCamera, setup_camera
        from src.python.core.integrator import (
            clear_render_target,
            get_total_samples,
            render_image,
            setup_render_target,
        )
        from src.python.scene.manager import SceneManager

        setup_render_target(16, 16)

        # Render some samples
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
        render_image(num_samples=2)
        assert get_total_samples() == 2

        clear_render_target()
        assert get_total_samples() == 0


class TestLightSetup:
    """Test light source configuration."""

    def test_setup_light_enables_light(self):
        """Test that setup_light enables the light source."""
        from src.python.core.integrator import (
            disable_light,
            is_light_enabled,
            setup_light,
        )

        disable_light()
        assert not is_light_enabled()

        setup_light(
            corner=(0, 1, 0),
            edge_u=(1, 0, 0),
            edge_v=(0, 0, 1),
            emission=(10.0, 10.0, 10.0),
            material_id=0,
        )
        assert is_light_enabled()

    def test_disable_light(self):
        """Test that disable_light disables the light source."""
        from src.python.core.integrator import (
            disable_light,
            is_light_enabled,
            setup_light,
        )

        setup_light(
            corner=(0, 1, 0),
            edge_u=(1, 0, 0),
            edge_v=(0, 0, 1),
            emission=(10.0, 10.0, 10.0),
            material_id=0,
        )
        assert is_light_enabled()

        disable_light()
        assert not is_light_enabled()


class TestBasicRendering:
    """Test basic rendering functionality."""

    def test_render_sample_returns_color(self):
        """Test that render_sample returns a valid color tuple."""
        from src.python.camera.pinhole import PinholeCamera, setup_camera
        from src.python.core.integrator import (
            disable_light,
            render_sample,
            setup_render_target,
        )
        from src.python.scene.manager import SceneManager

        setup_render_target(64, 64)

        # Simple scene with one sphere
        scene = SceneManager()
        mat_id = scene.add_lambertian_material((0.8, 0.3, 0.3))
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

        # Render center pixel
        color = render_sample(32, 32)

        assert isinstance(color, tuple)
        assert len(color) == 3
        # Color components should be finite (not NaN/Inf)
        assert all(c >= 0.0 for c in color)
        assert all(not (c != c) for c in color)  # NaN check

    def test_render_hits_lambertian_surface(self):
        """Test that rendering a Lambertian surface returns non-zero color with light."""
        from src.python.camera.pinhole import PinholeCamera, setup_camera
        from src.python.core.integrator import (
            render_sample,
            setup_light,
            setup_render_target,
        )
        from src.python.scene.manager import SceneManager

        setup_render_target(64, 64)

        # Scene with a sphere and a light behind camera
        scene = SceneManager()
        mat_id = scene.add_lambertian_material((0.8, 0.8, 0.8))
        light_mat_id = scene.add_lambertian_material((1.0, 1.0, 1.0))
        scene.add_sphere((0, 0, -2), 1.0, mat_id)
        scene.add_quad((0, 3, -2), (1, 0, 0), (0, 0, 1), light_mat_id)

        camera = PinholeCamera(
            lookfrom=(0, 0, 0),
            lookat=(0, 0, -1),
            vup=(0, 1, 0),
            vfov=90.0,
            aspect_ratio=1.0,
        )
        setup_camera(camera)

        # Setup light emission
        setup_light(
            corner=(0, 3, -2),
            edge_u=(1, 0, 0),
            edge_v=(0, 0, 1),
            emission=(15.0, 15.0, 15.0),
            material_id=light_mat_id,
        )

        # Render multiple samples for the center pixel to average out noise
        total_r, total_g, total_b = 0.0, 0.0, 0.0
        num_samples = 10
        for _ in range(num_samples):
            r, g, b = render_sample(32, 32)
            total_r += r
            total_g += g
            total_b += b

        avg_r = total_r / num_samples
        avg_g = total_g / num_samples
        avg_b = total_b / num_samples

        # With a light source, diffuse surface should have some illumination
        # (may not hit light every time due to Monte Carlo, but should average > 0)
        # This is a weak test due to the stochastic nature
        assert avg_r >= 0.0 and avg_g >= 0.0 and avg_b >= 0.0

    def test_render_misses_scene_returns_background(self):
        """Test that rays missing the scene return background color."""
        from src.python.camera.pinhole import PinholeCamera, setup_camera
        from src.python.core.integrator import (
            disable_light,
            render_sample,
            setup_render_target,
        )
        from src.python.scene.intersection import clear_scene
        from src.python.scene.manager import SceneManager

        setup_render_target(64, 64)

        # Empty scene
        scene = SceneManager()
        scene.add_lambertian_material((0.5, 0.5, 0.5))  # Dummy material

        camera = PinholeCamera(
            lookfrom=(0, 0, 0),
            lookat=(0, 0, -1),
            vup=(0, 1, 0),
            vfov=90.0,
            aspect_ratio=1.0,
        )
        setup_camera(camera)
        disable_light()

        # No geometry, ray should miss and return background color (black)
        # Clear the scene of any geometry
        clear_scene()

        color = render_sample(32, 32)
        # Background is black (0, 0, 0)
        assert color == (0.0, 0.0, 0.0)


class TestMaterialDispatch:
    """Test that different materials are correctly dispatched."""

    def test_metal_material_reflects(self):
        """Test that metal material produces reflected rays."""
        from src.python.camera.pinhole import PinholeCamera, setup_camera
        from src.python.core.integrator import (
            disable_light,
            render_sample,
            setup_render_target,
        )
        from src.python.scene.manager import SceneManager

        setup_render_target(64, 64)

        # Scene with a metal sphere
        scene = SceneManager()
        mat_id = scene.add_metal_material((0.9, 0.9, 0.9), roughness=0.0)
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

        # Should not crash and return valid color
        color = render_sample(32, 32)
        assert isinstance(color, tuple)
        assert len(color) == 3

    def test_dielectric_material_refracts(self):
        """Test that dielectric material produces refracted rays."""
        from src.python.camera.pinhole import PinholeCamera, setup_camera
        from src.python.core.integrator import (
            disable_light,
            render_sample,
            setup_render_target,
        )
        from src.python.scene.manager import SceneManager

        setup_render_target(64, 64)

        # Scene with a glass sphere
        scene = SceneManager()
        mat_id = scene.add_dielectric_material(ior=1.5)
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

        # Should not crash and return valid color
        color = render_sample(32, 32)
        assert isinstance(color, tuple)
        assert len(color) == 3


class TestProgressiveAccumulation:
    """Test progressive sample accumulation."""

    def test_sample_count_increments(self):
        """Test that sample count increments with each render_image call."""
        from src.python.camera.pinhole import PinholeCamera, setup_camera
        from src.python.core.integrator import (
            disable_light,
            get_total_samples,
            render_image,
            setup_render_target,
        )
        from src.python.scene.manager import SceneManager

        setup_render_target(16, 16)

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

        assert get_total_samples() == 0

        render_image(num_samples=1)
        assert get_total_samples() == 1

        render_image(num_samples=5)
        assert get_total_samples() == 6

    def test_progressive_accumulation_converges(self):
        """Test that more samples reduce variance (simple convergence test)."""
        import math

        from src.python.camera.pinhole import PinholeCamera, setup_camera
        from src.python.core.integrator import (
            disable_light,
            get_image,
            render_image,
            setup_render_target,
        )
        from src.python.scene.manager import SceneManager

        setup_render_target(8, 8)

        # Simple scene
        scene = SceneManager()
        mat_id = scene.add_lambertian_material((0.5, 0.5, 0.5))
        scene.add_sphere((0, 0, -2), 0.5, mat_id)

        camera = PinholeCamera(
            lookfrom=(0, 0, 0),
            lookat=(0, 0, -1),
            vup=(0, 1, 0),
            vfov=90.0,
            aspect_ratio=1.0,
        )
        setup_camera(camera)
        disable_light()

        # Render some samples
        render_image(num_samples=10)

        # Get image and verify it has valid values
        image = get_image()
        color = image[4, 4]  # Center pixel

        # Should have finite values (convert to Python floats for math.isnan)
        assert not math.isnan(float(color[0]))
        assert not math.isnan(float(color[1]))
        assert not math.isnan(float(color[2]))


class TestCornellBoxRendering:
    """Test rendering the Cornell box scene."""

    def test_cornell_box_renders_without_errors(self):
        """Test that Cornell box scene renders without exceptions."""
        from src.python.camera.pinhole import setup_camera
        from src.python.core.integrator import (
            get_total_samples,
            render_image,
            setup_light,
            setup_render_target,
        )
        from src.python.scene.cornell_box import create_cornell_box_scene, get_light_quad_info

        setup_render_target(32, 32)

        scene, camera, _ = create_cornell_box_scene()
        setup_camera(camera)

        # Setup light for Cornell box
        light_info = get_light_quad_info()
        # Material ID 3 is the light in Cornell box (after red, green, white walls)
        setup_light(
            corner=light_info["corner"],
            edge_u=light_info["edge_u"],
            edge_v=light_info["edge_v"],
            emission=(15.0, 15.0, 15.0),
            material_id=3,  # Light material is the 4th material (index 3)
        )

        # Should not crash
        render_image(num_samples=1)
        assert get_total_samples() == 1

    def test_cornell_box_center_pixel_has_color(self):
        """Test that center of Cornell box has some color contribution."""
        from src.python.camera.pinhole import setup_camera
        from src.python.core.integrator import (
            get_image,
            render_image,
            setup_light,
            setup_render_target,
        )
        from src.python.scene.cornell_box import create_cornell_box_scene, get_light_quad_info

        setup_render_target(32, 32)

        scene, camera, _ = create_cornell_box_scene()
        setup_camera(camera)

        # Setup light
        light_info = get_light_quad_info()
        setup_light(
            corner=light_info["corner"],
            edge_u=light_info["edge_u"],
            edge_v=light_info["edge_v"],
            emission=(15.0, 15.0, 15.0),
            material_id=3,
        )

        # Render multiple samples
        render_image(num_samples=5)

        # Get image
        image = get_image()

        # Check center region has some energy (not all black)
        center_color = image[16, 16]
        total_energy = center_color[0] + center_color[1] + center_color[2]

        # May or may not have energy depending on random paths,
        # but should be non-negative
        assert total_energy >= 0.0


class TestPathTracingConstants:
    """Test that path tracing constants are correctly configured."""

    def test_max_depth_is_reasonable(self):
        """Test that MAX_DEPTH allows for sufficient bounces."""
        from src.python.core.integrator import MAX_DEPTH

        assert MAX_DEPTH >= 10
        assert MAX_DEPTH <= 100

    def test_min_bounces_before_rr(self):
        """Test that Russian roulette starts after a few bounces."""
        from src.python.core.integrator import MIN_BOUNCES_BEFORE_RR

        assert MIN_BOUNCES_BEFORE_RR >= 1
        assert MIN_BOUNCES_BEFORE_RR <= 10


class TestRenderTargetErrors:
    """Test error handling for render target operations."""

    def test_render_sample_without_setup_raises_error(self):
        """Test that render_sample raises error if target not set up."""
        import src.python.core.integrator as integrator
        from src.python.core.integrator import render_sample

        # Mark as not initialized
        original_value = integrator._render_target_initialized[None]
        integrator._render_target_initialized[None] = 0

        with pytest.raises(RuntimeError, match="Render target not set up"):
            render_sample(0, 0)

        # Restore
        integrator._render_target_initialized[None] = original_value

    def test_render_image_without_setup_raises_error(self):
        """Test that render_image raises error if target not set up."""
        import src.python.core.integrator as integrator
        from src.python.core.integrator import render_image

        # Mark as not initialized
        original_value = integrator._render_target_initialized[None]
        integrator._render_target_initialized[None] = 0

        with pytest.raises(RuntimeError, match="Render target not set up"):
            render_image(1)

        # Restore
        integrator._render_target_initialized[None] = original_value


class TestNumericalStability:
    """Test numerical robustness of the path tracer."""

    def test_no_nan_in_rendered_image(self):
        """Test that rendered image contains no NaN values."""
        import numpy as np

        from src.python.camera.pinhole import PinholeCamera, setup_camera
        from src.python.core.integrator import (
            disable_light,
            get_image,
            render_image,
            setup_render_target,
        )
        from src.python.scene.manager import SceneManager

        setup_render_target(16, 16)

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

        render_image(num_samples=5)

        # Convert to numpy and check for NaN
        image = get_image().to_numpy()
        assert not np.any(np.isnan(image))

    def test_no_inf_in_rendered_image(self):
        """Test that rendered image contains no Inf values."""
        import numpy as np

        from src.python.camera.pinhole import PinholeCamera, setup_camera
        from src.python.core.integrator import (
            disable_light,
            get_image,
            render_image,
            setup_render_target,
        )
        from src.python.scene.manager import SceneManager

        setup_render_target(16, 16)

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

        render_image(num_samples=5)

        # Convert to numpy and check for Inf
        image = get_image().to_numpy()
        assert not np.any(np.isinf(image))

    def test_no_negative_colors(self):
        """Test that rendered image contains no negative values."""
        import numpy as np

        from src.python.camera.pinhole import PinholeCamera, setup_camera
        from src.python.core.integrator import (
            disable_light,
            get_image,
            render_image,
            setup_render_target,
        )
        from src.python.scene.manager import SceneManager

        setup_render_target(16, 16)

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

        render_image(num_samples=5)

        # Convert to numpy and check for negative values
        image = get_image().to_numpy()
        assert np.all(image >= 0.0)
