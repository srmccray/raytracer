"""Pytest configuration for raytracer tests.

This module provides shared fixtures for all test modules, including
Taichi initialization which must happen once per session.
"""

import pytest
import taichi as ti


@pytest.fixture(scope="session", autouse=True)
def init_taichi_session():
    """Initialize Taichi once for the entire test session.

    Using session scope prevents multiple ti.init() calls which can cause
    segmentation faults due to Taichi runtime conflicts.
    """
    ti.init(arch=ti.cpu, random_seed=42)
    yield
    # Note: We don't call ti.reset() here as it can cause issues
    # with subsequent tests if any cleanup happens after


@pytest.fixture(autouse=True)
def clear_all_scene_data():
    """Clear scene data before each test.

    This ensures tests are isolated from each other.
    """
    # Import here to avoid circular imports and ensure Taichi is initialized
    from src.python.materials.dielectric import clear_dielectric_materials
    from src.python.materials.lambertian import clear_lambertian_materials
    from src.python.materials.metal import clear_metal_materials
    from src.python.scene.intersection import clear_scene
    from src.python.scene.manager import _clear_material_tracking

    def _clear_all():
        clear_scene()
        clear_lambertian_materials()
        clear_metal_materials()
        clear_dielectric_materials()
        _clear_material_tracking()

        # Also clear integrator render target if it exists
        try:
            from src.python.core.integrator import clear_render_target, disable_light

            clear_render_target()
            disable_light()
        except (ImportError, RuntimeError):
            # Integrator not initialized or render target not set up yet
            pass

    # Clear everything before test
    _clear_all()

    yield

    # Clear everything after test
    _clear_all()
