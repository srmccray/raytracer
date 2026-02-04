"""Cornell box scene configuration.

This module provides a factory function to create the classic Cornell box scene,
a standard test scene used in computer graphics for evaluating global illumination
algorithms.

The Cornell box consists of:
- 5 walls forming an open box (left, right, back, floor, ceiling)
- Left wall: red diffuse
- Right wall: green diffuse
- Back, floor, ceiling: white diffuse
- 3 spheres with different materials (diffuse, metal, glass)
- Area light on the ceiling (emissive quad)

The classic Cornell box dimensions are approximately 555x555x555 units, but we
use a normalized scale where the box spans from 0 to 555 in each dimension,
with the camera positioned outside looking in through the open front.

Example:
    >>> import taichi as ti
    >>> ti.init(arch=ti.gpu)
    >>> from src.python.scene.cornell_box import create_cornell_box_scene
    >>> from src.python.camera.pinhole import PinholeCamera, setup_camera
    >>>
    >>> scene, camera = create_cornell_box_scene()
    >>> setup_camera(camera)
    >>> # Now render using the scene and camera
"""

from dataclasses import dataclass

from src.python.camera.pinhole import PinholeCamera
from src.python.scene.manager import SceneManager

# =============================================================================
# Cornell Box Parameters (for interactive preview)
# =============================================================================


@dataclass
class CornellBoxParams:
    """Parameters for configuring a Cornell box scene.

    This dataclass provides a convenient way to customize the Cornell box
    scene for interactive preview and rendering. All parameters have defaults
    matching the classic Cornell box configuration.

    Attributes:
        light_intensity: The intensity/brightness of the area light.
            Default is 15.0, which provides good illumination for the scene.
        light_color: RGB color of the light (each component in [0, 1]).
            Default is white (1.0, 1.0, 1.0).
        left_wall_color: RGB albedo of the left wall.
            Default is green (0.12, 0.45, 0.15).
        right_wall_color: RGB albedo of the right wall.
            Default is red (0.65, 0.05, 0.05).
        back_wall_color: RGB albedo of the back wall.
            Default is white (0.73, 0.73, 0.73).

    Example:
        >>> params = CornellBoxParams()
        >>> params.light_intensity
        15.0
        >>> params.left_wall_color
        (0.12, 0.45, 0.15)

        >>> # Custom parameters
        >>> custom = CornellBoxParams(
        ...     light_intensity=20.0,
        ...     light_color=(1.0, 0.9, 0.8),  # Warm light
        ...     left_wall_color=(0.2, 0.2, 0.8),  # Blue wall
        ... )
    """

    light_intensity: float = 15.0
    light_color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    left_wall_color: tuple[float, float, float] = (0.12, 0.45, 0.15)
    right_wall_color: tuple[float, float, float] = (0.65, 0.05, 0.05)
    back_wall_color: tuple[float, float, float] = (0.73, 0.73, 0.73)

# =============================================================================
# Cornell Box Constants
# =============================================================================

# Classic Cornell box dimensions (approximately 555x555x555 units)
BOX_SIZE = 555.0

# Wall colors (normalized RGB values matching original Cornell box measurements)
RED_WALL_ALBEDO = (0.65, 0.05, 0.05)
GREEN_WALL_ALBEDO = (0.12, 0.45, 0.15)
WHITE_WALL_ALBEDO = (0.73, 0.73, 0.73)

# Light color (bright white emission - handled by renderer, not material)
# For now, we use a bright white Lambertian as placeholder for emissive
LIGHT_ALBEDO = (1.0, 1.0, 1.0)

# Sphere materials
DIFFUSE_SPHERE_ALBEDO = (0.73, 0.73, 0.73)  # White diffuse
GLASS_SPHERE_IOR = 1.5  # Standard glass

# Metal sphere parameters (silver/chrome appearance)
METAL_SPHERE_ALBEDO = (0.95, 0.93, 0.88)  # Silver reflectance
METAL_SPHERE_ROUGHNESS = 0.3  # Slightly rough for soft reflections


# =============================================================================
# Cornell Box Factory
# =============================================================================


def create_cornell_box_scene(
    box_size: float = BOX_SIZE,
    params: CornellBoxParams | None = None,
) -> tuple[SceneManager, PinholeCamera, int]:
    """Create a Cornell box scene with standard configuration.

    Creates the classic Cornell box test scene with:
    - Red left wall, green right wall (colors configurable via params)
    - White back wall, floor, and ceiling (back wall color configurable)
    - Area light on ceiling (emissive, intensity/color configurable via params)
    - Three spheres: diffuse (white), metal (silver), glass

    The coordinate system places the box origin at (0, 0, 0) with:
    - X-axis: left to right (0 to box_size)
    - Y-axis: floor to ceiling (0 to box_size)
    - Z-axis: front to back (0 to box_size), camera looks toward +Z

    Args:
        box_size: The size of the box in each dimension. Default is 555.0
            (classic Cornell box dimensions).
        params: Optional CornellBoxParams for customizing light and wall colors.
            If None, uses default CornellBoxParams().

    Returns:
        A tuple of (SceneManager, PinholeCamera, light_material_id) where:
        - SceneManager contains all geometry and materials
        - PinholeCamera is configured for the standard view
        - light_material_id is the material ID of the area light (for dynamic updates)

    Example:
        >>> scene, camera, light_mat_id = create_cornell_box_scene()
        >>> print(f"Scene has {scene.get_quad_count()} quads")
        Scene has 6 quads
        >>> print(f"Scene has {scene.get_sphere_count()} spheres")
        Scene has 3 spheres

        >>> # With custom parameters
        >>> custom_params = CornellBoxParams(light_intensity=20.0)
        >>> scene, camera, light_mat_id = create_cornell_box_scene(params=custom_params)
    """
    # Use default params if none provided
    if params is None:
        params = CornellBoxParams()

    scene = SceneManager()

    # =========================================================================
    # Materials
    # =========================================================================

    # Wall materials (use params for configurable colors)
    # Note: left_wall uses right_wall_color (red) and right_wall uses left_wall_color (green)
    # because of the classic Cornell box convention where red is on the right as you look in
    red_mat = scene.add_lambertian_material(albedo=params.right_wall_color)
    green_mat = scene.add_lambertian_material(albedo=params.left_wall_color)
    white_mat = scene.add_lambertian_material(albedo=params.back_wall_color)

    # Light material (emissive - uses phosphorescent material for emission support)
    # The phosphorescent material computes emission as: glow_color * glow_intensity
    light_mat = scene.add_phosphorescent_material(
        albedo=LIGHT_ALBEDO,
        glow_color=params.light_color,
        glow_intensity=params.light_intensity,
    )

    # Sphere materials
    diffuse_mat = scene.add_lambertian_material(albedo=DIFFUSE_SPHERE_ALBEDO)
    metal_mat = scene.add_metal_material(
        albedo=METAL_SPHERE_ALBEDO,
        roughness=METAL_SPHERE_ROUGHNESS,
    )
    glass_mat = scene.add_dielectric_material(ior=GLASS_SPHERE_IOR)

    # =========================================================================
    # Walls (5 quads forming the box)
    # =========================================================================

    # Left wall (red) - YZ plane at x=0
    # Corner at (0, 0, 0), spans Y and Z
    scene.add_quad(
        corner=(0.0, 0.0, 0.0),
        edge_u=(0.0, box_size, 0.0),  # Up
        edge_v=(0.0, 0.0, box_size),  # Back
        material_id=red_mat,
    )

    # Right wall (green) - YZ plane at x=box_size
    # Corner at (box_size, 0, box_size), spans Y and -Z (to face inward)
    scene.add_quad(
        corner=(box_size, 0.0, box_size),
        edge_u=(0.0, box_size, 0.0),  # Up
        edge_v=(0.0, 0.0, -box_size),  # Front
        material_id=green_mat,
    )

    # Back wall (white) - XY plane at z=box_size
    # Corner at (0, 0, box_size), spans X and Y
    scene.add_quad(
        corner=(0.0, 0.0, box_size),
        edge_u=(box_size, 0.0, 0.0),  # Right
        edge_v=(0.0, box_size, 0.0),  # Up
        material_id=white_mat,
    )

    # Floor (white) - XZ plane at y=0
    # Corner at (0, 0, 0), spans X and Z
    scene.add_quad(
        corner=(0.0, 0.0, 0.0),
        edge_u=(box_size, 0.0, 0.0),  # Right
        edge_v=(0.0, 0.0, box_size),  # Back
        material_id=white_mat,
    )

    # Ceiling (white) - XZ plane at y=box_size
    # Corner at (0, box_size, box_size), spans X and -Z (to face downward)
    scene.add_quad(
        corner=(0.0, box_size, box_size),
        edge_u=(box_size, 0.0, 0.0),  # Right
        edge_v=(0.0, 0.0, -box_size),  # Front
        material_id=white_mat,
    )

    # =========================================================================
    # Area Light (on ceiling)
    # =========================================================================

    # Small light quad centered on ceiling
    # Light is smaller than ceiling (classic Cornell box light is ~130x105 units)
    light_width = 130.0
    light_depth = 105.0
    light_x_offset = (box_size - light_width) / 2.0
    light_z_offset = (box_size - light_depth) / 2.0

    # Light sits just below ceiling to avoid z-fighting
    light_y = box_size - 1.0

    scene.add_quad(
        corner=(light_x_offset, light_y, light_z_offset),
        edge_u=(light_width, 0.0, 0.0),  # Right
        edge_v=(0.0, 0.0, light_depth),  # Back
        material_id=light_mat,
    )

    # =========================================================================
    # Spheres (3 spheres with different materials)
    # =========================================================================

    # Sphere positions - arranged in the lower portion of the box
    # Classic Cornell box has tall and short blocks; we use spheres instead

    # Diffuse sphere (white) - left side, on floor
    diffuse_sphere_radius = 80.0
    diffuse_sphere_center = (
        box_size * 0.27,  # Left of center
        diffuse_sphere_radius,  # Resting on floor
        box_size * 0.35,  # Toward front
    )
    scene.add_sphere(
        center=diffuse_sphere_center,
        radius=diffuse_sphere_radius,
        material_id=diffuse_mat,
    )

    # Metal sphere (silver) - right side, on floor
    metal_sphere_radius = 80.0
    metal_sphere_center = (
        box_size * 0.73,  # Right of center
        metal_sphere_radius,  # Resting on floor
        box_size * 0.35,  # Toward front
    )
    scene.add_sphere(
        center=metal_sphere_center,
        radius=metal_sphere_radius,
        material_id=metal_mat,
    )

    # Glass sphere - center, on floor
    glass_sphere_radius = 80.0
    glass_sphere_center = (
        box_size * 0.5,  # Center
        glass_sphere_radius,  # Resting on floor
        box_size * 0.65,  # Toward back
    )
    scene.add_sphere(
        center=glass_sphere_center,
        radius=glass_sphere_radius,
        material_id=glass_mat,
    )

    # =========================================================================
    # Camera Setup
    # =========================================================================

    # Camera positioned outside the box, looking in through the open front
    # Classic Cornell box camera is at approximately z=-800 from box center
    camera_distance = 800.0
    camera = PinholeCamera(
        lookfrom=(
            box_size / 2.0,  # Centered on X
            box_size / 2.0,  # Centered on Y
            -camera_distance,  # In front of box
        ),
        lookat=(
            box_size / 2.0,  # Look at center X
            box_size / 2.0,  # Look at center Y
            box_size / 2.0,  # Look at center Z (middle of box)
        ),
        vup=(0.0, 1.0, 0.0),  # Y is up
        vfov=40.0,  # Classic Cornell box FOV
        aspect_ratio=1.0,  # Square image (classic Cornell box)
    )

    return scene, camera, light_mat


def get_light_quad_info(box_size: float = BOX_SIZE) -> dict[str, tuple[float, float, float]]:
    """Get the area light quad geometry for importance sampling.

    Returns the corner and edge vectors of the ceiling light quad,
    useful for implementing light sampling in the path tracer.

    Args:
        box_size: The size of the box. Default is 555.0.

    Returns:
        A dictionary with keys:
        - 'corner': The corner point (x, y, z) of the light quad
        - 'edge_u': The first edge vector (light width direction)
        - 'edge_v': The second edge vector (light depth direction)
        - 'center': The center point of the light
        - 'area': The area of the light (as a single float in a tuple for consistency)
    """
    light_width = 130.0
    light_depth = 105.0
    light_x_offset = (box_size - light_width) / 2.0
    light_z_offset = (box_size - light_depth) / 2.0
    light_y = box_size - 1.0

    corner = (light_x_offset, light_y, light_z_offset)
    edge_u = (light_width, 0.0, 0.0)
    edge_v = (0.0, 0.0, light_depth)

    center = (
        light_x_offset + light_width / 2.0,
        light_y,
        light_z_offset + light_depth / 2.0,
    )

    area = light_width * light_depth

    return {
        "corner": corner,
        "edge_u": edge_u,
        "edge_v": edge_v,
        "center": center,
        "area": (area, 0.0, 0.0),  # Packed as tuple for consistency
    }


def get_cornell_box_bounds(box_size: float = BOX_SIZE) -> dict[str, tuple[float, float, float]]:
    """Get the bounding box of the Cornell box scene.

    Useful for setting up BVH bounds or camera near/far planes.

    Args:
        box_size: The size of the box. Default is 555.0.

    Returns:
        A dictionary with keys:
        - 'min': The minimum corner (0, 0, 0)
        - 'max': The maximum corner (box_size, box_size, box_size)
        - 'center': The center of the box
        - 'size': The size in each dimension
    """
    return {
        "min": (0.0, 0.0, 0.0),
        "max": (box_size, box_size, box_size),
        "center": (box_size / 2.0, box_size / 2.0, box_size / 2.0),
        "size": (box_size, box_size, box_size),
    }
