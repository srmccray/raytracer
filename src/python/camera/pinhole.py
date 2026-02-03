"""Pinhole camera model for perspective projection ray generation.

This module implements a pinhole camera that generates primary rays for rendering.
The camera supports:
- Look-at positioning (lookfrom, lookat, vup)
- Vertical field of view specification
- Arbitrary aspect ratios
- Jittered sampling for anti-aliasing

The camera builds an orthonormal basis (u, v, w) from the view parameters:
- w: points from lookat toward lookfrom (opposite view direction)
- u: points right in the image plane
- v: points up in the image plane

All ray generation is Taichi-compatible for GPU acceleration.

Example:
    >>> import taichi as ti
    >>> ti.init(arch=ti.gpu)
    >>> from src.python.camera.pinhole import PinholeCamera, setup_camera, get_ray
    >>>
    >>> # Create camera looking at origin from z=3
    >>> camera = PinholeCamera(
    ...     lookfrom=(0.0, 0.0, 3.0),
    ...     lookat=(0.0, 0.0, 0.0),
    ...     vup=(0.0, 1.0, 0.0),
    ...     vfov=60.0,
    ...     aspect_ratio=16.0/9.0
    ... )
    >>> setup_camera(camera)
    >>>
    >>> # Generate ray for pixel center (0.5, 0.5)
    >>> @ti.kernel
    ... def render():
    ...     ray = get_ray(0.5, 0.5)  # Ray through image center
"""

import math
from dataclasses import dataclass

import numpy as np
import taichi as ti
import taichi.math as tm

from src.python.core.ray import Ray, make_ray, vec3

# =============================================================================
# Camera Data Structures
# =============================================================================


@dataclass
class PinholeCamera:
    """Configuration for a pinhole (perspective) camera.

    A pinhole camera produces perfect perspective projection with no
    depth of field effects. It is the simplest camera model for ray tracing.

    Attributes:
        lookfrom: Camera position in world space (x, y, z).
        lookat: Point the camera is looking at in world space (x, y, z).
        vup: Up direction vector for camera orientation (typically (0, 1, 0)).
        vfov: Vertical field of view in degrees (typically 40-90).
        aspect_ratio: Width divided by height of the output image.
    """

    lookfrom: tuple[float, float, float]
    lookat: tuple[float, float, float]
    vup: tuple[float, float, float]
    vfov: float
    aspect_ratio: float


# =============================================================================
# Taichi Fields for Camera State (GPU-accessible)
# =============================================================================

# Camera origin (position)
_camera_origin = ti.Vector.field(3, dtype=ti.f32, shape=())

# Orthonormal basis vectors
_camera_u = ti.Vector.field(3, dtype=ti.f32, shape=())  # Right
_camera_v = ti.Vector.field(3, dtype=ti.f32, shape=())  # Up
_camera_w = ti.Vector.field(3, dtype=ti.f32, shape=())  # Backward (opposite view)

# Viewport vectors for ray computation
_viewport_horizontal = ti.Vector.field(3, dtype=ti.f32, shape=())  # Full width
_viewport_vertical = ti.Vector.field(3, dtype=ti.f32, shape=())  # Full height
_lower_left_corner = ti.Vector.field(3, dtype=ti.f32, shape=())  # Lower-left of viewport


# =============================================================================
# Camera Setup (Python-side, called once per camera configuration)
# =============================================================================


def setup_camera(camera: PinholeCamera) -> None:
    """Initialize camera state from configuration.

    Computes the camera's orthonormal basis (u, v, w) and viewport geometry
    from the provided camera parameters. This must be called before rendering.

    The viewport is a virtual image plane at unit distance from the camera.
    Ray directions are computed by interpolating across this viewport.

    Args:
        camera: Camera configuration with position, orientation, and FOV.

    Note:
        This function writes to Taichi fields and should be called from
        Python (not from within a Taichi kernel).
    """
    # Convert FOV from degrees to radians
    theta = math.radians(camera.vfov)
    h = math.tan(theta / 2.0)

    # Viewport dimensions at unit distance
    viewport_height = 2.0 * h
    viewport_width = camera.aspect_ratio * viewport_height

    # Build orthonormal basis using NumPy (Python-side computation)
    lookfrom = np.array(camera.lookfrom, dtype=np.float32)
    lookat = np.array(camera.lookat, dtype=np.float32)
    vup = np.array(camera.vup, dtype=np.float32)

    # w points from lookat toward lookfrom (backward)
    w = lookfrom - lookat
    w = w / np.linalg.norm(w)

    # u points right (perpendicular to w and vup)
    u = np.cross(vup, w)
    u = u / np.linalg.norm(u)

    # v points up in the camera's frame
    v = np.cross(w, u)

    # Store basis vectors in Taichi fields
    _camera_origin[None] = lookfrom.tolist()
    _camera_u[None] = u.tolist()
    _camera_v[None] = v.tolist()
    _camera_w[None] = w.tolist()

    # Compute viewport vectors
    # horizontal spans the full width of the viewport
    horizontal = viewport_width * u
    # vertical spans the full height of the viewport
    vertical = viewport_height * v

    # Lower-left corner of the viewport (at z = -1 in camera space)
    # Origin - w (move forward) - horizontal/2 (left) - vertical/2 (down)
    lower_left = lookfrom - w - horizontal / 2.0 - vertical / 2.0

    _viewport_horizontal[None] = horizontal.tolist()
    _viewport_vertical[None] = vertical.tolist()
    _lower_left_corner[None] = lower_left.tolist()


# =============================================================================
# Ray Generation (Taichi-compatible, GPU-callable)
# =============================================================================


@ti.func
def get_ray(u: ti.f32, v: ti.f32) -> Ray:
    """Generate a ray through normalized image coordinates (u, v).

    Computes a ray from the camera origin through the point (u, v) on the
    virtual image plane. The coordinates are normalized:
    - u = 0: left edge of image
    - u = 1: right edge of image
    - v = 0: bottom edge of image
    - v = 1: top edge of image

    This function is designed to be called from within Taichi kernels.

    Args:
        u: Horizontal coordinate in [0, 1] (left to right).
        v: Vertical coordinate in [0, 1] (bottom to top).

    Returns:
        A Ray with origin at the camera position and direction toward
        the specified point on the image plane.
    """
    # Point on the viewport
    point_on_viewport = (
        _lower_left_corner[None] + u * _viewport_horizontal[None] + v * _viewport_vertical[None]
    )

    # Direction from camera origin to point on viewport
    origin = _camera_origin[None]
    direction = tm.normalize(point_on_viewport - origin)

    return make_ray(origin, direction)


@ti.func
def get_ray_jittered(pixel_i: ti.i32, pixel_j: ti.i32, width: ti.i32, height: ti.i32) -> Ray:
    """Generate a jittered ray for anti-aliasing.

    Adds random sub-pixel offset to the ray for stochastic anti-aliasing.
    When accumulated over multiple samples, this produces smooth edges.

    The jitter is uniformly distributed within the pixel area:
    - Adds random offset in [0, 1) to pixel coordinates
    - Converts to normalized [0, 1] image coordinates

    Args:
        pixel_i: Pixel x-coordinate (0 = left).
        pixel_j: Pixel y-coordinate (0 = bottom).
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        A Ray with random sub-pixel offset for anti-aliasing.

    Example:
        @ti.kernel
        def render():
            for i, j in image:
                ray = get_ray_jittered(i, j, width, height)
                color = trace(ray)
    """
    # Add random jitter within pixel [0, 1)
    jitter_u = ti.random(ti.f32)
    jitter_v = ti.random(ti.f32)

    # Convert pixel + jitter to normalized coordinates
    u = (ti.cast(pixel_i, ti.f32) + jitter_u) / ti.cast(width, ti.f32)
    v = (ti.cast(pixel_j, ti.f32) + jitter_v) / ti.cast(height, ti.f32)

    return get_ray(u, v)


@ti.func
def get_camera_origin() -> vec3:
    """Get the camera origin (position) in world space.

    Useful for computing view-dependent effects or checking ray distances.

    Returns:
        The camera position as a vec3.
    """
    return _camera_origin[None]


@ti.func
def get_camera_basis():
    """Get the camera's orthonormal basis vectors.

    Returns:
        A tuple (u, v, w) where:
        - u: Right direction in world space
        - v: Up direction in world space
        - w: Backward direction (opposite view direction)
    """
    return _camera_u[None], _camera_v[None], _camera_w[None]


# =============================================================================
# Utility Functions
# =============================================================================


def get_camera_info() -> dict[str, tuple[float, float, float]]:
    """Get current camera state for debugging.

    Returns a dictionary with camera vectors that can be inspected
    from Python. Useful for verifying camera setup.

    Returns:
        Dictionary with origin, u, v, w, horizontal, vertical, lower_left.
    """
    origin_vec = _camera_origin[None]
    u_vec = _camera_u[None]
    v_vec = _camera_v[None]
    w_vec = _camera_w[None]
    h_vec = _viewport_horizontal[None]
    vert_vec = _viewport_vertical[None]
    ll_vec = _lower_left_corner[None]

    return {
        "origin": (float(origin_vec[0]), float(origin_vec[1]), float(origin_vec[2])),
        "u": (float(u_vec[0]), float(u_vec[1]), float(u_vec[2])),
        "v": (float(v_vec[0]), float(v_vec[1]), float(v_vec[2])),
        "w": (float(w_vec[0]), float(w_vec[1]), float(w_vec[2])),
        "horizontal": (float(h_vec[0]), float(h_vec[1]), float(h_vec[2])),
        "vertical": (float(vert_vec[0]), float(vert_vec[1]), float(vert_vec[2])),
        "lower_left": (float(ll_vec[0]), float(ll_vec[1]), float(ll_vec[2])),
    }
