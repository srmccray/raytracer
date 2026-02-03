"""Camera module for view and ray generation.

This module provides camera models for generating primary rays:

Components:
    camera: Base camera interface
    pinhole: Simple pinhole (perspective) camera model
    thin_lens: Camera with depth of field (future)

Camera responsibilities:
    - Transform (u, v) image coordinates to world-space rays
    - Apply anti-aliasing jitter for sub-pixel sampling
    - Support look-at positioning with up vector
    - Compute field of view from sensor and focal length

Ray generation uses normalized device coordinates:
    u in [0, 1]: left to right across image
    v in [0, 1]: bottom to top across image

All camera ray generation is vectorized in Taichi kernels for
parallel primary ray creation across all pixels.
"""

# Imports will be added as modules are implemented
from .pinhole import (
    PinholeCamera,
    get_camera_basis,
    get_camera_info,
    get_camera_origin,
    get_ray,
    get_ray_jittered,
    setup_camera,
)

__all__ = [
    "PinholeCamera",
    "setup_camera",
    "get_ray",
    "get_ray_jittered",
    "get_camera_origin",
    "get_camera_basis",
    "get_camera_info",
]
