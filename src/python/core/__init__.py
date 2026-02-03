"""Core rendering module.

This module contains the fundamental building blocks for ray tracing:

Components:
    ray: Ray data structure and ray generation utilities
    intersection: Hit record structures and intersection testing
    integrator: Light transport algorithms (path tracing, Whitted-style)
    sampler: Random number generation and stratified sampling
    spectrum: Color representation and spectral utilities

The core module handles the rendering equation integration, implementing
Monte Carlo path tracing with Russian roulette termination, next event
estimation, and progressive accumulation for anti-aliasing.

All compute-intensive operations use Taichi kernels for GPU acceleration.
"""

# Imports will be added as modules are implemented
from .ray import (
    Ray,
    build_onb_from_normal,
    cross,
    dot,
    length,
    length_squared,
    local_to_world,
    make_ray,
    near_zero,
    normalize,
    random_cosine_direction,
    random_in_unit_disk,
    random_in_unit_sphere,
    random_on_hemisphere,
    random_unit_vector,
    ray_at,
    reflect,
    refract,
    sample_cosine_hemisphere,
    schlick_fresnel,
    vec3,
)

# Note: integrator and progressive are NOT imported here to avoid circular imports.
# Import directly from src.python.core.integrator or src.python.core.progressive when needed.
#
# For progressive rendering, use:
#   from src.python.core.progressive import ProgressiveRenderer

__all__ = [
    "Ray",
    "ray_at",
    "make_ray",
    "vec3",
    "length",
    "length_squared",
    "normalize",
    "dot",
    "cross",
    "reflect",
    "refract",
    "schlick_fresnel",
    "near_zero",
    "random_in_unit_sphere",
    "random_unit_vector",
    "random_on_hemisphere",
    "random_in_unit_disk",
    "random_cosine_direction",
    "build_onb_from_normal",
    "local_to_world",
    "sample_cosine_hemisphere",
]
