"""Geometry module for shape primitives and spatial acceleration.

This module provides geometric primitives and intersection algorithms:

Components:
    sphere: Sphere primitive with ray-sphere intersection
    quad: Axis-aligned and arbitrary quad primitives
    bvh: Bounding Volume Hierarchy for acceleration (future)
    aabb: Axis-Aligned Bounding Box utilities

All intersection routines are implemented as Taichi functions (@ti.func)
for GPU-accelerated parallel intersection testing. The primitives support
both closest-hit and any-hit queries for shadow rays.

Ray-object intersection follows the pattern:
    hit, t, normal, uv = intersect_shape(ray_origin, ray_direction, shape_data)
"""

# Imports will be added as modules are implemented
from .quad import Quad, hit_quad, make_quad, quad_area, quad_normal
from .sphere import HitRecord, Sphere, hit_sphere, make_sphere

__all__ = [
    "Sphere",
    "HitRecord",
    "hit_sphere",
    "make_sphere",
    "Quad",
    "hit_quad",
    "make_quad",
    "quad_area",
    "quad_normal",
]
