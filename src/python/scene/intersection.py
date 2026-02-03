"""Scene-level primitive intersection testing.

This module provides scene-level ray intersection testing that handles
multiple primitive types (spheres, quads) and returns the closest hit
with material information.

The scene stores primitives in Taichi fields for GPU-efficient access.
Each primitive has an associated material ID for shading.

Example:
    >>> import taichi as ti
    >>> ti.init(arch=ti.gpu)
    >>> from src.python.scene.intersection import (
    ...     SceneHitRecord, add_sphere, add_quad, intersect_scene, clear_scene
    ... )
    >>> clear_scene()
    >>> add_sphere(vec3(0, 0, -1), 0.5, material_id=0)
    >>> add_quad(vec3(-1, -0.5, -2), vec3(2, 0, 0), vec3(0, 1, 0), material_id=1)
    >>> # Use intersect_scene within a Taichi kernel
"""

import taichi as ti
import taichi.math as tm

from src.python.geometry.quad import Quad, hit_quad
from src.python.geometry.sphere import HitRecord, Sphere, hit_sphere

# Type alias for 3D vectors using Taichi's math module
vec3 = tm.vec3


@ti.dataclass
class SceneHitRecord:
    """Record of a ray-scene intersection with material information.

    Extends the basic HitRecord with material_id for scene-level queries.

    Attributes:
        hit: Whether the ray intersected any primitive (1 if hit, 0 if miss).
        t: The parameter value along the ray where intersection occurred.
            Only valid if hit == 1.
        point: The 3D point where the ray intersected the surface.
            Only valid if hit == 1.
        normal: The surface normal at the intersection point (unit length).
            Points toward the ray origin (outward for front face hits).
            Only valid if hit == 1.
        front_face: Whether the ray hit the front face (1) or back face (0).
            Only valid if hit == 1.
        material_id: The material ID of the hit primitive.
            Only valid if hit == 1. -1 indicates no material assigned.
    """

    hit: ti.i32
    t: ti.f32
    point: vec3
    normal: vec3
    front_face: ti.i32
    material_id: ti.i32


# Maximum number of primitives supported in the scene
MAX_SPHERES = 1024
MAX_QUADS = 1024

# Sphere storage: Structure of Arrays layout for GPU efficiency
sphere_centers = ti.Vector.field(3, dtype=ti.f32, shape=MAX_SPHERES)
sphere_radii = ti.field(dtype=ti.f32, shape=MAX_SPHERES)
sphere_material_ids = ti.field(dtype=ti.i32, shape=MAX_SPHERES)
num_spheres = ti.field(dtype=ti.i32, shape=())

# Quad storage: Structure of Arrays layout for GPU efficiency
# quad_corners stores the Q (corner point) of each quad
quad_corners = ti.Vector.field(3, dtype=ti.f32, shape=MAX_QUADS)
quad_edge_u = ti.Vector.field(3, dtype=ti.f32, shape=MAX_QUADS)
quad_edge_v = ti.Vector.field(3, dtype=ti.f32, shape=MAX_QUADS)
quad_material_ids = ti.field(dtype=ti.i32, shape=MAX_QUADS)
num_quads = ti.field(dtype=ti.i32, shape=())


def clear_scene() -> None:
    """Clear all primitives from the scene.

    Resets the primitive counts to zero. The actual field data is not
    cleared but will be overwritten when new primitives are added.
    """
    num_spheres[None] = 0
    num_quads[None] = 0


def add_sphere(center: vec3, radius: float, material_id: int = 0) -> int:
    """Add a sphere to the scene.

    Args:
        center: The center point of the sphere.
        radius: The radius of the sphere (should be positive).
        material_id: The material ID to associate with this sphere.

    Returns:
        The index of the added sphere.

    Raises:
        RuntimeError: If the maximum number of spheres is exceeded.
    """
    idx = num_spheres[None]
    if idx >= MAX_SPHERES:
        raise RuntimeError(f"Maximum number of spheres ({MAX_SPHERES}) exceeded")
    sphere_centers[idx] = center
    sphere_radii[idx] = radius
    sphere_material_ids[idx] = material_id
    num_spheres[None] = idx + 1
    return idx


def add_quad(q: vec3, u: vec3, v: vec3, material_id: int = 0) -> int:
    """Add a quad to the scene.

    The quad represents a parallelogram with vertices at Q, Q+u, Q+v, Q+u+v.

    Args:
        q: The corner point of the quad.
        u: Edge vector from q to adjacent corner.
        v: Edge vector from q to other adjacent corner.
        material_id: The material ID to associate with this quad.

    Returns:
        The index of the added quad.

    Raises:
        RuntimeError: If the maximum number of quads is exceeded.
    """
    idx = num_quads[None]
    if idx >= MAX_QUADS:
        raise RuntimeError(f"Maximum number of quads ({MAX_QUADS}) exceeded")
    quad_corners[idx] = q
    quad_edge_u[idx] = u
    quad_edge_v[idx] = v
    quad_material_ids[idx] = material_id
    num_quads[None] = idx + 1
    return idx


def get_sphere_count() -> int:
    """Get the number of spheres in the scene."""
    return int(num_spheres[None])


def get_quad_count() -> int:
    """Get the number of quads in the scene."""
    return int(num_quads[None])


@ti.func
def _hit_record_to_scene_hit_record(rec: HitRecord, material_id: ti.i32) -> SceneHitRecord:
    """Convert a HitRecord to a SceneHitRecord with material ID.

    Args:
        rec: The basic hit record from primitive intersection.
        material_id: The material ID of the hit primitive.

    Returns:
        A SceneHitRecord with the same hit data plus material_id.
    """
    return SceneHitRecord(
        hit=rec.hit,
        t=rec.t,
        point=rec.point,
        normal=rec.normal,
        front_face=rec.front_face,
        material_id=material_id,
    )


@ti.func
def _make_miss_record() -> SceneHitRecord:
    """Create a SceneHitRecord indicating no intersection.

    Returns:
        A SceneHitRecord with hit=0 and default values.
    """
    return SceneHitRecord(
        hit=0,
        t=0.0,
        point=vec3(0.0, 0.0, 0.0),
        normal=vec3(0.0, 0.0, 0.0),
        front_face=0,
        material_id=-1,
    )


@ti.func
def intersect_scene(
    ray_origin: vec3,
    ray_direction: vec3,
    t_min: ti.f32,
    t_max: ti.f32,
) -> SceneHitRecord:
    """Test ray against all primitives in the scene.

    Iterates through all spheres and quads, testing each for intersection
    and tracking the closest hit (smallest t > t_min).

    Args:
        ray_origin: The starting point of the ray.
        ray_direction: The direction vector of the ray.
        t_min: Minimum t value to consider a valid hit.
        t_max: Maximum t value to consider a valid hit.

    Returns:
        A SceneHitRecord containing the closest intersection, or a miss
        record if no intersection was found.
    """
    # Track the closest hit so far
    closest_t = t_max
    result = _make_miss_record()

    # Test all spheres
    n_spheres = num_spheres[None]
    for i in range(n_spheres):
        sphere = Sphere(center=sphere_centers[i], radius=sphere_radii[i])
        rec = hit_sphere(ray_origin, ray_direction, sphere, t_min, closest_t)
        if rec.hit == 1:
            closest_t = rec.t
            result = _hit_record_to_scene_hit_record(rec, sphere_material_ids[i])

    # Test all quads
    n_quads = num_quads[None]
    for i in range(n_quads):
        quad = Quad(Q=quad_corners[i], u=quad_edge_u[i], v=quad_edge_v[i])
        rec = hit_quad(ray_origin, ray_direction, quad, t_min, closest_t)
        if rec.hit == 1:
            closest_t = rec.t
            result = _hit_record_to_scene_hit_record(rec, quad_material_ids[i])

    return result


@ti.func
def intersect_scene_any(
    ray_origin: vec3,
    ray_direction: vec3,
    t_min: ti.f32,
    t_max: ti.f32,
) -> ti.i32:
    """Test if ray hits any primitive in the scene (shadow ray query).

    This is an optimized version that returns early on the first hit,
    useful for shadow ray testing where we only need to know if anything
    blocks the ray.

    Args:
        ray_origin: The starting point of the ray.
        ray_direction: The direction vector of the ray.
        t_min: Minimum t value to consider a valid hit.
        t_max: Maximum t value to consider a valid hit.

    Returns:
        1 if any primitive was hit, 0 otherwise.
    """
    hit_any = 0

    # Test all spheres (early exit on first hit)
    n_spheres = num_spheres[None]
    for i in range(n_spheres):
        if hit_any == 0:
            sphere = Sphere(center=sphere_centers[i], radius=sphere_radii[i])
            rec = hit_sphere(ray_origin, ray_direction, sphere, t_min, t_max)
            if rec.hit == 1:
                hit_any = 1

    # Test all quads (early exit on first hit)
    n_quads = num_quads[None]
    for i in range(n_quads):
        if hit_any == 0:
            quad = Quad(Q=quad_corners[i], u=quad_edge_u[i], v=quad_edge_v[i])
            rec = hit_quad(ray_origin, ray_direction, quad, t_min, t_max)
            if rec.hit == 1:
                hit_any = 1

    return hit_any
