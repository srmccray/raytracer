"""Sphere primitive with robust ray-sphere intersection.

This module provides a Sphere dataclass and intersection function using the
robust quadratic formula from Ray Tracing Gems to avoid floating-point artifacts.

The robust quadratic formula avoids catastrophic cancellation when b^2 is
nearly equal to 4ac by using a reformulated calculation that maintains
numerical stability.

Example:
    >>> import taichi as ti
    >>> ti.init(arch=ti.gpu)
    >>> from src.python.geometry.sphere import Sphere, HitRecord, hit_sphere
    >>> sphere = Sphere(center=ti.math.vec3(0, 0, -1), radius=0.5)
    >>> # Use hit_sphere within a Taichi kernel
"""

import taichi as ti
import taichi.math as tm

# Type alias for 3D vectors using Taichi's math module
vec3 = tm.vec3


@ti.dataclass
class Sphere:
    """A sphere defined by center point and radius.

    Attributes:
        center: The center point of the sphere (vec3).
        radius: The radius of the sphere (positive float).
    """

    center: vec3
    radius: ti.f32


@ti.dataclass
class HitRecord:
    """Record of a ray-sphere intersection.

    Attributes:
        hit: Whether the ray intersected the sphere (1 if hit, 0 if miss).
        t: The parameter value along the ray where intersection occurred.
            Only valid if hit == 1.
        point: The 3D point where the ray intersected the sphere.
            Only valid if hit == 1.
        normal: The surface normal at the intersection point (unit length,
            always points outward from the sphere center).
            Only valid if hit == 1.
        front_face: Whether the ray hit the front face (1) or back face (0).
            Front face means the ray hit from outside the sphere.
            Only valid if hit == 1.
    """

    hit: ti.i32
    t: ti.f32
    point: vec3
    normal: vec3
    front_face: ti.i32


@ti.func
def _solve_quadratic_robust(h: ti.f32, a: ti.f32, c: ti.f32, sqrt_d: ti.f32):
    """Solve quadratic equation using robust formula from Ray Tracing Gems.

    Solves a*t^2 + 2*h*t + c = 0 using a numerically stable method.

    Args:
        h: Half of the linear coefficient.
        a: Quadratic coefficient.
        c: Constant term.
        sqrt_d: Square root of discriminant (h^2 - a*c).

    Returns:
        Tuple of (t0, t1) where t0 <= t1.
    """
    # Robust quadratic formula: use sign of h to avoid catastrophic cancellation
    # q = -(h + sign(h) * sqrt(discriminant))
    sign_h = ti.select(h < 0.0, -1.0, 1.0)
    q = -(h + sign_h * sqrt_d)

    # Handle degenerate case where q is near zero (tangent ray)
    t0 = 0.0
    t1 = 0.0

    if ti.abs(q) < 1e-10:
        # Fall back to standard formula for edge cases
        t0 = (-h - sqrt_d) / a
        t1 = (-h + sqrt_d) / a
    else:
        t0 = q / a
        t1 = c / q

    # Ensure t0 <= t1
    if t0 > t1:
        temp = t0
        t0 = t1
        t1 = temp

    return t0, t1


@ti.func
def hit_sphere(
    ray_origin: vec3,
    ray_direction: vec3,
    sphere: Sphere,
    t_min: ti.f32,
    t_max: ti.f32,
) -> HitRecord:
    """Test for ray-sphere intersection using robust quadratic formula.

    Uses the numerically stable quadratic formula from Ray Tracing Gems
    (Chapter 7) to avoid floating-point precision issues when the
    discriminant is near zero.

    The ray-sphere intersection is found by solving:
        |ray_origin + t * ray_direction - center|^2 = radius^2

    Expanding and rearranging gives the quadratic equation:
        a*t^2 + 2*h*t + c = 0

    where:
        a = dot(direction, direction)
        h = dot(direction, oc)  (half of traditional b)
        c = dot(oc, oc) - radius^2
        oc = origin - center

    The robust formula uses:
        q = -(h + sign(h) * sqrt(discriminant))
        t0 = q / a
        t1 = c / q

    Args:
        ray_origin: The starting point of the ray.
        ray_direction: The direction vector of the ray (need not be normalized,
            but normalized is more efficient).
        sphere: The sphere to test intersection against.
        t_min: Minimum t value to consider a valid hit (avoids self-intersection).
        t_max: Maximum t value to consider a valid hit (for shadow rays, etc.).

    Returns:
        A HitRecord containing intersection information. Check hit field
        to determine if intersection occurred.
    """
    # Vector from sphere center to ray origin
    oc = ray_origin - sphere.center

    # Quadratic coefficients using half-b formulation for numerical stability
    # a*t^2 + 2*h*t + c = 0
    a = tm.dot(ray_direction, ray_direction)
    h = tm.dot(ray_direction, oc)  # This is half of the traditional 'b'
    c = tm.dot(oc, oc) - sphere.radius * sphere.radius

    # Discriminant (using half-b form: h^2 - ac instead of b^2 - 4ac)
    discriminant = h * h - a * c

    # Initialize result fields (Taichi requires outer-scope declaration)
    did_hit = 0
    hit_t = 0.0
    hit_point = vec3(0.0, 0.0, 0.0)
    hit_normal = vec3(0.0, 0.0, 0.0)
    is_front_face = 0

    if discriminant >= 0.0:
        sqrt_d = ti.sqrt(discriminant)

        # Get the two roots using robust quadratic formula
        t0, t1 = _solve_quadratic_robust(h, a, c, sqrt_d)

        # Find the first valid intersection in [t_min, t_max]
        t = t0
        valid = (t > t_min) and (t < t_max)

        if not valid:
            t = t1
            valid = (t > t_min) and (t < t_max)

        if valid:
            # Valid intersection found
            did_hit = 1
            hit_t = t
            hit_point = ray_origin + t * ray_direction

            # Outward normal: points from center to hit point
            outward_normal = (hit_point - sphere.center) / sphere.radius

            # Determine if front face or back face hit
            # Front face: ray direction and normal point in opposite directions
            if tm.dot(ray_direction, outward_normal) > 0.0:
                # Ray is inside the sphere, hitting back face
                is_front_face = 0
                hit_normal = -outward_normal
            else:
                is_front_face = 1
                hit_normal = outward_normal

    return HitRecord(
        hit=did_hit,
        t=hit_t,
        point=hit_point,
        normal=hit_normal,
        front_face=is_front_face,
    )


@ti.func
def make_sphere(center: vec3, radius: ti.f32) -> Sphere:
    """Create a sphere from center and radius.

    This is a convenience function for creating spheres within Taichi kernels.

    Args:
        center: The center point of the sphere.
        radius: The radius of the sphere (should be positive).

    Returns:
        A new Sphere instance.
    """
    return Sphere(center=center, radius=radius)
