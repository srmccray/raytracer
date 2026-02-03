"""Quad primitive with ray-quad intersection.

This module provides a Quad dataclass and intersection function for rendering
rectangular surfaces such as the walls of a Cornell box.

A quad is defined by:
- Q: A corner point of the quad
- u: Edge vector from Q to adjacent corner
- v: Edge vector from Q to other adjacent corner

The quad spans the parallelogram from Q to Q+u+v. The normal is computed as
normalize(cross(u, v)), pointing in the direction determined by the right-hand
rule.

Ray-quad intersection uses the parametric plane test:
1. Find where ray intersects the plane containing the quad
2. Check if the intersection point lies within the quad bounds

Example:
    >>> import taichi as ti
    >>> ti.init(arch=ti.gpu)
    >>> from src.python.geometry.quad import Quad, hit_quad
    >>> # Floor quad at y=0, spanning x=[0,1] and z=[0,1]
    >>> quad = Quad(
    ...     Q=ti.math.vec3(0, 0, 0),
    ...     u=ti.math.vec3(1, 0, 0),
    ...     v=ti.math.vec3(0, 0, 1)
    ... )
    >>> # Use hit_quad within a Taichi kernel
"""

import taichi as ti
import taichi.math as tm

from .sphere import HitRecord

# Type alias for 3D vectors using Taichi's math module
vec3 = tm.vec3


@ti.dataclass
class Quad:
    """A quad (parallelogram) defined by a corner point and two edge vectors.

    The quad represents the parallelogram with vertices at:
        Q, Q+u, Q+v, Q+u+v

    Attributes:
        Q: The corner point of the quad (vec3).
        u: Edge vector from Q to adjacent corner (vec3).
        v: Edge vector from Q to other adjacent corner (vec3).
    """

    Q: vec3
    u: vec3
    v: vec3


@ti.func
def _compute_quad_frame(quad: Quad):
    """Compute the quad's plane normal and basis vectors for intersection.

    For a quad with edges u and v, we compute:
    - normal: The plane normal (u x v), normalized
    - w: A helper vector for computing barycentric coordinates

    The intersection point P can be expressed as:
        P = Q + alpha * u + beta * v

    We solve for alpha and beta using the dot product with w vectors
    derived from the plane's parametric form.

    Args:
        quad: The quad to compute frame for.

    Returns:
        Tuple of (normal, D, w_u, w_v) where:
        - normal: Unit normal vector of the quad plane
        - d: Plane constant (distance from origin along normal)
        - w_u: Helper vector for computing alpha coordinate
        - w_v: Helper vector for computing beta coordinate
    """
    # Cross product gives normal direction
    n = tm.cross(quad.u, quad.v)
    normal = tm.normalize(n)

    # Plane equation: dot(normal, P) = d (plane constant)
    d = tm.dot(normal, quad.Q)

    # For solving the parametric coordinates (alpha, beta):
    # P = Q + alpha * u + beta * v
    # We need vectors w_u and w_v such that:
    # alpha = dot(w_u, P - Q)
    # beta = dot(w_v, P - Q)
    #
    # Using the formulas from "An Efficient Ray-Quadrilateral Intersection Test"
    # w = n / dot(n, n)
    # where n = u x v (unnormalized cross product)

    n_dot_n = tm.dot(n, n)

    # Guard against degenerate quad (u and v parallel)
    # In this case, set w vectors to zero
    w_u = vec3(0.0, 0.0, 0.0)
    w_v = vec3(0.0, 0.0, 0.0)

    if n_dot_n > 1e-10:
        # w_u = v x n / dot(n, n), w_v = n x u / dot(n, n)
        # These satisfy: dot(w_u, u) = 1, dot(w_u, v) = 0
        #                dot(w_v, u) = 0, dot(w_v, v) = 1
        w_u = tm.cross(quad.v, n) / n_dot_n
        w_v = tm.cross(n, quad.u) / n_dot_n

    return normal, d, w_u, w_v


@ti.func
def hit_quad(
    ray_origin: vec3,
    ray_direction: vec3,
    quad: Quad,
    t_min: ti.f32,
    t_max: ti.f32,
) -> HitRecord:
    """Test for ray-quad intersection.

    Uses parametric plane intersection followed by bounds checking:
    1. Compute where ray hits the plane containing the quad
    2. Express hit point in quad's local coordinates (alpha, beta)
    3. Check if 0 <= alpha <= 1 and 0 <= beta <= 1

    The ray-plane intersection is found by solving:
        ray_origin + t * ray_direction = Q + alpha * u + beta * v

    Taking dot product with the normal n = u x v:
        t = (D - dot(normal, ray_origin)) / dot(normal, ray_direction)

    Args:
        ray_origin: The starting point of the ray.
        ray_direction: The direction vector of the ray (need not be normalized).
        quad: The quad to test intersection against.
        t_min: Minimum t value to consider a valid hit (avoids self-intersection).
        t_max: Maximum t value to consider a valid hit.

    Returns:
        A HitRecord containing intersection information. Check hit field
        to determine if intersection occurred.
    """
    # Compute quad's plane frame
    normal, d, w_u, w_v = _compute_quad_frame(quad)

    # Check if ray is parallel to plane
    denom = tm.dot(normal, ray_direction)

    # Initialize result
    did_hit = 0
    hit_t = 0.0
    hit_point = vec3(0.0, 0.0, 0.0)
    hit_normal = vec3(0.0, 0.0, 0.0)
    is_front_face = 0

    # Ray not parallel to plane (denom != 0)
    if ti.abs(denom) > 1e-8:
        # Compute t parameter at plane intersection
        t = (d - tm.dot(normal, ray_origin)) / denom

        # Check if t is in valid range
        if t > t_min and t < t_max:
            # Compute hit point
            hit_point = ray_origin + t * ray_direction

            # Compute local coordinates (alpha, beta)
            # P = Q + alpha * u + beta * v
            # alpha = dot(w_u, P - Q)
            # beta = dot(w_v, P - Q)
            p_minus_q = hit_point - quad.Q
            alpha = tm.dot(w_u, p_minus_q)
            beta = tm.dot(w_v, p_minus_q)

            # Check if inside quad bounds [0, 1] x [0, 1]
            if alpha >= 0.0 and alpha <= 1.0 and beta >= 0.0 and beta <= 1.0:
                did_hit = 1
                hit_t = t

                # Determine front/back face based on ray direction vs normal
                if denom > 0.0:
                    # Ray hits back face (ray and normal point same direction)
                    is_front_face = 0
                    hit_normal = -normal
                else:
                    # Ray hits front face
                    is_front_face = 1
                    hit_normal = normal

    return HitRecord(
        hit=did_hit,
        t=hit_t,
        point=hit_point,
        normal=hit_normal,
        front_face=is_front_face,
    )


@ti.func
def make_quad(q: vec3, u: vec3, v: vec3) -> Quad:
    """Create a quad from corner point and edge vectors.

    This is a convenience function for creating quads within Taichi kernels.

    Args:
        q: The corner point of the quad.
        u: Edge vector from q to adjacent corner.
        v: Edge vector from q to other adjacent corner.

    Returns:
        A new Quad instance.
    """
    return Quad(Q=q, u=u, v=v)


@ti.func
def quad_normal(quad: Quad) -> vec3:
    """Compute the surface normal of a quad.

    The normal is computed as normalize(cross(u, v)), pointing in the
    direction determined by the right-hand rule.

    Args:
        quad: The quad to compute normal for.

    Returns:
        The unit normal vector.
    """
    return tm.normalize(tm.cross(quad.u, quad.v))


@ti.func
def quad_area(quad: Quad) -> ti.f32:
    """Compute the area of a quad.

    The area is the magnitude of the cross product of the edge vectors.

    Args:
        quad: The quad to compute area for.

    Returns:
        The area of the quad.
    """
    return tm.length(tm.cross(quad.u, quad.v))
