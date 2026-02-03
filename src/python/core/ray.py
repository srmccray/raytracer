"""Ray data structure and vector utilities for GPU-accelerated ray tracing.

This module provides the fundamental Ray dataclass and vector utility functions
for Monte Carlo ray tracing. All operations are designed to work within Taichi
kernels for GPU acceleration.

Example:
    >>> import taichi as ti
    >>> ti.init(arch=ti.gpu)
    >>> origin = ti.math.vec3(0.0, 0.0, 0.0)
    >>> direction = ti.math.vec3(0.0, 0.0, -1.0)
    >>> ray = Ray(origin=origin, direction=direction)
    >>> point = ray_at(ray, 5.0)  # Point 5 units along the ray
"""

import taichi as ti
import taichi.math as tm

# Type alias for 3D vectors using Taichi's math module
vec3 = tm.vec3


@ti.dataclass
class Ray:
    """A ray with an origin point and direction vector.

    Attributes:
        origin: The starting point of the ray (vec3).
        direction: The direction vector of the ray (vec3). Should be normalized
            for most operations, but this is not enforced to allow flexibility.
    """

    origin: vec3
    direction: vec3


@ti.func
def ray_at(ray: Ray, t: ti.f32) -> vec3:
    """Compute the point along the ray at parameter t.

    Args:
        ray: The ray to evaluate.
        t: The parameter value. Positive values are in front of the origin.

    Returns:
        The point ray.origin + t * ray.direction.
    """
    return ray.origin + t * ray.direction


@ti.func
def make_ray(origin: vec3, direction: vec3) -> Ray:
    """Create a ray from origin and direction.

    This is a convenience function for creating rays within Taichi kernels.

    Args:
        origin: The starting point of the ray.
        direction: The direction vector (should typically be normalized).

    Returns:
        A new Ray instance.
    """
    return Ray(origin=origin, direction=direction)


# =============================================================================
# Vector Utility Functions
# =============================================================================


@ti.func
def length(v: vec3) -> ti.f32:
    """Compute the length (magnitude) of a vector.

    Args:
        v: The input vector.

    Returns:
        The Euclidean length of the vector.
    """
    return tm.length(v)


@ti.func
def length_squared(v: vec3) -> ti.f32:
    """Compute the squared length of a vector.

    This is more efficient than length() when only comparing magnitudes,
    as it avoids the square root computation.

    Args:
        v: The input vector.

    Returns:
        The squared Euclidean length of the vector.
    """
    return tm.dot(v, v)


@ti.func
def normalize(v: vec3) -> vec3:
    """Normalize a vector to unit length.

    Args:
        v: The input vector.

    Returns:
        A unit vector in the same direction as v.
        If v is zero-length, returns a zero vector.
    """
    return tm.normalize(v)


@ti.func
def dot(a: vec3, b: vec3) -> ti.f32:
    """Compute the dot product of two vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        The dot product a . b.
    """
    return tm.dot(a, b)


@ti.func
def cross(a: vec3, b: vec3) -> vec3:
    """Compute the cross product of two vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        The cross product a x b.
    """
    return tm.cross(a, b)


@ti.func
def reflect(incident: vec3, normal: vec3) -> vec3:
    """Reflect an incident vector about a normal.

    Computes the reflection of the incident direction about the surface normal.
    The normal should be unit length for correct results.

    Args:
        incident: The incoming direction vector (pointing toward the surface).
        normal: The surface normal (should be normalized).

    Returns:
        The reflected direction vector.
    """
    return incident - 2.0 * tm.dot(incident, normal) * normal


@ti.func
def refract(incident: vec3, normal: vec3, eta: ti.f32) -> vec3:
    """Refract an incident vector through a surface.

    Computes the refracted direction using Snell's law. If total internal
    reflection occurs, returns a zero vector.

    Args:
        incident: The incoming direction vector (should be normalized).
        normal: The surface normal (should be normalized, pointing outward).
        eta: The ratio of refractive indices (n_incident / n_transmitted).

    Returns:
        The refracted direction vector, or zero vector if total internal
        reflection occurs.
    """
    cos_i = -tm.dot(incident, normal)
    sin2_t = eta * eta * (1.0 - cos_i * cos_i)
    result = vec3(0.0, 0.0, 0.0)
    if sin2_t <= 1.0:
        cos_t = ti.sqrt(1.0 - sin2_t)
        result = eta * incident + (eta * cos_i - cos_t) * normal
    return result


@ti.func
def schlick_fresnel(cosine: ti.f32, ref_idx: ti.f32) -> ti.f32:
    """Compute Fresnel reflectance using Schlick's approximation.

    This approximation is commonly used in ray tracing for efficiency.

    Args:
        cosine: Cosine of the angle between incident direction and normal.
        ref_idx: Ratio of refractive indices.

    Returns:
        The approximate Fresnel reflectance coefficient.
    """
    r0 = ((1.0 - ref_idx) / (1.0 + ref_idx)) ** 2
    return r0 + (1.0 - r0) * ((1.0 - cosine) ** 5)


@ti.func
def near_zero(v: vec3) -> ti.i32:
    """Check if a vector is near zero in all components.

    Useful for detecting degenerate cases in scattering.

    Args:
        v: The vector to check.

    Returns:
        1 if all components are near zero, 0 otherwise.
    """
    s = 1e-8
    return ti.abs(v.x) < s and ti.abs(v.y) < s and ti.abs(v.z) < s


# =============================================================================
# Random Sampling Utilities for Monte Carlo
# =============================================================================


@ti.func
def random_in_unit_sphere() -> vec3:
    """Generate a random point inside the unit sphere.

    Uses rejection sampling to generate uniformly distributed points
    within the unit sphere.

    Returns:
        A random point with length < 1.
    """
    p = vec3(0.0, 0.0, 0.0)
    found = False
    # Rejection sampling loop
    for _ in range(100):  # Max iterations to avoid infinite loops
        if not found:
            p = vec3(
                ti.random(ti.f32) * 2.0 - 1.0,
                ti.random(ti.f32) * 2.0 - 1.0,
                ti.random(ti.f32) * 2.0 - 1.0,
            )
            if length_squared(p) < 1.0:
                found = True
    return p


@ti.func
def random_unit_vector() -> vec3:
    """Generate a random unit vector uniformly distributed on the sphere.

    This is the normalized version of random_in_unit_sphere().

    Returns:
        A random unit vector.
    """
    return normalize(random_in_unit_sphere())


@ti.func
def random_on_hemisphere(normal: vec3) -> vec3:
    """Generate a random unit vector on the hemisphere defined by a normal.

    The returned vector will be in the same hemisphere as the normal
    (i.e., dot product with normal will be positive).

    Args:
        normal: The surface normal defining the hemisphere orientation.

    Returns:
        A random unit vector in the hemisphere around the normal.
    """
    on_sphere = random_unit_vector()
    result = on_sphere
    if tm.dot(on_sphere, normal) < 0.0:
        result = -on_sphere
    return result


@ti.func
def random_in_unit_disk() -> vec3:
    """Generate a random point inside the unit disk in the xy-plane.

    Useful for depth-of-field effects in camera simulation.

    Returns:
        A random point (x, y, 0) with x^2 + y^2 < 1.
    """
    p = vec3(0.0, 0.0, 0.0)
    found = False
    # Rejection sampling loop
    for _ in range(100):  # Max iterations to avoid infinite loops
        if not found:
            p = vec3(
                ti.random(ti.f32) * 2.0 - 1.0,
                ti.random(ti.f32) * 2.0 - 1.0,
                0.0,
            )
            if p.x * p.x + p.y * p.y < 1.0:
                found = True
    return p


@ti.func
def random_cosine_direction() -> vec3:
    """Generate a random direction with cosine-weighted distribution.

    This is useful for importance sampling diffuse surfaces.
    The distribution has PDF = cos(theta) / pi.

    Returns:
        A random direction in the local coordinate frame (z-up).
    """
    r1 = ti.random(ti.f32)
    r2 = ti.random(ti.f32)
    phi = 2.0 * tm.pi * r1
    sqrt_r2 = ti.sqrt(r2)
    x = ti.cos(phi) * sqrt_r2
    y = ti.sin(phi) * sqrt_r2
    z = ti.sqrt(1.0 - r2)
    return vec3(x, y, z)


@ti.func
def build_onb_from_normal(normal: vec3):
    """Build an orthonormal basis from a normal vector.

    Creates a local coordinate frame where the normal is the z-axis.
    Useful for transforming local-space samples to world space.

    Args:
        normal: The surface normal (should be normalized).

    Returns:
        A tuple (tangent, bitangent, normal) forming an orthonormal basis.
    """
    # Choose a vector not parallel to normal
    a = vec3(1.0, 0.0, 0.0)
    if ti.abs(normal.x) > 0.9:
        a = vec3(0.0, 1.0, 0.0)
    tangent = normalize(cross(a, normal))
    bitangent = cross(normal, tangent)
    return tangent, bitangent, normal


@ti.func
def local_to_world(local_dir: vec3, tangent: vec3, bitangent: vec3, normal: vec3) -> vec3:
    """Transform a direction from local to world coordinates.

    Args:
        local_dir: Direction in local coordinates (z-up).
        tangent: The x-axis of the local frame in world coordinates.
        bitangent: The y-axis of the local frame in world coordinates.
        normal: The z-axis of the local frame in world coordinates.

    Returns:
        The direction in world coordinates.
    """
    return local_dir.x * tangent + local_dir.y * bitangent + local_dir.z * normal


@ti.func
def sample_cosine_hemisphere(normal: vec3):
    """Cosine-weighted hemisphere sampling for diffuse surfaces.

    Generates a random direction weighted by cos(theta), which is the
    optimal importance sampling distribution for Lambertian BRDFs.

    Args:
        normal: The surface normal defining the hemisphere orientation.

    Returns:
        A tuple of (direction, pdf) where:
        - direction: The sampled direction in world space.
        - pdf: The probability density function value = cos(theta) / pi.
    """
    local_dir = random_cosine_direction()
    tangent, bitangent, n = build_onb_from_normal(normal)
    world_dir = local_to_world(local_dir, tangent, bitangent, n)
    pdf = tm.dot(world_dir, normal) / tm.pi
    return world_dir, pdf
