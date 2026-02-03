"""Lambertian (ideal diffuse) material implementation.

This module implements the Lambertian BRDF, which models ideal diffuse reflection
where incident light is scattered uniformly in all directions weighted by the
cosine of the angle from the surface normal.

The Lambertian BRDF is:
    f_r(wi, wo) = albedo / pi

The probability density function for cosine-weighted hemisphere sampling is:
    pdf(wi) = cos(theta) / pi

where theta is the angle between the sampled direction and the surface normal.

Example:
    >>> import taichi as ti
    >>> ti.init(arch=ti.gpu)
    >>> from src.python.materials.lambertian import (
    ...     scatter_lambertian, eval_lambertian, pdf_lambertian
    ... )
    >>> # Use within a Taichi kernel:
    >>> # direction, attenuation, pdf = scatter_lambertian(albedo, normal)
"""

import taichi as ti
import taichi.math as tm

from src.python.core.ray import (
    near_zero,
    sample_cosine_hemisphere,
)

# Type alias for 3D vectors
vec3 = tm.vec3


@ti.dataclass
class LambertianMaterial:
    """Lambertian (ideal diffuse) material properties.

    Attributes:
        albedo: The diffuse reflectance color (RGB, each component in [0, 1]).
            Represents the fraction of light reflected for each color channel.
            Values should be in [0, 1] to ensure energy conservation.
    """

    albedo: vec3


@ti.func
def eval_lambertian(albedo: vec3) -> vec3:
    """Evaluate the Lambertian BRDF.

    The Lambertian BRDF is constant for all directions:
        f_r = albedo / pi

    This function returns the BRDF value (not including the cosine term,
    which is applied separately in the rendering equation).

    Args:
        albedo: The diffuse reflectance color (RGB).

    Returns:
        The BRDF value (albedo / pi).
    """
    return albedo / tm.pi


@ti.func
def pdf_lambertian(normal: vec3, scattered_direction: vec3) -> ti.f32:
    """Compute the PDF for Lambertian cosine-weighted sampling.

    For cosine-weighted hemisphere sampling, the PDF is:
        pdf = cos(theta) / pi

    where theta is the angle between the scattered direction and the normal.

    Args:
        normal: The surface normal (should be normalized).
        scattered_direction: The sampled scatter direction (should be normalized).

    Returns:
        The probability density function value. Returns 0 if the direction
        is below the surface (negative cosine).
    """
    cos_theta = tm.dot(normal, scattered_direction)
    pdf = 0.0
    if cos_theta > 0.0:
        pdf = cos_theta / tm.pi
    return pdf


@ti.func
def scatter_lambertian(
    albedo: vec3,
    normal: vec3,
):
    """Sample a scattered ray direction for Lambertian material.

    Uses cosine-weighted hemisphere sampling for importance sampling,
    which is optimal for the Lambertian BRDF.

    The attenuation is computed as:
        attenuation = (BRDF * cos_theta) / pdf = albedo

    This simplification occurs because:
        BRDF = albedo / pi
        pdf = cos_theta / pi
        attenuation = (albedo / pi) * cos_theta / (cos_theta / pi) = albedo

    Args:
        albedo: The diffuse reflectance color (RGB, each component in [0, 1]).
        normal: The surface normal at the hit point (should be normalized).

    Returns:
        A tuple of (scattered_direction, attenuation, pdf) where:
        - scattered_direction: The sampled direction (normalized).
        - attenuation: The color attenuation for this scatter (equals albedo
          due to importance sampling cancellation).
        - pdf: The probability density function value for the sampled direction.
    """
    # Sample direction using cosine-weighted hemisphere sampling
    scattered_direction, pdf = sample_cosine_hemisphere(normal)

    # Handle degenerate case where sampled direction is near zero
    # (can happen due to floating point issues)
    if near_zero(scattered_direction):
        scattered_direction = normal

    # For Lambertian with cosine-weighted sampling:
    # weight = BRDF * cos_theta / pdf
    #        = (albedo / pi) * cos_theta / (cos_theta / pi)
    #        = albedo
    # This is the beauty of importance sampling - the cosine terms cancel
    attenuation = albedo

    return scattered_direction, attenuation, pdf


@ti.func
def scatter_lambertian_full(
    albedo: vec3,
    normal: vec3,
):
    """Sample a scattered ray with full BRDF information.

    This variant returns the BRDF value separately from the attenuation,
    useful for Multiple Importance Sampling (MIS) where we need to
    evaluate the BRDF independently.

    Args:
        albedo: The diffuse reflectance color (RGB).
        normal: The surface normal at the hit point (should be normalized).

    Returns:
        A tuple of (scattered_direction, attenuation, pdf, brdf) where:
        - scattered_direction: The sampled direction (normalized).
        - attenuation: The color attenuation (equals albedo).
        - pdf: The probability density function value.
        - brdf: The BRDF value (albedo / pi).
    """
    scattered_direction, attenuation, pdf = scatter_lambertian(albedo, normal)
    brdf = eval_lambertian(albedo)
    return scattered_direction, attenuation, pdf, brdf


# =============================================================================
# Material Field Storage (for scene-level material management)
# =============================================================================

# Maximum number of Lambertian materials in the scene
MAX_LAMBERTIAN_MATERIALS = 256

# Storage for Lambertian material properties
lambertian_albedos = ti.Vector.field(3, dtype=ti.f32, shape=MAX_LAMBERTIAN_MATERIALS)
num_lambertian_materials = ti.field(dtype=ti.i32, shape=())


def clear_lambertian_materials() -> None:
    """Clear all Lambertian materials.

    Resets the material count to zero. Existing data in the field will be
    overwritten when new materials are added.
    """
    num_lambertian_materials[None] = 0


def add_lambertian_material(albedo: tuple[float, float, float]) -> int:
    """Add a Lambertian material to the material registry.

    Args:
        albedo: The diffuse reflectance color as (R, G, B) tuple.
            Each component should be in [0, 1] for energy conservation.

    Returns:
        The index of the added material.

    Raises:
        RuntimeError: If the maximum number of materials is exceeded.
        ValueError: If any albedo component is outside [0, 1].
    """
    # Validate albedo for energy conservation
    for i, component in enumerate(albedo):
        if component < 0.0 or component > 1.0:
            raise ValueError(
                f"Albedo component {i} = {component} is outside [0, 1]. "
                "This would violate energy conservation."
            )

    idx = num_lambertian_materials[None]
    if idx >= MAX_LAMBERTIAN_MATERIALS:
        raise RuntimeError(
            f"Maximum number of Lambertian materials ({MAX_LAMBERTIAN_MATERIALS}) exceeded"
        )

    lambertian_albedos[idx] = vec3(albedo[0], albedo[1], albedo[2])
    num_lambertian_materials[None] = idx + 1
    return idx


def get_lambertian_material_count() -> int:
    """Get the number of Lambertian materials in the registry."""
    return int(num_lambertian_materials[None])


@ti.func
def get_lambertian_albedo(material_idx: ti.i32) -> vec3:
    """Get the albedo for a Lambertian material by index.

    Args:
        material_idx: The index of the material in the registry.

    Returns:
        The albedo color (RGB) for the material.
    """
    return lambertian_albedos[material_idx]


@ti.func
def scatter_lambertian_by_id(
    material_idx: ti.i32,
    normal: vec3,
):
    """Sample a scattered ray direction for a Lambertian material by index.

    Convenience function that looks up the albedo from the material registry
    and calls scatter_lambertian.

    Args:
        material_idx: The index of the material in the registry.
        normal: The surface normal at the hit point (should be normalized).

    Returns:
        A tuple of (scattered_direction, attenuation, pdf).
    """
    albedo = get_lambertian_albedo(material_idx)
    return scatter_lambertian(albedo, normal)
