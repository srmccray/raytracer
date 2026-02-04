"""Phosphorescent (glow-in-the-dark) material implementation.

This module implements a phosphorescent BSDF, which models materials that both
scatter light diffusely (like Lambertian) and emit light (glow). Phosphorescent
materials absorb light energy and re-emit it, creating a glow effect.

Physics Background:
    Real phosphorescence involves time-dependent energy storage and release.
    For rendering, we use a steady-state model where glow_intensity represents
    constant emission.

The material has two components:
    1. Scattering: Lambertian diffuse BRDF (f_r = albedo / pi)
    2. Emission: glow_color * glow_intensity (added to outgoing radiance)

Example:
    >>> import taichi as ti
    >>> ti.init(arch=ti.gpu)
    >>> from src.python.materials.phosphorescent import (
    ...     add_phosphorescent_material,
    ...     scatter_phosphorescent_by_id,
    ...     get_phosphorescent_emission_by_id,
    ... )
    >>> # Register a glowing green material
    >>> mat_idx = add_phosphorescent_material(
    ...     albedo=(0.2, 0.5, 0.2),
    ...     glow_color=(0.0, 1.0, 0.3),
    ...     glow_intensity=2.0,
    ... )
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
class PhosphorescentMaterial:
    """Phosphorescent (glow-in-the-dark) material properties.

    Attributes:
        albedo: The diffuse reflectance color (RGB, each component in [0, 1]).
            Represents the fraction of light reflected for each color channel.
        glow_color: The emission color (RGB). Can exceed 1.0 for HDR.
            Typically shifted toward longer wavelengths (green/yellow).
        glow_intensity: The emission strength multiplier.
            Typical range is 0-10, but can be higher for bright sources.
    """

    albedo: vec3
    glow_color: vec3
    glow_intensity: ti.f32


@ti.func
def eval_phosphorescent(albedo: vec3) -> vec3:
    """Evaluate the phosphorescent scattering BRDF (Lambertian).

    The scattering component is identical to Lambertian:
        f_r = albedo / pi

    Args:
        albedo: The diffuse reflectance color (RGB).

    Returns:
        The BRDF value (albedo / pi).
    """
    return albedo / tm.pi


@ti.func
def pdf_phosphorescent(normal: vec3, scattered_direction: vec3) -> ti.f32:
    """Compute the PDF for phosphorescent cosine-weighted sampling.

    Same as Lambertian:
        pdf = cos(theta) / pi

    Args:
        normal: The surface normal (should be normalized).
        scattered_direction: The sampled scatter direction (should be normalized).

    Returns:
        The probability density function value. Returns 0 if the direction
        is below the surface.
    """
    cos_theta = tm.dot(normal, scattered_direction)
    pdf = 0.0
    if cos_theta > 0.0:
        pdf = cos_theta / tm.pi
    return pdf


@ti.func
def scatter_phosphorescent(
    albedo: vec3,
    normal: vec3,
):
    """Sample a scattered ray direction for phosphorescent material.

    Uses cosine-weighted hemisphere sampling (same as Lambertian).
    The emission is handled separately via get_phosphorescent_emission.

    Args:
        albedo: The diffuse reflectance color (RGB, each component in [0, 1]).
        normal: The surface normal at the hit point (should be normalized).

    Returns:
        A tuple of (scattered_direction, attenuation, pdf) where:
        - scattered_direction: The sampled direction (normalized).
        - attenuation: The color attenuation (equals albedo due to importance sampling).
        - pdf: The probability density function value.
    """
    # Sample direction using cosine-weighted hemisphere sampling
    scattered_direction, pdf = sample_cosine_hemisphere(normal)

    # Handle degenerate case where sampled direction is near zero
    if near_zero(scattered_direction):
        scattered_direction = normal

    # For Lambertian with cosine-weighted sampling:
    # weight = BRDF * cos_theta / pdf = albedo
    attenuation = albedo

    return scattered_direction, attenuation, pdf


@ti.func
def get_phosphorescent_emission(glow_color: vec3, glow_intensity: ti.f32) -> vec3:
    """Get the emission radiance for a phosphorescent material.

    Args:
        glow_color: The emission color (RGB).
        glow_intensity: The emission strength multiplier.

    Returns:
        The emission radiance (glow_color * glow_intensity).
    """
    return glow_color * glow_intensity


@ti.func
def scatter_phosphorescent_full(
    albedo: vec3,
    glow_color: vec3,
    glow_intensity: ti.f32,
    normal: vec3,
):
    """Sample a scattered ray with full BRDF and emission information.

    Args:
        albedo: The diffuse reflectance color (RGB).
        glow_color: The emission color (RGB).
        glow_intensity: The emission strength multiplier.
        normal: The surface normal at the hit point (should be normalized).

    Returns:
        A tuple of (scattered_direction, attenuation, pdf, brdf, emission) where:
        - scattered_direction: The sampled direction (normalized).
        - attenuation: The color attenuation (equals albedo).
        - pdf: The probability density function value.
        - brdf: The BRDF value (albedo / pi).
        - emission: The emission radiance (glow_color * glow_intensity).
    """
    scattered_direction, attenuation, pdf = scatter_phosphorescent(albedo, normal)
    brdf = eval_phosphorescent(albedo)
    emission = get_phosphorescent_emission(glow_color, glow_intensity)
    return scattered_direction, attenuation, pdf, brdf, emission


# =============================================================================
# Material Field Storage (for scene-level material management)
# =============================================================================

# Maximum number of phosphorescent materials in the scene
MAX_PHOSPHORESCENT_MATERIALS = 64

# Storage for phosphorescent material properties
phosphorescent_albedos = ti.Vector.field(3, dtype=ti.f32, shape=MAX_PHOSPHORESCENT_MATERIALS)
phosphorescent_glow_colors = ti.Vector.field(3, dtype=ti.f32, shape=MAX_PHOSPHORESCENT_MATERIALS)
phosphorescent_glow_intensities = ti.field(dtype=ti.f32, shape=MAX_PHOSPHORESCENT_MATERIALS)
num_phosphorescent_materials = ti.field(dtype=ti.i32, shape=())


def clear_phosphorescent_materials() -> None:
    """Clear all phosphorescent materials.

    Resets the material count to zero. Existing data in the field will be
    overwritten when new materials are added.
    """
    num_phosphorescent_materials[None] = 0


def add_phosphorescent_material(
    albedo: tuple[float, float, float],
    glow_color: tuple[float, float, float],
    glow_intensity: float,
) -> int:
    """Add a phosphorescent material to the material registry.

    Args:
        albedo: The diffuse reflectance color as (R, G, B) tuple.
            Each component should be in [0, 1] for energy conservation.
        glow_color: The emission color as (R, G, B) tuple.
            Values can exceed 1.0 for HDR effects.
        glow_intensity: The emission strength. Typical range is 0-10.
            Must be non-negative.

    Returns:
        The index of the added material.

    Raises:
        RuntimeError: If the maximum number of materials is exceeded.
        ValueError: If any albedo component is outside [0, 1].
        ValueError: If glow_intensity is negative.
    """
    # Validate albedo for energy conservation
    for i, component in enumerate(albedo):
        if component < 0.0 or component > 1.0:
            raise ValueError(
                f"Albedo component {i} = {component} is outside [0, 1]. "
                "This would violate energy conservation."
            )

    # Validate glow_color (can be any non-negative value)
    for i, component in enumerate(glow_color):
        if component < 0.0:
            raise ValueError(f"Glow color component {i} = {component} is negative.")

    # Validate glow_intensity
    if glow_intensity < 0.0:
        raise ValueError(f"Glow intensity = {glow_intensity} is negative.")

    idx = num_phosphorescent_materials[None]
    if idx >= MAX_PHOSPHORESCENT_MATERIALS:
        raise RuntimeError(
            f"Maximum number of phosphorescent materials "
            f"({MAX_PHOSPHORESCENT_MATERIALS}) exceeded"
        )

    phosphorescent_albedos[idx] = vec3(albedo[0], albedo[1], albedo[2])
    phosphorescent_glow_colors[idx] = vec3(glow_color[0], glow_color[1], glow_color[2])
    phosphorescent_glow_intensities[idx] = glow_intensity
    num_phosphorescent_materials[None] = idx + 1
    return idx


def get_phosphorescent_material_count() -> int:
    """Get the number of phosphorescent materials in the registry."""
    return int(num_phosphorescent_materials[None])


@ti.func
def get_phosphorescent_albedo(material_idx: ti.i32) -> vec3:
    """Get the albedo for a phosphorescent material by index.

    Args:
        material_idx: The index of the material in the registry.

    Returns:
        The albedo color (RGB) for the material.
    """
    return phosphorescent_albedos[material_idx]


@ti.func
def get_phosphorescent_glow_color(material_idx: ti.i32) -> vec3:
    """Get the glow color for a phosphorescent material by index.

    Args:
        material_idx: The index of the material in the registry.

    Returns:
        The glow color (RGB) for the material.
    """
    return phosphorescent_glow_colors[material_idx]


@ti.func
def get_phosphorescent_glow_intensity(material_idx: ti.i32) -> ti.f32:
    """Get the glow intensity for a phosphorescent material by index.

    Args:
        material_idx: The index of the material in the registry.

    Returns:
        The glow intensity for the material.
    """
    return phosphorescent_glow_intensities[material_idx]


@ti.func
def scatter_phosphorescent_by_id(
    material_idx: ti.i32,
    normal: vec3,
):
    """Sample a scattered ray direction for a phosphorescent material by index.

    Convenience function that looks up the albedo from the material registry
    and calls scatter_phosphorescent.

    Args:
        material_idx: The index of the material in the registry.
        normal: The surface normal at the hit point (should be normalized).

    Returns:
        A tuple of (scattered_direction, attenuation, pdf).
    """
    albedo = get_phosphorescent_albedo(material_idx)
    return scatter_phosphorescent(albedo, normal)


@ti.func
def get_phosphorescent_emission_by_id(material_idx: ti.i32) -> vec3:
    """Get the emission radiance for a phosphorescent material by index.

    Convenience function for the integrator to query emission.

    Args:
        material_idx: The index of the material in the registry.

    Returns:
        The emission radiance (glow_color * glow_intensity).
    """
    glow_color = get_phosphorescent_glow_color(material_idx)
    glow_intensity = get_phosphorescent_glow_intensity(material_idx)
    return get_phosphorescent_emission(glow_color, glow_intensity)
