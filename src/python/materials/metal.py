"""Metal (specular reflective) material implementation.

This module implements the metal BSDF, which models specular reflection with
optional roughness (fuzziness). Perfect metals (roughness=0) produce mirror-like
reflections, while rougher metals scatter reflected rays within a cone.

The reflection formula is:
    R = I - 2(I . N)N

where I is the incident direction and N is the surface normal.

For rough metals, the reflected direction is perturbed by a random offset scaled
by the roughness parameter, modeling microfacet scattering.

Example:
    >>> import taichi as ti
    >>> ti.init(arch=ti.gpu)
    >>> from src.python.materials.metal import scatter_metal
    >>> # Use within a Taichi kernel:
    >>> # direction, attenuation, did_scatter = scatter_metal(
    >>> #     albedo, roughness, incident_dir, normal
    >>> # )
"""

import taichi as ti
import taichi.math as tm

from src.python.core.ray import (
    random_in_unit_sphere,
    reflect,
)

# Type alias for 3D vectors
vec3 = tm.vec3


@ti.dataclass
class MetalMaterial:
    """Metal (specular reflective) material properties.

    Attributes:
        albedo: The reflective color (RGB, each component in [0, 1]).
            Represents the color tint of reflected light.
        roughness: The surface roughness/fuzziness in [0, 1].
            0 = perfect mirror, 1 = maximum fuzziness.
    """

    albedo: vec3
    roughness: ti.f32


@ti.func
def scatter_metal(
    albedo: vec3,
    roughness: ti.f32,
    incident_direction: vec3,
    normal: vec3,
):
    """Compute scattered ray direction for metal material.

    Reflects the incident ray about the surface normal, then optionally
    perturbs the reflected direction based on roughness. The ray is absorbed
    if the scattered direction ends up below the surface.

    For metals, no energy is absorbed on reflection (attenuation = albedo).
    The metal simply tints the reflected light by its albedo color.

    Args:
        albedo: The reflective color (RGB, each component in [0, 1]).
        roughness: The surface roughness in [0, 1]. 0 = perfect mirror.
        incident_direction: The incoming ray direction (should be normalized).
        normal: The surface normal (should be normalized).

    Returns:
        A tuple of (scattered_direction, attenuation, did_scatter) where:
        - scattered_direction: The reflected direction (normalized).
        - attenuation: The color attenuation (equals albedo for metals).
        - did_scatter: 1 if the ray scattered above surface, 0 if absorbed.
    """
    # Compute perfect reflection
    reflected = reflect(incident_direction, normal)

    # Add roughness perturbation
    # Scale random offset by roughness (0 = perfect mirror, 1 = max fuzz)
    fuzz_offset = roughness * random_in_unit_sphere()
    scattered_direction = tm.normalize(reflected + fuzz_offset)

    # Check if scattered ray is above surface
    # If dot product with normal is positive, ray scatters; otherwise absorbed
    did_scatter = 1
    if tm.dot(scattered_direction, normal) <= 0.0:
        did_scatter = 0
        # Set to zero vector to indicate no valid scatter
        scattered_direction = vec3(0.0, 0.0, 0.0)

    # Metal attenuation is simply the albedo (no energy absorbed)
    attenuation = albedo

    return scattered_direction, attenuation, did_scatter


@ti.func
def scatter_metal_full(
    albedo: vec3,
    roughness: ti.f32,
    incident_direction: vec3,
    normal: vec3,
):
    """Scatter metal ray with PDF information.

    This variant returns a PDF value for consistency with the material
    interface, though for specular reflection the PDF is technically a
    delta function (infinite at the perfect reflection direction, zero elsewhere).

    For practical purposes with rough metals, we return a nominal PDF of 1.0.

    Args:
        albedo: The reflective color (RGB).
        roughness: The surface roughness in [0, 1].
        incident_direction: The incoming ray direction (should be normalized).
        normal: The surface normal (should be normalized).

    Returns:
        A tuple of (scattered_direction, attenuation, pdf, did_scatter) where:
        - scattered_direction: The reflected direction (normalized).
        - attenuation: The color attenuation (equals albedo).
        - pdf: Nominal PDF value (1.0 for specular materials).
        - did_scatter: 1 if the ray scattered above surface, 0 if absorbed.
    """
    scattered_direction, attenuation, did_scatter = scatter_metal(
        albedo, roughness, incident_direction, normal
    )
    # For specular reflection, PDF is a delta function.
    # We use 1.0 as a nominal value for sampling weight calculations.
    pdf = 1.0
    return scattered_direction, attenuation, pdf, did_scatter


# =============================================================================
# Material Field Storage (for scene-level material management)
# =============================================================================

# Maximum number of metal materials in the scene
MAX_METAL_MATERIALS = 256

# Storage for metal material properties
metal_albedos = ti.Vector.field(3, dtype=ti.f32, shape=MAX_METAL_MATERIALS)
metal_roughnesses = ti.field(dtype=ti.f32, shape=MAX_METAL_MATERIALS)
num_metal_materials = ti.field(dtype=ti.i32, shape=())


def clear_metal_materials() -> None:
    """Clear all metal materials.

    Resets the material count to zero. Existing data in the field will be
    overwritten when new materials are added.
    """
    num_metal_materials[None] = 0


def add_metal_material(
    albedo: tuple[float, float, float],
    roughness: float = 0.0,
) -> int:
    """Add a metal material to the material registry.

    Args:
        albedo: The reflective color as (R, G, B) tuple.
            Each component should be in [0, 1].
        roughness: The surface roughness in [0, 1]. Default is 0 (perfect mirror).
            Values are clamped to [0, 1].

    Returns:
        The index of the added material.

    Raises:
        RuntimeError: If the maximum number of materials is exceeded.
        ValueError: If any albedo component is outside [0, 1].
        ValueError: If roughness is outside [0, 1].
    """
    # Validate albedo components
    for i, component in enumerate(albedo):
        if component < 0.0 or component > 1.0:
            raise ValueError(
                f"Albedo component {i} = {component} is outside [0, 1]. "
                "This would violate energy conservation."
            )

    # Validate roughness
    if roughness < 0.0 or roughness > 1.0:
        raise ValueError(
            f"Roughness = {roughness} is outside [0, 1]. "
            "Roughness must be between 0 (perfect mirror) and 1 (maximum fuzz)."
        )

    idx = num_metal_materials[None]
    if idx >= MAX_METAL_MATERIALS:
        raise RuntimeError(
            f"Maximum number of metal materials ({MAX_METAL_MATERIALS}) exceeded"
        )

    metal_albedos[idx] = vec3(albedo[0], albedo[1], albedo[2])
    metal_roughnesses[idx] = roughness
    num_metal_materials[None] = idx + 1
    return idx


def get_metal_material_count() -> int:
    """Get the number of metal materials in the registry."""
    return int(num_metal_materials[None])


@ti.func
def get_metal_albedo(material_idx: ti.i32) -> vec3:
    """Get the albedo for a metal material by index.

    Args:
        material_idx: The index of the material in the registry.

    Returns:
        The albedo color (RGB) for the material.
    """
    return metal_albedos[material_idx]


@ti.func
def get_metal_roughness(material_idx: ti.i32) -> ti.f32:
    """Get the roughness for a metal material by index.

    Args:
        material_idx: The index of the material in the registry.

    Returns:
        The roughness value for the material.
    """
    return metal_roughnesses[material_idx]


@ti.func
def scatter_metal_by_id(
    material_idx: ti.i32,
    incident_direction: vec3,
    normal: vec3,
):
    """Sample a scattered ray direction for a metal material by index.

    Convenience function that looks up the albedo and roughness from the
    material registry and calls scatter_metal.

    Args:
        material_idx: The index of the material in the registry.
        incident_direction: The incoming ray direction (should be normalized).
        normal: The surface normal at the hit point (should be normalized).

    Returns:
        A tuple of (scattered_direction, attenuation, did_scatter).
    """
    albedo = get_metal_albedo(material_idx)
    roughness = get_metal_roughness(material_idx)
    return scatter_metal(albedo, roughness, incident_direction, normal)
