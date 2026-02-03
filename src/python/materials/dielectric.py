"""Dielectric (glass/water) material implementation.

This module implements the dielectric BSDF, which models transparent materials
like glass and water with refraction and Fresnel reflectance.

Key physics:
    - Snell's law for refraction: n1 * sin(theta1) = n2 * sin(theta2)
    - Schlick's approximation for Fresnel reflectance
    - Total internal reflection when sin(theta_t) > 1

The material randomly chooses between reflection and refraction based on
the Fresnel reflectance probability, which increases at grazing angles.

Example:
    >>> import taichi as ti
    >>> ti.init(arch=ti.gpu)
    >>> from src.python.materials.dielectric import scatter_dielectric
    >>> # Use within a Taichi kernel:
    >>> # direction, attenuation, did_scatter = scatter_dielectric(
    >>> #     ior, incident_dir, normal, front_face
    >>> # )
"""

import taichi as ti
import taichi.math as tm

from src.python.core.ray import (
    reflect,
    refract,
    schlick_fresnel,
)

# Type alias for 3D vectors
vec3 = tm.vec3


@ti.dataclass
class DielectricMaterial:
    """Dielectric (glass/water) material properties.

    Attributes:
        ior: Index of refraction. Common values:
            - Air: 1.0
            - Water: 1.33
            - Glass: 1.5
            - Diamond: 2.4
    """

    ior: ti.f32


@ti.func
def scatter_dielectric(
    ior: ti.f32,
    incident_direction: vec3,
    normal: vec3,
    front_face: ti.i32,
):
    """Compute scattered ray direction for dielectric material.

    Dielectrics (glass, water, etc.) both reflect and refract light.
    The probability of reflection vs refraction is determined by the
    Fresnel equations (using Schlick's approximation).

    Total internal reflection occurs when light travels from a denser
    medium to a less dense medium at a steep enough angle.

    Args:
        ior: Index of refraction of the material.
        incident_direction: The incoming ray direction (should be normalized).
        normal: The surface normal (should be normalized, pointing outward
            from the surface toward the incident ray).
        front_face: 1 if ray is hitting the outside of the surface,
            0 if ray is inside the material hitting from within.

    Returns:
        A tuple of (scattered_direction, attenuation, did_scatter) where:
        - scattered_direction: The reflected or refracted direction (normalized).
        - attenuation: The color attenuation (white for clear glass).
        - did_scatter: Always 1 for dielectrics (they always scatter).
    """
    # Dielectrics don't absorb light - attenuation is white
    attenuation = vec3(1.0, 1.0, 1.0)

    # Determine the refraction ratio based on which side we're hitting
    # If hitting from outside (front_face=1): eta = 1/ior (air to glass)
    # If hitting from inside (front_face=0): eta = ior (glass to air)
    refraction_ratio = 1.0 / ior
    if front_face == 0:
        refraction_ratio = ior

    # Compute cosine of incident angle
    cos_theta = tm.min(-tm.dot(incident_direction, normal), 1.0)
    sin_theta = tm.sqrt(1.0 - cos_theta * cos_theta)

    # Check for total internal reflection
    # This occurs when sin(theta_t) = (n1/n2) * sin(theta_i) > 1
    cannot_refract = refraction_ratio * sin_theta > 1.0

    # Compute Fresnel reflectance using Schlick's approximation
    reflectance = schlick_fresnel(cos_theta, refraction_ratio)

    # Decide between reflection and refraction
    # Reflect if: total internal reflection OR random < Fresnel reflectance
    scattered_direction = vec3(0.0, 0.0, 0.0)
    if cannot_refract or ti.random() < reflectance:
        # Reflect
        scattered_direction = reflect(incident_direction, normal)
    else:
        # Refract
        scattered_direction = refract(incident_direction, normal, refraction_ratio)

    # Normalize the result (should already be normalized, but ensure it)
    scattered_direction = tm.normalize(scattered_direction)

    # Dielectrics always scatter (no absorption)
    did_scatter = 1

    return scattered_direction, attenuation, did_scatter


@ti.func
def scatter_dielectric_full(
    ior: ti.f32,
    incident_direction: vec3,
    normal: vec3,
    front_face: ti.i32,
):
    """Scatter dielectric ray with PDF information.

    This variant returns a PDF value for consistency with the material
    interface, though for specular reflection/refraction the PDF is
    technically a delta function.

    Args:
        ior: Index of refraction of the material.
        incident_direction: The incoming ray direction (should be normalized).
        normal: The surface normal (should be normalized).
        front_face: 1 if ray is hitting the outside of the surface,
            0 if ray is inside the material hitting from within.

    Returns:
        A tuple of (scattered_direction, attenuation, pdf, did_scatter) where:
        - scattered_direction: The reflected or refracted direction (normalized).
        - attenuation: The color attenuation (white for clear glass).
        - pdf: Nominal PDF value (1.0 for specular materials).
        - did_scatter: Always 1 for dielectrics.
    """
    scattered_direction, attenuation, did_scatter = scatter_dielectric(
        ior, incident_direction, normal, front_face
    )
    # For specular reflection/refraction, PDF is a delta function.
    # We use 1.0 as a nominal value for sampling weight calculations.
    pdf = 1.0
    return scattered_direction, attenuation, pdf, did_scatter


@ti.func
def will_reflect(
    ior: ti.f32,
    incident_direction: vec3,
    normal: vec3,
    front_face: ti.i32,
) -> ti.i32:
    """Determine if total internal reflection will occur.

    Utility function to check if the ray will be totally internally
    reflected (no refraction possible).

    Args:
        ior: Index of refraction of the material.
        incident_direction: The incoming ray direction (should be normalized).
        normal: The surface normal (should be normalized).
        front_face: 1 if ray is hitting the outside of the surface,
            0 if ray is inside the material hitting from within.

    Returns:
        1 if total internal reflection will occur, 0 otherwise.
    """
    refraction_ratio = 1.0 / ior
    if front_face == 0:
        refraction_ratio = ior

    cos_theta = tm.min(-tm.dot(incident_direction, normal), 1.0)
    sin_theta = tm.sqrt(1.0 - cos_theta * cos_theta)

    return 1 if refraction_ratio * sin_theta > 1.0 else 0


@ti.func
def fresnel_reflectance(
    ior: ti.f32,
    incident_direction: vec3,
    normal: vec3,
    front_face: ti.i32,
) -> ti.f32:
    """Compute Fresnel reflectance using Schlick's approximation.

    Args:
        ior: Index of refraction of the material.
        incident_direction: The incoming ray direction (should be normalized).
        normal: The surface normal (should be normalized).
        front_face: 1 if ray is hitting the outside of the surface,
            0 if ray is inside the material hitting from within.

    Returns:
        The Fresnel reflectance coefficient in [0, 1].
    """
    refraction_ratio = 1.0 / ior
    if front_face == 0:
        refraction_ratio = ior

    cos_theta = tm.min(-tm.dot(incident_direction, normal), 1.0)
    return schlick_fresnel(cos_theta, refraction_ratio)


# =============================================================================
# Material Field Storage (for scene-level material management)
# =============================================================================

# Maximum number of dielectric materials in the scene
MAX_DIELECTRIC_MATERIALS = 256

# Storage for dielectric material properties
dielectric_iors = ti.field(dtype=ti.f32, shape=MAX_DIELECTRIC_MATERIALS)
num_dielectric_materials = ti.field(dtype=ti.i32, shape=())


def clear_dielectric_materials() -> None:
    """Clear all dielectric materials.

    Resets the material count to zero. Existing data in the field will be
    overwritten when new materials are added.
    """
    num_dielectric_materials[None] = 0


def add_dielectric_material(ior: float = 1.5) -> int:
    """Add a dielectric material to the material registry.

    Args:
        ior: Index of refraction. Default is 1.5 (typical glass).
            Must be >= 1.0 (physically meaningful IOR).
            Common values:
            - Air: 1.0
            - Water: 1.33
            - Glass: 1.5
            - Diamond: 2.4

    Returns:
        The index of the added material.

    Raises:
        RuntimeError: If the maximum number of materials is exceeded.
        ValueError: If IOR is less than 1.0.
    """
    if ior < 1.0:
        raise ValueError(
            f"Index of refraction = {ior} is less than 1.0. "
            "IOR must be >= 1.0 for physically meaningful materials."
        )

    idx = num_dielectric_materials[None]
    if idx >= MAX_DIELECTRIC_MATERIALS:
        raise RuntimeError(
            f"Maximum number of dielectric materials ({MAX_DIELECTRIC_MATERIALS}) exceeded"
        )

    dielectric_iors[idx] = ior
    num_dielectric_materials[None] = idx + 1
    return idx


def get_dielectric_material_count() -> int:
    """Get the number of dielectric materials in the registry."""
    return int(num_dielectric_materials[None])


@ti.func
def get_dielectric_ior(material_idx: ti.i32) -> ti.f32:
    """Get the IOR for a dielectric material by index.

    Args:
        material_idx: The index of the material in the registry.

    Returns:
        The index of refraction for the material.
    """
    return dielectric_iors[material_idx]


@ti.func
def scatter_dielectric_by_id(
    material_idx: ti.i32,
    incident_direction: vec3,
    normal: vec3,
    front_face: ti.i32,
):
    """Sample a scattered ray direction for a dielectric material by index.

    Convenience function that looks up the IOR from the material registry
    and calls scatter_dielectric.

    Args:
        material_idx: The index of the material in the registry.
        incident_direction: The incoming ray direction (should be normalized).
        normal: The surface normal at the hit point (should be normalized).
        front_face: 1 if ray is hitting the outside of the surface,
            0 if ray is inside the material hitting from within.

    Returns:
        A tuple of (scattered_direction, attenuation, did_scatter).
    """
    ior = get_dielectric_ior(material_idx)
    return scatter_dielectric(ior, incident_direction, normal, front_face)
