"""Materials module for BRDF/BSDF models.

This module implements physically-based material models for light scattering:

Components:
    lambertian: Ideal diffuse (Lambertian) reflection
    metal: Specular reflection with optional roughness
    dielectric: Glass-like materials with refraction (Fresnel equations)
    phosphorescent: Glow-in-the-dark materials (diffuse + emission)
    material: Base material interface and material registry

Each material provides:
    - sample(): Importance sample a scattering direction
    - eval(): Evaluate BSDF for given directions
    - pdf(): Probability density for a sampled direction

Materials follow energy conservation principles and support:
    - Cosine-weighted hemisphere sampling for diffuse
    - GGX microfacet distribution for rough metals (future)
    - Schlick Fresnel approximation for dielectrics

All BSDF computations are implemented as Taichi functions for GPU execution.
"""

# Lambertian (ideal diffuse) material
# Dielectric (glass/water) material
from .dielectric import (
    DielectricMaterial,
    add_dielectric_material,
    clear_dielectric_materials,
    fresnel_reflectance,
    get_dielectric_ior,
    get_dielectric_material_count,
    scatter_dielectric,
    scatter_dielectric_by_id,
    scatter_dielectric_full,
    will_reflect,
)
from .lambertian import (
    LambertianMaterial,
    add_lambertian_material,
    clear_lambertian_materials,
    eval_lambertian,
    get_lambertian_albedo,
    get_lambertian_material_count,
    pdf_lambertian,
    scatter_lambertian,
    scatter_lambertian_by_id,
    scatter_lambertian_full,
)

# Metal (specular reflective) material
from .metal import (
    MetalMaterial,
    add_metal_material,
    clear_metal_materials,
    get_metal_albedo,
    get_metal_material_count,
    get_metal_roughness,
    scatter_metal,
    scatter_metal_by_id,
    scatter_metal_full,
)

# Phosphorescent (glow-in-the-dark) material
from .phosphorescent import (
    PhosphorescentMaterial,
    add_phosphorescent_material,
    clear_phosphorescent_materials,
    eval_phosphorescent,
    get_phosphorescent_albedo,
    get_phosphorescent_emission,
    get_phosphorescent_emission_by_id,
    get_phosphorescent_glow_color,
    get_phosphorescent_glow_intensity,
    get_phosphorescent_material_count,
    pdf_phosphorescent,
    scatter_phosphorescent,
    scatter_phosphorescent_by_id,
    scatter_phosphorescent_full,
)

__all__ = [
    # Lambertian
    "LambertianMaterial",
    "eval_lambertian",
    "pdf_lambertian",
    "scatter_lambertian",
    "scatter_lambertian_full",
    "scatter_lambertian_by_id",
    "add_lambertian_material",
    "clear_lambertian_materials",
    "get_lambertian_material_count",
    "get_lambertian_albedo",
    # Metal
    "MetalMaterial",
    "scatter_metal",
    "scatter_metal_full",
    "scatter_metal_by_id",
    "add_metal_material",
    "clear_metal_materials",
    "get_metal_material_count",
    "get_metal_albedo",
    "get_metal_roughness",
    # Dielectric
    "DielectricMaterial",
    "scatter_dielectric",
    "scatter_dielectric_full",
    "scatter_dielectric_by_id",
    "add_dielectric_material",
    "clear_dielectric_materials",
    "get_dielectric_material_count",
    "get_dielectric_ior",
    "fresnel_reflectance",
    "will_reflect",
    # Phosphorescent
    "PhosphorescentMaterial",
    "eval_phosphorescent",
    "pdf_phosphorescent",
    "scatter_phosphorescent",
    "scatter_phosphorescent_full",
    "scatter_phosphorescent_by_id",
    "add_phosphorescent_material",
    "clear_phosphorescent_materials",
    "get_phosphorescent_material_count",
    "get_phosphorescent_albedo",
    "get_phosphorescent_glow_color",
    "get_phosphorescent_glow_intensity",
    "get_phosphorescent_emission",
    "get_phosphorescent_emission_by_id",
]
