"""Scene module for scene management and hit records.

This module handles scene representation and ray-scene queries:

Components:
    scene: Scene container holding objects, materials, and lights
    hit_record: Structure for storing intersection results
    world: World-space utilities and environment sampling
    manager: Unified scene manager coordinating primitives and materials

The scene module manages:
    - Object storage in GPU-friendly Taichi fields
    - Material ID assignment and lookup
    - Light source enumeration for direct lighting
    - Environment map sampling (future)

Scene data is organized for efficient GPU access:
    - Structure-of-Arrays layout for geometric data
    - Contiguous material ID arrays
    - Pre-computed light sampling CDFs
"""

# Scene intersection and hit records
# Cornell box scene
from .cornell_box import (
    BOX_SIZE,
    create_cornell_box_scene,
    get_cornell_box_bounds,
    get_light_quad_info,
)
from .intersection import (
    MAX_QUADS,
    MAX_SPHERES,
    SceneHitRecord,
    add_quad,
    add_sphere,
    clear_scene,
    get_quad_count,
    get_sphere_count,
    intersect_scene,
    intersect_scene_any,
)

# Scene manager for coordinating primitives and materials
from .manager import (
    MAX_MATERIALS,
    MaterialInfo,
    MaterialType,
    QuadInfo,
    SceneConfig,
    SceneManager,
    SphereInfo,
    get_material_type,
    get_material_type_index,
    material_type_indices,
    material_types,
    num_materials,
)

__all__ = [
    # Intersection module
    "SceneHitRecord",
    "add_sphere",
    "add_quad",
    "clear_scene",
    "get_sphere_count",
    "get_quad_count",
    "intersect_scene",
    "intersect_scene_any",
    "MAX_SPHERES",
    "MAX_QUADS",
    # Manager module
    "SceneManager",
    "MaterialType",
    "MaterialInfo",
    "SphereInfo",
    "QuadInfo",
    "SceneConfig",
    "MAX_MATERIALS",
    "get_material_type",
    "get_material_type_index",
    "material_types",
    "material_type_indices",
    "num_materials",
    # Cornell box module
    "create_cornell_box_scene",
    "get_cornell_box_bounds",
    "get_light_quad_info",
    "BOX_SIZE",
]
