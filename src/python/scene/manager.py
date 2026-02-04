"""Unified scene manager for coordinating primitives and materials.

This module provides a high-level scene management API that coordinates
primitive storage (spheres, quads) with material assignment. It tracks
which material type (Lambertian, Metal, Dielectric) each material ID
corresponds to, enabling proper material dispatch in the path tracer.

The SceneManager maintains:
- A unified material_id space across all material types
- Mapping from material_id to (material_type, type_local_index)
- High-level methods for adding objects with materials in one call
- Scene serialization/configuration support

Example:
    >>> import taichi as ti
    >>> ti.init(arch=ti.gpu)
    >>> from src.python.scene.manager import SceneManager
    >>> scene = SceneManager()
    >>> mat_id = scene.add_lambertian_material(albedo=(0.8, 0.3, 0.3))
    >>> scene.add_sphere_with_material(center=(0, 0, -1), radius=0.5, material_id=mat_id)
    >>> # Use scene.get_material_type(mat_id) in path tracer for dispatch
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import taichi as ti
import taichi.math as tm

from src.python.materials.dielectric import (
    add_dielectric_material,
    clear_dielectric_materials,
)
from src.python.materials.lambertian import (
    add_lambertian_material,
    clear_lambertian_materials,
)
from src.python.materials.metal import (
    add_metal_material,
    clear_metal_materials,
)
from src.python.materials.phosphorescent import (
    add_phosphorescent_material as _add_phosphorescent_material,
)
from src.python.materials.phosphorescent import (
    clear_phosphorescent_materials,
)
from src.python.scene.intersection import (
    MAX_QUADS,
    MAX_SPHERES,
    add_quad,
    add_sphere,
    clear_scene,
    get_quad_count,
    get_sphere_count,
)

# Type alias for 3D vectors
vec3 = tm.vec3


class MaterialType(IntEnum):
    """Enumeration of supported material types.

    Used for material dispatch in the path tracer to determine which
    scattering function to call.
    """

    LAMBERTIAN = 0
    METAL = 1
    DIELECTRIC = 2
    PHOSPHORESCENT = 3


# Maximum number of materials across all types
MAX_MATERIALS = 1024  # 256 per type * 4 types

# Taichi fields for GPU-side material type lookup
# material_types[i] stores the MaterialType for material_id i
material_types = ti.field(dtype=ti.i32, shape=MAX_MATERIALS)
# material_type_indices[i] stores the type-local index for material_id i
# (e.g., if material_id 5 is the 2nd metal material, material_type_indices[5] = 1)
material_type_indices = ti.field(dtype=ti.i32, shape=MAX_MATERIALS)
num_materials = ti.field(dtype=ti.i32, shape=())


def _clear_material_tracking() -> None:
    """Clear the material tracking fields."""
    num_materials[None] = 0


@ti.func
def get_material_type(material_id: ti.i32) -> ti.i32:
    """Get the material type for a given material ID.

    This is a Taichi function for use in GPU kernels.

    Args:
        material_id: The unified material ID.

    Returns:
        The material type as an integer (see MaterialType enum).
        Returns -1 for invalid material IDs.
    """
    result = -1
    if 0 <= material_id < num_materials[None]:
        result = material_types[material_id]
    return result


@ti.func
def get_material_type_index(material_id: ti.i32) -> ti.i32:
    """Get the type-local index for a given material ID.

    This is used to look up material properties in the type-specific
    material arrays (e.g., lambertian_albedos[type_index]).

    Args:
        material_id: The unified material ID.

    Returns:
        The index into the type-specific material array.
        Returns -1 for invalid material IDs.
    """
    result = -1
    if 0 <= material_id < num_materials[None]:
        result = material_type_indices[material_id]
    return result


@dataclass
class MaterialInfo:
    """Information about a registered material.

    Attributes:
        material_id: The unified material ID.
        material_type: The type of material (Lambertian, Metal, Dielectric).
        type_index: The index within the type-specific material array.
        params: The material parameters as provided during creation.
    """

    material_id: int
    material_type: MaterialType
    type_index: int
    params: dict[str, Any]


@dataclass
class SphereInfo:
    """Information about a sphere in the scene.

    Attributes:
        sphere_index: The index in the sphere storage arrays.
        center: The center of the sphere.
        radius: The radius of the sphere.
        material_id: The material ID assigned to the sphere.
    """

    sphere_index: int
    center: tuple[float, float, float]
    radius: float
    material_id: int


@dataclass
class QuadInfo:
    """Information about a quad in the scene.

    Attributes:
        quad_index: The index in the quad storage arrays.
        corner: The corner point (Q) of the quad.
        edge_u: The first edge vector.
        edge_v: The second edge vector.
        material_id: The material ID assigned to the quad.
    """

    quad_index: int
    corner: tuple[float, float, float]
    edge_u: tuple[float, float, float]
    edge_v: tuple[float, float, float]
    material_id: int


@dataclass
class SceneConfig:
    """Configuration for scene serialization.

    Attributes:
        materials: List of material configurations.
        spheres: List of sphere configurations.
        quads: List of quad configurations.
    """

    materials: list[dict[str, Any]] = field(default_factory=list)
    spheres: list[dict[str, Any]] = field(default_factory=list)
    quads: list[dict[str, Any]] = field(default_factory=list)


class SceneManager:
    """Unified scene manager coordinating primitives and materials.

    The SceneManager provides a high-level API for building scenes with
    automatic material tracking. It maintains a unified material_id space
    that maps to type-specific material registries, enabling the path tracer
    to dispatch to the correct scattering function.

    Attributes:
        materials: List of MaterialInfo for all registered materials.
        spheres: List of SphereInfo for all spheres in the scene.
        quads: List of QuadInfo for all quads in the scene.

    Example:
        >>> scene = SceneManager()
        >>> # Add materials
        >>> red_diffuse = scene.add_lambertian_material(albedo=(0.8, 0.1, 0.1))
        >>> gold_metal = scene.add_metal_material(albedo=(0.8, 0.6, 0.2), roughness=0.3)
        >>> glass = scene.add_dielectric_material(ior=1.5)
        >>> # Add objects with materials
        >>> scene.add_sphere_with_material((0, 0, -1), 0.5, red_diffuse)
        >>> scene.add_sphere_with_material((1, 0, -1), 0.5, gold_metal)
        >>> scene.add_sphere_with_material((-1, 0, -1), 0.5, glass)
    """

    def __init__(self) -> None:
        """Initialize an empty scene."""
        self.materials: list[MaterialInfo] = []
        self.spheres: list[SphereInfo] = []
        self.quads: list[QuadInfo] = []
        self._clear_all()

    def _clear_all(self) -> None:
        """Clear all scene data including Taichi fields."""
        # Clear primitive storage
        clear_scene()
        # Clear material registries
        clear_lambertian_materials()
        clear_metal_materials()
        clear_dielectric_materials()
        clear_phosphorescent_materials()
        # Clear material tracking
        _clear_material_tracking()
        # Clear local tracking
        self.materials.clear()
        self.spheres.clear()
        self.quads.clear()

    def clear(self) -> None:
        """Clear the entire scene (primitives and materials).

        Resets all Taichi fields and internal tracking structures.
        """
        self._clear_all()

    # =========================================================================
    # Material Management
    # =========================================================================

    def add_lambertian_material(
        self,
        albedo: tuple[float, float, float],
    ) -> int:
        """Add a Lambertian (diffuse) material to the scene.

        Args:
            albedo: The diffuse reflectance color as (R, G, B) tuple.
                Each component should be in [0, 1] for energy conservation.

        Returns:
            The unified material ID for this material.

        Raises:
            RuntimeError: If the maximum number of materials is exceeded.
            ValueError: If any albedo component is outside [0, 1].
        """
        # Add to type-specific registry
        type_index = add_lambertian_material(albedo)

        # Assign unified material ID
        material_id = num_materials[None]
        if material_id >= MAX_MATERIALS:
            raise RuntimeError(f"Maximum number of materials ({MAX_MATERIALS}) exceeded")

        # Update Taichi fields
        material_types[material_id] = int(MaterialType.LAMBERTIAN)
        material_type_indices[material_id] = type_index
        num_materials[None] = material_id + 1

        # Track locally
        info = MaterialInfo(
            material_id=material_id,
            material_type=MaterialType.LAMBERTIAN,
            type_index=type_index,
            params={"albedo": albedo},
        )
        self.materials.append(info)

        return material_id

    def add_metal_material(
        self,
        albedo: tuple[float, float, float],
        roughness: float = 0.0,
    ) -> int:
        """Add a metal (specular reflective) material to the scene.

        Args:
            albedo: The reflective color as (R, G, B) tuple.
                Each component should be in [0, 1].
            roughness: The surface roughness in [0, 1]. Default is 0 (perfect mirror).

        Returns:
            The unified material ID for this material.

        Raises:
            RuntimeError: If the maximum number of materials is exceeded.
            ValueError: If any albedo component is outside [0, 1].
            ValueError: If roughness is outside [0, 1].
        """
        # Add to type-specific registry
        type_index = add_metal_material(albedo, roughness)

        # Assign unified material ID
        material_id = num_materials[None]
        if material_id >= MAX_MATERIALS:
            raise RuntimeError(f"Maximum number of materials ({MAX_MATERIALS}) exceeded")

        # Update Taichi fields
        material_types[material_id] = int(MaterialType.METAL)
        material_type_indices[material_id] = type_index
        num_materials[None] = material_id + 1

        # Track locally
        info = MaterialInfo(
            material_id=material_id,
            material_type=MaterialType.METAL,
            type_index=type_index,
            params={"albedo": albedo, "roughness": roughness},
        )
        self.materials.append(info)

        return material_id

    def add_dielectric_material(
        self,
        ior: float = 1.5,
    ) -> int:
        """Add a dielectric (glass/water) material to the scene.

        Args:
            ior: Index of refraction. Default is 1.5 (typical glass).
                Common values: Air=1.0, Water=1.33, Glass=1.5, Diamond=2.4

        Returns:
            The unified material ID for this material.

        Raises:
            RuntimeError: If the maximum number of materials is exceeded.
            ValueError: If IOR is less than 1.0.
        """
        # Add to type-specific registry
        type_index = add_dielectric_material(ior)

        # Assign unified material ID
        material_id = num_materials[None]
        if material_id >= MAX_MATERIALS:
            raise RuntimeError(f"Maximum number of materials ({MAX_MATERIALS}) exceeded")

        # Update Taichi fields
        material_types[material_id] = int(MaterialType.DIELECTRIC)
        material_type_indices[material_id] = type_index
        num_materials[None] = material_id + 1

        # Track locally
        info = MaterialInfo(
            material_id=material_id,
            material_type=MaterialType.DIELECTRIC,
            type_index=type_index,
            params={"ior": ior},
        )
        self.materials.append(info)

        return material_id

    def add_phosphorescent_material(
        self,
        albedo: tuple[float, float, float],
        glow_color: tuple[float, float, float],
        glow_intensity: float,
    ) -> int:
        """Add a phosphorescent (glow-in-the-dark) material to the scene.

        Phosphorescent materials both scatter light diffusely (like Lambertian)
        and emit light. The emission is constant and does not depend on incoming
        light.

        Args:
            albedo: The diffuse reflectance color as (R, G, B) tuple.
                Each component should be in [0, 1] for energy conservation.
            glow_color: The emission color as (R, G, B) tuple.
                Values can exceed 1.0 for HDR effects.
            glow_intensity: The emission strength. Typical range is 0-10.
                Must be non-negative.

        Returns:
            The unified material ID for this material.

        Raises:
            RuntimeError: If the maximum number of materials is exceeded.
            ValueError: If any albedo component is outside [0, 1].
            ValueError: If any glow_color component is negative.
            ValueError: If glow_intensity is negative.
        """
        # Add to type-specific registry (validation happens there)
        type_index = _add_phosphorescent_material(albedo, glow_color, glow_intensity)

        # Assign unified material ID
        material_id = num_materials[None]
        if material_id >= MAX_MATERIALS:
            raise RuntimeError(f"Maximum number of materials ({MAX_MATERIALS}) exceeded")

        # Update Taichi fields
        material_types[material_id] = int(MaterialType.PHOSPHORESCENT)
        material_type_indices[material_id] = type_index
        num_materials[None] = material_id + 1

        # Track locally
        info = MaterialInfo(
            material_id=material_id,
            material_type=MaterialType.PHOSPHORESCENT,
            type_index=type_index,
            params={
                "albedo": albedo,
                "glow_color": glow_color,
                "glow_intensity": glow_intensity,
            },
        )
        self.materials.append(info)

        return material_id

    def get_material_count(self) -> int:
        """Get the total number of materials in the scene."""
        return int(num_materials[None])

    def get_material_info(self, material_id: int) -> MaterialInfo | None:
        """Get information about a material by ID.

        Args:
            material_id: The unified material ID.

        Returns:
            MaterialInfo for the material, or None if not found.
        """
        if 0 <= material_id < len(self.materials):
            return self.materials[material_id]
        return None

    def get_material_type_python(self, material_id: int) -> MaterialType | None:
        """Get the material type for a given material ID (Python side).

        This is a Python function for use outside of Taichi kernels.
        For GPU-side lookup, use the get_material_type() Taichi function.

        Args:
            material_id: The unified material ID.

        Returns:
            The MaterialType, or None for invalid material IDs.
        """
        if 0 <= material_id < len(self.materials):
            return self.materials[material_id].material_type
        return None

    # =========================================================================
    # Primitive Management
    # =========================================================================

    def add_sphere(
        self,
        center: tuple[float, float, float],
        radius: float,
        material_id: int,
    ) -> int:
        """Add a sphere to the scene.

        Args:
            center: The center point of the sphere as (x, y, z).
            radius: The radius of the sphere (should be positive).
            material_id: The unified material ID to assign to the sphere.

        Returns:
            The index of the added sphere.

        Raises:
            RuntimeError: If the maximum number of spheres is exceeded.
            ValueError: If material_id is invalid.
        """
        if material_id < 0 or material_id >= num_materials[None]:
            raise ValueError(f"Invalid material_id: {material_id}")

        center_vec = vec3(center[0], center[1], center[2])
        sphere_index = add_sphere(center_vec, radius, material_id)

        info = SphereInfo(
            sphere_index=sphere_index,
            center=center,
            radius=radius,
            material_id=material_id,
        )
        self.spheres.append(info)

        return sphere_index

    def add_quad(
        self,
        corner: tuple[float, float, float],
        edge_u: tuple[float, float, float],
        edge_v: tuple[float, float, float],
        material_id: int,
    ) -> int:
        """Add a quad (parallelogram) to the scene.

        The quad represents a parallelogram with vertices at:
        corner, corner+edge_u, corner+edge_v, corner+edge_u+edge_v

        Args:
            corner: The corner point (Q) of the quad as (x, y, z).
            edge_u: The first edge vector as (x, y, z).
            edge_v: The second edge vector as (x, y, z).
            material_id: The unified material ID to assign to the quad.

        Returns:
            The index of the added quad.

        Raises:
            RuntimeError: If the maximum number of quads is exceeded.
            ValueError: If material_id is invalid.
        """
        if material_id < 0 or material_id >= num_materials[None]:
            raise ValueError(f"Invalid material_id: {material_id}")

        q = vec3(corner[0], corner[1], corner[2])
        u = vec3(edge_u[0], edge_u[1], edge_u[2])
        v = vec3(edge_v[0], edge_v[1], edge_v[2])
        quad_index = add_quad(q, u, v, material_id)

        info = QuadInfo(
            quad_index=quad_index,
            corner=corner,
            edge_u=edge_u,
            edge_v=edge_v,
            material_id=material_id,
        )
        self.quads.append(info)

        return quad_index

    # =========================================================================
    # Convenience Methods (add object with material in one call)
    # =========================================================================

    def add_sphere_with_material(
        self,
        center: tuple[float, float, float],
        radius: float,
        material_id: int,
    ) -> tuple[int, int]:
        """Add a sphere with a pre-created material.

        This is a convenience alias for add_sphere() that makes the intent
        clearer when building scenes.

        Args:
            center: The center point of the sphere as (x, y, z).
            radius: The radius of the sphere.
            material_id: The unified material ID from add_*_material().

        Returns:
            Tuple of (sphere_index, material_id).
        """
        sphere_index = self.add_sphere(center, radius, material_id)
        return sphere_index, material_id

    def add_quad_with_material(
        self,
        corner: tuple[float, float, float],
        edge_u: tuple[float, float, float],
        edge_v: tuple[float, float, float],
        material_id: int,
    ) -> tuple[int, int]:
        """Add a quad with a pre-created material.

        This is a convenience alias for add_quad() that makes the intent
        clearer when building scenes.

        Args:
            corner: The corner point of the quad as (x, y, z).
            edge_u: The first edge vector as (x, y, z).
            edge_v: The second edge vector as (x, y, z).
            material_id: The unified material ID from add_*_material().

        Returns:
            Tuple of (quad_index, material_id).
        """
        quad_index = self.add_quad(corner, edge_u, edge_v, material_id)
        return quad_index, material_id

    def add_lambertian_sphere(
        self,
        center: tuple[float, float, float],
        radius: float,
        albedo: tuple[float, float, float],
    ) -> tuple[int, int]:
        """Add a sphere with a new Lambertian material.

        Convenience method that creates a Lambertian material and sphere
        in one call.

        Args:
            center: The center point of the sphere as (x, y, z).
            radius: The radius of the sphere.
            albedo: The diffuse reflectance color as (R, G, B).

        Returns:
            Tuple of (sphere_index, material_id).
        """
        material_id = self.add_lambertian_material(albedo)
        sphere_index = self.add_sphere(center, radius, material_id)
        return sphere_index, material_id

    def add_metal_sphere(
        self,
        center: tuple[float, float, float],
        radius: float,
        albedo: tuple[float, float, float],
        roughness: float = 0.0,
    ) -> tuple[int, int]:
        """Add a sphere with a new metal material.

        Convenience method that creates a metal material and sphere in one call.

        Args:
            center: The center point of the sphere as (x, y, z).
            radius: The radius of the sphere.
            albedo: The reflective color as (R, G, B).
            roughness: The surface roughness in [0, 1].

        Returns:
            Tuple of (sphere_index, material_id).
        """
        material_id = self.add_metal_material(albedo, roughness)
        sphere_index = self.add_sphere(center, radius, material_id)
        return sphere_index, material_id

    def add_dielectric_sphere(
        self,
        center: tuple[float, float, float],
        radius: float,
        ior: float = 1.5,
    ) -> tuple[int, int]:
        """Add a sphere with a new dielectric material.

        Convenience method that creates a dielectric material and sphere
        in one call.

        Args:
            center: The center point of the sphere as (x, y, z).
            radius: The radius of the sphere.
            ior: Index of refraction. Default is 1.5 (glass).

        Returns:
            Tuple of (sphere_index, material_id).
        """
        material_id = self.add_dielectric_material(ior)
        sphere_index = self.add_sphere(center, radius, material_id)
        return sphere_index, material_id

    def add_lambertian_quad(
        self,
        corner: tuple[float, float, float],
        edge_u: tuple[float, float, float],
        edge_v: tuple[float, float, float],
        albedo: tuple[float, float, float],
    ) -> tuple[int, int]:
        """Add a quad with a new Lambertian material.

        Convenience method that creates a Lambertian material and quad
        in one call.

        Args:
            corner: The corner point of the quad as (x, y, z).
            edge_u: The first edge vector.
            edge_v: The second edge vector.
            albedo: The diffuse reflectance color as (R, G, B).

        Returns:
            Tuple of (quad_index, material_id).
        """
        material_id = self.add_lambertian_material(albedo)
        quad_index = self.add_quad(corner, edge_u, edge_v, material_id)
        return quad_index, material_id

    def add_phosphorescent_sphere(
        self,
        center: tuple[float, float, float],
        radius: float,
        albedo: tuple[float, float, float],
        glow_color: tuple[float, float, float],
        glow_intensity: float,
    ) -> tuple[int, int]:
        """Add a sphere with a new phosphorescent material.

        Convenience method that creates a phosphorescent material and sphere
        in one call.

        Args:
            center: The center point of the sphere as (x, y, z).
            radius: The radius of the sphere.
            albedo: The diffuse reflectance color as (R, G, B).
            glow_color: The emission color as (R, G, B).
            glow_intensity: The emission strength.

        Returns:
            Tuple of (sphere_index, material_id).
        """
        material_id = self.add_phosphorescent_material(albedo, glow_color, glow_intensity)
        sphere_index = self.add_sphere(center, radius, material_id)
        return sphere_index, material_id

    def add_phosphorescent_quad(
        self,
        corner: tuple[float, float, float],
        edge_u: tuple[float, float, float],
        edge_v: tuple[float, float, float],
        albedo: tuple[float, float, float],
        glow_color: tuple[float, float, float],
        glow_intensity: float,
    ) -> tuple[int, int]:
        """Add a quad with a new phosphorescent material.

        Convenience method that creates a phosphorescent material and quad
        in one call.

        Args:
            corner: The corner point of the quad as (x, y, z).
            edge_u: The first edge vector.
            edge_v: The second edge vector.
            albedo: The diffuse reflectance color as (R, G, B).
            glow_color: The emission color as (R, G, B).
            glow_intensity: The emission strength.

        Returns:
            Tuple of (quad_index, material_id).
        """
        material_id = self.add_phosphorescent_material(albedo, glow_color, glow_intensity)
        quad_index = self.add_quad(corner, edge_u, edge_v, material_id)
        return quad_index, material_id

    # =========================================================================
    # Scene Queries
    # =========================================================================

    def get_sphere_count(self) -> int:
        """Get the number of spheres in the scene."""
        return get_sphere_count()

    def get_quad_count(self) -> int:
        """Get the number of quads in the scene."""
        return get_quad_count()

    def get_primitive_count(self) -> int:
        """Get the total number of primitives in the scene."""
        return self.get_sphere_count() + self.get_quad_count()

    # =========================================================================
    # Scene Serialization
    # =========================================================================

    def to_config(self) -> SceneConfig:
        """Export the scene to a configuration object.

        Returns:
            A SceneConfig containing all materials and primitives.
        """
        config = SceneConfig()

        # Export materials
        for mat in self.materials:
            mat_config: dict[str, Any] = {
                "type": mat.material_type.name.lower(),
                **mat.params,
            }
            config.materials.append(mat_config)

        # Export spheres
        for sphere in self.spheres:
            sphere_config = {
                "center": list(sphere.center),
                "radius": sphere.radius,
                "material_id": sphere.material_id,
            }
            config.spheres.append(sphere_config)

        # Export quads
        for quad in self.quads:
            quad_config = {
                "corner": list(quad.corner),
                "edge_u": list(quad.edge_u),
                "edge_v": list(quad.edge_v),
                "material_id": quad.material_id,
            }
            config.quads.append(quad_config)

        return config

    def from_config(self, config: SceneConfig) -> None:
        """Load a scene from a configuration object.

        Clears the current scene and loads the configuration.

        Args:
            config: The scene configuration to load.

        Raises:
            ValueError: If the configuration contains invalid data.
        """
        self.clear()

        # Load materials first (needed for primitives)
        for mat_config in config.materials:
            mat_type = mat_config.get("type", "").lower()
            if mat_type == "lambertian":
                albedo_list = mat_config.get("albedo", [0.5, 0.5, 0.5])
                albedo: tuple[float, float, float] = (
                    albedo_list[0],
                    albedo_list[1],
                    albedo_list[2],
                )
                self.add_lambertian_material(albedo)
            elif mat_type == "metal":
                albedo_list = mat_config.get("albedo", [0.8, 0.8, 0.8])
                albedo = (albedo_list[0], albedo_list[1], albedo_list[2])
                roughness = mat_config.get("roughness", 0.0)
                self.add_metal_material(albedo, roughness)
            elif mat_type == "dielectric":
                ior = mat_config.get("ior", 1.5)
                self.add_dielectric_material(ior)
            elif mat_type == "phosphorescent":
                albedo_list = mat_config.get("albedo", [0.5, 0.5, 0.5])
                albedo = (albedo_list[0], albedo_list[1], albedo_list[2])
                glow_color_list = mat_config.get("glow_color", [0.0, 1.0, 0.0])
                glow_color: tuple[float, float, float] = (
                    glow_color_list[0],
                    glow_color_list[1],
                    glow_color_list[2],
                )
                glow_intensity = mat_config.get("glow_intensity", 1.0)
                self.add_phosphorescent_material(albedo, glow_color, glow_intensity)
            else:
                raise ValueError(f"Unknown material type: {mat_type}")

        # Load spheres
        for sphere_config in config.spheres:
            center_list = sphere_config.get("center", [0, 0, 0])
            center: tuple[float, float, float] = (
                center_list[0],
                center_list[1],
                center_list[2],
            )
            radius = sphere_config.get("radius", 1.0)
            material_id = sphere_config.get("material_id", 0)
            self.add_sphere(center, radius, material_id)

        # Load quads
        for quad_config in config.quads:
            corner_list = quad_config.get("corner", [0, 0, 0])
            corner: tuple[float, float, float] = (
                corner_list[0],
                corner_list[1],
                corner_list[2],
            )
            edge_u_list = quad_config.get("edge_u", [1, 0, 0])
            edge_u: tuple[float, float, float] = (
                edge_u_list[0],
                edge_u_list[1],
                edge_u_list[2],
            )
            edge_v_list = quad_config.get("edge_v", [0, 1, 0])
            edge_v: tuple[float, float, float] = (
                edge_v_list[0],
                edge_v_list[1],
                edge_v_list[2],
            )
            material_id = quad_config.get("material_id", 0)
            self.add_quad(corner, edge_u, edge_v, material_id)

    def to_dict(self) -> dict[str, Any]:
        """Export the scene to a dictionary (for JSON serialization).

        Returns:
            A dictionary representation of the scene.
        """
        config = self.to_config()
        return {
            "materials": config.materials,
            "spheres": config.spheres,
            "quads": config.quads,
        }

    def from_dict(self, data: dict[str, Any]) -> None:
        """Load a scene from a dictionary.

        Args:
            data: Dictionary with 'materials', 'spheres', 'quads' keys.
        """
        config = SceneConfig(
            materials=data.get("materials", []),
            spheres=data.get("spheres", []),
            quads=data.get("quads", []),
        )
        self.from_config(config)

    # =========================================================================
    # Capacity Information
    # =========================================================================

    @staticmethod
    def get_max_spheres() -> int:
        """Get the maximum number of spheres supported."""
        return MAX_SPHERES

    @staticmethod
    def get_max_quads() -> int:
        """Get the maximum number of quads supported."""
        return MAX_QUADS

    @staticmethod
    def get_max_materials() -> int:
        """Get the maximum number of materials supported."""
        return MAX_MATERIALS
