"""Unit tests for the SceneManager.

Tests cover:
- Material registration (Lambertian, Metal, Dielectric)
- Material type tracking and lookup
- Primitive addition with materials
- Convenience methods (add_*_sphere, add_*_quad)
- Scene serialization (to_config, from_config)
- Scene clearing
- GPU-side material type dispatch
"""

import pytest
import taichi as ti


@pytest.fixture
def fresh_scene():
    """Create a fresh SceneManager for each test."""
    from src.python.scene.manager import SceneManager

    scene = SceneManager()
    yield scene
    scene.clear()


class TestMaterialRegistration:
    """Tests for material registration."""

    def test_add_lambertian_material(self, fresh_scene):
        """Test adding a Lambertian material."""
        mat_id = fresh_scene.add_lambertian_material(albedo=(0.8, 0.3, 0.3))
        assert mat_id == 0
        assert fresh_scene.get_material_count() == 1

    def test_add_metal_material(self, fresh_scene):
        """Test adding a metal material."""
        mat_id = fresh_scene.add_metal_material(albedo=(0.8, 0.6, 0.2), roughness=0.3)
        assert mat_id == 0
        assert fresh_scene.get_material_count() == 1

    def test_add_dielectric_material(self, fresh_scene):
        """Test adding a dielectric material."""
        mat_id = fresh_scene.add_dielectric_material(ior=1.5)
        assert mat_id == 0
        assert fresh_scene.get_material_count() == 1

    def test_add_multiple_materials(self, fresh_scene):
        """Test adding multiple materials of different types."""
        id0 = fresh_scene.add_lambertian_material(albedo=(0.8, 0.3, 0.3))
        id1 = fresh_scene.add_metal_material(albedo=(0.8, 0.6, 0.2), roughness=0.0)
        id2 = fresh_scene.add_dielectric_material(ior=1.5)
        id3 = fresh_scene.add_lambertian_material(albedo=(0.1, 0.8, 0.1))

        assert id0 == 0
        assert id1 == 1
        assert id2 == 2
        assert id3 == 3
        assert fresh_scene.get_material_count() == 4

    def test_material_validation_albedo(self, fresh_scene):
        """Test that invalid albedo raises ValueError."""
        with pytest.raises(ValueError):
            fresh_scene.add_lambertian_material(albedo=(1.5, 0.0, 0.0))

        with pytest.raises(ValueError):
            fresh_scene.add_metal_material(albedo=(-0.1, 0.5, 0.5))

    def test_material_validation_roughness(self, fresh_scene):
        """Test that invalid roughness raises ValueError."""
        with pytest.raises(ValueError):
            fresh_scene.add_metal_material(albedo=(0.8, 0.8, 0.8), roughness=1.5)

        with pytest.raises(ValueError):
            fresh_scene.add_metal_material(albedo=(0.8, 0.8, 0.8), roughness=-0.1)

    def test_material_validation_ior(self, fresh_scene):
        """Test that invalid IOR raises ValueError."""
        with pytest.raises(ValueError):
            fresh_scene.add_dielectric_material(ior=0.5)


class TestMaterialTypeTracking:
    """Tests for material type tracking."""

    def test_get_material_type_python(self, fresh_scene):
        """Test getting material type from Python side."""
        from src.python.scene.manager import MaterialType

        fresh_scene.add_lambertian_material(albedo=(0.5, 0.5, 0.5))
        fresh_scene.add_metal_material(albedo=(0.8, 0.8, 0.8))
        fresh_scene.add_dielectric_material(ior=1.5)

        assert fresh_scene.get_material_type_python(0) == MaterialType.LAMBERTIAN
        assert fresh_scene.get_material_type_python(1) == MaterialType.METAL
        assert fresh_scene.get_material_type_python(2) == MaterialType.DIELECTRIC
        assert fresh_scene.get_material_type_python(99) is None

    def test_get_material_info(self, fresh_scene):
        """Test getting full material info."""
        from src.python.scene.manager import MaterialType

        fresh_scene.add_metal_material(albedo=(0.8, 0.6, 0.2), roughness=0.3)

        info = fresh_scene.get_material_info(0)
        assert info is not None
        assert info.material_id == 0
        assert info.material_type == MaterialType.METAL
        assert info.type_index == 0
        assert info.params["albedo"] == (0.8, 0.6, 0.2)
        assert info.params["roughness"] == 0.3

    def test_get_material_type_gpu(self, fresh_scene):
        """Test getting material type from GPU kernel."""
        from src.python.scene.manager import MaterialType, get_material_type

        fresh_scene.add_lambertian_material(albedo=(0.5, 0.5, 0.5))
        fresh_scene.add_metal_material(albedo=(0.8, 0.8, 0.8))
        fresh_scene.add_dielectric_material(ior=1.5)

        result = ti.field(dtype=ti.i32, shape=4)

        @ti.kernel
        def test_kernel():
            result[0] = get_material_type(0)
            result[1] = get_material_type(1)
            result[2] = get_material_type(2)
            result[3] = get_material_type(99)  # Invalid

        test_kernel()

        assert result[0] == int(MaterialType.LAMBERTIAN)
        assert result[1] == int(MaterialType.METAL)
        assert result[2] == int(MaterialType.DIELECTRIC)
        assert result[3] == -1  # Invalid

    def test_get_material_type_index_gpu(self, fresh_scene):
        """Test getting material type index from GPU kernel."""
        from src.python.scene.manager import get_material_type_index

        # Add materials in mixed order
        fresh_scene.add_lambertian_material(albedo=(0.5, 0.5, 0.5))  # id=0, lambertian[0]
        fresh_scene.add_metal_material(albedo=(0.8, 0.8, 0.8))  # id=1, metal[0]
        fresh_scene.add_lambertian_material(albedo=(0.2, 0.2, 0.8))  # id=2, lambertian[1]
        fresh_scene.add_dielectric_material(ior=1.5)  # id=3, dielectric[0]

        result = ti.field(dtype=ti.i32, shape=5)

        @ti.kernel
        def test_kernel():
            result[0] = get_material_type_index(0)  # lambertian[0]
            result[1] = get_material_type_index(1)  # metal[0]
            result[2] = get_material_type_index(2)  # lambertian[1]
            result[3] = get_material_type_index(3)  # dielectric[0]
            result[4] = get_material_type_index(99)  # Invalid

        test_kernel()

        assert result[0] == 0  # First lambertian
        assert result[1] == 0  # First metal
        assert result[2] == 1  # Second lambertian
        assert result[3] == 0  # First dielectric
        assert result[4] == -1  # Invalid


class TestPrimitiveAddition:
    """Tests for adding primitives with materials."""

    def test_add_sphere_with_material(self, fresh_scene):
        """Test adding a sphere with a material."""
        mat_id = fresh_scene.add_lambertian_material(albedo=(0.8, 0.3, 0.3))
        sphere_idx = fresh_scene.add_sphere(
            center=(0.0, 0.0, -1.0), radius=0.5, material_id=mat_id
        )

        assert sphere_idx == 0
        assert fresh_scene.get_sphere_count() == 1

    def test_add_quad_with_material(self, fresh_scene):
        """Test adding a quad with a material."""
        mat_id = fresh_scene.add_lambertian_material(albedo=(0.5, 0.5, 0.5))
        quad_idx = fresh_scene.add_quad(
            corner=(-1.0, -1.0, -2.0),
            edge_u=(2.0, 0.0, 0.0),
            edge_v=(0.0, 2.0, 0.0),
            material_id=mat_id,
        )

        assert quad_idx == 0
        assert fresh_scene.get_quad_count() == 1

    def test_add_sphere_invalid_material(self, fresh_scene):
        """Test that adding a sphere with invalid material_id raises ValueError."""
        with pytest.raises(ValueError):
            fresh_scene.add_sphere(center=(0.0, 0.0, 0.0), radius=1.0, material_id=999)

    def test_add_quad_invalid_material(self, fresh_scene):
        """Test that adding a quad with invalid material_id raises ValueError."""
        with pytest.raises(ValueError):
            fresh_scene.add_quad(
                corner=(0.0, 0.0, 0.0),
                edge_u=(1.0, 0.0, 0.0),
                edge_v=(0.0, 1.0, 0.0),
                material_id=999,
            )


class TestConvenienceMethods:
    """Tests for convenience methods (add_*_sphere, etc.)."""

    def test_add_lambertian_sphere(self, fresh_scene):
        """Test add_lambertian_sphere convenience method."""
        sphere_idx, mat_id = fresh_scene.add_lambertian_sphere(
            center=(0.0, 0.0, -1.0), radius=0.5, albedo=(0.8, 0.3, 0.3)
        )

        assert sphere_idx == 0
        assert mat_id == 0
        assert fresh_scene.get_sphere_count() == 1
        assert fresh_scene.get_material_count() == 1

    def test_add_metal_sphere(self, fresh_scene):
        """Test add_metal_sphere convenience method."""
        from src.python.scene.manager import MaterialType

        sphere_idx, mat_id = fresh_scene.add_metal_sphere(
            center=(1.0, 0.0, -1.0), radius=0.5, albedo=(0.8, 0.6, 0.2), roughness=0.3
        )

        assert sphere_idx == 0
        assert mat_id == 0
        assert fresh_scene.get_material_type_python(mat_id) == MaterialType.METAL

    def test_add_dielectric_sphere(self, fresh_scene):
        """Test add_dielectric_sphere convenience method."""
        from src.python.scene.manager import MaterialType

        sphere_idx, mat_id = fresh_scene.add_dielectric_sphere(
            center=(-1.0, 0.0, -1.0), radius=0.5, ior=1.5
        )

        assert sphere_idx == 0
        assert mat_id == 0
        assert fresh_scene.get_material_type_python(mat_id) == MaterialType.DIELECTRIC

    def test_add_lambertian_quad(self, fresh_scene):
        """Test add_lambertian_quad convenience method."""
        quad_idx, mat_id = fresh_scene.add_lambertian_quad(
            corner=(-1.0, -1.0, -2.0),
            edge_u=(2.0, 0.0, 0.0),
            edge_v=(0.0, 2.0, 0.0),
            albedo=(0.8, 0.8, 0.8),
        )

        assert quad_idx == 0
        assert mat_id == 0
        assert fresh_scene.get_quad_count() == 1


class TestSceneClearing:
    """Tests for scene clearing."""

    def test_clear_scene(self, fresh_scene):
        """Test that clear() resets all scene data."""
        # Add materials and primitives
        mat_id = fresh_scene.add_lambertian_material(albedo=(0.5, 0.5, 0.5))
        fresh_scene.add_sphere(center=(0.0, 0.0, -1.0), radius=0.5, material_id=mat_id)
        fresh_scene.add_quad(
            corner=(-1.0, -1.0, -2.0),
            edge_u=(2.0, 0.0, 0.0),
            edge_v=(0.0, 2.0, 0.0),
            material_id=mat_id,
        )

        assert fresh_scene.get_material_count() == 1
        assert fresh_scene.get_sphere_count() == 1
        assert fresh_scene.get_quad_count() == 1

        # Clear
        fresh_scene.clear()

        assert fresh_scene.get_material_count() == 0
        assert fresh_scene.get_sphere_count() == 0
        assert fresh_scene.get_quad_count() == 0


class TestSceneSerialization:
    """Tests for scene serialization."""

    def test_to_config(self, fresh_scene):
        """Test exporting scene to config."""
        # Build a simple scene
        mat0 = fresh_scene.add_lambertian_material(albedo=(0.8, 0.3, 0.3))
        mat1 = fresh_scene.add_metal_material(albedo=(0.8, 0.6, 0.2), roughness=0.3)
        fresh_scene.add_sphere(center=(0.0, 0.0, -1.0), radius=0.5, material_id=mat0)
        fresh_scene.add_quad(
            corner=(-1.0, -1.0, -2.0),
            edge_u=(2.0, 0.0, 0.0),
            edge_v=(0.0, 2.0, 0.0),
            material_id=mat1,
        )

        config = fresh_scene.to_config()

        assert len(config.materials) == 2
        assert config.materials[0]["type"] == "lambertian"
        assert config.materials[1]["type"] == "metal"
        assert len(config.spheres) == 1
        assert len(config.quads) == 1

    def test_from_config(self, fresh_scene):
        """Test loading scene from config."""
        from src.python.scene.manager import MaterialType, SceneConfig

        config = SceneConfig(
            materials=[
                {"type": "lambertian", "albedo": [0.8, 0.3, 0.3]},
                {"type": "metal", "albedo": [0.8, 0.6, 0.2], "roughness": 0.3},
                {"type": "dielectric", "ior": 1.5},
            ],
            spheres=[
                {"center": [0.0, 0.0, -1.0], "radius": 0.5, "material_id": 0},
                {"center": [1.0, 0.0, -1.0], "radius": 0.5, "material_id": 1},
            ],
            quads=[
                {
                    "corner": [-1.0, -1.0, -2.0],
                    "edge_u": [2.0, 0.0, 0.0],
                    "edge_v": [0.0, 2.0, 0.0],
                    "material_id": 2,
                }
            ],
        )

        fresh_scene.from_config(config)

        assert fresh_scene.get_material_count() == 3
        assert fresh_scene.get_sphere_count() == 2
        assert fresh_scene.get_quad_count() == 1
        assert fresh_scene.get_material_type_python(0) == MaterialType.LAMBERTIAN
        assert fresh_scene.get_material_type_python(1) == MaterialType.METAL
        assert fresh_scene.get_material_type_python(2) == MaterialType.DIELECTRIC

    def test_to_dict_from_dict(self, fresh_scene):
        """Test round-trip serialization via dict."""
        from src.python.scene.manager import MaterialType, SceneManager

        # Build a scene
        mat0 = fresh_scene.add_lambertian_material(albedo=(0.8, 0.3, 0.3))
        mat1 = fresh_scene.add_dielectric_material(ior=1.5)
        fresh_scene.add_sphere(center=(0.0, 0.0, -1.0), radius=0.5, material_id=mat0)
        fresh_scene.add_sphere(center=(1.0, 0.0, -1.0), radius=0.5, material_id=mat1)

        # Export
        data = fresh_scene.to_dict()

        # Import into a new scene
        scene2 = SceneManager()
        scene2.from_dict(data)

        assert scene2.get_material_count() == 2
        assert scene2.get_sphere_count() == 2
        assert scene2.get_material_type_python(0) == MaterialType.LAMBERTIAN
        assert scene2.get_material_type_python(1) == MaterialType.DIELECTRIC

        # Clean up
        scene2.clear()

    def test_from_config_invalid_material_type(self, fresh_scene):
        """Test that invalid material type raises ValueError."""
        from src.python.scene.manager import SceneConfig

        config = SceneConfig(
            materials=[{"type": "unknown_material"}],
            spheres=[],
            quads=[],
        )

        with pytest.raises(ValueError, match="Unknown material type"):
            fresh_scene.from_config(config)


class TestSceneQueries:
    """Tests for scene query methods."""

    def test_get_primitive_count(self, fresh_scene):
        """Test get_primitive_count returns sum of all primitives."""
        mat_id = fresh_scene.add_lambertian_material(albedo=(0.5, 0.5, 0.5))
        fresh_scene.add_sphere(center=(0.0, 0.0, -1.0), radius=0.5, material_id=mat_id)
        fresh_scene.add_sphere(center=(1.0, 0.0, -1.0), radius=0.5, material_id=mat_id)
        fresh_scene.add_quad(
            corner=(-1.0, -1.0, -2.0),
            edge_u=(2.0, 0.0, 0.0),
            edge_v=(0.0, 2.0, 0.0),
            material_id=mat_id,
        )

        assert fresh_scene.get_primitive_count() == 3


class TestCapacityInfo:
    """Tests for capacity information methods."""

    def test_capacity_methods(self, fresh_scene):
        """Test capacity getter methods."""
        assert fresh_scene.get_max_spheres() > 0
        assert fresh_scene.get_max_quads() > 0
        assert fresh_scene.get_max_materials() > 0


class TestIntegrationWithIntersection:
    """Tests that SceneManager works with intersection system."""

    def test_intersection_returns_correct_material_id(self, fresh_scene):
        """Test that intersection returns the material_id set by SceneManager."""
        from src.python.scene.intersection import intersect_scene, vec3

        # Add materials with distinct IDs
        mat0 = fresh_scene.add_lambertian_material(albedo=(0.8, 0.3, 0.3))
        mat1 = fresh_scene.add_metal_material(albedo=(0.8, 0.6, 0.2), roughness=0.0)

        # Add spheres at different distances
        fresh_scene.add_sphere(center=(0.0, 0.0, -3.0), radius=0.5, material_id=mat0)
        fresh_scene.add_sphere(center=(0.0, 0.0, -5.0), radius=0.5, material_id=mat1)

        hit = ti.field(dtype=ti.i32, shape=())
        material_id = ti.field(dtype=ti.i32, shape=())

        @ti.kernel
        def test_kernel():
            ray_origin = vec3(0.0, 0.0, 0.0)
            ray_direction = vec3(0.0, 0.0, -1.0)
            rec = intersect_scene(ray_origin, ray_direction, 0.001, 1000.0)
            hit[None] = rec.hit
            material_id[None] = rec.material_id

        test_kernel()

        assert hit[None] == 1
        assert material_id[None] == mat0  # Should hit the closer sphere


class TestMaterialDispatchIntegration:
    """Tests for full material dispatch workflow."""

    def test_scatter_dispatch_by_material_type(self, fresh_scene):
        """Test that we can dispatch to correct scatter function based on type."""
        from src.python.materials.dielectric import scatter_dielectric_by_id
        from src.python.materials.lambertian import scatter_lambertian_by_id
        from src.python.materials.metal import scatter_metal_by_id
        from src.python.scene.manager import (
            MaterialType,
            get_material_type,
            get_material_type_index,
        )

        # Build scene with all material types (IDs are 0, 1, 2 respectively)
        fresh_scene.add_lambertian_material(albedo=(0.8, 0.3, 0.3))  # id=0
        fresh_scene.add_metal_material(albedo=(0.8, 0.6, 0.2), roughness=0.0)  # id=1
        fresh_scene.add_dielectric_material(ior=1.5)  # id=2

        # Test GPU-side dispatch
        scattered_valid = ti.field(dtype=ti.i32, shape=3)

        @ti.kernel
        def test_dispatch():
            normal = ti.Vector([0.0, 1.0, 0.0])
            incident = ti.Vector([0.0, -1.0, 0.0])

            # Test Lambertian dispatch
            mat_id = 0
            mat_type = get_material_type(mat_id)
            type_idx = get_material_type_index(mat_id)
            if mat_type == int(MaterialType.LAMBERTIAN):
                direction, attenuation, pdf = scatter_lambertian_by_id(type_idx, normal)
                scattered_valid[0] = 1 if direction.norm() > 0.0 else 0

            # Test Metal dispatch
            mat_id = 1
            mat_type = get_material_type(mat_id)
            type_idx = get_material_type_index(mat_id)
            if mat_type == int(MaterialType.METAL):
                direction, attenuation, did_scatter = scatter_metal_by_id(
                    type_idx, incident, normal
                )
                scattered_valid[1] = did_scatter

            # Test Dielectric dispatch
            mat_id = 2
            mat_type = get_material_type(mat_id)
            type_idx = get_material_type_index(mat_id)
            if mat_type == int(MaterialType.DIELECTRIC):
                direction, attenuation, did_scatter = scatter_dielectric_by_id(
                    type_idx, incident, normal, 1  # front_face
                )
                scattered_valid[2] = did_scatter

        test_dispatch()

        assert scattered_valid[0] == 1  # Lambertian scattered
        assert scattered_valid[1] == 1  # Metal scattered (should reflect upward)
        assert scattered_valid[2] == 1  # Dielectric scattered


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
