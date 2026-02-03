"""Unit tests for scene-level intersection.

Tests cover:
- SceneHitRecord with material_id
- Single primitive intersection (sphere and quad)
- Multiple primitives with closest hit selection
- Mixed primitive types (spheres and quads)
- Shadow ray queries (any hit)
- Scene clearing and primitive counts
"""

import pytest
import taichi as ti


class TestSceneHitRecordBasics:
    """Tests for SceneHitRecord dataclass."""

    def test_scene_hit_record_has_material_id(self):
        """Test that SceneHitRecord includes material_id field."""
        from src.python.scene.intersection import SceneHitRecord, vec3

        result_material_id = ti.field(dtype=ti.i32, shape=())

        @ti.kernel
        def test_kernel():
            rec = SceneHitRecord(
                hit=1,
                t=5.0,
                point=vec3(0.0, 0.0, 0.0),
                normal=vec3(0.0, 0.0, 1.0),
                front_face=1,
                material_id=42,
            )
            result_material_id[None] = rec.material_id

        test_kernel()
        assert result_material_id[None] == 42

    def test_scene_hit_record_miss_has_negative_material_id(self):
        """Test that miss records have material_id = -1."""
        from src.python.scene.intersection import _make_miss_record

        result_hit = ti.field(dtype=ti.i32, shape=())
        result_material_id = ti.field(dtype=ti.i32, shape=())

        @ti.kernel
        def test_kernel():
            rec = _make_miss_record()
            result_hit[None] = rec.hit
            result_material_id[None] = rec.material_id

        test_kernel()
        assert result_hit[None] == 0
        assert result_material_id[None] == -1


class TestScenePrimitiveStorage:
    """Tests for scene primitive storage and management."""

    def test_add_sphere(self):
        """Test adding a sphere to the scene."""
        from src.python.scene.intersection import add_sphere, get_sphere_count, vec3

        assert get_sphere_count() == 0
        idx = add_sphere(vec3(1.0, 2.0, 3.0), 0.5, material_id=1)
        assert idx == 0
        assert get_sphere_count() == 1

    def test_add_quad(self):
        """Test adding a quad to the scene."""
        from src.python.scene.intersection import add_quad, get_quad_count, vec3

        assert get_quad_count() == 0
        idx = add_quad(vec3(0.0, 0.0, 0.0), vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0), material_id=2)
        assert idx == 0
        assert get_quad_count() == 1

    def test_clear_scene(self):
        """Test clearing all primitives from the scene."""
        from src.python.scene.intersection import (
            add_quad,
            add_sphere,
            clear_scene,
            get_quad_count,
            get_sphere_count,
            vec3,
        )

        add_sphere(vec3(0.0, 0.0, 0.0), 1.0, material_id=0)
        add_sphere(vec3(1.0, 0.0, 0.0), 0.5, material_id=1)
        add_quad(vec3(0.0, 0.0, 0.0), vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0), material_id=2)

        assert get_sphere_count() == 2
        assert get_quad_count() == 1

        clear_scene()

        assert get_sphere_count() == 0
        assert get_quad_count() == 0

    def test_multiple_spheres(self):
        """Test adding multiple spheres."""
        from src.python.scene.intersection import add_sphere, get_sphere_count, vec3

        for i in range(5):
            idx = add_sphere(vec3(float(i), 0.0, 0.0), 0.5, material_id=i)
            assert idx == i

        assert get_sphere_count() == 5

    def test_multiple_quads(self):
        """Test adding multiple quads."""
        from src.python.scene.intersection import add_quad, get_quad_count, vec3

        for i in range(5):
            idx = add_quad(
                vec3(0.0, float(i), 0.0),
                vec3(1.0, 0.0, 0.0),
                vec3(0.0, 0.0, 1.0),
                material_id=i + 10,
            )
            assert idx == i

        assert get_quad_count() == 5


class TestSingleSphereIntersection:
    """Tests for scene intersection with a single sphere."""

    def test_hit_single_sphere(self):
        """Test ray hitting a single sphere in the scene."""
        from src.python.scene.intersection import add_sphere, intersect_scene, vec3

        add_sphere(vec3(0.0, 0.0, -3.0), 1.0, material_id=5)

        hit = ti.field(dtype=ti.i32, shape=())
        t_val = ti.field(dtype=ti.f32, shape=())
        material_id = ti.field(dtype=ti.i32, shape=())

        @ti.kernel
        def test_kernel():
            ray_origin = vec3(0.0, 0.0, 0.0)
            ray_direction = vec3(0.0, 0.0, -1.0)
            rec = intersect_scene(ray_origin, ray_direction, 0.001, 1000.0)
            hit[None] = rec.hit
            t_val[None] = rec.t
            material_id[None] = rec.material_id

        test_kernel()
        assert hit[None] == 1
        assert abs(t_val[None] - 2.0) < 1e-5  # Hit at z=-2 (front of sphere)
        assert material_id[None] == 5

    def test_miss_single_sphere(self):
        """Test ray missing a single sphere in the scene."""
        from src.python.scene.intersection import add_sphere, intersect_scene, vec3

        add_sphere(vec3(0.0, 0.0, -3.0), 1.0, material_id=5)

        hit = ti.field(dtype=ti.i32, shape=())

        @ti.kernel
        def test_kernel():
            # Ray parallel to sphere, offset to miss
            ray_origin = vec3(5.0, 0.0, 0.0)
            ray_direction = vec3(0.0, 0.0, -1.0)
            rec = intersect_scene(ray_origin, ray_direction, 0.001, 1000.0)
            hit[None] = rec.hit

        test_kernel()
        assert hit[None] == 0


class TestSingleQuadIntersection:
    """Tests for scene intersection with a single quad."""

    def test_hit_single_quad(self):
        """Test ray hitting a single quad in the scene."""
        from src.python.scene.intersection import add_quad, intersect_scene, vec3

        # Quad at z=-5 spanning [-1,1] x [-1,1]
        add_quad(vec3(-1.0, -1.0, -5.0), vec3(2.0, 0.0, 0.0), vec3(0.0, 2.0, 0.0), material_id=7)

        hit = ti.field(dtype=ti.i32, shape=())
        t_val = ti.field(dtype=ti.f32, shape=())
        material_id = ti.field(dtype=ti.i32, shape=())

        @ti.kernel
        def test_kernel():
            ray_origin = vec3(0.0, 0.0, 0.0)
            ray_direction = vec3(0.0, 0.0, -1.0)
            rec = intersect_scene(ray_origin, ray_direction, 0.001, 1000.0)
            hit[None] = rec.hit
            t_val[None] = rec.t
            material_id[None] = rec.material_id

        test_kernel()
        assert hit[None] == 1
        assert abs(t_val[None] - 5.0) < 1e-5
        assert material_id[None] == 7

    def test_miss_single_quad(self):
        """Test ray missing a single quad in the scene."""
        from src.python.scene.intersection import add_quad, intersect_scene, vec3

        # Quad at z=-5
        add_quad(vec3(-1.0, -1.0, -5.0), vec3(2.0, 0.0, 0.0), vec3(0.0, 2.0, 0.0), material_id=7)

        hit = ti.field(dtype=ti.i32, shape=())

        @ti.kernel
        def test_kernel():
            # Ray outside quad bounds
            ray_origin = vec3(5.0, 0.0, 0.0)
            ray_direction = vec3(0.0, 0.0, -1.0)
            rec = intersect_scene(ray_origin, ray_direction, 0.001, 1000.0)
            hit[None] = rec.hit

        test_kernel()
        assert hit[None] == 0


class TestMultiplePrimitivesClosestHit:
    """Tests for closest hit selection with multiple primitives."""

    def test_closest_of_two_spheres(self):
        """Test that the closest of two spheres is returned."""
        from src.python.scene.intersection import add_sphere, intersect_scene, vec3

        # Closer sphere at z=-2
        add_sphere(vec3(0.0, 0.0, -2.0), 0.5, material_id=1)
        # Farther sphere at z=-5
        add_sphere(vec3(0.0, 0.0, -5.0), 0.5, material_id=2)

        hit = ti.field(dtype=ti.i32, shape=())
        t_val = ti.field(dtype=ti.f32, shape=())
        material_id = ti.field(dtype=ti.i32, shape=())

        @ti.kernel
        def test_kernel():
            ray_origin = vec3(0.0, 0.0, 0.0)
            ray_direction = vec3(0.0, 0.0, -1.0)
            rec = intersect_scene(ray_origin, ray_direction, 0.001, 1000.0)
            hit[None] = rec.hit
            t_val[None] = rec.t
            material_id[None] = rec.material_id

        test_kernel()
        assert hit[None] == 1
        # Should hit the closer sphere (front at z=-1.5, so t=1.5)
        assert abs(t_val[None] - 1.5) < 1e-5
        assert material_id[None] == 1

    def test_closest_of_two_spheres_reversed_order(self):
        """Test closest hit when farther sphere is added first."""
        from src.python.scene.intersection import add_sphere, intersect_scene, vec3

        # Farther sphere added first at z=-5
        add_sphere(vec3(0.0, 0.0, -5.0), 0.5, material_id=2)
        # Closer sphere at z=-2
        add_sphere(vec3(0.0, 0.0, -2.0), 0.5, material_id=1)

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
        # Should still hit the closer sphere (material_id=1)
        assert material_id[None] == 1

    def test_closest_of_two_quads(self):
        """Test that the closest of two quads is returned."""
        from src.python.scene.intersection import add_quad, intersect_scene, vec3

        # Closer quad at z=-2
        add_quad(vec3(-1.0, -1.0, -2.0), vec3(2.0, 0.0, 0.0), vec3(0.0, 2.0, 0.0), material_id=10)
        # Farther quad at z=-5
        add_quad(vec3(-1.0, -1.0, -5.0), vec3(2.0, 0.0, 0.0), vec3(0.0, 2.0, 0.0), material_id=11)

        hit = ti.field(dtype=ti.i32, shape=())
        t_val = ti.field(dtype=ti.f32, shape=())
        material_id = ti.field(dtype=ti.i32, shape=())

        @ti.kernel
        def test_kernel():
            ray_origin = vec3(0.0, 0.0, 0.0)
            ray_direction = vec3(0.0, 0.0, -1.0)
            rec = intersect_scene(ray_origin, ray_direction, 0.001, 1000.0)
            hit[None] = rec.hit
            t_val[None] = rec.t
            material_id[None] = rec.material_id

        test_kernel()
        assert hit[None] == 1
        assert abs(t_val[None] - 2.0) < 1e-5
        assert material_id[None] == 10


class TestMixedPrimitiveTypes:
    """Tests for scenes with both spheres and quads."""

    def test_sphere_closer_than_quad(self):
        """Test sphere is returned when closer than quad."""
        from src.python.scene.intersection import add_quad, add_sphere, intersect_scene, vec3

        # Closer sphere at z=-2
        add_sphere(vec3(0.0, 0.0, -2.0), 0.5, material_id=1)
        # Farther quad at z=-5
        add_quad(vec3(-1.0, -1.0, -5.0), vec3(2.0, 0.0, 0.0), vec3(0.0, 2.0, 0.0), material_id=10)

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
        assert material_id[None] == 1  # Sphere material

    def test_quad_closer_than_sphere(self):
        """Test quad is returned when closer than sphere."""
        from src.python.scene.intersection import add_quad, add_sphere, intersect_scene, vec3

        # Farther sphere at z=-5
        add_sphere(vec3(0.0, 0.0, -5.0), 0.5, material_id=1)
        # Closer quad at z=-2
        add_quad(vec3(-1.0, -1.0, -2.0), vec3(2.0, 0.0, 0.0), vec3(0.0, 2.0, 0.0), material_id=10)

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
        assert material_id[None] == 10  # Quad material

    def test_mixed_scene_with_multiple_primitives(self):
        """Test complex scene with multiple spheres and quads."""
        from src.python.scene.intersection import add_quad, add_sphere, intersect_scene, vec3

        # Add several primitives
        add_sphere(vec3(0.0, 0.0, -10.0), 1.0, material_id=1)  # Far
        add_quad(
            vec3(-1.0, -1.0, -8.0), vec3(2.0, 0.0, 0.0), vec3(0.0, 2.0, 0.0), material_id=2
        )  # Medium far
        add_sphere(vec3(0.0, 0.0, -3.0), 0.5, material_id=3)  # Closest!
        add_quad(
            vec3(-1.0, -1.0, -6.0), vec3(2.0, 0.0, 0.0), vec3(0.0, 2.0, 0.0), material_id=4
        )  # Medium

        hit = ti.field(dtype=ti.i32, shape=())
        material_id = ti.field(dtype=ti.i32, shape=())
        t_val = ti.field(dtype=ti.f32, shape=())

        @ti.kernel
        def test_kernel():
            ray_origin = vec3(0.0, 0.0, 0.0)
            ray_direction = vec3(0.0, 0.0, -1.0)
            rec = intersect_scene(ray_origin, ray_direction, 0.001, 1000.0)
            hit[None] = rec.hit
            material_id[None] = rec.material_id
            t_val[None] = rec.t

        test_kernel()
        assert hit[None] == 1
        # The closest sphere at z=-3 with radius 0.5 should be hit
        # Front of sphere is at z=-2.5, so t=2.5
        assert abs(t_val[None] - 2.5) < 1e-5
        assert material_id[None] == 3


class TestShadowRayQuery:
    """Tests for any-hit shadow ray queries."""

    def test_any_hit_with_occlusion(self):
        """Test any-hit returns 1 when something blocks the ray."""
        from src.python.scene.intersection import add_sphere, intersect_scene_any, vec3

        add_sphere(vec3(0.0, 0.0, -3.0), 1.0, material_id=0)

        hit_any = ti.field(dtype=ti.i32, shape=())

        @ti.kernel
        def test_kernel():
            ray_origin = vec3(0.0, 0.0, 0.0)
            ray_direction = vec3(0.0, 0.0, -1.0)
            hit_any[None] = intersect_scene_any(ray_origin, ray_direction, 0.001, 1000.0)

        test_kernel()
        assert hit_any[None] == 1

    def test_any_hit_without_occlusion(self):
        """Test any-hit returns 0 when nothing blocks the ray."""
        from src.python.scene.intersection import add_sphere, intersect_scene_any, vec3

        add_sphere(vec3(5.0, 0.0, -3.0), 1.0, material_id=0)  # Off to the side

        hit_any = ti.field(dtype=ti.i32, shape=())

        @ti.kernel
        def test_kernel():
            ray_origin = vec3(0.0, 0.0, 0.0)
            ray_direction = vec3(0.0, 0.0, -1.0)
            hit_any[None] = intersect_scene_any(ray_origin, ray_direction, 0.001, 1000.0)

        test_kernel()
        assert hit_any[None] == 0

    def test_any_hit_with_quad(self):
        """Test any-hit with quad occlusion."""
        from src.python.scene.intersection import add_quad, intersect_scene_any, vec3

        add_quad(vec3(-1.0, -1.0, -3.0), vec3(2.0, 0.0, 0.0), vec3(0.0, 2.0, 0.0), material_id=0)

        hit_any = ti.field(dtype=ti.i32, shape=())

        @ti.kernel
        def test_kernel():
            ray_origin = vec3(0.0, 0.0, 0.0)
            ray_direction = vec3(0.0, 0.0, -1.0)
            hit_any[None] = intersect_scene_any(ray_origin, ray_direction, 0.001, 1000.0)

        test_kernel()
        assert hit_any[None] == 1

    def test_any_hit_empty_scene(self):
        """Test any-hit with empty scene."""
        from src.python.scene.intersection import intersect_scene_any, vec3

        hit_any = ti.field(dtype=ti.i32, shape=())

        @ti.kernel
        def test_kernel():
            ray_origin = vec3(0.0, 0.0, 0.0)
            ray_direction = vec3(0.0, 0.0, -1.0)
            hit_any[None] = intersect_scene_any(ray_origin, ray_direction, 0.001, 1000.0)

        test_kernel()
        assert hit_any[None] == 0


class TestEmptyScene:
    """Tests for empty scene behavior."""

    def test_intersect_empty_scene(self):
        """Test intersection with empty scene returns miss."""
        from src.python.scene.intersection import intersect_scene, vec3

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
        assert hit[None] == 0
        assert material_id[None] == -1


class TestTBounds:
    """Tests for t_min and t_max bounds."""

    def test_hit_rejected_by_t_min(self):
        """Test that hits before t_min are rejected."""
        from src.python.scene.intersection import add_sphere, intersect_scene, vec3

        add_sphere(vec3(0.0, 0.0, -1.5), 0.5, material_id=1)  # Front at z=-1, so t=1

        hit = ti.field(dtype=ti.i32, shape=())

        @ti.kernel
        def test_kernel():
            ray_origin = vec3(0.0, 0.0, 0.0)
            ray_direction = vec3(0.0, 0.0, -1.0)
            # t_min=2.0 should reject the hit at t=1.0
            rec = intersect_scene(ray_origin, ray_direction, 2.0, 1000.0)
            hit[None] = rec.hit

        test_kernel()
        assert hit[None] == 0

    def test_hit_rejected_by_t_max(self):
        """Test that hits after t_max are rejected."""
        from src.python.scene.intersection import add_sphere, intersect_scene, vec3

        add_sphere(vec3(0.0, 0.0, -100.0), 1.0, material_id=1)  # Front at z=-99, so t=99

        hit = ti.field(dtype=ti.i32, shape=())

        @ti.kernel
        def test_kernel():
            ray_origin = vec3(0.0, 0.0, 0.0)
            ray_direction = vec3(0.0, 0.0, -1.0)
            # t_max=50.0 should reject the hit at t=99
            rec = intersect_scene(ray_origin, ray_direction, 0.001, 50.0)
            hit[None] = rec.hit

        test_kernel()
        assert hit[None] == 0


class TestNormalAndFrontFace:
    """Tests for normal and front_face propagation."""

    def test_sphere_normal_propagated(self):
        """Test that sphere normal is correctly propagated."""
        from src.python.scene.intersection import add_sphere, intersect_scene, vec3

        add_sphere(vec3(0.0, 0.0, -3.0), 1.0, material_id=0)

        normal = ti.field(dtype=ti.math.vec3, shape=())
        front_face = ti.field(dtype=ti.i32, shape=())

        @ti.kernel
        def test_kernel():
            ray_origin = vec3(0.0, 0.0, 0.0)
            ray_direction = vec3(0.0, 0.0, -1.0)
            rec = intersect_scene(ray_origin, ray_direction, 0.001, 1000.0)
            normal[None] = rec.normal
            front_face[None] = rec.front_face

        test_kernel()
        n = normal[None]
        # Normal should point toward ray (positive z)
        assert abs(n[0]) < 1e-5
        assert abs(n[1]) < 1e-5
        assert abs(n[2] - 1.0) < 1e-5
        assert front_face[None] == 1

    def test_quad_normal_propagated(self):
        """Test that quad normal is correctly propagated."""
        from src.python.scene.intersection import add_quad, intersect_scene, vec3

        add_quad(vec3(-1.0, -1.0, -3.0), vec3(2.0, 0.0, 0.0), vec3(0.0, 2.0, 0.0), material_id=0)

        normal = ti.field(dtype=ti.math.vec3, shape=())
        front_face = ti.field(dtype=ti.i32, shape=())

        @ti.kernel
        def test_kernel():
            ray_origin = vec3(0.0, 0.0, 0.0)
            ray_direction = vec3(0.0, 0.0, -1.0)
            rec = intersect_scene(ray_origin, ray_direction, 0.001, 1000.0)
            normal[None] = rec.normal
            front_face[None] = rec.front_face

        test_kernel()
        n = normal[None]
        # Normal should point toward ray (positive z)
        assert abs(n[0]) < 1e-5
        assert abs(n[1]) < 1e-5
        assert abs(n[2] - 1.0) < 1e-5
        assert front_face[None] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
