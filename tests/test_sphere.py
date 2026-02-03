"""Unit tests for sphere intersection.

Tests cover:
- Ray hitting sphere from outside (front face)
- Ray missing sphere
- Ray starting inside sphere (back face)
- Ray tangent to sphere
- Numerical stability edge cases
"""

import pytest
import taichi as ti


class TestSphereBasics:
    """Tests for Sphere dataclass and basic operations."""

    def test_make_sphere(self):
        """Test make_sphere convenience function."""
        from src.python.geometry.sphere import make_sphere, vec3

        center_result = ti.field(dtype=ti.math.vec3, shape=())
        radius_result = ti.field(dtype=ti.f32, shape=())

        @ti.kernel
        def test_kernel():
            sphere = make_sphere(vec3(1.0, 2.0, 3.0), 0.5)
            center_result[None] = sphere.center
            radius_result[None] = sphere.radius

        test_kernel()
        c = center_result[None]
        assert abs(c[0] - 1.0) < 1e-6
        assert abs(c[1] - 2.0) < 1e-6
        assert abs(c[2] - 3.0) < 1e-6
        assert abs(radius_result[None] - 0.5) < 1e-6


class TestSphereIntersection:
    """Tests for ray-sphere intersection."""

    def test_hit_sphere_direct_hit(self):
        """Test ray hitting sphere head-on from outside."""
        from src.python.geometry.sphere import Sphere, hit_sphere, vec3

        hit = ti.field(dtype=ti.i32, shape=())
        t_val = ti.field(dtype=ti.f32, shape=())
        point = ti.field(dtype=ti.math.vec3, shape=())
        normal = ti.field(dtype=ti.math.vec3, shape=())
        front_face = ti.field(dtype=ti.i32, shape=())

        @ti.kernel
        def test_kernel():
            # Ray from z=5 pointing toward origin
            ray_origin = vec3(0.0, 0.0, 5.0)
            ray_direction = vec3(0.0, 0.0, -1.0)
            # Sphere at origin with radius 1
            sphere = Sphere(center=vec3(0.0, 0.0, 0.0), radius=1.0)

            record = hit_sphere(ray_origin, ray_direction, sphere, 0.001, 1000.0)
            hit[None] = record.hit
            t_val[None] = record.t
            point[None] = record.point
            normal[None] = record.normal
            front_face[None] = record.front_face

        test_kernel()
        assert hit[None] == 1
        # Should hit at z=1 (front of sphere), so t=4
        assert abs(t_val[None] - 4.0) < 1e-5
        # Hit point should be (0, 0, 1)
        p = point[None]
        assert abs(p[0]) < 1e-5
        assert abs(p[1]) < 1e-5
        assert abs(p[2] - 1.0) < 1e-5
        # Normal should point outward: (0, 0, 1)
        n = normal[None]
        assert abs(n[0]) < 1e-5
        assert abs(n[1]) < 1e-5
        assert abs(n[2] - 1.0) < 1e-5
        # Should be front face
        assert front_face[None] == 1

    def test_hit_sphere_miss(self):
        """Test ray missing sphere entirely."""
        from src.python.geometry.sphere import Sphere, hit_sphere, vec3

        hit = ti.field(dtype=ti.i32, shape=())

        @ti.kernel
        def test_kernel():
            # Ray parallel to sphere, offset to miss
            ray_origin = vec3(5.0, 0.0, 0.0)
            ray_direction = vec3(0.0, 0.0, -1.0)
            sphere = Sphere(center=vec3(0.0, 0.0, 0.0), radius=1.0)

            record = hit_sphere(ray_origin, ray_direction, sphere, 0.001, 1000.0)
            hit[None] = record.hit

        test_kernel()
        assert hit[None] == 0

    def test_hit_sphere_inside(self):
        """Test ray starting inside sphere (back face hit)."""
        from src.python.geometry.sphere import Sphere, hit_sphere, vec3

        hit = ti.field(dtype=ti.i32, shape=())
        t_val = ti.field(dtype=ti.f32, shape=())
        normal = ti.field(dtype=ti.math.vec3, shape=())
        front_face = ti.field(dtype=ti.i32, shape=())

        @ti.kernel
        def test_kernel():
            # Ray starting at center of sphere
            ray_origin = vec3(0.0, 0.0, 0.0)
            ray_direction = vec3(0.0, 0.0, 1.0)
            sphere = Sphere(center=vec3(0.0, 0.0, 0.0), radius=1.0)

            record = hit_sphere(ray_origin, ray_direction, sphere, 0.001, 1000.0)
            hit[None] = record.hit
            t_val[None] = record.t
            normal[None] = record.normal
            front_face[None] = record.front_face

        test_kernel()
        assert hit[None] == 1
        # Should hit at z=1 (radius), so t=1
        assert abs(t_val[None] - 1.0) < 1e-5
        # Normal should point inward (opposite to outward normal) for back face
        n = normal[None]
        assert abs(n[0]) < 1e-5
        assert abs(n[1]) < 1e-5
        assert abs(n[2] - (-1.0)) < 1e-5
        # Should be back face
        assert front_face[None] == 0

    def test_hit_sphere_tangent(self):
        """Test ray tangent to sphere (grazing hit)."""
        from src.python.geometry.sphere import Sphere, hit_sphere, vec3

        hit = ti.field(dtype=ti.i32, shape=())
        t_val = ti.field(dtype=ti.f32, shape=())
        point = ti.field(dtype=ti.math.vec3, shape=())

        @ti.kernel
        def test_kernel():
            # Ray tangent to unit sphere at (1, 0, 0)
            ray_origin = vec3(1.0, 0.0, -5.0)
            ray_direction = vec3(0.0, 0.0, 1.0)
            sphere = Sphere(center=vec3(0.0, 0.0, 0.0), radius=1.0)

            record = hit_sphere(ray_origin, ray_direction, sphere, 0.001, 1000.0)
            hit[None] = record.hit
            t_val[None] = record.t
            point[None] = record.point

        test_kernel()
        assert hit[None] == 1
        # Should hit at z=0, so t=5
        assert abs(t_val[None] - 5.0) < 1e-4
        # Hit point should be (1, 0, 0)
        p = point[None]
        assert abs(p[0] - 1.0) < 1e-4
        assert abs(p[1]) < 1e-4
        assert abs(p[2]) < 1e-4

    def test_hit_sphere_t_min_boundary(self):
        """Test that hits before t_min are rejected."""
        from src.python.geometry.sphere import Sphere, hit_sphere, vec3

        hit = ti.field(dtype=ti.i32, shape=())

        @ti.kernel
        def test_kernel():
            # Ray very close to sphere surface
            ray_origin = vec3(0.0, 0.0, 1.001)
            ray_direction = vec3(0.0, 0.0, -1.0)
            sphere = Sphere(center=vec3(0.0, 0.0, 0.0), radius=1.0)

            # t_min=0.01 should reject the first hit at t~0.001
            record = hit_sphere(ray_origin, ray_direction, sphere, 0.01, 1000.0)
            hit[None] = record.hit

        test_kernel()
        # Should still hit (second intersection at back of sphere)
        assert hit[None] == 1

    def test_hit_sphere_t_max_boundary(self):
        """Test that hits after t_max are rejected."""
        from src.python.geometry.sphere import Sphere, hit_sphere, vec3

        hit = ti.field(dtype=ti.i32, shape=())

        @ti.kernel
        def test_kernel():
            ray_origin = vec3(0.0, 0.0, 100.0)
            ray_direction = vec3(0.0, 0.0, -1.0)
            sphere = Sphere(center=vec3(0.0, 0.0, 0.0), radius=1.0)

            # t_max=50 should reject the hit at t=99
            record = hit_sphere(ray_origin, ray_direction, sphere, 0.001, 50.0)
            hit[None] = record.hit

        test_kernel()
        assert hit[None] == 0

    def test_hit_sphere_behind_ray(self):
        """Test that spheres behind ray origin are missed."""
        from src.python.geometry.sphere import Sphere, hit_sphere, vec3

        hit = ti.field(dtype=ti.i32, shape=())

        @ti.kernel
        def test_kernel():
            # Ray pointing away from sphere
            ray_origin = vec3(0.0, 0.0, 5.0)
            ray_direction = vec3(0.0, 0.0, 1.0)  # Pointing away
            sphere = Sphere(center=vec3(0.0, 0.0, 0.0), radius=1.0)

            record = hit_sphere(ray_origin, ray_direction, sphere, 0.001, 1000.0)
            hit[None] = record.hit

        test_kernel()
        assert hit[None] == 0

    def test_hit_sphere_normalized_normal(self):
        """Test that returned normal is unit length."""
        from src.python.geometry.sphere import Sphere, hit_sphere, vec3

        normal_length = ti.field(dtype=ti.f32, shape=())

        @ti.kernel
        def test_kernel():
            ray_origin = vec3(3.0, 4.0, 10.0)
            ray_direction = vec3(-0.3, -0.4, -1.0)
            sphere = Sphere(center=vec3(0.0, 0.0, 0.0), radius=5.0)

            record = hit_sphere(ray_origin, ray_direction, sphere, 0.001, 1000.0)
            n = record.normal
            normal_length[None] = ti.sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2])

        test_kernel()
        # Normal should be unit length
        assert abs(normal_length[None] - 1.0) < 1e-5

    def test_hit_sphere_oblique_angle(self):
        """Test ray hitting sphere at oblique angle."""
        import taichi.math as tm

        from src.python.geometry.sphere import Sphere, hit_sphere, vec3

        hit = ti.field(dtype=ti.i32, shape=())
        point = ti.field(dtype=ti.math.vec3, shape=())
        front_face = ti.field(dtype=ti.i32, shape=())

        @ti.kernel
        def test_kernel():
            # Ray at 45 degrees
            ray_origin = vec3(5.0, 0.0, 5.0)
            ray_direction = tm.normalize(vec3(-1.0, 0.0, -1.0))
            sphere = Sphere(center=vec3(0.0, 0.0, 0.0), radius=1.0)

            record = hit_sphere(ray_origin, ray_direction, sphere, 0.001, 1000.0)
            hit[None] = record.hit
            point[None] = record.point
            front_face[None] = record.front_face

        test_kernel()
        assert hit[None] == 1
        assert front_face[None] == 1
        # Hit point should be on sphere surface (distance from origin = 1)
        p = point[None]
        dist = (p[0] ** 2 + p[1] ** 2 + p[2] ** 2) ** 0.5
        assert abs(dist - 1.0) < 1e-4


class TestRobustQuadratic:
    """Tests for numerical robustness of the quadratic formula."""

    def test_near_tangent_stability(self):
        """Test stability when discriminant is nearly zero."""
        from src.python.geometry.sphere import Sphere, hit_sphere, vec3

        hit = ti.field(dtype=ti.i32, shape=())
        point = ti.field(dtype=ti.math.vec3, shape=())

        @ti.kernel
        def test_kernel():
            # Ray just barely grazing sphere
            # Offset by a tiny amount from exact tangent
            ray_origin = vec3(1.0 + 1e-7, 0.0, -5.0)
            ray_direction = vec3(0.0, 0.0, 1.0)
            sphere = Sphere(center=vec3(0.0, 0.0, 0.0), radius=1.0)

            record = hit_sphere(ray_origin, ray_direction, sphere, 0.001, 1000.0)
            hit[None] = record.hit
            point[None] = record.point

        test_kernel()
        # Should either hit or miss cleanly, no NaN or garbage
        h = hit[None]
        assert h == 0 or h == 1
        if h == 1:
            p = point[None]
            # No NaN values
            assert p[0] == p[0]  # NaN != NaN
            assert p[1] == p[1]
            assert p[2] == p[2]

    def test_large_sphere_large_distance(self):
        """Test numerical stability with large values."""
        from src.python.geometry.sphere import Sphere, hit_sphere, vec3

        hit = ti.field(dtype=ti.i32, shape=())
        t_val = ti.field(dtype=ti.f32, shape=())

        @ti.kernel
        def test_kernel():
            ray_origin = vec3(0.0, 0.0, 1e6)
            ray_direction = vec3(0.0, 0.0, -1.0)
            sphere = Sphere(center=vec3(0.0, 0.0, 0.0), radius=1000.0)

            record = hit_sphere(ray_origin, ray_direction, sphere, 0.001, 1e10)
            hit[None] = record.hit
            t_val[None] = record.t

        test_kernel()
        assert hit[None] == 1
        # Should hit at z=1000, so t ~ 999000
        # Note: f32 has ~7 significant digits, so at 1e6 we expect error of ~10-100
        assert abs(t_val[None] - 999000.0) < 100.0

    def test_small_sphere(self):
        """Test with very small sphere."""
        from src.python.geometry.sphere import Sphere, hit_sphere, vec3

        hit = ti.field(dtype=ti.i32, shape=())
        t_val = ti.field(dtype=ti.f32, shape=())

        @ti.kernel
        def test_kernel():
            ray_origin = vec3(0.0, 0.0, 1.0)
            ray_direction = vec3(0.0, 0.0, -1.0)
            sphere = Sphere(center=vec3(0.0, 0.0, 0.0), radius=0.001)

            record = hit_sphere(ray_origin, ray_direction, sphere, 1e-6, 1000.0)
            hit[None] = record.hit
            t_val[None] = record.t

        test_kernel()
        assert hit[None] == 1
        # Should hit at z=0.001, so t ~ 0.999
        assert abs(t_val[None] - 0.999) < 1e-4

    def test_unnormalized_ray_direction(self):
        """Test that intersection works with unnormalized ray direction."""
        from src.python.geometry.sphere import Sphere, hit_sphere, vec3

        hit = ti.field(dtype=ti.i32, shape=())
        point = ti.field(dtype=ti.math.vec3, shape=())

        @ti.kernel
        def test_kernel():
            ray_origin = vec3(0.0, 0.0, 5.0)
            # Unnormalized direction (length = 2)
            ray_direction = vec3(0.0, 0.0, -2.0)
            sphere = Sphere(center=vec3(0.0, 0.0, 0.0), radius=1.0)

            record = hit_sphere(ray_origin, ray_direction, sphere, 0.001, 1000.0)
            hit[None] = record.hit
            point[None] = record.point

        test_kernel()
        assert hit[None] == 1
        # Hit point should still be at (0, 0, 1) regardless of direction length
        p = point[None]
        assert abs(p[0]) < 1e-5
        assert abs(p[1]) < 1e-5
        assert abs(p[2] - 1.0) < 1e-5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
