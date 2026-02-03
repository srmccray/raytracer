"""Unit tests for quad intersection.

Tests cover:
- Ray hitting quad center (front and back face)
- Ray hitting quad at edges and corners
- Ray missing quad (outside bounds)
- Ray parallel to quad plane (no intersection)
- Numerical stability edge cases
"""

import pytest
import taichi as ti


class TestQuadBasics:
    """Tests for Quad dataclass and basic operations."""

    def test_make_quad(self):
        """Test make_quad convenience function."""
        from src.python.geometry.quad import make_quad, vec3

        q_result = ti.field(dtype=ti.math.vec3, shape=())
        u_result = ti.field(dtype=ti.math.vec3, shape=())
        v_result = ti.field(dtype=ti.math.vec3, shape=())

        @ti.kernel
        def test_kernel():
            quad = make_quad(
                vec3(1.0, 2.0, 3.0),
                vec3(1.0, 0.0, 0.0),
                vec3(0.0, 1.0, 0.0),
            )
            q_result[None] = quad.Q
            u_result[None] = quad.u
            v_result[None] = quad.v

        test_kernel()
        q = q_result[None]
        u = u_result[None]
        v = v_result[None]
        assert abs(q[0] - 1.0) < 1e-6
        assert abs(q[1] - 2.0) < 1e-6
        assert abs(q[2] - 3.0) < 1e-6
        assert abs(u[0] - 1.0) < 1e-6
        assert abs(u[1]) < 1e-6
        assert abs(u[2]) < 1e-6
        assert abs(v[0]) < 1e-6
        assert abs(v[1] - 1.0) < 1e-6
        assert abs(v[2]) < 1e-6

    def test_quad_normal(self):
        """Test quad_normal computation."""
        from src.python.geometry.quad import Quad, quad_normal, vec3

        normal_result = ti.field(dtype=ti.math.vec3, shape=())

        @ti.kernel
        def test_kernel():
            # Quad in xy-plane, normal should point in +z direction
            quad = Quad(
                Q=vec3(0.0, 0.0, 0.0),
                u=vec3(1.0, 0.0, 0.0),
                v=vec3(0.0, 1.0, 0.0),
            )
            normal_result[None] = quad_normal(quad)

        test_kernel()
        n = normal_result[None]
        assert abs(n[0]) < 1e-6
        assert abs(n[1]) < 1e-6
        assert abs(n[2] - 1.0) < 1e-6

    def test_quad_area(self):
        """Test quad_area computation."""
        from src.python.geometry.quad import Quad, quad_area, vec3

        area_result = ti.field(dtype=ti.f32, shape=())

        @ti.kernel
        def test_kernel():
            # 2x3 quad, area should be 6
            quad = Quad(
                Q=vec3(0.0, 0.0, 0.0),
                u=vec3(2.0, 0.0, 0.0),
                v=vec3(0.0, 3.0, 0.0),
            )
            area_result[None] = quad_area(quad)

        test_kernel()
        assert abs(area_result[None] - 6.0) < 1e-5


class TestQuadIntersection:
    """Tests for ray-quad intersection."""

    def test_hit_quad_center(self):
        """Test ray hitting quad center from front."""
        from src.python.geometry.quad import Quad, hit_quad, vec3

        hit = ti.field(dtype=ti.i32, shape=())
        t_val = ti.field(dtype=ti.f32, shape=())
        point = ti.field(dtype=ti.math.vec3, shape=())
        normal = ti.field(dtype=ti.math.vec3, shape=())
        front_face = ti.field(dtype=ti.i32, shape=())

        @ti.kernel
        def test_kernel():
            # Quad at z=0, spanning [-1,1] x [-1,1]
            quad = Quad(
                Q=vec3(-1.0, -1.0, 0.0),
                u=vec3(2.0, 0.0, 0.0),
                v=vec3(0.0, 2.0, 0.0),
            )
            # Ray from z=5 pointing toward origin (center of quad)
            ray_origin = vec3(0.0, 0.0, 5.0)
            ray_direction = vec3(0.0, 0.0, -1.0)

            record = hit_quad(ray_origin, ray_direction, quad, 0.001, 1000.0)
            hit[None] = record.hit
            t_val[None] = record.t
            point[None] = record.point
            normal[None] = record.normal
            front_face[None] = record.front_face

        test_kernel()
        assert hit[None] == 1
        assert abs(t_val[None] - 5.0) < 1e-5
        p = point[None]
        assert abs(p[0]) < 1e-5
        assert abs(p[1]) < 1e-5
        assert abs(p[2]) < 1e-5
        n = normal[None]
        # Normal should point toward ray (positive z)
        assert abs(n[0]) < 1e-5
        assert abs(n[1]) < 1e-5
        assert abs(n[2] - 1.0) < 1e-5
        assert front_face[None] == 1

    def test_hit_quad_back_face(self):
        """Test ray hitting quad center from back."""
        from src.python.geometry.quad import Quad, hit_quad, vec3

        hit = ti.field(dtype=ti.i32, shape=())
        normal = ti.field(dtype=ti.math.vec3, shape=())
        front_face = ti.field(dtype=ti.i32, shape=())

        @ti.kernel
        def test_kernel():
            # Quad at z=0, spanning [-1,1] x [-1,1]
            quad = Quad(
                Q=vec3(-1.0, -1.0, 0.0),
                u=vec3(2.0, 0.0, 0.0),
                v=vec3(0.0, 2.0, 0.0),
            )
            # Ray from z=-5 pointing toward origin (hitting from back)
            ray_origin = vec3(0.0, 0.0, -5.0)
            ray_direction = vec3(0.0, 0.0, 1.0)

            record = hit_quad(ray_origin, ray_direction, quad, 0.001, 1000.0)
            hit[None] = record.hit
            normal[None] = record.normal
            front_face[None] = record.front_face

        test_kernel()
        assert hit[None] == 1
        n = normal[None]
        # Normal should point toward ray (negative z for back face)
        assert abs(n[0]) < 1e-5
        assert abs(n[1]) < 1e-5
        assert abs(n[2] - (-1.0)) < 1e-5
        assert front_face[None] == 0

    def test_hit_quad_edge(self):
        """Test ray hitting quad at edge (alpha=1, beta=0.5)."""
        from src.python.geometry.quad import Quad, hit_quad, vec3

        hit = ti.field(dtype=ti.i32, shape=())
        point = ti.field(dtype=ti.math.vec3, shape=())

        @ti.kernel
        def test_kernel():
            # Unit quad at origin
            quad = Quad(
                Q=vec3(0.0, 0.0, 0.0),
                u=vec3(1.0, 0.0, 0.0),
                v=vec3(0.0, 1.0, 0.0),
            )
            # Ray hitting at (1.0, 0.5, 0.0) - edge of quad
            ray_origin = vec3(1.0, 0.5, 5.0)
            ray_direction = vec3(0.0, 0.0, -1.0)

            record = hit_quad(ray_origin, ray_direction, quad, 0.001, 1000.0)
            hit[None] = record.hit
            point[None] = record.point

        test_kernel()
        assert hit[None] == 1
        p = point[None]
        assert abs(p[0] - 1.0) < 1e-5
        assert abs(p[1] - 0.5) < 1e-5
        assert abs(p[2]) < 1e-5

    def test_hit_quad_corner(self):
        """Test ray hitting quad at corner (alpha=0, beta=0)."""
        from src.python.geometry.quad import Quad, hit_quad, vec3

        hit = ti.field(dtype=ti.i32, shape=())
        point = ti.field(dtype=ti.math.vec3, shape=())

        @ti.kernel
        def test_kernel():
            # Unit quad at origin
            quad = Quad(
                Q=vec3(0.0, 0.0, 0.0),
                u=vec3(1.0, 0.0, 0.0),
                v=vec3(0.0, 1.0, 0.0),
            )
            # Ray hitting at Q corner
            ray_origin = vec3(0.0, 0.0, 5.0)
            ray_direction = vec3(0.0, 0.0, -1.0)

            record = hit_quad(ray_origin, ray_direction, quad, 0.001, 1000.0)
            hit[None] = record.hit
            point[None] = record.point

        test_kernel()
        assert hit[None] == 1
        p = point[None]
        assert abs(p[0]) < 1e-5
        assert abs(p[1]) < 1e-5
        assert abs(p[2]) < 1e-5

    def test_miss_quad_outside_bounds(self):
        """Test ray missing quad (outside alpha/beta bounds)."""
        from src.python.geometry.quad import Quad, hit_quad, vec3

        hit = ti.field(dtype=ti.i32, shape=())

        @ti.kernel
        def test_kernel():
            # Unit quad at origin
            quad = Quad(
                Q=vec3(0.0, 0.0, 0.0),
                u=vec3(1.0, 0.0, 0.0),
                v=vec3(0.0, 1.0, 0.0),
            )
            # Ray pointing at (-0.5, 0.5, 0) - outside quad
            ray_origin = vec3(-0.5, 0.5, 5.0)
            ray_direction = vec3(0.0, 0.0, -1.0)

            record = hit_quad(ray_origin, ray_direction, quad, 0.001, 1000.0)
            hit[None] = record.hit

        test_kernel()
        assert hit[None] == 0

    def test_miss_quad_parallel_ray(self):
        """Test ray parallel to quad plane (no intersection)."""
        from src.python.geometry.quad import Quad, hit_quad, vec3

        hit = ti.field(dtype=ti.i32, shape=())

        @ti.kernel
        def test_kernel():
            # Quad in xy-plane at z=0
            quad = Quad(
                Q=vec3(0.0, 0.0, 0.0),
                u=vec3(1.0, 0.0, 0.0),
                v=vec3(0.0, 1.0, 0.0),
            )
            # Ray parallel to quad plane
            ray_origin = vec3(0.5, 0.5, 1.0)
            ray_direction = vec3(1.0, 0.0, 0.0)

            record = hit_quad(ray_origin, ray_direction, quad, 0.001, 1000.0)
            hit[None] = record.hit

        test_kernel()
        assert hit[None] == 0

    def test_miss_quad_behind_ray(self):
        """Test quad behind ray origin."""
        from src.python.geometry.quad import Quad, hit_quad, vec3

        hit = ti.field(dtype=ti.i32, shape=())

        @ti.kernel
        def test_kernel():
            # Quad at z=0
            quad = Quad(
                Q=vec3(0.0, 0.0, 0.0),
                u=vec3(1.0, 0.0, 0.0),
                v=vec3(0.0, 1.0, 0.0),
            )
            # Ray pointing away from quad
            ray_origin = vec3(0.5, 0.5, 5.0)
            ray_direction = vec3(0.0, 0.0, 1.0)  # Pointing away

            record = hit_quad(ray_origin, ray_direction, quad, 0.001, 1000.0)
            hit[None] = record.hit

        test_kernel()
        assert hit[None] == 0

    def test_hit_quad_t_min_boundary(self):
        """Test that hits before t_min are rejected."""
        from src.python.geometry.quad import Quad, hit_quad, vec3

        hit = ti.field(dtype=ti.i32, shape=())

        @ti.kernel
        def test_kernel():
            quad = Quad(
                Q=vec3(0.0, 0.0, 0.0),
                u=vec3(1.0, 0.0, 0.0),
                v=vec3(0.0, 1.0, 0.0),
            )
            # Ray very close to quad
            ray_origin = vec3(0.5, 0.5, 0.001)
            ray_direction = vec3(0.0, 0.0, -1.0)

            # t_min=0.01 should reject hit at t~0.001
            record = hit_quad(ray_origin, ray_direction, quad, 0.01, 1000.0)
            hit[None] = record.hit

        test_kernel()
        assert hit[None] == 0

    def test_hit_quad_t_max_boundary(self):
        """Test that hits after t_max are rejected."""
        from src.python.geometry.quad import Quad, hit_quad, vec3

        hit = ti.field(dtype=ti.i32, shape=())

        @ti.kernel
        def test_kernel():
            quad = Quad(
                Q=vec3(0.0, 0.0, 0.0),
                u=vec3(1.0, 0.0, 0.0),
                v=vec3(0.0, 1.0, 0.0),
            )
            ray_origin = vec3(0.5, 0.5, 100.0)
            ray_direction = vec3(0.0, 0.0, -1.0)

            # t_max=50 should reject hit at t=100
            record = hit_quad(ray_origin, ray_direction, quad, 0.001, 50.0)
            hit[None] = record.hit

        test_kernel()
        assert hit[None] == 0


class TestQuadCornellBox:
    """Tests simulating Cornell box wall setup."""

    def test_floor_quad(self):
        """Test floor quad (y=0 plane)."""
        from src.python.geometry.quad import Quad, hit_quad, vec3

        hit = ti.field(dtype=ti.i32, shape=())
        point = ti.field(dtype=ti.math.vec3, shape=())
        normal = ti.field(dtype=ti.math.vec3, shape=())

        @ti.kernel
        def test_kernel():
            # Floor at y=0, spanning x=[0,555] and z=[0,555] (typical Cornell box)
            floor = Quad(
                Q=vec3(0.0, 0.0, 0.0),
                u=vec3(555.0, 0.0, 0.0),
                v=vec3(0.0, 0.0, 555.0),
            )
            # Ray from above, hitting floor center
            ray_origin = vec3(277.5, 500.0, 277.5)
            ray_direction = vec3(0.0, -1.0, 0.0)

            record = hit_quad(ray_origin, ray_direction, floor, 0.001, 1000.0)
            hit[None] = record.hit
            point[None] = record.point
            normal[None] = record.normal

        test_kernel()
        assert hit[None] == 1
        p = point[None]
        assert abs(p[0] - 277.5) < 1e-3
        assert abs(p[1]) < 1e-3
        assert abs(p[2] - 277.5) < 1e-3
        # Normal should point up (+y)
        n = normal[None]
        assert abs(n[0]) < 1e-5
        assert abs(n[1] - 1.0) < 1e-5
        assert abs(n[2]) < 1e-5

    def test_back_wall_quad(self):
        """Test back wall quad (z=555 plane)."""
        from src.python.geometry.quad import Quad, hit_quad, vec3

        hit = ti.field(dtype=ti.i32, shape=())
        normal = ti.field(dtype=ti.math.vec3, shape=())

        @ti.kernel
        def test_kernel():
            # Back wall at z=555
            back_wall = Quad(
                Q=vec3(0.0, 0.0, 555.0),
                u=vec3(555.0, 0.0, 0.0),
                v=vec3(0.0, 555.0, 0.0),
            )
            # Ray from front, hitting back wall
            ray_origin = vec3(277.5, 277.5, 0.0)
            ray_direction = vec3(0.0, 0.0, 1.0)

            record = hit_quad(ray_origin, ray_direction, back_wall, 0.001, 1000.0)
            hit[None] = record.hit
            normal[None] = record.normal

        test_kernel()
        assert hit[None] == 1
        # Normal should point toward viewer (-z)
        n = normal[None]
        assert abs(n[0]) < 1e-5
        assert abs(n[1]) < 1e-5
        assert abs(n[2] - (-1.0)) < 1e-5

    def test_left_wall_quad(self):
        """Test left (red) wall quad (x=0 plane)."""
        from src.python.geometry.quad import Quad, hit_quad, vec3

        hit = ti.field(dtype=ti.i32, shape=())
        normal = ti.field(dtype=ti.math.vec3, shape=())

        @ti.kernel
        def test_kernel():
            # Left wall at x=0
            left_wall = Quad(
                Q=vec3(0.0, 0.0, 0.0),
                u=vec3(0.0, 0.0, 555.0),
                v=vec3(0.0, 555.0, 0.0),
            )
            # Ray from right, hitting left wall
            ray_origin = vec3(500.0, 277.5, 277.5)
            ray_direction = vec3(-1.0, 0.0, 0.0)

            record = hit_quad(ray_origin, ray_direction, left_wall, 0.001, 1000.0)
            hit[None] = record.hit
            normal[None] = record.normal

        test_kernel()
        assert hit[None] == 1
        # Normal should point right (+x)
        n = normal[None]
        assert abs(n[0] - 1.0) < 1e-5
        assert abs(n[1]) < 1e-5
        assert abs(n[2]) < 1e-5


class TestQuadNumericalStability:
    """Tests for numerical robustness."""

    def test_oblique_angle_hit(self):
        """Test ray hitting quad at oblique angle."""
        from src.python.geometry.quad import Quad, hit_quad, vec3

        hit = ti.field(dtype=ti.i32, shape=())
        point = ti.field(dtype=ti.math.vec3, shape=())

        @ti.kernel
        def test_kernel():
            quad = Quad(
                Q=vec3(0.0, 0.0, 0.0),
                u=vec3(2.0, 0.0, 0.0),
                v=vec3(0.0, 2.0, 0.0),
            )
            # Ray at 45 degrees
            ray_origin = vec3(1.0, 1.0, 5.0)
            ray_direction = ti.math.normalize(vec3(0.1, -0.1, -1.0))

            record = hit_quad(ray_origin, ray_direction, quad, 0.001, 1000.0)
            hit[None] = record.hit
            point[None] = record.point

        test_kernel()
        assert hit[None] == 1
        # Hit point should be on plane (z=0)
        p = point[None]
        assert abs(p[2]) < 1e-4

    def test_near_edge_stability(self):
        """Test numerical stability near quad edges."""
        from src.python.geometry.quad import Quad, hit_quad, vec3

        hit = ti.field(dtype=ti.i32, shape=())

        @ti.kernel
        def test_kernel():
            quad = Quad(
                Q=vec3(0.0, 0.0, 0.0),
                u=vec3(1.0, 0.0, 0.0),
                v=vec3(0.0, 1.0, 0.0),
            )
            # Ray just barely inside edge (1.0 - epsilon)
            ray_origin = vec3(0.9999999, 0.5, 5.0)
            ray_direction = vec3(0.0, 0.0, -1.0)

            record = hit_quad(ray_origin, ray_direction, quad, 0.001, 1000.0)
            hit[None] = record.hit

        test_kernel()
        # Should hit (barely inside)
        assert hit[None] == 1

    def test_near_edge_miss(self):
        """Test ray just outside quad edge."""
        from src.python.geometry.quad import Quad, hit_quad, vec3

        hit = ti.field(dtype=ti.i32, shape=())

        @ti.kernel
        def test_kernel():
            quad = Quad(
                Q=vec3(0.0, 0.0, 0.0),
                u=vec3(1.0, 0.0, 0.0),
                v=vec3(0.0, 1.0, 0.0),
            )
            # Ray just outside edge (1.0 + epsilon)
            ray_origin = vec3(1.0001, 0.5, 5.0)
            ray_direction = vec3(0.0, 0.0, -1.0)

            record = hit_quad(ray_origin, ray_direction, quad, 0.001, 1000.0)
            hit[None] = record.hit

        test_kernel()
        # Should miss (just outside)
        assert hit[None] == 0

    def test_normalized_normal_output(self):
        """Test that returned normal is unit length."""
        from src.python.geometry.quad import Quad, hit_quad, vec3

        normal_length = ti.field(dtype=ti.f32, shape=())

        @ti.kernel
        def test_kernel():
            # Non-unit edge vectors
            quad = Quad(
                Q=vec3(0.0, 0.0, 0.0),
                u=vec3(5.0, 0.0, 0.0),
                v=vec3(0.0, 3.0, 0.0),
            )
            ray_origin = vec3(2.5, 1.5, 5.0)
            ray_direction = vec3(0.0, 0.0, -1.0)

            record = hit_quad(ray_origin, ray_direction, quad, 0.001, 1000.0)
            n = record.normal
            normal_length[None] = ti.sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2])

        test_kernel()
        assert abs(normal_length[None] - 1.0) < 1e-5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
