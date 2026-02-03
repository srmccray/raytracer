"""Unit tests for the ray module.

Tests cover:
- Ray dataclass and ray_at function
- Vector utility functions (dot, cross, normalize, length, reflect, refract)
- Random sampling functions for Monte Carlo
"""

import pytest
import taichi as ti


class TestRayBasics:
    """Tests for Ray dataclass and basic operations."""

    def test_ray_at_origin(self):
        """Test ray_at returns origin when t=0."""
        from src.python.core.ray import Ray, ray_at, vec3

        result = ti.field(dtype=ti.math.vec3, shape=())

        @ti.kernel
        def test_kernel():
            origin = vec3(1.0, 2.0, 3.0)
            direction = vec3(0.0, 0.0, -1.0)
            ray = Ray(origin=origin, direction=direction)
            result[None] = ray_at(ray, 0.0)

        test_kernel()
        r = result[None]
        assert abs(r[0] - 1.0) < 1e-6
        assert abs(r[1] - 2.0) < 1e-6
        assert abs(r[2] - 3.0) < 1e-6

    def test_ray_at_positive_t(self):
        """Test ray_at computes correct point along ray."""
        from src.python.core.ray import Ray, ray_at, vec3

        result = ti.field(dtype=ti.math.vec3, shape=())

        @ti.kernel
        def test_kernel():
            origin = vec3(0.0, 0.0, 0.0)
            direction = vec3(1.0, 0.0, 0.0)
            ray = Ray(origin=origin, direction=direction)
            result[None] = ray_at(ray, 5.0)

        test_kernel()
        r = result[None]
        assert abs(r[0] - 5.0) < 1e-6
        assert abs(r[1] - 0.0) < 1e-6
        assert abs(r[2] - 0.0) < 1e-6

    def test_ray_at_negative_t(self):
        """Test ray_at handles negative t (behind origin)."""
        from src.python.core.ray import Ray, ray_at, vec3

        result = ti.field(dtype=ti.math.vec3, shape=())

        @ti.kernel
        def test_kernel():
            origin = vec3(0.0, 0.0, 0.0)
            direction = vec3(0.0, 1.0, 0.0)
            ray = Ray(origin=origin, direction=direction)
            result[None] = ray_at(ray, -3.0)

        test_kernel()
        r = result[None]
        assert abs(r[0] - 0.0) < 1e-6
        assert abs(r[1] - (-3.0)) < 1e-6
        assert abs(r[2] - 0.0) < 1e-6

    def test_make_ray(self):
        """Test make_ray convenience function."""
        from src.python.core.ray import make_ray, ray_at, vec3

        result = ti.field(dtype=ti.math.vec3, shape=())

        @ti.kernel
        def test_kernel():
            ray = make_ray(vec3(1.0, 1.0, 1.0), vec3(0.0, 0.0, 1.0))
            result[None] = ray_at(ray, 2.0)

        test_kernel()
        r = result[None]
        assert abs(r[0] - 1.0) < 1e-6
        assert abs(r[1] - 1.0) < 1e-6
        assert abs(r[2] - 3.0) < 1e-6


class TestVectorUtilities:
    """Tests for vector utility functions."""

    def test_length(self):
        """Test vector length computation."""
        from src.python.core.ray import length, vec3

        result = ti.field(dtype=ti.f32, shape=())

        @ti.kernel
        def test_kernel():
            v = vec3(3.0, 4.0, 0.0)
            result[None] = length(v)

        test_kernel()
        assert abs(result[None] - 5.0) < 1e-6

    def test_length_squared(self):
        """Test squared length avoids sqrt."""
        from src.python.core.ray import length_squared, vec3

        result = ti.field(dtype=ti.f32, shape=())

        @ti.kernel
        def test_kernel():
            v = vec3(3.0, 4.0, 0.0)
            result[None] = length_squared(v)

        test_kernel()
        assert abs(result[None] - 25.0) < 1e-6

    def test_normalize(self):
        """Test vector normalization."""
        from src.python.core.ray import length, normalize, vec3

        result_vec = ti.field(dtype=ti.math.vec3, shape=())
        result_len = ti.field(dtype=ti.f32, shape=())

        @ti.kernel
        def test_kernel():
            v = vec3(3.0, 4.0, 0.0)
            n = normalize(v)
            result_vec[None] = n
            result_len[None] = length(n)

        test_kernel()
        assert abs(result_len[None] - 1.0) < 1e-6
        assert abs(result_vec[None][0] - 0.6) < 1e-6
        assert abs(result_vec[None][1] - 0.8) < 1e-6

    def test_dot_product(self):
        """Test dot product computation."""
        from src.python.core.ray import dot, vec3

        result = ti.field(dtype=ti.f32, shape=())

        @ti.kernel
        def test_kernel():
            a = vec3(1.0, 2.0, 3.0)
            b = vec3(4.0, 5.0, 6.0)
            result[None] = dot(a, b)

        test_kernel()
        # 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assert abs(result[None] - 32.0) < 1e-6

    def test_dot_product_perpendicular(self):
        """Test dot product of perpendicular vectors is zero."""
        from src.python.core.ray import dot, vec3

        result = ti.field(dtype=ti.f32, shape=())

        @ti.kernel
        def test_kernel():
            a = vec3(1.0, 0.0, 0.0)
            b = vec3(0.0, 1.0, 0.0)
            result[None] = dot(a, b)

        test_kernel()
        assert abs(result[None]) < 1e-6

    def test_cross_product(self):
        """Test cross product computation."""
        from src.python.core.ray import cross, vec3

        result = ti.field(dtype=ti.math.vec3, shape=())

        @ti.kernel
        def test_kernel():
            a = vec3(1.0, 0.0, 0.0)
            b = vec3(0.0, 1.0, 0.0)
            result[None] = cross(a, b)

        test_kernel()
        r = result[None]
        assert abs(r[0] - 0.0) < 1e-6
        assert abs(r[1] - 0.0) < 1e-6
        assert abs(r[2] - 1.0) < 1e-6

    def test_reflect(self):
        """Test reflection about a normal."""
        from src.python.core.ray import normalize, reflect, vec3

        result = ti.field(dtype=ti.math.vec3, shape=())

        @ti.kernel
        def test_kernel():
            # Incident ray at 45 degrees
            incident = normalize(vec3(1.0, -1.0, 0.0))
            normal = vec3(0.0, 1.0, 0.0)
            result[None] = reflect(incident, normal)

        test_kernel()
        r = result[None]
        # Should reflect to (1, 1, 0) normalized
        expected = 1.0 / (2.0**0.5)
        assert abs(r[0] - expected) < 1e-5
        assert abs(r[1] - expected) < 1e-5
        assert abs(r[2] - 0.0) < 1e-6

    def test_refract_no_tir(self):
        """Test refraction when no total internal reflection."""
        from src.python.core.ray import length, refract, vec3

        result = ti.field(dtype=ti.math.vec3, shape=())
        result_len = ti.field(dtype=ti.f32, shape=())

        @ti.kernel
        def test_kernel():
            # Ray going straight down into surface
            incident = vec3(0.0, -1.0, 0.0)
            normal = vec3(0.0, 1.0, 0.0)
            eta = 1.0 / 1.5  # Air to glass
            r = refract(incident, normal, eta)
            result[None] = r
            result_len[None] = length(r)

        test_kernel()
        # Straight down should continue straight down (maybe with different magnitude)
        # The formula gives: eta * I + (eta * cos_i - cos_t) * N
        # With incident straight down and normal up, cos_i = 1
        # sin2_t = eta^2 * 0 = 0, cos_t = 1
        # result = eta * (0, -1, 0) + (eta - 1) * (0, 1, 0)
        #        = (0, -eta, 0) + (0, eta-1, 0)
        #        = (0, -1, 0)
        r = result[None]
        assert abs(r[0] - 0.0) < 1e-5
        assert abs(r[1] - (-1.0)) < 1e-5
        assert abs(r[2] - 0.0) < 1e-5

    def test_refract_tir(self):
        """Test total internal reflection returns zero vector."""
        from src.python.core.ray import normalize, refract, vec3

        result = ti.field(dtype=ti.math.vec3, shape=())

        @ti.kernel
        def test_kernel():
            # Ray at steep angle from inside glass
            incident = normalize(vec3(0.9, -0.1, 0.0))
            normal = vec3(0.0, 1.0, 0.0)
            eta = 1.5  # Glass to air (high eta causes TIR at steep angles)
            result[None] = refract(incident, normal, eta)

        test_kernel()
        r = result[None]
        # TIR should give zero vector
        assert abs(r[0]) < 1e-5
        assert abs(r[1]) < 1e-5
        assert abs(r[2]) < 1e-5

    def test_schlick_fresnel(self):
        """Test Schlick's Fresnel approximation."""
        from src.python.core.ray import schlick_fresnel

        result = ti.field(dtype=ti.f32, shape=())

        @ti.kernel
        def test_grazing():
            # At grazing angle (cosine = 0), Fresnel should be 1
            result[None] = schlick_fresnel(0.0, 1.5)

        test_grazing()
        assert abs(result[None] - 1.0) < 1e-5

        @ti.kernel
        def test_normal():
            # At normal incidence (cosine = 1), should be r0
            result[None] = schlick_fresnel(1.0, 1.5)

        test_normal()
        # r0 = ((1-1.5)/(1+1.5))^2 = (-0.5/2.5)^2 = 0.04
        assert abs(result[None] - 0.04) < 1e-5

    def test_near_zero(self):
        """Test near_zero detection."""
        from src.python.core.ray import near_zero, vec3

        result_zero = ti.field(dtype=ti.i32, shape=())
        result_nonzero = ti.field(dtype=ti.i32, shape=())

        @ti.kernel
        def test_kernel():
            v_zero = vec3(1e-9, 1e-9, 1e-9)
            v_nonzero = vec3(0.1, 0.0, 0.0)
            result_zero[None] = near_zero(v_zero)
            result_nonzero[None] = near_zero(v_nonzero)

        test_kernel()
        assert result_zero[None] == 1
        assert result_nonzero[None] == 0


class TestRandomSampling:
    """Tests for random sampling functions."""

    def test_random_in_unit_sphere_bounds(self):
        """Test random_in_unit_sphere returns points inside unit sphere."""
        from src.python.core.ray import length_squared, random_in_unit_sphere

        max_len_sq = ti.field(dtype=ti.f32, shape=())
        max_len_sq[None] = 0.0

        @ti.kernel
        def test_kernel():
            for _ in range(1000):
                p = random_in_unit_sphere()
                len_sq = length_squared(p)
                ti.atomic_max(max_len_sq[None], len_sq)

        test_kernel()
        assert max_len_sq[None] < 1.0

    def test_random_unit_vector_length(self):
        """Test random_unit_vector returns unit length vectors."""
        from src.python.core.ray import length, random_unit_vector

        min_len = ti.field(dtype=ti.f32, shape=())
        max_len = ti.field(dtype=ti.f32, shape=())
        min_len[None] = 10.0
        max_len[None] = 0.0

        @ti.kernel
        def test_kernel():
            for _ in range(1000):
                v = random_unit_vector()
                vec_len = length(v)
                ti.atomic_min(min_len[None], vec_len)
                ti.atomic_max(max_len[None], vec_len)

        test_kernel()
        assert abs(min_len[None] - 1.0) < 0.1
        assert abs(max_len[None] - 1.0) < 0.1

    def test_random_on_hemisphere_orientation(self):
        """Test random_on_hemisphere returns vectors in correct hemisphere."""
        from src.python.core.ray import dot, random_on_hemisphere, vec3

        min_dot = ti.field(dtype=ti.f32, shape=())
        min_dot[None] = 10.0

        @ti.kernel
        def test_kernel():
            normal = vec3(0.0, 1.0, 0.0)
            for _ in range(1000):
                v = random_on_hemisphere(normal)
                d = dot(v, normal)
                ti.atomic_min(min_dot[None], d)

        test_kernel()
        # All vectors should have positive dot product with normal
        assert min_dot[None] >= 0.0

    def test_random_in_unit_disk_bounds(self):
        """Test random_in_unit_disk returns points in xy plane inside unit disk."""
        from src.python.core.ray import random_in_unit_disk

        max_r_sq = ti.field(dtype=ti.f32, shape=())
        max_z = ti.field(dtype=ti.f32, shape=())
        max_r_sq[None] = 0.0
        max_z[None] = 0.0

        @ti.kernel
        def test_kernel():
            for _ in range(1000):
                p = random_in_unit_disk()
                r_sq = p.x * p.x + p.y * p.y
                ti.atomic_max(max_r_sq[None], r_sq)
                ti.atomic_max(max_z[None], ti.abs(p.z))

        test_kernel()
        assert max_r_sq[None] < 1.0
        assert max_z[None] < 1e-6  # z should always be 0

    def test_sample_cosine_hemisphere_pdf(self):
        """Test cosine hemisphere sampling returns valid PDF."""
        from src.python.core.ray import sample_cosine_hemisphere, vec3

        min_pdf = ti.field(dtype=ti.f32, shape=())
        max_pdf = ti.field(dtype=ti.f32, shape=())
        min_pdf[None] = 10.0
        max_pdf[None] = 0.0

        @ti.kernel
        def test_kernel():
            normal = vec3(0.0, 0.0, 1.0)
            for _ in range(1000):
                direction, pdf = sample_cosine_hemisphere(normal)
                ti.atomic_min(min_pdf[None], pdf)
                ti.atomic_max(max_pdf[None], pdf)

        test_kernel()
        # PDF should be between 0 and 1/pi (max when cos_theta = 1)
        assert min_pdf[None] >= 0.0
        assert max_pdf[None] <= 1.0 / 3.14159 + 0.01

    def test_build_onb_orthogonality(self):
        """Test that build_onb creates orthonormal basis."""
        from src.python.core.ray import build_onb_from_normal, dot, length, vec3

        dot_tn = ti.field(dtype=ti.f32, shape=())
        dot_bn = ti.field(dtype=ti.f32, shape=())
        dot_tb = ti.field(dtype=ti.f32, shape=())
        len_t = ti.field(dtype=ti.f32, shape=())
        len_b = ti.field(dtype=ti.f32, shape=())

        @ti.kernel
        def test_kernel():
            normal = vec3(0.5, 0.5, 0.707)
            t, b, n = build_onb_from_normal(normal)
            dot_tn[None] = dot(t, n)
            dot_bn[None] = dot(b, n)
            dot_tb[None] = dot(t, b)
            len_t[None] = length(t)
            len_b[None] = length(b)

        test_kernel()
        # Vectors should be orthogonal (use slightly looser tolerance for floating point)
        assert abs(dot_tn[None]) < 1e-4
        assert abs(dot_bn[None]) < 1e-4
        assert abs(dot_tb[None]) < 1e-4
        # Tangent and bitangent should be unit length
        assert abs(len_t[None] - 1.0) < 1e-4
        assert abs(len_b[None] - 1.0) < 1e-4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
