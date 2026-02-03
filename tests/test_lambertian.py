"""Unit tests for the Lambertian material module.

Tests cover:
- Lambertian BRDF evaluation (albedo / pi)
- PDF calculation for cosine-weighted sampling
- Scatter function (direction, attenuation, pdf)
- Energy conservation (attenuation <= 1)
- Material registry operations
"""

import math

import pytest
import taichi as ti


class TestLambertianBrdf:
    """Tests for Lambertian BRDF evaluation."""

    def test_eval_lambertian_white(self):
        """Test BRDF for white (1,1,1) albedo."""
        from src.python.materials.lambertian import eval_lambertian

        result = ti.Vector.field(3, dtype=ti.f32, shape=())

        @ti.kernel
        def test_kernel():
            albedo = ti.math.vec3(1.0, 1.0, 1.0)
            result[None] = eval_lambertian(albedo)

        test_kernel()
        r = result[None]
        expected = 1.0 / math.pi
        assert abs(r[0] - expected) < 1e-6
        assert abs(r[1] - expected) < 1e-6
        assert abs(r[2] - expected) < 1e-6

    def test_eval_lambertian_colored(self):
        """Test BRDF for colored (0.5, 0.3, 0.1) albedo."""
        from src.python.materials.lambertian import eval_lambertian

        result = ti.Vector.field(3, dtype=ti.f32, shape=())

        @ti.kernel
        def test_kernel():
            albedo = ti.math.vec3(0.5, 0.3, 0.1)
            result[None] = eval_lambertian(albedo)

        test_kernel()
        r = result[None]
        assert abs(r[0] - 0.5 / math.pi) < 1e-6
        assert abs(r[1] - 0.3 / math.pi) < 1e-6
        assert abs(r[2] - 0.1 / math.pi) < 1e-6

    def test_eval_lambertian_black(self):
        """Test BRDF for black (0,0,0) albedo returns zero."""
        from src.python.materials.lambertian import eval_lambertian

        result = ti.Vector.field(3, dtype=ti.f32, shape=())

        @ti.kernel
        def test_kernel():
            albedo = ti.math.vec3(0.0, 0.0, 0.0)
            result[None] = eval_lambertian(albedo)

        test_kernel()
        r = result[None]
        assert abs(r[0]) < 1e-6
        assert abs(r[1]) < 1e-6
        assert abs(r[2]) < 1e-6


class TestLambertianPdf:
    """Tests for PDF calculation."""

    def test_pdf_normal_direction(self):
        """Test PDF when scattered direction equals normal (cos_theta = 1)."""
        from src.python.materials.lambertian import pdf_lambertian

        result = ti.field(dtype=ti.f32, shape=())

        @ti.kernel
        def test_kernel():
            normal = ti.math.vec3(0.0, 1.0, 0.0)
            scattered = ti.math.vec3(0.0, 1.0, 0.0)  # Same as normal
            result[None] = pdf_lambertian(normal, scattered)

        test_kernel()
        # PDF = cos(0) / pi = 1 / pi
        expected = 1.0 / math.pi
        assert abs(result[None] - expected) < 1e-6

    def test_pdf_45_degrees(self):
        """Test PDF at 45 degree angle."""
        from src.python.materials.lambertian import pdf_lambertian

        result = ti.field(dtype=ti.f32, shape=())

        @ti.kernel
        def test_kernel():
            normal = ti.math.vec3(0.0, 1.0, 0.0)
            # 45 degrees: normalize(1, 1, 0)
            inv_sqrt2 = 1.0 / ti.sqrt(2.0)
            scattered = ti.math.vec3(inv_sqrt2, inv_sqrt2, 0.0)
            result[None] = pdf_lambertian(normal, scattered)

        test_kernel()
        # PDF = cos(45 deg) / pi = (1/sqrt(2)) / pi
        expected = (1.0 / math.sqrt(2.0)) / math.pi
        assert abs(result[None] - expected) < 1e-5

    def test_pdf_grazing_angle(self):
        """Test PDF at 90 degrees (grazing angle) is zero."""
        from src.python.materials.lambertian import pdf_lambertian

        result = ti.field(dtype=ti.f32, shape=())

        @ti.kernel
        def test_kernel():
            normal = ti.math.vec3(0.0, 1.0, 0.0)
            scattered = ti.math.vec3(1.0, 0.0, 0.0)  # Perpendicular to normal
            result[None] = pdf_lambertian(normal, scattered)

        test_kernel()
        # PDF = cos(90 deg) / pi = 0
        assert abs(result[None]) < 1e-6

    def test_pdf_below_surface(self):
        """Test PDF returns zero for directions below surface."""
        from src.python.materials.lambertian import pdf_lambertian

        result = ti.field(dtype=ti.f32, shape=())

        @ti.kernel
        def test_kernel():
            normal = ti.math.vec3(0.0, 1.0, 0.0)
            scattered = ti.math.vec3(0.0, -1.0, 0.0)  # Opposite to normal
            result[None] = pdf_lambertian(normal, scattered)

        test_kernel()
        assert result[None] == 0.0


class TestScatterLambertian:
    """Tests for the scatter function."""

    def test_scatter_direction_in_hemisphere(self):
        """Test that scattered direction is always in the correct hemisphere."""
        from src.python.materials.lambertian import scatter_lambertian

        min_dot = ti.field(dtype=ti.f32, shape=())
        min_dot[None] = 10.0

        @ti.kernel
        def test_kernel():
            albedo = ti.math.vec3(0.5, 0.5, 0.5)
            normal = ti.math.vec3(0.0, 1.0, 0.0)
            for i in range(1000):
                direction, attenuation, pdf = scatter_lambertian(albedo, normal)
                d = ti.math.dot(direction, normal)
                ti.atomic_min(min_dot[None], d)

        test_kernel()
        # All scattered directions should be in the hemisphere (positive dot product)
        assert min_dot[None] >= 0.0

    def test_scatter_direction_normalized(self):
        """Test that scattered direction is normalized."""
        from src.python.materials.lambertian import scatter_lambertian

        min_len = ti.field(dtype=ti.f32, shape=())
        max_len = ti.field(dtype=ti.f32, shape=())
        min_len[None] = 10.0
        max_len[None] = 0.0

        @ti.kernel
        def test_kernel():
            albedo = ti.math.vec3(0.5, 0.5, 0.5)
            normal = ti.math.vec3(0.0, 0.0, 1.0)
            for i in range(1000):
                direction, attenuation, pdf = scatter_lambertian(albedo, normal)
                length = ti.math.length(direction)
                ti.atomic_min(min_len[None], length)
                ti.atomic_max(max_len[None], length)

        test_kernel()
        assert abs(min_len[None] - 1.0) < 0.01
        assert abs(max_len[None] - 1.0) < 0.01

    def test_scatter_attenuation_equals_albedo(self):
        """Test that attenuation equals albedo (importance sampling property)."""
        from src.python.materials.lambertian import scatter_lambertian

        result_attenuation = ti.Vector.field(3, dtype=ti.f32, shape=())

        @ti.kernel
        def test_kernel():
            albedo = ti.math.vec3(0.7, 0.3, 0.5)
            normal = ti.math.vec3(0.0, 1.0, 0.0)
            _, attenuation, _ = scatter_lambertian(albedo, normal)
            result_attenuation[None] = attenuation

        test_kernel()
        r = result_attenuation[None]
        # Due to importance sampling, attenuation should exactly equal albedo
        assert abs(r[0] - 0.7) < 1e-6
        assert abs(r[1] - 0.3) < 1e-6
        assert abs(r[2] - 0.5) < 1e-6

    def test_scatter_pdf_positive(self):
        """Test that PDF is always positive for valid samples."""
        from src.python.materials.lambertian import scatter_lambertian

        min_pdf = ti.field(dtype=ti.f32, shape=())
        min_pdf[None] = 10.0

        @ti.kernel
        def test_kernel():
            albedo = ti.math.vec3(0.5, 0.5, 0.5)
            normal = ti.math.vec3(0.0, 1.0, 0.0)
            for i in range(1000):
                direction, attenuation, pdf = scatter_lambertian(albedo, normal)
                ti.atomic_min(min_pdf[None], pdf)

        test_kernel()
        assert min_pdf[None] > 0.0

    def test_scatter_pdf_bounded(self):
        """Test that PDF is bounded by 1/pi (maximum when cos_theta = 1)."""
        from src.python.materials.lambertian import scatter_lambertian

        max_pdf = ti.field(dtype=ti.f32, shape=())
        max_pdf[None] = 0.0

        @ti.kernel
        def test_kernel():
            albedo = ti.math.vec3(0.5, 0.5, 0.5)
            normal = ti.math.vec3(0.0, 1.0, 0.0)
            for i in range(1000):
                direction, attenuation, pdf = scatter_lambertian(albedo, normal)
                ti.atomic_max(max_pdf[None], pdf)

        test_kernel()
        # Maximum PDF is 1/pi when cos_theta = 1
        assert max_pdf[None] <= 1.0 / math.pi + 0.01


class TestEnergyConservation:
    """Tests for energy conservation properties."""

    def test_attenuation_bounded_by_one(self):
        """Test that attenuation never exceeds 1 (energy conservation)."""
        from src.python.materials.lambertian import scatter_lambertian

        max_component = ti.field(dtype=ti.f32, shape=())
        max_component[None] = 0.0

        @ti.kernel
        def test_kernel():
            # Test with white albedo (worst case for energy)
            albedo = ti.math.vec3(1.0, 1.0, 1.0)
            normal = ti.math.vec3(0.0, 1.0, 0.0)
            for i in range(1000):
                direction, attenuation, pdf = scatter_lambertian(albedo, normal)
                ti.atomic_max(max_component[None], attenuation.x)
                ti.atomic_max(max_component[None], attenuation.y)
                ti.atomic_max(max_component[None], attenuation.z)

        test_kernel()
        assert max_component[None] <= 1.0 + 1e-6

    def test_brdf_times_cosine_bounded(self):
        """Test that BRDF * cos_theta <= 1 for energy conservation."""
        from src.python.materials.lambertian import eval_lambertian

        max_reflectance = ti.field(dtype=ti.f32, shape=())
        max_reflectance[None] = 0.0

        @ti.kernel
        def test_kernel():
            # White albedo gives maximum BRDF
            albedo = ti.math.vec3(1.0, 1.0, 1.0)
            brdf = eval_lambertian(albedo)
            # Maximum cos_theta is 1
            max_cos_theta = 1.0
            max_reflectance[None] = brdf.x * max_cos_theta

        test_kernel()
        # BRDF * cos_theta should be <= 1 for energy conservation
        # For Lambertian: (1/pi) * 1 = 1/pi < 1
        assert max_reflectance[None] <= 1.0


class TestMaterialRegistry:
    """Tests for material registry operations."""

    def test_add_and_get_material(self):
        """Test adding a material and retrieving its albedo."""
        from src.python.materials.lambertian import (
            add_lambertian_material,
            clear_lambertian_materials,
            get_lambertian_albedo,
        )

        clear_lambertian_materials()
        idx = add_lambertian_material((0.8, 0.2, 0.4))

        result = ti.Vector.field(3, dtype=ti.f32, shape=())

        @ti.kernel
        def test_kernel(mat_idx: ti.i32):
            result[None] = get_lambertian_albedo(mat_idx)

        test_kernel(idx)
        r = result[None]
        assert abs(r[0] - 0.8) < 1e-6
        assert abs(r[1] - 0.2) < 1e-6
        assert abs(r[2] - 0.4) < 1e-6

    def test_material_count(self):
        """Test that material count is tracked correctly."""
        from src.python.materials.lambertian import (
            add_lambertian_material,
            clear_lambertian_materials,
            get_lambertian_material_count,
        )

        clear_lambertian_materials()
        assert get_lambertian_material_count() == 0

        add_lambertian_material((0.5, 0.5, 0.5))
        assert get_lambertian_material_count() == 1

        add_lambertian_material((0.3, 0.3, 0.3))
        assert get_lambertian_material_count() == 2

    def test_multiple_materials(self):
        """Test adding multiple materials with different albedos."""
        from src.python.materials.lambertian import (
            add_lambertian_material,
            clear_lambertian_materials,
            get_lambertian_albedo,
        )

        clear_lambertian_materials()
        idx0 = add_lambertian_material((1.0, 0.0, 0.0))  # Red
        idx1 = add_lambertian_material((0.0, 1.0, 0.0))  # Green
        idx2 = add_lambertian_material((0.0, 0.0, 1.0))  # Blue

        result = ti.Vector.field(3, dtype=ti.f32, shape=())

        @ti.kernel
        def test_kernel(mat_idx: ti.i32):
            result[None] = get_lambertian_albedo(mat_idx)

        test_kernel(idx0)
        assert abs(result[None][0] - 1.0) < 1e-6
        assert abs(result[None][1] - 0.0) < 1e-6
        assert abs(result[None][2] - 0.0) < 1e-6

        test_kernel(idx1)
        assert abs(result[None][0] - 0.0) < 1e-6
        assert abs(result[None][1] - 1.0) < 1e-6
        assert abs(result[None][2] - 0.0) < 1e-6

        test_kernel(idx2)
        assert abs(result[None][0] - 0.0) < 1e-6
        assert abs(result[None][1] - 0.0) < 1e-6
        assert abs(result[None][2] - 1.0) < 1e-6

    def test_albedo_validation_negative(self):
        """Test that negative albedo values are rejected."""
        from src.python.materials.lambertian import (
            add_lambertian_material,
            clear_lambertian_materials,
        )

        clear_lambertian_materials()
        with pytest.raises(ValueError, match="outside"):
            add_lambertian_material((-0.1, 0.5, 0.5))

    def test_albedo_validation_greater_than_one(self):
        """Test that albedo values > 1 are rejected."""
        from src.python.materials.lambertian import (
            add_lambertian_material,
            clear_lambertian_materials,
        )

        clear_lambertian_materials()
        with pytest.raises(ValueError, match="outside"):
            add_lambertian_material((0.5, 1.1, 0.5))

    def test_scatter_by_id(self):
        """Test scattering using material index."""
        from src.python.materials.lambertian import (
            add_lambertian_material,
            clear_lambertian_materials,
            scatter_lambertian_by_id,
        )

        clear_lambertian_materials()
        idx = add_lambertian_material((0.6, 0.4, 0.2))

        result_attenuation = ti.Vector.field(3, dtype=ti.f32, shape=())

        @ti.kernel
        def test_kernel(mat_idx: ti.i32):
            normal = ti.math.vec3(0.0, 1.0, 0.0)
            _, attenuation, _ = scatter_lambertian_by_id(mat_idx, normal)
            result_attenuation[None] = attenuation

        test_kernel(idx)
        r = result_attenuation[None]
        assert abs(r[0] - 0.6) < 1e-6
        assert abs(r[1] - 0.4) < 1e-6
        assert abs(r[2] - 0.2) < 1e-6


class TestLambertianMaterialDataclass:
    """Tests for the LambertianMaterial dataclass."""

    def test_dataclass_creation(self):
        """Test creating a LambertianMaterial dataclass."""
        from src.python.materials.lambertian import LambertianMaterial

        result = ti.Vector.field(3, dtype=ti.f32, shape=())

        @ti.kernel
        def test_kernel():
            mat = LambertianMaterial(albedo=ti.math.vec3(0.5, 0.6, 0.7))
            result[None] = mat.albedo

        test_kernel()
        r = result[None]
        assert abs(r[0] - 0.5) < 1e-6
        assert abs(r[1] - 0.6) < 1e-6
        assert abs(r[2] - 0.7) < 1e-6


class TestScatterFullVariant:
    """Tests for scatter_lambertian_full function."""

    def test_full_returns_brdf(self):
        """Test that scatter_lambertian_full returns correct BRDF value."""
        from src.python.materials.lambertian import scatter_lambertian_full

        result_brdf = ti.Vector.field(3, dtype=ti.f32, shape=())

        @ti.kernel
        def test_kernel():
            albedo = ti.math.vec3(0.8, 0.4, 0.2)
            normal = ti.math.vec3(0.0, 1.0, 0.0)
            _, _, _, brdf = scatter_lambertian_full(albedo, normal)
            result_brdf[None] = brdf

        test_kernel()
        r = result_brdf[None]
        assert abs(r[0] - 0.8 / math.pi) < 1e-6
        assert abs(r[1] - 0.4 / math.pi) < 1e-6
        assert abs(r[2] - 0.2 / math.pi) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
