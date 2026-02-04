"""Tests for the phosphorescent material module."""

import math

import pytest
import taichi as ti
import taichi.math as tm

# Note: Taichi is initialized by conftest.py
# We import the phosphorescent module in a fixture to ensure Taichi is ready

vec3 = tm.vec3


# =============================================================================
# Fixtures for module imports (after Taichi initialization)
# =============================================================================


@pytest.fixture
def phosphorescent():
    """Import and return the phosphorescent module."""
    from src.python.materials import phosphorescent

    return phosphorescent


# =============================================================================
# Test Classes
# =============================================================================


class TestPhosphorescentMaterialDataclass:
    """Tests for PhosphorescentMaterial dataclass."""

    def test_create_material(self, phosphorescent) -> None:
        """Test creating a PhosphorescentMaterial instance."""
        # Note: Taichi dataclasses work differently - they are used in kernels
        # This test just ensures the class exists
        assert phosphorescent.PhosphorescentMaterial is not None


class TestMaterialRegistry:
    """Tests for material registry functions."""

    def test_clear_materials(self, phosphorescent) -> None:
        """Test clearing materials resets count to zero."""
        phosphorescent.clear_phosphorescent_materials()
        assert phosphorescent.get_phosphorescent_material_count() == 0

    def test_add_material(self, phosphorescent) -> None:
        """Test adding a material returns valid index."""
        phosphorescent.clear_phosphorescent_materials()

        idx = phosphorescent.add_phosphorescent_material(
            albedo=(0.2, 0.5, 0.2),
            glow_color=(0.0, 1.0, 0.3),
            glow_intensity=2.0,
        )

        assert idx == 0
        assert phosphorescent.get_phosphorescent_material_count() == 1

    def test_add_multiple_materials(self, phosphorescent) -> None:
        """Test adding multiple materials increments index."""
        phosphorescent.clear_phosphorescent_materials()

        idx1 = phosphorescent.add_phosphorescent_material(
            albedo=(0.2, 0.5, 0.2),
            glow_color=(0.0, 1.0, 0.3),
            glow_intensity=2.0,
        )
        idx2 = phosphorescent.add_phosphorescent_material(
            albedo=(0.8, 0.1, 0.1),
            glow_color=(1.0, 0.0, 0.0),
            glow_intensity=5.0,
        )

        assert idx1 == 0
        assert idx2 == 1
        assert phosphorescent.get_phosphorescent_material_count() == 2

    def test_albedo_validation(self, phosphorescent) -> None:
        """Test albedo validation rejects out-of-range values."""
        phosphorescent.clear_phosphorescent_materials()

        # Albedo too high
        with pytest.raises(ValueError, match="outside.*0, 1"):
            phosphorescent.add_phosphorescent_material(
                albedo=(1.5, 0.5, 0.5),
                glow_color=(1.0, 1.0, 1.0),
                glow_intensity=1.0,
            )

        # Albedo negative
        with pytest.raises(ValueError, match="outside.*0, 1"):
            phosphorescent.add_phosphorescent_material(
                albedo=(-0.1, 0.5, 0.5),
                glow_color=(1.0, 1.0, 1.0),
                glow_intensity=1.0,
            )

    def test_glow_color_validation(self, phosphorescent) -> None:
        """Test glow color validation rejects negative values."""
        phosphorescent.clear_phosphorescent_materials()

        with pytest.raises(ValueError, match="negative"):
            phosphorescent.add_phosphorescent_material(
                albedo=(0.5, 0.5, 0.5),
                glow_color=(-0.1, 1.0, 1.0),
                glow_intensity=1.0,
            )

    def test_glow_intensity_validation(self, phosphorescent) -> None:
        """Test glow intensity validation rejects negative values."""
        phosphorescent.clear_phosphorescent_materials()

        with pytest.raises(ValueError, match="negative"):
            phosphorescent.add_phosphorescent_material(
                albedo=(0.5, 0.5, 0.5),
                glow_color=(1.0, 1.0, 1.0),
                glow_intensity=-1.0,
            )


class TestScatterDirectionHemisphere:
    """Tests verifying scatter directions are always in the correct hemisphere."""

    def test_scatter_direction_always_in_hemisphere(self, phosphorescent) -> None:
        """Test that all scattered directions have positive dot product with normal."""
        phosphorescent.clear_phosphorescent_materials()
        phosphorescent.add_phosphorescent_material(
            albedo=(0.5, 0.5, 0.5),
            glow_color=(0.0, 1.0, 0.0),
            glow_intensity=1.0,
        )

        num_samples = 1000
        min_dot = ti.field(dtype=ti.f32, shape=())
        min_dot[None] = 10.0

        @ti.kernel
        def test_kernel():
            normal = vec3(0.0, 1.0, 0.0)
            for i in range(num_samples):
                direction, _, _ = phosphorescent.scatter_phosphorescent_by_id(0, normal)
                dot = tm.dot(direction, normal)
                ti.atomic_min(min_dot[None], dot)

        test_kernel()
        # All scattered directions should be in the hemisphere (positive dot product)
        assert min_dot[None] >= 0.0, f"Found direction below surface with dot={min_dot[None]}"

    def test_scatter_direction_normalized(self, phosphorescent) -> None:
        """Test that scattered direction is always normalized."""
        phosphorescent.clear_phosphorescent_materials()
        phosphorescent.add_phosphorescent_material(
            albedo=(0.5, 0.5, 0.5),
            glow_color=(0.0, 1.0, 0.0),
            glow_intensity=1.0,
        )

        num_samples = 500
        min_len = ti.field(dtype=ti.f32, shape=())
        max_len = ti.field(dtype=ti.f32, shape=())
        min_len[None] = 10.0
        max_len[None] = 0.0

        @ti.kernel
        def test_kernel():
            normal = vec3(0.0, 0.0, 1.0)
            for i in range(num_samples):
                direction, _, _ = phosphorescent.scatter_phosphorescent_by_id(0, normal)
                length = tm.length(direction)
                ti.atomic_min(min_len[None], length)
                ti.atomic_max(max_len[None], length)

        test_kernel()
        assert abs(min_len[None] - 1.0) < 0.01, f"Direction not normalized: min_len={min_len[None]}"
        assert abs(max_len[None] - 1.0) < 0.01, f"Direction not normalized: max_len={max_len[None]}"

    def test_scatter_direction_with_tilted_normal(self, phosphorescent) -> None:
        """Test scatter works correctly with non-axis-aligned normal."""
        phosphorescent.clear_phosphorescent_materials()
        phosphorescent.add_phosphorescent_material(
            albedo=(0.3, 0.7, 0.3),
            glow_color=(0.0, 1.0, 0.3),
            glow_intensity=1.0,
        )

        num_samples = 500
        min_dot = ti.field(dtype=ti.f32, shape=())
        min_dot[None] = 10.0

        @ti.kernel
        def test_kernel():
            # Tilted normal
            normal = tm.normalize(vec3(1.0, 1.0, 1.0))
            for i in range(num_samples):
                direction, _, _ = phosphorescent.scatter_phosphorescent_by_id(0, normal)
                dot = tm.dot(direction, normal)
                ti.atomic_min(min_dot[None], dot)

        test_kernel()
        assert min_dot[None] >= 0.0, f"Direction below tilted surface with dot={min_dot[None]}"


class TestGlowIntensityBounds:
    """Tests for glow intensity behavior and bounds."""

    def test_zero_glow_intensity_produces_zero_emission(self, phosphorescent) -> None:
        """Test that zero glow intensity produces no emission."""
        phosphorescent.clear_phosphorescent_materials()
        phosphorescent.add_phosphorescent_material(
            albedo=(0.5, 0.5, 0.5),
            glow_color=(1.0, 1.0, 1.0),
            glow_intensity=0.0,
        )

        result_emission = ti.Vector.field(3, dtype=ti.f32, shape=())

        @ti.kernel
        def test_kernel():
            result_emission[None] = phosphorescent.get_phosphorescent_emission_by_id(0)

        test_kernel()
        emission = result_emission[None]

        assert abs(emission[0]) < 1e-6
        assert abs(emission[1]) < 1e-6
        assert abs(emission[2]) < 1e-6

    def test_high_glow_intensity_scales_correctly(self, phosphorescent) -> None:
        """Test that high glow intensity scales emission linearly."""
        phosphorescent.clear_phosphorescent_materials()
        phosphorescent.add_phosphorescent_material(
            albedo=(0.5, 0.5, 0.5),
            glow_color=(1.0, 0.5, 0.25),
            glow_intensity=10.0,
        )

        result_emission = ti.Vector.field(3, dtype=ti.f32, shape=())

        @ti.kernel
        def test_kernel():
            result_emission[None] = phosphorescent.get_phosphorescent_emission_by_id(0)

        test_kernel()
        emission = result_emission[None]

        # Expected: (1.0, 0.5, 0.25) * 10.0 = (10.0, 5.0, 2.5)
        assert abs(emission[0] - 10.0) < 1e-5
        assert abs(emission[1] - 5.0) < 1e-5
        assert abs(emission[2] - 2.5) < 1e-5

    def test_glow_intensity_validation_boundary(self, phosphorescent) -> None:
        """Test that glow intensity at boundary (0.0) is valid."""
        phosphorescent.clear_phosphorescent_materials()

        # Should not raise an error
        idx = phosphorescent.add_phosphorescent_material(
            albedo=(0.5, 0.5, 0.5),
            glow_color=(1.0, 1.0, 1.0),
            glow_intensity=0.0,
        )
        assert idx == 0

    def test_emission_components_non_negative(self, phosphorescent) -> None:
        """Test that emission components are always non-negative."""
        phosphorescent.clear_phosphorescent_materials()
        phosphorescent.add_phosphorescent_material(
            albedo=(0.5, 0.5, 0.5),
            glow_color=(0.5, 0.8, 0.3),
            glow_intensity=5.0,
        )

        result_emission = ti.Vector.field(3, dtype=ti.f32, shape=())

        @ti.kernel
        def test_kernel():
            result_emission[None] = phosphorescent.get_phosphorescent_emission_by_id(0)

        test_kernel()
        emission = result_emission[None]

        assert emission[0] >= 0.0, f"Emission R is negative: {emission[0]}"
        assert emission[1] >= 0.0, f"Emission G is negative: {emission[1]}"
        assert emission[2] >= 0.0, f"Emission B is negative: {emission[2]}"


class TestEnergyConservation:
    """Tests for energy conservation properties of phosphorescent material."""

    def test_attenuation_bounded_by_one(self, phosphorescent) -> None:
        """Test that attenuation never exceeds 1 (energy conservation for scattering)."""
        phosphorescent.clear_phosphorescent_materials()
        # Use maximum valid albedo
        phosphorescent.add_phosphorescent_material(
            albedo=(1.0, 1.0, 1.0),
            glow_color=(1.0, 1.0, 1.0),
            glow_intensity=10.0,
        )

        max_component = ti.field(dtype=ti.f32, shape=())
        max_component[None] = 0.0

        @ti.kernel
        def test_kernel():
            normal = vec3(0.0, 1.0, 0.0)
            for i in range(1000):
                _, attenuation, _ = phosphorescent.scatter_phosphorescent_by_id(0, normal)
                ti.atomic_max(max_component[None], attenuation.x)
                ti.atomic_max(max_component[None], attenuation.y)
                ti.atomic_max(max_component[None], attenuation.z)

        test_kernel()
        assert max_component[None] <= 1.0 + 1e-6, f"Attenuation exceeds 1: {max_component[None]}"

    def test_brdf_times_cosine_bounded(self, phosphorescent) -> None:
        """Test that BRDF * cos_theta <= 1 for energy conservation."""
        result_brdf = ti.Vector.field(3, dtype=ti.f32, shape=())

        @ti.kernel
        def test_kernel():
            # White albedo gives maximum BRDF
            albedo = vec3(1.0, 1.0, 1.0)
            brdf = phosphorescent.eval_phosphorescent(albedo)
            result_brdf[None] = brdf

        test_kernel()
        brdf = result_brdf[None]

        # Maximum cos_theta is 1, so BRDF * cos_theta = BRDF
        # For Lambertian: (1/pi) * 1 = 1/pi < 1
        assert brdf[0] <= 1.0, f"BRDF exceeds 1: {brdf[0]}"

    def test_pdf_bounded(self, phosphorescent) -> None:
        """Test that PDF is bounded by 1/pi (maximum when cos_theta = 1)."""
        phosphorescent.clear_phosphorescent_materials()
        phosphorescent.add_phosphorescent_material(
            albedo=(0.5, 0.5, 0.5),
            glow_color=(0.0, 1.0, 0.0),
            glow_intensity=1.0,
        )

        max_pdf = ti.field(dtype=ti.f32, shape=())
        max_pdf[None] = 0.0

        @ti.kernel
        def test_kernel():
            normal = vec3(0.0, 1.0, 0.0)
            for i in range(1000):
                _, _, pdf = phosphorescent.scatter_phosphorescent_by_id(0, normal)
                ti.atomic_max(max_pdf[None], pdf)

        test_kernel()
        # Maximum PDF is 1/pi when cos_theta = 1
        assert max_pdf[None] <= 1.0 / math.pi + 0.01, f"PDF exceeds 1/pi: {max_pdf[None]}"

    def test_pdf_positive(self, phosphorescent) -> None:
        """Test that PDF is always positive for valid samples."""
        phosphorescent.clear_phosphorescent_materials()
        phosphorescent.add_phosphorescent_material(
            albedo=(0.5, 0.5, 0.5),
            glow_color=(0.0, 1.0, 0.0),
            glow_intensity=1.0,
        )

        min_pdf = ti.field(dtype=ti.f32, shape=())
        min_pdf[None] = 10.0

        @ti.kernel
        def test_kernel():
            normal = vec3(0.0, 1.0, 0.0)
            for i in range(1000):
                _, _, pdf = phosphorescent.scatter_phosphorescent_by_id(0, normal)
                ti.atomic_min(min_pdf[None], pdf)

        test_kernel()
        assert min_pdf[None] > 0.0, f"PDF is non-positive: {min_pdf[None]}"


class TestTaichiFunctions:
    """Tests for Taichi kernel functions."""

    def test_emission_by_id(self, phosphorescent) -> None:
        """Test get_phosphorescent_emission_by_id returns correct emission."""
        phosphorescent.clear_phosphorescent_materials()
        phosphorescent.add_phosphorescent_material(
            albedo=(0.2, 0.5, 0.2),
            glow_color=(0.0, 1.0, 0.3),
            glow_intensity=2.0,
        )

        result_emission = ti.Vector.field(3, dtype=ti.f32, shape=())

        @ti.kernel
        def test_kernel():
            result_emission[None] = phosphorescent.get_phosphorescent_emission_by_id(0)

        test_kernel()
        emission = result_emission[None]

        # Expected: glow_color * glow_intensity = (0, 2.0, 0.6)
        assert abs(emission[0] - 0.0) < 1e-6
        assert abs(emission[1] - 2.0) < 1e-6
        assert abs(emission[2] - 0.6) < 1e-6

    def test_scatter_by_id(self, phosphorescent) -> None:
        """Test scatter_phosphorescent_by_id scatters correctly."""
        phosphorescent.clear_phosphorescent_materials()
        phosphorescent.add_phosphorescent_material(
            albedo=(0.2, 0.5, 0.2),
            glow_color=(0.0, 1.0, 0.3),
            glow_intensity=2.0,
        )

        result_direction = ti.Vector.field(3, dtype=ti.f32, shape=())
        result_attenuation = ti.Vector.field(3, dtype=ti.f32, shape=())
        result_pdf = ti.field(dtype=ti.f32, shape=())

        @ti.kernel
        def test_kernel():
            normal = vec3(0.0, 1.0, 0.0)
            direction, attenuation, pdf = phosphorescent.scatter_phosphorescent_by_id(0, normal)
            result_direction[None] = direction
            result_attenuation[None] = attenuation
            result_pdf[None] = pdf

        test_kernel()
        direction = result_direction[None]
        attenuation = result_attenuation[None]
        pdf = result_pdf[None]

        # Direction should be in upper hemisphere (normal is +Y)
        assert direction[1] >= 0.0, f"Direction should be in upper hemisphere, got y={direction[1]}"

        # Attenuation should equal albedo
        assert abs(attenuation[0] - 0.2) < 1e-6
        assert abs(attenuation[1] - 0.5) < 1e-6
        assert abs(attenuation[2] - 0.2) < 1e-6

        # PDF should be positive
        assert pdf > 0.0, f"PDF should be positive, got {pdf}"

    def test_property_getters(self, phosphorescent) -> None:
        """Test individual property getters return correct values."""
        phosphorescent.clear_phosphorescent_materials()
        phosphorescent.add_phosphorescent_material(
            albedo=(0.2, 0.5, 0.2),
            glow_color=(0.0, 1.0, 0.3),
            glow_intensity=2.0,
        )

        result_emission = ti.Vector.field(3, dtype=ti.f32, shape=())
        result_albedo = ti.Vector.field(3, dtype=ti.f32, shape=())

        @ti.kernel
        def test_kernel():
            albedo = phosphorescent.get_phosphorescent_albedo(0)
            glow_color = phosphorescent.get_phosphorescent_glow_color(0)
            glow_intensity = phosphorescent.get_phosphorescent_glow_intensity(0)
            result_emission[None] = glow_color * glow_intensity
            result_albedo[None] = albedo

        test_kernel()
        emission = result_emission[None]
        albedo = result_albedo[None]

        # Check emission (glow_color * glow_intensity)
        assert abs(emission[0] - 0.0) < 1e-6
        assert abs(emission[1] - 2.0) < 1e-6
        assert abs(emission[2] - 0.6) < 1e-6

        # Check albedo
        assert abs(albedo[0] - 0.2) < 1e-6
        assert abs(albedo[1] - 0.5) < 1e-6
        assert abs(albedo[2] - 0.2) < 1e-6

    def test_eval_and_pdf(self, phosphorescent) -> None:
        """Test eval_phosphorescent and pdf_phosphorescent."""
        result_brdf = ti.Vector.field(3, dtype=ti.f32, shape=())
        result_pdf = ti.field(dtype=ti.f32, shape=())

        @ti.kernel
        def test_kernel():
            albedo = vec3(0.5, 0.5, 0.5)
            normal = vec3(0.0, 1.0, 0.0)
            scattered = vec3(0.0, 1.0, 0.0)  # Same as normal

            brdf = phosphorescent.eval_phosphorescent(albedo)
            pdf = phosphorescent.pdf_phosphorescent(normal, scattered)

            result_brdf[None] = brdf
            result_pdf[None] = pdf

        test_kernel()
        brdf = result_brdf[None]
        pdf = result_pdf[None]

        # BRDF = albedo / pi = (0.5, 0.5, 0.5) / pi
        expected_brdf = 0.5 / math.pi
        assert abs(brdf[0] - expected_brdf) < 1e-6
        assert abs(brdf[1] - expected_brdf) < 1e-6
        assert abs(brdf[2] - expected_brdf) < 1e-6

        # PDF = cos(theta) / pi = 1.0 / pi (since direction == normal)
        expected_pdf = 1.0 / math.pi
        assert abs(pdf - expected_pdf) < 1e-6

    def test_scatter_direct(self, phosphorescent) -> None:
        """Test scatter_phosphorescent directly with given albedo."""
        result_direction = ti.Vector.field(3, dtype=ti.f32, shape=())
        result_attenuation = ti.Vector.field(3, dtype=ti.f32, shape=())
        result_pdf = ti.field(dtype=ti.f32, shape=())

        @ti.kernel
        def test_kernel():
            albedo = vec3(0.3, 0.6, 0.3)
            normal = vec3(0.0, 1.0, 0.0)

            direction, attenuation, pdf = phosphorescent.scatter_phosphorescent(albedo, normal)
            result_direction[None] = direction
            result_attenuation[None] = attenuation
            result_pdf[None] = pdf

        test_kernel()
        direction = result_direction[None]
        attenuation = result_attenuation[None]
        pdf = result_pdf[None]

        # Direction should be normalized
        length_sq = direction[0] ** 2 + direction[1] ** 2 + direction[2] ** 2
        assert abs(length_sq - 1.0) < 1e-5, f"Direction not normalized: length_sq={length_sq}"

        # Direction should be in upper hemisphere
        assert direction[1] >= 0.0

        # Attenuation should equal provided albedo
        assert abs(attenuation[0] - 0.3) < 1e-6
        assert abs(attenuation[1] - 0.6) < 1e-6
        assert abs(attenuation[2] - 0.3) < 1e-6

        # PDF should be positive
        assert pdf > 0.0

    def test_get_emission_direct(self, phosphorescent) -> None:
        """Test get_phosphorescent_emission directly."""
        result_emission = ti.Vector.field(3, dtype=ti.f32, shape=())

        @ti.kernel
        def test_kernel():
            glow_color = vec3(0.0, 1.0, 0.5)
            glow_intensity: ti.f32 = 3.0

            emission = phosphorescent.get_phosphorescent_emission(glow_color, glow_intensity)
            result_emission[None] = emission

        test_kernel()
        emission = result_emission[None]

        # Expected: (0.0, 1.0, 0.5) * 3.0 = (0.0, 3.0, 1.5)
        assert abs(emission[0] - 0.0) < 1e-6
        assert abs(emission[1] - 3.0) < 1e-6
        assert abs(emission[2] - 1.5) < 1e-6

    def test_scatter_full(self, phosphorescent) -> None:
        """Test scatter_phosphorescent_full returns all components."""
        result_direction = ti.Vector.field(3, dtype=ti.f32, shape=())
        result_attenuation = ti.Vector.field(3, dtype=ti.f32, shape=())
        result_pdf = ti.field(dtype=ti.f32, shape=())
        result_brdf = ti.Vector.field(3, dtype=ti.f32, shape=())
        result_emission = ti.Vector.field(3, dtype=ti.f32, shape=())

        @ti.kernel
        def test_kernel():
            albedo = vec3(0.4, 0.4, 0.4)
            glow_color = vec3(1.0, 0.5, 0.0)
            glow_intensity: ti.f32 = 2.0
            normal = vec3(0.0, 1.0, 0.0)

            direction, attenuation, pdf, brdf, emission = (
                phosphorescent.scatter_phosphorescent_full(
                    albedo, glow_color, glow_intensity, normal
                )
            )
            result_direction[None] = direction
            result_attenuation[None] = attenuation
            result_pdf[None] = pdf
            result_brdf[None] = brdf
            result_emission[None] = emission

        test_kernel()
        direction = result_direction[None]
        attenuation = result_attenuation[None]
        pdf = result_pdf[None]
        brdf = result_brdf[None]
        emission = result_emission[None]

        # Direction should be in upper hemisphere
        assert direction[1] >= 0.0

        # Attenuation should equal albedo
        assert abs(attenuation[0] - 0.4) < 1e-6
        assert abs(attenuation[1] - 0.4) < 1e-6
        assert abs(attenuation[2] - 0.4) < 1e-6

        # PDF should be positive
        assert pdf > 0.0

        # BRDF = albedo / pi
        expected_brdf = 0.4 / math.pi
        assert abs(brdf[0] - expected_brdf) < 1e-6
        assert abs(brdf[1] - expected_brdf) < 1e-6
        assert abs(brdf[2] - expected_brdf) < 1e-6

        # Emission = glow_color * glow_intensity = (1.0, 0.5, 0.0) * 2.0
        assert abs(emission[0] - 2.0) < 1e-6
        assert abs(emission[1] - 1.0) < 1e-6
        assert abs(emission[2] - 0.0) < 1e-6


class TestSceneManagerIntegration:
    """Tests for phosphorescent material integration with SceneManager."""

    def test_scene_manager_phosphorescent_material(self) -> None:
        """Test adding phosphorescent material through SceneManager."""
        from src.python.scene.manager import MaterialType, SceneManager

        scene = SceneManager()
        mat_id = scene.add_phosphorescent_material(
            albedo=(0.2, 0.5, 0.2),
            glow_color=(0.0, 1.0, 0.3),
            glow_intensity=2.0,
        )

        assert mat_id == 0
        assert scene.get_material_count() == 1
        assert scene.get_material_type_python(mat_id) == MaterialType.PHOSPHORESCENT

    def test_scene_manager_phosphorescent_sphere(self) -> None:
        """Test adding phosphorescent sphere through SceneManager convenience method."""
        from src.python.scene.manager import MaterialType, SceneManager

        scene = SceneManager()
        sphere_idx, mat_id = scene.add_phosphorescent_sphere(
            center=(0.0, 0.0, -2.0),
            radius=0.5,
            albedo=(0.2, 0.5, 0.2),
            glow_color=(0.0, 1.0, 0.3),
            glow_intensity=2.0,
        )

        assert sphere_idx == 0
        assert mat_id == 0
        assert scene.get_sphere_count() == 1
        assert scene.get_material_type_python(mat_id) == MaterialType.PHOSPHORESCENT

    def test_scene_manager_material_info(self) -> None:
        """Test that material info is correctly stored for phosphorescent."""
        from src.python.scene.manager import MaterialType, SceneManager

        scene = SceneManager()
        mat_id = scene.add_phosphorescent_material(
            albedo=(0.3, 0.6, 0.3),
            glow_color=(0.0, 1.0, 0.5),
            glow_intensity=3.0,
        )

        info = scene.get_material_info(mat_id)
        assert info is not None
        assert info.material_type == MaterialType.PHOSPHORESCENT
        assert info.params["albedo"] == (0.3, 0.6, 0.3)
        assert info.params["glow_color"] == (0.0, 1.0, 0.5)
        assert info.params["glow_intensity"] == 3.0

    def test_scene_manager_gpu_dispatch(self) -> None:
        """Test GPU-side material type lookup for phosphorescent."""
        from src.python.scene.manager import (
            MaterialType,
            SceneManager,
            get_material_type,
            get_material_type_index,
        )

        scene = SceneManager()
        mat_id = scene.add_phosphorescent_material(
            albedo=(0.2, 0.5, 0.2),
            glow_color=(0.0, 1.0, 0.3),
            glow_intensity=2.0,
        )

        result_type = ti.field(dtype=ti.i32, shape=())
        result_idx = ti.field(dtype=ti.i32, shape=())

        @ti.kernel
        def test_kernel():
            result_type[None] = get_material_type(0)
            result_idx[None] = get_material_type_index(0)

        test_kernel()

        assert result_type[None] == int(MaterialType.PHOSPHORESCENT)
        assert result_idx[None] == 0  # First phosphorescent material


class TestIntegratorDispatch:
    """Tests for phosphorescent dispatch in integrator."""

    def test_integrator_scatter_dispatch(self) -> None:
        """Test that integrator correctly dispatches to phosphorescent scatter."""
        from src.python.materials.phosphorescent import scatter_phosphorescent_by_id
        from src.python.scene.manager import (
            MaterialType,
            SceneManager,
            get_material_type,
            get_material_type_index,
        )

        scene = SceneManager()
        scene.add_phosphorescent_material(
            albedo=(0.2, 0.5, 0.2),
            glow_color=(0.0, 1.0, 0.3),
            glow_intensity=2.0,
        )

        scattered_valid = ti.field(dtype=ti.i32, shape=())
        attenuation_result = ti.Vector.field(3, dtype=ti.f32, shape=())

        @ti.kernel
        def test_dispatch():
            normal = vec3(0.0, 1.0, 0.0)

            mat_id = 0
            mat_type = get_material_type(mat_id)
            type_idx = get_material_type_index(mat_id)

            if mat_type == int(MaterialType.PHOSPHORESCENT):
                direction, attenuation, pdf = scatter_phosphorescent_by_id(type_idx, normal)
                scattered_valid[None] = 1 if tm.length(direction) > 0.0 else 0
                attenuation_result[None] = attenuation

        test_dispatch()

        assert scattered_valid[None] == 1
        # Attenuation should equal albedo
        assert abs(attenuation_result[None][0] - 0.2) < 1e-6
        assert abs(attenuation_result[None][1] - 0.5) < 1e-6
        assert abs(attenuation_result[None][2] - 0.2) < 1e-6

    def test_integrator_emission_dispatch(self) -> None:
        """Test that integrator correctly retrieves phosphorescent emission."""
        from src.python.materials.phosphorescent import get_phosphorescent_emission_by_id
        from src.python.scene.manager import (
            MaterialType,
            SceneManager,
            get_material_type,
            get_material_type_index,
        )

        scene = SceneManager()
        scene.add_phosphorescent_material(
            albedo=(0.2, 0.5, 0.2),
            glow_color=(0.0, 1.0, 0.3),
            glow_intensity=2.0,
        )

        emission_result = ti.Vector.field(3, dtype=ti.f32, shape=())

        @ti.kernel
        def test_emission():
            mat_id = 0
            mat_type = get_material_type(mat_id)
            type_idx = get_material_type_index(mat_id)

            if mat_type == int(MaterialType.PHOSPHORESCENT):
                emission_result[None] = get_phosphorescent_emission_by_id(type_idx)

        test_emission()

        # Expected: glow_color * glow_intensity = (0, 1, 0.3) * 2.0 = (0, 2.0, 0.6)
        assert abs(emission_result[None][0] - 0.0) < 1e-6
        assert abs(emission_result[None][1] - 2.0) < 1e-6
        assert abs(emission_result[None][2] - 0.6) < 1e-6


class TestPhosphorescentRendering:
    """Integration tests for rendering scenes with phosphorescent materials."""

    def test_phosphorescent_sphere_renders_without_errors(self) -> None:
        """Test that a scene with phosphorescent sphere renders without errors."""
        import numpy as np

        from src.python.camera.pinhole import PinholeCamera, setup_camera
        from src.python.core.integrator import (
            disable_light,
            get_image,
            render_image,
            setup_render_target,
        )
        from src.python.scene.manager import SceneManager

        setup_render_target(32, 32)

        # Create scene with phosphorescent sphere
        scene = SceneManager()
        scene.add_phosphorescent_sphere(
            center=(0.0, 0.0, -2.0),
            radius=0.5,
            albedo=(0.2, 0.5, 0.2),
            glow_color=(0.0, 1.0, 0.3),
            glow_intensity=2.0,
        )

        camera = PinholeCamera(
            lookfrom=(0.0, 0.0, 0.0),
            lookat=(0.0, 0.0, -1.0),
            vup=(0.0, 1.0, 0.0),
            vfov=90.0,
            aspect_ratio=1.0,
        )
        setup_camera(camera)
        disable_light()  # Rely on phosphorescent emission

        # Should not crash
        render_image(num_samples=2)

        # Verify no NaN/Inf
        image = get_image().to_numpy()
        assert not np.any(np.isnan(image)), "Image contains NaN"
        assert not np.any(np.isinf(image)), "Image contains Inf"

    def test_phosphorescent_emission_contributes_to_image(self) -> None:
        """Test that phosphorescent emission contributes light to the rendered image."""
        import numpy as np

        from src.python.camera.pinhole import PinholeCamera, setup_camera
        from src.python.core.integrator import (
            disable_light,
            get_image,
            render_image,
            setup_render_target,
        )
        from src.python.scene.manager import SceneManager

        setup_render_target(32, 32)

        # Create scene with brightly glowing phosphorescent sphere
        scene = SceneManager()
        scene.add_phosphorescent_sphere(
            center=(0.0, 0.0, -2.0),
            radius=1.0,
            albedo=(0.1, 0.1, 0.1),
            glow_color=(0.0, 10.0, 0.0),  # Very bright green glow
            glow_intensity=5.0,
        )

        camera = PinholeCamera(
            lookfrom=(0.0, 0.0, 0.0),
            lookat=(0.0, 0.0, -1.0),
            vup=(0.0, 1.0, 0.0),
            vfov=90.0,
            aspect_ratio=1.0,
        )
        setup_camera(camera)
        disable_light()  # No other light source

        # Render more samples to accumulate emission
        render_image(num_samples=8)

        # Get image and check center region
        image = get_image().to_numpy()

        # Extract the active 32x32 region
        center_region = image[12:20, 12:20, :]

        # The center should have some illumination from the glowing sphere
        total_energy = np.sum(center_region)
        assert total_energy > 0.0, "Phosphorescent emission should contribute light to image"

        # Green channel should be dominant (we set glow_color to green)
        green_energy = np.sum(center_region[:, :, 1])
        red_energy = np.sum(center_region[:, :, 0])
        blue_energy = np.sum(center_region[:, :, 2])

        # Allow for some tolerance as colors mix during path tracing
        # but green should still be the strongest component
        if total_energy > 0.1:  # Only check if there's meaningful illumination
            assert green_energy >= red_energy, "Green should be dominant"
            assert green_energy >= blue_energy, "Green should be dominant"

    def test_phosphorescent_mixed_with_other_materials(self) -> None:
        """Test rendering scene with phosphorescent and other materials together."""
        import numpy as np

        from src.python.camera.pinhole import PinholeCamera, setup_camera
        from src.python.core.integrator import (
            disable_light,
            get_image,
            render_image,
            setup_render_target,
        )
        from src.python.scene.manager import SceneManager

        setup_render_target(32, 32)

        # Create scene with mixed materials
        scene = SceneManager()

        # Phosphorescent sphere (emitting)
        scene.add_phosphorescent_sphere(
            center=(-1.0, 0.0, -3.0),
            radius=0.5,
            albedo=(0.2, 0.5, 0.2),
            glow_color=(0.0, 5.0, 0.0),
            glow_intensity=2.0,
        )

        # Lambertian sphere
        scene.add_lambertian_sphere(
            center=(0.0, 0.0, -3.0),
            radius=0.5,
            albedo=(0.8, 0.3, 0.3),
        )

        # Metal sphere
        scene.add_metal_sphere(
            center=(1.0, 0.0, -3.0),
            radius=0.5,
            albedo=(0.8, 0.8, 0.8),
            roughness=0.1,
        )

        camera = PinholeCamera(
            lookfrom=(0.0, 0.0, 0.0),
            lookat=(0.0, 0.0, -1.0),
            vup=(0.0, 1.0, 0.0),
            vfov=90.0,
            aspect_ratio=1.0,
        )
        setup_camera(camera)
        disable_light()

        # Should render without errors
        render_image(num_samples=4)

        # Verify no NaN/Inf
        image = get_image().to_numpy()
        assert not np.any(np.isnan(image)), "Image contains NaN with mixed materials"
        assert not np.any(np.isinf(image)), "Image contains Inf with mixed materials"
        assert np.all(image >= 0.0), "Image contains negative values"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
