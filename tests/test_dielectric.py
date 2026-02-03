"""Unit tests for the Dielectric material module.

Tests cover:
- Refraction (Snell's law)
- Total internal reflection (TIR)
- Fresnel reflectance (Schlick approximation)
- Random reflection/refraction probability
- Energy conservation (attenuation = white)
- Material registry operations
- IOR validation
"""

import math

import pytest
import taichi as ti


class TestRefraction:
    """Tests for refraction (Snell's law)."""

    def test_refraction_normal_incidence(self):
        """Test refraction at normal incidence (straight on).

        At normal incidence, the ray should continue straight through
        (no bending) regardless of IOR.
        """
        from src.python.core.ray import refract

        result_dir = ti.Vector.field(3, dtype=ti.f32, shape=())

        @ti.kernel
        def test_kernel():
            ior = 1.5  # Glass
            incident = ti.math.vec3(0.0, -1.0, 0.0)  # Straight down
            normal = ti.math.vec3(0.0, 1.0, 0.0)  # Pointing up

            # At normal incidence with low Fresnel reflectance,
            # most rays will refract. We directly test refraction here.
            refracted = refract(incident, normal, 1.0 / ior)
            result_dir[None] = refracted

        test_kernel()
        d = result_dir[None]
        # At normal incidence, refracted ray should still point down
        # (slight deviation allowed due to IOR)
        assert d[1] < 0.0  # Still pointing downward
        assert abs(d[0]) < 0.01  # No significant x deviation
        assert abs(d[2]) < 0.01  # No significant z deviation

    def test_refraction_snells_law(self):
        """Test that refraction follows Snell's law."""
        from src.python.core.ray import refract

        result_dir = ti.Vector.field(3, dtype=ti.f32, shape=())

        @ti.kernel
        def test_kernel():
            # 45 degree incidence
            inv_sqrt2 = 1.0 / ti.sqrt(2.0)
            incident = ti.math.vec3(inv_sqrt2, -inv_sqrt2, 0.0)
            normal = ti.math.vec3(0.0, 1.0, 0.0)

            # Air to glass (IOR 1.5)
            eta = 1.0 / 1.5
            refracted = refract(incident, normal, eta)
            result_dir[None] = refracted

        test_kernel()
        d = result_dir[None]

        # Verify Snell's law: sin(theta1) / sin(theta2) = n2 / n1 = 1.5
        # sin(45) = 1/sqrt(2)
        # sin(theta2) = sin(45) / 1.5 = 1/(sqrt(2)*1.5)
        # theta2 = arcsin(0.4714) approx 28.1 degrees
        sin_theta1 = 1.0 / math.sqrt(2.0)
        sin_theta2 = sin_theta1 / 1.5

        # The x component of refracted direction should be sin(theta2)
        length = math.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2)
        if length > 0.01:  # Only check if we got a valid refraction
            d_normalized = [d[0] / length, d[1] / length, d[2] / length]
            # x component should be approximately sin(theta2)
            assert abs(d_normalized[0] - sin_theta2) < 0.01

    def test_refraction_air_to_glass(self):
        """Test refraction from air into glass."""
        from src.python.core.ray import refract

        result_dir = ti.Vector.field(3, dtype=ti.f32, shape=())

        @ti.kernel
        def test_kernel():
            # Incident at 30 degrees from normal
            incident = ti.math.normalize(ti.math.vec3(0.5, -0.866, 0.0))
            normal = ti.math.vec3(0.0, 1.0, 0.0)
            eta = 1.0 / 1.5  # Air to glass

            refracted = refract(incident, normal, eta)
            result_dir[None] = refracted

        test_kernel()
        d = result_dir[None]

        # Refracted ray should bend toward normal (smaller x component)
        incident_x = 0.5 / math.sqrt(0.5**2 + 0.866**2)
        # For air to glass, refracted ray bends toward normal
        assert abs(d[0]) < abs(incident_x)
        assert d[1] < 0  # Still going down

    def test_refraction_glass_to_air(self):
        """Test refraction from glass into air (without TIR)."""
        from src.python.core.ray import refract

        result_dir = ti.Vector.field(3, dtype=ti.f32, shape=())

        @ti.kernel
        def test_kernel():
            # Small incident angle (well below critical angle)
            incident = ti.math.normalize(ti.math.vec3(0.2, -0.98, 0.0))
            normal = ti.math.vec3(0.0, 1.0, 0.0)
            eta = 1.5  # Glass to air

            refracted = refract(incident, normal, eta)
            result_dir[None] = refracted

        test_kernel()
        d = result_dir[None]
        length = math.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2)

        # Should refract (not TIR) at this small angle
        assert length > 0.5  # Got a valid refraction
        # Ray should bend away from normal (larger x component)
        assert abs(d[0]) > 0.2  # Bent away from normal


class TestTotalInternalReflection:
    """Tests for total internal reflection (TIR)."""

    def test_tir_critical_angle_exceeded(self):
        """Test TIR when critical angle is exceeded."""
        from src.python.materials.dielectric import will_reflect

        result = ti.field(dtype=ti.i32, shape=())

        @ti.kernel
        def test_kernel():
            ior = 1.5
            # Critical angle for glass is arcsin(1/1.5) = 41.8 degrees
            # 60 degree incident angle (well beyond critical)
            incident = ti.math.normalize(ti.math.vec3(0.866, -0.5, 0.0))
            normal = ti.math.vec3(0.0, 1.0, 0.0)
            front_face = 0  # Inside glass, hitting boundary

            result[None] = will_reflect(ior, incident, normal, front_face)

        test_kernel()
        # Should have TIR (will_reflect = 1)
        assert result[None] == 1

    def test_tir_below_critical_angle(self):
        """Test no TIR when below critical angle."""
        from src.python.materials.dielectric import will_reflect

        result = ti.field(dtype=ti.i32, shape=())

        @ti.kernel
        def test_kernel():
            ior = 1.5
            # 20 degree incident angle (below critical angle of 41.8)
            incident = ti.math.normalize(ti.math.vec3(0.342, -0.940, 0.0))
            normal = ti.math.vec3(0.0, 1.0, 0.0)
            front_face = 0  # Inside glass

            result[None] = will_reflect(ior, incident, normal, front_face)

        test_kernel()
        # Should not have TIR (will_reflect = 0)
        assert result[None] == 0

    def test_tir_only_from_denser_medium(self):
        """Test that TIR only occurs when going from denser to less dense medium."""
        from src.python.materials.dielectric import will_reflect

        result_front = ti.field(dtype=ti.i32, shape=())
        result_back = ti.field(dtype=ti.i32, shape=())

        @ti.kernel
        def test_kernel():
            ior = 1.5
            # Same steep angle
            incident = ti.math.normalize(ti.math.vec3(0.866, -0.5, 0.0))
            normal = ti.math.vec3(0.0, 1.0, 0.0)

            # From outside (air to glass) - no TIR possible
            result_front[None] = will_reflect(ior, incident, normal, 1)

            # From inside (glass to air) - TIR possible
            result_back[None] = will_reflect(ior, incident, normal, 0)

        test_kernel()
        # No TIR from air to glass
        assert result_front[None] == 0
        # TIR from glass to air at steep angle
        assert result_back[None] == 1

    def test_tir_scatter_reflects(self):
        """Test that scatter_dielectric reflects when TIR occurs."""
        from src.python.core.ray import reflect
        from src.python.materials.dielectric import scatter_dielectric

        result_dir = ti.Vector.field(3, dtype=ti.f32, shape=())
        expected_dir = ti.Vector.field(3, dtype=ti.f32, shape=())

        @ti.kernel
        def test_kernel():
            ior = 1.5
            # Steep angle causing TIR
            incident = ti.math.normalize(ti.math.vec3(0.866, -0.5, 0.0))
            normal = ti.math.vec3(0.0, 1.0, 0.0)
            front_face = 0  # Inside glass

            # Force reflection by making TIR happen
            # Get the expected reflection direction
            expected_dir[None] = reflect(incident, normal)

            # Multiple samples - all should reflect due to TIR
            direction, _, _ = scatter_dielectric(ior, incident, normal, front_face)
            result_dir[None] = direction

        test_kernel()
        r = result_dir[None]
        e = expected_dir[None]

        # Direction should be the reflection (TIR)
        assert abs(r[0] - e[0]) < 0.01
        assert abs(r[1] - e[1]) < 0.01
        assert abs(r[2] - e[2]) < 0.01


class TestFresnelReflectance:
    """Tests for Fresnel reflectance (Schlick approximation)."""

    def test_fresnel_normal_incidence(self):
        """Test Fresnel reflectance at normal incidence.

        At normal incidence, R = ((n1-n2)/(n1+n2))^2
        For glass (n=1.5): R = ((1-1.5)/(1+1.5))^2 = 0.04
        """
        from src.python.materials.dielectric import fresnel_reflectance

        result = ti.field(dtype=ti.f32, shape=())

        @ti.kernel
        def test_kernel():
            ior = 1.5
            # Normal incidence (straight on)
            incident = ti.math.vec3(0.0, -1.0, 0.0)
            normal = ti.math.vec3(0.0, 1.0, 0.0)
            front_face = 1

            result[None] = fresnel_reflectance(ior, incident, normal, front_face)

        test_kernel()
        # Expected: ((1-1.5)/(1+1.5))^2 = 0.04
        expected = ((1.0 - 1.5) / (1.0 + 1.5)) ** 2
        assert abs(result[None] - expected) < 0.01

    def test_fresnel_grazing_angle(self):
        """Test Fresnel reflectance at grazing angle.

        At grazing angles, reflectance should approach 1.0.
        """
        from src.python.materials.dielectric import fresnel_reflectance

        result = ti.field(dtype=ti.f32, shape=())

        @ti.kernel
        def test_kernel():
            ior = 1.5
            # Grazing angle (nearly parallel to surface)
            incident = ti.math.normalize(ti.math.vec3(0.999, -0.045, 0.0))
            normal = ti.math.vec3(0.0, 1.0, 0.0)
            front_face = 1

            result[None] = fresnel_reflectance(ior, incident, normal, front_face)

        test_kernel()
        # At grazing angles, Fresnel reflectance should be high (close to 1)
        assert result[None] > 0.5

    def test_fresnel_increases_with_angle(self):
        """Test that Fresnel reflectance increases with incident angle."""
        from src.python.materials.dielectric import fresnel_reflectance

        result_normal = ti.field(dtype=ti.f32, shape=())
        result_45deg = ti.field(dtype=ti.f32, shape=())
        result_75deg = ti.field(dtype=ti.f32, shape=())

        @ti.kernel
        def test_kernel():
            ior = 1.5
            normal = ti.math.vec3(0.0, 1.0, 0.0)
            front_face = 1

            # Normal incidence
            incident_normal = ti.math.vec3(0.0, -1.0, 0.0)
            result_normal[None] = fresnel_reflectance(
                ior, incident_normal, normal, front_face
            )

            # 45 degrees
            inv_sqrt2 = 1.0 / ti.sqrt(2.0)
            incident_45 = ti.math.vec3(inv_sqrt2, -inv_sqrt2, 0.0)
            result_45deg[None] = fresnel_reflectance(ior, incident_45, normal, front_face)

            # 75 degrees
            incident_75 = ti.math.normalize(ti.math.vec3(0.966, -0.259, 0.0))
            result_75deg[None] = fresnel_reflectance(ior, incident_75, normal, front_face)

        test_kernel()
        # Fresnel reflectance should increase with angle
        assert result_normal[None] < result_45deg[None]
        assert result_45deg[None] < result_75deg[None]

    def test_fresnel_probability_distribution(self):
        """Test that reflection/refraction follows Fresnel probability."""
        from src.python.core.ray import reflect
        from src.python.materials.dielectric import scatter_dielectric

        num_samples = 5000
        reflect_count = ti.field(dtype=ti.i32, shape=())

        @ti.kernel
        def test_kernel():
            ior = 1.5
            # 60 degree incidence for moderate Fresnel
            incident = ti.math.normalize(ti.math.vec3(0.866, -0.5, 0.0))
            normal = ti.math.vec3(0.0, 1.0, 0.0)
            front_face = 1

            reflect_dir = reflect(incident, normal)

            count = 0
            for i in range(num_samples):
                direction, _, _ = scatter_dielectric(ior, incident, normal, front_face)
                # Check if this sample reflected (matches reflect direction)
                diff = ti.math.length(direction - reflect_dir)
                if diff < 0.01:
                    count += 1
            reflect_count[None] = count

        test_kernel()

        # Get expected Fresnel reflectance
        from src.python.materials.dielectric import fresnel_reflectance

        expected_r = ti.field(dtype=ti.f32, shape=())

        @ti.kernel
        def get_fresnel():
            ior = 1.5
            incident = ti.math.normalize(ti.math.vec3(0.866, -0.5, 0.0))
            normal = ti.math.vec3(0.0, 1.0, 0.0)
            expected_r[None] = fresnel_reflectance(ior, incident, normal, 1)

        get_fresnel()

        # Check that reflection probability roughly matches Fresnel
        observed_ratio = reflect_count[None] / num_samples
        expected_ratio = expected_r[None]

        # Allow 10% tolerance due to random sampling
        assert abs(observed_ratio - expected_ratio) < 0.1


class TestAttenuation:
    """Tests for attenuation (energy conservation)."""

    def test_attenuation_is_white(self):
        """Test that dielectric attenuation is white (1,1,1)."""
        from src.python.materials.dielectric import scatter_dielectric

        result_attenuation = ti.Vector.field(3, dtype=ti.f32, shape=())

        @ti.kernel
        def test_kernel():
            ior = 1.5
            incident = ti.math.vec3(0.0, -1.0, 0.0)
            normal = ti.math.vec3(0.0, 1.0, 0.0)
            front_face = 1

            _, attenuation, _ = scatter_dielectric(ior, incident, normal, front_face)
            result_attenuation[None] = attenuation

        test_kernel()
        a = result_attenuation[None]
        # Clear glass has white attenuation
        assert abs(a[0] - 1.0) < 1e-6
        assert abs(a[1] - 1.0) < 1e-6
        assert abs(a[2] - 1.0) < 1e-6

    def test_always_scatters(self):
        """Test that dielectric always scatters (did_scatter = 1)."""
        from src.python.materials.dielectric import scatter_dielectric

        num_samples = 100
        scatter_flags = ti.field(dtype=ti.i32, shape=num_samples)

        @ti.kernel
        def test_kernel():
            ior = 1.5
            incident = ti.math.vec3(0.0, -1.0, 0.0)
            normal = ti.math.vec3(0.0, 1.0, 0.0)
            front_face = 1

            for i in range(num_samples):
                _, _, did_scatter = scatter_dielectric(
                    ior, incident, normal, front_face
                )
                scatter_flags[i] = did_scatter

        test_kernel()

        # All samples should scatter
        for i in range(num_samples):
            assert scatter_flags[i] == 1


class TestScatteredDirectionNormalized:
    """Tests for direction normalization."""

    def test_scattered_direction_is_normalized(self):
        """Test that scattered direction is a unit vector."""
        from src.python.materials.dielectric import scatter_dielectric

        num_samples = 500
        lengths = ti.field(dtype=ti.f32, shape=num_samples)

        @ti.kernel
        def test_kernel():
            ior = 1.5
            incident = ti.math.normalize(ti.math.vec3(0.5, -0.866, 0.0))
            normal = ti.math.vec3(0.0, 1.0, 0.0)
            front_face = 1

            for i in range(num_samples):
                direction, _, _ = scatter_dielectric(ior, incident, normal, front_face)
                lengths[i] = ti.math.length(direction)

        test_kernel()

        min_len = min(lengths[i] for i in range(num_samples))
        max_len = max(lengths[i] for i in range(num_samples))

        assert abs(min_len - 1.0) < 0.01
        assert abs(max_len - 1.0) < 0.01


class TestMaterialRegistry:
    """Tests for material registry operations."""

    def test_add_and_get_material(self):
        """Test adding a material and retrieving its IOR."""
        from src.python.materials.dielectric import (
            add_dielectric_material,
            clear_dielectric_materials,
            get_dielectric_ior,
        )

        clear_dielectric_materials()
        idx = add_dielectric_material(ior=1.5)

        result = ti.field(dtype=ti.f32, shape=())

        @ti.kernel
        def test_kernel(mat_idx: ti.i32):
            result[None] = get_dielectric_ior(mat_idx)

        test_kernel(idx)
        assert abs(result[None] - 1.5) < 1e-6

    def test_material_count(self):
        """Test that material count is tracked correctly."""
        from src.python.materials.dielectric import (
            add_dielectric_material,
            clear_dielectric_materials,
            get_dielectric_material_count,
        )

        clear_dielectric_materials()
        assert get_dielectric_material_count() == 0

        add_dielectric_material(ior=1.5)
        assert get_dielectric_material_count() == 1

        add_dielectric_material(ior=1.33)
        assert get_dielectric_material_count() == 2

    def test_multiple_materials(self):
        """Test adding multiple materials with different IORs."""
        from src.python.materials.dielectric import (
            add_dielectric_material,
            clear_dielectric_materials,
            get_dielectric_ior,
        )

        clear_dielectric_materials()
        idx_glass = add_dielectric_material(ior=1.5)  # Glass
        idx_water = add_dielectric_material(ior=1.33)  # Water
        idx_diamond = add_dielectric_material(ior=2.4)  # Diamond

        result = ti.field(dtype=ti.f32, shape=())

        @ti.kernel
        def test_kernel(mat_idx: ti.i32):
            result[None] = get_dielectric_ior(mat_idx)

        test_kernel(idx_glass)
        assert abs(result[None] - 1.5) < 1e-6

        test_kernel(idx_water)
        assert abs(result[None] - 1.33) < 1e-6

        test_kernel(idx_diamond)
        assert abs(result[None] - 2.4) < 1e-6

    def test_default_ior_is_glass(self):
        """Test that default IOR is 1.5 (glass)."""
        from src.python.materials.dielectric import (
            add_dielectric_material,
            clear_dielectric_materials,
            get_dielectric_ior,
        )

        clear_dielectric_materials()
        idx = add_dielectric_material()  # No IOR specified

        result = ti.field(dtype=ti.f32, shape=())

        @ti.kernel
        def test_kernel(mat_idx: ti.i32):
            result[None] = get_dielectric_ior(mat_idx)

        test_kernel(idx)
        assert abs(result[None] - 1.5) < 1e-6

    def test_scatter_by_id(self):
        """Test scattering using material index."""
        from src.python.materials.dielectric import (
            add_dielectric_material,
            clear_dielectric_materials,
            scatter_dielectric_by_id,
        )

        clear_dielectric_materials()
        idx = add_dielectric_material(ior=1.5)

        result_attenuation = ti.Vector.field(3, dtype=ti.f32, shape=())
        result_scatter = ti.field(dtype=ti.i32, shape=())

        @ti.kernel
        def test_kernel(mat_idx: ti.i32):
            incident = ti.math.vec3(0.0, -1.0, 0.0)
            normal = ti.math.vec3(0.0, 1.0, 0.0)
            front_face = 1
            _, attenuation, did_scatter = scatter_dielectric_by_id(
                mat_idx, incident, normal, front_face
            )
            result_attenuation[None] = attenuation
            result_scatter[None] = did_scatter

        test_kernel(idx)
        a = result_attenuation[None]
        assert abs(a[0] - 1.0) < 1e-6
        assert abs(a[1] - 1.0) < 1e-6
        assert abs(a[2] - 1.0) < 1e-6
        assert result_scatter[None] == 1


class TestValidation:
    """Tests for parameter validation."""

    def test_ior_validation_below_one(self):
        """Test that IOR < 1 is rejected."""
        from src.python.materials.dielectric import (
            add_dielectric_material,
            clear_dielectric_materials,
        )

        clear_dielectric_materials()
        with pytest.raises(ValueError, match="less than 1.0"):
            add_dielectric_material(ior=0.9)

    def test_ior_boundary_value_one(self):
        """Test that IOR = 1.0 is valid (though physically unusual)."""
        from src.python.materials.dielectric import (
            add_dielectric_material,
            clear_dielectric_materials,
            get_dielectric_ior,
        )

        clear_dielectric_materials()
        idx = add_dielectric_material(ior=1.0)

        result = ti.field(dtype=ti.f32, shape=())

        @ti.kernel
        def test_kernel(mat_idx: ti.i32):
            result[None] = get_dielectric_ior(mat_idx)

        test_kernel(idx)
        assert abs(result[None] - 1.0) < 1e-6

    def test_high_ior_valid(self):
        """Test that high IOR values are valid (e.g., diamond = 2.4)."""
        from src.python.materials.dielectric import (
            add_dielectric_material,
            clear_dielectric_materials,
            get_dielectric_ior,
        )

        clear_dielectric_materials()
        idx = add_dielectric_material(ior=2.4)

        result = ti.field(dtype=ti.f32, shape=())

        @ti.kernel
        def test_kernel(mat_idx: ti.i32):
            result[None] = get_dielectric_ior(mat_idx)

        test_kernel(idx)
        assert abs(result[None] - 2.4) < 1e-6


class TestDielectricMaterialDataclass:
    """Tests for the DielectricMaterial dataclass."""

    def test_dataclass_creation(self):
        """Test creating a DielectricMaterial dataclass."""
        from src.python.materials.dielectric import DielectricMaterial

        result = ti.field(dtype=ti.f32, shape=())

        @ti.kernel
        def test_kernel():
            mat = DielectricMaterial(ior=1.5)
            result[None] = mat.ior

        test_kernel()
        assert abs(result[None] - 1.5) < 1e-6


class TestScatterFullVariant:
    """Tests for scatter_dielectric_full function."""

    def test_full_returns_pdf(self):
        """Test that scatter_dielectric_full returns nominal PDF."""
        from src.python.materials.dielectric import scatter_dielectric_full

        result_pdf = ti.field(dtype=ti.f32, shape=())
        result_scatter = ti.field(dtype=ti.i32, shape=())

        @ti.kernel
        def test_kernel():
            ior = 1.5
            incident = ti.math.vec3(0.0, -1.0, 0.0)
            normal = ti.math.vec3(0.0, 1.0, 0.0)
            front_face = 1

            _, _, pdf, did_scatter = scatter_dielectric_full(
                ior, incident, normal, front_face
            )
            result_pdf[None] = pdf
            result_scatter[None] = did_scatter

        test_kernel()
        # PDF should be 1.0 (nominal value for specular)
        assert abs(result_pdf[None] - 1.0) < 1e-6
        assert result_scatter[None] == 1


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_ior_one_no_refraction(self):
        """Test that IOR=1 (no refraction) passes ray through unchanged."""
        from src.python.core.ray import refract

        result_dir = ti.Vector.field(3, dtype=ti.f32, shape=())

        @ti.kernel
        def test_kernel():
            incident = ti.math.normalize(ti.math.vec3(0.5, -0.866, 0.0))
            normal = ti.math.vec3(0.0, 1.0, 0.0)
            eta = 1.0  # No refraction

            refracted = refract(incident, normal, eta)
            result_dir[None] = refracted

        test_kernel()
        d = result_dir[None]

        # Should pass through unchanged
        expected = [0.5 / math.sqrt(0.5**2 + 0.866**2), -0.866 / math.sqrt(0.5**2 + 0.866**2), 0.0]
        assert abs(d[0] - expected[0]) < 0.01
        assert abs(d[1] - expected[1]) < 0.01
        assert abs(d[2] - expected[2]) < 0.01

    def test_critical_angle_boundary(self):
        """Test behavior right at the critical angle."""
        from src.python.materials.dielectric import will_reflect

        result_below = ti.field(dtype=ti.i32, shape=())
        result_above = ti.field(dtype=ti.i32, shape=())

        @ti.kernel
        def test_kernel():
            ior = 1.5
            normal = ti.math.vec3(0.0, 1.0, 0.0)
            front_face = 0  # Inside glass

            # Critical angle = arcsin(1/1.5) = 41.81 degrees
            # Just below critical angle (40 degrees)
            incident_below = ti.math.normalize(ti.math.vec3(0.643, -0.766, 0.0))
            result_below[None] = will_reflect(ior, incident_below, normal, front_face)

            # Just above critical angle (45 degrees)
            inv_sqrt2 = 1.0 / ti.sqrt(2.0)
            incident_above = ti.math.vec3(inv_sqrt2, -inv_sqrt2, 0.0)
            result_above[None] = will_reflect(ior, incident_above, normal, front_face)

        test_kernel()
        # Below critical angle: should not have TIR
        assert result_below[None] == 0
        # Above critical angle: should have TIR
        assert result_above[None] == 1

    def test_back_face_handling(self):
        """Test that back face (inside material) is handled correctly."""
        from src.python.materials.dielectric import scatter_dielectric

        num_samples = 100
        directions = ti.Vector.field(3, dtype=ti.f32, shape=num_samples)

        @ti.kernel
        def test_kernel():
            ior = 1.5
            # Small angle (no TIR)
            incident = ti.math.normalize(ti.math.vec3(0.2, -0.98, 0.0))
            normal = ti.math.vec3(0.0, 1.0, 0.0)
            front_face = 0  # Inside material

            for i in range(100):
                direction, _, _ = scatter_dielectric(ior, incident, normal, front_face)
                directions[i] = direction

        test_kernel()

        # All directions should be valid (normalized)
        for i in range(100):
            d = directions[i]
            length = math.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2)
            assert abs(length - 1.0) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
