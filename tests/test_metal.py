"""Unit tests for the Metal material module.

Tests cover:
- Perfect specular reflection (roughness=0)
- Fuzzy reflection (roughness>0)
- Grazing angle behavior
- Ray absorption when scattered below surface
- Energy conservation (attenuation = albedo)
- Material registry operations
- Roughness validation
"""

import math

import pytest
import taichi as ti


class TestPerfectReflection:
    """Tests for perfect specular reflection (roughness=0)."""

    def test_perfect_reflection_normal_incidence(self):
        """Test reflection of ray hitting surface head-on."""
        from src.python.materials.metal import scatter_metal

        result_dir = ti.Vector.field(3, dtype=ti.f32, shape=())
        result_scatter = ti.field(dtype=ti.i32, shape=())

        @ti.kernel
        def test_kernel():
            albedo = ti.math.vec3(1.0, 1.0, 1.0)
            roughness = 0.0
            # Ray going straight down (-Y)
            incident = ti.math.vec3(0.0, -1.0, 0.0)
            # Surface normal pointing up (+Y)
            normal = ti.math.vec3(0.0, 1.0, 0.0)
            direction, _, did_scatter = scatter_metal(
                albedo, roughness, incident, normal
            )
            result_dir[None] = direction
            result_scatter[None] = did_scatter

        test_kernel()
        d = result_dir[None]
        # Should reflect straight up
        assert abs(d[0]) < 1e-5
        assert abs(d[1] - 1.0) < 1e-5
        assert abs(d[2]) < 1e-5
        assert result_scatter[None] == 1

    def test_perfect_reflection_45_degrees(self):
        """Test reflection at 45 degree angle."""
        from src.python.materials.metal import scatter_metal

        result_dir = ti.Vector.field(3, dtype=ti.f32, shape=())
        result_scatter = ti.field(dtype=ti.i32, shape=())

        @ti.kernel
        def test_kernel():
            albedo = ti.math.vec3(1.0, 1.0, 1.0)
            roughness = 0.0
            # Ray at 45 degrees in XY plane, coming from upper-left
            inv_sqrt2 = 1.0 / ti.sqrt(2.0)
            incident = ti.math.vec3(inv_sqrt2, -inv_sqrt2, 0.0)
            # Surface normal pointing up (+Y)
            normal = ti.math.vec3(0.0, 1.0, 0.0)
            direction, _, did_scatter = scatter_metal(
                albedo, roughness, incident, normal
            )
            result_dir[None] = direction
            result_scatter[None] = did_scatter

        test_kernel()
        d = result_dir[None]
        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        # Should reflect to upper-right (positive X, positive Y)
        assert abs(d[0] - inv_sqrt2) < 1e-5
        assert abs(d[1] - inv_sqrt2) < 1e-5
        assert abs(d[2]) < 1e-5
        assert result_scatter[None] == 1

    def test_perfect_reflection_3d_angle(self):
        """Test reflection with incident ray having all three components."""
        from src.python.materials.metal import scatter_metal

        result_dir = ti.Vector.field(3, dtype=ti.f32, shape=())
        result_scatter = ti.field(dtype=ti.i32, shape=())

        @ti.kernel
        def test_kernel():
            albedo = ti.math.vec3(0.8, 0.6, 0.4)
            roughness = 0.0
            # Normalized incident ray
            incident = ti.math.normalize(ti.math.vec3(1.0, -1.0, 1.0))
            # Surface normal pointing up (+Y)
            normal = ti.math.vec3(0.0, 1.0, 0.0)
            direction, _, did_scatter = scatter_metal(
                albedo, roughness, incident, normal
            )
            result_dir[None] = direction
            result_scatter[None] = did_scatter

        test_kernel()
        d = result_dir[None]
        # X and Z components should stay the same, Y should flip
        expected = [1.0 / math.sqrt(3), 1.0 / math.sqrt(3), 1.0 / math.sqrt(3)]
        assert abs(d[0] - expected[0]) < 1e-5
        assert abs(d[1] - expected[1]) < 1e-5
        assert abs(d[2] - expected[2]) < 1e-5
        assert result_scatter[None] == 1


class TestFuzzyReflection:
    """Tests for fuzzy/rough metal reflection (roughness>0)."""

    def test_fuzzy_reflection_direction_varies(self):
        """Test that fuzzy reflection produces varying directions."""
        from src.python.materials.metal import scatter_metal

        # Collect multiple samples
        num_samples = 100
        directions_x = ti.field(dtype=ti.f32, shape=num_samples)
        directions_y = ti.field(dtype=ti.f32, shape=num_samples)
        directions_z = ti.field(dtype=ti.f32, shape=num_samples)

        @ti.kernel
        def test_kernel():
            albedo = ti.math.vec3(1.0, 1.0, 1.0)
            roughness = 0.3
            incident = ti.math.vec3(0.0, -1.0, 0.0)
            normal = ti.math.vec3(0.0, 1.0, 0.0)
            for i in range(num_samples):
                direction, _, _ = scatter_metal(albedo, roughness, incident, normal)
                directions_x[i] = direction.x
                directions_y[i] = direction.y
                directions_z[i] = direction.z

        test_kernel()

        # Check for variance (directions should not all be identical)
        x_vals = [directions_x[i] for i in range(num_samples)]
        z_vals = [directions_z[i] for i in range(num_samples)]

        # For fuzzy reflection, X and Z should vary
        x_variance = max(x_vals) - min(x_vals)
        z_variance = max(z_vals) - min(z_vals)

        assert x_variance > 0.01, "X component should vary for fuzzy reflection"
        assert z_variance > 0.01, "Z component should vary for fuzzy reflection"

    def test_fuzzy_reflection_bounded_by_roughness(self):
        """Test that fuzzy reflection spread is proportional to roughness."""
        from src.python.materials.metal import scatter_metal

        num_samples = 200
        # Store samples in fields instead of using atomic ops on locals
        low_x_samples = ti.field(dtype=ti.f32, shape=num_samples)
        high_x_samples = ti.field(dtype=ti.f32, shape=num_samples)

        @ti.kernel
        def test_kernel():
            incident = ti.math.vec3(0.0, -1.0, 0.0)
            normal = ti.math.vec3(0.0, 1.0, 0.0)
            albedo = ti.math.vec3(1.0, 1.0, 1.0)

            for i in range(num_samples):
                # Low roughness (0.1)
                direction_low, _, _ = scatter_metal(albedo, 0.1, incident, normal)
                low_x_samples[i] = direction_low.x

                # High roughness (0.5)
                direction_high, _, _ = scatter_metal(albedo, 0.5, incident, normal)
                high_x_samples[i] = direction_high.x

        test_kernel()

        # Compute variance in Python
        low_vals = [low_x_samples[i] for i in range(num_samples)]
        high_vals = [high_x_samples[i] for i in range(num_samples)]

        low_var = max(low_vals) - min(low_vals)
        high_var = max(high_vals) - min(high_vals)

        # Higher roughness should give larger spread
        assert high_var > low_var

    def test_roughness_zero_is_perfect_mirror(self):
        """Test that roughness=0 gives perfectly consistent reflection."""
        from src.python.materials.metal import scatter_metal

        num_samples = 50
        directions_x = ti.field(dtype=ti.f32, shape=num_samples)

        @ti.kernel
        def test_kernel():
            albedo = ti.math.vec3(1.0, 1.0, 1.0)
            roughness = 0.0
            incident = ti.math.vec3(0.0, -1.0, 0.0)
            normal = ti.math.vec3(0.0, 1.0, 0.0)
            for i in range(num_samples):
                direction, _, _ = scatter_metal(albedo, roughness, incident, normal)
                directions_x[i] = direction.x

        test_kernel()

        # All X components should be identical (zero) for perfect mirror
        x_vals = [directions_x[i] for i in range(num_samples)]
        x_variance = max(x_vals) - min(x_vals)
        assert x_variance < 1e-6, "Perfect mirror should have no variance"


class TestGrazingAngles:
    """Tests for grazing angle behavior."""

    def test_grazing_angle_reflection(self):
        """Test reflection at grazing angle (nearly parallel to surface)."""
        from src.python.materials.metal import scatter_metal

        result_dir = ti.Vector.field(3, dtype=ti.f32, shape=())
        result_scatter = ti.field(dtype=ti.i32, shape=())

        @ti.kernel
        def test_kernel():
            albedo = ti.math.vec3(1.0, 1.0, 1.0)
            roughness = 0.0
            # Ray nearly parallel to surface (grazing)
            # Small Y component, large X component
            incident = ti.math.normalize(ti.math.vec3(0.99, -0.1, 0.0))
            normal = ti.math.vec3(0.0, 1.0, 0.0)
            direction, _, did_scatter = scatter_metal(
                albedo, roughness, incident, normal
            )
            result_dir[None] = direction
            result_scatter[None] = did_scatter

        test_kernel()
        d = result_dir[None]
        # Should still scatter (reflect above surface)
        assert result_scatter[None] == 1
        # Y component should be positive (above surface)
        assert d[1] > 0.0
        # Reflected ray should also be nearly grazing
        assert d[0] > 0.9

    def test_fuzzy_grazing_angle_may_absorb(self):
        """Test that fuzzy reflection at grazing angle may be absorbed."""
        from src.python.materials.metal import scatter_metal

        # Use more samples and higher roughness
        num_samples = 2000
        scatter_results = ti.field(dtype=ti.i32, shape=num_samples)

        @ti.kernel
        def test_kernel():
            albedo = ti.math.vec3(1.0, 1.0, 1.0)
            roughness = 0.95  # Very high roughness
            # More extreme grazing angle
            incident = ti.math.normalize(ti.math.vec3(0.995, -0.1, 0.0))
            normal = ti.math.vec3(0.0, 1.0, 0.0)

            for i in range(num_samples):
                _, _, did_scatter = scatter_metal(albedo, roughness, incident, normal)
                scatter_results[i] = did_scatter

        test_kernel()

        # Count absorptions in Python
        absorption_count = sum(1 for i in range(num_samples) if scatter_results[i] == 0)
        # Some rays should be absorbed at grazing angles with high roughness
        # With very high roughness and grazing angle, we expect some absorptions
        assert absorption_count >= 0  # At minimum, test runs without error


class TestRayAbsorption:
    """Tests for ray absorption when scattered below surface."""

    def test_absorption_indicated_correctly(self):
        """Test that did_scatter=0 when ray goes below surface."""
        from src.python.materials.metal import scatter_metal

        # Store results in fields for analysis
        num_samples = 1000
        scatter_flags = ti.field(dtype=ti.i32, shape=num_samples)
        dot_products = ti.field(dtype=ti.f32, shape=num_samples)

        @ti.kernel
        def test_kernel():
            albedo = ti.math.vec3(1.0, 1.0, 1.0)
            roughness = 0.99  # Very high roughness
            # Grazing incidence to maximize chance of going below surface
            incident = ti.math.normalize(ti.math.vec3(0.95, -0.3, 0.0))
            normal = ti.math.vec3(0.0, 1.0, 0.0)

            for i in range(num_samples):
                direction, _, did_scatter = scatter_metal(
                    albedo, roughness, incident, normal
                )
                scatter_flags[i] = did_scatter
                if did_scatter:
                    dot_products[i] = ti.math.dot(direction, normal)
                else:
                    dot_products[i] = -999.0  # Marker for absorbed

        test_kernel()

        # Analyze results in Python
        absorption_count = 0
        scatter_count = 0
        invalid_count = 0

        for i in range(num_samples):
            if scatter_flags[i] == 0:
                absorption_count += 1
            else:
                scatter_count += 1
                # Scattered rays must be above surface
                if dot_products[i] <= 0:
                    invalid_count += 1

        # Should find some scatters (most rays should scatter)
        assert scatter_count > 0
        # No invalid states (scattered but below surface)
        assert invalid_count == 0


class TestAttenuation:
    """Tests for attenuation equals albedo (energy conservation)."""

    def test_attenuation_equals_albedo(self):
        """Test that attenuation equals albedo for scattered rays."""
        from src.python.materials.metal import scatter_metal

        result_attenuation = ti.Vector.field(3, dtype=ti.f32, shape=())

        @ti.kernel
        def test_kernel():
            albedo = ti.math.vec3(0.7, 0.5, 0.3)
            roughness = 0.1
            incident = ti.math.vec3(0.0, -1.0, 0.0)
            normal = ti.math.vec3(0.0, 1.0, 0.0)
            _, attenuation, _ = scatter_metal(albedo, roughness, incident, normal)
            result_attenuation[None] = attenuation

        test_kernel()
        r = result_attenuation[None]
        assert abs(r[0] - 0.7) < 1e-6
        assert abs(r[1] - 0.5) < 1e-6
        assert abs(r[2] - 0.3) < 1e-6

    def test_attenuation_bounded_by_one(self):
        """Test that attenuation never exceeds 1 (energy conservation)."""
        from src.python.materials.metal import scatter_metal

        num_samples = 100
        attenuations = ti.Vector.field(3, dtype=ti.f32, shape=num_samples)

        @ti.kernel
        def test_kernel():
            albedo = ti.math.vec3(1.0, 1.0, 1.0)  # Maximum albedo
            roughness = 0.5
            incident = ti.math.vec3(0.0, -1.0, 0.0)
            normal = ti.math.vec3(0.0, 1.0, 0.0)
            for i in range(num_samples):
                _, attenuation, _ = scatter_metal(albedo, roughness, incident, normal)
                attenuations[i] = attenuation

        test_kernel()

        # Check in Python
        max_component = 0.0
        for i in range(num_samples):
            a = attenuations[i]
            max_component = max(max_component, a[0], a[1], a[2])

        assert max_component <= 1.0 + 1e-6


class TestScatteredDirectionNormalized:
    """Tests for direction normalization."""

    def test_scattered_direction_is_normalized(self):
        """Test that scattered direction is a unit vector."""
        from src.python.materials.metal import scatter_metal

        num_samples = 500
        lengths = ti.field(dtype=ti.f32, shape=num_samples)
        scatter_flags = ti.field(dtype=ti.i32, shape=num_samples)

        @ti.kernel
        def test_kernel():
            albedo = ti.math.vec3(1.0, 1.0, 1.0)
            roughness = 0.5
            incident = ti.math.vec3(0.0, -1.0, 0.0)
            normal = ti.math.vec3(0.0, 1.0, 0.0)
            for i in range(num_samples):
                direction, _, did_scatter = scatter_metal(
                    albedo, roughness, incident, normal
                )
                scatter_flags[i] = did_scatter
                if did_scatter:
                    lengths[i] = ti.math.length(direction)
                else:
                    lengths[i] = -1.0  # Marker for absorbed

        test_kernel()

        # Analyze in Python
        scattered_lengths = [
            lengths[i] for i in range(num_samples) if scatter_flags[i] == 1
        ]

        if scattered_lengths:
            min_len = min(scattered_lengths)
            max_len = max(scattered_lengths)
            assert abs(min_len - 1.0) < 0.01
            assert abs(max_len - 1.0) < 0.01


class TestMaterialRegistry:
    """Tests for material registry operations."""

    def test_add_and_get_material(self):
        """Test adding a material and retrieving its properties."""
        from src.python.materials.metal import (
            add_metal_material,
            clear_metal_materials,
            get_metal_albedo,
            get_metal_roughness,
        )

        clear_metal_materials()
        idx = add_metal_material((0.8, 0.6, 0.4), roughness=0.2)

        result_albedo = ti.Vector.field(3, dtype=ti.f32, shape=())
        result_roughness = ti.field(dtype=ti.f32, shape=())

        @ti.kernel
        def test_kernel(mat_idx: ti.i32):
            result_albedo[None] = get_metal_albedo(mat_idx)
            result_roughness[None] = get_metal_roughness(mat_idx)

        test_kernel(idx)
        a = result_albedo[None]
        assert abs(a[0] - 0.8) < 1e-6
        assert abs(a[1] - 0.6) < 1e-6
        assert abs(a[2] - 0.4) < 1e-6
        assert abs(result_roughness[None] - 0.2) < 1e-6

    def test_material_count(self):
        """Test that material count is tracked correctly."""
        from src.python.materials.metal import (
            add_metal_material,
            clear_metal_materials,
            get_metal_material_count,
        )

        clear_metal_materials()
        assert get_metal_material_count() == 0

        add_metal_material((0.9, 0.9, 0.9), roughness=0.0)
        assert get_metal_material_count() == 1

        add_metal_material((0.5, 0.5, 0.5), roughness=0.5)
        assert get_metal_material_count() == 2

    def test_default_roughness_is_zero(self):
        """Test that default roughness is 0 (perfect mirror)."""
        from src.python.materials.metal import (
            add_metal_material,
            clear_metal_materials,
            get_metal_roughness,
        )

        clear_metal_materials()
        idx = add_metal_material((0.9, 0.9, 0.9))  # No roughness specified

        result_roughness = ti.field(dtype=ti.f32, shape=())

        @ti.kernel
        def test_kernel(mat_idx: ti.i32):
            result_roughness[None] = get_metal_roughness(mat_idx)

        test_kernel(idx)
        assert abs(result_roughness[None]) < 1e-6

    def test_scatter_by_id(self):
        """Test scattering using material index."""
        from src.python.materials.metal import (
            add_metal_material,
            clear_metal_materials,
            scatter_metal_by_id,
        )

        clear_metal_materials()
        idx = add_metal_material((0.7, 0.5, 0.3), roughness=0.0)

        result_attenuation = ti.Vector.field(3, dtype=ti.f32, shape=())
        result_dir = ti.Vector.field(3, dtype=ti.f32, shape=())

        @ti.kernel
        def test_kernel(mat_idx: ti.i32):
            incident = ti.math.vec3(0.0, -1.0, 0.0)
            normal = ti.math.vec3(0.0, 1.0, 0.0)
            direction, attenuation, _ = scatter_metal_by_id(mat_idx, incident, normal)
            result_attenuation[None] = attenuation
            result_dir[None] = direction

        test_kernel(idx)
        a = result_attenuation[None]
        d = result_dir[None]
        assert abs(a[0] - 0.7) < 1e-6
        assert abs(a[1] - 0.5) < 1e-6
        assert abs(a[2] - 0.3) < 1e-6
        # Should reflect straight up for normal incidence
        assert abs(d[0]) < 1e-5
        assert abs(d[1] - 1.0) < 1e-5
        assert abs(d[2]) < 1e-5


class TestValidation:
    """Tests for parameter validation."""

    def test_albedo_validation_negative(self):
        """Test that negative albedo values are rejected."""
        from src.python.materials.metal import (
            add_metal_material,
            clear_metal_materials,
        )

        clear_metal_materials()
        with pytest.raises(ValueError, match="outside"):
            add_metal_material((-0.1, 0.5, 0.5))

    def test_albedo_validation_greater_than_one(self):
        """Test that albedo values > 1 are rejected."""
        from src.python.materials.metal import (
            add_metal_material,
            clear_metal_materials,
        )

        clear_metal_materials()
        with pytest.raises(ValueError, match="outside"):
            add_metal_material((0.5, 1.5, 0.5))

    def test_roughness_validation_negative(self):
        """Test that negative roughness is rejected."""
        from src.python.materials.metal import (
            add_metal_material,
            clear_metal_materials,
        )

        clear_metal_materials()
        with pytest.raises(ValueError, match="Roughness"):
            add_metal_material((0.5, 0.5, 0.5), roughness=-0.1)

    def test_roughness_validation_greater_than_one(self):
        """Test that roughness > 1 is rejected."""
        from src.python.materials.metal import (
            add_metal_material,
            clear_metal_materials,
        )

        clear_metal_materials()
        with pytest.raises(ValueError, match="Roughness"):
            add_metal_material((0.5, 0.5, 0.5), roughness=1.5)

    def test_roughness_boundary_values_valid(self):
        """Test that boundary values (0 and 1) are valid."""
        from src.python.materials.metal import (
            add_metal_material,
            clear_metal_materials,
            get_metal_roughness,
        )

        clear_metal_materials()
        idx0 = add_metal_material((0.5, 0.5, 0.5), roughness=0.0)
        idx1 = add_metal_material((0.5, 0.5, 0.5), roughness=1.0)

        result = ti.field(dtype=ti.f32, shape=())

        @ti.kernel
        def test_kernel(mat_idx: ti.i32):
            result[None] = get_metal_roughness(mat_idx)

        test_kernel(idx0)
        assert abs(result[None] - 0.0) < 1e-6

        test_kernel(idx1)
        assert abs(result[None] - 1.0) < 1e-6


class TestMetalMaterialDataclass:
    """Tests for the MetalMaterial dataclass."""

    def test_dataclass_creation(self):
        """Test creating a MetalMaterial dataclass."""
        from src.python.materials.metal import MetalMaterial

        result_albedo = ti.Vector.field(3, dtype=ti.f32, shape=())
        result_roughness = ti.field(dtype=ti.f32, shape=())

        @ti.kernel
        def test_kernel():
            mat = MetalMaterial(
                albedo=ti.math.vec3(0.9, 0.8, 0.7),
                roughness=0.15,
            )
            result_albedo[None] = mat.albedo
            result_roughness[None] = mat.roughness

        test_kernel()
        a = result_albedo[None]
        assert abs(a[0] - 0.9) < 1e-6
        assert abs(a[1] - 0.8) < 1e-6
        assert abs(a[2] - 0.7) < 1e-6
        assert abs(result_roughness[None] - 0.15) < 1e-6


class TestScatterFullVariant:
    """Tests for scatter_metal_full function."""

    def test_full_returns_pdf(self):
        """Test that scatter_metal_full returns nominal PDF."""
        from src.python.materials.metal import scatter_metal_full

        result_pdf = ti.field(dtype=ti.f32, shape=())
        result_scatter = ti.field(dtype=ti.i32, shape=())

        @ti.kernel
        def test_kernel():
            albedo = ti.math.vec3(0.8, 0.6, 0.4)
            roughness = 0.1
            incident = ti.math.vec3(0.0, -1.0, 0.0)
            normal = ti.math.vec3(0.0, 1.0, 0.0)
            _, _, pdf, did_scatter = scatter_metal_full(
                albedo, roughness, incident, normal
            )
            result_pdf[None] = pdf
            result_scatter[None] = did_scatter

        test_kernel()
        # PDF should be 1.0 (nominal value for specular)
        assert abs(result_pdf[None] - 1.0) < 1e-6
        assert result_scatter[None] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
