"""Unit tests for the pinhole camera module.

Tests cover:
- Camera setup and orthonormal basis computation
- Ray generation for center and corner pixels
- Ray direction correctness
- Jittered sampling for anti-aliasing
- Edge cases (different FOV, aspect ratios, camera orientations)
"""

import math

import pytest
import taichi as ti


class TestCameraSetup:
    """Tests for camera setup and basis computation."""

    def test_orthonormal_basis_default_orientation(self):
        """Test that u, v, w form orthonormal basis for default camera."""
        from src.python.camera.pinhole import PinholeCamera, get_camera_info, setup_camera

        camera = PinholeCamera(
            lookfrom=(0.0, 0.0, 3.0),
            lookat=(0.0, 0.0, 0.0),
            vup=(0.0, 1.0, 0.0),
            vfov=90.0,
            aspect_ratio=1.0,
        )
        setup_camera(camera)

        info = get_camera_info()
        u = info["u"]
        v = info["v"]
        w = info["w"]

        # Check orthogonality (dot products should be zero)
        dot_uv = u[0] * v[0] + u[1] * v[1] + u[2] * v[2]
        dot_uw = u[0] * w[0] + u[1] * w[1] + u[2] * w[2]
        dot_vw = v[0] * w[0] + v[1] * w[1] + v[2] * w[2]

        assert abs(dot_uv) < 1e-6, f"u and v not orthogonal: dot={dot_uv}"
        assert abs(dot_uw) < 1e-6, f"u and w not orthogonal: dot={dot_uw}"
        assert abs(dot_vw) < 1e-6, f"v and w not orthogonal: dot={dot_vw}"

        # Check unit length
        len_u = math.sqrt(u[0] ** 2 + u[1] ** 2 + u[2] ** 2)
        len_v = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
        len_w = math.sqrt(w[0] ** 2 + w[1] ** 2 + w[2] ** 2)

        assert abs(len_u - 1.0) < 1e-6, f"u not unit length: {len_u}"
        assert abs(len_v - 1.0) < 1e-6, f"v not unit length: {len_v}"
        assert abs(len_w - 1.0) < 1e-6, f"w not unit length: {len_w}"

    def test_basis_directions_looking_at_negative_z(self):
        """Test basis vectors when camera looks down -z axis."""
        from src.python.camera.pinhole import PinholeCamera, get_camera_info, setup_camera

        # Camera at origin looking toward -z
        camera = PinholeCamera(
            lookfrom=(0.0, 0.0, 0.0),
            lookat=(0.0, 0.0, -1.0),
            vup=(0.0, 1.0, 0.0),
            vfov=90.0,
            aspect_ratio=1.0,
        )
        setup_camera(camera)

        info = get_camera_info()

        # w should point toward +z (opposite view direction)
        assert abs(info["w"][0]) < 1e-6
        assert abs(info["w"][1]) < 1e-6
        assert abs(info["w"][2] - 1.0) < 1e-6

        # u should point toward +x (right)
        assert abs(info["u"][0] - 1.0) < 1e-6
        assert abs(info["u"][1]) < 1e-6
        assert abs(info["u"][2]) < 1e-6

        # v should point toward +y (up)
        assert abs(info["v"][0]) < 1e-6
        assert abs(info["v"][1] - 1.0) < 1e-6
        assert abs(info["v"][2]) < 1e-6

    def test_origin_set_correctly(self):
        """Test that camera origin is set to lookfrom."""
        from src.python.camera.pinhole import PinholeCamera, get_camera_info, setup_camera

        lookfrom = (1.5, 2.5, 3.5)
        camera = PinholeCamera(
            lookfrom=lookfrom,
            lookat=(0.0, 0.0, 0.0),
            vup=(0.0, 1.0, 0.0),
            vfov=60.0,
            aspect_ratio=16.0 / 9.0,
        )
        setup_camera(camera)

        info = get_camera_info()
        origin = info["origin"]

        assert abs(origin[0] - lookfrom[0]) < 1e-6
        assert abs(origin[1] - lookfrom[1]) < 1e-6
        assert abs(origin[2] - lookfrom[2]) < 1e-6


class TestRayGeneration:
    """Tests for ray generation functions."""

    def test_center_ray_direction(self):
        """Test that center ray (0.5, 0.5) points at lookat."""
        from src.python.camera.pinhole import PinholeCamera, get_ray, setup_camera

        camera = PinholeCamera(
            lookfrom=(0.0, 0.0, 3.0),
            lookat=(0.0, 0.0, 0.0),
            vup=(0.0, 1.0, 0.0),
            vfov=90.0,
            aspect_ratio=1.0,
        )
        setup_camera(camera)

        result_origin = ti.field(dtype=ti.math.vec3, shape=())
        result_dir = ti.field(dtype=ti.math.vec3, shape=())

        @ti.kernel
        def test_kernel():
            ray = get_ray(0.5, 0.5)
            result_origin[None] = ray.origin
            result_dir[None] = ray.direction

        test_kernel()

        origin = result_origin[None]
        direction = result_dir[None]

        # Origin should be at lookfrom
        assert abs(origin[0] - 0.0) < 1e-6
        assert abs(origin[1] - 0.0) < 1e-6
        assert abs(origin[2] - 3.0) < 1e-6

        # Direction should point toward lookat (i.e., toward -z)
        assert abs(direction[0]) < 1e-5
        assert abs(direction[1]) < 1e-5
        assert direction[2] < 0  # Should point toward -z

    def test_corner_rays_symmetric(self):
        """Test that corner rays are symmetric around center."""
        from src.python.camera.pinhole import PinholeCamera, get_ray, setup_camera

        camera = PinholeCamera(
            lookfrom=(0.0, 0.0, 1.0),
            lookat=(0.0, 0.0, 0.0),
            vup=(0.0, 1.0, 0.0),
            vfov=90.0,
            aspect_ratio=1.0,
        )
        setup_camera(camera)

        # Store rays for all corners
        ray_00 = ti.field(dtype=ti.math.vec3, shape=())
        ray_01 = ti.field(dtype=ti.math.vec3, shape=())
        ray_10 = ti.field(dtype=ti.math.vec3, shape=())
        ray_11 = ti.field(dtype=ti.math.vec3, shape=())

        @ti.kernel
        def test_kernel():
            ray_00[None] = get_ray(0.0, 0.0).direction
            ray_01[None] = get_ray(0.0, 1.0).direction
            ray_10[None] = get_ray(1.0, 0.0).direction
            ray_11[None] = get_ray(1.0, 1.0).direction

        test_kernel()

        d00 = ray_00[None]
        d01 = ray_01[None]
        d10 = ray_10[None]
        d11 = ray_11[None]

        # For symmetric camera (aspect=1), x components should be mirrored
        # d00 and d01 should have same x (left side)
        # d10 and d11 should have same x (right side, opposite sign)
        assert abs(d00[0] - d01[0]) < 1e-5, "Left corners x should match"
        assert abs(d10[0] - d11[0]) < 1e-5, "Right corners x should match"
        assert abs(d00[0] + d10[0]) < 1e-5, "Left and right x should be symmetric"

        # Similarly for y components
        assert abs(d00[1] - d10[1]) < 1e-5, "Bottom corners y should match"
        assert abs(d01[1] - d11[1]) < 1e-5, "Top corners y should match"
        assert abs(d00[1] + d01[1]) < 1e-5, "Top and bottom y should be symmetric"

    def test_ray_origin_is_camera_origin(self):
        """Test that all rays originate from camera position."""
        from src.python.camera.pinhole import PinholeCamera, get_ray, setup_camera

        lookfrom = (5.0, 3.0, 2.0)
        camera = PinholeCamera(
            lookfrom=lookfrom,
            lookat=(0.0, 0.0, 0.0),
            vup=(0.0, 1.0, 0.0),
            vfov=60.0,
            aspect_ratio=1.5,
        )
        setup_camera(camera)

        origins = ti.field(dtype=ti.math.vec3, shape=(4,))

        @ti.kernel
        def test_kernel():
            origins[0] = get_ray(0.0, 0.0).origin
            origins[1] = get_ray(1.0, 1.0).origin
            origins[2] = get_ray(0.5, 0.5).origin
            origins[3] = get_ray(0.25, 0.75).origin

        test_kernel()

        for i in range(4):
            origin = origins[i]
            assert abs(origin[0] - lookfrom[0]) < 1e-6
            assert abs(origin[1] - lookfrom[1]) < 1e-6
            assert abs(origin[2] - lookfrom[2]) < 1e-6

    def test_ray_direction_normalized(self):
        """Test that ray directions are unit length."""
        from src.python.camera.pinhole import PinholeCamera, get_ray, setup_camera
        from src.python.core.ray import length

        camera = PinholeCamera(
            lookfrom=(0.0, 0.0, 5.0),
            lookat=(0.0, 0.0, 0.0),
            vup=(0.0, 1.0, 0.0),
            vfov=45.0,
            aspect_ratio=2.0,
        )
        setup_camera(camera)

        lengths = ti.field(dtype=ti.f32, shape=(5,))

        @ti.kernel
        def test_kernel():
            lengths[0] = length(get_ray(0.0, 0.0).direction)
            lengths[1] = length(get_ray(1.0, 0.0).direction)
            lengths[2] = length(get_ray(0.0, 1.0).direction)
            lengths[3] = length(get_ray(1.0, 1.0).direction)
            lengths[4] = length(get_ray(0.5, 0.5).direction)

        test_kernel()

        for i in range(5):
            assert abs(lengths[i] - 1.0) < 1e-5, f"Ray {i} direction not normalized: {lengths[i]}"


class TestJitteredSampling:
    """Tests for jittered ray generation for anti-aliasing."""

    def test_jittered_rays_vary(self):
        """Test that jittered rays produce different directions."""
        from src.python.camera.pinhole import PinholeCamera, get_ray_jittered, setup_camera

        camera = PinholeCamera(
            lookfrom=(0.0, 0.0, 2.0),
            lookat=(0.0, 0.0, 0.0),
            vup=(0.0, 1.0, 0.0),
            vfov=90.0,
            aspect_ratio=1.0,
        )
        setup_camera(camera)

        # Generate multiple jittered rays for same pixel
        directions = ti.field(dtype=ti.math.vec3, shape=(10,))

        @ti.kernel
        def test_kernel():
            for i in range(10):
                directions[i] = get_ray_jittered(50, 50, 100, 100).direction

        test_kernel()

        # Check that not all directions are identical
        d0 = directions[0]
        all_same = True
        for i in range(1, 10):
            di = directions[i]
            if abs(di[0] - d0[0]) > 1e-6 or abs(di[1] - d0[1]) > 1e-6:
                all_same = False
                break

        assert not all_same, "Jittered rays should produce varying directions"

    def test_jittered_rays_in_pixel_bounds(self):
        """Test that jittered rays stay within pixel boundaries."""
        from src.python.camera.pinhole import (
            PinholeCamera,
            get_ray,
            get_ray_jittered,
            setup_camera,
        )

        camera = PinholeCamera(
            lookfrom=(0.0, 0.0, 1.0),
            lookat=(0.0, 0.0, 0.0),
            vup=(0.0, 1.0, 0.0),
            vfov=90.0,
            aspect_ratio=1.0,
        )
        setup_camera(camera)

        width, height = 100, 100
        pixel_i, pixel_j = 50, 50

        # Get the four corners of this pixel
        corner_dirs = ti.field(dtype=ti.math.vec3, shape=(4,))

        @ti.kernel
        def get_corners():
            u0 = ti.cast(pixel_i, ti.f32) / ti.cast(width, ti.f32)
            u1 = ti.cast(pixel_i + 1, ti.f32) / ti.cast(width, ti.f32)
            v0 = ti.cast(pixel_j, ti.f32) / ti.cast(height, ti.f32)
            v1 = ti.cast(pixel_j + 1, ti.f32) / ti.cast(height, ti.f32)

            corner_dirs[0] = get_ray(u0, v0).direction
            corner_dirs[1] = get_ray(u1, v0).direction
            corner_dirs[2] = get_ray(u0, v1).direction
            corner_dirs[3] = get_ray(u1, v1).direction

        get_corners()

        # Get bounds for x and y components
        min_x = min(corner_dirs[i][0] for i in range(4))
        max_x = max(corner_dirs[i][0] for i in range(4))
        min_y = min(corner_dirs[i][1] for i in range(4))
        max_y = max(corner_dirs[i][1] for i in range(4))

        # Generate jittered rays and check they're within bounds
        jittered_dirs = ti.field(dtype=ti.math.vec3, shape=(100,))

        @ti.kernel
        def get_jittered():
            for i in range(100):
                jittered_dirs[i] = get_ray_jittered(pixel_i, pixel_j, width, height).direction

        get_jittered()

        # Small tolerance for floating point
        eps = 1e-5
        for i in range(100):
            d = jittered_dirs[i]
            assert d[0] >= min_x - eps and d[0] <= max_x + eps, f"x out of bounds: {d[0]}"
            assert d[1] >= min_y - eps and d[1] <= max_y + eps, f"y out of bounds: {d[1]}"


class TestFieldOfView:
    """Tests for field of view calculation."""

    def test_fov_90_viewport_size(self):
        """Test that 90 degree FOV produces correct viewport geometry."""
        from src.python.camera.pinhole import PinholeCamera, get_camera_info, setup_camera

        # With vfov=90 degrees and aspect=1, at distance 1, viewport should be 2x2
        # tan(45) = 1, so half_height = 1, full_height = 2
        camera = PinholeCamera(
            lookfrom=(0.0, 0.0, 1.0),
            lookat=(0.0, 0.0, 0.0),
            vup=(0.0, 1.0, 0.0),
            vfov=90.0,
            aspect_ratio=1.0,
        )
        setup_camera(camera)

        info = get_camera_info()
        horizontal = info["horizontal"]
        vertical = info["vertical"]

        # Viewport should span 2 units in each direction
        h_len = math.sqrt(horizontal[0] ** 2 + horizontal[1] ** 2 + horizontal[2] ** 2)
        v_len = math.sqrt(vertical[0] ** 2 + vertical[1] ** 2 + vertical[2] ** 2)

        assert abs(h_len - 2.0) < 1e-5, f"Horizontal span wrong: {h_len}"
        assert abs(v_len - 2.0) < 1e-5, f"Vertical span wrong: {v_len}"

    def test_narrow_fov(self):
        """Test narrow FOV produces smaller viewport."""
        from src.python.camera.pinhole import PinholeCamera, get_camera_info, setup_camera

        # With vfov=60 degrees: tan(30) = 0.577, half_height = 0.577
        # full_height = 1.155 (approximately)
        camera = PinholeCamera(
            lookfrom=(0.0, 0.0, 1.0),
            lookat=(0.0, 0.0, 0.0),
            vup=(0.0, 1.0, 0.0),
            vfov=60.0,
            aspect_ratio=1.0,
        )
        setup_camera(camera)

        info = get_camera_info()
        vertical = info["vertical"]
        v_len = math.sqrt(vertical[0] ** 2 + vertical[1] ** 2 + vertical[2] ** 2)

        expected = 2.0 * math.tan(math.radians(30.0))
        assert abs(v_len - expected) < 1e-5, f"Vertical span wrong: {v_len}, expected {expected}"

    def test_aspect_ratio_affects_horizontal(self):
        """Test that aspect ratio scales horizontal viewport."""
        from src.python.camera.pinhole import PinholeCamera, get_camera_info, setup_camera

        # 16:9 aspect ratio
        camera = PinholeCamera(
            lookfrom=(0.0, 0.0, 1.0),
            lookat=(0.0, 0.0, 0.0),
            vup=(0.0, 1.0, 0.0),
            vfov=90.0,
            aspect_ratio=16.0 / 9.0,
        )
        setup_camera(camera)

        info = get_camera_info()
        horizontal = info["horizontal"]
        vertical = info["vertical"]

        h_len = math.sqrt(horizontal[0] ** 2 + horizontal[1] ** 2 + horizontal[2] ** 2)
        v_len = math.sqrt(vertical[0] ** 2 + vertical[1] ** 2 + vertical[2] ** 2)

        # Horizontal should be aspect_ratio * vertical
        assert abs(h_len / v_len - 16.0 / 9.0) < 1e-5


class TestCameraOrientation:
    """Tests for different camera orientations."""

    def test_looking_down(self):
        """Test camera looking straight down."""
        from src.python.camera.pinhole import PinholeCamera, get_camera_info, setup_camera

        camera = PinholeCamera(
            lookfrom=(0.0, 10.0, 0.0),
            lookat=(0.0, 0.0, 0.0),
            vup=(0.0, 0.0, -1.0),  # -z is "up" when looking down
            vfov=90.0,
            aspect_ratio=1.0,
        )
        setup_camera(camera)

        info = get_camera_info()

        # w should point up (opposite view direction)
        assert abs(info["w"][1] - 1.0) < 1e-5

    def test_tilted_camera(self):
        """Test camera with tilted orientation."""
        from src.python.camera.pinhole import PinholeCamera, get_camera_info, setup_camera

        camera = PinholeCamera(
            lookfrom=(5.0, 5.0, 5.0),
            lookat=(0.0, 0.0, 0.0),
            vup=(0.0, 1.0, 0.0),
            vfov=60.0,
            aspect_ratio=1.0,
        )
        setup_camera(camera)

        info = get_camera_info()

        # Verify orthonormality still holds
        u, v, w = info["u"], info["v"], info["w"]

        dot_uv = u[0] * v[0] + u[1] * v[1] + u[2] * v[2]
        dot_uw = u[0] * w[0] + u[1] * w[1] + u[2] * w[2]
        dot_vw = v[0] * w[0] + v[1] * w[1] + v[2] * w[2]

        assert abs(dot_uv) < 1e-5
        assert abs(dot_uw) < 1e-5
        assert abs(dot_vw) < 1e-5


class TestCameraUtilityFunctions:
    """Tests for utility functions."""

    def test_get_camera_origin_function(self):
        """Test get_camera_origin returns correct position."""
        from src.python.camera.pinhole import (
            PinholeCamera,
            get_camera_origin,
            setup_camera,
        )

        lookfrom = (1.0, 2.0, 3.0)
        camera = PinholeCamera(
            lookfrom=lookfrom,
            lookat=(0.0, 0.0, 0.0),
            vup=(0.0, 1.0, 0.0),
            vfov=90.0,
            aspect_ratio=1.0,
        )
        setup_camera(camera)

        result = ti.field(dtype=ti.math.vec3, shape=())

        @ti.kernel
        def test_kernel():
            result[None] = get_camera_origin()

        test_kernel()

        origin = result[None]
        assert abs(origin[0] - lookfrom[0]) < 1e-6
        assert abs(origin[1] - lookfrom[1]) < 1e-6
        assert abs(origin[2] - lookfrom[2]) < 1e-6

    def test_get_camera_basis_function(self):
        """Test get_camera_basis returns orthonormal vectors."""
        from src.python.camera.pinhole import (
            PinholeCamera,
            get_camera_basis,
            setup_camera,
        )

        camera = PinholeCamera(
            lookfrom=(0.0, 0.0, 5.0),
            lookat=(0.0, 0.0, 0.0),
            vup=(0.0, 1.0, 0.0),
            vfov=90.0,
            aspect_ratio=1.0,
        )
        setup_camera(camera)

        u_result = ti.field(dtype=ti.math.vec3, shape=())
        v_result = ti.field(dtype=ti.math.vec3, shape=())
        w_result = ti.field(dtype=ti.math.vec3, shape=())

        @ti.kernel
        def test_kernel():
            u, v, w = get_camera_basis()
            u_result[None] = u
            v_result[None] = v
            w_result[None] = w

        test_kernel()

        u = u_result[None]
        v = v_result[None]
        w = w_result[None]

        # Check u points right (+x)
        assert abs(u[0] - 1.0) < 1e-5
        # Check v points up (+y)
        assert abs(v[1] - 1.0) < 1e-5
        # Check w points backward (+z)
        assert abs(w[2] - 1.0) < 1e-5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
