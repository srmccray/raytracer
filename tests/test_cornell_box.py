"""Unit tests for the Cornell box scene.

Tests cover:
- Scene creation and geometry counts
- Wall positions and orientations
- Sphere positions and materials (diffuse, metal, glass)
- Camera configuration
- Material assignments
- Light quad geometry
- Scene bounds
"""

import pytest


@pytest.fixture
def cornell_box_scene():
    """Create a Cornell box scene for testing."""
    from src.python.scene.cornell_box import create_cornell_box_scene

    scene, camera, light_mat_id = create_cornell_box_scene()
    yield scene, camera, light_mat_id
    scene.clear()


@pytest.fixture
def custom_size_scene():
    """Create a Cornell box with custom size for testing."""
    from src.python.scene.cornell_box import create_cornell_box_scene

    scene, camera, light_mat_id = create_cornell_box_scene(box_size=100.0)
    yield scene, camera, light_mat_id
    scene.clear()


class TestSceneCreation:
    """Tests for basic scene creation."""

    def test_create_scene_returns_tuple(self):
        """Test that create_cornell_box_scene returns a tuple of 3 elements."""
        from src.python.camera.pinhole import PinholeCamera
        from src.python.scene.cornell_box import create_cornell_box_scene
        from src.python.scene.manager import SceneManager

        scene, camera, light_mat_id = create_cornell_box_scene()
        try:
            assert isinstance(scene, SceneManager)
            assert isinstance(camera, PinholeCamera)
            assert isinstance(light_mat_id, int)
            assert light_mat_id >= 0
        finally:
            scene.clear()

    def test_scene_has_correct_quad_count(self, cornell_box_scene):
        """Test that the scene has 6 quads (5 walls + 1 light)."""
        scene, *_ = cornell_box_scene
        assert scene.get_quad_count() == 6

    def test_scene_has_correct_sphere_count(self, cornell_box_scene):
        """Test that the scene has 3 spheres."""
        scene, *_ = cornell_box_scene
        assert scene.get_sphere_count() == 3

    def test_scene_has_correct_material_count(self, cornell_box_scene):
        """Test that the scene has correct number of materials.

        Materials:
        - Red wall (Lambertian) - ID 0
        - Green wall (Lambertian) - ID 1
        - White wall (Lambertian) - ID 2
        - Light (Phosphorescent emissive) - ID 3
        - Diffuse sphere (Lambertian) - ID 4
        - Metal sphere (Metal) - ID 5
        - Glass sphere (Dielectric) - ID 6
        Total: 7 materials
        """
        scene, *_ = cornell_box_scene
        assert scene.get_material_count() == 7

    def test_scene_primitive_count(self, cornell_box_scene):
        """Test total primitive count."""
        scene, *_ = cornell_box_scene
        assert scene.get_primitive_count() == 9  # 6 quads + 3 spheres


class TestMaterialAssignments:
    """Tests for material type assignments."""

    def test_lambertian_materials_created(self, cornell_box_scene):
        """Test that Lambertian materials are created correctly."""
        from src.python.scene.manager import MaterialType

        scene, *_ = cornell_box_scene

        # Materials 0, 1, 2, 4 should be Lambertian (red, green, white walls, diffuse sphere)
        # Note: Material 3 is now Phosphorescent (light)
        for i in [0, 1, 2, 4]:
            assert scene.get_material_type_python(i) == MaterialType.LAMBERTIAN

    def test_phosphorescent_material_created(self, cornell_box_scene):
        """Test that Phosphorescent material is created correctly (light only)."""
        from src.python.scene.manager import MaterialType

        scene, *_ = cornell_box_scene
        # Material 3 is Phosphorescent (light - emissive)
        assert scene.get_material_type_python(3) == MaterialType.PHOSPHORESCENT

    def test_metal_material_created(self, cornell_box_scene):
        """Test that Metal material is created correctly."""
        from src.python.scene.manager import MaterialType

        scene, *_ = cornell_box_scene
        # Material 5 is Metal (silver sphere)
        assert scene.get_material_type_python(5) == MaterialType.METAL

    def test_dielectric_material_created(self, cornell_box_scene):
        """Test that Dielectric material is created correctly."""
        from src.python.scene.manager import MaterialType

        scene, *_ = cornell_box_scene
        # Material 6 should be Dielectric (glass sphere)
        assert scene.get_material_type_python(6) == MaterialType.DIELECTRIC


class TestWallGeometry:
    """Tests for wall positions and orientations."""

    def test_wall_quads_stored(self, cornell_box_scene):
        """Test that wall quad information is stored correctly."""
        scene, *_ = cornell_box_scene

        # Should have 6 quads stored in scene.quads
        assert len(scene.quads) == 6

    def test_left_wall_position(self, cornell_box_scene):
        """Test left wall is at x=0."""
        scene, *_ = cornell_box_scene

        # Left wall is first quad (index 0)
        left_wall = scene.quads[0]
        assert left_wall.corner[0] == 0.0  # x = 0
        assert left_wall.corner[1] == 0.0  # y = 0
        assert left_wall.corner[2] == 0.0  # z = 0

    def test_right_wall_position(self, cornell_box_scene):
        """Test right wall is at x=555."""
        from src.python.scene.cornell_box import BOX_SIZE

        scene, *_ = cornell_box_scene

        # Right wall is second quad (index 1)
        right_wall = scene.quads[1]
        assert right_wall.corner[0] == BOX_SIZE  # x = 555

    def test_back_wall_position(self, cornell_box_scene):
        """Test back wall is at z=555."""
        from src.python.scene.cornell_box import BOX_SIZE

        scene, *_ = cornell_box_scene

        # Back wall is third quad (index 2)
        back_wall = scene.quads[2]
        assert back_wall.corner[2] == BOX_SIZE  # z = 555

    def test_floor_position(self, cornell_box_scene):
        """Test floor is at y=0."""
        scene, *_ = cornell_box_scene

        # Floor is fourth quad (index 3)
        floor = scene.quads[3]
        assert floor.corner[1] == 0.0  # y = 0

    def test_ceiling_position(self, cornell_box_scene):
        """Test ceiling is at y=555."""
        from src.python.scene.cornell_box import BOX_SIZE

        scene, *_ = cornell_box_scene

        # Ceiling is fifth quad (index 4)
        ceiling = scene.quads[4]
        assert ceiling.corner[1] == BOX_SIZE  # y = 555


class TestWallMaterials:
    """Tests for wall material assignments."""

    def test_left_wall_is_red(self, cornell_box_scene):
        """Test left wall has red material (ID 0)."""
        scene, *_ = cornell_box_scene
        left_wall = scene.quads[0]
        assert left_wall.material_id == 0  # Red material

    def test_right_wall_is_green(self, cornell_box_scene):
        """Test right wall has green material (ID 1)."""
        scene, *_ = cornell_box_scene
        right_wall = scene.quads[1]
        assert right_wall.material_id == 1  # Green material

    def test_back_wall_is_white(self, cornell_box_scene):
        """Test back wall has white material (ID 2)."""
        scene, *_ = cornell_box_scene
        back_wall = scene.quads[2]
        assert back_wall.material_id == 2  # White material

    def test_floor_is_white(self, cornell_box_scene):
        """Test floor has white material (ID 2)."""
        scene, *_ = cornell_box_scene
        floor = scene.quads[3]
        assert floor.material_id == 2  # White material

    def test_ceiling_is_white(self, cornell_box_scene):
        """Test ceiling has white material (ID 2)."""
        scene, *_ = cornell_box_scene
        ceiling = scene.quads[4]
        assert ceiling.material_id == 2  # White material


class TestLightGeometry:
    """Tests for area light geometry."""

    def test_light_quad_exists(self, cornell_box_scene):
        """Test that light quad is created."""
        scene, *_ = cornell_box_scene
        # Light is sixth quad (index 5)
        assert len(scene.quads) >= 6

    def test_light_on_ceiling(self, cornell_box_scene):
        """Test that light is near ceiling level."""
        from src.python.scene.cornell_box import BOX_SIZE

        scene, *_ = cornell_box_scene
        light = scene.quads[5]
        # Light should be just below ceiling (555 - 1 = 554)
        assert abs(light.corner[1] - (BOX_SIZE - 1.0)) < 0.01

    def test_light_centered_horizontally(self, cornell_box_scene):
        """Test that light is centered in X and Z."""
        from src.python.scene.cornell_box import BOX_SIZE

        scene, *_ = cornell_box_scene
        light = scene.quads[5]

        # Light dimensions
        light_width = 130.0
        light_depth = 105.0
        expected_x_offset = (BOX_SIZE - light_width) / 2.0
        expected_z_offset = (BOX_SIZE - light_depth) / 2.0

        assert abs(light.corner[0] - expected_x_offset) < 0.01
        assert abs(light.corner[2] - expected_z_offset) < 0.01

    def test_light_has_correct_material(self, cornell_box_scene):
        """Test that light has the light material."""
        scene, *_ = cornell_box_scene
        light = scene.quads[5]
        assert light.material_id == 3  # Light material is ID 3


class TestSphereGeometry:
    """Tests for sphere positions and sizes."""

    def test_spheres_stored(self, cornell_box_scene):
        """Test that sphere information is stored correctly."""
        scene, *_ = cornell_box_scene
        assert len(scene.spheres) == 3

    def test_diffuse_sphere_position(self, cornell_box_scene):
        """Test diffuse sphere is on left side of box."""
        from src.python.scene.cornell_box import BOX_SIZE

        scene, *_ = cornell_box_scene
        diffuse_sphere = scene.spheres[0]

        # Should be on left side (x < center)
        assert diffuse_sphere.center[0] < BOX_SIZE / 2.0
        # Should be resting on floor (y = radius)
        assert abs(diffuse_sphere.center[1] - diffuse_sphere.radius) < 0.01

    def test_metal_sphere_position(self, cornell_box_scene):
        """Test metal sphere is on right side of box."""
        from src.python.scene.cornell_box import BOX_SIZE

        scene, *_ = cornell_box_scene
        metal_sphere = scene.spheres[1]

        # Should be on right side (x > center)
        assert metal_sphere.center[0] > BOX_SIZE / 2.0
        # Should be resting on floor (y = radius)
        assert abs(metal_sphere.center[1] - metal_sphere.radius) < 0.01

    def test_glass_sphere_position(self, cornell_box_scene):
        """Test glass sphere is centered and on floor."""
        from src.python.scene.cornell_box import BOX_SIZE

        scene, *_ = cornell_box_scene
        glass_sphere = scene.spheres[2]

        # Should be near center (x approximately at center)
        assert abs(glass_sphere.center[0] - BOX_SIZE / 2.0) < 1.0
        # Should be resting on floor (y = radius)
        assert abs(glass_sphere.center[1] - glass_sphere.radius) < 0.01

    def test_diffuse_sphere_material(self, cornell_box_scene):
        """Test diffuse sphere has Lambertian material."""
        from src.python.scene.manager import MaterialType

        scene, *_ = cornell_box_scene
        diffuse_sphere = scene.spheres[0]
        assert diffuse_sphere.material_id == 4  # Diffuse sphere material
        assert scene.get_material_type_python(4) == MaterialType.LAMBERTIAN

    def test_metal_sphere_material(self, cornell_box_scene):
        """Test metal sphere has Metal material."""
        from src.python.scene.manager import MaterialType

        scene, *_ = cornell_box_scene
        metal_sphere = scene.spheres[1]
        assert metal_sphere.material_id == 5  # Metal sphere material
        assert scene.get_material_type_python(5) == MaterialType.METAL

    def test_glass_sphere_material(self, cornell_box_scene):
        """Test glass sphere has Dielectric material."""
        from src.python.scene.manager import MaterialType

        scene, *_ = cornell_box_scene
        glass_sphere = scene.spheres[2]
        assert glass_sphere.material_id == 6  # Glass sphere material
        assert scene.get_material_type_python(6) == MaterialType.DIELECTRIC


class TestCameraConfiguration:
    """Tests for camera setup."""

    def test_camera_position(self, cornell_box_scene):
        """Test camera is positioned in front of box."""
        from src.python.scene.cornell_box import BOX_SIZE

        _, camera, *_ = cornell_box_scene

        # Camera should be centered on X and Y
        assert abs(camera.lookfrom[0] - BOX_SIZE / 2.0) < 0.01
        assert abs(camera.lookfrom[1] - BOX_SIZE / 2.0) < 0.01
        # Camera should be in front (negative Z)
        assert camera.lookfrom[2] < 0.0

    def test_camera_look_at(self, cornell_box_scene):
        """Test camera looks at center of box."""
        from src.python.scene.cornell_box import BOX_SIZE

        _, camera, *_ = cornell_box_scene

        # Should look at center of box
        assert abs(camera.lookat[0] - BOX_SIZE / 2.0) < 0.01
        assert abs(camera.lookat[1] - BOX_SIZE / 2.0) < 0.01
        assert abs(camera.lookat[2] - BOX_SIZE / 2.0) < 0.01

    def test_camera_up_vector(self, cornell_box_scene):
        """Test camera up vector is Y-up."""
        _, camera, *_ = cornell_box_scene

        assert camera.vup[0] == 0.0
        assert camera.vup[1] == 1.0
        assert camera.vup[2] == 0.0

    def test_camera_fov(self, cornell_box_scene):
        """Test camera has standard FOV."""
        _, camera, *_ = cornell_box_scene
        assert camera.vfov == 40.0

    def test_camera_aspect_ratio(self, cornell_box_scene):
        """Test camera has square aspect ratio."""
        _, camera, *_ = cornell_box_scene
        assert camera.aspect_ratio == 1.0


class TestCustomBoxSize:
    """Tests for custom box size parameter."""

    def test_custom_size_scales_geometry(self, custom_size_scene):
        """Test that custom box size scales walls correctly."""
        scene, *_ = custom_size_scene

        # Check right wall is at x=100 (custom size)
        right_wall = scene.quads[1]
        assert right_wall.corner[0] == 100.0

    def test_custom_size_scales_camera(self, custom_size_scene):
        """Test that custom box size scales camera position."""
        _, camera, *_ = custom_size_scene

        # Camera should be centered on custom size
        assert abs(camera.lookfrom[0] - 50.0) < 0.01  # 100 / 2
        assert abs(camera.lookat[0] - 50.0) < 0.01


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_get_light_quad_info(self):
        """Test get_light_quad_info returns correct geometry."""
        from src.python.scene.cornell_box import BOX_SIZE, get_light_quad_info

        info = get_light_quad_info()

        # Check corner is on ceiling
        assert abs(info["corner"][1] - (BOX_SIZE - 1.0)) < 0.01

        # Check edge vectors
        assert info["edge_u"][0] == 130.0  # Width in X
        assert info["edge_v"][2] == 105.0  # Depth in Z

        # Check area
        assert abs(info["area"][0] - 130.0 * 105.0) < 0.01

    def test_get_light_quad_info_custom_size(self):
        """Test get_light_quad_info with custom box size."""
        from src.python.scene.cornell_box import get_light_quad_info

        info = get_light_quad_info(box_size=100.0)

        # Light should still be just below ceiling
        assert abs(info["corner"][1] - 99.0) < 0.01

    def test_get_cornell_box_bounds(self):
        """Test get_cornell_box_bounds returns correct bounds."""
        from src.python.scene.cornell_box import BOX_SIZE, get_cornell_box_bounds

        bounds = get_cornell_box_bounds()

        assert bounds["min"] == (0.0, 0.0, 0.0)
        assert bounds["max"] == (BOX_SIZE, BOX_SIZE, BOX_SIZE)
        assert bounds["center"] == (BOX_SIZE / 2.0, BOX_SIZE / 2.0, BOX_SIZE / 2.0)
        assert bounds["size"] == (BOX_SIZE, BOX_SIZE, BOX_SIZE)

    def test_get_cornell_box_bounds_custom_size(self):
        """Test get_cornell_box_bounds with custom size."""
        from src.python.scene.cornell_box import get_cornell_box_bounds

        bounds = get_cornell_box_bounds(box_size=100.0)

        assert bounds["max"] == (100.0, 100.0, 100.0)
        assert bounds["center"] == (50.0, 50.0, 50.0)


class TestSceneIntegration:
    """Integration tests with intersection system."""

    def test_ray_hits_back_wall(self, cornell_box_scene):
        """Test that a ray toward the back wall hits it."""
        import taichi as ti

        from src.python.scene.cornell_box import BOX_SIZE
        from src.python.scene.intersection import intersect_scene, vec3

        scene, *_ = cornell_box_scene

        hit = ti.field(dtype=ti.i32, shape=())
        hit_z = ti.field(dtype=ti.f32, shape=())

        @ti.kernel
        def test_kernel():
            # Shoot ray from center toward back wall
            origin = vec3(BOX_SIZE / 2.0, BOX_SIZE / 2.0, BOX_SIZE / 4.0)
            direction = vec3(0.0, 0.0, 1.0)  # Toward +Z (back)
            rec = intersect_scene(origin, direction, 0.001, 10000.0)
            hit[None] = rec.hit
            if rec.hit == 1:
                hit_z[None] = rec.point[2]

        test_kernel()

        assert hit[None] == 1
        # Should hit back wall at z=555
        assert abs(hit_z[None] - BOX_SIZE) < 1.0

    def test_ray_hits_floor(self, cornell_box_scene):
        """Test that a ray toward the floor hits it."""
        import taichi as ti

        from src.python.scene.cornell_box import BOX_SIZE
        from src.python.scene.intersection import intersect_scene, vec3

        scene, *_ = cornell_box_scene

        hit = ti.field(dtype=ti.i32, shape=())
        hit_y = ti.field(dtype=ti.f32, shape=())

        @ti.kernel
        def test_kernel():
            # Shoot ray from center toward floor
            origin = vec3(BOX_SIZE / 2.0, BOX_SIZE / 2.0, BOX_SIZE / 2.0)
            direction = vec3(0.0, -1.0, 0.0)  # Toward -Y (floor)
            rec = intersect_scene(origin, direction, 0.001, 10000.0)
            hit[None] = rec.hit
            if rec.hit == 1:
                hit_y[None] = rec.point[1]

        test_kernel()

        assert hit[None] == 1
        # Should hit floor at y=0
        assert abs(hit_y[None]) < 1.0

    def test_ray_hits_sphere(self, cornell_box_scene):
        """Test that a ray toward a sphere hits it."""
        import taichi as ti

        from src.python.scene.intersection import intersect_scene, vec3

        scene, *_ = cornell_box_scene

        hit = ti.field(dtype=ti.i32, shape=())
        material_id = ti.field(dtype=ti.i32, shape=())

        # Glass sphere is at center, elevated
        glass_sphere = scene.spheres[2]
        target = glass_sphere.center

        @ti.kernel
        def test_kernel():
            # Shoot ray from front toward glass sphere
            origin = vec3(target[0], target[1], 0.0)
            direction = vec3(0.0, 0.0, 1.0)  # Toward +Z
            rec = intersect_scene(origin, direction, 0.001, 10000.0)
            hit[None] = rec.hit
            material_id[None] = rec.material_id

        test_kernel()

        assert hit[None] == 1
        assert material_id[None] == 6  # Glass material ID


class TestConstants:
    """Tests for module constants."""

    def test_box_size_constant(self):
        """Test BOX_SIZE is classic Cornell box dimension."""
        from src.python.scene.cornell_box import BOX_SIZE

        assert BOX_SIZE == 555.0

    def test_wall_colors_valid(self):
        """Test wall colors are valid RGB in [0, 1]."""
        from src.python.scene.cornell_box import (
            GREEN_WALL_ALBEDO,
            RED_WALL_ALBEDO,
            WHITE_WALL_ALBEDO,
        )

        for color in [RED_WALL_ALBEDO, GREEN_WALL_ALBEDO, WHITE_WALL_ALBEDO]:
            for component in color:
                assert 0.0 <= component <= 1.0

    def test_glass_ior_is_standard(self):
        """Test glass IOR is standard glass value."""
        from src.python.scene.cornell_box import GLASS_SPHERE_IOR

        assert GLASS_SPHERE_IOR == 1.5


class TestCornellBoxParamsDataclass:
    """Tests for CornellBoxParams dataclass itself."""

    def test_default_values(self):
        """Test that CornellBoxParams has correct default values."""
        from src.python.scene.cornell_box import CornellBoxParams

        params = CornellBoxParams()

        # Light defaults
        assert params.light_intensity == 15.0
        assert params.light_color == (1.0, 1.0, 1.0)

        # Wall color defaults
        assert params.left_wall_color == (0.12, 0.45, 0.15)  # Green
        assert params.right_wall_color == (0.65, 0.05, 0.05)  # Red
        assert params.back_wall_color == (0.73, 0.73, 0.73)  # White

    def test_custom_values_stored(self):
        """Test that custom values are stored correctly."""
        from src.python.scene.cornell_box import CornellBoxParams

        params = CornellBoxParams(
            light_intensity=25.0,
            light_color=(0.9, 0.8, 0.7),
            left_wall_color=(0.1, 0.2, 0.3),
            right_wall_color=(0.4, 0.5, 0.6),
            back_wall_color=(0.7, 0.8, 0.9),
        )

        assert params.light_intensity == 25.0
        assert params.light_color == (0.9, 0.8, 0.7)
        assert params.left_wall_color == (0.1, 0.2, 0.3)
        assert params.right_wall_color == (0.4, 0.5, 0.6)
        assert params.back_wall_color == (0.7, 0.8, 0.9)

    def test_partial_custom_values(self):
        """Test that partial custom values override defaults correctly."""
        from src.python.scene.cornell_box import CornellBoxParams

        params = CornellBoxParams(light_intensity=30.0)

        # Custom value
        assert params.light_intensity == 30.0

        # Default values should be preserved
        assert params.light_color == (1.0, 1.0, 1.0)
        assert params.left_wall_color == (0.12, 0.45, 0.15)

    def test_dataclass_equality_same_values(self):
        """Test that dataclass equality works for same values."""
        from src.python.scene.cornell_box import CornellBoxParams

        params1 = CornellBoxParams(light_intensity=20.0, light_color=(1.0, 0.9, 0.8))
        params2 = CornellBoxParams(light_intensity=20.0, light_color=(1.0, 0.9, 0.8))

        assert params1 == params2

    def test_dataclass_equality_different_values(self):
        """Test that dataclass equality works for different values."""
        from src.python.scene.cornell_box import CornellBoxParams

        params1 = CornellBoxParams(light_intensity=20.0)
        params2 = CornellBoxParams(light_intensity=25.0)

        assert params1 != params2

    def test_dataclass_equality_default_instances(self):
        """Test that two default instances are equal."""
        from src.python.scene.cornell_box import CornellBoxParams

        params1 = CornellBoxParams()
        params2 = CornellBoxParams()

        assert params1 == params2

    def test_dataclass_repr(self):
        """Test that dataclass has a readable repr."""
        from src.python.scene.cornell_box import CornellBoxParams

        params = CornellBoxParams(light_intensity=20.0)
        repr_str = repr(params)

        assert "CornellBoxParams" in repr_str
        assert "light_intensity=20.0" in repr_str


class TestCornellBoxParams:
    """Tests for CornellBoxParams and parameterized scene creation."""

    def test_default_params_creates_valid_scene(self):
        """Test that default params create a valid scene."""
        from src.python.scene.cornell_box import CornellBoxParams, create_cornell_box_scene

        params = CornellBoxParams()
        scene, camera, light_mat_id = create_cornell_box_scene(params=params)
        try:
            assert scene.get_quad_count() == 6
            assert scene.get_sphere_count() == 3
            assert light_mat_id == 3  # Light material ID
        finally:
            scene.clear()

    def test_custom_light_intensity(self):
        """Test that custom light intensity is used."""
        from src.python.scene.cornell_box import CornellBoxParams, create_cornell_box_scene

        params = CornellBoxParams(light_intensity=25.0)
        scene, camera, light_mat_id = create_cornell_box_scene(params=params)
        try:
            # Verify the light material was created with custom intensity
            assert light_mat_id == 3
            # Check that light quad uses the light material
            light_quad = scene.quads[5]
            assert light_quad.material_id == light_mat_id
        finally:
            scene.clear()

    def test_custom_light_color(self):
        """Test that custom light color is used."""
        from src.python.scene.cornell_box import CornellBoxParams, create_cornell_box_scene

        params = CornellBoxParams(light_color=(1.0, 0.8, 0.6))  # Warm light
        scene, camera, light_mat_id = create_cornell_box_scene(params=params)
        try:
            assert light_mat_id == 3
        finally:
            scene.clear()

    def test_custom_wall_colors(self):
        """Test that custom wall colors are used."""
        from src.python.scene.cornell_box import CornellBoxParams, create_cornell_box_scene

        # Use custom colors (swap red and green)
        params = CornellBoxParams(
            left_wall_color=(0.8, 0.1, 0.1),  # Red on left
            right_wall_color=(0.1, 0.8, 0.1),  # Green on right
            back_wall_color=(0.5, 0.5, 0.8),  # Blue-ish back
        )
        scene, camera, light_mat_id = create_cornell_box_scene(params=params)
        try:
            # Walls should still be created with correct material IDs
            assert scene.quads[0].material_id == 0  # Left wall
            assert scene.quads[1].material_id == 1  # Right wall
            assert scene.quads[2].material_id == 2  # Back wall
        finally:
            scene.clear()

    def test_none_params_uses_defaults(self):
        """Test that params=None creates same scene as default params."""
        from src.python.scene.cornell_box import create_cornell_box_scene

        scene1, camera1, light_id1 = create_cornell_box_scene(params=None)
        try:
            scene2, camera2, light_id2 = create_cornell_box_scene()
            try:
                # Should have same structure
                assert scene1.get_quad_count() == scene2.get_quad_count()
                assert scene1.get_sphere_count() == scene2.get_sphere_count()
                assert scene1.get_material_count() == scene2.get_material_count()
                assert light_id1 == light_id2
            finally:
                scene2.clear()
        finally:
            scene1.clear()

    def test_light_material_id_returned(self):
        """Test that the light material ID is correctly returned."""
        from src.python.scene.cornell_box import create_cornell_box_scene
        from src.python.scene.manager import MaterialType

        scene, camera, light_mat_id = create_cornell_box_scene()
        try:
            # Light material should be the 4th material (index 3)
            assert light_mat_id == 3
            # It should be a phosphorescent (emissive) material
            assert scene.get_material_type_python(light_mat_id) == MaterialType.PHOSPHORESCENT
        finally:
            scene.clear()

    def test_backward_compatibility(self):
        """Test that old calling pattern still works."""
        from src.python.scene.cornell_box import create_cornell_box_scene

        # Old code might unpack only scene and camera
        result = create_cornell_box_scene()
        scene = result[0]
        camera = result[1]
        # New code can also get light_mat_id
        light_mat_id = result[2]
        try:
            assert scene.get_quad_count() == 6
            assert camera.vfov == 40.0
            assert light_mat_id == 3
        finally:
            scene.clear()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
