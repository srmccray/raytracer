"""Path tracing integrator for Monte Carlo light transport.

This module implements the main rendering kernel using unbiased path tracing
with material-based scattering, Russian roulette termination, and progressive
sample accumulation.

The path tracer solves the rendering equation by tracing rays from the camera
through the scene, bouncing off surfaces according to their material properties,
and accumulating radiance along each path.

Key features:
    - Material dispatch (Lambertian, Metal, Dielectric)
    - Russian roulette termination after minimum bounces
    - Progressive sample accumulation for convergence
    - Area light emission support
    - Self-intersection avoidance with ray offset

Example:
    >>> import taichi as ti
    >>> ti.init(arch=ti.gpu)
    >>> from src.python.core.integrator import (
    ...     render_sample, render_image, setup_render_target
    ... )
    >>> from src.python.scene.cornell_box import create_cornell_box_scene
    >>> from src.python.camera.pinhole import setup_camera
    >>>
    >>> scene, camera = create_cornell_box_scene()
    >>> setup_camera(camera)
    >>> setup_render_target(512, 512)
    >>> render_image(num_samples=100)
"""

import taichi as ti
import taichi.math as tm

from src.python.camera.pinhole import get_ray_jittered
from src.python.materials.dielectric import (
    get_dielectric_ior,
    scatter_dielectric,
)
from src.python.materials.lambertian import (
    get_lambertian_albedo,
    scatter_lambertian,
)
from src.python.materials.metal import (
    get_metal_albedo,
    get_metal_roughness,
    scatter_metal,
)
from src.python.scene.intersection import intersect_scene
from src.python.scene.manager import (
    MaterialType,
    get_material_type,
    get_material_type_index,
)

# Type alias for 3D vectors
vec3 = tm.vec3

# =============================================================================
# Rendering Constants
# =============================================================================

# Maximum ray bounces (path length)
MAX_DEPTH = 50

# Minimum bounces before Russian roulette can terminate paths
MIN_BOUNCES_BEFORE_RR = 3

# Russian roulette survival probability cap
MAX_RR_PROBABILITY = 0.95

# Ray offset epsilon to avoid self-intersection
RAY_EPSILON = 1e-4

# t_min and t_max for ray intersection
T_MIN = 1e-4
T_MAX = 1e10

# Background/environment color (dark gray for indirect illumination)
BACKGROUND_COLOR = vec3(0.0, 0.0, 0.0)

# =============================================================================
# Light Source Configuration
# =============================================================================

# Area light emission fields (configured by setup_light)
_light_enabled = ti.field(dtype=ti.i32, shape=())
_light_corner = ti.Vector.field(3, dtype=ti.f32, shape=())
_light_edge_u = ti.Vector.field(3, dtype=ti.f32, shape=())
_light_edge_v = ti.Vector.field(3, dtype=ti.f32, shape=())
_light_emission = ti.Vector.field(3, dtype=ti.f32, shape=())
_light_material_id = ti.field(dtype=ti.i32, shape=())
_light_normal = ti.Vector.field(3, dtype=ti.f32, shape=())


def setup_light(
    corner: tuple[float, float, float],
    edge_u: tuple[float, float, float],
    edge_v: tuple[float, float, float],
    emission: tuple[float, float, float],
    material_id: int,
) -> None:
    """Configure the area light source for emission.

    The light is defined as a parallelogram with vertices at:
    corner, corner+edge_u, corner+edge_v, corner+edge_u+edge_v

    Args:
        corner: The corner point of the light quad.
        edge_u: First edge vector.
        edge_v: Second edge vector.
        emission: The emitted radiance (RGB).
        material_id: The material ID assigned to the light quad.
    """
    import math

    _light_enabled[None] = 1
    _light_corner[None] = [corner[0], corner[1], corner[2]]
    _light_edge_u[None] = [edge_u[0], edge_u[1], edge_u[2]]
    _light_edge_v[None] = [edge_v[0], edge_v[1], edge_v[2]]
    _light_emission[None] = [emission[0], emission[1], emission[2]]
    _light_material_id[None] = material_id

    # Compute light normal (cross product of edges, normalized) using pure Python
    # Cross product: u x v
    nx = edge_u[1] * edge_v[2] - edge_u[2] * edge_v[1]
    ny = edge_u[2] * edge_v[0] - edge_u[0] * edge_v[2]
    nz = edge_u[0] * edge_v[1] - edge_u[1] * edge_v[0]

    # Normalize
    length = math.sqrt(nx * nx + ny * ny + nz * nz)
    if length > 1e-8:
        nx /= length
        ny /= length
        nz /= length

    _light_normal[None] = [nx, ny, nz]


def disable_light() -> None:
    """Disable the area light (for scenes without emission)."""
    _light_enabled[None] = 0


def is_light_enabled() -> bool:
    """Check if the area light is enabled."""
    return bool(_light_enabled[None])


# =============================================================================
# Render Target (Image Buffer)
# =============================================================================

# Maximum supported image dimensions (preallocated to avoid kernel recompilation)
MAX_IMAGE_WIDTH = 2048
MAX_IMAGE_HEIGHT = 2048

# Image dimensions (actual active size)
_image_width = ti.field(dtype=ti.i32, shape=())
_image_height = ti.field(dtype=ti.i32, shape=())

# Color accumulation buffer (preallocated to max size)
_color_buffer = ti.Vector.field(3, dtype=ti.f32, shape=(MAX_IMAGE_WIDTH, MAX_IMAGE_HEIGHT))

# Sample count per pixel (preallocated to max size)
_sample_count = ti.field(dtype=ti.i32, shape=(MAX_IMAGE_WIDTH, MAX_IMAGE_HEIGHT))

# Flag to track if render target is initialized
_render_target_initialized = ti.field(dtype=ti.i32, shape=())


def setup_render_target(width: int, height: int) -> None:
    """Initialize the render target buffers.

    Sets the active image dimensions and clears the buffers.
    The buffers are preallocated to MAX_IMAGE_WIDTH x MAX_IMAGE_HEIGHT
    to avoid Taichi kernel recompilation issues.

    Args:
        width: Image width in pixels (max MAX_IMAGE_WIDTH).
        height: Image height in pixels (max MAX_IMAGE_HEIGHT).

    Raises:
        ValueError: If dimensions exceed maximum supported size.
    """
    if width > MAX_IMAGE_WIDTH or height > MAX_IMAGE_HEIGHT:
        raise ValueError(
            f"Image dimensions ({width}x{height}) exceed maximum supported "
            f"({MAX_IMAGE_WIDTH}x{MAX_IMAGE_HEIGHT})"
        )

    _image_width[None] = width
    _image_height[None] = height
    _render_target_initialized[None] = 1

    # Clear buffers
    clear_render_target()


def clear_render_target() -> None:
    """Clear the render target buffers to zero."""
    _color_buffer.fill(0.0)
    _sample_count.fill(0)


def get_image_dimensions() -> tuple[int, int]:
    """Get the current render target dimensions.

    Returns:
        Tuple of (width, height).
    """
    return int(_image_width[None]), int(_image_height[None])


def _check_render_target_initialized() -> None:
    """Check if render target is initialized and raise if not."""
    if _render_target_initialized[None] == 0:
        raise RuntimeError("Render target not set up. Call setup_render_target() first.")


def get_image() -> "ti.MatrixField":
    """Get the color buffer.

    Note: This returns the full preallocated buffer. Use get_image_dimensions()
    to determine the active region.

    Returns:
        A Taichi field containing the RGB image.

    Raises:
        RuntimeError: If render target has not been set up.
    """
    _check_render_target_initialized()
    return _color_buffer


def get_sample_count() -> "ti.ScalarField":
    """Get the sample count field.

    Note: This returns the full preallocated buffer. Use get_image_dimensions()
    to determine the active region.

    Returns:
        A Taichi field containing the sample count per pixel.

    Raises:
        RuntimeError: If render target has not been set up.
    """
    _check_render_target_initialized()
    return _sample_count


# =============================================================================
# Material Dispatch
# =============================================================================


@ti.func
def _scatter_material(
    material_id: ti.i32,
    incident_direction: vec3,
    hit_point: vec3,
    normal: vec3,
    front_face: ti.i32,
):
    """Dispatch to the appropriate material scattering function.

    Based on the material type, calls the corresponding scatter function
    and returns the scattered ray direction and attenuation.

    Args:
        material_id: The unified material ID.
        incident_direction: The incoming ray direction (normalized).
        hit_point: The intersection point on the surface.
        normal: The surface normal (normalized, facing toward ray).
        front_face: 1 if hit front face, 0 if back face.

    Returns:
        A tuple of (scattered_direction, attenuation, did_scatter) where:
        - scattered_direction: The new ray direction (normalized).
        - attenuation: The color attenuation for this bounce.
        - did_scatter: 1 if ray scattered, 0 if absorbed.
    """
    mat_type = get_material_type(material_id)
    type_index = get_material_type_index(material_id)

    # Default values
    scattered_direction = vec3(0.0, 0.0, 0.0)
    attenuation = vec3(0.0, 0.0, 0.0)
    did_scatter = 0

    if mat_type == int(MaterialType.LAMBERTIAN):
        albedo = get_lambertian_albedo(type_index)
        scattered_direction, attenuation, _ = scatter_lambertian(albedo, normal)
        did_scatter = 1

    elif mat_type == int(MaterialType.METAL):
        albedo = get_metal_albedo(type_index)
        roughness = get_metal_roughness(type_index)
        scattered_direction, attenuation, did_scatter = scatter_metal(
            albedo, roughness, incident_direction, normal
        )

    elif mat_type == int(MaterialType.DIELECTRIC):
        ior = get_dielectric_ior(type_index)
        scattered_direction, attenuation, did_scatter = scatter_dielectric(
            ior, incident_direction, normal, front_face
        )

    return scattered_direction, attenuation, did_scatter


@ti.func
def _get_emission(material_id: ti.i32, hit_point: vec3, normal: vec3) -> vec3:
    """Get the emission from a surface if it is an emitter.

    Currently only the configured area light emits.

    Args:
        material_id: The material ID of the hit surface.
        hit_point: The intersection point.
        normal: The surface normal.

    Returns:
        The emitted radiance (RGB), or zero if not an emitter.
    """
    emission = vec3(0.0, 0.0, 0.0)

    if _light_enabled[None] == 1:
        if material_id == _light_material_id[None]:
            # Check that we're hitting the front face of the light
            # (light normal should point toward camera)
            emission = _light_emission[None]

    return emission


# =============================================================================
# Path Tracing Core
# =============================================================================


@ti.func
def _offset_ray_origin(point: vec3, normal: vec3, direction: vec3) -> vec3:
    """Offset ray origin to avoid self-intersection.

    Pushes the point slightly along the geometric normal in the direction
    the ray will travel (above surface for reflection, below for refraction).

    Args:
        point: The intersection point.
        normal: The geometric surface normal.
        direction: The scattered ray direction.

    Returns:
        The offset origin point.
    """
    # If direction is going into the surface (dot < 0), offset inward
    offset_dir = normal
    if tm.dot(direction, normal) < 0.0:
        offset_dir = -normal
    return point + RAY_EPSILON * offset_dir


@ti.func
def trace_path(
    pixel_i: ti.i32,
    pixel_j: ti.i32,
    width: ti.i32,
    height: ti.i32,
) -> vec3:
    """Trace a single path from the camera through the scene.

    Implements unbiased Monte Carlo path tracing with:
    - Material-based scattering
    - Russian roulette termination after MIN_BOUNCES_BEFORE_RR bounces
    - Emission accumulation from light sources
    - Background color for escaped rays

    Args:
        pixel_i: Pixel x-coordinate (0 = left).
        pixel_j: Pixel y-coordinate (0 = bottom).
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        The estimated radiance (RGB) for this path sample.
    """
    # Generate camera ray with jitter for anti-aliasing
    ray = get_ray_jittered(pixel_i, pixel_j, width, height)
    origin = ray.origin
    direction = ray.direction

    # Accumulated radiance for this path
    radiance = vec3(0.0, 0.0, 0.0)

    # Throughput (product of all BSDF * cos / pdf terms along path)
    throughput = vec3(1.0, 1.0, 1.0)

    # Active flag for path continuation (Taichi doesn't support break in ti.func loops)
    active = 1

    # Path tracing loop
    for depth in range(MAX_DEPTH):
        if active == 1:
            # Intersect ray with scene
            hit_record = intersect_scene(origin, direction, T_MIN, T_MAX)

            if hit_record.hit == 0:
                # Ray escaped - add background contribution
                radiance += throughput * BACKGROUND_COLOR
                active = 0
            else:
                # Extract hit information
                hit_point = hit_record.point
                normal = hit_record.normal
                front_face = hit_record.front_face
                material_id = hit_record.material_id

                # Add emission from hit surface (if emitter)
                emission = _get_emission(material_id, hit_point, normal)
                radiance += throughput * emission

                # Scatter ray according to material
                scattered_direction, attenuation, did_scatter = _scatter_material(
                    material_id, direction, hit_point, normal, front_face
                )

                if did_scatter == 0:
                    # Ray was absorbed
                    active = 0
                else:
                    # Update throughput
                    throughput *= attenuation

                    # Russian roulette termination after minimum bounces
                    if depth >= MIN_BOUNCES_BEFORE_RR:
                        # Survival probability based on throughput luminance
                        luminance = (
                            0.2126 * throughput.x + 0.7152 * throughput.y + 0.0722 * throughput.z
                        )
                        rr_prob = tm.min(luminance, MAX_RR_PROBABILITY)

                        if ti.random(ti.f32) > rr_prob:
                            # Path terminated by Russian roulette
                            active = 0
                        else:
                            # Compensate for termination probability
                            throughput /= rr_prob

                    # Set up next ray (only if still active)
                    if active == 1:
                        origin = _offset_ray_origin(hit_point, normal, scattered_direction)
                        direction = scattered_direction

    return radiance


@ti.func
def render_sample_impl(pixel_i: ti.i32, pixel_j: ti.i32, width: ti.i32, height: ti.i32) -> vec3:
    """Render a single sample for a pixel.

    This is the main entry point for the path tracer, called once per sample.

    Args:
        pixel_i: Pixel x-coordinate (0 = left).
        pixel_j: Pixel y-coordinate (0 = bottom).
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        The estimated radiance (RGB) for this sample.
    """
    return trace_path(pixel_i, pixel_j, width, height)


# =============================================================================
# Rendering Kernels
# =============================================================================


@ti.kernel
def _render_one_spp(width: ti.i32, height: ti.i32):
    """Render one sample per pixel and accumulate.

    Traces one path through each pixel and progressively accumulates
    the result into the color buffer.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
    """
    for i, j in ti.ndrange(width, height):
        # Trace path for this pixel
        color = render_sample_impl(i, j, width, height)

        # Clamp negative values (numerical errors)
        color = tm.max(color, vec3(0.0, 0.0, 0.0))

        # Check for NaN/Inf and replace with zero
        for c in ti.static(range(3)):
            if tm.isnan(color[c]) or tm.isinf(color[c]):
                color[c] = 0.0

        # Progressive accumulation using running average
        _sample_count[i, j] += 1
        n = _sample_count[i, j]

        # Running average: avg_n = avg_{n-1} + (x_n - avg_{n-1}) / n
        _color_buffer[i, j] += (color - _color_buffer[i, j]) / ti.cast(n, ti.f32)


@ti.kernel
def _render_single_pixel(pixel_i: ti.i32, pixel_j: ti.i32, width: ti.i32, height: ti.i32) -> vec3:
    """Render a single sample for a specific pixel.

    Used for testing and debugging individual pixel rendering.

    Args:
        pixel_i: Pixel x-coordinate.
        pixel_j: Pixel y-coordinate.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        The rendered color for this sample.
    """
    return render_sample_impl(pixel_i, pixel_j, width, height)


# =============================================================================
# Public Rendering API
# =============================================================================


def render_sample(pixel_i: int, pixel_j: int) -> tuple[float, float, float]:
    """Render a single sample for a specific pixel.

    This is a Python-callable function for testing. For production rendering,
    use render_image() which processes all pixels in parallel.

    Args:
        pixel_i: Pixel x-coordinate (0 = left).
        pixel_j: Pixel y-coordinate (0 = bottom).

    Returns:
        Tuple of (R, G, B) color values.

    Raises:
        RuntimeError: If render target has not been set up.
    """
    _check_render_target_initialized()

    width, height = get_image_dimensions()
    color = _render_single_pixel(pixel_i, pixel_j, width, height)

    return (float(color[0]), float(color[1]), float(color[2]))


def render_image(num_samples: int = 1) -> None:
    """Render the image with the specified number of samples per pixel.

    Progressively accumulates samples into the color buffer. Can be called
    multiple times to add more samples for convergence.

    Args:
        num_samples: Number of samples to render per pixel.

    Raises:
        RuntimeError: If render target has not been set up.
    """
    _check_render_target_initialized()

    width, height = get_image_dimensions()

    for _ in range(num_samples):
        _render_one_spp(width, height)


def get_total_samples() -> int:
    """Get the total number of samples rendered so far.

    Returns the sample count from pixel (0, 0), which should be the same
    for all pixels after calling render_image().

    Returns:
        The number of samples per pixel.

    Raises:
        RuntimeError: If render target has not been set up.
    """
    _check_render_target_initialized()

    return int(_sample_count[0, 0])


def get_normalized_image_numpy():
    """Get the rendered image as a NumPy array.

    Returns the color buffer with values in [0, 1] range (clamped).
    The array shape is (height, width, 3) with dtype float32.

    Returns:
        NumPy array of shape (height, width, 3).

    Raises:
        RuntimeError: If render target has not been set up.
    """
    import numpy as np

    _check_render_target_initialized()

    width, height = get_image_dimensions()

    # Get raw image data (full buffer)
    full_image = _color_buffer.to_numpy()

    # Extract active region
    image = full_image[:width, :height, :]

    # Transpose from (width, height, 3) to (height, width, 3) for standard image format
    image = np.transpose(image, (1, 0, 2))

    # Flip vertically (Taichi uses bottom-left origin, images use top-left)
    image = np.flipud(image)

    # Clamp to [0, 1]
    image = np.clip(image, 0.0, 1.0)

    return image.astype(np.float32)


def save_image(filepath: str, gamma: float = 2.2) -> None:
    """Save the rendered image to a file.

    Applies gamma correction and saves as PNG or other format based on extension.

    Args:
        filepath: Path to save the image (e.g., "output.png").
        gamma: Gamma correction value. Default is 2.2 for sRGB.

    Raises:
        RuntimeError: If render target has not been set up.
    """
    import numpy as np
    from PIL import Image as PILImage

    image = get_normalized_image_numpy()

    # Apply gamma correction
    image = np.power(image, 1.0 / gamma)

    # Convert to 8-bit
    image_8bit = (image * 255).astype(np.uint8)

    # Save using PIL
    pil_image = PILImage.fromarray(image_8bit, mode="RGB")
    pil_image.save(filepath)
