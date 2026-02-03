# FRD: Cornell Box Raytracer with Three Spheres

**Created:** 2026-02-02
**Tier:** MEDIUM
**Beads Feature ID:** raytracer-fj3
**Triage Scores:** Complexity 6/10, Risk 4/10
**Status:** In Review

## Problem Statement

This is a greenfield raytracer project with established directory structure but no implementation. The goal is to create a foundational raytracing system that renders a classic Cornell box scene with three spheres, serving as the basis for future rendering experiments.

The Cornell box is a standard test scene in computer graphics, originally created at Cornell University to validate global illumination algorithms. It provides a controlled environment with known geometry and lighting that makes it easy to verify rendering correctness.

### Current State

- Project structure exists: `src/python/{core, geometry, materials, scene, camera, preview}`
- Dependencies configured: Taichi 1.7+, NumPy, Pillow, Matplotlib
- All source directories are empty -- no rendering code exists

### Limitations

1. No ray generation or intersection testing
2. No material system for light interaction
3. No scene representation for geometry
4. No integrator for light transport computation
5. No output mechanism for rendered images

### Impact

- **Rendering capability:** Without this foundation, no rendering is possible
- **Performance:** Taichi GPU acceleration enables real-time preview and fast convergence
- **Correctness:** Cornell box provides ground truth for validating future features

### Success Metrics

- Cornell box renders with correct geometry (5 walls + 3 spheres)
- Soft shadows from area light source
- Color bleeding visible (red/green walls reflecting onto spheres)
- Renders 512x512 at 64 spp in under 10 seconds on GPU
- Visual output matches reference Cornell box appearance

## Proposed Solution

### Overview

Implement a Monte Carlo path tracer using Taichi for GPU acceleration. The system will use a "one big kernel" architecture for optimal GPU performance, with vectorized data structures for geometry and materials.

### Mathematical Foundation

#### The Rendering Equation

The path tracer solves the rendering equation:

```
L_o(x, w_o) = L_e(x, w_o) + integral_Omega f_r(x, w_i, w_o) L_i(x, w_i) |cos(theta_i)| dw_i
```

Where:
- `L_o(x, w_o)` = outgoing radiance at point x in direction w_o
- `L_e(x, w_o)` = emitted radiance (for light sources)
- `f_r` = BRDF (bidirectional reflectance distribution function)
- `L_i(x, w_i)` = incoming radiance from direction w_i
- `theta_i` = angle between surface normal and incoming direction

#### Ray-Sphere Intersection

Using the robust quadratic formula from Ray Tracing Gems to avoid floating-point precision issues:

```
Given ray: P(t) = O + t*D
Sphere: |P - C|^2 = r^2

Substituting: |O + t*D - C|^2 = r^2
Let f = O - C (vector from center to origin)

t^2(D.D) + 2t(D.f) + (f.f - r^2) = 0

Standard form: at^2 + bt + c = 0
Where: a = D.D, b = 2(D.f), c = f.f - r^2

Discriminant: b^2 - 4ac
Roots: t = (-b +/- sqrt(discriminant)) / 2a
```

For numerical stability, use the reformulation:

```
q = -0.5 * (b + sign(b) * sqrt(discriminant))
t1 = q / a
t2 = c / q
```

#### Lambertian BRDF

For diffuse materials:

```
f_r = albedo / pi
```

PDF for cosine-weighted hemisphere sampling:

```
p(w_i) = cos(theta) / pi
```

#### Fresnel Equations (Schlick Approximation)

For reflective/refractive materials:

```
F(theta) = F_0 + (1 - F_0)(1 - cos(theta))^5

Where F_0 = ((n1 - n2) / (n1 + n2))^2
```

Typical values:
- Dielectrics (glass, plastic): F_0 ~ 0.04
- Metals: F_0 = material color (high values)

### Key Components

1. **Ray Generator (camera module)**
   - Pinhole camera model for simplicity
   - Generates primary rays from camera through pixel centers
   - Supports jittered sampling for anti-aliasing

2. **Geometry System (geometry module)**
   - Sphere primitive with robust intersection
   - Quad primitive for Cornell box walls
   - Hit record structure for intersection data

3. **Material System (materials module)**
   - Lambertian diffuse (walls, matte spheres)
   - Metal with controllable roughness
   - Dielectric (glass) with refraction

4. **Path Integrator (core module)**
   - Monte Carlo path tracing with Russian Roulette termination
   - Next Event Estimation (NEE) for direct light sampling
   - Maximum bounce depth limit (safety)

5. **Scene Manager (scene module)**
   - Cornell box scene definition
   - Vectorized storage for GPU efficiency
   - Area light representation

6. **Preview System (preview module)**
   - Progressive rendering display
   - Image output to PNG

## Technical Approach

### Taichi Implementation

#### Architecture Decision: One Big Kernel

Based on research into Taichi raytracing implementations, a consolidated kernel approach outperforms microkernel architectures on GPU. The main rendering kernel will handle ray generation, intersection, shading, and accumulation in a single pass per sample.

Reference: [Ray Tracing One Weekend in Taichi](https://github.com/bsavery/ray-tracing-one-weekend-taichi)

#### Field Layout

```python
import taichi as ti

ti.init(arch=ti.gpu)  # Prefer Metal on macOS, CUDA on Linux/Windows

# Image accumulation buffer
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(width, height))
sample_count = ti.field(dtype=ti.i32, shape=())

# Geometry: Vectorized sphere storage
sphere_centers = ti.Vector.field(3, dtype=ti.f32, shape=max_spheres)
sphere_radii = ti.field(dtype=ti.f32, shape=max_spheres)
sphere_materials = ti.field(dtype=ti.i32, shape=max_spheres)  # Material index

# Geometry: Vectorized quad storage (for walls)
quad_corners = ti.Vector.field(3, dtype=ti.f32, shape=(max_quads, 4))
quad_normals = ti.Vector.field(3, dtype=ti.f32, shape=max_quads)
quad_materials = ti.field(dtype=ti.i32, shape=max_quads)

# Materials: Vectorized storage
material_albedo = ti.Vector.field(3, dtype=ti.f32, shape=max_materials)
material_type = ti.field(dtype=ti.i32, shape=max_materials)  # 0=diffuse, 1=metal, 2=dielectric
material_param = ti.field(dtype=ti.f32, shape=max_materials)  # roughness or IOR

# Area light
light_center = ti.Vector.field(3, dtype=ti.f32, shape=())
light_size = ti.Vector.field(2, dtype=ti.f32, shape=())
light_emission = ti.Vector.field(3, dtype=ti.f32, shape=())
```

#### Main Render Kernel Structure

```python
@ti.kernel
def render_sample():
    for i, j in pixels:
        # Generate camera ray with jitter
        ray_origin, ray_dir = generate_camera_ray(i, j)

        # Trace path
        color = trace_path(ray_origin, ray_dir, max_depth=8)

        # Accumulate sample
        pixels[i, j] += color

@ti.func
def trace_path(origin: ti.math.vec3, direction: ti.math.vec3, max_depth: int) -> ti.math.vec3:
    throughput = ti.math.vec3(1.0, 1.0, 1.0)
    radiance = ti.math.vec3(0.0, 0.0, 0.0)

    ray_o = origin
    ray_d = direction

    for bounce in range(max_depth):
        # Find closest intersection
        hit, t, normal, material_id = intersect_scene(ray_o, ray_d)

        if not hit:
            break

        hit_point = ray_o + t * ray_d

        # Check if we hit light
        if is_light(hit_point):
            radiance += throughput * light_emission[None]
            break

        # Sample material and update path
        new_dir, pdf, brdf = sample_material(material_id, ray_d, normal)

        # Update throughput
        cos_theta = ti.abs(ti.math.dot(new_dir, normal))
        throughput *= brdf * cos_theta / pdf

        # Russian Roulette termination
        if bounce > 3:
            p = ti.max(throughput[0], ti.max(throughput[1], throughput[2]))
            if ti.random() > p:
                break
            throughput /= p

        ray_o = hit_point + 0.001 * normal  # Offset to avoid self-intersection
        ray_d = new_dir

    return radiance
```

#### Performance Considerations

- **Block dimension:** Use `ti.loop_config(block_dim=256)` for optimal GPU occupancy
- **Memory layout:** Dense fields for small, fixed-size data; avoid sparse structures initially
- **Atomic operations:** Use `+=` for pixel accumulation (Taichi handles atomics)
- **Self-intersection:** Offset hit points by small epsilon in normal direction

Reference: [Taichi Performance Tuning](https://docs.taichi-lang.cn/en/docs/performance/)

### Module Structure

```
src/python/
+-- core/
|   +-- __init__.py
|   +-- integrator.py      # Path tracing logic
|   +-- renderer.py        # Main render loop, sample accumulation
+-- geometry/
|   +-- __init__.py
|   +-- primitives.py      # Sphere, Quad intersection functions
|   +-- hit.py             # Hit record struct
+-- materials/
|   +-- __init__.py
|   +-- material.py        # Material types, BRDF sampling
+-- scene/
|   +-- __init__.py
|   +-- scene.py           # Scene definition, geometry storage
|   +-- cornell_box.py     # Cornell box scene setup
+-- camera/
|   +-- __init__.py
|   +-- camera.py          # Pinhole camera, ray generation
+-- preview/
|   +-- __init__.py
|   +-- display.py         # Matplotlib preview, PNG export
```

### Integration with Existing Code

This is a greenfield implementation. All modules will be created from scratch, with clear interfaces between components:

- **Camera** produces rays
- **Geometry** tests ray intersections
- **Materials** sample scattered directions
- **Integrator** orchestrates the path tracing algorithm
- **Scene** provides geometry and material data to the integrator
- **Preview** displays and exports results

## Cornell Box Scene Specification

### Geometry

Based on the original Cornell box dimensions (scaled to unit cube for simplicity):

```
# Walls (quads)
- Left wall (red):   x = -1, from (-1,-1,-1) to (-1,1,1)
- Right wall (green): x = 1, from (1,-1,-1) to (1,1,1)
- Back wall (white):  z = -1, from (-1,-1,-1) to (1,1,-1)
- Floor (white):      y = -1, from (-1,-1,-1) to (1,-1,1)
- Ceiling (white):    y = 1, from (-1,1,-1) to (1,1,1)

# Spheres
- Large diffuse white sphere: center (0.4, -0.6, -0.2), radius 0.4
- Medium metal sphere: center (-0.4, -0.7, 0.3), radius 0.3
- Small glass sphere: center (0.0, -0.8, 0.5), radius 0.2

# Area light (on ceiling)
- Center: (0, 0.99, 0)
- Size: 0.5 x 0.5
- Emission: (15, 15, 15) or similar bright white
```

### Materials

| Material | Type | Albedo | Parameter |
|----------|------|--------|-----------|
| Red | Lambertian | (0.65, 0.05, 0.05) | - |
| Green | Lambertian | (0.12, 0.45, 0.15) | - |
| White | Lambertian | (0.73, 0.73, 0.73) | - |
| Metal | Metal | (0.8, 0.8, 0.9) | roughness=0.1 |
| Glass | Dielectric | (1.0, 1.0, 1.0) | IOR=1.5 |

### Camera

- Position: (0, 0, 3.5)
- Look at: (0, 0, 0)
- Up: (0, 1, 0)
- Field of view: 40 degrees
- Aspect ratio: 1:1

## Testing Strategy

### Unit Tests

1. **Ray-sphere intersection**
   - Ray hitting sphere center
   - Ray tangent to sphere
   - Ray missing sphere
   - Ray originating inside sphere

2. **Material sampling**
   - Lambertian samples in hemisphere
   - Metal reflection direction
   - Dielectric refraction with correct Fresnel

3. **Mathematical properties**
   - BRDF energy conservation (integral over hemisphere <= 1)
   - Cosine-weighted sampling PDF integrates to 1

### Visual Regression Tests

1. **Cornell box reference render**
   - Compare to known-correct reference at 1024 spp
   - RMSE < 0.02 for diffuse-only scene

2. **Material tests**
   - Single sphere per material type on gray background
   - Verify specular highlights and refraction

### Performance Benchmarks

1. **Render time**
   - 512x512 @ 64 spp < 10 seconds on integrated GPU
   - 512x512 @ 1024 spp < 120 seconds

2. **Convergence**
   - Variance decreases with sqrt(samples)
   - No fireflies (runaway bright pixels)

## Acceptance Criteria

- [ ] Camera generates correct primary rays through each pixel
- [ ] Ray-sphere intersection is robust (no z-fighting or missed hits)
- [ ] Ray-quad intersection works for all Cornell box walls
- [ ] Lambertian diffuse material scatters correctly
- [ ] Metal material reflects with controllable roughness
- [ ] Dielectric material refracts with correct Fresnel
- [ ] Path tracer accumulates samples progressively
- [ ] Russian Roulette terminates paths without bias
- [ ] Cornell box renders with visible color bleeding
- [ ] Soft shadows appear under area light
- [ ] Three spheres (diffuse, metal, glass) render correctly
- [ ] Output can be saved to PNG file
- [ ] Reference image matches expected output (visual inspection)
- [ ] Performance meets target (512x512 @ 64 spp < 10s)

## Open Questions

- [ ] Should we implement BVH acceleration for this initial version, or defer to a later feature? (Recommendation: defer -- 3 spheres + 5 quads don't need acceleration)
- [ ] Should the preview use Matplotlib's interactive mode or a separate window? (Recommendation: Matplotlib for simplicity, defer GUI to later)
- [ ] Should we support multiple area lights or just one? (Recommendation: one for now)

## References

- [PBRT 3rd Edition](https://www.pbr-book.org/) - Managing Rounding Error, Path Tracing
- [Ray Tracing Gems - Precision Improvements for Ray/Sphere Intersection](https://link.springer.com/chapter/10.1007/978-1-4842-4427-2_7) - NVIDIA Research
- [Ray Tracing in One Weekend (Taichi)](https://github.com/bsavery/ray-tracing-one-weekend-taichi) - GPU architecture patterns
- [Taichi Performance Tuning](https://docs.taichi-lang.cn/en/docs/performance/) - GPU kernel optimization
- [Crash Course in BRDF Implementation](https://boksajak.github.io/files/CrashCourseBRDF.pdf) - Material models
- [Cornell Box Reference](https://www.graphics.cornell.edu/online/box/) - Original scene specification

---

## Implementation Tasks

| # | Title | Agent | Depends On | Description |
|---|-------|-------|------------|-------------|
| 1 | Implement core vector math and ray structure | renderer-core | - | Create basic ti.Vector operations, Ray dataclass |
| 2 | Implement pinhole camera with ray generation | renderer-core | 1 | Camera class with generate_ray() using jittered sampling |
| 3 | Implement sphere intersection | renderer-core | 1 | Robust ray-sphere intersection with hit record |
| 4 | Implement quad intersection | renderer-core | 1 | Ray-quad intersection for box walls |
| 5 | Implement Lambertian material | material-system | 1 | Diffuse BRDF with cosine-weighted sampling |
| 6 | Implement metal material | material-system | 5 | Specular reflection with roughness |
| 7 | Implement dielectric material | material-system | 5 | Refraction with Fresnel/Schlick |
| 8 | Implement scene manager | scene-system | 3, 4 | Vectorized geometry storage, material assignment |
| 9 | Implement Cornell box scene | scene-system | 8 | Specific scene definition with 5 walls + 3 spheres |
| 10 | Implement path integrator | renderer-core | 5, 6, 7, 8 | Main rendering kernel with Russian Roulette |
| 11 | Implement progressive renderer | renderer-core | 10 | Sample accumulation, tonemapping |
| 12 | Implement preview/export | preview-system | 11 | Matplotlib display, PNG export |
| 13 | Integration test with Cornell box | renderer-core | 9, 12 | End-to-end rendering verification |

---

## Next Agent to Invoke

**Agent:** `frd-refiner`

**Context to provide:**
- Feature slug: `cornell-box-raytracer`
- Tier: MEDIUM
- Beads feature ID: raytracer-fj3
- FRD location: `.claude_docs/features/cornell-box-raytracer/frd.md`
- This is a greenfield implementation establishing core raytracing architecture
- Key research sources have been incorporated (Taichi patterns, numerical stability, BRDF models)

**Beads update (for orchestrator):**
`bd update raytracer-fj3 --notes="FRD created at .claude_docs/features/cornell-box-raytracer/frd.md"`

**After that agent completes:**
The FRD Refiner will validate technical completeness, check for gaps in the specification, and potentially add refinements. Once refinement is complete, implementation can proceed with the task breakdown, starting with the core ray/vector structures.
