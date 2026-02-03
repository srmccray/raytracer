"""Python implementation of the Taichi-based raytracer.

This package provides GPU-accelerated ray tracing using Taichi, with support for:
- Path tracing with multiple importance sampling
- Various material models (Lambertian, metal, dielectric)
- Geometric primitives (spheres, quads)
- Progressive rendering with accumulation

Subpackages:
    core: Ray generation, vector utilities, integrators, and rendering loop
    geometry: Shape primitives and intersection algorithms
    materials: BRDF/BSDF material models
    scene: Scene management and hit record structures
    camera: Camera models with ray generation
    preview: Output rendering and preview utilities
"""

__version__ = "0.1.0"
