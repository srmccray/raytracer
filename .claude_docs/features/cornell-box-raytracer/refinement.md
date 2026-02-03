# Refinement Notes: Cornell Box Raytracer with Three Spheres

**Refined:** 2026-02-02
**FRD Location:** `/Users/stephen/Projects/raytracer/.claude_docs/features/cornell-box-raytracer/frd.md`
**Beads Feature ID:** raytracer-fj3

## Codebase Alignment

### Verified Assumptions

| FRD Claim | Validation | Evidence |
|-----------|------------|----------|
| Project structure exists with empty src directories | Confirmed | `src/python/{core, geometry, materials, scene, camera, preview}` directories exist and are empty |
| Taichi 1.7+ dependency | Confirmed | `pyproject.toml:7` specifies `taichi>=1.7.0` |
| NumPy dependency | Confirmed | `pyproject.toml:8` specifies `numpy>=1.26.0` |
| Pillow dependency | Confirmed | `pyproject.toml:9` specifies `Pillow>=10.0.0` |
| Matplotlib dependency | Confirmed | `pyproject.toml:10` specifies `matplotlib>=3.8.0` |
| Python 3.11+ requirement | Confirmed | `pyproject.toml:5` specifies `requires-python = ">=3.11"` |
| Dev dependencies include pytest | Confirmed | `pyproject.toml:14` includes `pytest>=8.0.0` |
| Test directories prepared | Confirmed | `tests/reference/` and `tests/scenes/` exist (empty) |

### Corrections Needed

- **None identified** - FRD accurately reflects the greenfield state

### Minor Observations

1. The `scenes/` directory exists at project root (empty) - could be used for scene definition files if JSON/YAML scene format is added later
2. `experiments/` directory exists - good for prototyping before integrating into main modules
3. Project uses ruff for linting and mypy for type checking (strict mode) - code must be type-annotated

## Key Files

### Will Create

| File | Purpose |
|------|---------|
| `src/python/core/__init__.py` | Core module exports |
| `src/python/core/integrator.py` | Path tracing logic with main render kernel |
| `src/python/core/renderer.py` | Main render loop, sample accumulation |
| `src/python/geometry/__init__.py` | Geometry module exports |
| `src/python/geometry/primitives.py` | Sphere, Quad intersection functions |
| `src/python/geometry/hit.py` | Hit record struct/dataclass |
| `src/python/materials/__init__.py` | Materials module exports |
| `src/python/materials/material.py` | Material types, BRDF sampling |
| `src/python/scene/__init__.py` | Scene module exports |
| `src/python/scene/scene.py` | Scene definition, vectorized geometry storage |
| `src/python/scene/cornell_box.py` | Cornell box scene setup |
| `src/python/camera/__init__.py` | Camera module exports |
| `src/python/camera/camera.py` | Pinhole camera, ray generation |
| `src/python/preview/__init__.py` | Preview module exports |
| `src/python/preview/display.py` | Matplotlib preview, PNG export |

### Will Create (Tests)

| File | Purpose |
|------|---------|
| `tests/test_geometry.py` | Ray-sphere, ray-quad intersection tests |
| `tests/test_materials.py` | BRDF sampling validation |
| `tests/test_camera.py` | Camera ray generation tests |
| `tests/test_integration.py` | End-to-end render tests |

### Reference (read-only)

| File | Purpose |
|------|---------|
| `pyproject.toml` | Dependency versions, tool configuration |
| `AGENTS.md` | Workflow guidance for beads integration |

## Taichi-Specific Notes

### Field Layout Validation
- **Compatible** - FRD proposes standard dense field layouts that work well with Taichi 1.7+
- `ti.Vector.field(3, dtype=ti.f32)` for pixel buffer and geometry data is optimal
- Vectorized storage for spheres/quads/materials allows efficient GPU iteration

### Kernel Integration
- "One big kernel" architecture is confirmed best practice for Taichi GPU path tracing
- `@ti.kernel` for main render loop, `@ti.func` for intersection/sampling helpers
- No existing kernels to integrate with (greenfield)

### Memory Impact
- Estimated GPU memory for 512x512:
  - Pixel buffer: 512 * 512 * 3 * 4 bytes = 3 MB
  - Geometry fields (max 10 spheres, 10 quads, 10 materials): < 1 KB
  - Total: < 5 MB - negligible for any modern GPU

### Platform Considerations
- macOS: Will use Metal backend (`ti.init(arch=ti.metal)` or `ti.gpu`)
- Linux/Windows: Will use CUDA if available, fall back to CPU
- FRD should note explicit backend selection for testing reproducibility

## Quality Gate Alignment

The project has strict tooling configured:

1. **mypy strict mode** - All code must have type annotations
   - Taichi kernels and funcs need careful typing (use `ti.types` module)
   - Consider `# type: ignore` comments where Taichi's dynamic types conflict

2. **ruff linting** - Rules: E, F, W, I, N, UP
   - Import sorting (I) will be enforced
   - Naming conventions (N) must follow PEP 8

3. **pytest** - Test discovery in `src/python` and `tests`
   - Test files: `*_test.py` or `test_*.py`

## Suggested Tasks (for beads)

| # | Title | Agent | Depends On | Description |
|---|-------|-------|------------|-------------|
| 1 | Create module structure with __init__.py files | renderer-core | - | Set up all Python package structure with empty __init__.py files and docstrings |
| 2 | Implement ray structure and vector utilities | renderer-core | 1 | Create Ray dataclass/struct with origin and direction, add ti.math vector helpers |
| 3 | Implement pinhole camera with ray generation | renderer-core | 2 | Camera class with generate_ray(u, v) supporting jittered sampling |
| 4 | Implement sphere intersection | renderer-core | 2 | Robust ray-sphere intersection using numerically stable quadratic formula |
| 5 | Implement quad intersection | renderer-core | 2 | Ray-quad intersection for axis-aligned and arbitrary quads |
| 6 | Implement hit record and scene intersection | scene-system | 4, 5 | HitRecord struct, scene-level intersection testing all primitives |
| 7 | Implement Lambertian material | material-system | 2 | Diffuse BRDF with cosine-weighted hemisphere sampling |
| 8 | Implement metal material | material-system | 7 | Specular reflection with roughness parameter |
| 9 | Implement dielectric material | material-system | 7 | Glass with refraction and Schlick Fresnel approximation |
| 10 | Implement scene manager with vectorized storage | scene-system | 6 | Taichi fields for geometry and materials, add/query interface |
| 11 | Implement Cornell box scene definition | scene-system | 10 | Hardcoded Cornell box with 5 walls, 3 spheres, area light |
| 12 | Implement path integrator kernel | renderer-core | 7, 8, 9, 10 | Main @ti.kernel with path tracing, Russian Roulette, NEE |
| 13 | Implement progressive renderer | renderer-core | 12 | Sample accumulation loop, tonemapping, convergence tracking |
| 14 | Implement preview and PNG export | preview-system | 13 | Matplotlib display with progressive updates, Pillow PNG save |
| 15 | Integration test: render Cornell box | renderer-core | 11, 14 | End-to-end test rendering full scene, save reference image |

## Blockers / Concerns

1. **No reference images yet** - `tests/reference/` is empty. First successful render should be saved as ground truth for regression testing.

2. **Taichi type annotations** - mypy strict mode may conflict with Taichi's dynamic typing in kernels. May need strategic `# type: ignore` or wrapper patterns.

3. **CI/CD not configured** - No GitHub Actions or similar. Quality gates exist but are manual. Consider adding automated testing.

## Ready for Implementation

- [x] FRD assumptions validated against codebase
- [x] No major blockers identified
- [x] Directory structure confirmed ready
- [x] Dependencies properly specified
- [x] Task breakdown provided
- [ ] Reference images (will be created during implementation)

---

## Next Agent to Invoke

**Agent:** `renderer-core`

**Context to provide:**
- Feature slug: `cornell-box-raytracer`
- Tier: MEDIUM
- Beads feature ID: raytracer-fj3
- Refinement summary: Greenfield project validated. All directories exist and are empty. Dependencies confirmed. Ready for implementation starting with module structure and ray/camera primitives.
- Key files: Start with `src/python/core/`, `src/python/geometry/`, `src/python/camera/`

**Beads update (for orchestrator):**
`bd update raytracer-fj3 --notes="Refinement complete. Codebase validated. Ready for implementation."`

**After that agent completes:**
The renderer-core agent will implement the foundational ray structure, camera, and geometry primitives. Subsequent agents (material-system, scene-system, preview-system) can work in parallel on their respective modules once dependencies are met.
