# 3D-FractalParameterAI Project Context

## Project Overview
3D extension of FractalParameterAI for computing fractal dimensions of 3D surfaces from OpenFOAM RT (Rayleigh-Taylor) instability simulations.

## Repository Structure
```
fractal3d/                 # Core library
├── mesh_io.py            # TriangleMesh, BoundingBox3D, VTK/STL readers
├── box_counting_3d.py    # Cube counting, Akenine-Möller intersection test
├── surface_features.py   # Connected components, curvature, classification
├── fast_counting.py      # Numba JIT acceleration (20-450x speedup)
└── spatial_hash.py       # Spatial indexing

analyze_rt_interfaces.py  # Main analysis script for temporal evolution
```

## Related Projects
- **FractalParameterAI** (sibling directory): 2D version with AI parameter optimization
- **FastFractalAnalyzer**: High-performance fractal computation library

## Key Technical Decisions
- Cubes (not rectangles) for isotropic fractal scaling
- Numba JIT for performance (20-450x speedup over pure Python)
- Triangle-centric cube counting approach
- Akenine-Möller SAT algorithm for triangle-cube intersection

## OpenFOAM/ParaView Workflow
- Case directory: `~/OpenFOAM/douglass-v2506/run/RT_Dalziel_3D_isoFoam`
- Domain: 0.4m x 0.5m x 0.2m
- Export script: `~/export_isosurfaces.py` (uses VTK writers directly, not ParaView SaveData)
- ParaView 6.0.1 installed on server

## Validated Results (Dalziel 3D RT)
| Time | D | Components | Classification |
|------|---|------------|----------------|
| 1.0 | 1.91 | 1 | smooth |
| 5.0 | 2.27 | 91 | highly_fragmented |
| 10.0 | 2.36 | 216 | turbulent_mixing |
| 15.0 | 2.49 | 408 | turbulent_mixing |
| 20.0 | 2.65 | 414 | turbulent_mixing |

## Surface Classification Types
- `smooth`: D ≈ 2.0, low complexity
- `moderate`: intermediate roughness
- `complex`: high roughness/compactness
- `highly_fragmented`: many disconnected components
- `turbulent_mixing`: late-stage RT, high fragmentation + roughness

## Common Commands
```bash
# Analyze all timesteps
python analyze_rt_interfaces.py /path/to/isosurfaces/ -o results.csv --plot evolution.png

# Single file analysis
python analyze_rt_interfaces.py interface_t5.0000.vtp

# Time range subset
python analyze_rt_interfaces.py /path/to/isosurfaces/ --t-min 5 --t-max 15
```

## Next Steps
1. Verify OpenFOAM parameters match 2D case
2. Run full 100-timestep analysis on server
3. Compare 2D vs 3D fractal dimension evolution
4. Implement AI parameter optimization (port from 2D)

## Server Access
- Claude Code installed on remote server via npm
- Connect via SSH, run `claude` in project directory
