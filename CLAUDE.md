# 3D-FractalParameterAI - Session Summary

## Project Purpose
Extend the 2D FractalParameterAI framework to compute fractal dimensions of 3D surfaces extracted from OpenFOAM Rayleigh-Taylor (RT) instability simulations.

---

## What Was Built This Session

### 1. Repository Setup
- GitHub repo: https://github.com/rwdlnk/3D-FractalParameterAI
- MIT License (rwdlnk), README, CONTRIBUTING, CODE_OF_CONDUCT
- GitHub issue/PR templates

### 2. Core Library (`fractal3d/`)
| Module | Purpose |
|--------|---------|
| `mesh_io.py` | `TriangleMesh`, `BoundingBox3D` classes, VTK/STL file readers |
| `box_counting_3d.py` | 3D cube counting with Akenine-Möller triangle-cube intersection (SAT algorithm) |
| `surface_features.py` | Connected component detection, curvature estimation, complexity metrics, surface classification |
| `fast_counting.py` | Numba JIT acceleration achieving **20-450x speedup** |
| `spatial_hash.py` | Spatial indexing for large meshes |

### 3. Analysis Script
`analyze_rt_interfaces.py` - Command-line tool for batch analysis:
```bash
python analyze_rt_interfaces.py /path/to/isosurfaces/ -o results.csv --plot evolution.png
python analyze_rt_interfaces.py /path/to/isosurfaces/ --t-min 5 --t-max 15
python analyze_rt_interfaces.py interface_t5.0000.vtp
```

---

## Server Setup

### Claude Code Installation
Installed on remote server via:
```bash
npm install -g @anthropic-ai/claude-code
```
User can run `claude` in project directory on server.

### ParaView 6.0.1 Isosurface Export
Created `~/export_isosurfaces.py` that:
- Reads OpenFOAM case with `alpha.water` field
- Extracts 0.5 isosurface (interface between fluids)
- Uses VTK writers directly (ParaView's SaveData is broken)
- Handles multi-block datasets from OpenFOAM
- Exports 100 timesteps to VTP format

---

## OpenFOAM Simulation Details
- **Case**: `~/OpenFOAM/douglass-v2506/run/RT_Dalziel_3D_isoFoam`
- **Domain**: 0.4m × 0.5m × 0.2m (x, y, z)
- **Timestep**: 0.2s intervals, t=0.2 to t=20.0 (100 files)
- **Output**: `isosurfaces/interface_t*.vtp`

---

## Validated Results

First successful 3D RT fractal analysis showing physical evolution:

| Time | Triangles | Components | **D** | R² | Classification |
|------|-----------|------------|-------|-----|----------------|
| 1.0 | 9,632 | 1 | **1.91** | 0.996 | smooth |
| 5.0 | 64,050 | 91 | **2.27** | 0.999 | highly_fragmented |
| 10.0 | 198,832 | 216 | **2.36** | 0.999 | turbulent_mixing |
| 15.0 | 375,218 | 408 | **2.49** | 0.997 | turbulent_mixing |
| 20.0 | 541,881 | 414 | **2.65** | 0.997 | turbulent_mixing |

**Key Finding**: D evolves from ~2.0 (flat surface) to ~2.65 (turbulent mixing), correctly tracking RT instability development. The t=5 interface was visually confirmed to be complex (not flat as initially assumed).

---

## Key Technical Decisions

1. **Cubes not rectangles**: Isotropic scaling required for valid fractal dimension
2. **Numba over spatial hash**: JIT compilation gave 20-450x speedup; spatial hash less impactful for this use case
3. **Triangle-centric counting**: More efficient than cube-centric for surface meshes
4. **VTK writers directly**: ParaView's SaveData broken in both 5.10.1 and 6.0.1

---

## Surface Classification System

| Type | Characteristics | Parameter Adjustment |
|------|-----------------|---------------------|
| `smooth` | D ≈ 2.0, low complexity, single component | Coarser analysis |
| `moderate` | Intermediate roughness | Default parameters |
| `complex` | High roughness or compactness | Finer resolution |
| `highly_fragmented` | Many disconnected components | Smaller cubes |
| `turbulent_mixing` | Late-stage RT, high fragmentation + roughness | Finest resolution |

---

## Related Projects
- **FractalParameterAI** (`../FractalParameterAI/`): 2D version with AI parameter optimization
- **FastFractalAnalyzer**: High-performance fractal computation library

---

## Next Steps
1. **Verify OpenFOAM parameters** match 2D Dalziel case
2. **Clone repo on server**: `git clone https://github.com/rwdlnk/3D-FractalParameterAI.git`
3. **Run full 100-timestep analysis** on server
4. **Compare 2D vs 3D** fractal dimension evolution
5. **Port AI parameter optimization** from 2D FractalParameterAI

---

## Files on Server
- Export script: `~/export_isosurfaces.py`
- Isosurfaces: `~/OpenFOAM/douglass-v2506/run/RT_Dalziel_3D_isoFoam/isosurfaces/`
- 100 VTP files, 485MB total (192KB early → 12MB late)

---

## Session Date
December 4, 2025
