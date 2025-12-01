"""
3D Fractal Parameter AI - Surface fractal dimension analysis.

This package provides tools for computing fractal dimensions of 3D surfaces
using cube-counting (3D box-counting) methods.
"""

from .box_counting_3d import (
    CubeCounter,
    compute_fractal_dimension_3d,
    FractalDimensionResult,
    analyze_mesh,
)
from .mesh_io import (
    load_mesh,
    TriangleMesh,
    BoundingBox3D,
)
from .surface_features import (
    extract_surface_features,
    find_connected_components,
    suggest_parameters,
    SurfaceFeatures,
    ConnectedComponent,
    classify_surface_type,
    print_feature_summary,
)

__version__ = "0.1.0"
__all__ = [
    # Box counting
    "CubeCounter",
    "compute_fractal_dimension_3d",
    "FractalDimensionResult",
    "analyze_mesh",
    # Mesh I/O
    "load_mesh",
    "TriangleMesh",
    "BoundingBox3D",
    # Surface features
    "extract_surface_features",
    "find_connected_components",
    "suggest_parameters",
    "SurfaceFeatures",
    "ConnectedComponent",
    "classify_surface_type",
    "print_feature_summary",
]
