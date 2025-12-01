"""
3D Box (Cube) Counting Algorithm for Surface Fractal Dimension.

This module implements cube-counting methods for computing the fractal
dimension of 3D surfaces. For a surface embedded in 3D space, the
fractal dimension D satisfies: N(δ) ~ δ^(-D), where N(δ) is the number
of cubes of size δ that intersect the surface.

For smooth surfaces: D = 2.0
For fractal surfaces: 2.0 < D < 3.0
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
from scipy import stats
from .mesh_io import TriangleMesh, BoundingBox3D


@dataclass
class BoxCountResult:
    """Result of a single box count at a specific scale."""
    delta: float  # Cube size
    n_boxes: int  # Number of cubes intersecting surface
    grid_dims: Tuple[int, int, int]  # Grid dimensions (nx, ny, nz)


@dataclass
class FractalDimensionResult:
    """Result of fractal dimension calculation."""
    dimension: float  # Estimated fractal dimension
    r_squared: float  # R² of log-log regression
    std_error: float  # Standard error of dimension estimate
    intercept: float  # Intercept of log-log regression
    deltas: List[float]  # Cube sizes used
    n_boxes: List[int]  # Box counts at each scale
    log_inv_delta: np.ndarray  # log(1/δ) values
    log_n_boxes: np.ndarray  # log(N) values


def triangle_cube_intersection(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray,
                                cube_min: np.ndarray, cube_max: np.ndarray) -> bool:
    """
    Test if a triangle intersects an axis-aligned cube.

    Uses the Separating Axis Theorem (SAT) with optimizations.
    Based on Akenine-Möller's triangle-box intersection algorithm.

    Args:
        v0, v1, v2: Triangle vertices, each shape (3,)
        cube_min: Minimum corner of cube (x, y, z)
        cube_max: Maximum corner of cube (x, y, z)

    Returns:
        True if triangle intersects cube, False otherwise
    """
    # Translate so cube center is at origin
    cube_center = (cube_min + cube_max) / 2
    half_size = (cube_max - cube_min) / 2

    # Translate triangle to cube-centered coordinates
    t0 = v0 - cube_center
    t1 = v1 - cube_center
    t2 = v2 - cube_center

    # Triangle edges
    e0 = t1 - t0
    e1 = t2 - t1
    e2 = t0 - t2

    # Test 1: AABB overlap test (bounding box of triangle vs cube)
    tri_min = np.minimum(np.minimum(t0, t1), t2)
    tri_max = np.maximum(np.maximum(t0, t1), t2)

    if np.any(tri_min > half_size) or np.any(tri_max < -half_size):
        return False

    # Test 2: Triangle normal axis
    normal = np.cross(e0, e1)
    d = -np.dot(normal, t0)

    # Project cube onto triangle normal
    r = np.sum(half_size * np.abs(normal))
    if abs(d) > r:
        return False

    # Test 3: 9 cross product axes (3 edges × 3 cube axes)
    # These are the most expensive tests, so we do them last

    axes = [
        # Edge 0 × cube axes
        np.array([0, -e0[2], e0[1]]),  # e0 × X
        np.array([e0[2], 0, -e0[0]]),  # e0 × Y
        np.array([-e0[1], e0[0], 0]),  # e0 × Z
        # Edge 1 × cube axes
        np.array([0, -e1[2], e1[1]]),  # e1 × X
        np.array([e1[2], 0, -e1[0]]),  # e1 × Y
        np.array([-e1[1], e1[0], 0]),  # e1 × Z
        # Edge 2 × cube axes
        np.array([0, -e2[2], e2[1]]),  # e2 × X
        np.array([e2[2], 0, -e2[0]]),  # e2 × Y
        np.array([-e2[1], e2[0], 0]),  # e2 × Z
    ]

    for axis in axes:
        # Skip degenerate axes
        axis_len_sq = np.dot(axis, axis)
        if axis_len_sq < 1e-12:
            continue

        # Project triangle vertices onto axis
        p0 = np.dot(t0, axis)
        p1 = np.dot(t1, axis)
        p2 = np.dot(t2, axis)

        # Project cube onto axis (half-extent along axis)
        r = np.sum(half_size * np.abs(axis))

        # Check for separation
        if min(p0, p1, p2) > r or max(p0, p1, p2) < -r:
            return False

    # No separating axis found - triangle and cube intersect
    return True


def triangle_cube_intersection_vectorized(triangles: np.ndarray,
                                           cube_min: np.ndarray,
                                           cube_max: np.ndarray) -> np.ndarray:
    """
    Vectorized triangle-cube intersection test for multiple triangles.

    Args:
        triangles: Array of shape (N, 3, 3) - N triangles, 3 vertices each, 3D coords
        cube_min: Minimum corner of cube (x, y, z)
        cube_max: Maximum corner of cube (x, y, z)

    Returns:
        Boolean array of shape (N,) indicating intersection for each triangle
    """
    n_tris = len(triangles)
    if n_tris == 0:
        return np.array([], dtype=bool)

    # Translate so cube center is at origin
    cube_center = (cube_min + cube_max) / 2
    half_size = (cube_max - cube_min) / 2

    # Translate triangles: (N, 3, 3)
    t = triangles - cube_center

    # Quick AABB rejection
    tri_min = t.min(axis=1)  # (N, 3)
    tri_max = t.max(axis=1)  # (N, 3)

    aabb_reject = np.any(tri_min > half_size, axis=1) | np.any(tri_max < -half_size, axis=1)

    # For triangles that pass AABB test, do full SAT test
    result = np.zeros(n_tris, dtype=bool)
    candidates = np.where(~aabb_reject)[0]

    for i in candidates:
        if triangle_cube_intersection(triangles[i, 0], triangles[i, 1],
                                       triangles[i, 2], cube_min, cube_max):
            result[i] = True

    return result


class CubeCounter:
    """
    Counts cubes intersecting a triangulated surface mesh.

    Uses spatial hashing for efficiency with large meshes.
    """

    def __init__(self, mesh: TriangleMesh):
        """
        Initialize counter with a triangle mesh.

        Args:
            mesh: TriangleMesh object
        """
        self.mesh = mesh
        self._triangle_vertices = mesh.get_all_triangle_vertices()
        self._triangle_bboxes = self._compute_triangle_bboxes()

    def _compute_triangle_bboxes(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute axis-aligned bounding boxes for all triangles."""
        tri_min = self._triangle_vertices.min(axis=1)  # (N, 3)
        tri_max = self._triangle_vertices.max(axis=1)  # (N, 3)
        return tri_min, tri_max

    def count_cubes(self, delta: float,
                    domain: Optional[BoundingBox3D] = None) -> BoxCountResult:
        """
        Count cubes of size delta that intersect the mesh surface.

        Args:
            delta: Cube side length
            domain: Optional custom domain (defaults to mesh bounding box)

        Returns:
            BoxCountResult with count and grid information
        """
        if domain is None:
            domain = self.mesh.bbox

        # Compute grid dimensions
        nx = int(np.ceil(domain.width / delta))
        ny = int(np.ceil(domain.height / delta))
        nz = int(np.ceil(domain.depth / delta))

        # Ensure at least 1 cell in each dimension
        nx = max(1, nx)
        ny = max(1, ny)
        nz = max(1, nz)

        # Set of occupied cube indices
        occupied = set()

        # Triangle bounding boxes
        tri_min, tri_max = self._triangle_bboxes

        # For each triangle, find candidate cubes and test intersection
        for t_idx in range(self.mesh.n_triangles):
            # Get triangle's bounding box indices
            t_min = tri_min[t_idx]
            t_max = tri_max[t_idx]

            # Cube index range for this triangle
            i_min = max(0, int((t_min[0] - domain.min_x) / delta))
            i_max = min(nx - 1, int((t_max[0] - domain.min_x) / delta))
            j_min = max(0, int((t_min[1] - domain.min_y) / delta))
            j_max = min(ny - 1, int((t_max[1] - domain.min_y) / delta))
            k_min = max(0, int((t_min[2] - domain.min_z) / delta))
            k_max = min(nz - 1, int((t_max[2] - domain.min_z) / delta))

            # Get triangle vertices
            v0 = self._triangle_vertices[t_idx, 0]
            v1 = self._triangle_vertices[t_idx, 1]
            v2 = self._triangle_vertices[t_idx, 2]

            # Test each candidate cube
            for i in range(i_min, i_max + 1):
                for j in range(j_min, j_max + 1):
                    for k in range(k_min, k_max + 1):
                        # Skip if already occupied
                        if (i, j, k) in occupied:
                            continue

                        # Compute cube bounds
                        cube_min = np.array([
                            domain.min_x + i * delta,
                            domain.min_y + j * delta,
                            domain.min_z + k * delta
                        ])
                        cube_max = cube_min + delta

                        # Test intersection
                        if triangle_cube_intersection(v0, v1, v2, cube_min, cube_max):
                            occupied.add((i, j, k))

        return BoxCountResult(
            delta=delta,
            n_boxes=len(occupied),
            grid_dims=(nx, ny, nz)
        )


def compute_fractal_dimension_3d(mesh: TriangleMesh,
                                  initial_delta: Optional[float] = None,
                                  delta_factor: float = 1.5,
                                  num_steps: int = 15,
                                  min_delta: Optional[float] = None,
                                  max_delta: Optional[float] = None) -> FractalDimensionResult:
    """
    Compute fractal dimension of a 3D surface using cube counting.

    The fractal dimension D is estimated from the slope of log(N) vs log(1/δ),
    where N(δ) is the number of cubes of size δ that intersect the surface.

    For a smooth surface: D ≈ 2.0
    For a fractal surface: 2.0 < D < 3.0

    Args:
        mesh: TriangleMesh to analyze
        initial_delta: Starting cube size (default: 1/10 of characteristic length)
        delta_factor: Factor to reduce delta by each step (default: 1.5)
        num_steps: Number of scales to analyze (default: 15)
        min_delta: Minimum cube size (default: characteristic_length / 500)
        max_delta: Maximum cube size (default: characteristic_length)

    Returns:
        FractalDimensionResult with dimension estimate and statistics
    """
    char_len = mesh.bbox.characteristic_length

    # Set defaults based on characteristic length
    if initial_delta is None:
        initial_delta = char_len / 10

    if min_delta is None:
        min_delta = char_len / 500

    if max_delta is None:
        max_delta = char_len

    # Generate delta sequence (geometric progression)
    deltas = []
    delta = initial_delta
    for _ in range(num_steps):
        if delta < min_delta:
            break
        if delta > max_delta:
            delta = delta / delta_factor
            continue
        deltas.append(delta)
        delta = delta / delta_factor

    if len(deltas) < 3:
        raise ValueError(f"Insufficient scale range: only {len(deltas)} valid delta values")

    # Count cubes at each scale
    counter = CubeCounter(mesh)
    n_boxes = []

    for delta in deltas:
        result = counter.count_cubes(delta)
        n_boxes.append(result.n_boxes)
        print(f"  δ = {delta:.6f}: {result.n_boxes} cubes "
              f"(grid: {result.grid_dims[0]}×{result.grid_dims[1]}×{result.grid_dims[2]})")

    # Filter out zero counts
    valid_mask = np.array(n_boxes) > 0
    if np.sum(valid_mask) < 3:
        raise ValueError(f"Insufficient non-zero box counts: {np.sum(valid_mask)}")

    deltas = np.array(deltas)[valid_mask]
    n_boxes = np.array(n_boxes)[valid_mask]

    # Linear regression in log-log space
    log_inv_delta = np.log(1.0 / deltas)
    log_n_boxes = np.log(n_boxes)

    slope, intercept, r_value, p_value, std_err = stats.linregress(
        log_inv_delta, log_n_boxes
    )

    return FractalDimensionResult(
        dimension=slope,
        r_squared=r_value ** 2,
        std_error=std_err,
        intercept=intercept,
        deltas=deltas.tolist(),
        n_boxes=n_boxes.tolist(),
        log_inv_delta=log_inv_delta,
        log_n_boxes=log_n_boxes,
    )


def analyze_mesh(filename: str,
                 initial_delta: Optional[float] = None,
                 delta_factor: float = 1.5,
                 num_steps: int = 15,
                 verbose: bool = True) -> FractalDimensionResult:
    """
    Convenience function to analyze a mesh file.

    Args:
        filename: Path to mesh file (.vtk, .vtp, or .stl)
        initial_delta: Starting cube size (auto-computed if None)
        delta_factor: Factor to reduce delta by each step
        num_steps: Number of scales to analyze
        verbose: Print progress information

    Returns:
        FractalDimensionResult
    """
    from .mesh_io import load_mesh

    if verbose:
        print(f"Loading mesh: {filename}")

    mesh = load_mesh(filename)

    if verbose:
        print(f"  Vertices: {mesh.n_vertices}")
        print(f"  Triangles: {mesh.n_triangles}")
        print(f"  Bounding box: ({mesh.bbox.width:.4f} × "
              f"{mesh.bbox.height:.4f} × {mesh.bbox.depth:.4f})")
        print(f"  Surface area: {mesh.surface_area:.4f}")
        print(f"\nComputing fractal dimension...")

    result = compute_fractal_dimension_3d(
        mesh,
        initial_delta=initial_delta,
        delta_factor=delta_factor,
        num_steps=num_steps,
    )

    if verbose:
        print(f"\nResults:")
        print(f"  Fractal dimension: {result.dimension:.4f}")
        print(f"  R²: {result.r_squared:.6f}")
        print(f"  Standard error: {result.std_error:.6f}")

    return result
