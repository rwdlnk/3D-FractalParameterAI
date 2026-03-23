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
                                  max_delta: Optional[float] = None,
                                  use_gpu: str = 'auto') -> FractalDimensionResult:
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
        use_gpu: 'auto' (GPU if available), 'always', or 'never'

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

    # Count cubes at each scale using grid-optimized counting
    # (multiple offsets, take minimum count — matches multifractal Phase 2)
    from .fast_counting import FastCubeCounter
    counter = FastCubeCounter(mesh, use_gpu=use_gpu)
    bbox = mesh.bbox
    n_boxes = []

    for delta in deltas:
        # Adaptive offsets: more offsets at finer scales (same as multifractal)
        if delta < 0.005:
            offset_fracs = np.linspace(0, 0.75, 4)  # 4^3 = 64 tests
        elif delta < 0.02:
            offset_fracs = np.linspace(0, 0.5, 3)   # 3^3 = 27 tests
        else:
            offset_fracs = np.linspace(0, 0.5, 2)   # 2^3 = 8 tests

        best_n = float('inf')
        best_dims = (0, 0, 0)
        grid_tests = 0
        for dx_frac in offset_fracs:
            for dy_frac in offset_fracs:
                for dz_frac in offset_fracs:
                    grid_tests += 1
                    shifted = BoundingBox3D(
                        min_x=bbox.min_x + dx_frac * delta,
                        max_x=bbox.max_x,
                        min_y=bbox.min_y + dy_frac * delta,
                        max_y=bbox.max_y,
                        min_z=bbox.min_z + dz_frac * delta,
                        max_z=bbox.max_z)
                    result = counter.count_cubes(delta, domain=shifted)
                    if result.n_boxes < best_n:
                        best_n = result.n_boxes
                        best_dims = result.grid_dims

        n_boxes.append(best_n)
        print(f"  δ = {delta:.6f}: {best_n} cubes "
              f"(grid: {best_dims[0]}×{best_dims[1]}×{best_dims[2]}) "
              f"[{grid_tests} offsets]")

    # Filter out zero counts
    valid_mask = np.array(n_boxes) > 0
    if np.sum(valid_mask) < 3:
        raise ValueError(f"Insufficient non-zero box counts: {np.sum(valid_mask)}")

    deltas = np.array(deltas)[valid_mask]
    n_boxes = np.array(n_boxes)[valid_mask]

    # Log-log space
    log_inv_delta = np.log(1.0 / deltas)
    log_n_boxes = np.log(n_boxes)

    # Enhanced boundary removal: detect and trim scales where the
    # log-log slope deviates significantly from the middle region
    if len(deltas) > 8:
        n = len(log_inv_delta)
        seg_size = max(3, n // 4)
        if n >= 3 * seg_size:
            try:
                sl_first, _, r2_first, _, _ = stats.linregress(
                    log_inv_delta[:seg_size], log_n_boxes[:seg_size])
                sl_mid, _, r2_mid, _, _ = stats.linregress(
                    log_inv_delta[seg_size:2*seg_size], log_n_boxes[seg_size:2*seg_size])
                sl_last, _, r2_last, _, _ = stats.linregress(
                    log_inv_delta[-seg_size:], log_n_boxes[-seg_size:])
                trim_start, trim_end = 0, 0
                if sl_mid != 0:
                    if abs(sl_first - sl_mid) / abs(sl_mid) > 0.15 or r2_first < 0.95:
                        trim_start = 1
                    if abs(sl_last - sl_mid) / abs(sl_mid) > 0.15 or r2_last < 0.95:
                        trim_end = 1
                if (trim_start or trim_end) and n > (trim_start + trim_end) + 5:
                    deltas = deltas[trim_start:n - trim_end if trim_end else n]
                    n_boxes = n_boxes[trim_start:n - trim_end if trim_end else n]
                    log_inv_delta = log_inv_delta[trim_start:n - trim_end if trim_end else n]
                    log_n_boxes = log_n_boxes[trim_start:n - trim_end if trim_end else n]
            except Exception:
                pass

    # Find optimal scaling region: sliding window, select best R² × range
    best_score = -1.0
    best_slope, best_err, best_r2 = np.nan, np.nan, 0.0
    best_start, best_end = 0, len(log_inv_delta)
    min_pts = min(5, len(log_inv_delta))

    for n_pts in range(min_pts, len(log_inv_delta) + 1):
        for start in range(len(log_inv_delta) - n_pts + 1):
            end = start + n_pts
            x = log_inv_delta[start:end]
            y = log_n_boxes[start:end]
            sl, _, rv, _, se = stats.linregress(x, y)
            r2 = rv ** 2
            scaling_range = abs(x[-1] - x[0])
            if r2 >= 0.97 and scaling_range >= 1.0:
                score = r2 * scaling_range
                if score > best_score:
                    best_score = score
                    best_slope = sl
                    best_err = se
                    best_r2 = r2
                    best_start = start
                    best_end = end

    if np.isnan(best_slope):
        # Fallback: use all data
        best_slope, intercept, rv, _, best_err = stats.linregress(
            log_inv_delta, log_n_boxes)
        best_r2 = rv ** 2
        best_start = 0
        best_end = len(log_inv_delta)
    else:
        _, intercept, _, _, _ = stats.linregress(
            log_inv_delta[best_start:best_end], log_n_boxes[best_start:best_end])

    fit_deltas = deltas[best_start:best_end]
    fit_nboxes = n_boxes[best_start:best_end]

    return FractalDimensionResult(
        dimension=best_slope,
        r_squared=best_r2,
        std_error=best_err,
        intercept=intercept,
        deltas=fit_deltas.tolist(),
        n_boxes=fit_nboxes.tolist(),
        log_inv_delta=log_inv_delta[best_start:best_end],
        log_n_boxes=log_n_boxes[best_start:best_end],
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
