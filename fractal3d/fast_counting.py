"""
High-performance cube counting using NumPy vectorization and optional Numba JIT.

This module provides optimized implementations of cube counting that can be
10-100x faster than pure Python for large meshes.
"""

import numpy as np
from typing import Tuple, Optional, Set
from .mesh_io import TriangleMesh, BoundingBox3D
from .box_counting_3d import BoxCountResult

# Try to import numba for JIT compilation
try:
    from numba import jit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Dummy decorator
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range


@jit(nopython=True, cache=True)
def _triangle_cube_intersection_numba(
    v0_x, v0_y, v0_z,
    v1_x, v1_y, v1_z,
    v2_x, v2_y, v2_z,
    cube_min_x, cube_min_y, cube_min_z,
    cube_max_x, cube_max_y, cube_max_z
) -> bool:
    """
    Numba-optimized triangle-cube intersection test.

    All arguments are scalars for maximum performance.
    """
    # Cube center and half-size
    cx = (cube_min_x + cube_max_x) * 0.5
    cy = (cube_min_y + cube_max_y) * 0.5
    cz = (cube_min_z + cube_max_z) * 0.5
    hx = (cube_max_x - cube_min_x) * 0.5
    hy = (cube_max_y - cube_min_y) * 0.5
    hz = (cube_max_z - cube_min_z) * 0.5

    # Translate triangle
    t0_x, t0_y, t0_z = v0_x - cx, v0_y - cy, v0_z - cz
    t1_x, t1_y, t1_z = v1_x - cx, v1_y - cy, v1_z - cz
    t2_x, t2_y, t2_z = v2_x - cx, v2_y - cy, v2_z - cz

    # AABB test
    tri_min_x = min(t0_x, t1_x, t2_x)
    tri_max_x = max(t0_x, t1_x, t2_x)
    tri_min_y = min(t0_y, t1_y, t2_y)
    tri_max_y = max(t0_y, t1_y, t2_y)
    tri_min_z = min(t0_z, t1_z, t2_z)
    tri_max_z = max(t0_z, t1_z, t2_z)

    if tri_min_x > hx or tri_max_x < -hx:
        return False
    if tri_min_y > hy or tri_max_y < -hy:
        return False
    if tri_min_z > hz or tri_max_z < -hz:
        return False

    # Edge vectors
    e0_x, e0_y, e0_z = t1_x - t0_x, t1_y - t0_y, t1_z - t0_z
    e1_x, e1_y, e1_z = t2_x - t1_x, t2_y - t1_y, t2_z - t1_z
    e2_x, e2_y, e2_z = t0_x - t2_x, t0_y - t2_y, t0_z - t2_z

    # Triangle normal test
    n_x = e0_y * e1_z - e0_z * e1_y
    n_y = e0_z * e1_x - e0_x * e1_z
    n_z = e0_x * e1_y - e0_y * e1_x

    d = -(n_x * t0_x + n_y * t0_y + n_z * t0_z)
    r = hx * abs(n_x) + hy * abs(n_y) + hz * abs(n_z)

    if abs(d) > r:
        return False

    # Cross product axes (9 tests)
    # e0 × X = (0, -e0_z, e0_y)
    p0 = t0_y * e0_z - t0_z * e0_y
    p1 = t1_y * e0_z - t1_z * e0_y
    p2 = t2_y * e0_z - t2_z * e0_y
    r = hy * abs(e0_z) + hz * abs(e0_y)
    if min(p0, p1, p2) > r or max(p0, p1, p2) < -r:
        return False

    # e0 × Y = (e0_z, 0, -e0_x)
    p0 = t0_z * e0_x - t0_x * e0_z
    p1 = t1_z * e0_x - t1_x * e0_z
    p2 = t2_z * e0_x - t2_x * e0_z
    r = hx * abs(e0_z) + hz * abs(e0_x)
    if min(p0, p1, p2) > r or max(p0, p1, p2) < -r:
        return False

    # e0 × Z = (-e0_y, e0_x, 0)
    p0 = t0_x * e0_y - t0_y * e0_x
    p1 = t1_x * e0_y - t1_y * e0_x
    p2 = t2_x * e0_y - t2_y * e0_x
    r = hx * abs(e0_y) + hy * abs(e0_x)
    if min(p0, p1, p2) > r or max(p0, p1, p2) < -r:
        return False

    # e1 × X
    p0 = t0_y * e1_z - t0_z * e1_y
    p1 = t1_y * e1_z - t1_z * e1_y
    p2 = t2_y * e1_z - t2_z * e1_y
    r = hy * abs(e1_z) + hz * abs(e1_y)
    if min(p0, p1, p2) > r or max(p0, p1, p2) < -r:
        return False

    # e1 × Y
    p0 = t0_z * e1_x - t0_x * e1_z
    p1 = t1_z * e1_x - t1_x * e1_z
    p2 = t2_z * e1_x - t2_x * e1_z
    r = hx * abs(e1_z) + hz * abs(e1_x)
    if min(p0, p1, p2) > r or max(p0, p1, p2) < -r:
        return False

    # e1 × Z
    p0 = t0_x * e1_y - t0_y * e1_x
    p1 = t1_x * e1_y - t1_y * e1_x
    p2 = t2_x * e1_y - t2_y * e1_x
    r = hx * abs(e1_y) + hy * abs(e1_x)
    if min(p0, p1, p2) > r or max(p0, p1, p2) < -r:
        return False

    # e2 × X
    p0 = t0_y * e2_z - t0_z * e2_y
    p1 = t1_y * e2_z - t1_z * e2_y
    p2 = t2_y * e2_z - t2_z * e2_y
    r = hy * abs(e2_z) + hz * abs(e2_y)
    if min(p0, p1, p2) > r or max(p0, p1, p2) < -r:
        return False

    # e2 × Y
    p0 = t0_z * e2_x - t0_x * e2_z
    p1 = t1_z * e2_x - t1_x * e2_z
    p2 = t2_z * e2_x - t2_x * e2_z
    r = hx * abs(e2_z) + hz * abs(e2_x)
    if min(p0, p1, p2) > r or max(p0, p1, p2) < -r:
        return False

    # e2 × Z
    p0 = t0_x * e2_y - t0_y * e2_x
    p1 = t1_x * e2_y - t1_y * e2_x
    p2 = t2_x * e2_y - t2_y * e2_x
    r = hx * abs(e2_y) + hy * abs(e2_x)
    if min(p0, p1, p2) > r or max(p0, p1, p2) < -r:
        return False

    return True


@jit(nopython=True, parallel=True, cache=True)
def _count_cubes_numba(
    triangle_vertices: np.ndarray,  # (N, 3, 3)
    delta: float,
    domain_min: np.ndarray,  # (3,)
    domain_max: np.ndarray,  # (3,)
) -> np.ndarray:
    """
    Numba-accelerated cube counting with parallel execution.

    Returns a flat array of occupied cube indices encoded as i*ny*nz + j*nz + k.
    """
    n_triangles = triangle_vertices.shape[0]

    # Grid dimensions
    nx = max(1, int(np.ceil((domain_max[0] - domain_min[0]) / delta)))
    ny = max(1, int(np.ceil((domain_max[1] - domain_min[1]) / delta)))
    nz = max(1, int(np.ceil((domain_max[2] - domain_min[2]) / delta)))

    inv_delta = 1.0 / delta

    # Use a large array to collect results (will deduplicate later)
    # Maximum possible cubes touched per triangle
    max_cubes_per_tri = 27  # 3x3x3 worst case for small triangles
    results = np.empty(n_triangles * max_cubes_per_tri, dtype=np.int64)
    result_counts = np.zeros(n_triangles, dtype=np.int64)

    for t_idx in prange(n_triangles):
        v0 = triangle_vertices[t_idx, 0]
        v1 = triangle_vertices[t_idx, 1]
        v2 = triangle_vertices[t_idx, 2]

        # Triangle bounding box
        t_min_x = min(v0[0], v1[0], v2[0])
        t_max_x = max(v0[0], v1[0], v2[0])
        t_min_y = min(v0[1], v1[1], v2[1])
        t_max_y = max(v0[1], v1[1], v2[1])
        t_min_z = min(v0[2], v1[2], v2[2])
        t_max_z = max(v0[2], v1[2], v2[2])

        # Cube index range
        i_min = max(0, int((t_min_x - domain_min[0]) * inv_delta))
        i_max = min(nx - 1, int((t_max_x - domain_min[0]) * inv_delta))
        j_min = max(0, int((t_min_y - domain_min[1]) * inv_delta))
        j_max = min(ny - 1, int((t_max_y - domain_min[1]) * inv_delta))
        k_min = max(0, int((t_min_z - domain_min[2]) * inv_delta))
        k_max = min(nz - 1, int((t_max_z - domain_min[2]) * inv_delta))

        count = 0
        base_idx = t_idx * max_cubes_per_tri

        for i in range(i_min, i_max + 1):
            for j in range(j_min, j_max + 1):
                for k in range(k_min, k_max + 1):
                    cube_min_x = domain_min[0] + i * delta
                    cube_min_y = domain_min[1] + j * delta
                    cube_min_z = domain_min[2] + k * delta
                    cube_max_x = cube_min_x + delta
                    cube_max_y = cube_min_y + delta
                    cube_max_z = cube_min_z + delta

                    if _triangle_cube_intersection_numba(
                        v0[0], v0[1], v0[2],
                        v1[0], v1[1], v1[2],
                        v2[0], v2[1], v2[2],
                        cube_min_x, cube_min_y, cube_min_z,
                        cube_max_x, cube_max_y, cube_max_z
                    ):
                        if count < max_cubes_per_tri:
                            # Encode cube index
                            cube_idx = i * ny * nz + j * nz + k
                            results[base_idx + count] = cube_idx
                            count += 1

        result_counts[t_idx] = count

    return results, result_counts, nx, ny, nz


class FastCubeCounter:
    """
    High-performance cube counter using Numba JIT compilation.

    Falls back to pure NumPy if Numba is not available.
    """

    def __init__(self, mesh: TriangleMesh):
        """Initialize with mesh."""
        self.mesh = mesh
        self._triangle_vertices = mesh.get_all_triangle_vertices().astype(np.float64)
        self._domain_min = np.array([
            mesh.bbox.min_x, mesh.bbox.min_y, mesh.bbox.min_z
        ], dtype=np.float64)
        self._domain_max = np.array([
            mesh.bbox.max_x, mesh.bbox.max_y, mesh.bbox.max_z
        ], dtype=np.float64)

    def count_cubes(self, delta: float,
                    domain: Optional[BoundingBox3D] = None) -> BoxCountResult:
        """
        Count cubes using Numba-accelerated algorithm.

        Args:
            delta: Cube side length
            domain: Optional custom domain

        Returns:
            BoxCountResult
        """
        if domain is not None:
            domain_min = np.array([domain.min_x, domain.min_y, domain.min_z])
            domain_max = np.array([domain.max_x, domain.max_y, domain.max_z])
        else:
            domain_min = self._domain_min
            domain_max = self._domain_max

        if HAS_NUMBA:
            results, counts, nx, ny, nz = _count_cubes_numba(
                self._triangle_vertices, delta, domain_min, domain_max
            )

            # Collect unique cube indices
            occupied = set()
            max_per_tri = 27
            for t_idx in range(len(counts)):
                base = t_idx * max_per_tri
                for i in range(counts[t_idx]):
                    occupied.add(results[base + i])

            return BoxCountResult(
                delta=delta,
                n_boxes=len(occupied),
                grid_dims=(nx, ny, nz)
            )
        else:
            # Fallback to Python implementation
            from .box_counting_3d import CubeCounter
            counter = CubeCounter(self.mesh)
            return counter.count_cubes(delta, domain)


def count_cubes_fast(mesh: TriangleMesh, delta: float,
                     domain: Optional[BoundingBox3D] = None) -> BoxCountResult:
    """
    Convenience function for fast cube counting.

    Args:
        mesh: TriangleMesh to analyze
        delta: Cube side length
        domain: Optional custom domain

    Returns:
        BoxCountResult
    """
    counter = FastCubeCounter(mesh)
    return counter.count_cubes(delta, domain)


def check_numba_available() -> bool:
    """Check if Numba is available for acceleration."""
    return HAS_NUMBA
