"""
GPU-accelerated cube counting using Numba CUDA.

Provides a CUDA kernel for the Separating Axis Theorem (SAT) triangle-cube
intersection test, writing results to a boolean occupied grid. This eliminates
the expensive CPU-side set() deduplication and exploits massive parallelism
(one thread per triangle).

Falls back gracefully when CUDA is not available.
"""

import numpy as np
from typing import Tuple

# Check CUDA availability
try:
    from numba import cuda
    if cuda.is_available():
        HAS_CUDA = True
    else:
        HAS_CUDA = False
except (ImportError, Exception):
    HAS_CUDA = False


if HAS_CUDA:
    from numba import cuda as _cuda
    import math

    @_cuda.jit(device=True)
    def _min3(a, b, c):
        """Device function: minimum of 3 floats."""
        t = a if a < b else b
        return t if t < c else c

    @_cuda.jit(device=True)
    def _max3(a, b, c):
        """Device function: maximum of 3 floats."""
        t = a if a > b else b
        return t if t > c else c

    @_cuda.jit(device=True)
    def _sat_test(
        v0_x, v0_y, v0_z,
        v1_x, v1_y, v1_z,
        v2_x, v2_y, v2_z,
        cube_min_x, cube_min_y, cube_min_z,
        cube_max_x, cube_max_y, cube_max_z
    ):
        """
        Device function: SAT triangle-cube intersection test.

        Returns 1 if intersecting, 0 if not.
        Identical logic to the CPU Numba version in fast_counting.py.
        """
        # Cube center and half-size
        cx = (cube_min_x + cube_max_x) * 0.5
        cy = (cube_min_y + cube_max_y) * 0.5
        cz = (cube_min_z + cube_max_z) * 0.5
        hx = (cube_max_x - cube_min_x) * 0.5
        hy = (cube_max_y - cube_min_y) * 0.5
        hz = (cube_max_z - cube_min_z) * 0.5

        # Translate triangle so cube center is at origin
        t0_x = v0_x - cx; t0_y = v0_y - cy; t0_z = v0_z - cz
        t1_x = v1_x - cx; t1_y = v1_y - cy; t1_z = v1_z - cz
        t2_x = v2_x - cx; t2_y = v2_y - cy; t2_z = v2_z - cz

        # AABB overlap test (3 axes)
        if _min3(t0_x, t1_x, t2_x) > hx or _max3(t0_x, t1_x, t2_x) < -hx:
            return 0
        if _min3(t0_y, t1_y, t2_y) > hy or _max3(t0_y, t1_y, t2_y) < -hy:
            return 0
        if _min3(t0_z, t1_z, t2_z) > hz or _max3(t0_z, t1_z, t2_z) < -hz:
            return 0

        # Edge vectors
        e0_x = t1_x - t0_x; e0_y = t1_y - t0_y; e0_z = t1_z - t0_z
        e1_x = t2_x - t1_x; e1_y = t2_y - t1_y; e1_z = t2_z - t1_z
        e2_x = t0_x - t2_x; e2_y = t0_y - t2_y; e2_z = t0_z - t2_z

        # Triangle normal test
        n_x = e0_y * e1_z - e0_z * e1_y
        n_y = e0_z * e1_x - e0_x * e1_z
        n_z = e0_x * e1_y - e0_y * e1_x
        d = -(n_x * t0_x + n_y * t0_y + n_z * t0_z)
        r = hx * abs(n_x) + hy * abs(n_y) + hz * abs(n_z)
        if abs(d) > r:
            return 0

        # 9 cross-product axis tests (edge x cube-face-normal)
        # e0 x X = (0, -e0_z, e0_y)
        p0 = t0_y * e0_z - t0_z * e0_y
        p1 = t1_y * e0_z - t1_z * e0_y
        p2 = t2_y * e0_z - t2_z * e0_y
        r = hy * abs(e0_z) + hz * abs(e0_y)
        if _min3(p0, p1, p2) > r or _max3(p0, p1, p2) < -r:
            return 0

        # e0 x Y = (e0_z, 0, -e0_x)
        p0 = t0_z * e0_x - t0_x * e0_z
        p1 = t1_z * e0_x - t1_x * e0_z
        p2 = t2_z * e0_x - t2_x * e0_z
        r = hx * abs(e0_z) + hz * abs(e0_x)
        if _min3(p0, p1, p2) > r or _max3(p0, p1, p2) < -r:
            return 0

        # e0 x Z = (-e0_y, e0_x, 0)
        p0 = t0_x * e0_y - t0_y * e0_x
        p1 = t1_x * e0_y - t1_y * e0_x
        p2 = t2_x * e0_y - t2_y * e0_x
        r = hx * abs(e0_y) + hy * abs(e0_x)
        if _min3(p0, p1, p2) > r or _max3(p0, p1, p2) < -r:
            return 0

        # e1 x X
        p0 = t0_y * e1_z - t0_z * e1_y
        p1 = t1_y * e1_z - t1_z * e1_y
        p2 = t2_y * e1_z - t2_z * e1_y
        r = hy * abs(e1_z) + hz * abs(e1_y)
        if _min3(p0, p1, p2) > r or _max3(p0, p1, p2) < -r:
            return 0

        # e1 x Y
        p0 = t0_z * e1_x - t0_x * e1_z
        p1 = t1_z * e1_x - t1_x * e1_z
        p2 = t2_z * e1_x - t2_x * e1_z
        r = hx * abs(e1_z) + hz * abs(e1_x)
        if _min3(p0, p1, p2) > r or _max3(p0, p1, p2) < -r:
            return 0

        # e1 x Z
        p0 = t0_x * e1_y - t0_y * e1_x
        p1 = t1_x * e1_y - t1_y * e1_x
        p2 = t2_x * e1_y - t2_y * e1_x
        r = hx * abs(e1_y) + hy * abs(e1_x)
        if _min3(p0, p1, p2) > r or _max3(p0, p1, p2) < -r:
            return 0

        # e2 x X
        p0 = t0_y * e2_z - t0_z * e2_y
        p1 = t1_y * e2_z - t1_z * e2_y
        p2 = t2_y * e2_z - t2_z * e2_y
        r = hy * abs(e2_z) + hz * abs(e2_y)
        if _min3(p0, p1, p2) > r or _max3(p0, p1, p2) < -r:
            return 0

        # e2 x Y
        p0 = t0_z * e2_x - t0_x * e2_z
        p1 = t1_z * e2_x - t1_x * e2_z
        p2 = t2_z * e2_x - t2_x * e2_z
        r = hx * abs(e2_z) + hz * abs(e2_x)
        if _min3(p0, p1, p2) > r or _max3(p0, p1, p2) < -r:
            return 0

        # e2 x Z
        p0 = t0_x * e2_y - t0_y * e2_x
        p1 = t1_x * e2_y - t1_y * e2_x
        p2 = t2_x * e2_y - t2_y * e2_x
        r = hx * abs(e2_y) + hy * abs(e2_x)
        if _min3(p0, p1, p2) > r or _max3(p0, p1, p2) < -r:
            return 0

        return 1

    @_cuda.jit
    def _count_cubes_kernel(
        tri_verts,      # float64[n_triangles, 3, 3]
        delta,          # float64
        inv_delta,      # float64
        domain_min_x, domain_min_y, domain_min_z,
        nx, ny, nz,
        occupied        # uint8[nx * ny * nz] — flattened boolean grid
    ):
        """
        CUDA kernel: one thread per triangle.

        For each triangle, compute bounding box -> candidate cube range,
        run SAT test on each candidate, mark occupied grid.
        Multiple threads writing 1 to the same cell is idempotent (no atomics needed).
        """
        tid = _cuda.grid(1)
        n_triangles = tri_verts.shape[0]
        if tid >= n_triangles:
            return

        # Load triangle vertices
        v0_x = tri_verts[tid, 0, 0]
        v0_y = tri_verts[tid, 0, 1]
        v0_z = tri_verts[tid, 0, 2]
        v1_x = tri_verts[tid, 1, 0]
        v1_y = tri_verts[tid, 1, 1]
        v1_z = tri_verts[tid, 1, 2]
        v2_x = tri_verts[tid, 2, 0]
        v2_y = tri_verts[tid, 2, 1]
        v2_z = tri_verts[tid, 2, 2]

        # Triangle bounding box -> cube index range
        t_min_x = _min3(v0_x, v1_x, v2_x)
        t_max_x = _max3(v0_x, v1_x, v2_x)
        t_min_y = _min3(v0_y, v1_y, v2_y)
        t_max_y = _max3(v0_y, v1_y, v2_y)
        t_min_z = _min3(v0_z, v1_z, v2_z)
        t_max_z = _max3(v0_z, v1_z, v2_z)

        i_min = int((t_min_x - domain_min_x) * inv_delta)
        i_max = int((t_max_x - domain_min_x) * inv_delta)
        j_min = int((t_min_y - domain_min_y) * inv_delta)
        j_max = int((t_max_y - domain_min_y) * inv_delta)
        k_min = int((t_min_z - domain_min_z) * inv_delta)
        k_max = int((t_max_z - domain_min_z) * inv_delta)

        # Clamp to grid bounds
        if i_min < 0: i_min = 0
        if j_min < 0: j_min = 0
        if k_min < 0: k_min = 0
        if i_max >= nx: i_max = nx - 1
        if j_max >= ny: j_max = ny - 1
        if k_max >= nz: k_max = nz - 1

        # Test each candidate cube
        for i in range(i_min, i_max + 1):
            cube_min_x = domain_min_x + i * delta
            cube_max_x = cube_min_x + delta
            for j in range(j_min, j_max + 1):
                cube_min_y = domain_min_y + j * delta
                cube_max_y = cube_min_y + delta
                for k in range(k_min, k_max + 1):
                    cube_min_z = domain_min_z + k * delta
                    cube_max_z = cube_min_z + delta

                    if _sat_test(
                        v0_x, v0_y, v0_z,
                        v1_x, v1_y, v1_z,
                        v2_x, v2_y, v2_z,
                        cube_min_x, cube_min_y, cube_min_z,
                        cube_max_x, cube_max_y, cube_max_z
                    ):
                        idx = i * (ny * nz) + j * nz + k
                        occupied[idx] = 1

    @_cuda.jit
    def _measure_cubes_kernel(
        tri_verts,      # float64[n_triangles, 3, 3]
        tri_areas,      # float64[n_triangles]
        delta,          # float64
        inv_delta,      # float64
        domain_min_x, domain_min_y, domain_min_z,
        nx, ny, nz,
        cube_counts,    # int32[nx * ny * nz] — number of triangles per cube
        cube_areas,     # float64[nx * ny * nz] — accumulated area per cube
        tri_n_cubes,    # int32[n_triangles] — how many cubes each triangle hits
    ):
        """
        CUDA kernel for multifractal measure computation.

        Two passes needed for area measure:
        Pass 1: Count cubes per triangle (tri_n_cubes) and count triangles per cube (cube_counts).
        Pass 2 (separate kernel): Distribute area using tri_n_cubes.

        This kernel does pass 1: counts + accumulates area equally.
        For area measure, we use atomicAdd to accumulate area fractions.
        For count measure, we use atomicAdd to count triangles.
        """
        tid = _cuda.grid(1)
        n_triangles = tri_verts.shape[0]
        if tid >= n_triangles:
            return

        v0_x = tri_verts[tid, 0, 0]; v0_y = tri_verts[tid, 0, 1]; v0_z = tri_verts[tid, 0, 2]
        v1_x = tri_verts[tid, 1, 0]; v1_y = tri_verts[tid, 1, 1]; v1_z = tri_verts[tid, 1, 2]
        v2_x = tri_verts[tid, 2, 0]; v2_y = tri_verts[tid, 2, 1]; v2_z = tri_verts[tid, 2, 2]

        t_min_x = _min3(v0_x, v1_x, v2_x); t_max_x = _max3(v0_x, v1_x, v2_x)
        t_min_y = _min3(v0_y, v1_y, v2_y); t_max_y = _max3(v0_y, v1_y, v2_y)
        t_min_z = _min3(v0_z, v1_z, v2_z); t_max_z = _max3(v0_z, v1_z, v2_z)

        i_min = int((t_min_x - domain_min_x) * inv_delta)
        i_max = int((t_max_x - domain_min_x) * inv_delta)
        j_min = int((t_min_y - domain_min_y) * inv_delta)
        j_max = int((t_max_y - domain_min_y) * inv_delta)
        k_min = int((t_min_z - domain_min_z) * inv_delta)
        k_max = int((t_max_z - domain_min_z) * inv_delta)

        if i_min < 0: i_min = 0
        if j_min < 0: j_min = 0
        if k_min < 0: k_min = 0
        if i_max >= nx: i_max = nx - 1
        if j_max >= ny: j_max = ny - 1
        if k_max >= nz: k_max = nz - 1

        # First: count how many cubes this triangle intersects
        n_hits = 0
        for i in range(i_min, i_max + 1):
            cube_min_x = domain_min_x + i * delta
            cube_max_x = cube_min_x + delta
            for j in range(j_min, j_max + 1):
                cube_min_y = domain_min_y + j * delta
                cube_max_y = cube_min_y + delta
                for k in range(k_min, k_max + 1):
                    cube_min_z = domain_min_z + k * delta
                    cube_max_z = cube_min_z + delta
                    if _sat_test(
                        v0_x, v0_y, v0_z, v1_x, v1_y, v1_z, v2_x, v2_y, v2_z,
                        cube_min_x, cube_min_y, cube_min_z,
                        cube_max_x, cube_max_y, cube_max_z
                    ):
                        n_hits += 1

        if n_hits == 0:
            return

        tri_n_cubes[tid] = n_hits
        area = tri_areas[tid]
        area_per_cube = area / n_hits

        # Second: accumulate measures
        for i in range(i_min, i_max + 1):
            cube_min_x = domain_min_x + i * delta
            cube_max_x = cube_min_x + delta
            for j in range(j_min, j_max + 1):
                cube_min_y = domain_min_y + j * delta
                cube_max_y = cube_min_y + delta
                for k in range(k_min, k_max + 1):
                    cube_min_z = domain_min_z + k * delta
                    cube_max_z = cube_min_z + delta
                    if _sat_test(
                        v0_x, v0_y, v0_z, v1_x, v1_y, v1_z, v2_x, v2_y, v2_z,
                        cube_min_x, cube_min_y, cube_min_z,
                        cube_max_x, cube_max_y, cube_max_z
                    ):
                        idx = i * (ny * nz) + j * nz + k
                        _cuda.atomic.add(cube_counts, idx, 1)
                        _cuda.atomic.add(cube_areas, idx, area_per_cube)


def count_cubes_gpu(
    triangle_vertices: np.ndarray,
    delta: float,
    domain_min: np.ndarray,
    domain_max: np.ndarray,
) -> tuple:
    """
    GPU-accelerated cube counting.

    Args:
        triangle_vertices: float64 array of shape (n_triangles, 3, 3)
        delta: Cube side length
        domain_min: (3,) array of domain minimum coordinates
        domain_max: (3,) array of domain maximum coordinates

    Returns:
        (n_occupied, nx, ny, nz): count of occupied cubes and grid dimensions
    """
    if not HAS_CUDA:
        raise RuntimeError("CUDA not available")

    nx = max(1, int(np.ceil((domain_max[0] - domain_min[0]) / delta)))
    ny = max(1, int(np.ceil((domain_max[1] - domain_min[1]) / delta)))
    nz = max(1, int(np.ceil((domain_max[2] - domain_min[2]) / delta)))
    grid_size = nx * ny * nz

    inv_delta = 1.0 / delta
    n_triangles = triangle_vertices.shape[0]

    # Allocate device arrays
    d_verts = cuda.to_device(np.ascontiguousarray(triangle_vertices))
    d_occupied = cuda.to_device(np.zeros(grid_size, dtype=np.uint8))

    # Launch kernel
    threads_per_block = 256
    blocks = (n_triangles + threads_per_block - 1) // threads_per_block

    _count_cubes_kernel[blocks, threads_per_block](
        d_verts, delta, inv_delta,
        domain_min[0], domain_min[1], domain_min[2],
        nx, ny, nz,
        d_occupied
    )

    # Count occupied cubes — copy back and sum on CPU
    occupied = d_occupied.copy_to_host()
    n_occupied = int(occupied.sum())

    return n_occupied, nx, ny, nz


def compute_cube_measures_gpu(
    triangle_vertices: np.ndarray,
    triangle_areas: np.ndarray,
    delta: float,
    domain_min: np.ndarray,
    domain_max: np.ndarray,
    measure: str = 'area',
) -> tuple:
    """
    GPU-accelerated cube measure computation for multifractal analysis.

    Args:
        triangle_vertices: float64 array (n_triangles, 3, 3)
        triangle_areas: float64 array (n_triangles,)
        delta: Cube side length
        domain_min: (3,) domain minimum
        domain_max: (3,) domain maximum
        measure: 'area' or 'count'

    Returns:
        (measures, n_occupied): non-zero measures array and occupied count
    """
    if not HAS_CUDA:
        raise RuntimeError("CUDA not available")

    nx = max(1, int(np.ceil((domain_max[0] - domain_min[0]) / delta)))
    ny = max(1, int(np.ceil((domain_max[1] - domain_min[1]) / delta)))
    nz = max(1, int(np.ceil((domain_max[2] - domain_min[2]) / delta)))
    grid_size = nx * ny * nz

    inv_delta = 1.0 / delta
    n_triangles = triangle_vertices.shape[0]

    # Allocate device arrays
    d_verts = cuda.to_device(np.ascontiguousarray(triangle_vertices))
    d_areas = cuda.to_device(np.ascontiguousarray(triangle_areas.astype(np.float64)))
    d_counts = cuda.to_device(np.zeros(grid_size, dtype=np.int32))
    d_cube_areas = cuda.to_device(np.zeros(grid_size, dtype=np.float64))
    d_tri_n_cubes = cuda.to_device(np.zeros(n_triangles, dtype=np.int32))

    # Launch kernel
    threads_per_block = 256
    blocks = (n_triangles + threads_per_block - 1) // threads_per_block

    _measure_cubes_kernel[blocks, threads_per_block](
        d_verts, d_areas, delta, inv_delta,
        domain_min[0], domain_min[1], domain_min[2],
        nx, ny, nz,
        d_counts, d_cube_areas, d_tri_n_cubes
    )

    if measure == 'area':
        cube_values = d_cube_areas.copy_to_host()
        mask = cube_values > 0
    else:
        cube_values = d_counts.copy_to_host().astype(np.float64)
        mask = cube_values > 0

    measures = cube_values[mask]
    n_occupied = int(mask.sum())

    return measures, n_occupied


def check_cuda_available() -> bool:
    """Check if CUDA is available for GPU acceleration."""
    return HAS_CUDA
