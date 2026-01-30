#!/usr/bin/env python3
"""
3D Multifractal Analysis for Surface Meshes.

Computes multifractal spectrum for 3D surfaces:
- Generalized dimensions D_q
- Singularity spectrum f(alpha)
- Multifractal width and asymmetry
- Cube-counting based analysis using triangle surface areas as the measure

This module extends the 2D multifractal formalism to 3D triangulated surfaces.

Reference:
    Adapted from the 2D implementation in FractalParameterAI by extending
    the box-counting approach to cube-counting with surface area measures.
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import time

from .mesh_io import TriangleMesh, BoundingBox3D

# Try to import numba for JIT compilation
try:
    from numba import jit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range


@dataclass
class MultifractalSpectrum3D:
    """Container for 3D multifractal analysis results."""
    # Generalized dimensions
    d0: float  # Capacity dimension (box-counting dimension)
    d1: float  # Information dimension
    d2: float  # Correlation dimension
    d_inf: float  # D_infinity (minimum value of D_q)
    d_minus_inf: float  # D_minus_infinity (maximum value of D_q)

    # Singularity spectrum
    alpha_min: float  # Minimum singularity strength
    alpha_max: float  # Maximum singularity strength
    alpha_0: float  # alpha at f(alpha) maximum
    f_alpha_max: float  # Maximum f(alpha) value

    # Spectrum characteristics
    spectrum_width: float  # Delta_alpha = alpha_max - alpha_min
    spectrum_asymmetry: float  # (alpha_0 - alpha_min) / (alpha_max - alpha_min)

    # Full spectrum data
    q_values: np.ndarray  # Array of q values used
    d_q: np.ndarray  # D_q for each q
    tau_q: np.ndarray  # tau(q) for each q
    alpha: np.ndarray  # alpha values for f(alpha) curve
    f_alpha: np.ndarray  # f(alpha) values

    # Quality metrics
    d_q_r_squared: np.ndarray  # R-squared for each D_q fit
    mean_r_squared: float  # Mean R-squared across all q

    # Analysis metadata
    n_scales: int  # Number of cube sizes used
    total_surface_area: float  # Total mesh surface area
    analysis_time: float


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
    Uses Separating Axis Theorem (SAT).
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

    # Cross product axes (9 tests) - e0 x X, Y, Z
    p0 = t0_y * e0_z - t0_z * e0_y
    p1 = t1_y * e0_z - t1_z * e0_y
    p2 = t2_y * e0_z - t2_z * e0_y
    r = hy * abs(e0_z) + hz * abs(e0_y)
    if min(p0, p1, p2) > r or max(p0, p1, p2) < -r:
        return False

    p0 = t0_z * e0_x - t0_x * e0_z
    p1 = t1_z * e0_x - t1_x * e0_z
    p2 = t2_z * e0_x - t2_x * e0_z
    r = hx * abs(e0_z) + hz * abs(e0_x)
    if min(p0, p1, p2) > r or max(p0, p1, p2) < -r:
        return False

    p0 = t0_x * e0_y - t0_y * e0_x
    p1 = t1_x * e0_y - t1_y * e0_x
    p2 = t2_x * e0_y - t2_y * e0_x
    r = hx * abs(e0_y) + hy * abs(e0_x)
    if min(p0, p1, p2) > r or max(p0, p1, p2) < -r:
        return False

    # e1 x X, Y, Z
    p0 = t0_y * e1_z - t0_z * e1_y
    p1 = t1_y * e1_z - t1_z * e1_y
    p2 = t2_y * e1_z - t2_z * e1_y
    r = hy * abs(e1_z) + hz * abs(e1_y)
    if min(p0, p1, p2) > r or max(p0, p1, p2) < -r:
        return False

    p0 = t0_z * e1_x - t0_x * e1_z
    p1 = t1_z * e1_x - t1_x * e1_z
    p2 = t2_z * e1_x - t2_x * e1_z
    r = hx * abs(e1_z) + hz * abs(e1_x)
    if min(p0, p1, p2) > r or max(p0, p1, p2) < -r:
        return False

    p0 = t0_x * e1_y - t0_y * e1_x
    p1 = t1_x * e1_y - t1_y * e1_x
    p2 = t2_x * e1_y - t2_y * e1_x
    r = hx * abs(e1_y) + hy * abs(e1_x)
    if min(p0, p1, p2) > r or max(p0, p1, p2) < -r:
        return False

    # e2 x X, Y, Z
    p0 = t0_y * e2_z - t0_z * e2_y
    p1 = t1_y * e2_z - t1_z * e2_y
    p2 = t2_y * e2_z - t2_z * e2_y
    r = hy * abs(e2_z) + hz * abs(e2_y)
    if min(p0, p1, p2) > r or max(p0, p1, p2) < -r:
        return False

    p0 = t0_z * e2_x - t0_x * e2_z
    p1 = t1_z * e2_x - t1_x * e2_z
    p2 = t2_z * e2_x - t2_x * e2_z
    r = hx * abs(e2_z) + hz * abs(e2_x)
    if min(p0, p1, p2) > r or max(p0, p1, p2) < -r:
        return False

    p0 = t0_x * e2_y - t0_y * e2_x
    p1 = t1_x * e2_y - t1_y * e2_x
    p2 = t2_x * e2_y - t2_y * e2_x
    r = hx * abs(e2_y) + hy * abs(e2_x)
    if min(p0, p1, p2) > r or max(p0, p1, p2) < -r:
        return False

    return True


def _triangle_cube_intersection_python(v0, v1, v2, cube_min, cube_max) -> bool:
    """Pure Python triangle-cube intersection for fallback."""
    # Cube center and half-size
    cube_center = (cube_min + cube_max) / 2
    half_size = (cube_max - cube_min) / 2

    # Translate triangle
    t0 = v0 - cube_center
    t1 = v1 - cube_center
    t2 = v2 - cube_center

    # AABB test
    tri_min = np.minimum(np.minimum(t0, t1), t2)
    tri_max = np.maximum(np.maximum(t0, t1), t2)

    if np.any(tri_min > half_size) or np.any(tri_max < -half_size):
        return False

    # Edge vectors
    e0 = t1 - t0
    e1 = t2 - t1
    e2 = t0 - t2

    # Triangle normal test
    normal = np.cross(e0, e1)
    d = -np.dot(normal, t0)
    r = np.sum(half_size * np.abs(normal))

    if abs(d) > r:
        return False

    # Cross product axes (simplified - check main axes)
    axes = [
        np.array([0, -e0[2], e0[1]]),
        np.array([e0[2], 0, -e0[0]]),
        np.array([-e0[1], e0[0], 0]),
        np.array([0, -e1[2], e1[1]]),
        np.array([e1[2], 0, -e1[0]]),
        np.array([-e1[1], e1[0], 0]),
        np.array([0, -e2[2], e2[1]]),
        np.array([e2[2], 0, -e2[0]]),
        np.array([-e2[1], e2[0], 0]),
    ]

    for axis in axes:
        axis_len_sq = np.dot(axis, axis)
        if axis_len_sq < 1e-12:
            continue

        p0 = np.dot(t0, axis)
        p1 = np.dot(t1, axis)
        p2 = np.dot(t2, axis)
        r = np.sum(half_size * np.abs(axis))

        if min(p0, p1, p2) > r or max(p0, p1, p2) < -r:
            return False

    return True


class MultifractalAnalyzer3D:
    """
    Computes multifractal spectrum for 3D triangulated surfaces.

    The multifractal formalism for surfaces:
    1. Partition 3D space into cubes of size epsilon
    2. Compute surface area measure mu_i(epsilon) in each cube
    3. Normalize to get probabilities p_i(epsilon) = mu_i / sum(mu_i)
    4. Calculate partition function: Z_q(epsilon) = sum(p_i^q)
    5. Compute tau(q) from scaling: Z_q(epsilon) ~ epsilon^tau(q)
    6. Generalized dimensions: D_q = tau(q) / (q - 1)
    7. Singularity spectrum: alpha(q) = d(tau)/dq, f(alpha) = q*alpha - tau

    For a monofractal surface, D_q is constant for all q.
    For a multifractal surface, D_q varies with q.
    """

    def __init__(self,
                 q_min: float = -5.0,
                 q_max: float = 5.0,
                 n_q: int = 21,
                 debug: bool = False):
        """
        Initialize multifractal analyzer.

        Args:
            q_min: Minimum q value (negative q emphasizes sparse regions)
            q_max: Maximum q value (positive q emphasizes dense regions)
            n_q: Number of q values
            debug: Enable debug output
        """
        self.q_min = q_min
        self.q_max = q_max
        self.n_q = n_q
        self.debug = debug

        # Create q values array, avoiding q=1 (singularity in D_q formula)
        q_values = np.linspace(q_min, q_max, n_q)
        # Remove q values very close to 1
        self.q_values = q_values[np.abs(q_values - 1.0) > 0.1]

    def analyze(self,
                mesh: TriangleMesh,
                initial_delta: Optional[float] = None,
                delta_factor: float = 1.5,
                num_steps: int = 12) -> Optional[MultifractalSpectrum3D]:
        """
        Compute multifractal spectrum from triangulated surface.

        Args:
            mesh: TriangleMesh object
            initial_delta: Initial cube size (default: char_length / 5)
            delta_factor: Factor for cube size progression
            num_steps: Number of cube sizes

        Returns:
            MultifractalSpectrum3D object or None on failure
        """
        start_time = time.time()

        if self.debug:
            print(f"   Computing 3D multifractal spectrum...")
            print(f"      Triangles: {mesh.n_triangles}")
            print(f"      Q range: [{self.q_min}, {self.q_max}] ({len(self.q_values)} values)")
            print(f"      Scales: {num_steps}")

        # Get triangle data
        triangle_vertices = mesh.get_all_triangle_vertices()
        triangle_areas = mesh.triangle_areas()
        total_area = np.sum(triangle_areas)

        if total_area <= 0:
            if self.debug:
                print(f"   Error: Zero total surface area")
            return None

        # Set default initial delta based on characteristic length
        char_len = mesh.bbox.characteristic_length
        if initial_delta is None:
            initial_delta = char_len / 5

        min_delta = char_len / 200  # Prevent too small cubes

        # Generate cube sizes (geometric progression, decreasing)
        deltas = []
        delta = initial_delta
        for _ in range(num_steps):
            if delta < min_delta:
                break
            deltas.append(delta)
            delta = delta / delta_factor

        deltas = np.array(deltas)

        if self.debug:
            print(f"      Cube sizes: {deltas[0]:.6f} to {deltas[-1]:.6f}")

        # ============================================
        # 1. Compute cube areas for each scale
        # ============================================
        cube_data = []
        for delta in deltas:
            areas_per_cube, n_cubes = self._compute_cube_areas(
                mesh, triangle_vertices, triangle_areas, delta
            )

            if len(areas_per_cube) > 0:
                # Normalize to get probabilities
                probabilities = areas_per_cube / total_area
                cube_data.append({
                    'delta': delta,
                    'areas': areas_per_cube,
                    'probabilities': probabilities,
                    'n_cubes': n_cubes
                })
                if self.debug:
                    print(f"      delta={delta:.5f}: {n_cubes} cubes with surface")
            else:
                if self.debug:
                    print(f"      Warning: No cubes at delta={delta:.6f}")

        if len(cube_data) < 3:
            if self.debug:
                print(f"   Error: Insufficient scales ({len(cube_data)}) for analysis")
            return None

        # ============================================
        # 2. Compute partition function Z_q(epsilon) for each q
        # ============================================
        tau_q = []
        d_q = []
        tau_r_squared = []

        for q in self.q_values:
            # Compute Z_q for each scale
            z_q = []
            deltas_used = []

            for data in cube_data:
                probs = data['probabilities']
                delta = data['delta']

                # Filter out zero probabilities for stability
                probs_nonzero = probs[probs > 1e-15]

                if len(probs_nonzero) == 0:
                    continue

                # Partition function: Z_q(epsilon) = sum(p_i^q)
                if q == 0:
                    # For q = 0: Z_0 = number of non-empty cubes
                    z = len(probs_nonzero)
                elif q > 0:
                    z = np.sum(probs_nonzero ** q)
                else:
                    # For negative q, need special handling to avoid overflow
                    # Only include cubes with significant probability
                    z = np.sum(probs_nonzero ** q)

                if z > 0 and np.isfinite(z):
                    z_q.append(z)
                    deltas_used.append(delta)

            if len(z_q) >= 3:
                # Fit tau(q) from Z_q(epsilon) ~ epsilon^tau(q)
                log_delta = np.log(deltas_used)
                log_z = np.log(z_q)

                # Linear regression: log(Z_q) = tau(q) * log(epsilon) + const
                coeffs = np.polyfit(log_delta, log_z, 1)
                tau = coeffs[0]

                # Compute R-squared
                predicted = coeffs[0] * log_delta + coeffs[1]
                ss_res = np.sum((log_z - predicted) ** 2)
                ss_tot = np.sum((log_z - np.mean(log_z)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

                tau_q.append(tau)
                tau_r_squared.append(r_squared)

                # Generalized dimension: D_q = tau(q) / (q - 1)
                if abs(q - 1.0) > 0.01:
                    dq = tau / (q - 1.0)
                else:
                    dq = np.nan  # D_1 handled separately
                d_q.append(dq)
            else:
                tau_q.append(np.nan)
                d_q.append(np.nan)
                tau_r_squared.append(0.0)

        tau_q = np.array(tau_q)
        d_q = np.array(d_q)
        tau_r_squared = np.array(tau_r_squared)

        # Remove NaN values
        valid_mask = ~np.isnan(d_q)
        q_valid = self.q_values[valid_mask]
        d_q_valid = d_q[valid_mask]
        tau_q_valid = tau_q[valid_mask]
        r_squared_valid = tau_r_squared[valid_mask]

        if len(d_q_valid) < 3:
            if self.debug:
                print(f"   Error: Insufficient valid D_q values ({len(d_q_valid)})")
            return None

        # ============================================
        # 3. Extract key dimensions
        # ============================================
        # D_0: capacity dimension (box-counting dimension)
        d0_idx = np.argmin(np.abs(q_valid))
        d0 = d_q_valid[d0_idx]

        # D_1: information dimension (interpolate near q=1)
        below_1 = q_valid < 1.0
        above_1 = q_valid > 1.0
        if np.any(below_1) and np.any(above_1):
            q_below = q_valid[below_1][-1]
            q_above = q_valid[above_1][0]
            d_below = d_q_valid[below_1][-1]
            d_above = d_q_valid[above_1][0]
            d1 = d_below + (1.0 - q_below) * (d_above - d_below) / (q_above - q_below)
        else:
            d1 = d0

        # D_2: correlation dimension
        d2_idx = np.argmin(np.abs(q_valid - 2.0))
        d2 = d_q_valid[d2_idx]

        # D_inf and D_-inf
        d_inf = np.min(d_q_valid)
        d_minus_inf = np.max(d_q_valid)

        if self.debug:
            print(f"      D_0 = {d0:.4f}, D_1 = {d1:.4f}, D_2 = {d2:.4f}")

        # ============================================
        # 4. Compute singularity spectrum f(alpha)
        # ============================================
        # alpha(q) = d(tau)/dq
        # f(alpha) = q * alpha - tau

        # Compute derivative d(tau)/dq numerically
        alpha_values = np.gradient(tau_q_valid, q_valid)
        f_alpha_values = q_valid * alpha_values - tau_q_valid

        # Find spectrum characteristics
        f_max_idx = np.argmax(f_alpha_values)
        alpha_0 = alpha_values[f_max_idx]
        f_alpha_max = f_alpha_values[f_max_idx]

        alpha_min = np.min(alpha_values)
        alpha_max = np.max(alpha_values)
        spectrum_width = alpha_max - alpha_min

        # Asymmetry: (alpha_0 - alpha_min) / (alpha_max - alpha_min)
        if spectrum_width > 1e-6:
            spectrum_asymmetry = (alpha_0 - alpha_min) / spectrum_width
        else:
            spectrum_asymmetry = 0.5

        if self.debug:
            print(f"      Spectrum width Delta_alpha = {spectrum_width:.4f}")
            print(f"      Asymmetry = {spectrum_asymmetry:.4f}")

        # ============================================
        # 5. Package results
        # ============================================
        analysis_time = time.time() - start_time
        mean_r_squared = np.mean(r_squared_valid)

        spectrum = MultifractalSpectrum3D(
            d0=d0,
            d1=d1,
            d2=d2,
            d_inf=d_inf,
            d_minus_inf=d_minus_inf,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            alpha_0=alpha_0,
            f_alpha_max=f_alpha_max,
            spectrum_width=spectrum_width,
            spectrum_asymmetry=spectrum_asymmetry,
            q_values=q_valid,
            d_q=d_q_valid,
            tau_q=tau_q_valid,
            alpha=alpha_values,
            f_alpha=f_alpha_values,
            d_q_r_squared=r_squared_valid,
            mean_r_squared=mean_r_squared,
            n_scales=len(cube_data),
            total_surface_area=total_area,
            analysis_time=analysis_time
        )

        if self.debug:
            print(f"      Analysis complete in {analysis_time:.2f}s")

        return spectrum

    def _compute_cube_areas(self,
                            mesh: TriangleMesh,
                            triangle_vertices: np.ndarray,
                            triangle_areas: np.ndarray,
                            delta: float) -> Tuple[np.ndarray, int]:
        """
        Compute total surface area in each cube.

        For multifractal analysis, we need the measure (surface area) in each cube,
        not just binary occupancy.

        Args:
            mesh: TriangleMesh
            triangle_vertices: (N, 3, 3) array of triangle vertices
            triangle_areas: (N,) array of triangle areas
            delta: Cube side length

        Returns:
            Tuple of (areas_array, n_cubes) where areas_array contains
            the total surface area in each occupied cube
        """
        bbox = mesh.bbox

        # Grid dimensions
        nx = max(1, int(np.ceil(bbox.width / delta)))
        ny = max(1, int(np.ceil(bbox.height / delta)))
        nz = max(1, int(np.ceil(bbox.depth / delta)))

        # Dictionary to accumulate area in each cube
        cube_areas = {}

        # For each triangle, find which cubes it intersects and distribute area
        for t_idx in range(mesh.n_triangles):
            v0 = triangle_vertices[t_idx, 0]
            v1 = triangle_vertices[t_idx, 1]
            v2 = triangle_vertices[t_idx, 2]
            area = triangle_areas[t_idx]

            if area <= 0:
                continue

            # Triangle bounding box
            t_min = np.minimum(np.minimum(v0, v1), v2)
            t_max = np.maximum(np.maximum(v0, v1), v2)

            # Cube index range
            i_min = max(0, int((t_min[0] - bbox.min_x) / delta))
            i_max = min(nx - 1, int((t_max[0] - bbox.min_x) / delta))
            j_min = max(0, int((t_min[1] - bbox.min_y) / delta))
            j_max = min(ny - 1, int((t_max[1] - bbox.min_y) / delta))
            k_min = max(0, int((t_min[2] - bbox.min_z) / delta))
            k_max = min(nz - 1, int((t_max[2] - bbox.min_z) / delta))

            # Count intersecting cubes for this triangle
            intersecting_cubes = []

            for i in range(i_min, i_max + 1):
                for j in range(j_min, j_max + 1):
                    for k in range(k_min, k_max + 1):
                        cube_min = np.array([
                            bbox.min_x + i * delta,
                            bbox.min_y + j * delta,
                            bbox.min_z + k * delta
                        ])
                        cube_max = cube_min + delta

                        # Test intersection
                        if HAS_NUMBA:
                            intersects = _triangle_cube_intersection_numba(
                                v0[0], v0[1], v0[2],
                                v1[0], v1[1], v1[2],
                                v2[0], v2[1], v2[2],
                                cube_min[0], cube_min[1], cube_min[2],
                                cube_max[0], cube_max[1], cube_max[2]
                            )
                        else:
                            intersects = _triangle_cube_intersection_python(
                                v0, v1, v2, cube_min, cube_max
                            )

                        if intersects:
                            intersecting_cubes.append((i, j, k))

            # Distribute triangle area equally among intersecting cubes
            # This is an approximation - a more accurate method would compute
            # the exact area within each cube
            if len(intersecting_cubes) > 0:
                area_per_cube = area / len(intersecting_cubes)
                for cube_key in intersecting_cubes:
                    if cube_key not in cube_areas:
                        cube_areas[cube_key] = 0.0
                    cube_areas[cube_key] += area_per_cube

        # Convert to array
        if len(cube_areas) > 0:
            areas = np.array(list(cube_areas.values()))
        else:
            areas = np.array([])

        return areas, len(cube_areas)


def analyze_mesh_multifractal(filename: str,
                               q_min: float = -5.0,
                               q_max: float = 5.0,
                               n_q: int = 21,
                               initial_delta: Optional[float] = None,
                               delta_factor: float = 1.5,
                               num_steps: int = 12,
                               verbose: bool = True) -> Optional[MultifractalSpectrum3D]:
    """
    Convenience function to analyze a mesh file for multifractal properties.

    Args:
        filename: Path to mesh file (.vtk, .vtp, or .stl)
        q_min: Minimum q value
        q_max: Maximum q value
        n_q: Number of q values
        initial_delta: Initial cube size (auto-computed if None)
        delta_factor: Factor to reduce delta by each step
        num_steps: Number of scales to analyze
        verbose: Print progress information

    Returns:
        MultifractalSpectrum3D or None on failure
    """
    from .mesh_io import load_mesh

    if verbose:
        print(f"Loading mesh: {filename}")

    mesh = load_mesh(filename)

    if verbose:
        print(f"  Vertices: {mesh.n_vertices}")
        print(f"  Triangles: {mesh.n_triangles}")
        print(f"  Surface area: {mesh.surface_area:.6f}")
        print(f"\nComputing multifractal spectrum...")

    analyzer = MultifractalAnalyzer3D(
        q_min=q_min,
        q_max=q_max,
        n_q=n_q,
        debug=verbose
    )

    spectrum = analyzer.analyze(
        mesh,
        initial_delta=initial_delta,
        delta_factor=delta_factor,
        num_steps=num_steps
    )

    if verbose and spectrum:
        print(f"\n{'='*60}")
        print(f"MULTIFRACTAL ANALYSIS RESULTS")
        print(f"{'='*60}")

        print(f"\nGeneralized Dimensions:")
        print(f"   D_0 (capacity):    {spectrum.d0:.6f}")
        print(f"   D_1 (information): {spectrum.d1:.6f}")
        print(f"   D_2 (correlation): {spectrum.d2:.6f}")
        print(f"   D_inf:             {spectrum.d_inf:.6f}")
        print(f"   D_-inf:            {spectrum.d_minus_inf:.6f}")

        print(f"\nSingularity Spectrum:")
        print(f"   alpha_min: {spectrum.alpha_min:.6f}")
        print(f"   alpha_0:   {spectrum.alpha_0:.6f}")
        print(f"   alpha_max: {spectrum.alpha_max:.6f}")
        print(f"   f(alpha)_max: {spectrum.f_alpha_max:.6f}")

        print(f"\nSpectrum Characteristics:")
        print(f"   Width Delta_alpha: {spectrum.spectrum_width:.6f}")
        print(f"   Asymmetry: {spectrum.spectrum_asymmetry:.6f}")

        print(f"\nQuality Metrics:")
        print(f"   Mean R-squared: {spectrum.mean_r_squared:.6f}")
        print(f"   Scales used: {spectrum.n_scales}")
        print(f"   Analysis time: {spectrum.analysis_time:.2f}s")

        print(f"\n{'='*60}")

    return spectrum


def main():
    """Command-line interface for 3D multifractal analysis."""
    import argparse

    parser = argparse.ArgumentParser(
        description='3D Multifractal Analysis for Surface Meshes'
    )
    parser.add_argument('mesh_file', help='Path to mesh file (.vtk, .vtp, .stl)')
    parser.add_argument('--q-min', type=float, default=-5.0,
                        help='Minimum q value (default: -5.0)')
    parser.add_argument('--q-max', type=float, default=5.0,
                        help='Maximum q value (default: 5.0)')
    parser.add_argument('--n-q', type=int, default=21,
                        help='Number of q values (default: 21)')
    parser.add_argument('--num-steps', type=int, default=12,
                        help='Number of scales (default: 12)')
    parser.add_argument('--delta-factor', type=float, default=1.5,
                        help='Scale factor between cube sizes (default: 1.5)')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress verbose output')
    parser.add_argument('--output', '-o', type=str,
                        help='Output CSV file for results')

    args = parser.parse_args()

    spectrum = analyze_mesh_multifractal(
        args.mesh_file,
        q_min=args.q_min,
        q_max=args.q_max,
        n_q=args.n_q,
        num_steps=args.num_steps,
        delta_factor=args.delta_factor,
        verbose=not args.quiet
    )

    if spectrum and args.output:
        # Save D_q spectrum to CSV
        import csv
        with open(args.output, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['q', 'D_q', 'tau_q', 'alpha', 'f_alpha', 'R_squared'])
            for i in range(len(spectrum.q_values)):
                writer.writerow([
                    spectrum.q_values[i],
                    spectrum.d_q[i],
                    spectrum.tau_q[i],
                    spectrum.alpha[i],
                    spectrum.f_alpha[i],
                    spectrum.d_q_r_squared[i]
                ])
        print(f"\nResults saved to: {args.output}")

    return 0 if spectrum else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
