"""
3D Multifractal Analysis for Triangulated Surface Meshes.

Computes the multifractal spectrum of 3D surfaces using cube-based
partition functions. Extends the monofractal cube-counting approach
(N(delta) ~ delta^(-D)) to the full generalized dimension spectrum D_q.

Mathematical framework:
    1. Cover the surface with cubes of size delta
    2. Assign measure mu_i to each occupied cube
    3. Normalize: p_i = mu_i / sum(mu_i)
    4. Partition function: Z_q(delta) = sum(p_i^q)
    5. Mass exponent: tau(q) from log(Z_q) vs log(delta) regression
    6. Generalized dimensions: D_q = tau(q) / (q - 1)
    7. Singularity spectrum: f(alpha) via Legendre transform

Measure choices:
    - 'count': Number of triangles intersecting each cube
      (analogous to 2D segment count measure)
    - 'area': Triangle area distributed among intersecting cubes
      (physically meaningful: surface area density, preserves total measure)

References:
    - Halsey et al. (1986) "Fractal measures and their singularities"
    - Chhabra & Jensen (1989) "Direct determination of f(alpha) singularity spectrum"
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy import stats
import time
import os
import csv

from .mesh_io import TriangleMesh, BoundingBox3D
from .box_counting_3d import triangle_cube_intersection

# Try to import numba for JIT compilation
try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


@dataclass
class MultifractalResult3D:
    """Result of 3D multifractal analysis."""
    q_values: np.ndarray
    tau: np.ndarray          # Mass exponent tau(q)
    Dq: np.ndarray           # Generalized dimensions D_q
    alpha: np.ndarray        # Singularity strengths alpha(q)
    f_alpha: np.ndarray      # Singularity spectrum f(alpha)
    r_squared: np.ndarray    # R-squared of tau(q) fits
    D0: float                # Capacity dimension
    D1: float                # Information dimension
    D2: float                # Correlation dimension
    alpha_width: float       # Width of f(alpha) spectrum
    degree_multifractality: float  # D_q(max) - D_q(min)
    measure_type: str        # 'count' or 'area'
    cube_sizes: np.ndarray   # Cube sizes used
    n_occupied: List[int]    # Number of occupied cubes per scale


# Legacy alias for backward compatibility
MultifractalSpectrum3D = MultifractalResult3D


@jit(nopython=True, cache=True)
def _triangle_cube_intersection_numba(
    v0_x, v0_y, v0_z,
    v1_x, v1_y, v1_z,
    v2_x, v2_y, v2_z,
    cube_min_x, cube_min_y, cube_min_z,
    cube_max_x, cube_max_y, cube_max_z
) -> bool:
    """Numba-optimized triangle-cube intersection test (SAT algorithm)."""
    cx = (cube_min_x + cube_max_x) * 0.5
    cy = (cube_min_y + cube_max_y) * 0.5
    cz = (cube_min_z + cube_max_z) * 0.5
    hx = (cube_max_x - cube_min_x) * 0.5
    hy = (cube_max_y - cube_min_y) * 0.5
    hz = (cube_max_z - cube_min_z) * 0.5

    t0_x, t0_y, t0_z = v0_x - cx, v0_y - cy, v0_z - cz
    t1_x, t1_y, t1_z = v1_x - cx, v1_y - cy, v1_z - cz
    t2_x, t2_y, t2_z = v2_x - cx, v2_y - cy, v2_z - cz

    # AABB test
    if min(t0_x, t1_x, t2_x) > hx or max(t0_x, t1_x, t2_x) < -hx:
        return False
    if min(t0_y, t1_y, t2_y) > hy or max(t0_y, t1_y, t2_y) < -hy:
        return False
    if min(t0_z, t1_z, t2_z) > hz or max(t0_z, t1_z, t2_z) < -hz:
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

    # 9 cross product axis tests
    for ex, ey, ez in [(e0_x, e0_y, e0_z), (e1_x, e1_y, e1_z), (e2_x, e2_y, e2_z)]:
        # edge x X-axis = (0, -ez, ey)
        p0 = t0_y * ez - t0_z * ey
        p1 = t1_y * ez - t1_z * ey
        p2 = t2_y * ez - t2_z * ey
        r = hy * abs(ez) + hz * abs(ey)
        if min(p0, p1, p2) > r or max(p0, p1, p2) < -r:
            return False

        # edge x Y-axis = (ez, 0, -ex)
        p0 = t0_z * ex - t0_x * ez
        p1 = t1_z * ex - t1_x * ez
        p2 = t2_z * ex - t2_x * ez
        r = hx * abs(ez) + hz * abs(ex)
        if min(p0, p1, p2) > r or max(p0, p1, p2) < -r:
            return False

        # edge x Z-axis = (-ey, ex, 0)
        p0 = t0_x * ey - t0_y * ex
        p1 = t1_x * ey - t1_y * ex
        p2 = t2_x * ey - t2_y * ex
        r = hx * abs(ey) + hy * abs(ex)
        if min(p0, p1, p2) > r or max(p0, p1, p2) < -r:
            return False

    return True


def _test_intersection(v0, v1, v2, cube_min, cube_max) -> bool:
    """Test triangle-cube intersection using best available method."""
    if HAS_NUMBA:
        return _triangle_cube_intersection_numba(
            v0[0], v0[1], v0[2],
            v1[0], v1[1], v1[2],
            v2[0], v2[1], v2[2],
            cube_min[0], cube_min[1], cube_min[2],
            cube_max[0], cube_max[1], cube_max[2]
        )
    else:
        return triangle_cube_intersection(v0, v1, v2, cube_min, cube_max)


class MultifractalAnalyzer3D:
    """
    Multifractal analysis for 3D triangulated surface meshes.

    Uses cube-based partition functions with the Separating Axis Theorem
    (Akenine-Moller) for triangle-cube intersection testing.

    Supports two measure types:
    - 'count': triangle count per cube (interface complexity density)
    - 'area': triangle area distributed among cubes (surface area density)
    """

    def __init__(self, measure: str = 'area', verbose: bool = True):
        """
        Initialize 3D multifractal analyzer.

        Args:
            measure: Measure type - 'count' or 'area'
            verbose: Print progress information
        """
        if measure not in ('count', 'area'):
            raise ValueError(f"measure must be 'count' or 'area', got '{measure}'")
        self.measure = measure
        self.verbose = verbose
        self._D1_direct = np.nan  # Storage for q=1 result

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def compute_cube_measures(self, mesh: TriangleMesh, delta: float,
                              domain: Optional[BoundingBox3D] = None
                              ) -> Tuple[np.ndarray, int]:
        """
        Compute the measure mu_i for each occupied cube at scale delta.

        For 'area' measure: distributes each triangle's area equally
        among the cubes it intersects, preserving total surface area.

        For 'count' measure: counts the number of triangles intersecting
        each cube.

        Args:
            mesh: Surface mesh to analyze
            delta: Cube side length
            domain: Optional custom domain (defaults to mesh bounding box)

        Returns:
            (measures, n_occupied): Array of non-zero measures and count
        """
        if domain is None:
            domain = mesh.bbox

        nx = max(1, int(np.ceil(domain.width / delta)))
        ny = max(1, int(np.ceil(domain.height / delta)))
        nz = max(1, int(np.ceil(domain.depth / delta)))

        cube_measures = {}
        all_verts = mesh.get_all_triangle_vertices()
        tri_min = all_verts.min(axis=1)
        tri_max = all_verts.max(axis=1)
        inv_delta = 1.0 / delta

        if self.measure == 'area':
            areas = mesh.triangle_areas()

        for t_idx in range(mesh.n_triangles):
            v0 = all_verts[t_idx, 0]
            v1 = all_verts[t_idx, 1]
            v2 = all_verts[t_idx, 2]

            i_min = max(0, int((tri_min[t_idx, 0] - domain.min_x) * inv_delta))
            i_max = min(nx - 1, int((tri_max[t_idx, 0] - domain.min_x) * inv_delta))
            j_min = max(0, int((tri_min[t_idx, 1] - domain.min_y) * inv_delta))
            j_max = min(ny - 1, int((tri_max[t_idx, 1] - domain.min_y) * inv_delta))
            k_min = max(0, int((tri_min[t_idx, 2] - domain.min_z) * inv_delta))
            k_max = min(nz - 1, int((tri_max[t_idx, 2] - domain.min_z) * inv_delta))

            if self.measure == 'area':
                area = areas[t_idx]
                if area <= 0:
                    continue
                # First pass: find intersecting cubes
                intersecting = []
                for i in range(i_min, i_max + 1):
                    for j in range(j_min, j_max + 1):
                        for k in range(k_min, k_max + 1):
                            cube_min = np.array([
                                domain.min_x + i * delta,
                                domain.min_y + j * delta,
                                domain.min_z + k * delta
                            ])
                            cube_max = cube_min + delta
                            if _test_intersection(v0, v1, v2, cube_min, cube_max):
                                intersecting.append((i, j, k))
                # Distribute area equally among intersecting cubes
                if intersecting:
                    area_per_cube = area / len(intersecting)
                    for key in intersecting:
                        cube_measures[key] = cube_measures.get(key, 0.0) + area_per_cube
            else:
                # count measure
                for i in range(i_min, i_max + 1):
                    for j in range(j_min, j_max + 1):
                        for k in range(k_min, k_max + 1):
                            cube_min = np.array([
                                domain.min_x + i * delta,
                                domain.min_y + j * delta,
                                domain.min_z + k * delta
                            ])
                            cube_max = cube_min + delta
                            if _test_intersection(v0, v1, v2, cube_min, cube_max):
                                key = (i, j, k)
                                cube_measures[key] = cube_measures.get(key, 0) + 1

        measures = np.array(list(cube_measures.values()), dtype=np.float64)
        return measures, len(measures)

    def compute_multifractal_spectrum(
        self,
        mesh: TriangleMesh,
        q_values: Optional[np.ndarray] = None,
        delta_factor: float = 1.5,
        num_scales: int = 12,
        min_delta: Optional[float] = None,
        max_delta: Optional[float] = None,
    ) -> MultifractalResult3D:
        """
        Compute the full multifractal spectrum of a 3D surface.

        Args:
            mesh: TriangleMesh to analyze
            q_values: Array of q moments (default: -5 to 5 in steps of 0.5)
            delta_factor: Geometric ratio between successive cube sizes
            num_scales: Number of scales to analyze
            min_delta: Minimum cube size (default: char_length / 200)
            max_delta: Maximum cube size (default: char_length / 2)

        Returns:
            MultifractalResult3D with complete spectrum
        """
        if q_values is None:
            q_values = np.arange(-5, 5.1, 0.5)
        q_values = np.asarray(q_values, dtype=np.float64)

        char_len = mesh.bbox.characteristic_length

        if min_delta is None:
            min_delta = char_len / 200
        if max_delta is None:
            max_delta = char_len / 2

        # Generate geometric sequence of cube sizes (large to small)
        cube_sizes = []
        delta = max_delta
        for _ in range(num_scales):
            if delta < min_delta:
                break
            cube_sizes.append(delta)
            delta /= delta_factor
        cube_sizes = np.array(cube_sizes)
        n_scales = len(cube_sizes)

        if n_scales < 3:
            raise ValueError(f"Insufficient scale range: {n_scales} scales "
                             f"(need >= 3). Adjust min/max_delta or delta_factor.")

        self._log(f"Multifractal analysis: {n_scales} scales, "
                  f"{len(q_values)} q-values, measure='{self.measure}'")
        self._log(f"  Cube sizes: {cube_sizes[0]:.6f} to {cube_sizes[-1]:.6f}")
        self._log(f"  Mesh: {mesh.n_triangles} triangles, "
                  f"bbox: {mesh.bbox.width:.4f} x {mesh.bbox.height:.4f} x {mesh.bbox.depth:.4f}")

        # Step 1: Compute measures at each scale
        all_probabilities = []
        n_occupied_list = []
        t_start = time.time()

        for s_idx, delta in enumerate(cube_sizes):
            t0 = time.time()
            measures, n_occ = self.compute_cube_measures(mesh, delta)
            n_occupied_list.append(n_occ)

            if measures.size > 0:
                total = measures.sum()
                probabilities = measures / total
            else:
                probabilities = np.array([])

            all_probabilities.append(probabilities)
            self._log(f"  Scale {s_idx+1}/{n_scales}: delta={delta:.6f}, "
                      f"occupied={n_occ}, dt={time.time()-t0:.1f}s")

        self._log(f"  Cube counting total: {time.time()-t_start:.1f}s")

        # Step 2: Compute tau(q) via partition function regression
        taus = np.full(len(q_values), np.nan)
        r_squared = np.full(len(q_values), np.nan)

        for q_idx, q in enumerate(q_values):
            if abs(q - 1.0) < 1e-8:
                # q=1: information dimension via L'Hopital's rule
                # D_1 = lim_{eps->0} H(eps) / ln(1/eps)
                # H(eps) = -sum(p_i * ln(p_i))
                # Regress H vs ln(eps): slope gives D_1 = -slope
                entropy = np.full(n_scales, np.nan)
                for s, probs in enumerate(all_probabilities):
                    if probs.size > 0:
                        mask = probs > 1e-15
                        if np.any(mask):
                            entropy[s] = -np.sum(probs[mask] * np.log(probs[mask]))

                valid = ~np.isnan(entropy)
                if np.sum(valid) >= 3:
                    log_eps = np.log(cube_sizes[valid])
                    slope, _, r_val, _, _ = stats.linregress(log_eps, entropy[valid])
                    # tau(1) = 0 by definition (Z_1 = 1)
                    taus[q_idx] = 0.0
                    r_squared[q_idx] = r_val ** 2
                    self._D1_direct = -slope
            else:
                # General q: Z_q(epsilon) = sum(p_i^q)
                Z_q = np.full(n_scales, np.nan)
                for s, probs in enumerate(all_probabilities):
                    if probs.size > 0:
                        mask = probs > 1e-15
                        if np.any(mask):
                            Z_q[s] = np.sum(probs[mask] ** q)

                valid = ~np.isnan(Z_q) & (Z_q > 0)
                if np.sum(valid) >= 3:
                    log_eps = np.log(cube_sizes[valid])
                    log_Zq = np.log(Z_q[valid])
                    slope, _, r_val, _, _ = stats.linregress(log_eps, log_Zq)
                    taus[q_idx] = slope
                    r_squared[q_idx] = r_val ** 2

        # Step 3: Generalized dimensions D_q = tau(q) / (q - 1)
        Dq = np.full(len(q_values), np.nan)
        for i, q in enumerate(q_values):
            if np.isnan(taus[i]):
                continue
            if abs(q - 1.0) < 1e-8:
                Dq[i] = self._D1_direct
            else:
                Dq[i] = taus[i] / (q - 1)

        # Step 4: Singularity spectrum f(alpha) via Legendre transform
        alpha = np.full(len(q_values), np.nan)
        f_alpha_arr = np.full(len(q_values), np.nan)

        for i in range(len(q_values)):
            if np.isnan(taus[i]):
                continue
            # alpha = -d(tau)/dq (numerical derivative)
            if 0 < i < len(q_values) - 1:
                if not np.isnan(taus[i-1]) and not np.isnan(taus[i+1]):
                    alpha[i] = -(taus[i+1] - taus[i-1]) / (q_values[i+1] - q_values[i-1])
            elif i == 0:
                if not np.isnan(taus[i+1]):
                    alpha[i] = -(taus[i+1] - taus[i]) / (q_values[i+1] - q_values[i])
            else:
                if not np.isnan(taus[i-1]):
                    alpha[i] = -(taus[i] - taus[i-1]) / (q_values[i] - q_values[i-1])

            if not np.isnan(alpha[i]):
                f_alpha_arr[i] = q_values[i] * alpha[i] + taus[i]

        # Step 5: Extract key dimensions
        def _get_Dq(q_target):
            idx = np.argmin(np.abs(q_values - q_target))
            if abs(q_values[idx] - q_target) < 0.01:
                return Dq[idx]
            return np.nan

        D0 = _get_Dq(0.0)
        D1 = _get_Dq(1.0)
        D2 = _get_Dq(2.0)

        valid_alpha = alpha[~np.isnan(alpha)]
        alpha_width = (valid_alpha.max() - valid_alpha.min()) if len(valid_alpha) >= 2 else np.nan

        valid_Dq = Dq[~np.isnan(Dq)]
        degree_mf = (valid_Dq.max() - valid_Dq.min()) if len(valid_Dq) >= 3 else np.nan

        self._log(f"\nMultifractal spectrum (measure='{self.measure}'):")
        self._log(f"  D(0) = {D0:.4f}  (capacity dimension)")
        self._log(f"  D(1) = {D1:.4f}  (information dimension)")
        self._log(f"  D(2) = {D2:.4f}  (correlation dimension)")
        self._log(f"  alpha width = {alpha_width:.4f}")
        self._log(f"  Degree of multifractality = {degree_mf:.4f}")

        if not np.isnan(degree_mf):
            if degree_mf > 0.1:
                self._log(f"  -> Surface shows multifractal behavior")
            else:
                self._log(f"  -> Surface appears monofractal")

        return MultifractalResult3D(
            q_values=q_values,
            tau=taus,
            Dq=Dq,
            alpha=alpha,
            f_alpha=f_alpha_arr,
            r_squared=r_squared,
            D0=D0,
            D1=D1,
            D2=D2,
            alpha_width=alpha_width,
            degree_multifractality=degree_mf,
            measure_type=self.measure,
            cube_sizes=cube_sizes,
            n_occupied=n_occupied_list,
        )

    # Legacy API compatibility
    def analyze(self, mesh, initial_delta=None, delta_factor=1.5, num_steps=12):
        """Legacy interface - calls compute_multifractal_spectrum."""
        return self.compute_multifractal_spectrum(
            mesh, delta_factor=delta_factor, num_scales=num_steps,
            max_delta=initial_delta,
        )

    def save_results(self, result: MultifractalResult3D, output_dir: str,
                     label: str = ''):
        """Save multifractal results to CSV files."""
        os.makedirs(output_dir, exist_ok=True)
        prefix = f"{label}_" if label else ""

        spec_file = os.path.join(output_dir, f"{prefix}multifractal_spectrum_3d.csv")
        with open(spec_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['q', 'tau_q', 'D_q', 'alpha', 'f_alpha', 'R_squared'])
            for i in range(len(result.q_values)):
                writer.writerow([
                    f"{result.q_values[i]:.2f}",
                    f"{result.tau[i]:.6f}" if not np.isnan(result.tau[i]) else 'NaN',
                    f"{result.Dq[i]:.6f}" if not np.isnan(result.Dq[i]) else 'NaN',
                    f"{result.alpha[i]:.6f}" if not np.isnan(result.alpha[i]) else 'NaN',
                    f"{result.f_alpha[i]:.6f}" if not np.isnan(result.f_alpha[i]) else 'NaN',
                    f"{result.r_squared[i]:.6f}" if not np.isnan(result.r_squared[i]) else 'NaN',
                ])

        params_file = os.path.join(output_dir, f"{prefix}multifractal_params_3d.csv")
        with open(params_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['parameter', 'value'])
            writer.writerow(['measure', result.measure_type])
            writer.writerow(['D0', f"{result.D0:.6f}"])
            writer.writerow(['D1', f"{result.D1:.6f}"])
            writer.writerow(['D2', f"{result.D2:.6f}"])
            writer.writerow(['alpha_width', f"{result.alpha_width:.6f}"])
            writer.writerow(['degree_multifractality', f"{result.degree_multifractality:.6f}"])
            writer.writerow(['n_scales', len(result.cube_sizes)])
            writer.writerow(['delta_min', f"{result.cube_sizes[-1]:.6f}"])
            writer.writerow(['delta_max', f"{result.cube_sizes[0]:.6f}"])

        scale_file = os.path.join(output_dir, f"{prefix}multifractal_scales_3d.csv")
        with open(scale_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['delta', 'n_occupied'])
            for i in range(len(result.cube_sizes)):
                writer.writerow([f"{result.cube_sizes[i]:.6f}", result.n_occupied[i]])

        self._log(f"Results saved to {output_dir}")


def analyze_multifractal_3d(
    filename: str,
    measure: str = 'area',
    q_values: Optional[np.ndarray] = None,
    delta_factor: float = 1.5,
    num_scales: int = 12,
    output_dir: Optional[str] = None,
    verbose: bool = True,
) -> MultifractalResult3D:
    """
    Convenience function: load mesh and compute multifractal spectrum.

    Args:
        filename: Path to mesh file (.vtk, .vtp, .stl)
        measure: 'count' or 'area'
        q_values: Array of q moments (default: -5 to 5)
        delta_factor: Geometric ratio between cube sizes
        num_scales: Number of scales
        output_dir: Directory to save results (optional)
        verbose: Print progress

    Returns:
        MultifractalResult3D
    """
    from .mesh_io import load_mesh

    if verbose:
        print(f"Loading mesh: {filename}")

    mesh = load_mesh(filename)

    if verbose:
        print(f"  Vertices: {mesh.n_vertices}")
        print(f"  Triangles: {mesh.n_triangles}")
        print(f"  Surface area: {mesh.surface_area:.6f}")

    analyzer = MultifractalAnalyzer3D(measure=measure, verbose=verbose)
    result = analyzer.compute_multifractal_spectrum(
        mesh,
        q_values=q_values,
        delta_factor=delta_factor,
        num_scales=num_scales,
    )

    if output_dir:
        analyzer.save_results(result, output_dir)

    return result


# Legacy alias
analyze_mesh_multifractal = analyze_multifractal_3d


def main():
    """Command-line interface for 3D multifractal analysis."""
    import argparse

    parser = argparse.ArgumentParser(
        description='3D Multifractal Analysis for Surface Meshes'
    )
    parser.add_argument('mesh_file', help='Path to mesh file (.vtk, .vtp, .stl)')
    parser.add_argument('--measure', choices=['count', 'area'], default='area',
                        help='Measure type (default: area)')
    parser.add_argument('--q-min', type=float, default=-5.0,
                        help='Minimum q value (default: -5.0)')
    parser.add_argument('--q-max', type=float, default=5.0,
                        help='Maximum q value (default: 5.0)')
    parser.add_argument('--dq', type=float, default=0.5,
                        help='q step size (default: 0.5)')
    parser.add_argument('--num-scales', type=int, default=12,
                        help='Number of scales (default: 12)')
    parser.add_argument('--delta-factor', type=float, default=1.5,
                        help='Scale factor between cube sizes (default: 1.5)')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress verbose output')
    parser.add_argument('--output-dir', '-o', type=str,
                        help='Output directory for CSV results')

    args = parser.parse_args()

    q_values = np.arange(args.q_min, args.q_max + args.dq/2, args.dq)

    result = analyze_multifractal_3d(
        args.mesh_file,
        measure=args.measure,
        q_values=q_values,
        num_scales=args.num_scales,
        delta_factor=args.delta_factor,
        output_dir=args.output_dir,
        verbose=not args.quiet,
    )

    return 0 if result is not None else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
