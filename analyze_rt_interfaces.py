#!/usr/bin/env python3
"""
Analyze 3D RT interface isosurfaces for fractal dimension.

This script computes the fractal dimension and surface features for
VTP files exported from OpenFOAM/ParaView simulations.

Usage:
    # Analyze all VTP files in a directory
    python analyze_rt_interfaces.py /path/to/isosurfaces/

    # Analyze specific time range
    python analyze_rt_interfaces.py /path/to/isosurfaces/ --t-min 5.0 --t-max 15.0

    # Analyze a single file
    python analyze_rt_interfaces.py /path/to/interface_t5.0000.vtp

    # Save results to CSV
    python analyze_rt_interfaces.py /path/to/isosurfaces/ --output results.csv

    # Generate plot
    python analyze_rt_interfaces.py /path/to/isosurfaces/ --plot evolution.png
"""

import argparse
import os
import sys
import re
import time
import csv
from typing import List, Tuple, Optional, Dict
import numpy as np
from scipy import stats

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fractal3d import (
    load_mesh,
    extract_surface_features,
    FastCubeCounter,
    check_numba_available,
)


def extract_time_from_filename(filename: str) -> Optional[float]:
    """Extract time value from filename like 'interface_t5.0000.vtp'."""
    match = re.search(r'_t(\d+\.?\d*)', filename)
    if match:
        return float(match.group(1))
    return None


def find_vtp_files(path: str, t_min: Optional[float] = None,
                   t_max: Optional[float] = None) -> List[Tuple[float, str]]:
    """
    Find VTP files in directory, optionally filtering by time range.

    Returns:
        List of (time, filepath) tuples, sorted by time
    """
    if os.path.isfile(path):
        # Single file
        t = extract_time_from_filename(path)
        if t is not None:
            return [(t, path)]
        return [(0.0, path)]

    # Directory
    files = []
    for f in os.listdir(path):
        if f.endswith('.vtp'):
            t = extract_time_from_filename(f)
            if t is not None:
                # Apply time filter
                if t_min is not None and t < t_min:
                    continue
                if t_max is not None and t > t_max:
                    continue
                files.append((t, os.path.join(path, f)))

    return sorted(files, key=lambda x: x[0])


def analyze_single_file(filepath: str, num_scales: int = 10,
                        compute_curvature: bool = False,
                        verbose: bool = True) -> Dict:
    """
    Analyze a single VTP file.

    Returns:
        Dictionary with analysis results
    """
    if verbose:
        print(f"  Loading mesh...", end=" ", flush=True)

    t0 = time.time()
    mesh = load_mesh(filepath)
    load_time = time.time() - t0

    if verbose:
        print(f"{mesh.n_triangles:,} triangles ({load_time:.1f}s)")

    # Extract features
    if verbose:
        print(f"  Extracting features...", end=" ", flush=True)

    t0 = time.time()
    features = extract_surface_features(mesh, compute_curvature=compute_curvature)
    feature_time = time.time() - t0

    if verbose:
        print(f"({feature_time:.1f}s)")

    # Compute fractal dimension
    if verbose:
        print(f"  Computing fractal dimension...", end=" ", flush=True)

    t0 = time.time()
    counter = FastCubeCounter(mesh)
    char_len = mesh.bbox.characteristic_length

    deltas = []
    n_boxes = []
    delta = char_len / 5  # Start coarser

    for i in range(num_scales):
        result = counter.count_cubes(delta)
        if result.n_boxes > 0:
            deltas.append(delta)
            n_boxes.append(result.n_boxes)
        delta /= 1.5

    # Linear regression
    if len(deltas) >= 3:
        log_inv_delta = np.log(1.0 / np.array(deltas))
        log_n = np.log(np.array(n_boxes))
        slope, intercept, r, p, se = stats.linregress(log_inv_delta, log_n)
        D = slope
        R2 = r ** 2
    else:
        D = float('nan')
        R2 = float('nan')
        se = float('nan')

    fractal_time = time.time() - t0

    if verbose:
        print(f"D={D:.4f}, R²={R2:.5f} ({fractal_time:.1f}s)")

    return {
        'filepath': filepath,
        'n_vertices': mesh.n_vertices,
        'n_triangles': mesh.n_triangles,
        'surface_area': mesh.surface_area,
        'bbox_width': mesh.bbox.width,
        'bbox_height': mesh.bbox.height,
        'bbox_depth': mesh.bbox.depth,
        'char_length': char_len,
        'n_components': features.n_components,
        'fragmentation': features.fragmentation_index,
        'compactness': features.compactness,
        'roughness': features.roughness,
        'sphericity': features.sphericity,
        'complexity_score': features.complexity_score,
        'surface_type': features.surface_type,
        'fractal_dimension': D,
        'r_squared': R2,
        'std_error': se,
        'deltas': deltas,
        'n_boxes': n_boxes,
        'load_time': load_time,
        'feature_time': feature_time,
        'fractal_time': fractal_time,
    }


def analyze_directory(path: str, t_min: Optional[float] = None,
                      t_max: Optional[float] = None,
                      num_scales: int = 10,
                      compute_curvature: bool = False,
                      verbose: bool = True) -> List[Dict]:
    """
    Analyze all VTP files in a directory.

    Returns:
        List of result dictionaries, one per file
    """
    files = find_vtp_files(path, t_min, t_max)

    if not files:
        print(f"No VTP files found in {path}")
        return []

    if verbose:
        print(f"Found {len(files)} VTP files")
        print(f"Time range: {files[0][0]:.4f} to {files[-1][0]:.4f}")
        print(f"Numba acceleration: {check_numba_available()}")
        print()

    results = []

    for i, (t, filepath) in enumerate(files):
        if verbose:
            print(f"[{i+1}/{len(files)}] t = {t:.4f}")

        result = analyze_single_file(
            filepath,
            num_scales=num_scales,
            compute_curvature=compute_curvature,
            verbose=verbose
        )
        result['time'] = t
        results.append(result)

        if verbose:
            print()

    return results


def save_results_csv(results: List[Dict], output_path: str) -> None:
    """Save results to CSV file."""
    if not results:
        return

    # Define columns to save
    columns = [
        'time', 'n_triangles', 'n_components', 'fractal_dimension',
        'r_squared', 'std_error', 'fragmentation', 'compactness',
        'roughness', 'complexity_score', 'surface_type',
        'surface_area', 'char_length', 'bbox_width', 'bbox_height', 'bbox_depth'
    ]

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)

    print(f"Results saved to {output_path}")


def plot_results(results: List[Dict], output_path: str) -> None:
    """Generate plot of fractal dimension vs time."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    times = [r['time'] for r in results]
    D_values = [r['fractal_dimension'] for r in results]
    components = [r['n_components'] for r in results]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Fractal dimension
    ax1.plot(times, D_values, 'b-o', linewidth=2, markersize=4)
    ax1.axhline(y=2.0, color='gray', linestyle='--', label='D=2.0 (smooth surface)')
    ax1.set_ylabel('Fractal Dimension D', fontsize=12)
    ax1.set_ylim(1.8, 3.0)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title('3D RT Interface Temporal Evolution', fontsize=14)

    # Number of components
    ax2.plot(times, components, 'r-o', linewidth=2, markersize=4)
    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_ylabel('Connected Components', fontsize=12)
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Plot saved to {output_path}")


def print_summary(results: List[Dict]) -> None:
    """Print summary table of results."""
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Time':>8} {'Triangles':>10} {'Comps':>7} {'D':>8} {'R²':>9} {'Type':>20}")
    print("-" * 80)

    for r in results:
        print(f"{r['time']:>8.2f} {r['n_triangles']:>10,} {r['n_components']:>7} "
              f"{r['fractal_dimension']:>8.4f} {r['r_squared']:>9.5f} {r['surface_type']:>20}")

    print("=" * 80)

    # Statistics
    D_values = [r['fractal_dimension'] for r in results if not np.isnan(r['fractal_dimension'])]
    if D_values:
        print(f"\nFractal Dimension Statistics:")
        print(f"  Min:  {min(D_values):.4f}")
        print(f"  Max:  {max(D_values):.4f}")
        print(f"  Mean: {np.mean(D_values):.4f}")
        print(f"  Std:  {np.std(D_values):.4f}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze 3D RT interface isosurfaces for fractal dimension.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('path', help='Path to VTP file or directory containing VTP files')
    parser.add_argument('--t-min', type=float, default=None,
                        help='Minimum time to analyze')
    parser.add_argument('--t-max', type=float, default=None,
                        help='Maximum time to analyze')
    parser.add_argument('--num-scales', type=int, default=10,
                        help='Number of scales for box counting (default: 10)')
    parser.add_argument('--curvature', action='store_true',
                        help='Compute curvature statistics (slower)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output CSV file for results')
    parser.add_argument('--plot', '-p', type=str, default=None,
                        help='Output plot file (PNG)')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Reduce output verbosity')

    args = parser.parse_args()

    verbose = not args.quiet

    if verbose:
        print("=" * 80)
        print("3D RT INTERFACE FRACTAL ANALYSIS")
        print("=" * 80)
        print(f"Path: {args.path}")
        if args.t_min is not None or args.t_max is not None:
            print(f"Time range: {args.t_min} to {args.t_max}")
        print()

    # Run analysis
    if os.path.isfile(args.path):
        # Single file
        t = extract_time_from_filename(args.path)
        result = analyze_single_file(
            args.path,
            num_scales=args.num_scales,
            compute_curvature=args.curvature,
            verbose=verbose
        )
        result['time'] = t if t is not None else 0.0
        results = [result]
    else:
        # Directory
        results = analyze_directory(
            args.path,
            t_min=args.t_min,
            t_max=args.t_max,
            num_scales=args.num_scales,
            compute_curvature=args.curvature,
            verbose=verbose
        )

    if not results:
        return 1

    # Print summary
    if verbose:
        print_summary(results)

    # Save CSV
    if args.output:
        save_results_csv(results, args.output)

    # Generate plot
    if args.plot and len(results) > 1:
        plot_results(results, args.plot)

    return 0


if __name__ == '__main__':
    sys.exit(main())
