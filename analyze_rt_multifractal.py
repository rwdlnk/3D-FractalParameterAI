#!/usr/bin/env python3
"""
Analyze 3D RT interface data for multifractal properties.

Processes all timesteps in an OpenFOAM postProcessing/extractInterface directory
and computes generalized dimensions D_q for each timestep.
"""

import os
import sys
import argparse
import csv
import numpy as np
from pathlib import Path

# Add the package to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fractal3d.mesh_io import load_mesh
from fractal3d.multifractal_3d import MultifractalAnalyzer3D


def find_vtk_files(interface_dir: str):
    """Find all VTK files in timestep directories."""
    interface_path = Path(interface_dir)

    if not interface_path.exists():
        print(f"Error: Directory not found: {interface_dir}")
        return []

    # Find all timestep directories
    timesteps = []
    for item in interface_path.iterdir():
        if item.is_dir():
            try:
                t = float(item.name)
                # Look for VTK file in directory
                vtk_files = list(item.glob("*.vtk")) + list(item.glob("*.vtp"))
                if vtk_files:
                    timesteps.append((t, str(vtk_files[0])))
            except ValueError:
                continue

    # Sort by timestep
    timesteps.sort(key=lambda x: x[0])
    return timesteps


def main():
    parser = argparse.ArgumentParser(
        description='Analyze RT interfaces for multifractal properties'
    )
    parser.add_argument('interface_dir',
                        help='Path to extractInterface directory')
    parser.add_argument('--output', '-o', type=str, default='multifractal_results.csv',
                        help='Output CSV file (default: multifractal_results.csv)')
    parser.add_argument('--q-min', type=float, default=-5.0,
                        help='Minimum q value (default: -5.0)')
    parser.add_argument('--q-max', type=float, default=5.0,
                        help='Maximum q value (default: 5.0)')
    parser.add_argument('--n-q', type=int, default=21,
                        help='Number of q values (default: 21)')
    parser.add_argument('--num-steps', type=int, default=12,
                        help='Number of scales (default: 12)')
    parser.add_argument('--skip', type=int, default=1,
                        help='Process every Nth timestep (default: 1)')

    args = parser.parse_args()

    # Find all VTK files
    timesteps = find_vtk_files(args.interface_dir)

    if not timesteps:
        print("No VTK files found!")
        return 1

    print(f"Found {len(timesteps)} timesteps")

    # Apply skip
    if args.skip > 1:
        timesteps = timesteps[::args.skip]
        print(f"Processing every {args.skip}th timestep: {len(timesteps)} files")

    # Initialize analyzer
    analyzer = MultifractalAnalyzer3D(
        q_min=args.q_min,
        q_max=args.q_max,
        n_q=args.n_q,
        debug=False
    )

    # Results storage
    results = []

    # Process each timestep
    for i, (t, vtk_file) in enumerate(timesteps):
        print(f"\n[{i+1}/{len(timesteps)}] t = {t:.1f}s: {os.path.basename(vtk_file)}")

        try:
            # Load mesh
            mesh = load_mesh(vtk_file)
            print(f"   Triangles: {mesh.n_triangles}, Area: {mesh.surface_area:.4f}")

            # Compute multifractal spectrum
            spectrum = analyzer.analyze(mesh, num_steps=args.num_steps)

            if spectrum:
                result = {
                    'time': t,
                    'n_triangles': mesh.n_triangles,
                    'surface_area': mesh.surface_area,
                    'd0': spectrum.d0,
                    'd1': spectrum.d1,
                    'd2': spectrum.d2,
                    'd_inf': spectrum.d_inf,
                    'd_minus_inf': spectrum.d_minus_inf,
                    'alpha_min': spectrum.alpha_min,
                    'alpha_max': spectrum.alpha_max,
                    'alpha_0': spectrum.alpha_0,
                    'spectrum_width': spectrum.spectrum_width,
                    'spectrum_asymmetry': spectrum.spectrum_asymmetry,
                    'mean_r_squared': spectrum.mean_r_squared,
                    'analysis_time': spectrum.analysis_time
                }
                results.append(result)

                print(f"   D_0={spectrum.d0:.4f}, D_1={spectrum.d1:.4f}, D_2={spectrum.d2:.4f}")
                print(f"   Width={spectrum.spectrum_width:.4f}, R²={spectrum.mean_r_squared:.4f}")
            else:
                print(f"   Warning: Analysis failed")

        except Exception as e:
            print(f"   Error: {e}")
            continue

    # Save results
    if results:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

        print(f"\n{'='*60}")
        print(f"Results saved to: {args.output}")
        print(f"Processed {len(results)} timesteps")

        # Summary statistics
        d0_values = [r['d0'] for r in results]
        d1_values = [r['d1'] for r in results]
        d2_values = [r['d2'] for r in results]
        width_values = [r['spectrum_width'] for r in results]

        print(f"\nSummary:")
        print(f"   D_0: {np.min(d0_values):.4f} - {np.max(d0_values):.4f}")
        print(f"   D_1: {np.min(d1_values):.4f} - {np.max(d1_values):.4f}")
        print(f"   D_2: {np.min(d2_values):.4f} - {np.max(d2_values):.4f}")
        print(f"   Spectrum width: {np.min(width_values):.4f} - {np.max(width_values):.4f}")
        print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
