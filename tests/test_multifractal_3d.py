#!/usr/bin/env python3
"""
Tests for 3D multifractal analysis.

Validates that:
1. A smooth sphere is monofractal (D_q ~ 2.0 for all q, narrow alpha width)
2. A rough surface shows broader multifractal spectrum
3. Both 'count' and 'area' measures produce valid results
4. Results can be saved to CSV
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fractal3d.multifractal_3d import MultifractalAnalyzer3D
from tests.test_box_counting_3d import create_sphere_mesh, create_rough_surface


def test_sphere_monofractal():
    """Smooth sphere should be approximately monofractal with D_q ~ 2.0."""
    print("Testing sphere multifractal spectrum (expect monofractal, D~2.0)...")

    mesh = create_sphere_mesh(radius=1.0, n_subdivisions=3)
    print(f"  Mesh: {mesh.n_triangles} triangles")

    analyzer = MultifractalAnalyzer3D(measure='area', verbose=False)
    result = analyzer.compute_multifractal_spectrum(
        mesh,
        q_values=np.arange(-3, 3.1, 1.0),
        num_scales=8,
        delta_factor=1.5,
    )

    print(f"  D(0) = {result.D0:.4f}")
    print(f"  D(1) = {result.D1:.4f}")
    print(f"  D(2) = {result.D2:.4f}")
    print(f"  alpha width = {result.alpha_width:.4f}")
    print(f"  Degree of multifractality = {result.degree_multifractality:.4f}")

    # D(0) should be close to 2.0 for a smooth sphere
    assert 1.8 < result.D0 < 2.5, f"D(0) = {result.D0:.4f}, expected ~2.0"

    # All R-squared values should be reasonable
    valid_r2 = result.r_squared[~np.isnan(result.r_squared)]
    assert len(valid_r2) > 0, "No valid R-squared values"
    print(f"  Mean R-squared = {np.mean(valid_r2):.4f}")

    print("  PASSED!")


def test_rough_surface_spectrum():
    """Rough surface should show broader multifractal spectrum."""
    print("\nTesting rough surface multifractal spectrum...")

    mesh = create_rough_surface(size=2.0, n_divisions=50, roughness=0.15, n_octaves=5)
    print(f"  Mesh: {mesh.n_triangles} triangles")

    analyzer = MultifractalAnalyzer3D(measure='area', verbose=False)
    result = analyzer.compute_multifractal_spectrum(
        mesh,
        q_values=np.arange(-3, 3.1, 1.0),
        num_scales=8,
        delta_factor=1.5,
    )

    print(f"  D(0) = {result.D0:.4f}")
    print(f"  D(1) = {result.D1:.4f}")
    print(f"  D(2) = {result.D2:.4f}")
    print(f"  alpha width = {result.alpha_width:.4f}")
    print(f"  Degree of multifractality = {result.degree_multifractality:.4f}")

    # D(0) should be > 2.0 for a rough surface
    assert result.D0 > 2.0, f"D(0) = {result.D0:.4f}, expected > 2.0"

    # All R-squared values should be reasonable
    valid_r2 = result.r_squared[~np.isnan(result.r_squared)]
    assert np.all(valid_r2 > 0.8), f"Poor R-squared values: {valid_r2}"

    print("  PASSED!")


def test_count_measure():
    """Test that 'count' measure produces valid results."""
    print("\nTesting count measure...")

    mesh = create_sphere_mesh(radius=1.0, n_subdivisions=3)

    analyzer = MultifractalAnalyzer3D(measure='count', verbose=False)
    result = analyzer.compute_multifractal_spectrum(
        mesh,
        q_values=np.arange(-2, 2.1, 1.0),
        num_scales=6,
        delta_factor=1.5,
    )

    print(f"  D(0) = {result.D0:.4f}")
    print(f"  measure_type = {result.measure_type}")

    assert result.measure_type == 'count'
    assert 1.7 < result.D0 < 2.5, f"D(0) = {result.D0:.4f} with count measure"

    print("  PASSED!")


def test_area_measure():
    """Test that 'area' measure produces valid results."""
    print("\nTesting area measure...")

    mesh = create_sphere_mesh(radius=1.0, n_subdivisions=3)

    analyzer = MultifractalAnalyzer3D(measure='area', verbose=False)
    result = analyzer.compute_multifractal_spectrum(
        mesh,
        q_values=np.arange(-2, 2.1, 1.0),
        num_scales=6,
        delta_factor=1.5,
    )

    print(f"  D(0) = {result.D0:.4f}")
    print(f"  measure_type = {result.measure_type}")

    assert result.measure_type == 'area'
    assert 1.7 < result.D0 < 2.5, f"D(0) = {result.D0:.4f} with area measure"

    print("  PASSED!")


def test_save_results(tmp_dir='/tmp/test_mf3d'):
    """Test saving results to CSV."""
    print("\nTesting result saving...")

    mesh = create_sphere_mesh(radius=1.0, n_subdivisions=2)
    analyzer = MultifractalAnalyzer3D(measure='area', verbose=False)
    result = analyzer.compute_multifractal_spectrum(
        mesh,
        q_values=np.arange(-2, 2.1, 1.0),
        num_scales=5,
        delta_factor=1.5,
    )

    analyzer.save_results(result, tmp_dir, label='test')

    expected_files = [
        'test_multifractal_spectrum_3d.csv',
        'test_multifractal_params_3d.csv',
        'test_multifractal_scales_3d.csv',
    ]
    for f in expected_files:
        path = os.path.join(tmp_dir, f)
        assert os.path.exists(path), f"Missing output: {path}"
        print(f"  Found: {f}")

    print("  PASSED!")


def test_legacy_alias():
    """Test that legacy MultifractalSpectrum3D alias works."""
    print("\nTesting legacy alias...")

    from fractal3d.multifractal_3d import MultifractalSpectrum3D, MultifractalResult3D
    assert MultifractalSpectrum3D is MultifractalResult3D
    print("  MultifractalSpectrum3D == MultifractalResult3D: OK")

    from fractal3d.multifractal_3d import analyze_mesh_multifractal, analyze_multifractal_3d
    assert analyze_mesh_multifractal is analyze_multifractal_3d
    print("  analyze_mesh_multifractal == analyze_multifractal_3d: OK")

    print("  PASSED!")


def run_all_tests():
    """Run all multifractal 3D tests."""
    print("=" * 60)
    print("3D Multifractal Analysis Tests")
    print("=" * 60)

    test_sphere_monofractal()
    test_rough_surface_spectrum()
    test_count_measure()
    test_area_measure()
    test_save_results()
    test_legacy_alias()

    print("\n" + "=" * 60)
    print("All multifractal 3D tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
