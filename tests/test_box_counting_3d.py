#!/usr/bin/env python3
"""
Tests for 3D box counting fractal dimension calculation.

Includes tests with synthetic surfaces of known dimension.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fractal3d.mesh_io import TriangleMesh, BoundingBox3D
from fractal3d.box_counting_3d import (
    triangle_cube_intersection,
    CubeCounter,
    compute_fractal_dimension_3d,
)


def create_sphere_mesh(radius: float = 1.0, n_subdivisions: int = 3) -> TriangleMesh:
    """
    Create a triangulated sphere mesh using icosahedron subdivision.

    A smooth sphere should have fractal dimension D ≈ 2.0.
    """
    # Start with icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio

    vertices = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1],
    ], dtype=np.float64)

    # Normalize to unit sphere
    vertices = vertices / np.linalg.norm(vertices[0])

    # Icosahedron faces
    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ], dtype=np.int64)

    # Subdivide faces
    for _ in range(n_subdivisions):
        vertices, faces = _subdivide_mesh(vertices, faces)

    # Scale to desired radius
    vertices = vertices * radius

    return TriangleMesh.from_vertices_and_faces(vertices, faces)


def _subdivide_mesh(vertices: np.ndarray, faces: np.ndarray):
    """Subdivide each triangle into 4 triangles."""
    edge_midpoints = {}
    new_vertices = list(vertices)
    new_faces = []

    def get_midpoint(i1, i2):
        """Get or create midpoint vertex between two vertices."""
        key = (min(i1, i2), max(i1, i2))
        if key in edge_midpoints:
            return edge_midpoints[key]

        v1 = vertices[i1]
        v2 = vertices[i2]
        midpoint = (v1 + v2) / 2
        midpoint = midpoint / np.linalg.norm(midpoint)  # Project to sphere

        idx = len(new_vertices)
        new_vertices.append(midpoint)
        edge_midpoints[key] = idx
        return idx

    for face in faces:
        v0, v1, v2 = face

        # Get midpoint indices
        m01 = get_midpoint(v0, v1)
        m12 = get_midpoint(v1, v2)
        m20 = get_midpoint(v2, v0)

        # Create 4 new triangles
        new_faces.extend([
            [v0, m01, m20],
            [v1, m12, m01],
            [v2, m20, m12],
            [m01, m12, m20],
        ])

    return np.array(new_vertices), np.array(new_faces, dtype=np.int64)


def create_plane_mesh(size: float = 2.0, n_divisions: int = 10) -> TriangleMesh:
    """
    Create a flat triangulated plane.

    A flat plane should have fractal dimension D = 2.0 exactly.
    """
    x = np.linspace(-size/2, size/2, n_divisions + 1)
    y = np.linspace(-size/2, size/2, n_divisions + 1)

    vertices = []
    for yi in y:
        for xi in x:
            vertices.append([xi, yi, 0.0])

    vertices = np.array(vertices, dtype=np.float64)

    faces = []
    for j in range(n_divisions):
        for i in range(n_divisions):
            # Vertex indices for this grid cell
            v00 = j * (n_divisions + 1) + i
            v10 = v00 + 1
            v01 = v00 + (n_divisions + 1)
            v11 = v01 + 1

            # Two triangles per cell
            faces.append([v00, v10, v11])
            faces.append([v00, v11, v01])

    faces = np.array(faces, dtype=np.int64)

    return TriangleMesh.from_vertices_and_faces(vertices, faces)


def create_rough_surface(size: float = 2.0, n_divisions: int = 50,
                          roughness: float = 0.1, n_octaves: int = 4) -> TriangleMesh:
    """
    Create a rough surface with fractal-like properties.

    Uses multiple octaves of noise to create self-similar roughness.
    The fractal dimension should be > 2.0.
    """
    x = np.linspace(-size/2, size/2, n_divisions + 1)
    y = np.linspace(-size/2, size/2, n_divisions + 1)
    X, Y = np.meshgrid(x, y)

    # Generate multi-octave noise
    Z = np.zeros_like(X)
    amplitude = roughness
    frequency = 1.0

    np.random.seed(42)  # Reproducible

    for _ in range(n_octaves):
        # Simple noise approximation using random heights at grid points
        noise = np.random.randn(n_divisions + 1, n_divisions + 1)
        Z += amplitude * noise
        amplitude *= 0.5
        frequency *= 2.0

    # Create vertices
    vertices = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    # Create faces
    faces = []
    for j in range(n_divisions):
        for i in range(n_divisions):
            v00 = j * (n_divisions + 1) + i
            v10 = v00 + 1
            v01 = v00 + (n_divisions + 1)
            v11 = v01 + 1

            faces.append([v00, v10, v11])
            faces.append([v00, v11, v01])

    faces = np.array(faces, dtype=np.int64)

    return TriangleMesh.from_vertices_and_faces(vertices, faces)


def test_triangle_cube_intersection():
    """Test the triangle-cube intersection function."""
    print("Testing triangle-cube intersection...")

    # Triangle fully inside cube
    v0 = np.array([0.2, 0.2, 0.2])
    v1 = np.array([0.8, 0.2, 0.2])
    v2 = np.array([0.5, 0.8, 0.2])
    cube_min = np.array([0.0, 0.0, 0.0])
    cube_max = np.array([1.0, 1.0, 1.0])
    assert triangle_cube_intersection(v0, v1, v2, cube_min, cube_max), \
        "Triangle inside cube should intersect"

    # Triangle completely outside cube
    v0 = np.array([2.0, 2.0, 2.0])
    v1 = np.array([3.0, 2.0, 2.0])
    v2 = np.array([2.5, 3.0, 2.0])
    assert not triangle_cube_intersection(v0, v1, v2, cube_min, cube_max), \
        "Triangle outside cube should not intersect"

    # Triangle crossing cube face
    v0 = np.array([-0.5, 0.5, 0.5])
    v1 = np.array([0.5, 0.5, 0.5])
    v2 = np.array([0.0, 0.5, -0.5])
    cube_min = np.array([0.0, 0.0, 0.0])
    cube_max = np.array([1.0, 1.0, 1.0])
    assert triangle_cube_intersection(v0, v1, v2, cube_min, cube_max), \
        "Triangle crossing cube should intersect"

    print("  All intersection tests passed!")


def test_plane_dimension():
    """Test that a flat plane has dimension ≈ 2.0."""
    print("\nTesting flat plane (expected D ≈ 2.0)...")

    mesh = create_plane_mesh(size=2.0, n_divisions=20)
    print(f"  Vertices: {mesh.n_vertices}, Triangles: {mesh.n_triangles}")

    result = compute_fractal_dimension_3d(
        mesh,
        initial_delta=0.5,
        delta_factor=1.4,
        num_steps=10,
    )

    print(f"  Fractal dimension: {result.dimension:.4f}")
    print(f"  R²: {result.r_squared:.6f}")

    # Plane should have D ≈ 2.0 (within tolerance)
    assert 1.9 < result.dimension < 2.1, \
        f"Plane dimension {result.dimension} not close to 2.0"
    print("  PASSED!")


def test_sphere_dimension():
    """Test that a sphere has dimension ≈ 2.0."""
    print("\nTesting sphere (expected D ≈ 2.0)...")

    mesh = create_sphere_mesh(radius=1.0, n_subdivisions=3)
    print(f"  Vertices: {mesh.n_vertices}, Triangles: {mesh.n_triangles}")

    result = compute_fractal_dimension_3d(
        mesh,
        initial_delta=0.3,
        delta_factor=1.4,
        num_steps=10,
    )

    print(f"  Fractal dimension: {result.dimension:.4f}")
    print(f"  R²: {result.r_squared:.6f}")

    # Sphere should have D ≈ 2.0
    assert 1.9 < result.dimension < 2.2, \
        f"Sphere dimension {result.dimension} not close to 2.0"
    print("  PASSED!")


def test_rough_surface_dimension():
    """Test that a rough surface has dimension > 2.0."""
    print("\nTesting rough surface (expected D > 2.0)...")

    mesh = create_rough_surface(size=2.0, n_divisions=50, roughness=0.15, n_octaves=5)
    print(f"  Vertices: {mesh.n_vertices}, Triangles: {mesh.n_triangles}")

    result = compute_fractal_dimension_3d(
        mesh,
        initial_delta=0.3,
        delta_factor=1.4,
        num_steps=12,
    )

    print(f"  Fractal dimension: {result.dimension:.4f}")
    print(f"  R²: {result.r_squared:.6f}")

    # Rough surface should have D > 2.0
    assert result.dimension > 2.0, \
        f"Rough surface dimension {result.dimension} should be > 2.0"
    print("  PASSED!")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("3D Box Counting Tests")
    print("=" * 60)

    test_triangle_cube_intersection()
    test_plane_dimension()
    test_sphere_dimension()
    test_rough_surface_dimension()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
