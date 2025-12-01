"""
Spatial Hash Grid for accelerating triangle-cube intersection queries.

Provides O(1) average lookup for finding triangles that may intersect
a given region of space.
"""

import numpy as np
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
from .mesh_io import TriangleMesh, BoundingBox3D


class SpatialHashGrid:
    """
    Spatial hash grid for fast triangle lookup.

    Triangles are binned into cells based on their bounding boxes.
    Queries for a region return only triangles in overlapping cells.
    """

    def __init__(self, mesh: TriangleMesh, cell_size: Optional[float] = None):
        """
        Build spatial hash from mesh.

        Args:
            mesh: TriangleMesh to index
            cell_size: Size of hash cells (default: characteristic_length / 10)
        """
        self.mesh = mesh

        # Default cell size based on mesh
        if cell_size is None:
            cell_size = mesh.bbox.characteristic_length / 10

        self.cell_size = cell_size
        self.inv_cell_size = 1.0 / cell_size

        # Origin for cell indexing
        self.origin = np.array([
            mesh.bbox.min_x,
            mesh.bbox.min_y,
            mesh.bbox.min_z
        ])

        # Precompute triangle data
        self._triangle_vertices = mesh.get_all_triangle_vertices()
        self._tri_min = self._triangle_vertices.min(axis=1)
        self._tri_max = self._triangle_vertices.max(axis=1)

        # Build the hash grid
        self._grid: Dict[Tuple[int, int, int], List[int]] = defaultdict(list)
        self._build_grid()

        # Statistics
        self.n_cells = len(self._grid)
        self.n_triangles = mesh.n_triangles

    def _build_grid(self) -> None:
        """Build the spatial hash grid by binning triangles."""
        for tri_idx in range(self.mesh.n_triangles):
            # Get cell range for this triangle's bounding box
            min_cell = self._point_to_cell(self._tri_min[tri_idx])
            max_cell = self._point_to_cell(self._tri_max[tri_idx])

            # Add triangle to all cells it overlaps
            for i in range(min_cell[0], max_cell[0] + 1):
                for j in range(min_cell[1], max_cell[1] + 1):
                    for k in range(min_cell[2], max_cell[2] + 1):
                        self._grid[(i, j, k)].append(tri_idx)

    def _point_to_cell(self, point: np.ndarray) -> Tuple[int, int, int]:
        """Convert a point to cell indices."""
        cell = ((point - self.origin) * self.inv_cell_size).astype(int)
        return (int(cell[0]), int(cell[1]), int(cell[2]))

    def query_box(self, box_min: np.ndarray, box_max: np.ndarray) -> Set[int]:
        """
        Find all triangles that might intersect a box.

        Args:
            box_min: Minimum corner of query box
            box_max: Maximum corner of query box

        Returns:
            Set of triangle indices that may intersect the box
        """
        min_cell = self._point_to_cell(box_min)
        max_cell = self._point_to_cell(box_max)

        candidates = set()

        for i in range(min_cell[0], max_cell[0] + 1):
            for j in range(min_cell[1], max_cell[1] + 1):
                for k in range(min_cell[2], max_cell[2] + 1):
                    cell_key = (i, j, k)
                    if cell_key in self._grid:
                        candidates.update(self._grid[cell_key])

        return candidates

    def query_point(self, point: np.ndarray) -> List[int]:
        """Find all triangles in the cell containing a point."""
        cell = self._point_to_cell(point)
        return self._grid.get(cell, [])

    def get_triangle_vertices(self, tri_idx: int) -> np.ndarray:
        """Get the 3 vertices of a triangle."""
        return self._triangle_vertices[tri_idx]

    def get_triangle_bbox(self, tri_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get the bounding box of a triangle."""
        return self._tri_min[tri_idx], self._tri_max[tri_idx]

    def stats(self) -> Dict:
        """Get statistics about the spatial hash."""
        if self.n_cells == 0:
            return {
                "n_cells": 0,
                "n_triangles": self.n_triangles,
                "cell_size": self.cell_size,
                "avg_triangles_per_cell": 0,
                "max_triangles_per_cell": 0,
            }

        cell_sizes = [len(tris) for tris in self._grid.values()]

        return {
            "n_cells": self.n_cells,
            "n_triangles": self.n_triangles,
            "cell_size": self.cell_size,
            "avg_triangles_per_cell": np.mean(cell_sizes),
            "max_triangles_per_cell": max(cell_sizes),
            "min_triangles_per_cell": min(cell_sizes),
        }


class AcceleratedCubeCounter:
    """
    Cube counter with spatial hash acceleration.

    Significantly faster than brute-force for large meshes.
    Build cost is O(n_triangles), query cost is O(n_cubes × avg_triangles_per_cell).
    """

    def __init__(self, mesh: TriangleMesh, cell_size: Optional[float] = None):
        """
        Initialize accelerated counter.

        Args:
            mesh: TriangleMesh to analyze
            cell_size: Spatial hash cell size (default: auto-computed)
        """
        self.mesh = mesh
        self._triangle_vertices = mesh.get_all_triangle_vertices()

        # Build spatial hash
        if cell_size is None:
            # Use characteristic length / 5 for good balance
            cell_size = mesh.bbox.characteristic_length / 5

        self.spatial_hash = SpatialHashGrid(mesh, cell_size)

    def count_cubes(self, delta: float,
                    domain: Optional[BoundingBox3D] = None) -> 'BoxCountResult':
        """
        Count cubes of size delta that intersect the mesh surface.

        Uses spatial hash for acceleration.

        Args:
            delta: Cube side length
            domain: Optional custom domain (defaults to mesh bounding box)

        Returns:
            BoxCountResult with count and grid information
        """
        from .box_counting_3d import BoxCountResult, triangle_cube_intersection

        if domain is None:
            domain = self.mesh.bbox

        # Compute grid dimensions
        nx = max(1, int(np.ceil(domain.width / delta)))
        ny = max(1, int(np.ceil(domain.height / delta)))
        nz = max(1, int(np.ceil(domain.depth / delta)))

        # Set of occupied cube indices
        occupied = set()

        # For each cube, query spatial hash and test candidates
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    # Cube bounds
                    cube_min = np.array([
                        domain.min_x + i * delta,
                        domain.min_y + j * delta,
                        domain.min_z + k * delta
                    ])
                    cube_max = cube_min + delta

                    # Query spatial hash for candidate triangles
                    candidates = self.spatial_hash.query_box(cube_min, cube_max)

                    # Test each candidate
                    for tri_idx in candidates:
                        v0, v1, v2 = self._triangle_vertices[tri_idx]

                        if triangle_cube_intersection(v0, v1, v2, cube_min, cube_max):
                            occupied.add((i, j, k))
                            break  # Found intersection, no need to test more

        return BoxCountResult(
            delta=delta,
            n_boxes=len(occupied),
            grid_dims=(nx, ny, nz)
        )

    def count_cubes_fast(self, delta: float,
                         domain: Optional[BoundingBox3D] = None) -> 'BoxCountResult':
        """
        Faster cube counting using triangle-centric approach.

        Instead of iterating over cubes, iterates over triangles and marks
        the cubes they intersect. Better for sparse surfaces.

        Args:
            delta: Cube side length
            domain: Optional custom domain (defaults to mesh bounding box)

        Returns:
            BoxCountResult with count and grid information
        """
        from .box_counting_3d import BoxCountResult, triangle_cube_intersection

        if domain is None:
            domain = self.mesh.bbox

        # Compute grid dimensions
        nx = max(1, int(np.ceil(domain.width / delta)))
        ny = max(1, int(np.ceil(domain.height / delta)))
        nz = max(1, int(np.ceil(domain.depth / delta)))

        inv_delta = 1.0 / delta

        # Set of occupied cube indices
        occupied = set()

        # Triangle bounding boxes
        tri_min = self._triangle_vertices.min(axis=1)
        tri_max = self._triangle_vertices.max(axis=1)

        # For each triangle, find candidate cubes and test intersection
        for t_idx in range(self.mesh.n_triangles):
            t_min = tri_min[t_idx]
            t_max = tri_max[t_idx]

            # Cube index range for this triangle
            i_min = max(0, int((t_min[0] - domain.min_x) * inv_delta))
            i_max = min(nx - 1, int((t_max[0] - domain.min_x) * inv_delta))
            j_min = max(0, int((t_min[1] - domain.min_y) * inv_delta))
            j_max = min(ny - 1, int((t_max[1] - domain.min_y) * inv_delta))
            k_min = max(0, int((t_min[2] - domain.min_z) * inv_delta))
            k_max = min(nz - 1, int((t_max[2] - domain.min_z) * inv_delta))

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

                        # Cube bounds
                        cube_min = np.array([
                            domain.min_x + i * delta,
                            domain.min_y + j * delta,
                            domain.min_z + k * delta
                        ])
                        cube_max = cube_min + delta

                        if triangle_cube_intersection(v0, v1, v2, cube_min, cube_max):
                            occupied.add((i, j, k))

        return BoxCountResult(
            delta=delta,
            n_boxes=len(occupied),
            grid_dims=(nx, ny, nz)
        )


def create_counter(mesh: TriangleMesh,
                   accelerated: bool = True,
                   cell_size: Optional[float] = None) -> 'CubeCounter':
    """
    Factory function to create appropriate cube counter.

    Args:
        mesh: TriangleMesh to analyze
        accelerated: Use spatial hash acceleration (default: True)
        cell_size: Spatial hash cell size for accelerated counter

    Returns:
        CubeCounter or AcceleratedCubeCounter
    """
    if accelerated:
        return AcceleratedCubeCounter(mesh, cell_size)
    else:
        from .box_counting_3d import CubeCounter
        return CubeCounter(mesh)
