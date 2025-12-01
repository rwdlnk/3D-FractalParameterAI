"""
Mesh I/O and data structures for 3D surface analysis.

Supports reading VTK (.vtk, .vtp) and STL (.stl) mesh files.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
import os


@dataclass
class BoundingBox3D:
    """3D axis-aligned bounding box."""
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    min_z: float
    max_z: float

    @property
    def width(self) -> float:
        """Size in X direction."""
        return self.max_x - self.min_x

    @property
    def height(self) -> float:
        """Size in Y direction."""
        return self.max_y - self.min_y

    @property
    def depth(self) -> float:
        """Size in Z direction."""
        return self.max_z - self.min_z

    @property
    def dimensions(self) -> Tuple[float, float, float]:
        """Return (width, height, depth)."""
        return (self.width, self.height, self.depth)

    @property
    def max_dimension(self) -> float:
        """Largest dimension."""
        return max(self.width, self.height, self.depth)

    @property
    def min_dimension(self) -> float:
        """Smallest dimension."""
        return min(self.width, self.height, self.depth)

    @property
    def characteristic_length(self) -> float:
        """Characteristic length scale for surface analysis.

        For surfaces (which may be flat in one dimension), uses the
        maximum of the geometric mean and max dimension / sqrt(3).
        """
        dims = sorted([self.width, self.height, self.depth], reverse=True)
        # For truly 3D surfaces, use geometric mean
        if dims[2] > 1e-10:
            return (dims[0] * dims[1] * dims[2]) ** (1/3)
        # For 2D surfaces (flat in one dimension), use sqrt of area
        elif dims[1] > 1e-10:
            return (dims[0] * dims[1]) ** 0.5
        # For 1D (line), use length
        else:
            return dims[0]

    @property
    def volume(self) -> float:
        """Bounding box volume."""
        return self.width * self.height * self.depth

    @property
    def center(self) -> Tuple[float, float, float]:
        """Center point of bounding box."""
        return (
            (self.min_x + self.max_x) / 2,
            (self.min_y + self.max_y) / 2,
            (self.min_z + self.max_z) / 2,
        )

    def contains_point(self, x: float, y: float, z: float) -> bool:
        """Check if point is inside bounding box."""
        return (self.min_x <= x <= self.max_x and
                self.min_y <= y <= self.max_y and
                self.min_z <= z <= self.max_z)


@dataclass
class TriangleMesh:
    """
    Triangulated surface mesh.

    Attributes:
        vertices: Nx3 array of vertex coordinates
        triangles: Mx3 array of vertex indices (0-based)
        bbox: Bounding box of the mesh
    """
    vertices: np.ndarray  # Shape: (N, 3)
    triangles: np.ndarray  # Shape: (M, 3), dtype: int
    bbox: BoundingBox3D

    @property
    def n_vertices(self) -> int:
        """Number of vertices."""
        return len(self.vertices)

    @property
    def n_triangles(self) -> int:
        """Number of triangles."""
        return len(self.triangles)

    @property
    def surface_area(self) -> float:
        """Total surface area of mesh."""
        areas = self.triangle_areas()
        return np.sum(areas)

    def triangle_areas(self) -> np.ndarray:
        """Compute area of each triangle."""
        v0 = self.vertices[self.triangles[:, 0]]
        v1 = self.vertices[self.triangles[:, 1]]
        v2 = self.vertices[self.triangles[:, 2]]

        # Cross product gives area * 2
        cross = np.cross(v1 - v0, v2 - v0)
        areas = 0.5 * np.linalg.norm(cross, axis=1)
        return areas

    def get_triangle_vertices(self, triangle_idx: int) -> np.ndarray:
        """Get 3x3 array of vertex coordinates for a triangle."""
        idx = self.triangles[triangle_idx]
        return self.vertices[idx]

    def get_all_triangle_vertices(self) -> np.ndarray:
        """
        Get all triangle vertices as Mx3x3 array.

        Returns:
            Array of shape (M, 3, 3) where result[i] gives the 3 vertices
            of triangle i, each vertex being (x, y, z).
        """
        return self.vertices[self.triangles]

    @classmethod
    def from_vertices_and_faces(cls, vertices: np.ndarray,
                                 faces: np.ndarray) -> 'TriangleMesh':
        """Create mesh from vertex and face arrays."""
        vertices = np.asarray(vertices, dtype=np.float64)
        faces = np.asarray(faces, dtype=np.int64)

        # Compute bounding box
        bbox = BoundingBox3D(
            min_x=vertices[:, 0].min(),
            max_x=vertices[:, 0].max(),
            min_y=vertices[:, 1].min(),
            max_y=vertices[:, 1].max(),
            min_z=vertices[:, 2].min(),
            max_z=vertices[:, 2].max(),
        )

        return cls(vertices=vertices, triangles=faces, bbox=bbox)


def load_vtk(filename: str) -> TriangleMesh:
    """
    Load mesh from VTK file (.vtk or .vtp).

    Requires the vtk package.
    """
    try:
        import vtk
        from vtk.util.numpy_support import vtk_to_numpy
    except ImportError:
        raise ImportError("VTK package required. Install with: pip install vtk")

    # Determine reader based on extension
    ext = os.path.splitext(filename)[1].lower()

    if ext == '.vtp':
        reader = vtk.vtkXMLPolyDataReader()
    elif ext == '.vtk':
        reader = vtk.vtkPolyDataReader()
    else:
        raise ValueError(f"Unsupported VTK extension: {ext}")

    reader.SetFileName(filename)
    reader.Update()

    polydata = reader.GetOutput()

    # Extract vertices
    points = polydata.GetPoints()
    vertices = vtk_to_numpy(points.GetData())

    # Extract triangles
    polys = polydata.GetPolys()
    if polys.GetNumberOfCells() == 0:
        raise ValueError("No polygons found in VTK file")

    # Convert to numpy array of triangle indices
    cells = vtk_to_numpy(polys.GetConnectivityArray())
    offsets = vtk_to_numpy(polys.GetOffsetsArray())

    # Check that all polygons are triangles
    n_cells = len(offsets) - 1
    triangles = []

    for i in range(n_cells):
        start = offsets[i]
        end = offsets[i + 1]
        n_verts = end - start

        if n_verts == 3:
            triangles.append(cells[start:end])
        elif n_verts > 3:
            # Triangulate polygon using fan triangulation
            for j in range(1, n_verts - 1):
                triangles.append([cells[start], cells[start + j], cells[start + j + 1]])
        # Skip degenerate polygons with < 3 vertices

    triangles = np.array(triangles, dtype=np.int64)

    return TriangleMesh.from_vertices_and_faces(vertices, triangles)


def load_stl(filename: str) -> TriangleMesh:
    """
    Load mesh from STL file (ASCII or binary).
    """
    # Try to detect if ASCII or binary
    with open(filename, 'rb') as f:
        header = f.read(80)

    # ASCII STL starts with "solid"
    is_ascii = header.startswith(b'solid') and b'\x00' not in header

    if is_ascii:
        return _load_stl_ascii(filename)
    else:
        return _load_stl_binary(filename)


def _load_stl_ascii(filename: str) -> TriangleMesh:
    """Load ASCII STL file."""
    vertices = []
    triangles = []

    with open(filename, 'r') as f:
        current_triangle = []

        for line in f:
            line = line.strip().lower()

            if line.startswith('vertex'):
                parts = line.split()
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                vertices.append([x, y, z])
                current_triangle.append(len(vertices) - 1)

                if len(current_triangle) == 3:
                    triangles.append(current_triangle)
                    current_triangle = []

    vertices = np.array(vertices, dtype=np.float64)
    triangles = np.array(triangles, dtype=np.int64)

    return TriangleMesh.from_vertices_and_faces(vertices, triangles)


def _load_stl_binary(filename: str) -> TriangleMesh:
    """Load binary STL file."""
    with open(filename, 'rb') as f:
        # Skip 80-byte header
        f.read(80)

        # Read number of triangles
        n_triangles = np.frombuffer(f.read(4), dtype=np.uint32)[0]

        vertices = []
        triangles = []

        for i in range(n_triangles):
            # Read normal (3 floats) - we ignore this
            f.read(12)

            # Read 3 vertices (9 floats)
            v1 = np.frombuffer(f.read(12), dtype=np.float32)
            v2 = np.frombuffer(f.read(12), dtype=np.float32)
            v3 = np.frombuffer(f.read(12), dtype=np.float32)

            base_idx = len(vertices)
            vertices.extend([v1, v2, v3])
            triangles.append([base_idx, base_idx + 1, base_idx + 2])

            # Skip attribute byte count
            f.read(2)

    vertices = np.array(vertices, dtype=np.float64)
    triangles = np.array(triangles, dtype=np.int64)

    return TriangleMesh.from_vertices_and_faces(vertices, triangles)


def load_mesh(filename: str) -> TriangleMesh:
    """
    Load mesh from file (auto-detect format).

    Supported formats:
        - .vtk (VTK legacy PolyData)
        - .vtp (VTK XML PolyData)
        - .stl (STL ASCII or binary)

    Args:
        filename: Path to mesh file

    Returns:
        TriangleMesh object
    """
    ext = os.path.splitext(filename)[1].lower()

    if ext in ('.vtk', '.vtp'):
        return load_vtk(filename)
    elif ext == '.stl':
        return load_stl(filename)
    else:
        raise ValueError(f"Unsupported file format: {ext}")
