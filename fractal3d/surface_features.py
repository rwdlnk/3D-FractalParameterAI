"""
Surface Feature Extraction for 3D Fractal Analysis.

Computes geometric and topological features of triangulated surfaces
to guide AI-based parameter optimization for fractal dimension calculation.

Analogous to interface_features.py in the 2D FractalParameterAI project.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict, deque
from dataclasses import dataclass
from .mesh_io import TriangleMesh, BoundingBox3D


@dataclass
class ConnectedComponent:
    """A connected component of a mesh."""
    triangle_indices: np.ndarray  # Indices of triangles in this component
    n_triangles: int
    surface_area: float
    bbox: BoundingBox3D
    centroid: np.ndarray  # Center of mass


@dataclass
class SurfaceFeatures:
    """Comprehensive surface characterization features."""
    # Basic geometry
    n_vertices: int
    n_triangles: int
    surface_area: float
    bbox_volume: float
    characteristic_length: float

    # Connectivity
    n_components: int
    largest_component_fraction: float  # Fraction of area in largest component
    fragmentation_index: float  # 1 - largest_component_fraction
    component_size_std: float  # Std dev of component sizes (normalized)

    # Shape metrics
    compactness: float  # Surface area / convex hull area (≥1, higher = more complex)
    sphericity: float  # How sphere-like (1 = perfect sphere)
    aspect_ratio: float  # max_dim / min_dim of bounding box
    flatness: float  # min_dim / mid_dim (0-1, lower = flatter)
    elongation: float  # mid_dim / max_dim (0-1, lower = more elongated)

    # Surface complexity (3D analogs of 2D tortuosity)
    roughness: float  # Surface area / projected area estimate
    area_ratio: float  # Surface area / bbox surface area
    convexity: float  # Convex hull volume / bbox volume

    # Curvature statistics (if computed)
    mean_curvature_mean: Optional[float] = None
    mean_curvature_std: Optional[float] = None
    gaussian_curvature_mean: Optional[float] = None
    gaussian_curvature_std: Optional[float] = None

    # Triangle quality metrics
    min_triangle_area: float = 0.0
    max_triangle_area: float = 0.0
    mean_triangle_area: float = 0.0
    triangle_area_cv: float = 0.0  # Coefficient of variation

    # Derived complexity scores
    complexity_score: float = 0.0  # Combined complexity metric
    surface_type: str = "unknown"  # Classification for parameter selection

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {k: v for k, v in self.__dict__.items() if v is not None}


def build_adjacency_graph(mesh: TriangleMesh) -> Dict[int, Set[int]]:
    """
    Build triangle adjacency graph based on shared edges.

    Two triangles are adjacent if they share an edge (2 vertices).

    Returns:
        Dictionary mapping triangle index to set of adjacent triangle indices
    """
    # Build edge -> triangles mapping
    edge_to_triangles: Dict[Tuple[int, int], List[int]] = defaultdict(list)

    for tri_idx, triangle in enumerate(mesh.triangles):
        # Get the three edges (sorted vertex pairs)
        v0, v1, v2 = triangle
        edges = [
            tuple(sorted([v0, v1])),
            tuple(sorted([v1, v2])),
            tuple(sorted([v2, v0])),
        ]
        for edge in edges:
            edge_to_triangles[edge].append(tri_idx)

    # Build adjacency graph
    adjacency: Dict[int, Set[int]] = defaultdict(set)

    for triangles in edge_to_triangles.values():
        # All triangles sharing this edge are adjacent
        for i, tri1 in enumerate(triangles):
            for tri2 in triangles[i+1:]:
                adjacency[tri1].add(tri2)
                adjacency[tri2].add(tri1)

    return adjacency


def find_connected_components(mesh: TriangleMesh) -> List[ConnectedComponent]:
    """
    Find all connected components of a mesh using BFS.

    Returns:
        List of ConnectedComponent objects, sorted by surface area (largest first)
    """
    adjacency = build_adjacency_graph(mesh)
    visited = set()
    components = []

    triangle_areas = mesh.triangle_areas()
    all_vertices = mesh.get_all_triangle_vertices()  # (N, 3, 3)

    for start_tri in range(mesh.n_triangles):
        if start_tri in visited:
            continue

        # BFS to find all triangles in this component
        component_tris = []
        queue = deque([start_tri])
        visited.add(start_tri)

        while queue:
            tri_idx = queue.popleft()
            component_tris.append(tri_idx)

            for neighbor in adjacency.get(tri_idx, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        # Compute component properties
        tri_indices = np.array(component_tris, dtype=np.int64)
        area = triangle_areas[tri_indices].sum()

        # Get vertices of this component
        component_vertices = all_vertices[tri_indices].reshape(-1, 3)

        # Bounding box
        bbox = BoundingBox3D(
            min_x=component_vertices[:, 0].min(),
            max_x=component_vertices[:, 0].max(),
            min_y=component_vertices[:, 1].min(),
            max_y=component_vertices[:, 1].max(),
            min_z=component_vertices[:, 2].min(),
            max_z=component_vertices[:, 2].max(),
        )

        # Centroid (area-weighted)
        tri_centers = all_vertices[tri_indices].mean(axis=1)  # (n, 3)
        tri_areas = triangle_areas[tri_indices]
        if area > 0:
            centroid = np.average(tri_centers, weights=tri_areas, axis=0)
        else:
            centroid = tri_centers.mean(axis=0)

        components.append(ConnectedComponent(
            triangle_indices=tri_indices,
            n_triangles=len(tri_indices),
            surface_area=area,
            bbox=bbox,
            centroid=centroid,
        ))

    # Sort by surface area (largest first)
    components.sort(key=lambda c: c.surface_area, reverse=True)

    return components


def compute_vertex_normals(mesh: TriangleMesh) -> np.ndarray:
    """
    Compute vertex normals by averaging adjacent face normals.

    Returns:
        Array of shape (n_vertices, 3) with unit normal vectors
    """
    # Compute face normals
    v0 = mesh.vertices[mesh.triangles[:, 0]]
    v1 = mesh.vertices[mesh.triangles[:, 1]]
    v2 = mesh.vertices[mesh.triangles[:, 2]]

    face_normals = np.cross(v1 - v0, v2 - v0)
    face_areas = np.linalg.norm(face_normals, axis=1, keepdims=True)
    face_normals = np.divide(face_normals, face_areas,
                             where=face_areas > 1e-10,
                             out=np.zeros_like(face_normals))

    # Accumulate normals at vertices (area-weighted)
    vertex_normals = np.zeros((mesh.n_vertices, 3))

    for i, triangle in enumerate(mesh.triangles):
        for v_idx in triangle:
            vertex_normals[v_idx] += face_normals[i] * face_areas[i, 0]

    # Normalize
    norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    vertex_normals = np.divide(vertex_normals, norms,
                               where=norms > 1e-10,
                               out=np.zeros_like(vertex_normals))

    return vertex_normals


def estimate_mean_curvature(mesh: TriangleMesh) -> np.ndarray:
    """
    Estimate mean curvature at each vertex using the discrete Laplacian.

    Uses the cotangent Laplacian formula for approximation.

    Returns:
        Array of shape (n_vertices,) with mean curvature values
    """
    n_verts = mesh.n_vertices

    # Build cotan weights and compute Laplacian
    laplacian = np.zeros((n_verts, 3))
    areas = np.zeros(n_verts)

    for triangle in mesh.triangles:
        i, j, k = triangle
        vi, vj, vk = mesh.vertices[i], mesh.vertices[j], mesh.vertices[k]

        # Edge vectors
        eij = vj - vi
        ejk = vk - vj
        eki = vi - vk

        # Cotangent weights
        def cotan(v1, v2):
            cos_angle = np.dot(v1, v2)
            sin_angle = np.linalg.norm(np.cross(v1, v2))
            if sin_angle < 1e-10:
                return 0.0
            return cos_angle / sin_angle

        cot_i = cotan(-eki, eij)
        cot_j = cotan(-eij, ejk)
        cot_k = cotan(-ejk, eki)

        # Accumulate Laplacian contributions
        laplacian[i] += cot_j * (vk - vi) + cot_k * (vj - vi)
        laplacian[j] += cot_k * (vi - vj) + cot_i * (vk - vj)
        laplacian[k] += cot_i * (vj - vk) + cot_j * (vi - vk)

        # Accumulate areas (Voronoi regions approximation)
        tri_area = 0.5 * np.linalg.norm(np.cross(eij, -eki))
        areas[i] += tri_area / 3
        areas[j] += tri_area / 3
        areas[k] += tri_area / 3

    # Mean curvature = |Laplacian| / (4 * area)
    areas = np.maximum(areas, 1e-10)  # Avoid division by zero
    mean_curvature = np.linalg.norm(laplacian, axis=1) / (4 * areas)

    return mean_curvature


def estimate_gaussian_curvature(mesh: TriangleMesh) -> np.ndarray:
    """
    Estimate Gaussian curvature at each vertex using angle deficit.

    K = (2π - sum of angles) / area

    Returns:
        Array of shape (n_vertices,) with Gaussian curvature values
    """
    n_verts = mesh.n_vertices
    angle_sum = np.zeros(n_verts)
    areas = np.zeros(n_verts)

    for triangle in mesh.triangles:
        i, j, k = triangle
        vi, vj, vk = mesh.vertices[i], mesh.vertices[j], mesh.vertices[k]

        # Edge vectors from each vertex
        eij = vj - vi
        eik = vk - vi
        eji = vi - vj
        ejk = vk - vj
        eki = vi - vk
        ekj = vj - vk

        # Angles at each vertex
        def angle(v1, v2):
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
            cos_angle = np.clip(cos_angle, -1, 1)
            return np.arccos(cos_angle)

        angle_sum[i] += angle(eij, eik)
        angle_sum[j] += angle(eji, ejk)
        angle_sum[k] += angle(eki, ekj)

        # Area contribution
        tri_area = 0.5 * np.linalg.norm(np.cross(eij, eik))
        areas[i] += tri_area / 3
        areas[j] += tri_area / 3
        areas[k] += tri_area / 3

    # Gaussian curvature from angle deficit
    areas = np.maximum(areas, 1e-10)
    gaussian_curvature = (2 * np.pi - angle_sum) / areas

    return gaussian_curvature


def compute_convex_hull_properties(mesh: TriangleMesh) -> Tuple[float, float]:
    """
    Compute convex hull surface area and volume.

    Returns:
        (convex_hull_area, convex_hull_volume)
    """
    try:
        from scipy.spatial import ConvexHull

        # Need at least 4 non-coplanar points
        if mesh.n_vertices < 4:
            return mesh.surface_area, mesh.bbox.volume

        try:
            hull = ConvexHull(mesh.vertices)
            return hull.area, hull.volume
        except Exception:
            # Degenerate case (coplanar points, etc.)
            return mesh.surface_area, mesh.bbox.volume

    except ImportError:
        # scipy not available
        return mesh.surface_area, mesh.bbox.volume


def extract_surface_features(mesh: TriangleMesh,
                              compute_curvature: bool = True) -> SurfaceFeatures:
    """
    Extract comprehensive surface features for AI parameter optimization.

    Args:
        mesh: TriangleMesh to analyze
        compute_curvature: Whether to compute curvature statistics (slower)

    Returns:
        SurfaceFeatures object with all computed metrics
    """
    # Basic geometry
    surface_area = mesh.surface_area
    bbox = mesh.bbox
    bbox_volume = bbox.volume if bbox.volume > 0 else 1e-10
    char_length = bbox.characteristic_length

    # Connected components
    components = find_connected_components(mesh)
    n_components = len(components)

    if n_components > 0 and surface_area > 0:
        largest_fraction = components[0].surface_area / surface_area
        component_areas = np.array([c.surface_area for c in components])
        component_areas_norm = component_areas / surface_area
        size_std = np.std(component_areas_norm) if n_components > 1 else 0.0
    else:
        largest_fraction = 1.0
        size_std = 0.0

    fragmentation = 1.0 - largest_fraction

    # Shape metrics from bounding box
    dims = sorted([bbox.width, bbox.height, bbox.depth], reverse=True)
    max_dim, mid_dim, min_dim = dims[0], dims[1], dims[2]

    # Avoid division by zero
    max_dim = max(max_dim, 1e-10)
    mid_dim = max(mid_dim, 1e-10)
    min_dim = max(min_dim, 1e-10)

    aspect_ratio = max_dim / min_dim
    flatness = min_dim / mid_dim
    elongation = mid_dim / max_dim

    # Sphericity: ratio of surface area of equivalent sphere to actual surface area
    # For a sphere with same volume: A_sphere = (36π V²)^(1/3)
    if bbox_volume > 0:
        equivalent_sphere_area = (36 * np.pi * bbox_volume**2) ** (1/3)
        sphericity = equivalent_sphere_area / max(surface_area, 1e-10)
        sphericity = min(sphericity, 1.0)  # Cap at 1
    else:
        sphericity = 0.0

    # Convex hull properties
    hull_area, hull_volume = compute_convex_hull_properties(mesh)
    compactness = surface_area / max(hull_area, 1e-10)
    convexity = hull_volume / max(bbox_volume, 1e-10)

    # Surface complexity metrics
    bbox_surface_area = 2 * (bbox.width * bbox.height +
                             bbox.height * bbox.depth +
                             bbox.depth * bbox.width)
    bbox_surface_area = max(bbox_surface_area, 1e-10)
    area_ratio = surface_area / bbox_surface_area

    # Roughness: estimate as surface_area / projected_area
    # Use max cross-sectional area as projected area estimate
    projected_areas = [
        bbox.width * bbox.height,
        bbox.height * bbox.depth,
        bbox.depth * bbox.width,
    ]
    max_projected = max(projected_areas)
    roughness = surface_area / max(max_projected, 1e-10)

    # Triangle quality metrics
    tri_areas = mesh.triangle_areas()
    if len(tri_areas) > 0:
        min_tri_area = tri_areas.min()
        max_tri_area = tri_areas.max()
        mean_tri_area = tri_areas.mean()
        tri_area_std = tri_areas.std()
        tri_area_cv = tri_area_std / mean_tri_area if mean_tri_area > 0 else 0.0
    else:
        min_tri_area = max_tri_area = mean_tri_area = 0.0
        tri_area_cv = 0.0

    # Curvature statistics
    mean_curv_mean = mean_curv_std = None
    gauss_curv_mean = gauss_curv_std = None

    if compute_curvature and mesh.n_vertices > 3:
        try:
            mean_curvature = estimate_mean_curvature(mesh)
            # Filter outliers (curvature at boundary vertices can be extreme)
            valid_mask = np.isfinite(mean_curvature) & (np.abs(mean_curvature) < 1e6)
            if valid_mask.sum() > 0:
                mean_curv = mean_curvature[valid_mask]
                mean_curv_mean = float(np.mean(mean_curv))
                mean_curv_std = float(np.std(mean_curv))

            gaussian_curvature = estimate_gaussian_curvature(mesh)
            valid_mask = np.isfinite(gaussian_curvature) & (np.abs(gaussian_curvature) < 1e6)
            if valid_mask.sum() > 0:
                gauss_curv = gaussian_curvature[valid_mask]
                gauss_curv_mean = float(np.mean(gauss_curv))
                gauss_curv_std = float(np.std(gauss_curv))
        except Exception:
            pass  # Curvature computation can fail for degenerate meshes

    # Complexity score (combined metric)
    # Higher values indicate more complex surfaces needing finer analysis
    complexity_score = (
        0.3 * (compactness - 1.0) +  # Deviation from convex
        0.2 * fragmentation * 10 +    # Fragmentation penalty
        0.2 * (roughness - 2.0) / 10 +  # Excess roughness
        0.15 * np.log1p(n_components) +  # Component count
        0.15 * (1 - sphericity)  # Non-sphericity
    )
    complexity_score = max(0.0, complexity_score)

    # Surface classification for parameter selection
    surface_type = classify_surface_type(
        n_components=n_components,
        fragmentation=fragmentation,
        compactness=compactness,
        roughness=roughness,
        sphericity=sphericity,
        aspect_ratio=aspect_ratio,
        complexity_score=complexity_score,
    )

    return SurfaceFeatures(
        n_vertices=mesh.n_vertices,
        n_triangles=mesh.n_triangles,
        surface_area=surface_area,
        bbox_volume=bbox_volume,
        characteristic_length=char_length,
        n_components=n_components,
        largest_component_fraction=largest_fraction,
        fragmentation_index=fragmentation,
        component_size_std=size_std,
        compactness=compactness,
        sphericity=sphericity,
        aspect_ratio=aspect_ratio,
        flatness=flatness,
        elongation=elongation,
        roughness=roughness,
        area_ratio=area_ratio,
        convexity=convexity,
        mean_curvature_mean=mean_curv_mean,
        mean_curvature_std=mean_curv_std,
        gaussian_curvature_mean=gauss_curv_mean,
        gaussian_curvature_std=gauss_curv_std,
        min_triangle_area=min_tri_area,
        max_triangle_area=max_tri_area,
        mean_triangle_area=mean_tri_area,
        triangle_area_cv=tri_area_cv,
        complexity_score=complexity_score,
        surface_type=surface_type,
    )


def classify_surface_type(n_components: int,
                          fragmentation: float,
                          compactness: float,
                          roughness: float,
                          sphericity: float,
                          aspect_ratio: float,
                          complexity_score: float) -> str:
    """
    Classify surface type for parameter selection.

    Categories:
        - "smooth": Low complexity, nearly convex surfaces
        - "moderate": Moderate roughness/complexity
        - "complex": High roughness or fragmentation
        - "highly_fragmented": Many disconnected components
        - "turbulent_mixing": High fragmentation + roughness (RT late stage)

    Returns:
        Surface type string
    """
    # Highly fragmented (many components)
    if n_components > 10 or fragmentation > 0.5:
        if roughness > 5.0 or compactness > 2.0:
            return "turbulent_mixing"
        return "highly_fragmented"

    # Complex surfaces
    if complexity_score > 1.5 or compactness > 3.0 or roughness > 10.0:
        return "complex"

    # Moderate complexity
    if complexity_score > 0.5 or compactness > 1.5 or roughness > 4.0:
        return "moderate"

    # Smooth surfaces
    return "smooth"


def suggest_parameters(features: SurfaceFeatures) -> Dict[str, float]:
    """
    Suggest box-counting parameters based on surface features.

    This is the 3D analog of the 2D parameter suggestion system.

    Returns:
        Dictionary with suggested parameters:
        - initial_delta: Starting cube size
        - delta_factor: Scale reduction factor
        - num_steps: Number of scales to test
    """
    char_len = features.characteristic_length
    surface_type = features.surface_type

    # Base parameters
    params = {
        "initial_delta": char_len / 10,
        "delta_factor": 1.5,
        "num_steps": 15,
    }

    if surface_type == "smooth":
        # Smooth surfaces: coarser analysis is sufficient
        params["initial_delta"] = char_len / 8
        params["delta_factor"] = 1.6
        params["num_steps"] = 12

    elif surface_type == "moderate":
        # Moderate: standard parameters
        params["initial_delta"] = char_len / 10
        params["delta_factor"] = 1.5
        params["num_steps"] = 15

    elif surface_type == "complex":
        # Complex: finer resolution needed
        params["initial_delta"] = char_len / 15
        params["delta_factor"] = 1.4
        params["num_steps"] = 18

    elif surface_type == "highly_fragmented":
        # Fragmented: need to capture small fragments
        params["initial_delta"] = char_len / 15
        params["delta_factor"] = 1.4
        params["num_steps"] = 18

    elif surface_type == "turbulent_mixing":
        # Late-stage RT: finest resolution, many scales
        params["initial_delta"] = char_len / 20
        params["delta_factor"] = 1.3
        params["num_steps"] = 22

    # Adjust for high fragmentation
    if features.n_components > 5:
        # Use smaller cubes to capture small fragments
        params["initial_delta"] *= 0.8
        params["num_steps"] += 2

    # Adjust for high aspect ratio
    if features.aspect_ratio > 5:
        # Elongated surfaces may need more scales
        params["num_steps"] += 2

    return params


def print_feature_summary(features: SurfaceFeatures) -> None:
    """Print a formatted summary of surface features."""
    print("=" * 60)
    print("SURFACE FEATURE SUMMARY")
    print("=" * 60)

    print(f"\nGeometry:")
    print(f"  Vertices: {features.n_vertices}")
    print(f"  Triangles: {features.n_triangles}")
    print(f"  Surface area: {features.surface_area:.4f}")
    print(f"  Characteristic length: {features.characteristic_length:.4f}")

    print(f"\nConnectivity:")
    print(f"  Components: {features.n_components}")
    print(f"  Largest component: {features.largest_component_fraction*100:.1f}%")
    print(f"  Fragmentation index: {features.fragmentation_index:.3f}")

    print(f"\nShape:")
    print(f"  Compactness: {features.compactness:.3f} (1.0 = convex)")
    print(f"  Sphericity: {features.sphericity:.3f} (1.0 = sphere)")
    print(f"  Aspect ratio: {features.aspect_ratio:.2f}")
    print(f"  Roughness: {features.roughness:.3f}")

    if features.mean_curvature_mean is not None:
        print(f"\nCurvature:")
        print(f"  Mean curvature: {features.mean_curvature_mean:.4f} ± {features.mean_curvature_std:.4f}")
        print(f"  Gaussian curvature: {features.gaussian_curvature_mean:.4f} ± {features.gaussian_curvature_std:.4f}")

    print(f"\nClassification:")
    print(f"  Complexity score: {features.complexity_score:.3f}")
    print(f"  Surface type: {features.surface_type}")

    # Suggested parameters
    params = suggest_parameters(features)
    print(f"\nSuggested Parameters:")
    print(f"  initial_delta: {params['initial_delta']:.6f}")
    print(f"  delta_factor: {params['delta_factor']:.2f}")
    print(f"  num_steps: {params['num_steps']}")

    print("=" * 60)
