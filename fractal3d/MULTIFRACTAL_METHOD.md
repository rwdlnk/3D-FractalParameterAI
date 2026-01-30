# Multifractal Analysis: Step-by-Step Explanation

Both 2D and 3D codes follow the same mathematical framework but differ in how they handle the geometry (2D curves vs 3D surfaces).

---

## Mathematical Foundation

The **multifractal formalism** characterizes how a measure (interface length in 2D, surface area in 3D) is distributed across scales:

1. **Partition function**: Z_q(ε) = Σ p_i^q
2. **Scaling exponent**: τ(q) from Z_q(ε) ~ ε^τ(q)
3. **Generalized dimension**: D_q = τ(q) / (q - 1)
4. **Singularity spectrum**: α(q) = dτ/dq, f(α) = qα - τ

For a **monofractal**, D_q is constant for all q. For a **multifractal**, D_q varies with q.

---

## 2D Analysis (rt_multifractal_analysis.py)

**Input**: Interface segments as (N, 4) array [x1, y1, x2, y2]

### Step 1: Setup q-values
```
q_values = [-5.0, -4.5, ..., 4.5, 5.0]  (21 values, excluding q≈1)
```
Values near q=1 are excluded because D_q = τ(q)/(q-1) has a singularity there.

### Step 2: Generate box sizes
```python
deltas = [initial_delta * (delta_factor ** i) for i in range(num_steps)]
# Example: [0.025, 0.0375, 0.056, ...] (geometric progression, increasing)
```

### Step 3: For each box size δ, compute the measure in each box

**2D approach**:
- Create a grid of boxes of size δ
- For each interface segment:
  - Compute segment length
  - Find box containing segment midpoint
  - Add full segment length to that box
- Result: dictionary `{(ix, iy): total_length_in_box}`
- Normalize: p_i = length_i / total_length

```python
for seg in segments:
    x1, y1, x2, y2 = seg
    seg_length = sqrt((x2-x1)² + (y2-y1)²)
    mid_x, mid_y = (x1+x2)/2, (y1+y2)/2
    ix = int((mid_x - x_min) / delta)
    iy = int((mid_y - y_min) / delta)
    box_lengths[(ix, iy)] += seg_length
```

### Step 4: Compute partition function Z_q(ε) for each q

For each q value and each scale:
```python
if q != 0:
    Z_q = sum(p_i ** q)  # p_i = box_length / total_length
else:
    Z_q = count(p_i > 0)  # number of non-empty boxes
```

### Step 5: Fit τ(q) from scaling relation

For each q:
```python
log_delta = log(deltas)
log_Z = log(Z_q_values)
# Linear regression: log(Z_q) = τ(q) * log(ε) + const
tau, const = polyfit(log_delta, log_Z, 1)
```

### Step 6: Compute generalized dimensions D_q

```python
D_q = tau / (q - 1)  # for q ≠ 1
D_1 = linear_interpolation(D_0.9, D_1.1)  # special case
```

### Step 7: Compute singularity spectrum f(α)

```python
alpha = gradient(tau_q, q)  # numerical derivative dτ/dq
f_alpha = q * alpha - tau_q   # Legendre transform
```

---

## 3D Analysis (multifractal_3d.py)

**Input**: TriangleMesh object with vertices and triangle connectivity

### Steps 1-2: Same as 2D
Setup q-values and generate cube sizes (geometric progression, **decreasing** in 3D).

### Step 3: For each cube size δ, compute surface area in each cube

**3D approach**:
- Create a 3D grid of cubes of size δ
- For each triangle:
  - Compute triangle bounding box
  - Find range of cubes that might intersect
  - For each candidate cube, test triangle-cube intersection using **Separating Axis Theorem (SAT)**
  - Distribute triangle area equally among intersecting cubes

```python
for t_idx in range(n_triangles):
    v0, v1, v2 = triangle_vertices[t_idx]
    area = triangle_areas[t_idx]

    # Find candidate cubes from triangle bounding box
    for i in range(i_min, i_max+1):
        for j in range(j_min, j_max+1):
            for k in range(k_min, k_max+1):
                cube_min = [bbox.min + (i,j,k)*delta]
                cube_max = cube_min + delta

                if triangle_cube_intersects(v0, v1, v2, cube_min, cube_max):
                    intersecting_cubes.append((i, j, k))

    # Distribute area equally
    area_per_cube = area / len(intersecting_cubes)
    for cube in intersecting_cubes:
        cube_areas[cube] += area_per_cube
```

**Triangle-Cube Intersection Test**:

Uses the Separating Axis Theorem with 13 potential separating axes:
1. 3 cube face normals (X, Y, Z axes)
2. 1 triangle normal
3. 9 cross products (3 triangle edges × 3 cube axes)

If any axis separates the triangle from the cube, they don't intersect.

The test is **Numba JIT-compiled** for 20-450x speedup.

### Steps 4-7: Same as 2D
Compute partition functions, fit τ(q), compute D_q and f(α).

---

## Key Differences Summary

| Aspect | 2D | 3D |
|--------|----|----|
| **Input** | Line segments [x1,y1,x2,y2] | Triangle mesh (vertices + faces) |
| **Measure** | Segment length | Triangle surface area |
| **Box type** | 2D boxes (squares) | 3D cubes |
| **Intersection test** | Midpoint assignment | SAT-based triangle-cube test |
| **Area distribution** | Full segment to one box | Distributed among all intersecting cubes |
| **Acceleration** | Pure Python | Numba JIT compilation |
| **Scale direction** | Increasing δ | Decreasing δ |

---

## Physical Interpretation

| Dimension | Meaning |
|-----------|---------|
| **D₀** (capacity) | Box-counting dimension - how many boxes needed to cover the set |
| **D₁** (information) | Shannon entropy scaling - sensitivity to measure distribution |
| **D₂** (correlation) | Pair correlation scaling - clustering of the measure |
| **Δα** (spectrum width) | Degree of multifractality - 0 for monofractal, >0 for multifractal |

For the RT simulations:
- **t=0**: D₀ ≈ D₁ ≈ D₂ ≈ 2.0 (smooth surface, near-monofractal)
- **t=20**: D₂ > D₁ > D₀, Δα ≈ 1.9 (turbulent mixing, multifractal)

---

## Source Code Locations

- **2D**: `/media/rod/Research/ResearchII_III/ResearchIII/githubRepos/FractalParameterAI/integration/rt_multifractal_analysis.py`
- **3D**: `/media/rod/Research/ResearchII_III/ResearchIII/githubRepos/3D-FractalParameterAI/fractal3d/multifractal_3d.py`

---

*Document created: January 30, 2026*
