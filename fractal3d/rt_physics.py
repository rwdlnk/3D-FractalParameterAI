"""
Rayleigh-Taylor Physics Nondimensionalization for 3D Analysis.

Extends the 2D RTPhysics dataclass with domain width W (spanwise)
for use with 3D multifractal and fractal dimension analysis.

Domain convention for 3D RT:
    - H: domain height (gravity direction, Y)
    - L: domain length (streamwise, X)
    - W: domain width  (spanwise, Z)

Dimensionless variables:
    - Time:       tau = t * sqrt(A*g/H)
    - Length:     y/H (vertical), x/L (streamwise), z/W (spanwise)
    - Cube size:  delta/H
    - Velocity:   u / sqrt(A*g*H)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Union

ArrayLike = Union[float, np.ndarray, list]


@dataclass
class RTPhysics3D:
    """RT instability physics parameters for 3D nondimensionalization.

    Attributes:
        A: Atwood number (rho2 - rho1) / (rho2 + rho1)
        g: Gravitational acceleration (m/s^2)
        H: Domain height (m) - gravity direction (Y)
        L: Domain length (m) - streamwise (X)
        W: Domain width (m) - spanwise (Z)
    """
    A: float
    g: float
    H: float
    L: float
    W: float

    # Derived quantities
    tau_factor: float = field(init=False, repr=False)
    velocity_scale: float = field(init=False, repr=False)
    time_scale: float = field(init=False, repr=False)

    def __post_init__(self):
        if self.A <= 0 or self.A > 1:
            raise ValueError(f"Atwood number must be in (0, 1], got {self.A}")
        if self.g <= 0:
            raise ValueError(f"Gravity must be positive, got {self.g}")
        if self.H <= 0:
            raise ValueError(f"Domain height must be positive, got {self.H}")
        if self.L <= 0:
            raise ValueError(f"Domain length must be positive, got {self.L}")
        if self.W <= 0:
            raise ValueError(f"Domain width must be positive, got {self.W}")

        self.tau_factor = np.sqrt(self.A * self.g / self.H)
        self.velocity_scale = np.sqrt(self.A * self.g * self.H)
        self.time_scale = np.sqrt(self.H / (self.A * self.g))

    # --- Nondimensionalization ---

    def nondim_time(self, t: ArrayLike) -> np.ndarray:
        """Convert physical time to dimensionless time tau = t * sqrt(Ag/H)."""
        return np.asarray(t) * self.tau_factor

    def nondim_length(self, x: ArrayLike, scale: str = 'H') -> np.ndarray:
        """Normalize length by H, L, or W.

        Args:
            x: Physical length(s)
            scale: 'H' (height), 'L' (length/streamwise), or 'W' (width/spanwise)
        """
        scales = {'H': self.H, 'L': self.L, 'W': self.W}
        if scale not in scales:
            raise ValueError(f"scale must be 'H', 'L', or 'W', got '{scale}'")
        return np.asarray(x) / scales[scale]

    def nondim_cube_size(self, delta: ArrayLike) -> np.ndarray:
        """Normalize cube size by domain height: delta/H."""
        return np.asarray(delta) / self.H

    def nondim_wavenumber(self, f: ArrayLike, direction: str = 'L') -> np.ndarray:
        """Convert spatial frequency to dimensionless wavenumber.

        Args:
            f: Spatial frequency (1/m)
            direction: 'L' for streamwise (k = 2*pi*f*L) or 'W' for spanwise (k = 2*pi*f*W)
        """
        scales = {'L': self.L, 'W': self.W}
        if direction not in scales:
            raise ValueError(f"direction must be 'L' or 'W', got '{direction}'")
        return 2.0 * np.pi * np.asarray(f) * scales[direction]

    def nondim_velocity(self, u: ArrayLike) -> np.ndarray:
        """Normalize velocity by sqrt(AgH)."""
        return np.asarray(u) / self.velocity_scale

    # --- Inverse ---

    def dim_time(self, tau: ArrayLike) -> np.ndarray:
        """Convert dimensionless time back to physical time."""
        return np.asarray(tau) / self.tau_factor

    def dim_length(self, x_star: ArrayLike, scale: str = 'H') -> np.ndarray:
        """Convert normalized length back to physical length."""
        scales = {'H': self.H, 'L': self.L, 'W': self.W}
        if scale not in scales:
            raise ValueError(f"scale must be 'H', 'L', or 'W', got '{scale}'")
        return np.asarray(x_star) * scales[scale]

    def dim_cube_size(self, delta_star: ArrayLike) -> np.ndarray:
        """Convert normalized cube size back to physical size."""
        return np.asarray(delta_star) * self.H

    # --- Factory methods ---

    @classmethod
    def from_dalziel(cls, A: float = 0.5, g: float = 9.81,
                     H: float = 0.5, L: float = 0.4,
                     W: float = 0.4) -> 'RTPhysics3D':
        """Create with Dalziel (1999) experimental defaults.

        Default parameters: A=0.5, g=9.81, H=0.5m, L=0.4m, W=0.4m.
        """
        return cls(A=A, g=g, H=H, L=L, W=W)

    @classmethod
    def from_vtk_file(cls, vtk_path: str, A: float = 0.5,
                      g: float = 9.81) -> 'RTPhysics3D':
        """Auto-detect H, L, W from a 3D VTK rectilinear grid file.

        Convention: X -> L (length), Y -> H (height), Z -> W (width).

        Args:
            vtk_path: Path to a VTK file with 3D rectilinear grid
            A: Atwood number
            g: Gravitational acceleration
        """
        H, L, W = _read_domain_from_vtk_3d(vtk_path)
        return cls(A=A, g=g, H=H, L=L, W=W)

    def summary(self) -> str:
        """Return a formatted summary of the physics parameters."""
        lines = [
            f"RT Physics Parameters (3D):",
            f"  A = {self.A:.4f}  (Atwood number)",
            f"  g = {self.g:.4f}  m/s^2",
            f"  H = {self.H:.4f}  m  (domain height, Y)",
            f"  L = {self.L:.4f}  m  (domain length, X)",
            f"  W = {self.W:.4f}  m  (domain width, Z)",
            f"  tau_factor = sqrt(Ag/H) = {self.tau_factor:.4f}  s^-1",
            f"  velocity_scale = sqrt(AgH) = {self.velocity_scale:.4f}  m/s",
            f"  time_scale = sqrt(H/(Ag)) = {self.time_scale:.4f}  s",
        ]
        return "\n".join(lines)


def _read_domain_from_vtk_3d(vtk_path: str):
    """Extract domain H, L, W from 3D VTK rectilinear grid.

    Convention: X -> L (length), Y -> H (height), Z -> W (width).

    Returns:
        (H, L, W): Domain height, length, width in meters
    """
    with open(vtk_path, 'r') as f:
        lines = f.readlines()

    coords = {'X': [], 'Y': [], 'Z': []}

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        for axis in ('X', 'Y', 'Z'):
            if line.startswith(f"{axis}_COORDINATES"):
                n = int(line.split()[1])
                vals = []
                i += 1
                while len(vals) < n:
                    vals.extend(float(v) for v in lines[i].strip().split())
                    i += 1
                coords[axis] = vals
                break
        else:
            i += 1

    for axis in ('X', 'Y', 'Z'):
        if not coords[axis]:
            raise ValueError(f"Could not extract {axis} coordinates from {vtk_path}")

    L = max(coords['X']) - min(coords['X'])
    H = max(coords['Y']) - min(coords['Y'])
    W = max(coords['Z']) - min(coords['Z'])
    return H, L, W
