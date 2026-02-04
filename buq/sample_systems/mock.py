import numpy as np
from buq.systems import CollectiveVariableSystem
from scipy.interpolate import RegularGridInterpolator
from pathlib import Path

class Mock1DSystem(CollectiveVariableSystem):
    """
    A(x) = 0.5 * x^2  =>  dA/dx = x
    """
    def __init__(self):
        super().__init__(dim=1, bounds = (-2.0, 2.0))

    def write_plumed_input(self, x: np.ndarray) -> None:
        pass

    def run_simulation(self, x: np.ndarray) -> None:
        pass

    def get_force(self, x: np.ndarray) -> np.ndarray:
        return np.array([x[0]])  # dA/dx = x

    def true_fes(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x).ravel()
        return 0.5 * x**2
    
    def true_grad(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x).ravel()
        return x


class Mock2DSystem(CollectiveVariableSystem):
    """
    A(x,y) = 0.5 * x^2 + 0.33 *  y^3)  =>  grad A = [x, y^2]
    """
    def __init__(self):
        super().__init__(dim=2,bounds = (-2.0, 2.0, -2.0, 2.0))

    def write_plumed_input(self, x: np.ndarray) -> None:
        pass  # no-op

    def run_simulation(self, x: np.ndarray) -> None:
        pass  # no-op

    def get_force(self, x: np.ndarray) -> np.ndarray:
        return np.array([x[0], x[1] * x[1]])  # [dA/dx, dA/dy] = [x, y^2]
    
class AdipepFromGrid(CollectiveVariableSystem):
    """
    2D system using gradients from a metadynamics grid (phi, psi).
    Expects fes.dat columns: phi, psi, fes, dA/dphi, dA/dpsi on a regular 100x100 grid.
    """
    def __init__(self, fes_path: str = None):
        super().__init__(dim=2,bounds = (-np.pi, np.pi, -np.pi, np.pi))

        # Resolve fes.dat path
        if fes_path is None:
            # Look next to this module (mock.py)
            module_dir = Path(__file__).parent
            candidate = module_dir / "fes.dat"
            if candidate.exists():
                fes_path = str(candidate)
            else:
                # fallback to CWD
                cwd_candidate = Path("fes.dat")
                if not cwd_candidate.exists():
                    raise FileNotFoundError(
                        "fes.dat not found. Place it next to mock.py or in the current working directory, "
                        "or pass fes_path to MockAdipep(...)."
                    )
                fes_path = str(cwd_candidate)

        # Load and build grids/interpolators
        self._load_fes(fes_path)

    def _load_fes(self, path: str) -> None:
        data = np.loadtxt(path)
        # Expect 5 columns: phi, psi, fes, dphi, dpsi
        phi = data[:, 0].reshape(100, 100)
        psi = data[:, 1].reshape(100, 100)
        fes = data[:, 2].reshape(100, 100)
        dphi = data[:, 3].reshape(100, 100)  # dA/dphi
        dpsi = data[:, 4].reshape(100, 100)  # dA/dpsi

        # Grids (unique sorted coordinates)
        self.x_grid = np.unique(phi.ravel())  # phi
        self.y_grid = np.unique(psi.ravel())  # psi

        # Interpolators
        # Note: If your phi/psi matrices are transposed relative to dx/dy, you may need .T.
        # The following matches your previous working pattern (use .T):
        self._dphi_interp = RegularGridInterpolator(
            (self.x_grid, self.y_grid), dphi.T, bounds_error=False, fill_value=None
        )
        self._dpsi_interp = RegularGridInterpolator(
            (self.x_grid, self.y_grid), dpsi.T, bounds_error=False, fill_value=None
        )

        # Optional: store FES for reference plots (shift min to zero)
        self.fes_grid = fes.T - np.min(fes)

    def write_plumed_input(self, x: np.ndarray) -> None:
        pass  # no-op

    def run_simulation(self, x: np.ndarray) -> None:
        pass  # no-op

    def get_force(self, x: np.ndarray) -> np.ndarray:
        """
        Return gradient [dA/dphi, dA/dpsi] at (phi, psi).
        """
        phi_val = float(x[0])
        psi_val = float(x[1])
        dA_dphi = float(self._dphi_interp((phi_val, psi_val)))
        dA_dpsi = float(self._dpsi_interp((phi_val, psi_val)))
        return np.array([dA_dphi, dA_dpsi], dtype=float)
