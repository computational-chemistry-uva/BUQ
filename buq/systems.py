from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


class CollectiveVariableSystem(ABC):
    """
    Abstract interface for a system with 1D or 2D collective variables (CVs).

    Subclasses must implement:
      - write_plumed_input(x)
      - run_simulation(x)
      - get_force(x)

    and should define:
      - self.bounds:  (x_min, x_max) or (x_min, x_max, y_min, y_max)
    """

    def __init__(self, dim: int, bounds: Tuple[float, ...]):
        assert dim in (1, 2), "Only 1D and 2D CVs are supported for now."
        self.dim = dim
        self.bounds = bounds  # shape per dim: 1D -> (x_min, x_max), 2D -> (x_min, x_max, y_min, y_max)

    @abstractmethod
    def write_plumed_input(self, x: np.ndarray) -> None:
        """Create PLUMED input for CV point x (shape (dim,))."""

    @abstractmethod
    def run_simulation(self, x: np.ndarray) -> None:
        """Run the MD simulation corresponding to CV point x."""

    @abstractmethod
    def get_force(self, x: np.ndarray) -> np.ndarray:
        """Return dF/dx at position x, as an array of shape (dim,)."""