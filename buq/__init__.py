"""
buq: Bayesian Umbrella Quadrature library.
"""

from .systems import CollectiveVariableSystem
from .bq_runner import BQConfig, BayesianQuadratureRunner
from .kernels import (
    SumRBFWhiteGPy,
    SumMaternWhiteGPy,
)
from .integration import (
    integration_1D_trapz,
    integration_2D_rgrid,
    integrate_from_grad,
)

__all__ = [
    "CollectiveVariableSystem",
    "BQConfig",
    "BayesianQuadratureRunner",
    "SumRBFWhiteGPy",
    "SumMaternWhiteGPy",
    "integration_1D_trapz",
    "integration_2D_rgrid",
    "integrate_from_grad",
    "AdipepFromGrid",
    "Mock1DSystem",
    "Mock2DSystem",
]

__version__ = "0.1.0"