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
    integration_1D,
    integration_2D_rgrid,
    integrate_from_grad,
)

__all__ = [
    "CollectiveVariableSystem",
    "BQConfig",
    "BayesianQuadratureRunner",
    "SumRBFWhiteGPy",
    "SumMaternWhiteGPy",
    "integration_1D",
    "integration_2D_rgrid",
    "integrate_from_grad",
    "Mock1DSystem",
    "Mock2DSystem",
    "Adipep1DFromGrid",
    "Adipep2DFromGrid",
]

__version__ = "0.1.0"