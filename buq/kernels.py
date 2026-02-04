import numpy as np

from emukit.quadrature.interfaces import (
    IStandardKernel,
)



class SumRBFWhiteGPy(IStandardKernel):
    """
    Wrapper for a sum of GPy RBF and White kernels to be used in Emukit quadrature.

    Expects gpy_kernel to be something like RBF + White.
    Works for any input dimension (1D, 2D, ...).
    """

    def __init__(self, gpy_kernel):
        rbf, white = gpy_kernel.parts
        self.gpy_rbf = rbf
        self.gpy_white = white
        self.gpy_kernel = rbf + white

    @property
    def lengthscales(self) -> np.ndarray:
        if self.gpy_rbf.ARD:
            return self.gpy_rbf.lengthscale.values
        return np.full((self.gpy_rbf.input_dim,), float(self.gpy_rbf.lengthscale[0]))

    @property
    def variance(self) -> float:
        return float(self.gpy_rbf.variance[0])

    def K(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        return self.gpy_kernel.K(x1, x2)

    def dK_dx1(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        # x1: (n1, dim), x2: (n2, dim)
        ls2 = self.lengthscales**2
        scaled_vector_diff = np.swapaxes(
            (x1[None, :, :] - x2[:, None, :]) / ls2,
            0,
            -1,
        )  # (dim, n1, n2)
        return -self.K(x1, x2)[None, ...] * scaled_vector_diff

    def dKdiag_dx(self, x: np.ndarray) -> np.ndarray:
        dim = x.shape[1]
        n = x.shape[0]
        return np.zeros((dim, n))


class SumMaternWhiteGPy(IStandardKernel):
    """
    Wrapper for a sum of a Matern-type kernel and a White kernel for Emukit.

    Can wrap:
      - GPy.kern.Exponential (Matern 1/2)
      - GPy.kern.Matern32
      - GPy.kern.Matern52
    as long as the sum is matern + white and the API matches.

    The derivative formula used here is the same as for RBF, which is exact
    for RBF but an approximation for general Matérn kernels. This is what
    you have been using; if you ever need exact derivatives, we would need
    to implement the Matérn-specific formula.
    """

    def __init__(self, gpy_kernel):
        matern, white = gpy_kernel.parts
        self.gpy_matern = matern
        self.gpy_white = white
        self.gpy_kernel = matern + white

    @property
    def lengthscales(self) -> np.ndarray:
        if self.gpy_matern.ARD:
            return self.gpy_matern.lengthscale.values
        return np.full(
            (self.gpy_matern.input_dim,), float(self.gpy_matern.lengthscale[0])
        )

    @property
    def variance(self) -> float:
        return float(self.gpy_matern.variance[0])

    def K(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        return self.gpy_kernel.K(x1, x2)

    def dK_dx1(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        ls2 = self.lengthscales**2
        scaled_vector_diff = np.swapaxes(
            (x1[None, :, :] - x2[:, None, :]) / ls2,
            0,
            -1,
        )  # (dim, n1, n2)
        return -self.K(x1, x2)[None, ...] * scaled_vector_diff

    def dKdiag_dx(self, x: np.ndarray) -> np.ndarray:
        dim = x.shape[1]
        n = x.shape[0]
        return np.zeros((dim, n))

