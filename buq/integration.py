import sys
import numpy as np
from scipy import optimize as scipy_optimize

from scipy.integrate import cumulative_simpson


def integration_1D(
    x_grid: np.ndarray,
    dA_dx: np.ndarray,
) -> np.ndarray:
    """
    Integrate a 1D free-energy derivative dA/dx over x using cumulative trapezoidal rule.

    Parameters
    ----------
    x_grid : ndarray, shape (n,)
        Monotonic grid of x values.
    dA_dx : ndarray, shape (n,) or (n, 1)
        Free-energy derivative dA/dx evaluated at x_grid.

    Returns
    -------
    A : ndarray, shape (n,)
        Integrated free energy, shifted so that min(A) = 0.
    """
    x_grid = np.asarray(x_grid).ravel()
    dA_dx = np.asarray(dA_dx).ravel()

    # cumulative_trapezoid integrates y(x) -> ∫ y dx
    A = cumulative_simpson(dA_dx, x=x_grid, initial=0.0)
    # shift minimum to zero
    A -= np.min(A)
    return A


def integration_2D_rgrid(
    grid: np.ndarray,
    dA_grid: np.ndarray,
    integrator: str = "simpson+mini",
    fast: bool = False,
) -> np.ndarray:
    """
    Integrate a 2D regular/rectangular grid from its gradient.

    Parameters
    ----------
    grid : ndarray, shape (n_j, n_i, 2)
        Grid coordinates. Convention: grid[j, i, :] = (x_i, y_j).
    dA_grid : ndarray, shape (n_j, n_i, 2)
        Free-energy derivatives: dA_grid[..., 0] = dA/dx, dA_grid[..., 1] = dA/dy.
    integrator : {'trapz', 'simpson', 'trapz+mini', 'simpson+mini', 'fourier'}
        Integration algorithm (currently only Simpson variants are implemented
        explicitly here; the string is kept for compatibility / future use).
    fast : bool
        If True, limit the number of L-BFGS-B iterations and print callback progress.

    Returns
    -------
    A_grid : ndarray, shape (n_j, n_i)
        Integrated free energy on the grid, minimum value set to zero.
    """
    # grid-related definitions
    n_jg = grid.shape[0]
    n_ig = grid.shape[1]
    n_grid = n_jg * n_ig

    # spacing between points
    dx = abs(grid[0, 0, 0] - grid[0, 1, 0])
    dy = abs(grid[0, 0, 1] - grid[1, 0, 1])

    # initialize integrated surface matrix
    A_grid = np.zeros((n_jg, n_ig))

    # difference of gradients per grid point [Kästner 2009, Eq. 14]
    def D_tot(F_flat):
        F = F_flat.reshape(n_jg, n_ig)
        dFy, dFx = np.gradient(F, dy, dx)
        dF = np.stack((dFx, dFy), axis=-1)
        return np.sum((dA_grid - dF) ** 2) / n_grid

    def callback(A):
        print(f"Current loss: {D_tot(A):.6f}")

    # Simpson-like integration on the grid
    sys.stdout.write("# Integrating             - Simpson's rule ")
    for j in range(n_jg):
        for i in range(n_ig):
            if i == 0 and j == 0:
                A_grid[j, i] = 0.0  # reference point
            elif i == 0:
                # integrate along y
                A_grid[j, i] = (
                    A_grid[j - 1, i]
                    + (dA_grid[j - 1, i, 1] + dA_grid[j, i, 1]) * dy / 2.0
                )
            elif j == 0:
                # integrate along x
                A_grid[j, i] = (
                    A_grid[j, i - 1]
                    + (dA_grid[j, i - 1, 0] + dA_grid[j, i, 0]) * dx / 2.0
                )
            else:
                # 2D Simpson-like step
                A_grid[j, i] = (
                    A_grid[j - 1, i - 1]
                    + (
                        dA_grid[j - 1, i - 1, 0]
                        + dA_grid[j - 1, i, 0]
                        + dA_grid[j, i - 1, 0]
                        + dA_grid[j, i, 0]
                    )
                    * dx
                    / 4.0
                    + (
                        dA_grid[j - 1, i - 1, 1]
                        + dA_grid[j - 1, i, 1]
                        + dA_grid[j, i - 1, 1]
                        + dA_grid[j, i, 1]
                    )
                    * dy
                    / 4.0
                )

    # Optional real-space minimization to enforce consistency of gradients
    if "mini" in integrator:
        sys.stdout.write("+ Real Space Grid Mini ")
        sys.stdout.flush()

        options = {
            "maxfun": np.inf,
            "maxls": 50,
            "iprint": -1,
        }
        if fast:
            options["maxiter"] = 80
            options["iprint"] = 10
        else:
            options["maxiter"] = np.inf

        mini_result = scipy_optimize.minimize(
            D_tot,
            A_grid.ravel(),
            method="L-BFGS-B",
            options=options,
            callback=callback if fast else None,
        )
        if not mini_result.success:
            sys.stdout.write("\nWARNING: Minimization could not converge")
        A_grid = mini_result.x.reshape(n_jg, n_ig)

    sys.stdout.write(f"\n# Integration error:        {D_tot(A_grid.ravel()):.2f}\n\n")

    # set minimum to zero
    A_grid = A_grid - np.min(A_grid)

    return A_grid


def integrate_from_grad(
    coords: np.ndarray,
    grad: np.ndarray,
    integrator: str = "simpson+mini",
    fast: bool = False,
) -> np.ndarray:
    """
    Convenience wrapper that dispatches to 1D or 2D integration
    based on the shape of coords.

    Parameters
    ----------
    coords : ndarray
        - 1D: shape (n,) or (n, 1)
        - 2D: grid of shape (n_j, n_i, 2)
    grad : ndarray
        - 1D: shape (n,) or (n, 1), dA/dx
        - 2D: shape (n_j, n_i, 2), [dA/dx, dA/dy]
    integrator : str
        Only used for 2D currently ('simpson+mini', etc.).
    fast : bool
        Passed to 2D minimization routine.

    Returns
    -------
    A : ndarray
        - 1D: shape (n,)
        - 2D: shape (n_j, n_i)
    """
    coords = np.asarray(coords)

    # 1D case
    if coords.ndim == 1 or (coords.ndim == 2 and coords.shape[1] == 1):
        x_grid = coords.ravel()
        dA_dx = np.asarray(grad).ravel()
        return integration_1D(x_grid, dA_dx)

    # 2D case: expect (n_j, n_i, 2)
    if coords.ndim == 3 and coords.shape[2] == 2:
        return integration_2D_rgrid(coords, np.asarray(grad), integrator, fast)

    raise ValueError(
        f"Unsupported coords shape {coords.shape}. "
        "Expected (n,), (n,1), or (n_j, n_i, 2)."
    )