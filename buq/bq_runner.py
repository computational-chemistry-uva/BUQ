from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union, List
import numpy as np
import GPy
import matplotlib.pyplot as plt
from emukit.quadrature.acquisitions import IntegralVarianceReduction, UncertaintySampling, MutualInformation
from emukit.quadrature.methods import VanillaBayesianQuadrature
from emukit.model_wrappers.gpy_quadrature_wrappers import BaseGaussianProcessGPy
from emukit.quadrature.measures import LebesgueMeasure
from buq.systems import CollectiveVariableSystem
from buq.integration import integration_1D, integration_2D_rgrid
from buq.kernels import (
    SumRBFWhiteGPy,
    SumMaternWhiteGPy,
)
from emukit.quadrature.kernels import (
    QuadratureRBFLebesgueMeasure,
    QuadratureProductMatern52LebesgueMeasure,
    QuadratureProductMatern12LebesgueMeasure,
    QuadratureProductMatern32LebesgueMeasure,
)

ArrayLike = Union[np.ndarray, List[float]]


@dataclass
class BQConfig:
    kernel_type: str              # "RBF", "Matern52", "Matern32","Matern12"
    lengthscale: Union[float, np.ndarray]
    noise: float
    variance: float = 1.0
    n_queries: int = 0

    grid_size_1d: int = 100
    grid_size_2d: Tuple[int, int] = (50, 50)

    use_mini: bool = True
    fast_mini: bool = False

    optimize_hyperparams: bool = False

    acq_function: str = "IVR"     # "IVR", "US" or "MI"

    # NEW: how many variables are integrated over in the FES
    # If None: all dims are integrated (old behaviour).
    n_integrated: Optional[int] = None

    # NEW: for simple slice case with extra variables (e.g. dim=2, n_integrated=1),
    # extra_context gives the fixed value of the extra variable(s) for FES.
    # For dim=2, n_integrated=1: this is the y/psi value at which to compute A(x, y_ctx).
    extra_context: Optional[ArrayLike] = None


class BayesianQuadratureRunner:
    """
    High-level runner that:
      - interacts with a CollectiveVariableSystem,
      - fits GPs to the gradient (forces),
      - proposes new evaluation points,
      - integrates the gradient to obtain a free-energy estimate.

    Works for both 1D and 2D systems:
      - dim = 1: A(x)
      - dim = 2: A(x, y) or A(x, y_ctx) when using a slice.
    """

    def __init__(
        self,
        system: CollectiveVariableSystem,
        config: BQConfig,
        bounds: Optional[Tuple[float, ...]] = None,
    ):
        """
        Parameters
        ----------
        system : CollectiveVariableSystem
            The MD system implementation.
            Must have system.dim in {1,2} and a .bounds attribute.
        config : BQConfig
            Configuration for kernels, noise, grid sizes, etc.
        bounds : tuple or None
            If None, uses system.bounds.
            If provided, overrides system.bounds.
        """
        self.emukit_method = None
        self.acq_function = None
        self.system = system
        self.dim = system.dim
        self.config = config

        # How many dims are integrated in the FES?
        # None => integrate all dims (old behaviour).
        if self.config.n_integrated is None:
            self.n_integrated = self.dim
        else:
            self.n_integrated = int(self.config.n_integrated)
            if not (1 <= self.n_integrated <= self.dim):
                raise ValueError(f"n_integrated must be between 1 and dim={self.dim}")
        # Extra context for non-integrated dims (currently used for dim=2, n_integrated=1)
        self.extra_context = self.config.extra_context

        # --- default to system.bounds if not provided ---
        if bounds is None:
            if not hasattr(system, "bounds"):
                raise ValueError("System must define a 'bounds' attribute or pass bounds explicitly.")
            bounds = system.bounds

        if self.dim == 1:
            if len(bounds) != 2:
                raise ValueError("For 1D systems, bounds must be (x_min, x_max).")
            self.bounds_1d = (bounds[0], bounds[1])
            self.bounds_2d = None
        elif self.dim == 2:
            if len(bounds) != 4:
                raise ValueError("For 2D systems, bounds must be (x_min, x_max, y_min, y_max).")
            self.bounds_2d = (bounds[0], bounds[1], bounds[2], bounds[3])
            self.bounds_1d = None
        else:
            raise ValueError("BayesianQuadratureRunner currently supports only dim = 1 or 2.")

        # Data and models
        self.X_data: Optional[np.ndarray] = None       # shape (n_samples, dim)
        self.Y_data: Optional[np.ndarray] = None       # shape (n_samples, dim)

        # Grids for prediction / integration
        #  - 1D: x_grid_1d for FES; grid_flat for acquisition / derivatives
        #  - 2D: X_grid_2d, Y_grid_2d for FES/acq; grid_flat flattened version
        self.x_grid_1d: Optional[np.ndarray] = None    # shape (n,) for 1D or 1D slice
        self.X_grid_2d: Optional[np.ndarray] = None    # shape (nx, ny)
        self.Y_grid_2d: Optional[np.ndarray] = None    # shape (nx, ny)
        self.grid_flat: Optional[np.ndarray] = None    # shape (N, dim) flattened grid

        # Current free energy estimate
        self.current_fes_1d: Optional[np.ndarray] = None   # shape (n,)
        self.current_fes_2d: Optional[np.ndarray] = None   # shape (nx, ny)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def initialize(self, initial_points: np.ndarray) -> None:
        """
        Run simulations at initial points, fit initial GPs, and compute initial FES.

        Parameters
        ----------
        initial_points : ndarray, shape (n_init, dim)
            Initial CV locations at which to run the MD simulations.
        """
        initial_points = np.atleast_2d(initial_points)
        if initial_points.shape[1] != self.dim:
            raise ValueError(f"initial_points must have shape (n, {self.dim}).")

        # Run initial simulations
        forces = []
        for x in initial_points:
            print(f"Running initial simulation at x = {x}")
            try:
                f = self.system.get_force(x)  # shape (dim,) # check if simulation already done
            except Exception:
                self.system.run_simulation(x)
            f = self.system.get_force(x)  # shape (dim,)
            f = np.asarray(f).reshape(self.dim)
            forces.append(f)

        self.X_data = initial_points
        self.Y_data = np.vstack(forces)  # (n_samples, dim)

        # Build grid for prediction / integration / acquisition
        self._build_grid()

        # --- Build GPy kernel ---
        if self.config.kernel_type == "RBF":
            base = GPy.kern.RBF(self.dim, lengthscale=self.config.lengthscale,
                                variance=self.config.variance, ARD=True)
        elif self.config.kernel_type == "Matern52":
            base = GPy.kern.Matern52(self.dim, lengthscale=self.config.lengthscale,
                                     variance=self.config.variance, ARD=True)
        elif self.config.kernel_type == "Matern12":
            base = GPy.kern.Exponential(self.dim, lengthscale=self.config.lengthscale,
                                        variance=self.config.variance, ARD=True)
        elif self.config.kernel_type == "Matern32":
            base = GPy.kern.Matern32(self.dim, lengthscale=self.config.lengthscale,
                                     variance=self.config.variance, ARD=True)
        else:
            raise ValueError(f"Unknown kernel_type '{self.config.kernel_type}'.")

        white = GPy.kern.White(self.dim, variance=self.config.noise)
        kernel = base + white

        gpy_model = GPy.models.GPRegression(X=self.X_data, Y=self.Y_data, kernel=kernel)
        if self.config.optimize_hyperparams:
            gpy_model.optimize(messages=False, max_iters=100)

        # --- Wrap in Emukit kernel and measure ---
        # Measure is over all CV dimensions (integrated + extra)
        if self.dim == 1:
            bounds_list = [(self.bounds_1d[0], self.bounds_1d[1])]
        else:
            x_min, x_max, y_min, y_max = self.bounds_2d
            bounds_list = [(x_min, x_max), (y_min, y_max)]
        measure = LebesgueMeasure.from_bounds(bounds=bounds_list)

        if self.config.kernel_type == "RBF":
            emukit_kernel = SumRBFWhiteGPy(gpy_model.kern)
            quad_kernel = QuadratureRBFLebesgueMeasure(emukit_kernel, measure)
        elif self.config.kernel_type == "Matern52":
            emukit_kernel = SumMaternWhiteGPy(gpy_model.kern)
            quad_kernel = QuadratureProductMatern52LebesgueMeasure(emukit_kernel, measure)
        elif self.config.kernel_type == "Matern12":
            emukit_kernel = SumMaternWhiteGPy(gpy_model.kern)
            quad_kernel = QuadratureProductMatern12LebesgueMeasure(emukit_kernel, measure)
        elif self.config.kernel_type == "Matern32":
            emukit_kernel = SumMaternWhiteGPy(gpy_model.kern)
            quad_kernel = QuadratureProductMatern32LebesgueMeasure(emukit_kernel, measure)

        # --- Emukit base GP and VBQ method ---
        emu_base_gp = BaseGaussianProcessGPy(kern=quad_kernel, gpy_model=gpy_model)
        self.emukit_method = VanillaBayesianQuadrature(base_gp=emu_base_gp,
                                                       X=self.X_data, Y=self.Y_data)

        # Acquisition settings from config
        self.acq_function = self.config.acq_function  # "IVR", "MI", or "US"
        self._update_fes()

    def initialize_from_data(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        Initialize the runner directly from precomputed data, without
        calling system.get_force or system.run_simulation.

        Parameters
        ----------
        X : ndarray, shape (n_samples, dim)
            CV locations.
        Y : ndarray, shape (n_samples, dim)
            Corresponding mean forces at those CVs.
        """
        X = np.atleast_2d(X)
        Y = np.atleast_2d(Y)

        if X.shape[1] != self.dim:
            raise ValueError(f"X must have shape (n, {self.dim}); got {X.shape}.")
        if Y.shape[1] != self.dim:
            raise ValueError(f"Y must have shape (n, {self.dim}); got {Y.shape}.")

        self.X_data = X
        self.Y_data = Y

        # Build grid for prediction / integration / acquisition
        self._build_grid()

        # --- Build GPy kernel (same as in initialize) ---
        if self.config.kernel_type == "RBF":
            base = GPy.kern.RBF(self.dim, lengthscale=self.config.lengthscale,
                                variance=self.config.variance, ARD=True)
        elif self.config.kernel_type == "Matern52":
            base = GPy.kern.Matern52(self.dim, lengthscale=self.config.lengthscale,
                                     variance=self.config.variance, ARD=True)
        elif self.config.kernel_type == "Matern12":
            base = GPy.kern.Exponential(self.dim, lengthscale=self.config.lengthscale,
                                        variance=self.config.variance, ARD=True)
        elif self.config.kernel_type == "Matern32":
            base = GPy.kern.Matern32(self.dim, lengthscale=self.config.lengthscale,
                                     variance=self.config.variance, ARD=True)
        else:
            raise ValueError(f"Unknown kernel_type '{self.config.kernel_type}'.")

        white = GPy.kern.White(self.dim, variance=self.config.noise)
        kernel = base + white

        gpy_model = GPy.models.GPRegression(X=self.X_data, Y=self.Y_data, kernel=kernel)
        if self.config.optimize_hyperparams:
            gpy_model.optimize(messages=False, max_iters=100)

        # --- Wrap in Emukit kernel and measure ---
        if self.dim == 1:
            bounds_list = [(self.bounds_1d[0], self.bounds_1d[1])]
        else:
            x_min, x_max, y_min, y_max = self.bounds_2d
            bounds_list = [(x_min, x_max), (y_min, y_max)]
        measure = LebesgueMeasure.from_bounds(bounds=bounds_list)

        if self.config.kernel_type == "RBF":
            emukit_kernel = SumRBFWhiteGPy(gpy_model.kern)
            quad_kernel = QuadratureRBFLebesgueMeasure(emukit_kernel, measure)
        elif self.config.kernel_type == "Matern52":
            emukit_kernel = SumMaternWhiteGPy(gpy_model.kern)
            quad_kernel = QuadratureProductMatern52LebesgueMeasure(emukit_kernel, measure)
        elif self.config.kernel_type == "Matern12":
            emukit_kernel = SumMaternWhiteGPy(gpy_model.kern)
            quad_kernel = QuadratureProductMatern12LebesgueMeasure(emukit_kernel, measure)
        elif self.config.kernel_type == "Matern32":
            emukit_kernel = SumMaternWhiteGPy(gpy_model.kern)
            quad_kernel = QuadratureProductMatern32LebesgueMeasure(emukit_kernel, measure)

        emu_base_gp = BaseGaussianProcessGPy(kern=quad_kernel, gpy_model=gpy_model)
        self.emukit_method = VanillaBayesianQuadrature(base_gp=emu_base_gp,
                                                       X=self.X_data, Y=self.Y_data)

        self.acq_function = self.config.acq_function
        self._update_fes()

    def run(
        self,
        n_queries: Optional[int] = None,
        weight_var: float = 1.0,
        weight_fes: float = 0.0,
        weight_path: float = 0.0,
        sampling_grid: Optional[np.ndarray] = None,
    ) -> None:
        """
        Run the adaptive loop for n_queries steps.
        """
        if n_queries is None:
            n_queries = self.config.n_queries

        if n_queries <= 0:
            print("No adaptive queries requested (n_queries <= 0).")
            return

        for q in range(n_queries):
            print(f"\n=== Adaptive iteration {q + 1} / {n_queries} ===")
            self.run_one_query(
                weight_var=weight_var,
                weight_fes=weight_fes,
                weight_path=weight_path,
                sampling_grid=sampling_grid,
                compute_fes=False,
            )
        self._update_fes()  # always compute FES after all queries are done

    def run_one_query(
        self,
        weight_var: float = 1.0,
        weight_fes: float = 0.0,
        weight_path: float = 0.0,
        sampling_grid: Optional[np.ndarray] = None,
        compute_fes: bool = True,
    ) -> None:
        """
        Perform a single adaptive query.
        """
        x_next = self._select_next_point(
            weight_var=weight_var,
            weight_fes=weight_fes,
            weight_path=weight_path,
            sampling_grid=sampling_grid,
        )
        print(f"Selected next point: {x_next}")

        # Run simulation at x_next
        try:
            f_next = self.system.get_force(x_next)  # check if already run
        except Exception:
            self.system.run_simulation(x_next)
            f_next = self.system.get_force(x_next)
        f_next = np.asarray(f_next).reshape(1, self.dim)

        # Append data
        self.X_data = np.vstack([self.X_data, x_next.reshape(1, self.dim)])
        self.Y_data = np.vstack([self.Y_data, f_next])

        # Sync Emukit and recompute
        self.emukit_method.set_data(self.X_data, self.Y_data)
        if compute_fes:
            self._update_fes()

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_fes(
        self,
        savepath: Optional[str] = None,
        show: bool = True,
        true_fes_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        align_min: bool = True,
        true_label: str = "Analytical FES",
    ) -> None:
        """
        Plot the current free energy estimate.

        - 1D or 2D+slice (dim=2, n_integrated=1): line plot A(x).
        - 2D full (dim=2, n_integrated=2): contour plot A(x, y).
        """
        # Treat dim=2, n_integrated=1 as "1D FES" for plotting
        is_1d_fes = (self.dim == 1) or (self.dim == 2 and self.n_integrated == 1)

        if is_1d_fes:
            if self.current_fes_1d is None or self.x_grid_1d is None:
                raise RuntimeError("FES not available; call initialize() first.")

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(self.x_grid_1d, self.current_fes_1d, label="FES (BQ)", color="C0")
            if self.X_data is not None:
                xq = self.X_data[:, 0]
                xq_clipped = np.clip(xq, self.x_grid_1d[0], self.x_grid_1d[-1])
                yq = np.interp(xq_clipped, self.x_grid_1d, self.current_fes_1d)

                ax.scatter(xq_clipped, yq, color="red", marker="x", label="Queries")

            # Optional analytical/reference curve
            ref_func = true_fes_func
            if ref_func is None and hasattr(self.system, "true_fes"):
                ref_func = getattr(self.system, "true_fes")

            if callable(ref_func):
                y_true = np.asarray(ref_func(self.x_grid_1d)).ravel()
                if align_min:
                    y_true = y_true - np.min(y_true)
                ax.plot(self.x_grid_1d, y_true, label=true_label, color="C1", linestyle="--")

            ax.set_xlabel("CV")
            ax.set_ylabel("Free energy (arb. units)")
            title = "Bayesian Quadrature FES (1D"
            if self.dim == 2 and self.n_integrated == 1:
                title += " slice)"
            else:
                title += ")"
            ax.set_title(title)
            ax.legend()

        else:
            if self.current_fes_2d is None or self.X_grid_2d is None or self.Y_grid_2d is None:
                raise RuntimeError("FES not available; call initialize() first.")

            fig, ax = plt.subplots(figsize=(7, 6))
            contour = ax.contourf(self.X_grid_2d, self.Y_grid_2d, self.current_fes_2d,
                                  levels=100, cmap="viridis")
            cbar = fig.colorbar(contour, ax=ax)
            cbar.set_label("Free energy (arb. units)")
            if self.X_data is not None:
                ax.scatter(self.X_data[:, 0], self.X_data[:, 1],
                           color="white", edgecolor="black", s=20, label="Queries")
                ax.legend()
            ax.set_xlabel("CV1")
            ax.set_ylabel("CV2")
            ax.set_title("Bayesian Quadrature FES (2D)")

        if savepath is not None:
            plt.savefig(savepath, dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close()

    def plot_acq(
        self,
        weight_var: float = 1.0,
        weight_fes: float = 0.0,
        weight_path: float = 0.0,
        sampling_grid: Optional[np.ndarray] = None,
        savepath: Optional[str] = None,
        show: bool = True,
        full: bool = True,
    ) -> None:
        """
        Plot variance-based acquisition plus FES term and path term.
        """
        var_norm, fes_norm, acq = self._compute_acquisition_grid(
            weight_var=weight_var,
            weight_fes=weight_fes,
            weight_path=weight_path,
            sampling_grid=sampling_grid,
        )

        if self.dim == 1:
            fig, ax = plt.subplots(1, 1, figsize=(7, 4))
            x = self.x_grid_1d
            label = self.acq_function + " acquisition"
            ax.plot(x, var_norm, label=label, color="C0")
            ax.plot(x, fes_norm, label="Normalized FES", color="C1")
            ax.plot(x, acq, label="Combined acquisition", color="C3")
            ax.set_xlabel("CV")
            ax.set_ylabel("Value (normalized)")
            ax.set_title("IVR / FES / acquisition (1D)")
            ax.legend()

        else:
            X = self.X_grid_2d
            Y = self.Y_grid_2d
            var_grid = var_norm
            fes_grid = fes_norm
            acq_grid = acq

            if full:
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))

                # Plot 1: variance
                label = self.acq_function + " variance"
                contour1 = axes[0].contourf(X, Y, var_grid, levels=100, cmap="viridis")
                cbar1 = fig.colorbar(contour1, ax=axes[0], shrink=0.8, aspect=30, pad=0.02)
                cbar1.set_label(label)
                axes[0].set_title(f"{label} Contour")

                # Plot 2: scaled FES
                contour2 = axes[1].contourf(X, Y, fes_grid, levels=100, cmap="plasma")
                cbar2 = fig.colorbar(contour2, ax=axes[1], shrink=0.8, aspect=30, pad=0.02)
                cbar2.set_label("Scaled FES")
                axes[1].set_title("Scaled FES")

                # Plot 3: combined
                contour3 = axes[2].contourf(X, Y, acq_grid, levels=100, cmap="cividis")
                cbar3 = fig.colorbar(contour3, ax=axes[2], shrink=0.8, aspect=30, pad=0.02)
                cbar3.set_label("Combined acquisition")
                axes[2].set_title("Combined acquisition")

            else:
                fig, ax = plt.subplots(figsize=(7, 6))
                contour = ax.contourf(X, Y, acq_grid, levels=100, cmap="viridis")
                cbar = fig.colorbar(contour, ax=ax)
                cbar.set_label("Combined acquisition")
                ax.set_title("Acquisition")

        if savepath is not None:
            plt.savefig(savepath, dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close()

    def plot_derivatives(
        self,
        true_1d: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        true_2d: Optional[Callable[[np.ndarray, np.ndarray], tuple]] = None,
        savepath: Optional[str] = None,
        show: bool = True,
    ) -> None:
        """
        Plot predicted derivatives and (optionally) analytical derivatives.
        Uses the existing emukit_method; does not rebuild any models.
        """
        if self.grid_flat is None:
            self._build_grid()
        if self.emukit_method is None:
            raise RuntimeError("Emukit not initialized. Call initialize() first.")

        grad = self._predict_grad_on_grid()  # shape: (n,1) in 1D, (nx,ny,2) in 2D

        if self.dim == 1:
            if self.x_grid_1d is None:
                self._build_grid()
            x = self.x_grid_1d
            dA_dx = grad[:, 0]

            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(x, dA_dx, label="Predicted dA/dx", color="C0")

            # Observations
            if self.X_data is not None and self.Y_data is not None:
                ax.scatter(self.X_data[:, 0], self.Y_data[:, 0],
                           color="red", marker="x", label="Observed dA/dx")

            # Analytical overlay
            ref_fn = true_1d
            if ref_fn is None:
                if hasattr(self.system, "true_grad"):
                    ref_fn = getattr(self.system, "true_grad")
                elif hasattr(self.system, "true_force"):
                    ref_fn = getattr(self.system, "true_force")
            if callable(ref_fn):
                y_true = np.asarray(ref_fn(x)).ravel()
                ax.plot(x, y_true, label="Analytical dA/dx", color="C1", linestyle="--")

            ax.set_xlabel("CV")
            ax.set_ylabel("dA/dx")
            ax.set_title("Derivative (1D)")
            ax.legend()

            if savepath is not None:
                plt.savefig(savepath, dpi=300, bbox_inches="tight")
            if show:
                plt.show()
            else:
                plt.close()

        else:
            if self.X_grid_2d is None or self.Y_grid_2d is None:
                self._build_grid()
            X = self.X_grid_2d
            Y = self.Y_grid_2d
            Zx = grad[:, :, 0]
            Zy = grad[:, :, 1]

            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            # dA/dx
            c1 = axes[0].contourf(X, Y, Zx, levels=100, cmap="viridis")
            axes[0].set_title("Predicted dA/dx (2D)")
            axes[0].set_xlabel("CV1"); axes[0].set_ylabel("CV2")
            cb1 = fig.colorbar(c1, ax=axes[0]); cb1.set_label("dA/dx")
            if callable(true_2d):
                Zx_true, Zy_true = true_2d(X, Y)
                axes[0].contour(X, Y, Zx_true, levels=10, colors="k", linewidths=0.8)
            if self.X_data is not None:
                axes[0].scatter(self.X_data[:, 0], self.X_data[:, 1],
                                color="white", edgecolor="black", s=20, label="Queries")
                axes[0].legend(loc="upper right")

            # dA/dy
            c2 = axes[1].contourf(X, Y, Zy, levels=100, cmap="plasma")
            axes[1].set_title("Predicted dA/dy (2D)")
            axes[1].set_xlabel("CV1"); axes[1].set_ylabel("CV2")
            cb2 = fig.colorbar(c2, ax=axes[1]); cb2.set_label("dA/dy")
            if callable(true_2d):
                Zx_true, Zy_true = true_2d(X, Y)
                axes[1].contour(X, Y, Zy_true, levels=10, colors="k", linewidths=0.8)
            if self.X_data is not None:
                axes[1].scatter(self.X_data[:, 0], self.X_data[:, 1],
                                color="white", edgecolor="black", s=20, label="Queries")
                axes[1].legend(loc="upper right")

            fig.suptitle("Derivative contours (2D)")
            fig.tight_layout()

            if savepath is not None:
                plt.savefig(savepath, dpi=300, bbox_inches="tight")
            if show:
                plt.show()
            else:
                plt.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_grid(self) -> None:
        """
        Create the grid where we predict gradients / acquisitions, and integrate to get FES.
        """
        if self.dim == 1:
            x_min, x_max = self.bounds_1d
            self.x_grid_1d = np.linspace(x_min, x_max, num=self.config.grid_size_1d)
            self.grid_flat = self.x_grid_1d.reshape(-1, 1)  # (n, 1)
        else:  # dim == 2
            x_min, x_max, y_min, y_max = self.bounds_2d
            nx, ny = self.config.grid_size_2d
            x_grid = np.linspace(x_min, x_max, num=nx)
            y_grid = np.linspace(y_min, y_max, num=ny)
            self.X_grid_2d, self.Y_grid_2d = np.meshgrid(x_grid, y_grid, indexing="ij")
            self.grid_flat = np.vstack(
                [self.X_grid_2d.ravel(), self.Y_grid_2d.ravel()]
            ).T  # (nx * ny, 2)

    def _predict_grad_on_grid(self) -> np.ndarray:
        """
        Predict gradients on the full grid (used for full integration n_integrated == dim).
        """
        if self.grid_flat is None:
            self._build_grid()
        if self.emukit_method is None or not hasattr(self.emukit_method, "predict"):
            raise RuntimeError("Emukit not initialized. Call initialize() first.")

        mean, _ = self.emukit_method.predict(self.grid_flat)  # (N, dim)
        grad_flat = np.asarray(mean)
        if grad_flat.ndim == 1:
            grad_flat = grad_flat.reshape(-1, 1)

        if self.dim == 1:
            return grad_flat.reshape(-1, 1)
        else:
            nx, ny = self.config.grid_size_2d
            return grad_flat.reshape(nx, ny, self.dim)

    def _update_fes(self) -> None:
        """
        Compute current FES from the GP-predicted gradients.

        - If n_integrated == dim:
            * dim=1: integrate over x to get A(x).
            * dim=2: integrate over (x, y) to get A(x, y) on a 2D grid.
        - If dim=2 and n_integrated == 1:
            * integrate over the first CV only (x),
              treating the second CV (y) as an extra variable fixed
              at extra_context (or in the middle of its bounds).
        """
        # Full integration over all dims (original behaviour)
        if self.n_integrated == self.dim:
            grad = self._predict_grad_on_grid()
            if self.dim == 1:
                dA_dx = grad[:, 0]
                self.current_fes_1d = integration_1D(self.x_grid_1d, dA_dx)
            else:
                # Build grid as (Y, X) to match your integration routine
                XY_combined = np.stack((self.Y_grid_2d, self.X_grid_2d), axis=-1)  # (nx, ny, 2): (y, x)
                # Swap derivative components to (dA/dy, dA/dx) to match grid channels
                derivative_xy_combined = grad[:, :, [1, 0]]  # (nx, ny, 2): (dy, dx)

                self.current_fes_2d = integration_2D_rgrid(
                    XY_combined,
                    derivative_xy_combined,
                    integrator="simpson+mini" if self.config.use_mini else "simpson",
                    fast=self.config.fast_mini,
                )
        else:
            # Mixed mode: currently only support dim=2, n_integrated=1 (one integrated, one extra)
            if not (self.dim == 2 and self.n_integrated == 1):
                raise NotImplementedError(
                    f"_update_fes: mixed mode only implemented for dim=2, n_integrated=1; "
                    f"got dim={self.dim}, n_integrated={self.n_integrated}"
                )

            # Integrated variable = first CV (x), extra variable = second CV (y)
            x_min, x_max, y_min, y_max = self.bounds_2d
            if self.x_grid_1d is None:
                self.x_grid_1d = np.linspace(x_min, x_max, num=self.config.grid_size_1d)

            # Choose context value for extra variable (y)
            if self.extra_context is None:
                y_ctx = 0.5 * (y_min + y_max)
            else:
                y_ctx = float(np.atleast_1d(self.extra_context)[0])

            # Build prediction grid for FES slice: (x, y_ctx)
            X_pred_fes = np.column_stack(
                [self.x_grid_1d, y_ctx * np.ones_like(self.x_grid_1d)]
            )
            mean_fes, _ = self.emukit_method.predict(X_pred_fes)
            mean_fes = np.asarray(mean_fes).reshape(-1, self.dim)

            # Integrate derivative w.r.t integrated dim (first component)
            dA_dx = mean_fes[:, 0]
            self.current_fes_1d = integration_1D(self.x_grid_1d, dA_dx)

    def _compute_acquisition_grid(self, weight_var, weight_fes, weight_path, sampling_grid):
        if self.grid_flat is None:
            self._build_grid()
        if self.emukit_method is None:
            raise RuntimeError("Emukit not initialized. Call initialize() first.")
        if self.acq_function == "IVR":
            acquisition = IntegralVarianceReduction(self.emukit_method)
        elif self.acq_function == "MI":
            acquisition = MutualInformation(self.emukit_method)    
        elif self.acq_function == "US":
            acquisition = UncertaintySampling(self.emukit_method)

        else:
            raise RuntimeError("acq_function must be 'IVR', 'US', or 'MI'.")

        acq_flat = np.asarray(acquisition.evaluate(self.grid_flat)).ravel()
        amax = np.max(acq_flat)
        acq_norm_flat = acq_flat / amax if amax > 0 else np.zeros_like(acq_flat)

        # FES normalization
        if weight_fes != 0.0: #We dont want to recompute FES if we are not using it in the acquisition, to save time
            self._update_fes()
        fes = self.current_fes_1d if self.dim == 1 else self.current_fes_2d   
        fmax = np.max(fes)
        fes_norm = fes / fmax if fmax > 0 else np.zeros_like(fes)

        # Reshape + combine
        if self.dim == 1:
            acq_norm = acq_norm_flat
            sampling_term = 0.0 if sampling_grid is None else np.asarray(sampling_grid)
            if sampling_grid is not None and sampling_term.shape != acq_norm.shape:
                raise ValueError(f"sampling_grid shape {sampling_term.shape} must be {acq_norm.shape}")
        else:
            nx, ny = self.config.grid_size_2d
            acq_norm = acq_norm_flat.reshape(nx, ny)
            sampling_term = 0.0 if sampling_grid is None else np.asarray(sampling_grid)
            if sampling_grid is not None and sampling_term.shape != acq_norm.shape:
                raise ValueError(f"sampling_grid shape {sampling_term.shape} must be {(nx, ny)}")

        combined = weight_var * acq_norm - weight_fes * fes_norm + weight_path * sampling_term
        return acq_norm, fes_norm, combined


    def _select_next_point(
        self,
        weight_var: float,
        weight_fes: float,
        weight_path: float,
        sampling_grid: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Select next point by maximizing the combined acquisition.
        """
        _, _, acq = self._compute_acquisition_grid(
            weight_var=weight_var,
            weight_fes=weight_fes,
            weight_path=weight_path,
            sampling_grid=sampling_grid,
        )

        if self.dim == 1:
            idx = int(np.argmax(acq))
            x_next = self.x_grid_1d[idx]
            return np.array([x_next])
        else:
            nx, ny = self.config.grid_size_2d
            max_index = np.unravel_index(int(np.argmax(acq)), (nx, ny))
            i, j = max_index
            new_x = self.X_grid_2d[i, j]
            new_y = self.Y_grid_2d[i, j]
            return np.array([new_x, new_y])