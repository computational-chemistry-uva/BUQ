import numpy as np
from buq import BQConfig, BayesianQuadratureRunner
from buq.sample_systems import AdipepFromGrid


system = AdipepFromGrid()
bounds = (-np.pi, np.pi, -np.pi, np.pi)

config = BQConfig(
    kernel_type="Matern32",
    lengthscale=np.array([0.6]),  # ARD lengthscales for 2D
    noise=1e-6,
    variance=1.0,
    n_queries=3,
    grid_size_2d=(40, 40),
    use_mini=True,        # L-BFGS-B refinement in 2D integration
    fast_mini=True,       # keeps minimization fast
    acq_function="US",   # or "US"
)

initial_points = np.array([[-1.5, -1.5], [0.0, 0.0], [1.5, 1.5]])

runner = BayesianQuadratureRunner(system, bounds, config)
runner.initialize(initial_points)
print("Initial FES shape (nx, ny):", runner.current_fes_2d.shape)

# Adaptive steps using Emukit IVR
runner.run(n_queries=100, weight_var=1.0, weight_fes=0.0, weight_path=0.0)
print("Final #points:", runner.X_data.shape[0])

# Plots
runner.plot_fes(show=True)
runner.plot_ivr(show=True, full=True)
runner.plot_derivatives(show=True)


print(runner.current_fes_2d)
