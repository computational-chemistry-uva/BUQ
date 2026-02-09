import numpy as np
from buq import BQConfig, BayesianQuadratureRunner
from buq.sample_systems import Mock1DSystem

system = Mock1DSystem()
config = BQConfig(
    kernel_type="Matern32",
    lengthscale=0.9,
    noise=1e-6,
    acq_function="IVR",
    n_queries=3,
    grid_size_1d=200,
    
    use_mini=False,  # only relevant for 2D
)

initial_points = np.array([[-1.5], [0.0], [1.5]])

runner = BayesianQuadratureRunner(system, config)
runner.initialize(initial_points)
print("Initial FES shape:", runner.current_fes_1d.shape)



for i in range(5):
    runner.run_one_query(weight_fes=0.5, weight_var=0.5)
    runner.plot_fes(show=True)
    runner.plot_acq(show=True, weight_fes=0.5, weight_var=0.5)
    runner.plot_derivatives(show=True)

