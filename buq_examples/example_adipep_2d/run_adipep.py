import numpy as np
import matplotlib.pyplot as plt
from buq import BQConfig, BayesianQuadratureRunner
from adipep_2d import Adipep

system = Adipep(
    kappa_phi=200.0,
    kappa_psi=200.0,
    measure_after_ps=10.0,
    nsteps=50000,
)

config = BQConfig(
    kernel_type="Matern32",
    lengthscale=np.array([0.6]),
    noise=1e-6,
    variance=1.0,
    n_queries=3,
    grid_size_2d=(100, 100),
    use_mini=True,
    fast_mini=True,
    acq_function="IVR",
)

initial_points = np.array([[-1.5, -1.5], [0.0, 0.0], [1.5, 1.5]])

runner = BayesianQuadratureRunner(system, config)
runner.initialize(initial_points)
initial_queries = 100

runner.run(n_queries=initial_queries, weight_var=1.0, weight_fes=0.0, weight_path=0.0)

data = np.genfromtxt("simulations_essentials/fes.dat")
phi = data[:, 0].reshape(100, 100) #then reshape to 100x100 and transpose to get correct orientation for contourf
psi = data[:, 1].reshape(100, 100)
ground_truth_fes = data[:, 2].reshape(100, 100).T

# RMSD loop
rmsd = []
vmin, vmax = ground_truth_fes.min(), ground_truth_fes.max()
diff_min, diff_max = -2.5, 20

for i in range(initial_queries + 1, initial_queries + 11):
    runner.run_one_query(weight_var=1.0, weight_fes=0.0)
    bq_fes = runner.current_fes_2d

    rmsd_query = np.sqrt(np.mean((ground_truth_fes - bq_fes) ** 2))
    rmsd.append(rmsd_query)
    print(f"RMSD after {i} queries: {rmsd_query:.4f} kcal/mol")

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    # First plot: Ground Truth
    contour1 = axes[0, 0].contourf(psi, phi, ground_truth_fes, levels=100,
                                cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0, 0].set_title("Ground Truth Free Energy", fontsize=16)

    # Second plot: Prediction BQ
    contour2 = axes[0, 1].contourf(psi, phi, bq_fes, levels=100,
                                cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0, 1].set_title("Prediction using Bayesian Quadrature", fontsize=16)
    axes[0, 1].scatter(runner.X_data[:, 0], runner.X_data[:, 1], color="white", s=15)

    # Shared colorbar for first two
    cbar1 = fig.colorbar(contour1, ax=axes[0, :], shrink=0.8, location="right")
    cbar1.set_label("Free Energy (kcal/mol)", fontsize=14)
    cbar1.ax.tick_params(labelsize=12)

    diff = ground_truth_fes - bq_fes
    # Third plot: Difference
    contour3 = axes[1, 0].contourf(psi, phi, diff, levels=100,
                                cmap="coolwarm", vmin=diff_min, vmax=diff_max)
    axes[1, 0].set_title("Difference", fontsize=16)
    contour3.set_clim(diff_min, diff_max)

    cbar2 = fig.colorbar(contour3, ax=axes[1, 0], shrink=0.8, location="right")
    cbar2.set_label("Difference (kcal/mol)", fontsize=14)
    cbar2.set_ticks([-2.5, 0, 5, 10, 15, 20])
    cbar2.ax.tick_params(labelsize=12)
    # Fourth plot: RMSD 
    axes[1, 1].plot(np.arange(1, len(rmsd) + 1) + initial_queries, rmsd, marker="o", linestyle="-")
    axes[1, 1].set_title("RMSD", fontsize=16)
    axes[1, 1].set_xlabel("Query", fontsize=14)
    axes[1, 1].set_ylabel("RMSD (kcal/mol)", fontsize=14)
    axes[1, 1].tick_params(axis="both", labelsize=12)

    # Make first three square
    for ax in [axes[0, 0], axes[0, 1], axes[1, 0]]:
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel(r"$\phi$", fontsize=14)
        ax.set_ylabel(r"$\psi$", fontsize=14)
        ax.tick_params(axis="both", labelsize=12)

    # Title
    fig.suptitle(f"After {i} Queries", fontsize=18, fontweight="bold")

    plt.savefig(f"plots/fes_after_{i}.png", dpi=150)
    plt.show()
    
