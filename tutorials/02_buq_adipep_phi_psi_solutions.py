# %% [markdown]
# # Notebook 2 SOLUTIONS: Bayesian Quadrature on Alanine Dipeptide
#
# In the previous notebook we used **Bayesian Optimization (BO)** to find
# where the force is zero  i.e. the minima/maxima of the free energy surface.
#
# In this notebook we use **Bayesian Quadrature (BUQ)** to do something more
# ambitious: reconstruct the **entire free energy surface** A(phi, psi) by integrating
# the mean force over both dihedral angles.
#
# ### What is the difference?
#
# | | Bayesian Optimization | Bayesian Quadrature |
# |---|---|---|
# | **Goal** | Find minimum of f(x) | Compute integral of f(x) |
# | **Surrogate** | GP on \|force(phi)\| | GP on force(phi, psi) directly |
# | **Acquisition** | Reduce uncertainty near minimum | Reduce uncertainty of integral |
# | **Output** | Best (phi, psi) found | F(phi, psi) + error bars |
#
# ### What you will learn
# - How BUQ places new simulation points to reduce **integral uncertainty** in 2D
# - How kernel choice and hyperparameters affect the GP over the force
# - How the IVR acquisition function differs from EI/LCB
# - How sampling patterns differ between BO (notebook 1) and BUQ (this notebook)
#
# For more information, check out our paper: https://arxiv.org/abs/2601.08783
#
#
# > **Note:** We use `AdipepFromGrid`, which wraps the same metadynamics data
# > as notebook 1 but exposes both dihedral angles (phi, psi), so you can directly
# > compare sampling strategies.

# %% [markdown]
# ## 0. Imports

# %%
import numpy as np
import matplotlib.pyplot as plt

from buq import BQConfig, BayesianQuadratureRunner
from buq.sample_systems import AdipepFromGrid

# %% [markdown]
# ## 1. The system: alanine dipeptide (phi, psi)
#
# `AdipepFromGrid` wraps the precomputed metadynamics data.
# It exposes the mean force f(phi, psi) = dA/d(phi, psi) via `system.get_force(x)`.
#
# Let's first look at what the system gives us.

# %%
system = AdipepFromGrid()

# Evaluate FES on a dense grid for reference
phi_grid = np.linspace(system.bounds[0][0], system.bounds[0][1], 100)
psi_grid = np.linspace(system.bounds[1][0], system.bounds[1][1], 100)
PHI, PSI = np.meshgrid(phi_grid, psi_grid)

fes_ref = system.true_fes(PHI, PSI)
fes_ref -= np.min(fes_ref)

fig, ax = plt.subplots(figsize=(6, 5))
cf = ax.contourf(PHI, PSI, fes_ref, levels=20, cmap="viridis")
plt.colorbar(cf, ax=ax, label="A(phi, psi) (kJ/mol)")
ax.set_xlabel("phi (rad)")
ax.set_ylabel("psi (rad)")
ax.set_title("Reference Free Energy Surface")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 2. Configure BUQ
#
# `BQConfig` controls the GP kernel and the BUQ loop.
#
# Key parameters:
# - `kernel_type`: shape of the GP covariance -- `"Matern12"`, `"Matern32"`, `"Matern52"`, `"RBF"`
# - `lengthscale`: ARD lengthscales for each dimension (phi and psi), controls how quickly the GP varies
# - `noise`: assumed observation noise (white kernel)
# - `acq_function`: `"IVR"` (Integral Variance Reduction), `"US"` (Uncertainty Sampling), `"MI"` (Mutual Information)
# - `grid_size_2d`: resolution of the internal (phi, psi) grid -- in 2D this matters more, since a high resolution grid significantly slows down integral computation
# - `use_mini` / `fast_mini`: enable L-BFGS-B refinement for the 2D integration step
#
# ### TODO: try changing these settings and see what happens!
# - `kernel_type`
# - `lengthscale`
# - `noise`

# %%
# --- Settings: change these! ---
kernel_type = "Matern32"
lengthscale = np.array([0.6, 0.6])   # one per dimension (ARD)
noise       = 1e-6
# --------------------------------

config = BQConfig(
    kernel_type=kernel_type,
    lengthscale=lengthscale,
    noise=noise,
    variance=1.0,
    acq_function="IVR",
    grid_size_2d=(40, 40),
    use_mini=True,
    fast_mini=True,
)

# %% [markdown]
# ## 3. Initialize the runner
#
# We start with a small set of initial points where the force has already been
# evaluated. The runner fits an initial GP to these observations.
#
# ### TODO: try different initial point placements
# - What happens if all initial points are clustered in one region?

# %%
# --- Settings: change these! ---
initial_points = np.array([[-1.5, -1.5], [0.0, 0.0], [1.5, 1.5]])
# --------------------------------

runner = BayesianQuadratureRunner(system, config)
runner.initialize(initial_points)

print("Initial points:\n", initial_points)
print("Initial FES shape (nx, ny):", runner.current_fes_2d.shape)

# Plot initial state
runner.plot_fes(show=True)
runner.plot_acq(show=True, full=True)
runner.plot_derivatives(show=True)

# %% [markdown]
# **Questions:**
# - Where does the acquisition function suggest sampling next? Why?
# - How does the initial FES estimate compare to the reference?

# %% [markdown]
# ## 4. The BUQ loop
#
# At each step the runner:
# 1. Evaluates the IVR acquisition function on the (phi, psi) grid
# 2. Picks the (phi, psi) that maximally reduces integral variance
# 3. Queries the force at that point (`system.get_force`)
# 4. Updates the GP posterior and recomputes the 2D FES
#
# The acquisition is a weighted combination:
#
# $$\text{acq}(\phi, \psi) = (1 - w_\text{fes}) \cdot \text{IVR}(\phi, \psi) - w_\text{fes} \cdot \hat{F}(\phi, \psi)$$
#
# - `weight_var=1, weight_fes=0`: pure IVR -- reduces integral variance uniformly
# - `weight_fes > 0`: biases sampling toward high free energy regions (exploitation)
#
# ### TODO: try changing the weights and number of steps

# %%
# --- Settings: change these! ---
n_steps    = 5
weight_fes = 0.0
# --------------------------------

runner = BayesianQuadratureRunner(system, config)
runner.initialize(initial_points)

for i in range(n_steps):
    print(f"\n--- Step {i+1}/{n_steps} ---")
    runner.run_one_query(weight_fes=weight_fes)
    runner.plot_fes(show=True)
    runner.plot_acq(show=True, full=True)

print("Final FES shape:", runner.current_fes_2d.shape)

# %% [markdown]
# **Questions:**
# - How do the sampling locations compare to notebook 1 (BO)?
# - Does BUQ sample near the force zero crossings, or elsewhere?
# - What changes when you increase `weight_fes`?

# %% [markdown]
# ## 5. Effect of kernel and lengthscale
#
# The kernel controls the GP's assumptions about the smoothness of the force.
#
# - **Small lengthscale**: GP varies quickly -- can fit sharp features, but needs more points to cover the domain
# - **Large lengthscale**: GP varies slowly -- smooth FES, may miss sharp features
# - **RBF**: infinitely smooth (strong assumption)
# - **Matern12/32/52**: less smooth, more realistic for MD forces, where Matern12 is really for sharp, rugged derivatives, while Matern52 is smoother
#
# In 2D you can also use **anisotropic (ARD) lengthscales**, e.g. `[0.6, 1.2]`,
# if the force varies at different rates along phi and psi.
#
# ### TODO: run each cell and compare the FES reconstructions

# %%
# Helper -- no need to change this
def run_buq(kernel_type, lengthscale, noise=1e-6, n_steps=8,
            weight_var=1.0, weight_fes=0.0, title=""):
    """Run a full BUQ loop and return the runner."""
    cfg = BQConfig(
        kernel_type=kernel_type,
        lengthscale=np.atleast_1d(lengthscale),
        noise=noise,
        variance=1.0,
        acq_function="IVR",
        grid_size_2d=(40, 40),
        use_mini=True,
        fast_mini=True,
    )
    r = BayesianQuadratureRunner(system, cfg)
    r.initialize(np.array([[-1.5, -1.5], [0.0, 0.0], [1.5, 1.5]]))
    for _ in range(n_steps):
        r.run_one_query(weight_var=weight_var, weight_fes=weight_fes)
    print(f"\n{title}")
    r.plot_derivatives(show=True)
    r.plot_fes(show=True)
    return r

# %%
r_short   = run_buq("Matern32", lengthscale=[0.1, 0.1], title="Matern32, lengthscale=0.1")

# %%
r_default = run_buq("Matern32", lengthscale=[0.6, 0.6], title="Matern32, lengthscale=0.6")

# %%
r_long    = run_buq("Matern32", lengthscale=[2.0, 2.0], title="Matern32, lengthscale=2.0")

# %%
r_ard     = run_buq("Matern32", lengthscale=[0.6, 1.2], title="Matern32, ARD lengthscale=[0.6, 1.2]")

# %%
r_rbf     = run_buq("RBF",      lengthscale=[0.6, 0.6], title="RBF, lengthscale=0.6")

# %% [markdown]
# **Questions:**
# - Which kernel/lengthscale gives the best FES reconstruction?
# - With a short lengthscale: does the GP overfit the initial points?
# - With a long lengthscale: does the FES miss any features?
# - Does using ARD lengthscales improve the reconstruction?
# - Why might RBF be a poor choice for MD force data?

# %% [markdown]
# ## 6. Acquisition functions: IVR vs US vs MI
#
# So far we used **IVR** (Integral Variance Reduction), which directly minimizes
# the variance of the integral estimate.
#
# Two alternatives are available:
# - **US** (Uncertainty Sampling): samples where GP variance is highest
# - **MI** (Mutual Information): samples where information gain is highest
#
# IVR is the most principled choice for BUQ, but the others can be useful
# for comparison.
#
# ### TODO: run each cell and compare sampling patterns

# %%
r_ivr = run_buq("Matern32", lengthscale=[0.6, 0.6], title="IVR")

# %%
# Note: to use a different acquisition, change acq_function in BQConfig inside run_buq,
# or copy the helper and modify it:

def run_buq_acq(acq_function, title=""):
    cfg = BQConfig(
        kernel_type="Matern32",
        lengthscale=np.array([0.6, 0.6]),
        noise=1e-6,
        variance=1.0,
        acq_function=acq_function,
        grid_size_2d=(40, 40),
        use_mini=True,
        fast_mini=True,
    )
    r = BayesianQuadratureRunner(system, cfg)
    r.initialize(np.array([[-1.5, -1.5], [0.0, 0.0], [1.5, 1.5]]))
    for _ in range(8):
        r.run_one_query(weight_var=1.0, weight_fes=0.0)
    print(f"\n{title}")
    r.plot_fes(show=True)
    r.plot_acq(show=True, full=True)
    return r

# %%
r_us = run_buq_acq("US", title="Uncertainty Sampling")

# %%
r_mi = run_buq_acq("MI", title="Mutual Information")

# %% [markdown]
# **Questions:**
# - Do US and MI sample in different locations than IVR?
# - Which gives the best FES reconstruction after 8 queries?
# - Why is IVR the most natural choice for free energy estimation?

# %% [markdown]
# ## 7. BUQ vs BO: where do they sample?
#
# Let's directly compare the sampling locations:
# - **BO** (notebook 1): minimizes |force| -- clusters near zero crossings
# - **BUQ** (this notebook): minimizes integral variance -- spreads across the (phi, psi) domain
#
# The cell below runs a clean BUQ loop and plots the query locations on the FES.

# %%
cfg = BQConfig(
    kernel_type="Matern32",
    lengthscale=np.array([0.6, 0.6]),
    noise=1e-6,
    variance=1.0,
    acq_function="US",
    grid_size_2d=(40, 40),
    use_mini=True,
    fast_mini=True,
)
runner_compare = BayesianQuadratureRunner(system, cfg)
runner_compare.initialize(np.array([[-1.5, -1.5], [0.0, 0.0], [1.5, 1.5]]))

for _ in range(10):
    runner_compare.run_one_query(weight_var=1.0, weight_fes=0.0)

buq_queries = runner_compare.X_data  # shape (N, 2)

fig, ax = plt.subplots(figsize=(6, 5))
cf = ax.contourf(PHI, PSI, fes_ref, levels=20, cmap="viridis", alpha=0.8)
plt.colorbar(cf, ax=ax, label="A(phi, psi) (kJ/mol)")
ax.scatter(buq_queries[:, 0], buq_queries[:, 1],
           marker="x", s=80, color="white", linewidths=1.5,
           label="BUQ queries", zorder=5)
ax.set_xlabel("phi (rad)")
ax.set_ylabel("psi (rad)")
ax.set_title("BUQ query locations vs FES")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# **Key insight:**
# - BO clusters queries near the **minimum** of |force| (zero crossing = FES minimum)
# - BUQ spreads queries to reduce **integral uncertainty** across the whole (phi, psi) domain
#
# Neither is universally better -- it depends on your goal:
# - Want to find the **minimum quickly**? Use BO.
# - Want to reconstruct the **full FES accurately**? Use BUQ.

# %% [markdown]
# ## 8. Bonus: convergence of the barrier height
#
# As BUQ adds more points, the FES estimate should converge to the reference.
# Let's track the **barrier height** (max of FES) as a function of BUQ iteration.

# %%
cfg = BQConfig(
    kernel_type="Matern32",
    lengthscale=np.array([0.6, 0.6]),
    noise=1e-6,
    variance=1.0,
    acq_function="IVR",
    grid_size_2d=(40, 40),
    use_mini=True,
    fast_mini=True,
)
runner_conv = BayesianQuadratureRunner(system, cfg)
runner_conv.initialize(np.array([[-1.5, -1.5], [0.0, 0.0], [1.5, 1.5]]))

barrier_estimates = []
n_conv_steps = 15

for i in range(n_conv_steps):
    runner_conv.run_one_query(weight_var=1.0, weight_fes=0.0)
    fes_now = runner_conv.current_fes_2d.copy()
    fes_now -= np.min(fes_now)
    barrier_estimates.append(np.max(fes_now))

barrier_ref = np.max(fes_ref)

plt.figure(figsize=(7, 4))
plt.plot(range(1, n_conv_steps + 1), barrier_estimates, "o-",
         color="steelblue", label="BUQ estimate")
plt.axhline(barrier_ref, color="orange", linestyle="--",
            label="reference (metadynamics)")
plt.xlabel("BUQ iteration")
plt.ylabel("Barrier height (kJ/mol)")
plt.title("Convergence of FES barrier height")
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# **Questions:**
# - How many BUQ iterations does it take to converge to the reference barrier height?
# - Does changing the lengthscale affect convergence speed?
# - How would this compare to random sampling (no acquisition function)?

# %% [markdown]
# ## Summary
#
# | | Notebook 1: BO | Notebook 2: BUQ |
# |---|---|---|
# | **Objective** | Find zero of force | Reconstruct full FES |
# | **GP models** | \|force(phi, psi)\| | force(phi, psi) directly |
# | **Acquisition** | EI / PI / LCB | IVR / US / MI |
# | **Sampling** | Clusters near minimum | Spreads across domain |
# | **Output** | Best (phi, psi) found | F(phi, psi) with error bars |
#
# BUQ is the principled approach when you need the **full free energy surface**
# and want to know **how uncertain** your estimate is -- which is exactly what
# matters in enhanced sampling MD.
