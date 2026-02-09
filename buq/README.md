# `buq` — Bayesian Umbrella Quadrature core library

This package contains the core components of **Bayesian Umbrella Quadrature (BUQ)**:

- Gaussian process–based Bayesian quadrature on gradients (forces),
- integration of gradients into free-energy profiles/surfaces,
- plotting and acquisition utilities.

Higher-level runners and complete examples live in `buq_examples/`.

---

## Modules and main components

- `systems.py`  
  Defines the abstract base class `CollectiveVariableSystem`, which connects BUQ to your MD (or other) simulation code.

- `bq_runner.py`  
  Implements `BQConfig` and `BayesianQuadratureRunner`, which orchestrate:
  1. querying a system for forces,
  2. fitting a GPy+Emukit GP model,
  3. integrating gradients to a free-energy surface (FES),
  4. adaptively selecting new sampling points - different acquisitions are possible

- `integration.py`  
  Utilities to integrate gradients:
  - `integration_1D_trapz(x, dA_dx)`
  - `integration_2D_rgrid(grid, dA_grid, integrator="simpson+mini", fast=False)`
  - `integrate_from_grad(coords, grad, ...)`

- `kernels.py`  
  Wrappers to use GPy (RBF or Matern) kernels in Emukit quadrature:
  - `SumRBFWhiteGPy`
  - `SumMaternWhiteGPy`

- `sample_systems/mock.py`  
  Toy systems for testing and examples:
  - `Mock1DSystem` (analytic test function 1D),
  - `Mock2DSystem` (analytic test function 2D),
  - `AdipepFromGrid` (alanine dipeptide from a precomputed grid, in 2D).

---

## `CollectiveVariableSystem`

`CollectiveVariableSystem` (in `systems.py`) describes your system in terms of collective variables (CVs):

- `dim` ∈ {1, 2} — number of CVs.
- `bounds` — CV domain:
  - 1D: `(x_min, x_max)`
  - 2D: `(x_min, x_max, y_min, y_max)`

Other parameters are recommended, such as the strength of the kappa, equilibration time, etc. 
Subclasses must implement:

- `write_plumed_input(x)` – prepare input for an umbrella simulation at CV point `x`.
- `run_simulation(x)` – run the corresponding biased simulation - possible to connect to Gromacs, LAMMPS, OpenMM, Ace....
- `get_force(x)` – return the mean force / gradient at `x` as an array of shape `(dim,)`.

They may optionally implement:

- `true_fes(x)` – reference free energy (for validation/plots).
- `true_grad(x)` / `true_force(x)` – reference gradient.

---

## `BQConfig` and `BayesianQuadratureRunner`

Both are defined in `bq_runner.py`:

- `BQConfig` – holds configuration for the BUQ run, including:

  - GP kernel:
    - `kernel_type`: `"RBF"`, `"Matern52"`, `"Matern32"`, `"Matern12"`;
    - `lengthscale`, `variance`, `noise`.
  - Acquisition:
    - `acq_function`: `"IVR"`, `"US"`, or `"MI"`.
  - Integration grids:
    - `grid_size_1d` for 1D,
    - `grid_size_2d = (nx, ny)` for 2D.
  - 2D integration refinement:
    - `use_mini`, `fast_mini`.
  - Hyperparameter optimization:
    - `optimize_hyperparams` (use GPy optimizer or not).

- `BayesianQuadratureRunner` – high-level driver that:

  1. queries a `CollectiveVariableSystem` for forces,
  2. fits a GPy+Emukit GP model to gradient data,
  3. builds a grid and integrates gradients to obtain the FES,
  4. adaptively chooses new sampling points via a variance-based acquisition.

A typical construction looks like:

    from buq.bq_runner import BQConfig, BayesianQuadratureRunner

    system = ...  # a CollectiveVariableSystem
    config = BQConfig(...)

    runner = BayesianQuadratureRunner(
        system=system,
        config=config,
        bounds=None,  # optional; default: system.bounds
    )

Key methods:

- `initialize(initial_points)`  
  Run simulations at initial CV locations, fit the initial GP, build the integration grid, and compute an initial FES.

- `run(n_queries=None, ...)`  
  Perform multiple adaptive queries (using the chosen acquisition function), updating the GP and FES.

- `run_one_query(...)`  
  Single adaptive step, useful for custom loops.

- Plotting helpers:
  - `plot_fes(...)` – current free energy (1D or 2D).
  - `plot_acq(...)` – acquisition components.
  - `plot_derivatives(...)` – predicted gradients .

---

## Integration helpers

Defined in `integration.py`:

- `integration_1D_trapz(x, dA_dx)`  
  1D cumulative trapezoidal integration, returns `A(x)` with `min(A) = 0`.

- `integration_2D_rgrid(grid, dA_grid, integrator="simpson+mini", fast=False)`  
  2D integration on a regular grid, with optional real-space minimization to enforce gradient consistency.

- `integrate_from_grad(coords, grad, ...)`  
  Dispatches to the 1D or 2D integrator depending on the shape of `coords`.

---

## Kernel wrappers

Defined in `kernels.py`:

- `SumRBFWhiteGPy`  
  Wraps a GPy kernel of the form `RBF + White` for use with Emukit quadrature.

- `SumMaternWhiteGPy`  
  Wraps Matérn-type kernels plus white noise (`Exponential`, `Matern32`, `Matern52` + `White`).

Both implement Emukit’s kernel interface, providing `K`, `dK_dx1`, and access to `lengthscales` and `variance`.

---

## Sample systems

Defined in `sample_systems/mock.py`:

- `Mock1DSystem`  
  Analytic 1D system:
  - `A(x) = 0.5 * x^2`
  - `dA/dx = x`  
  Provides `get_force`, `true_fes`, and `true_grad`.

- `Mock2DSystem`  
  Analytic 2D system:
  - `A(x, y) = 0.5 * x^2 + 0.33 * y^3` (up to a constant)
  - `grad A = [x, y^2]`  
  Provides `get_force`.

- `AdipepFromGrid`  
  2D alanine dipeptide system (φ, ψ), reading gradients from a regular `fes.dat` grid (metadynamics output with columns: φ, ψ, FES, dA/dφ, dA/dψ).  
  Builds interpolators for the gradients and exposes them via `get_force`.

Higher-level scripts that actually run BUQ on these systems are in `buq_examples/`.

---

## Minimal 1D example

For full, ready-to-run scripts, see `buq_examples/`.  
A minimal pattern using `Mock1DSystem` looks like:

    import numpy as np
    from buq.bq_runner import BQConfig, BayesianQuadratureRunner
    from buq.sample_systems.mock import Mock1DSystem

    # 1. System
    system = Mock1DSystem()

    # 2. Configuration
    config = BQConfig(
        kernel_type="RBF",
        lengthscale=0.3,
        noise=1e-4,
        variance=1.0,
        n_queries=20,
        grid_size_1d=200,
        optimize_hyperparams=False,
        acq_function="IVR",
    )

    # 3. Runner
    runner = BayesianQuadratureRunner(system=system, config=config)

    # 4. Initial design
    x_init = np.linspace(-2.0, 2.0, 5).reshape(-1, 1)
    runner.initialize(x_init)

    # 5. Adaptive BUQ steps
    runner.run()

    # 6. Plot free energy and compare to analytic reference
    runner.plot_fes(true_fes_func=system.true_fes)
