# BUQ Tutorials

These tutorials introduce **Bayesian Optimization (BO)** and **Bayesian Quadrature (BUQ)** for free energy estimation in molecular dynamics, using alanine dipeptide as a running example.

## Overview

| Notebook | Topic | Key concepts |
|----------|-------|--------------|
| `01_bayesopt_adipep_phi.py` | Bayesian Optimization | GP surrogate, acquisition functions (EI/PI/LCB), exploration vs exploitation |
| `02_buq_adipep_phi.py` | Bayesian Quadrature | IVR acquisition, kernel choice, FES reconstruction with uncertainty |

The two notebooks are designed to be run in order. Notebook 1 finds *where* the free energy minimum is; notebook 2 reconstructs the *full* free energy surface.

## Setup

### 1. Create the conda environment

From the **repo root**:

```bash
conda env create -f environment_buq.yml
conda activate buq_env
```

### 2. Register the kernel in VS Code

```bash
python -m ipykernel install --user --name buq_env --display-name "buq_env"
```

Then open VS Code, select **buq_env** as the kernel (bottom-right corner), and you're ready to go.

### 3. Data files

The tutorials expect the following file in the `tutorials/` directory:

```
tutorials/
├── fes_adipep_phi.dat       ← precomputed metadynamics FES (provided)
├── 01_bayesopt_adipep_phi.py
└── 02_buq_adipep_phi.py
```

## Running the notebooks

Open either `.py` file in VS Code. Each `# %%` block is an interactive cell — click **Run Cell** to execute it, or use `Shift+Enter`.

> **Tip:** Run cells top to bottom on your first pass. Later sections have `# 🔧 TODO` markers where you are encouraged to change settings and re-run.

## What you will learn

### Notebook 1 — Bayesian Optimization
- How a **Gaussian Process** models an unknown function from sparse observations
- How **acquisition functions** (EI, PI, LCB) guide the search for the force minimum
- The trade-off between **exploration and exploitation**
- How the GP posterior over the force connects to the free energy surface

### Notebook 2 — Bayesian Quadrature
- How BUQ integrates the mean force to reconstruct **F(φ) with uncertainty bands**
- How the **IVR acquisition function** reduces integral variance
- How **kernel choice and lengthscale** affect the GP over the force
- Why BUQ samples differently from BO — and when each approach is preferable

## Background

Both methods use a **Gaussian Process** as a surrogate for the mean force f(φ) = dA/dφ. The key difference is the objective:

- **BO** minimizes |f(φ)| to find zero crossings (= FES minima/maxima)
- **BUQ** minimizes the variance of ∫f(φ)dφ to reconstruct the full FES

For more details, see the main `BUQ` package documentation.

