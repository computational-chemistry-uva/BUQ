Bayesian Umbrella Quadrature (BUQ) developed by Eline Kempkes and Alberto Pérez de Alba Ortíz 
----------------------------------------------------------------------------------------------



This repository contains code and data for **Bayesian Umbrella Quadrature (BUQ)**, a method to reconstruct free energy landscapes from biased molecular dynamics simulations using Bayesian quadrature.

The repository is organized into two main parts:

- **`buq/`** – the reusable Python package implementing BUQ (collective‑variable interface, Gaussian process models, Bayesian quadrature driver, and utilities).
- **`BUQ_paper/`** – the case studies and scripts used to generate the results in the accompanying paper.
- **`buq_examples/`** – some examples on how to use buq

All dependencies (Conda + pip) are specified in `environment_buq.yml`.

---

## 1. Repository structure

```text
.
├── buq/                         # Python package: generic BUQ implementation
│   ├── __init__.py
│   ├── systems.py               # Abstract interface for MD systems (CVs, forces, bias)
│   ├── kernels.py               # GP kernel wrappers used in BUQ
│   ├── integration.py           # Integrating gradients → free energy
│   ├── bq_runner.py             # High-level Bayesian quadrature driver
│   ├── sample_systems/
│   │   ├── __init__.py
│   │   └── mock.py              # Minimal / toy system example
│   └── README.md                # Package-level documentation
│
├── buq_examples/                # Example scripts & small applications of BUQ
│   ├── example_adipep_2d/
│   │   ├── simulations_essentials/  # MD input / essentials for alanine dipeptide example
│   │   ├── adipep_2d.py             # System definition & BUQ setup for 2D alanine dipeptide
│   │   └── run_adipep.py            # Driver script to run BUQ on the example
│   ├── example_adipep_from_grid_2d.py  # BUQ using precomputed grid data for alanine dipeptide
│   ├── example_runner_1d.py           # 1D illustrative BUQ example
│   └── example_runner_2d.py           # 2D illustrative BUQ example
│
├── BUQ_paper/                   # Full reproducibility package for the BUQ paper
│   └── ...                      # See BUQ_paper/README.md for detailed instructions
│
├── environment_buq.yml          # Conda environment specification (all dependencies)
├── pyproject.toml               # Build and packaging configuration
└── README.md                    # This file