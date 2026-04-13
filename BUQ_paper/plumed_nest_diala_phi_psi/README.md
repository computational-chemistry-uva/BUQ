# PLUMED input for alanine dipeptide with restraints on their trigynomic functions

This folder contains the PLUMED input and minimal scripts used in the
Bayesian Umbrella Quadrature (BUQ) study for alanine dipeptide (diala).

We perform restrained MD simulations of alanine dipeptide (diala) where the
backbone torsions φ and ψ are biased. To handle the **periodicity** of
these dihedral angles in a smooth way, we do not restrain φ and ψ
directly. Instead, we bias their trigonometric projections:

- `sin_phi`, `cos_phi` for φ  
- `sin_psi`, `cos_psi` for ψ  

These variables live on the unit circle and are continuous even when the
angle crosses the −π/π boundary, which makes the moving restraints and
subsequent force estimation numerically more robust.

The resulting COLVAR data are then post‑processed (e.g. with BUQ) to reconstruct the free energy landscape.

This directory is intended as a self‑contained PLUMED‑NEST “egg” for the PLUMED part of the work. The full BUQ implementation and additional analysis code are in the main BUQ repository:

- https://github.com/computational-chemistry-uva/BUQ

---

## Contents

Top‑level files:

- `plumed.dat`  
  Representative PLUMED input file for a dialanine φ/ψ umbrella window.
  This is the file used for the PLUMED‑NEST `plumed driver --parse-only`
  compatibility test.

- `diala.pdb`  
  Structure file used in `MOLINFO STRUCTURE=diala.pdb`.

- `diala.gro`, `md.tpr`, `run.mdp`  
  Example GROMACS input files for running a restrained MD simulation
  with PLUMED (`md.tpr` is the pre‑compiled GROMACS run input).

- `scripts/`  
  - `write_plumed_files.py`  
    Contains minimal helper functions:
    - `write_plumed_file(phi, psi, ...)`: generate a PLUMED input file
      with moving restraints for a target `(φ, ψ)` (radians).
    - `get_force(phi_value, psi_value, ...)`: read a COLVAR file and
      compute the mean forces `[dF/dφ, dF/dψ]` from the deviations in
      `sin/cos` between the umbrella target and the sampled values.

- `example/`  
  A small working example for one window:
  - `plumed_-1.508_1.194.dat`  
    A PLUMED file generated for `φ = -1.508`, `ψ = 1.194` (radians).
  - `diala.pdb`  
    Local copy of the structure for convenience.
  - `Colvars/COLVAR_-1.508_1.194`  
    Example COLVAR file produced by a restrained MD run with that PLUMED
    input.
  - `run_-1508_1194.py`  
    Example Python script that runs a single restrained MD simulation
    with GROMACS using `plumed_-1.508_1.194.dat` and writes the COLVAR
    file.

---

## Requirements

- **PLUMED**: 2.9.0 

- **MD engine**: GROMACS 2023.2
 
- **Python**: 3.9+ with `numpy`  
  (for `scripts/write_plumed_files.py` and the example run/analysis)

---

## Testing the PLUMED input (PLUMED‑NEST check)

From inside this directory (`plumed_nest_diala_phi_psi`):

```
plumed driver --natoms 100000 --parse-only --kt 2.49 --plumed plumed.dat
```

Generating a PLUMED file for a given (φ, ψ). The function write_plumed_file in scripts/write_plumed_files.py creates a PLUMED input with moving restraints on sin/cos of φ and ψ.

## Example usage:
```
cd scripts
python -c "from write_plumed_files import write_plumed_file; write_plumed_file(-1.508, 1.194)"
```

This will generate the file ../Colvars/plumed_-1.508_1.194.dat with blocks of the form:
```
restraint_phi_cos, restraint_phi_sin, restraint_psi_cos, restraint_psi_sin
```

Each MOVINGRESTRAINT ramps the restraint center and force constant in time from an initial value (current φ/ψ) to a target value (desired φ/ψ), using the parameters, which of course, can be changed:
```
equisteps = 500
build_up_kappa_steps = 1500 (500 + 1000)
moving_speed = 1000 steps per radian
```

The resulting simulation writes COLVAR data for:

```
sin_phi, cos_phi, sin_psi, cos_psi to Colvars/COLVAR_-1.508_1.194
```
This COLVAR can be processed using:

```
cd scripts
python -c "from write_plumed_files import get_force; import numpy as np; print(get_force(-1.508, 1.194))"
```
This is also worked out in the provided example.

# Relation to the BUQ paper
These PLUMED inputs and example simulations correspond to the alanine dipeptide case study in the BUQ paper:

Title: Bayesian umbrella quadrature accelerates free-energy calculations across diverse molecular systems and processes

Authors: Eline K. Kempkes, Alberto Pérez de Alba Ortíz

DOI: https://doi.org/10.48550/arXiv.2601.08783

Please cite this paper if you use this setup.

For details of the BUQ method and the free‑energy reconstruction based on these forces, see:

https://github.com/computational-chemistry-uva/BUQ