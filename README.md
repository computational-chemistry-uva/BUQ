Bayesian Umbrella Quadrature (BUQ) developed by Eline Kempkes and Alberto Pérez de Alba Ortíz 
----------------------------------------------------------------------------------------------



Overview

This repository contains the full set of scripts, input files, and results used to produce the Bayesian Umbrella Quadrature (BUQ) results in the accompanying paper. Three molecular systems are included: alanine dipeptide in vacuo, the water–ice phase transition, and the SN2 reaction. Each system directory contains (i) all simulation input files required to reproduce the biased MD or metadynamics runs, (ii) the BUQ implentation of that system, and (iii) the final data and outputs used in the paper.

⸻

**1. Alanine Dipeptide in Vacuo
**

_Bayesian Quadrature
_
basyian_quadrature/ contains the BUQ implementation for alanine dipeptide.
	•	bq_adipep.py
Main script for running BUQ.
	•	simulation_essentials/
Full GROMACS input set used for generating MD data.
	•	final_result_0.2_ns_Matern52_ls_0.75_w_0.1_n0.0_acq_IVR_q_100/
Output directory for the run used in the paper.
Kernel: Matern-5/2, lengthscale 0.75,exploitation weight 0.1, 0.0 noise, IVR acquisition, 100 queries.
Key files:
	•	AD_0.2_ns_Matern52_ls_0.75_w_0.1_n0.0_acq_IVR_q_100_adipep.txt
RMSD with respect to the FES after every BUQ query.
	•	AD_0.2_ns_Matern52_ls_0.75_w_0.1_n0.0_acq_IVR_q_100all_data.dat
All queried samples: φ, ψ, and sampled ∂A/∂φ, ∂A/∂ψ.

_Converged Free Energy Landscape via Metadynamics
_
Getting_Converged_Landscape_Metadynamics/ holds the metadynamics runs used to compute a converged landscape and the associated convergence analysis.

_Uniform Grid Sampling
_
Getting_Grid_Bayesianquad/ provides the uniform grid reference data.
	•	umbrella_sampling_on_a_grid_ad.py generates the grid samples.
	•	grid_sampling_result/ contains the final dataset from this grid-based sampling.

⸻

2. Water → Ice Phase Transition

_Simulation Input
_
simulation_essentials/ includes all LAMMPS files required for the biased MD runs.

_Bayesian Quadrature
_
bayesquad_pt_ice.py performs BUQ for the phase transition system.

_Final Results
_
results_PT_15.0_ns_Matern12_ls_20.0_w_0.1_n0.0_kappa_es_100_adaptive0_queries_15_full/
Run used in the paper.
Kernel: Matern-1/2, lengthscale 20.0, exploitation weight 0.1, κ = 100, noise 0.0, 15 adaptive queries. (with the 4 initial, 19 in total)

Key data files:
	•	PT_15.0_ns_Matern12_ls_20.0_w_0.1_n0.0_kappa_es_100_adaptive0_queries_15_fullall_data.dat
Sampled N_ice values and sampled ∂A/∂N_ice 
	•	PT_15.0_ns_Matern12_ls_20.0_w_0.1_n0.0_kappa_es_100_adaptive0_queries_15_fullall_data_rerun.dat
Corrected dataset in which one outlier point was replaced with a rerun.
See rerun_point/ for the rerun calculation.

⸻

3. SN2 Reaction

_Simulation Input
_
mace_2_swa.model and p.xyz are the required files for generating MD samples.

_Bayesian Quadrature
_
bq_chemical.py executes BUQ for the SN2 system.

_Final Results
_
results_CR_Matern52_ls_0.2_0.2_w_fes0.1__w_path0.0_n0.0_full_50/
Run used in the paper.
Kernel parameters:  Matern-5/2, lengthscale 0.2 for both dimensions, exploitation weight 0.1, noise 0.0,  50 adaptive queries. (with the 3 initial, 53 in total), but only used 50 for creating the plot in the paper.
Key dataset:
	•	CR_Matern52_ls_0.2_0.2_w_fes0.1__w_path0.0_n0.0_full_50all_data.dat
Query data including (d1, d2, and sampled ∂A/∂d1, ∂A/∂d2).


