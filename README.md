Bayesian Umbrella Quadrature (BUQ) developed by Eline Kempkes and Alberto Pérez de Alba Ortíz 
----------------------------------------------------------------------------------------------



Overview

This repository contains the full set of scripts, input files, and results used to produce the Bayesian Umbrella Quadrature (BUQ) results in the accompanying paper. Three molecular systems are included: alanine dipeptide in vacuo, the water–ice phase transition, and the SN2 reaction. Each system directory contains (i) all simulation input files required to reproduce the biased MD or metadynamics runs, (ii) the BUQ implentation of that system, and (iii) the final data and outputs used in the paper.

------------

## ****1. Alanine Dipeptide in Vacuo****

### _Bayesian Quadrature_

**Bayesian_Quadrature/**  contains the BUQ implementation for alanine dipeptide. 

**_Simulation Input_** :

 **simulation_essentials/** : is the full GROMACS input set used for generating MD data.

**_Bayesian Quadrature_**

_bq_adipep.py:_ Main script for running BUQ. It uses the parameters defined in params.csv . 

**_Final Results_**

**final_result_0.2_ns_Matern52_ls_0.75_w_0.1_n0.0_acq_IVR_q_100/** :Output directory for the run used in the paper.

The settings are,	Kernel: Matern-5/2, lengthscale 0.75,exploitation weight 0.1, 0.0 noise, IVR acquisition, 100 queries.
In this result directory, there are two key files: 

_AD_0.2_ns_Matern52_ls_0.75_w_0.1_n0.0_acq_IVR_q_100_adipep.txt:_ RMSD with respect to the FES after every BUQ query. 

_AD_0.2_ns_Matern52_ls_0.75_w_0.1_n0.0_acq_IVR_q_100all_data.dat_: All queried samples: φ, ψ, and sampled ∂A/∂φ, ∂A/∂ψ.





### **Getting_Converged_Landscape_Metadynamics** :
holds the metadynamics runs used to compute a converged landscape (**running_metadynamics**) and the associated convergence analysis (**results_metadynamics**)






 ### **Getting_Grid_Bayesianquad/** 
 provides the uniform grid reference data. _umbrella_sampling_on_a_grid_ad.py:_ generates the grid samples. **grid_sampling_result/:** contains the final dataset from this grid-based sampling.



------------

## **2. Water → Ice Phase Transition**

**_Simulation Input_** :

**simulation_essentials/** includes all LAMMPS files required for the biased MD runs.

**_Bayesian Quadrature_**

_bayesquad_pt_ice.py_ performs BUQ for the phase transition system.

**_Final Results_**

**results_PT_15.0_ns_Matern12_ls_20.0_w_0.1_n0.0_kappa_es_100_adaptive0_queries_15_full/** -->  Run used in the paper with settings: Kernel: Matern-1/2, lengthscale 20.0, exploitation weight 0.1, κ = 100, noise 0.0, 15 adaptive queries. (with the 4 initial, 19 in total). 

We have the following key data files:


_PT_15.0_ns_Matern12_ls_20.0_w_0.1_n0.0_kappa_es_100_adaptive0_queries_15_fullall_data.dat:_ Sampled N_ice values and sampled ∂A/∂N_ice 

_PT_15.0_ns_Matern12_ls_20.0_w_0.1_n0.0_kappa_es_100_adaptive0_queries_15_fullall_data_rerun.dat:_
Corrected dataset in which one outlier point was replaced with a rerun. See **rerun_point/** for the rerun calculation.


--------------

## **3. SN2 Reaction**

**_Simulation Input_**

mace_2_swa.model and p.xyz are the required files for generating MD samples.

**_Bayesian Quadrature_**

_bq_chemical.py_ executes BUQ for the SN2 system.

**_Final Results_**

**results_CR_Matern52_ls_0.2_0.2_w_fes0.1__w_path0.0_n0.0_full_50/** --> Run used in the paper.

With the parameters:  Matern-5/2, lengthscale 0.2 for both dimensions, exploitation weight 0.1, noise 0.0,  50 adaptive queries. (with the 3 initial, 53 in total), but only used 50 for creating the plot in the paper. 
Here, the most important datafile is _CR_Matern52_ls_0.2_0.2_w_fes0.1__w_path0.0_n0.0_full_50all_data.dat_ , Which has all the query data including (d1, d2, and sampled ∂A/∂d1, ∂A/∂d2).


