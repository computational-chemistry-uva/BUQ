

**simulation_essentials/**

	•	Contains all LAMMPS and PLUMED files required to run the biased MD simulations used to evaluate $$\partial A / \partial N_{ice}.$$


**bayesquad_pt_ice.py**

	•	Main Python script performing Bayesian quadrature for the phase transition system.
  
	•	Reads parameters from params.csv and launches MD evaluations as required by the acquisition strategy.

**params.csv**

	•	Parameter table used for SLURM array jobs. Each row corresponds to one Bayesian quadrature run.

**rerun point**

	•	The rerun input and output are stored in rerun_point/.


**Final Results**

results_PT_15.0_ns_Matern12_ls_20.0_w_0.1_n0.0_kappa_es_100_adaptive0_queries_15_full/ :Production run used in the paper.
  
Settings:
  
	•	Kernel: Matérn 1/2
  
	•	Lengthscale: 20.0
  
	•	Exploitation weight: 0.1
  
	•	\kappa = 100
  
	•	Noise: 0.0
  
	•	Adaptive queries: 15 (4 initial + 15 adaptive = 19 total evaluations)


***Key Data Files***

	•	PT_15.0_ns_Matern12_ls_20.0_w_0.1_n0.0_kappa_es_100_adaptive0_queries_15_fullall_data.dat (Sampled N_{ice} values and corresponding \partial A / \partial N_{ice}.)
  
	•	PT_15.0_ns_Matern12_ls_20.0_w_0.1_n0.0_kappa_es_100_adaptive0_queries_15_fullall_data_rerun.dat (Corrected dataset where one outlier was replaced by a rerun calculation.)

  
  

**Running the Code**
The calculations were executed on an HPC system using SLURM. Each Bayesian quadrature run is launched as a SLURM array job, with one array index per row of params.csv.

Execution flow:
	1.	A job-specific scratch directory is created on local storage.
	2.	simulation_essentials/, bayesquad_pt_ice.py, and params.csv are copied to scratch.
	3.	The Python script is executed inside a Conda environment with LAMMPS and PLUMED available.
	4.	On completion (or cancellation), all outputs are copied back to the submission directory under a results folder named after the run.

The SLURM submission script used for production runs is provided in the repository and can be adapted to local cluster configurations (partition, walltime, GPUs, modules).

