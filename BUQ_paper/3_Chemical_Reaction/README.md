
**Simulation Input**	

•	MACE_2_swa.model and p.xyz

Required to generate MD samples for the SN2 reaction using a MACE potential.

**bq_chemical.py**

Executes Bayesian quadrature for the SN2 system. Parameters are read from params.csv.

**Final Results**

results_CR_Matern52_ls_0.2_0.2_w_fes0.1__w_path0.0_n0.0_full_50/ :Production run used in the paper.

Settings:

	•	Kernel: Matérn 5/2
  
	•	Lengthscales: 0.2, 0.2
  
	•	Exploitation weight: 0.1
  
	•	Noise: 0.0
  
	•	Adaptive queries: 50 (3 initial + 50 adaptive = 53 total evaluations)
  
	•	Only 50 evaluations were used to generate the plots shown in the paper.

**Key Data File**

	•	CR_Matern52_ls_0.2_0.2_w_fes0.1__w_path0.0_n0.0_full_50all_data.dat
  
Contains all query points (d_1, d_2) and sampled gradients.
