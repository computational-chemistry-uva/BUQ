

#change the run command when running on the cluster


import sys
import numpy as np
import GPy
from emukit.quadrature.methods import VanillaBayesianQuadrature
from emukit.model_wrappers.gpy_quadrature_wrappers import \
    BaseGaussianProcessGPy, RBFGPy
from emukit.quadrature.kernels import QuadratureRBFLebesgueMeasure, LebesgueEmbedding,QuadratureProductMatern12
from typing import Union
from emukit.quadrature.measures import LebesgueMeasure
from emukit.quadrature.acquisitions import IntegralVarianceReduction
from emukit.core.optimization import GradientAcquisitionOptimizer
from emukit.core.parameter_space import ParameterSpace
from emukit.quadrature.methods import VanillaBayesianQuadrature
from emukit.model_wrappers.gpy_quadrature_wrappers import \
    BaseGaussianProcessGPy, RBFGPy
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
import os
import subprocess
from emukit.quadrature.kernels import QuadratureKernel
from emukit.model_wrappers import GPyModelWrapper
import emukit.model_wrappers.gpy_quadrature_wrappers as emuwrap
from numpy import ndarray
from scipy import optimize as scipy_optimize
import glob
from matplotlib.colors import TwoSlopeNorm,Normalize
import csv
from emukit.quadrature.interfaces import (
    IRBF,
    IBaseGaussianProcess,
    IBrownian,
    IProductBrownian,
    IProductMatern12,
    IProductMatern32,
    IProductMatern52,
    IStandardKernel
)
from typing import List, Optional, Tuple, Union
from scipy.interpolate import interp1d



class SumRBFWhiteGPy(IStandardKernel):
    """
    Wrapper for a sum of GPy RBF and White kernels to be used in Emukit quadrature.

    :param gpy_rbf: An RBF kernel from GPy.
    :param gpy_white: A White kernel from GPy.
    """

    def __init__(self, gpy_kernel):
        
        gpy_rbf = gpy_model.kern.parts[0]
        gpy_white = gpy_model.kern.parts[1]
        self.gpy_rbf = gpy_rbf
        self.gpy_white = gpy_white
        self.gpy_kernel = gpy_rbf + gpy_white  # Sum of kernels

    @property
    def lengthscales(self) -> np.ndarray:
       if self.gpy_rbf.ARD:
           return self.gpy_rbf.lengthscale.values
       return np.full((self.gpy_rbf.input_dim,), self.gpy_rbf.lengthscale[0])

    @property
    def variance(self) -> float:
        """Returns the variance of the total"""
        return self.gpy_rbf.variance.values[0] # I dont think we need to add the variance of the white


    def K(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """Computes the full kernel matrix (RBF + White)."""
        return self.gpy_kernel.K(x1, x2)

    def dK_dx1(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        scaled_vector_diff = np.swapaxes((x1[None, :, :] - x2[:, None, :]) / self.lengthscales**2, 0, -1)
        return -self.K(x1, x2)[None, ...] * scaled_vector_diff


    def dKdiag_dx(self, x: np.ndarray) -> np.ndarray:
        return np.zeros((x.shape[1], x.shape[0]))


class SumMatern52WhiteGPy(IStandardKernel):
    """
    Wrapper for a sum of GPy Matern and White kernels to be used in Emukit quadrature.

    :param gpy_rbf: An RBF kernel from GPy.
    :param gpy_white: A White kernel from GPy.
    """

    def __init__(self, gpy_kernel):
        
        gpy_matern = gpy_model.kern.parts[0]
        gpy_white = gpy_model.kern.parts[1]
        self.gpy_matern = gpy_matern
        self.gpy_white = gpy_white
        self.gpy_kernel = gpy_matern + gpy_white  # Sum of kernels

    @property
    def lengthscales(self) -> np.ndarray:
       if self.gpy_matern.ARD:
           return self.gpy_matern.lengthscale.values
       return np.full((self.gpy_matern.input_dim,), self.gpy_matern.lengthscale[0])

    @property
    def variance(self) -> float:
        """Returns the variance of the total"""
        return self.gpy_matern.variance.values[0] # I dont think we need to add the variance of the white


    def K(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """Computes the full kernel matrix (RBF + White)."""
        return self.gpy_kernel.K(x1, x2)

    def dK_dx1(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        scaled_vector_diff = np.swapaxes((x1[None, :, :] - x2[:, None, :]) / self.lengthscales**2, 0, -1)
        return -self.K(x1, x2)[None, ...] * scaled_vector_diff


    def dKdiag_dx(self, x: np.ndarray) -> np.ndarray:
        return np.zeros((x.shape[1], x.shape[0]))


class QuadratureProductMatern12LebesgueMeasure(QuadratureProductMatern12, LebesgueEmbedding):
    """A product Matern12 kernel augmented with integrability w.r.t. the standard Lebesgue measure.

    .. seealso::
       * :class:`emukit.quadrature.interfaces.IProductMatern12`
       * :class:`emukit.quadrature.kernels.QuadratureProductMatern12`
       * :class:`emukit.quadrature.measures.LebesgueMeasure`

    :param matern_kernel: The standard EmuKit product Matern12 kernel.
    :param measure: The Lebesgue measure.
    :param variable_names: The (variable) name(s) of the integral.

    """

    def __init__(self, matern_kernel: IProductMatern12, measure: LebesgueMeasure, variable_names: str = "") -> None:
        super().__init__(matern_kernel=matern_kernel, measure=measure, variable_names=variable_names)

    def _scale(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return self.variance * z
    def _get_univariate_parameters(self, dim: int) -> dict:
        lengthscales = self.lengthscales
        # Handle isotropic vs anisotropic lengthscales
        if np.ndim(lengthscales) == 0 or lengthscales.size == 1:
            ls = float(lengthscales)
        else:
            ls = float(lengthscales[dim])
        return {
            "domain": self.measure.domain.bounds[dim],
            "lengthscale": ls,
            "normalize": self.measure.is_normalized,
        }
    def _qK_1d(self, x: np.ndarray, **parameters) -> np.ndarray:
        a, b = parameters["domain"]
        lengthscale = parameters["lengthscale"]
        normalization = 1 / (b - a) if parameters["normalize"] else 1.0
        first_term = -np.exp((a - x) / lengthscale)
        second_term = -np.exp((x - b) / lengthscale)
        return normalization * lengthscale * (2.0 + first_term + second_term)

    def _qKq_1d(self, **parameters) -> float:
        a, b = parameters["domain"]
        lengthscale = parameters["lengthscale"]
        normalization = 1 / (b - a) if parameters["normalize"] else 1.0
        qKq = 2.0 * lengthscale * ((b - a) + lengthscale * (np.exp(-(b - a) / lengthscale) - 1.0))
        return float(qKq) * normalization**2

    def _dqK_dx_1d(self, x: np.ndarray, **parameters) -> np.ndarray:
        a, b = parameters["domain"]
        lengthscale = parameters["lengthscale"]
        normalization = 1 / (b - a) if parameters["normalize"] else 1.0
        first_term = np.exp((a - x) / lengthscale)
        second_term = -np.exp((x - b) / lengthscale)
        return (first_term + second_term) * normalization

def get_force(es, kappa_es, measure_after_ps=1000):
    es_label = f"{es:.3f}".replace('.', '_')
    pattern = f"colvars/COLVAR_{es_label}.*"

    # Glob for the matching file
    files = glob.glob(pattern)

    if len(files) != 1:
        raise FileNotFoundError(f"Expected exactly one COLVAR file, found {len(files)} matching: {pattern}")

    filename = files[0]
    data = np.genfromtxt(filename)

    mask = data[:,0]> measure_after_ps
    data = data[mask]
    es_real = np.mean(data[:,1])

    force_es = np.mean((es_real - es )*kappa_es)
    return np.array([-force_es])



def write_custom_plumed_file(es_target, kappa_es,steeringsteps= 250000, equil_steps=250000):
    """
    Generate a PLUMED input file with custom environment similarity restraints.
    
    Parameters:
    es_target (float): Target value for environment similarity restraint in STEP2.
    kappa_es (float): Force constant for environment similarity restraint.
    """
    # Build file name
    es_label = f"{es_target:.3f}".replace('.', '_')
    file_path = f"colvars/plumed_{es_label}.dat"
    
    with open(file_path, "w") as file:
        # VOLUME
        file.write("vol: VOLUME\n\n")
        
        # ENVIRONMENTSIMILARITY block 1
        file.write("ENVIRONMENTSIMILARITY ...\n")
        file.write(" SPECIES=1-864:3\n")
        file.write(" SIGMA=0.055\n")
        file.write(" CRYSTAL_STRUCTURE=CUSTOM\n")
        file.write(" LABEL=refcv\n")
        file.write(" REFERENCE_1=Environments/IceIhExtendedEnvironments/env1.pdb\n")
        file.write(" REFERENCE_2=Environments/IceIhExtendedEnvironments/env2.pdb\n")
        file.write(" REFERENCE_3=Environments/IceIhExtendedEnvironments/env3.pdb\n")
        file.write(" REFERENCE_4=Environments/IceIhExtendedEnvironments/env4.pdb\n")
        file.write(" MORE_THAN={RATIONAL R_0=0.5 NN=12 MM=24}\n")
        file.write(" MEAN\n")
        file.write("... ENVIRONMENTSIMILARITY\n\n")
        
        # ENVIRONMENTSIMILARITY block 2
        file.write("ENVIRONMENTSIMILARITY ...\n")
        file.write(" SPECIES=1-864:3\n")
        file.write(" SIGMA=0.055\n")
        file.write(" CRYSTAL_STRUCTURE=CUSTOM\n")
        file.write(" LABEL=refcv2\n")
        file.write(" REFERENCE_1=Environments/IceIcExtendedEnvironments/env1.pdb\n")
        file.write(" REFERENCE_2=Environments/IceIcExtendedEnvironments/env2.pdb\n")
        file.write(" MORE_THAN={RATIONAL R_0=0.5 NN=12 MM=24}\n")
        file.write(" MEAN\n")
        file.write("... ENVIRONMENTSIMILARITY\n\n")
        
        # diff MATHEVAL
        file.write("diff: MATHEVAL ARG=refcv2.mean,refcv.mean FUNC=((x-0.26)/(0.58-0.26)-(y-0.29)/(0.80-0.29)) PERIODIC=NO\n\n")
        
        # UPPER_WALLS
        file.write("UPPER_WALLS ARG=diff AT=0.04 KAPPA=100000 EXP=2 LABEL=uwall\n\n")
        
        # Q6
        file.write("Q6 SPECIES=1-288:3 SWITCH={CUBIC D_0=0.3 D_MAX=0.35} VMEAN LABEL=q6\n\n")
        
        # diff2 MATHEVAL
        file.write("diff2: MATHEVAL ARG=q6.vmean,refcv.mean FUNC=((x-0.0668781995)/(0.39184059-0.0668781995)-(y-0.2899390548628429)/(0.7838534089775562-0.2899390548628429)) PERIODIC=NO\n\n")
        
        # MOVINGRESTRAINT
        file.write("restraint_more_es: MOVINGRESTRAINT ...\n")
        file.write(" ARG=refcv.morethan\n")
        file.write(" STEP0=0 AT0=288 KAPPA0=0\n")
        file.write(" STEP1={} AT1=288 KAPPA1={}\n".format(steeringsteps,kappa_es))
        file.write(" STEP2={} AT2={} KAPPA2={}\n".format(equil_steps+steeringsteps, es_target, kappa_es))
        file.write("...\n\n")
        
        # PRINT
        file.write("PRINT STRIDE=500 ARG=refcv.morethan,vol,* FILE=colvars/COLVAR_{}\n".format(es_label))


def run_command(command):
    # Build the full srun command for the cluster
    setup_cmds = f"mpirun -np 32 /home/ekempke/software/lammps_ice/build/lmp -partition 1x32 {command} &> lmp.out"

    # Use current environment variables (assuming conda env already active)
    env = os.environ.copy()

    try:
        subprocess.run(setup_cmds, shell=True, check=True, executable='/bin/bash', env=env)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Command failed: {command}")
        print(e)



def write_lammps_input(es, nsteps):
    es_label = f"{es:.3f}".replace('.', '_')
    
    filename = f"colvars/input_{es}"
    content = f"""variable    temperature equal 300.0
variable    tempDamp equal 100.0

variable    pressure equal 1.
variable    pressureDamp equal 1000.0 # This is 1 ps

variable    seed equal 745821

units       real
atom_style  full

read_data   water.data.0

variable    out_freq equal 1000
variable    out_freq2 equal 1000

timestep    2.0

neigh_modify    delay 7 every 1

include     in.tip4p

thermo          ${{out_freq}}
thermo_style    custom step temp pe etotal epair emol press lx ly lz vol pxx pyy pzz pxy pxz pyz

restart     ${{out_freq}} restart.lmp restart2.lmp

dump            myDump all atom ${{out_freq2}} colvars/dump_{es_label}.water
dump_modify     myDump append yes

fix             1 all plumed plumedfile colvars/plumed_{es_label}.dat outfile plumed_out{es}
fix             2 all shake 1e-6 200 0 b 1 a 1
fix             3 all nph iso ${{pressure}} ${{pressure}} ${{pressureDamp}}
fix             4 all temp/csvr ${{temperature}} ${{temperature}} ${{tempDamp}} ${{seed}}
velocity        all create ${{temperature}} ${{seed}} dist gaussian

run             {nsteps}

write_data      data_{es_label}.final

write_restart   restart.lmp
"""

    with open(filename, "w") as f:
        f.write(content)





task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))

# Read CSV manually
with open("params.csv", "r") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

row = rows[task_id]

# Extract parameters
kappa_es_base = int(row["kappa_es"])
adaptive_kappa = int(row["adaptive_kappa"])
ns = float(row["ns"])
lengthscale = (float(row["lengthscale"]))
queries = int(row["queries"])
weight_acq_fes = float(row["weight_acq_fes"])
noise = float(row["noise"])
full = row["full"].strip().lower() == "true"  # Convert "True"/"False" to boolean

# Create run name


measure_after_ps = 2000
kappa_es = kappa_es_base  #kj?
nsteps = int(500000 * ns )
build_up_steps = 1000
second_equil= 50000-build_up_steps #half a ns 


init_es = np.array([1.6, 3.0, 284.160 , 285.0 ])
der_1 = []   
for i in range(len(init_es)):
    es = init_es[i]
    write_custom_plumed_file(es,kappa_es=kappa_es, equil_steps= second_equil, steeringsteps=build_up_steps)
    write_lammps_input(es,nsteps)
    command = f"-in colvars/input_{es}"
    run_command(command)
    forces = get_force(es,kappa_es, measure_after_ps = measure_after_ps)
    der_1.append(forces[0])

der_1 = np.array(der_1)
X_data = init_es.reshape(-1, 1)
force_data = der_1.reshape(-1, 1)  # 2D output



kernel_type = "Matern12"
kernel1 = GPy.kern.Exponential(1, lengthscale=lengthscale, variance=1, ARD=True)
kernel2 = GPy.kern.src.static.White(1,variance = noise)
kernel = kernel1 + kernel2
gpy_model = GPy.models.GPRegression(X=X_data, Y=force_data, kernel=kernel)
emukit_kernel = SumMatern52WhiteGPy(gpy_model.kern)




full_str = "full" if full else "nofull"
name = f"PT_{ns}_ns_{kernel_type}_ls_{lengthscale}_w_{weight_acq_fes}_n{noise}_kappa_es_{kappa_es_base}_adaptive{adaptive_kappa}_queries_{queries}_{full_str}"

# Save run name
with open("run_name.txt", "w") as f:
    f.write(name)




es_lb= 0.0
es_ub = 288.0

rmsd = []
save_queries= []

# Create a 2D meshgrid


fes_analytical_data = np.genfromtxt("fes_analytical.data")


x_grid = fes_analytical_data[:,0]

f_analytical = fes_analytical_data[:,1] #inkjoule/mol

x_pred = x_grid.reshape(-1, 1)

#analytical = analytical/4.184


def write_result(rmsd_array, lengthscale, queries, weight_acq_fes, kernel_type, noise, kappa_vol):
    """
    Saves the experimental settings and RMSD results to a file.
    """
    # Automatically set the name based on lengthscale and weight_acq_fes
    file_path = f"{name}_PT.txt"
    
    # Open the file for writing
    with open(file_path, "w") as f:
        # Write the header with all settings
        f.write(f"Experiment Name: {name}\n")
        f.write(f"Lengthscale: {lengthscale}\n")
        f.write(f"Queries: {queries}\n")
        f.write(f"Weight Acquisition FES: {weight_acq_fes}\n")
        f.write(f"Kernel: {kernel_type}\n")
        f.write(f"noise: {noise}\n")
        f.write(f"kappa_vol: {kappa_vol}\n")

        f.write("\nRMSD Results:\n")

        # Save the RMSD array
        np.savetxt(f, rmsd_array, fmt="%.6f")
    
    print(f"Results saved to {file_path}")


#%% changeable parameters


emukit_measure = LebesgueMeasure.from_bounds(bounds=[(es_lb, es_ub)])
emukit_qrbf = QuadratureProductMatern12LebesgueMeasure(emukit_kernel, emukit_measure)
emukit_model = BaseGaussianProcessGPy(kern=emukit_qrbf, gpy_model=gpy_model)

# Bayesian Quadrature method
emukit_method = VanillaBayesianQuadrature(base_gp=emukit_model, X=X_data, Y=force_data)
ivr_acquisition = IntegralVarianceReduction(emukit_method)
space = ParameterSpace(emukit_method.reasonable_box_bounds.convert_to_list_of_continuous_parameters())
optimizer = GradientAcquisitionOptimizer(space)
weight_acq_ivr = 1.0 - weight_acq_fes
predicted_derivative, _ = emukit_method.predict(x_pred)
bq_integral = cumulative_trapezoid(predicted_derivative[:, 0], x_grid, initial=0)
bq_integral -= np.min(bq_integral)

#%%
        
with open(name +"all_data.dat", "w") as f:
    for i in range(len(emukit_method.X)):
        f.write(f"{i+1} \t {emukit_method.X[i]} \t {emukit_method.Y[i]} \n")
    

for i in range(1,queries+1): #10
    print(f" -------------- BaysOpt loop, query {i} ---------------")
    
    ivr_plot = ivr_acquisition.evaluate(x_pred).flatten()
    ivr_plot = ivr_plot / np.max(ivr_plot)

    scaled_free_energy = bq_integral / np.max(bq_integral)  # voor introduction alpha
    together = +weight_acq_ivr * ivr_plot - weight_acq_fes * scaled_free_energy

    max_index_together = np.unravel_index(np.argmax(together), together.shape)  # alpha kan hier weer bij

    new_x_ivr = x_grid[max_index_together]
    
    plt.figure()

    plt.plot(x_grid, ivr_plot, label="IVR Acquisition", color='green')
    plt.plot(x_grid, scaled_free_energy, label="Scaled Free Energy", color='blue')
    plt.plot(x_grid, together, label="Combined Acquisition", color='red')
    plt.axvline(new_x_ivr, color='black', linestyle='--', label='Next Query Point')
    plt.title(f"Acquisition Functions and Next Query Point (Iter {i})")
    plt.savefig(name + f"acq_iter_{i}.png")


    free_energy_value  = scaled_free_energy[max_index_together]
    
    kappa_es = kappa_es_base + adaptive_kappa * free_energy_value
    print(free_energy_value)
    print(kappa_es)
    print(f"going to run a simulation at {new_x_ivr }")
    write_custom_plumed_file(new_x_ivr, kappa_es=kappa_es, equil_steps= second_equil, steeringsteps=build_up_steps)
    write_lammps_input(new_x_ivr,nsteps)
    command = f"-in colvars/input_{new_x_ivr}"
      
    run_command(command)

    force_x  = get_force(new_x_ivr,kappa_es, measure_after_ps = measure_after_ps) 
    force_xy =np.array([force_x])
    xy_new = np.array([[new_x_ivr]])
    X_data = np.vstack([X_data, xy_new])


    force_data = np.vstack([force_data, force_xy])
    emukit_method.set_data(X_data, force_data)
    with open(name + "all_data.dat", "a") as f:  # 'a' mode to append
        f.write(f"{i} \t {emukit_method.X[-1]} \t   {emukit_method.Y[-1]} \n")
    
        
    predicted_derivative, _ = emukit_method.predict(x_pred)
    bq_integral = cumulative_trapezoid(predicted_derivative[:, 0], x_grid, initial=0)
    bq_integral -= np.min(bq_integral)
    
    rmsd_query = np.sqrt(np.mean((f_analytical - bq_integral) ** 2))
    rmsd.append(rmsd_query)
    save_queries.append(i)
    if full:
                # Create new plot each time
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot true and predicted integrals
        ax.plot(x_grid, f_analytical - np.min(f_analytical), label="True Integral (Shifted)", color='black', linewidth=2)
        ax.plot(x_grid, bq_integral - np.min(bq_integral), '--', label=f"Predicted Integral (Iter {i+1})", color='blue')
        
        # Interpolate predicted integral to get y-values for queried X_data
        queried_integral_interp = interp1d(x_grid, bq_integral - np.min(bq_integral), kind='linear', bounds_error=False, fill_value="extrapolate")
        queried_y = queried_integral_interp(X_data[:, 0])
        
        # Plot queried points on the predicted curve
        ax.scatter(X_data[:, 0], queried_y, color='red', label="Queried Points", zorder=5)

        plt.savefig(name + f"fes_iter_{i}.png")

predicted_derivative, _ = emukit_method.predict(x_pred)
bq_integral = cumulative_trapezoid(predicted_derivative[:, 0], x_grid, initial=0)
bq_integral -= np.min(bq_integral)

fig, ax = plt.subplots(figsize=(10, 6))

# Plot true and predicted integrals
ax.plot(x_grid, f_analytical - np.min(f_analytical), label="True Integral (Shifted)", color='black', linewidth=2)
ax.plot(x_grid, bq_integral - np.min(bq_integral), '--', label=f"Predicted Integral (Iter {i+1})", color='blue')

# Interpolate predicted integral to get y-values for queried X_data
queried_integral_interp = interp1d(x_grid, bq_integral - np.min(bq_integral), kind='linear', bounds_error=False, fill_value="extrapolate")
queried_y = queried_integral_interp(X_data[:, 0])

# Plot queried points on the predicted curve
ax.scatter(X_data[:, 0], queried_y, color='red', label="Queried Points", zorder=5)

# Adjust layout
plt.savefig(name+"fes.png")

# Show the combined figure
plt.show()

rmsd_arr= np.array(rmsd)
write_result(rmsd_arr, lengthscale, queries, weight_acq_fes,kernel_type, noise, kappa_es)





