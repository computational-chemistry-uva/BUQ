

#change the run command when running on the cluster


import sys
import numpy as np
import GPy
from emukit.quadrature.methods import VanillaBayesianQuadrature
from emukit.model_wrappers.gpy_quadrature_wrappers import \
    BaseGaussianProcessGPy, RBFGPy
from emukit.quadrature.kernels import QuadratureRBFLebesgueMeasure
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
from ase import units
from ase.md.bussi import Bussi
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.io import read, write
from emukit.quadrature.kernels import QuadratureProductMatern52,LebesgueEmbedding
from typing import Union
from mace.calculators import MACECalculator

from ase.calculators.plumed import Plumed

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


class QuadratureProductMatern52LebesgueMeasure(QuadratureProductMatern52, LebesgueEmbedding):
    """A product Matern52 kernel augmented with integrability w.r.t. the standard Lebesgue measure.

    .. seealso::
       * :class:`emukit.quadrature.interfaces.IProductMatern52`
       * :class:`emukit.quadrature.kernels.QuadratureProductMatern52`
       * :class:`emukit.quadrature.measures.LebesgueMeasure`

    :param matern_kernel: The standard EmuKit product Matern52 kernel.
    :param measure: The Lebesgue measure.
    :param variable_names: The (variable) name(s) of the integral.

    """

    def __init__(self, matern_kernel: IProductMatern52, measure: LebesgueMeasure, variable_names: str = "") -> None:
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
        s5 = np.sqrt(5)
        first_term = 16 * lengthscale / (3 * s5)
        second_term = (
            -np.exp(s5 * (x - b) / lengthscale)
            / (15 * lengthscale)
            * (8 * s5 * lengthscale**2 + 25 * lengthscale * (b - x) + 5 * s5 * (b - x) ** 2)
        )
        third_term = (
            -np.exp(s5 * (a - x) / lengthscale)
            / (15 * lengthscale)
            * (8 * s5 * lengthscale**2 + 25 * lengthscale * (x - a) + 5 * s5 * (a - x) ** 2)
        )
        return (first_term + second_term + third_term) * normalization

    def _qKq_1d(self, **parameters) -> float:
        a, b = parameters["domain"]
        lengthscale = parameters["lengthscale"]
        normalization = 1 / (b - a) if parameters["normalize"] else 1.0
        c = np.sqrt(5) * (b - a)
        bracket_term = 5 * a**2 - 10 * a * b + 5 * b**2 + 7 * c * lengthscale + 15 * lengthscale**2
        qKq = (2 * lengthscale * (8 * c - 15 * lengthscale) + 2 * np.exp(-c / lengthscale) * bracket_term) / 15
        return float(qKq) * normalization**2

    def _dqK_dx_1d(self, x: np.ndarray, **parameters) -> np.ndarray:
        a, b = parameters["domain"]
        lengthscale = parameters["lengthscale"]
        normalization = 1 / (b - a) if parameters["normalize"] else 1.0
        s5 = np.sqrt(5)
        first_exp = -np.exp(s5 * (x - b) / lengthscale) / (15 * lengthscale)
        first_term = first_exp * (15 * lengthscale - 15 * s5 * (x - b) + 25 / lengthscale * (x - b) ** 2)
        second_exp = -np.exp(s5 * (a - x) / lengthscale) / (15 * lengthscale)
        second_term = second_exp * (-15 * lengthscale + 15 * s5 * (a - x) - 25 / lengthscale * (a - x) ** 2)
        return (first_term + second_term) * normalization

def do_simulation(d2,d1):

    timestep = 0.5 * units.fs
    atoms = read('p.xyz', '0')
    potential = MACECalculator(model_paths='MACE_2_swa.model', device='cuda')

    # pulling rc=d2-d1 does not work, C-F bond is too strong.
    # "steer: MOVINGRESTRAINT ARG=rc STEP0=5000 AT0=2.0 KAPPA0=50000.00 STEP1=255000 AT1=-1.0",
    # notice we start from the lowest energy, with C bound to F.
    bias = [f"UNITS LENGTH=A TIME=ps ENERGY=kcal/mol",
            "d1: DISTANCE ATOMS=1,4 NOPBC",
            "d2: DISTANCE ATOMS=1,5 NOPBC",
            "rc: COMBINE ARG=d1,d2 COEFFICIENTS=-1,1 PERIODIC=NO",
            f"steer: MOVINGRESTRAINT ARG=d1,d2 STEP0=1000 AT0=2.64,1.84 KAPPA0=1000.0,100.0 STEP1=5000 AT1={d1},{d2}",
            "ener: ENERGY",
            "an: ANGLE ATOMS=1,2,4,5 NOPBC",
            "res: RESTRAINT ARG=an AT=pi*0.5 KAPPA=100.0",
            "an2: ANGLE ATOMS=1,5,4 NOPBC",
            "res2: RESTRAINT ARG=an2 AT=0.0 KAPPA=100.0",
            f"PRINT ARG=* STRIDE=100 FILE=colvars/COLVAR_{d1}_{d2}",
            "FLUSH STRIDE=500"]

    atoms.calc = Plumed(calc=potential,
                        input=bias,
                        timestep=timestep,
                        atoms=atoms)

    MaxwellBoltzmannDistribution(atoms, temperature_K=300)

    dyn = Bussi(atoms, timestep, temperature_K=300, taut=100*timestep,
            logfile=f'colvars/log_{d1}_{d2}', loginterval=500)
    def write_frame():
        dyn.atoms.write(f'colvars/t_{d1}_{d2}.xyz', append=True)
    dyn.attach(write_frame, interval=500)

    dyn.run(80000)


task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))

# Read CSV manually
with open("params.csv", "r") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

row = rows[task_id]

# Extract parameters

lengthscale = (float(row["lengthscale_0"]), float(row["lengthscale_1"]))
queries = int(row["queries"])
weight_acq_fes = float(row["weight_acq_fes"])
weight_path = float(row["weight_path"]) 

noise = float(row["noise"])
kernel_type = row["kernel_type"]
full = row["full"].strip().lower() == "true"  # Convert "True"/"False" to boolean
# Create run name


measure_after_ps = 10


# full=False

# kernel_type = "Matern52" #RBF

def get_force(d2,d1,kappa_d2=100,kappa_d1= 1000, measure_after_ps=measure_after_ps):
    pattern = f"colvars/COLVAR_{d1}_{d2}*"

    # Glob for the matching file
    files = glob.glob(pattern)

    if len(files) != 1:
        raise FileNotFoundError(f"Expected exactly one COLVAR file, found {len(files)} matching: {pattern}")

    filename = files[0]
    data = np.genfromtxt(filename)

    mask = data[:,0]> measure_after_ps
    data = data[mask]
    d1_real = np.mean((data[:,1]))
    d2_real = np.mean((data[:,2]))
    

    force_d2 = np.mean((d2_real - d2 )*kappa_d2)
    force_d1 = np.mean((d1_real - d1 )*kappa_d1)
    return np.array([-force_d2,-force_d1])


d2_lb = 1.8
d2_ub = 3.5

d1_lb= 1.2
d1_ub = 2.8


#%% standard parameters DO NOT CHANGE
init_d2 = np.array([ 1.84,  2.6,3.468 ])
der_d2 = []

init_d1 = np.array([ 2.64,1.6,1.413])
der_d1 = []   
for i in range(3):
    d2 = init_d2[i]
    d1 = init_d1[i]
    print(f"Running simulation for d2={d2}, d1={d1}")
    do_simulation(d2,d1)
    forces = get_force(d2,d1, measure_after_ps = measure_after_ps)
    der_d2.append(forces[0])
    der_d1.append(forces[1])

der_d1 = np.array(der_d1)
der_d2 = np.array(der_d2)

X_data = np.column_stack((init_d2, init_d1))
force_data =  np.column_stack((der_d2, der_d1))
force_data = np.array(force_data).reshape(-1, 2)  # 2D output


if kernel_type == "RBF":
    kernel1 = GPy.kern.RBF(2, lengthscale=lengthscale, variance=1, ARD=True)
    kernel2 = GPy.kern.src.static.White(2,variance = noise)
    kernel = kernel1 + kernel2
    gpy_model = GPy.models.GPRegression(X=X_data, Y=force_data, kernel=kernel)
    emukit_kernel = SumRBFWhiteGPy(gpy_model.kern)
    emukit_measure = LebesgueMeasure.from_bounds(bounds=[(d2_lb, d2_ub), (d1_lb, d1_ub)])
    emukit_qrbf = QuadratureRBFLebesgueMeasure(emukit_kernel, emukit_measure)       

elif kernel_type == "Matern52":
    kernel1 = GPy.kern.Matern52(2, lengthscale=lengthscale, variance=1, ARD=True)
    kernel2 = GPy.kern.src.static.White(2,variance = noise)
    kernel = kernel1 + kernel2
    gpy_model = GPy.models.GPRegression(X=X_data, Y=force_data, kernel=kernel)
    emukit_kernel = SumMatern52WhiteGPy(gpy_model.kern)
    emukit_measure = LebesgueMeasure.from_bounds(bounds=[(d2_lb, d2_ub), (d1_lb, d1_ub)])
    emukit_qrbf = QuadratureProductMatern52LebesgueMeasure(emukit_kernel, emukit_measure)       





lengthscale_str = "_".join(map(str, lengthscale))
full_str = "full" if full else "nofull"
name = f"PT_{kernel_type}_ls_{lengthscale_str}_w_fes{weight_acq_fes}__w_path{weight_path}_n{noise}_{full_str}_{queries}"

# Save run name
with open("run_name.txt", "w") as f:
    f.write(name)




rmsd = []
save_queries= []

# Create a 2D m

x_grid = np.linspace(d2_lb, d2_ub, num=60)
y_grid = np.linspace(d1_lb, d1_ub, num= 60)
X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
X_flat = np.vstack([X.ravel(), Y.ravel()]).T



lower_line = 3.1 - 0.7 * X
upper_line = 3.96 - 0.7 * X

# Create mask: 1 if Y between the lines, else 0
sampling_grid = np.logical_and(Y >= lower_line, Y <= upper_line).astype(int)

def integration_2D_rgrid(
        grid: ndarray,
        dA_grid: ndarray,
        integrator: str = 'simpson+mini',
        fast: str="no") -> ndarray:
    '''
        Integration of a 2D regular/rectangular surface from its gradient

        Parameters
        ----------
        grid : ndarray(grid_f*n_j,grid_f*n_i,2)
            matrix of grid coordinates
        dA_grid : ndarray(grid_f*n_j,grid_f*n_i,2)
            matrix of free energy derivatives
        integrator : {trapz, simpson, trapz+mini, simpson+mini, fourier}, optional
            integration algorithm (default: 'simpson+mini')

        Returns
        -------
        A_grid : ndarray(grid_f*n_j,grid_f*n_i)
            matrix of integrated free energy,
            minimum value set to zero
        important!! first y, then x:
            
             
        XY_combined = np.stack((Y,X),axis=-1)
        derivative_xy_combined = np.stack((y_derivative,x_derivative),axis=-1)
           
    '''

    # check integrator
    if integrator not in {'trapz', 'simpson', 'trapz+mini', 'simpson+mini', 'fourier'}:
        raise ValueError(f"Integrator '{integrator}' not recognized")

    ## grid related definitions
    n_ig = grid.shape[1]
    n_jg = grid.shape[0]
    n_grid = n_jg * n_ig
    dx, dy = abs(grid[0,0,0] - grid[0,1,0]), abs(grid[0,0,1] - grid[1,0,1])    # space between points
    # initialize integrated surface matrix
    A_grid = np.zeros((n_jg,n_ig))

    # difference of gradients per grid point [Kästner 2009 - Eq.14] (optimization format)
    def D_tot(F):
        F = F.reshape(n_jg,n_ig)
        dFy, dFx = np.gradient(F,dy,dx)
        dF = np.stack((dFx,dFy), axis=-1)
        return np.sum((dA_grid - dF)**2) / n_grid

    def callback(A):
        print(f"Current loss: {D_tot(A):.6f}")
  
    ## Simpson's rule integration
    sys.stdout.write("# Integrating             - Simpson's rule ")
    for j in range(n_jg):
        for i in range(n_ig):
            if i == 0 and j == 0:
                A_grid[j, i] = 0  # corner point to zero
            elif i == 0:
                A_grid[j, i] = A_grid[j-1, i] + (dA_grid[j-1, i, 1] + dA_grid[j, i, 1]) * dy / 2
            elif j == 0:
                A_grid[j, i] = A_grid[j, i-1] + (dA_grid[j, i-1, 0] + dA_grid[j, i, 0]) * dx / 2
            else:
                A_grid[j, i] = A_grid[j-1, i-1] \
                               + (dA_grid[j-1, i-1, 0] + dA_grid[j-1, i, 0] + dA_grid[j, i-1, 0] + dA_grid[j, i, 0]) * dx / 6 \
                               + (dA_grid[j-1, i-1, 1] + dA_grid[j-1, i, 1] + dA_grid[j, i-1, 1] + dA_grid[j, i, 1]) * dy / 6

    ## real-space grid minimization
    # TODO: Global optimization methods -> Differential Evolution
    # FIXME: Now minimization of the squared difference of gradients
    #        per grid point instead of the derivative of difference
    #        of gradients (it matters?)
    if 'mini' in integrator:
        sys.stdout.write("+ Real Space Grid Mini ")
        sys.stdout.flush()
        # L-BFGS-B minimization of sumation of square of gradient differences
        if fast == "no":
            mini_result = scipy_optimize.minimize(D_tot, A_grid.ravel(), method='L-BFGS-B', options={'maxfun':np.inf, 'maxiter':np.inf, 'maxls':50, 'iprint':-1})
        
        if fast =="yes":
            mini_result = scipy_optimize.minimize(D_tot, A_grid.ravel(), method='L-BFGS-B', options={'maxfun':np.inf, 'maxiter':80, 'maxls':50, 'iprint':10}, callback=callback)

        if not mini_result.success:
            sys.stdout.write("\nWARNING: Minimization could not converge")
        A_grid = mini_result.x.reshape(n_jg,n_ig)



    # integration error
    sys.stdout.write(f"\n# Integration error:        {D_tot(A_grid.ravel()):.2f}\n\n")

    # set minimum to zero
    A_grid = A_grid - np.min(A_grid)

    # return integrated surface
    return A_grid


def write_result(rmsd_array, lengthscale, queries, weight_acq_fes, kernel_type, noise):
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

        f.write("\nRMSD Results:\n")

        # Save the RMSD array
        np.savetxt(f, rmsd_array, fmt="%.6f")
    
    print(f"Results saved to {file_path}")


#%% changeable parameters


emukit_model = BaseGaussianProcessGPy(kern=emukit_qrbf, gpy_model=gpy_model)

# Bayesian Quadrature method
emukit_method = VanillaBayesianQuadrature(base_gp=emukit_model, X=X_data, Y=force_data)
ivr_acquisition = IntegralVarianceReduction(emukit_method)
space = ParameterSpace(emukit_method.reasonable_box_bounds.convert_to_list_of_continuous_parameters())
optimizer = GradientAcquisitionOptimizer(space)
weight_acq_ivr = 1.0 - weight_acq_fes - weight_path
predicted_derivatives, _ = emukit_method.predict(X_flat)
predicted_derivatives = predicted_derivatives.reshape(X.shape[0], Y.shape[1], 2)
XY_combined = np.stack((Y,X),axis=-1)
derivative_xy_combined = np.stack((predicted_derivatives[:, :, 1],predicted_derivatives[:, :, 0]),axis=-1)
bq_int = integration_2D_rgrid(XY_combined,derivative_xy_combined, "simpson+mini")
bq_int = bq_int 
#%%
        
with open(name +"all_data.dat", "w") as f:
    for i in range(len(emukit_method.X)):
        f.write(f"{i+1} \t {emukit_method.X[i][0]} \t {emukit_method.X[i][1]} \t {emukit_method.Y[i][0]} \t {emukit_method.Y[i][1]}  \n")
    

for i in range(1,queries+1): #10
    print(f" -------------- BaysOpt loop, query {i} ---------------")
    
    ivr_plot = ivr_acquisition.evaluate(X_flat)
    ivr_plot = ivr_plot.reshape(X.shape) / np.max(ivr_plot)
    
    scaled_free_energy = bq_int/np.max(bq_int)# voor introduction alpha
    together = + weight_acq_ivr*ivr_plot -weight_acq_fes* scaled_free_energy + weight_path * sampling_grid
   
    max_index_together = np.unravel_index(np.argmax(together), together.shape) #alpha kan hier weer bij
    new_x_ivr = X[max_index_together]
    new_y_ivr = Y[max_index_together]   

    free_energy_value  = scaled_free_energy[max_index_together]
    

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # --- Plot 1: IVR ---
    contour1 = axes[0].contourf(X, Y, ivr_plot, levels=100, cmap="viridis")
    cbar1 = fig.colorbar(contour1, ax=axes[0], shrink=0.8, aspect=30, pad=0.02)
    cbar1.set_label("IVR")
    axes[0].set_title(f"IVR Contour Plot, {i}")

    # --- Plot 2: Scaled Free Energy ---
    contour2 = axes[1].contourf(X, Y, scaled_free_energy, levels=100, cmap="plasma")
    cbar2 = fig.colorbar(contour2, ax=axes[1], shrink=0.8, aspect=30, pad=0.02)
    cbar2.set_label("Scaled Free Energy")
    axes[1].set_title(f"Scaled Free Energy, {i}")

    # --- Plot 3: Together ---
    contour3 = axes[2].contourf(X, Y, together, levels=100, cmap="cividis")
    cbar3 = fig.colorbar(contour3, ax=axes[2], shrink=0.8, aspect=30, pad=0.02)
    cbar3.set_label("Combined")
    axes[2].set_title(f"Combined acqui, {i}")

    plt.savefig(name + f"_acqui_{i}.png", dpi=300)
    plt.show()






    print(f"going to run a simulation at {new_x_ivr} {new_y_ivr} ")
    
    do_simulation(new_x_ivr,new_y_ivr)

    force_x, force_y  = get_force(new_x_ivr,new_y_ivr, measure_after_ps = measure_after_ps) 
    force_xy =np.array([force_x,force_y])
    xy_new = np.array([[new_x_ivr,new_y_ivr]])
    X_data = np.append(X_data, xy_new, axis=0)
    force_data = np.vstack([force_data, force_xy])
    emukit_method.set_data(X_data, force_data)
    with open(name + "all_data.dat", "a") as f:  # 'a' mode to append
        f.write(f"{i} \t {emukit_method.X[-1][0]} \t {emukit_method.X[-1][1]}   \t {emukit_method.Y[-1][0]} \t {emukit_method.Y[-1][1]} \n")
    
        
    predicted_derivatives, _ = emukit_method.predict(X_flat)
    predicted_derivatives = predicted_derivatives.reshape(X.shape[0], Y.shape[1], 2)
    XY_combined = np.stack((Y,X),axis=-1)
    derivative_xy_combined = np.stack((predicted_derivatives[:, :, 1],predicted_derivatives[:, :, 0]),axis=-1)
 
    bq_int = integration_2D_rgrid(XY_combined,derivative_xy_combined, "simpson+mini")



    save_queries.append(i)
    if full:
         fig = plt.figure(figsize=(14, 12))
         ax = plt.gca()

         contour = ax.contourf(X, Y, bq_int, levels=100, cmap="viridis")
         ax.set_title("Prediction using Bayesian Quadrature", fontsize=16)
         ax.scatter(emukit_method.X[:, 0], emukit_method.X[:, 1], color="white")

         plt.colorbar(contour, ax=ax)  # Optional: adds color scale
         plt.savefig(name + f"fes_after_{i}.png")
            # Show the combined figure
         plt.show()
            

predicted_derivatives, _ = emukit_method.predict(X_flat)
predicted_derivatives = predicted_derivatives.reshape(X.shape[0], Y.shape[1], 2)
XY_combined = np.stack((Y,X),axis=-1)
derivative_xy_combined = np.stack((predicted_derivatives[:, :, 1],predicted_derivatives[:, :, 0]),axis=-1)

bq_int = integration_2D_rgrid(XY_combined,derivative_xy_combined, "simpson+mini")
fig = plt.figure(figsize=(14, 12))
ax = plt.gca()

contour = ax.contourf(X, Y, bq_int, levels=100, cmap="viridis")
ax.set_title("Prediction using Bayesian Quadrature", fontsize=16)
ax.scatter(emukit_method.X[:, 0], emukit_method.X[:, 1], color="white")

# Adjust layout
plt.savefig(name+"fes.png")

# Show the combined figure
plt.show()

rmsd_arr= np.array(rmsd)
write_result(rmsd_arr, lengthscale, queries, weight_acq_fes,kernel_type, noise)





