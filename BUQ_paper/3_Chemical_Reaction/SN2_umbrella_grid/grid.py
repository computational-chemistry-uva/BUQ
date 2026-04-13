
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


from typing import List, Optional, Tuple, Union

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


def do_simulation(d2,d1):

    timestep = 0.5 * units.fs
    atoms = read('p.xyz', '0')
    potential = MACECalculator(model_paths='MACE_2_swa.model', device='cpu')

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

measure_after_ps = 10


# grid size
N_D1 = 10
N_D2 = 10
der_d1 = []
der_d2 = []
if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python opes_sn2.py <task_id>")

    task_id = int(sys.argv[1])  # will come from SLURM_ARRAY_TASK_ID

    if not (0 <= task_id < N_D1 * N_D2):
        raise ValueError(f"task_id {task_id} out of range [0, {N_D1 * N_D2 - 1}]")

    # 1D index → 2D (i,j)
    i = task_id // N_D2   # 0..N_D1-1
    j = task_id %  N_D2   # 0..N_D2-1

    d1_vals = np.linspace(d1_lb, d1_ub, N_D1)
    d2_vals = np.linspace(d2_lb, d2_ub, N_D2)

    d1 = float(d1_vals[i])
    d2 = float(d2_vals[j])

    # common directory for all COLVAR files
    os.makedirs("colvars", exist_ok=True)

    print(f"Running simulation for task_id={task_id}, i={i}, j={j}, d1={d1:.4f}, d2={d2:.4f}")
    # do_simulation(d2, d1)

    forces = get_force(d2,d1, measure_after_ps = measure_after_ps)
    der_d2.append(forces[0])
    der_d1.append(forces[1])



der_d1 = np.array(der_d1)
der_d2 = np.array(der_d2)

X_data = np.column_stack((init_d2, init_d1))
force_data =  np.column_stack((der_d2, der_d1))
force_data = np.array(force_data).reshape(-1, 2)  # 2D output

lengthscale = 0.2
kernel1 = GPy.kern.Matern52(2, lengthscale=lengthscale, variance=1, ARD=True)
kernel2 = GPy.kern.src.static.White(2,variance = noise)
kernel = kernel1 + kernel2
gpy_model = GPy.models.GPRegression(X=X_data, Y=force_data, kernel=kernel)
emukit_kernel = SumMatern52WhiteGPy(gpy_model.kern)
emukit_measure = LebesgueMeasure.from_bounds(bounds=[(d2_lb, d2_ub), (d1_lb, d1_ub)])
emukit_qrbf = QuadratureProductMatern52LebesgueMeasure(emukit_kernel, emukit_measure)       



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
    

fig = plt.figure(figsize=(14, 12))
ax = plt.gca()

contour = ax.contourf(X, Y, bq_int, levels=100, cmap="viridis")
ax.set_title("Prediction using Bayesian Quadrature", fontsize=16)
ax.scatter(emukit_method.X[:, 0], emukit_method.X[:, 1], color="white")

plt.colorbar(contour, ax=ax)  # Optional: adds color scale
plt.savefig(name + f"fes_after.png")
# Show the combined figure
plt.show()


