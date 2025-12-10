#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 16:31:07 2024

@author: eline
"""

import numpy as np
import GPy
from emukit.quadrature.methods import VanillaBayesianQuadrature
from emukit.model_wrappers.gpy_quadrature_wrappers import \
    BaseGaussianProcessGPy, RBFGPy
from emukit.quadrature.kernels import QuadratureProductMatern52,LebesgueEmbedding
from typing import Union
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
from scipy.interpolate import RegularGridInterpolator
from numpy import ndarray
from scipy import optimize as scipy_optimize
import emukit.quadrature.acquisitions as emu_acqui
import csv
import sys
import emukit.model_wrappers.gpy_quadrature_wrappers as emuwrap

import os
import subprocess
from emukit.quadrature.kernels import QuadratureKernel
from emukit.model_wrappers import GPyModelWrapper
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
    Wrapper for a sum of GPy RBF and White kernels to be used in Emukit quadrature.

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

def get_force(phi_value,psi_value, kappa_phi=200, kappa_psi=200,measure_after_ps=1000):
    data_file = open("Colvars/COLVAR_{:.3f}_{:.3f}".format(phi_value,psi_value))

    data =np.genfromtxt(data_file)
    data_file.close()
    
    mask = data[:,0]> measure_after_ps
    data = data[mask]
    sin_phi_real = np.mean(data[:,1])
    cos_phi_real = np.mean(data[:,2])

    sin_psi_real = np.mean(data[:,3])
    cos_psi_real = np.mean(data[:,4])
    
    sin_phi_umbrella = np.sin(phi_value)
    cos_phi_umbrella = np.cos(phi_value)
    
    sin_psi_umbrella = np.sin(psi_value)
    cos_psi_umbrella = np.cos(psi_value)
    
    
    
    force_sin_phi = (sin_phi_real - sin_phi_umbrella )*kappa_phi #to kcal
    force_cos_phi = ((cos_phi_real - cos_phi_umbrella ))*kappa_phi
    force_sin_psi = ((sin_psi_real - sin_psi_umbrella ))*kappa_psi
    force_cos_psi = ((cos_psi_real - cos_psi_umbrella ))*kappa_psi


    sign_force_phi = -1 if np.arctan2(sin_phi_real, cos_phi_real) < np.arctan2(sin_phi_umbrella, cos_phi_umbrella) else 1
    force_phi = np.sqrt(force_sin_phi**2 + force_cos_phi**2) * sign_force_phi
    
    sign_force_psi = -1 if np.arctan2(sin_psi_real, cos_psi_real) < np.arctan2(sin_psi_umbrella, cos_psi_umbrella) else 1
    force_psi = np.sqrt(force_sin_psi**2 + force_cos_psi**2) * sign_force_psi
      

    return np.array([-force_phi,-force_psi])      
def write_plumed_file(phi, psi, kappa_phi=200,kappa_psi=200):
  
    equisteps= 500
    moving_speed = 1000 
    build_up_kappa_steps = 1000 +equisteps
    
    file = open("Colvars/plumed_{:.3f}_{:.3f}.dat".format(phi,psi), "w")
    file.write("#vim:ft=plumed \n")
    file.write("MOLINFO STRUCTURE=diala.pdb \n")
    file.write("UNITS LENGTH=A TIME=ps ENERGY=kcal/mol\n")

    file.write("phi: TORSION ATOMS=@phi-2 \n")
    file.write("psi: TORSION ATOMS=@psi-2 \n")


    file.write("cos_phi: MATHEVAL arg=phi FUNC=cos(x) PERIODIC=NO \n")
    file.write("sin_phi: MATHEVAL arg=phi FUNC=sin(x) PERIODIC=NO \n")
    file.write("cos_psi: MATHEVAL arg=psi FUNC=cos(x) PERIODIC=NO \n")
    file.write("sin_psi: MATHEVAL arg=psi FUNC=sin(x) PERIODIC=NO \n")

    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    sin_psi = np.sin(psi)
    cos_psi = np.cos(psi)
    
    
    #moving restraint --> starting
    #moving restraint --> starting from 
    distance_phi = np.abs( current_phi - phi)
    distance_psi = np.abs( current_phi - psi)
    

    step_to_phi = int(build_up_kappa_steps+ distance_phi*moving_speed)
    step_to_psi = int(build_up_kappa_steps+ distance_psi*moving_speed)
    file.write("restraint_phi_cos: MOVINGRESTRAINT ... \n ARG=cos_phi \n STEP0={} AT0={} KAPPA0=0 \n".format(equisteps,np.cos(current_phi)))
    file.write("STEP1={} AT1={} KAPPA1={} \n".format(build_up_kappa_steps,np.cos(current_phi),kappa_phi))
    file.write("STEP2={} AT2={} KAPPA2={} \n".format( step_to_phi, cos_phi,kappa_phi))
    file.write("...\n")
    file.write("restraint_phi_sin: MOVINGRESTRAINT ... \n ARG=sin_phi \n STEP0={} AT0={} KAPPA0=0 \n".format(equisteps,np.sin(current_phi)))
    file.write("STEP1={} AT1={} KAPPA1={} \n".format(build_up_kappa_steps,np.sin(current_phi),kappa_phi))
    file.write("STEP2={} AT2={} KAPPA2={} \n".format( step_to_phi, sin_phi,kappa_phi))
    file.write("...\n")
    
    file.write("restraint_psi_cos: MOVINGRESTRAINT ... \n ARG=cos_psi \n STEP0={} AT0={} KAPPA0=0 \n".format(equisteps,np.cos(current_psi)))
    file.write("STEP1={} AT1={} KAPPA1={} \n".format(build_up_kappa_steps,np.cos(current_psi),kappa_psi))
    file.write("STEP2={} AT2={} KAPPA2={} \n".format( step_to_psi, cos_psi,kappa_psi))
    file.write("...\n")
    file.write("restraint_psi_sin: MOVINGRESTRAINT ... \n ARG=sin_psi \n STEP0={} AT0={} KAPPA0=0 \n".format(equisteps,np.sin(current_psi)))
    file.write("STEP1={} AT1={} KAPPA1={} \n".format(build_up_kappa_steps,np.sin(current_psi),kappa_psi))
    file.write("STEP2={} AT2={} KAPPA2={} \n".format( step_to_psi, sin_psi,kappa_psi))
    file.write("...\n")
    
    
    
    file.write("PRINT ARG=sin_phi,cos_phi,sin_psi,cos_psi,*.* FILE=Colvars/COLVAR_{:.3f}_{:.3f} STRIDE=100".format(phi,psi))
    file.close()

    maximum_steps = np.max([step_to_phi,step_to_psi])
def run_command(command):
    # Set up GROMACS environment variables and preserve the existing PATH
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = '/usr/local/gromacs/lib:' + env.get('LD_LIBRARY_PATH', '')
    env['PATH'] = '/usr/local/gromacs/bin:' + env.get('PATH', '/bin:/usr/bin:/usr/local/bin')  # Preserve system PATH
    
    # Execute the command
    try:
        subprocess.run(command, shell=True, check=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running command: {command}")
        print(e)


positions_1 = np.array([ -1.507964473999999999e+00 , 8.796459429999999857e-01])
positions_2 = np.array([1.193805207999999896e+00,  -8.796459429999999857e-01 ])
current_psi =  0.354882
current_phi = -1.401141


task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))

# Read CSV manually
with open("params.csv", "r") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

row = rows[task_id]

# Extract parameters

lengthscale = (float(row["lengthscale"]))
acq_function = row["acq_function"]
queries = int(row["queries"])
weight_acq_fes = float(row["weight_acq_fes"])
noise = float(row["noise"])
kernel_type = row["kernel_type"]
full = row["full"].strip().lower() == "true"  # Convert "True"/"False" to boolean



ns = 0.2
kappa =200
measure_after_ps = 100 #0.5ns maximum steering i saw was 12 ps
nsteps = 500000 * ns 
nsteps = int(nsteps)  


name = f"AD_{ns}_ns_{kernel_type}_ls_{lengthscale}_w_{weight_acq_fes}_n{noise}_acq_{acq_function}_q_{queries}"
with open('run_name.txt', 'w') as f:
    f.write(name)



##initialpoints!
write_plumed_file(positions_1[0], positions_2[0],kappa_phi=kappa, kappa_psi=kappa)
run_command("rm *#*")
command = "srun --mpi=pmix_v4 gmx_mpi mdrun -s finaltop.tpr -v -c diala.gro  -plumed  Colvars/plumed_{:.3f}_{:.3f}.dat -nsteps {}".format(positions_1[0], positions_2[0],nsteps)
run_command(command)
der_1 = get_force(positions_1[0], positions_2[0], measure_after_ps= measure_after_ps)     
write_plumed_file(positions_1[1], positions_2[1],kappa_phi=kappa, kappa_psi=kappa)
run_command("rm *#*")
command = "srun --mpi=pmix_v4 gmx_mpi mdrun -s finaltop.tpr -v -c diala.gro  -plumed  Colvars/plumed_{:.3f}_{:.3f}.dat -nsteps {}".format(positions_1[1], positions_2[1],nsteps)
run_command(command)
der_2 = get_force(positions_1[1], positions_2[1], measure_after_ps= measure_after_ps)    
X_data = np.column_stack((positions_1, positions_2))
force_data =  np.column_stack((der_1, der_2))
force_data = np.array(force_data).reshape(-1, 2)  # 2D output




lb_1 = -np.pi 
ub_1 = np.pi
lb_2=-np.pi
ub_2= np.pi


if kernel_type == "RBF":
    kernel1 = GPy.kern.RBF(2, lengthscale=lengthscale, variance=1, ARD=True)
    kernel2 = GPy.kern.src.static.White(2,variance = noise)
    kernel = kernel1 + kernel2
    gpy_model = GPy.models.GPRegression(X=X_data, Y=force_data, kernel=kernel)
    emukit_kernel = SumRBFWhiteGPy(gpy_model.kern)
    emukit_measure = LebesgueMeasure.from_bounds(bounds=[(lb_1, ub_1), (lb_2, ub_2)])
    emukit_qrbf = QuadratureRBFLebesgueMeasure(emukit_kernel, emukit_measure)       

elif kernel_type == "Matern52":
    kernel1 = GPy.kern.Matern52(2, lengthscale=lengthscale, variance=1, ARD=True)
    kernel2 = GPy.kern.src.static.White(2,variance = noise)
    kernel = kernel1 + kernel2
    gpy_model = GPy.models.GPRegression(X=X_data, Y=force_data, kernel=kernel)
    emukit_kernel = SumMatern52WhiteGPy(gpy_model.kern)
    emukit_measure = LebesgueMeasure.from_bounds(bounds=[(lb_1, ub_1), (lb_2, ub_2)])
    emukit_qrbf = QuadratureProductMatern52LebesgueMeasure(emukit_kernel, emukit_measure)       


print(kernel_type)

name = f"AD_{ns}_ns_{kernel_type}_ls_{lengthscale}_w_{weight_acq_fes}_n{noise}_acq_{acq_function}_q_{queries}"



with open('run_name.txt', 'w') as f:
    f.write(name)


#%% standard parameters DO NOT CHANGE
def get_metadynamics( derivatives="no"):
    metafile = open("fes_final.dat")
    data = np.genfromtxt(metafile)
    metafile.close()
    fes = data[:,2]
    ow1=data[:,0].reshape(100,100)
    ow2=data[:,1].reshape(100,100)
    dx = data[:,3].reshape(100,100)
    dy = data[:,4].reshape(100,100)
    if derivatives=="yes":
        return ow1,ow2,dx,dy
    return (fes - np.min(fes)).reshape(100,100)





rmsd = []
save_queries= []

x_grid = np.linspace(lb_1, ub_1, num=100)
y_grid = np.linspace(lb_2, ub_2, num=100)

# Create a 2D meshgrid
X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
analytical = get_metadynamics()

ow1,ow2, x_derivative,y_derivative = get_metadynamics(derivatives="yes")


def write_result(rmsd_array, lengthscale, queries, weight_acq_fes, kernel_type, noise):
    """
    Saves the experimental settings and RMSD results to a file.
    """
    # Automatically set the name based on lengthscale and weight_acq_fes
    file_path = f"{name}_adipep.txt"
    
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
                               + (dA_grid[j-1, i-1, 0] + dA_grid[j-1, i, 0] + dA_grid[j, i-1, 0] + dA_grid[j, i, 0]) * dx / 4 \
                               + (dA_grid[j-1, i-1, 1] + dA_grid[j-1, i, 1] + dA_grid[j, i-1, 1] + dA_grid[j, i, 1]) * dy / 4

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


    
#%% some initialization:


emukit_model = BaseGaussianProcessGPy(kern=emukit_qrbf, gpy_model=gpy_model)

# Bayesian Quadrature method
emukit_method = VanillaBayesianQuadrature(base_gp=emukit_model, X=X_data, Y=force_data)


if acq_function == "IVR":
    acquisition = IntegralVarianceReduction(emukit_method)

elif acq_function == "US":
    acquisition = emu_acqui.UncertaintySampling(emukit_method)



space = ParameterSpace(emukit_method.reasonable_box_bounds.convert_to_list_of_continuous_parameters())
optimizer = GradientAcquisitionOptimizer(space)
X_flat = np.vstack([X.ravel(), Y.ravel()]).T

predicted_derivatives, _ = emukit_method.predict(X_flat)
predicted_derivatives = predicted_derivatives.reshape(X.shape[0], Y.shape[1], 2)
XY_combined = np.stack((Y,X),axis=-1)
derivative_xy_combined = np.stack((predicted_derivatives[:, :, 1],predicted_derivatives[:, :, 0]),axis=-1)
bq_int = integration_2D_rgrid(XY_combined,derivative_xy_combined, "simpson+mini")
bq_int = bq_int.T

weight_acq_ivr = 1.0 - weight_acq_fes
#%%

        
with open(name +"all_data.dat", "w") as f:
    for i in range(len(emukit_method.X)):
        f.write(f"{i+1} \t {emukit_method.X[i][0]} \t {emukit_method.X[i][1]} \t {emukit_method.Y[i][0]} \t {emukit_method.Y[i][1]}  \n")
    


for i in range(1,queries+1): #10
    print(f" -------------- BaysOpt loop, query {i} ---------------")
    
    ivr_plot = acquisition.evaluate(X_flat)
    ivr_plot = ivr_plot.reshape(X.shape) 
    
    if np.min(ivr_plot) < 0:
        with open(name +"all_data.dat", "w") as f:
            f.write("Acquisition function has negative values, stopping here\n")
        sys.exit("Acquisition function has negative values, stopping here")

    
    ivr_plot = ivr_plot/ np.max(ivr_plot)
    
    scaled_free_energy = bq_int.T/np.max(bq_int)# voor introduction alpha
    together = -weight_acq_fes* scaled_free_energy + weight_acq_ivr*ivr_plot
   
    
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
    axes[2].set_title(f"Combined Contour, {i}")

    plt.savefig(name + f"_acqui_{i}.png", dpi=300)
    plt.show()
    
    max_index_together = np.unravel_index(np.argmax(together), together.shape) #alpha kan hier weer bij
   
    new_x_ivr = X[max_index_together]
    new_y_ivr = Y[max_index_together]   


    print(f"going to run a simulation at {new_x_ivr} {new_y_ivr} ")
    
        #do simulation
       
        
    write_plumed_file(new_x_ivr, new_y_ivr,kappa_phi=kappa, kappa_psi=kappa)
    run_command("rm *#*")
    command = "srun --mpi=pmix_v4 gmx_mpi mdrun -s finaltop.tpr -v -c diala.gro  -plumed  Colvars/plumed_{:.3f}_{:.3f}.dat -nsteps {}".format(new_x_ivr, new_y_ivr,nsteps)
       
    # command = "gmx mdrun -s md_start.tpr  -v  -c pp4_em.gro -plumed  Colvars/plumed_{:.3f}_{:.3f}_{:.3f}.dat -nsteps {}".format(new_x_ivr, new_y_ivr,new_z_ivr,nsteps)

    run_command(command)
    
    
    force_x, force_y = get_force(new_x_ivr, new_y_ivr,kappa_phi= kappa, kappa_psi= kappa, measure_after_ps= measure_after_ps) 
    
        
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
    bq_int = bq_int.T
    rmsd_query = np.sqrt(np.mean((analytical - bq_int) ** 2))
    rmsd.append(rmsd_query)
    save_queries.append(i)
    if full:
           fig, axes = plt.subplots(2, 2, figsize=(14, 12))
           vmin = analytical.min()
           vmax = analytical.max()
            
            # First plot: Converged Metadynamics
           contour1 = axes[0, 0].contourf(X, Y, analytical.T, levels=100, cmap="viridis", vmin=vmin, vmax=vmax)
           axes[0, 0].set_title("Ground Truth Free Energy", fontsize=16)
            
            # Second plot: Predicted BQ
           contour2 = axes[0, 1].contourf(X, Y, bq_int.T, levels=100, cmap="viridis", vmin=vmin, vmax=vmax)
           axes[0, 1].set_title("Prediction using Bayesian Quadrature", fontsize=16)
           axes[0, 1].scatter(emukit_method.X[:, 0], emukit_method.X[:, 1], color="white")
            
            # Shared colorbar for first two plots
           cbar1 = fig.colorbar(contour1, ax=axes[0, :], shrink=0.8, location="right")
           cbar1.set_label("Free Energy (kcal/mol)", fontsize=14)
            
            # Third plot: Difference (using its own colormap)
           contour3 = axes[1, 0].contourf(X, Y, analytical.T - bq_int.T, levels=100, cmap="coolwarm")
           axes[1, 0].set_title("Difference", fontsize=16)
           cbar2 = fig.colorbar(contour3, ax=axes[1, 0], shrink=0.8, location="right")
           cbar2.set_label("Difference (kcal/mol)", fontsize=14)
            
            # Compute and plot RMSD
           axes[1, 1].plot(save_queries, rmsd, marker="o", linestyle="-")
            
      
            # Overall figure title
           fig.suptitle(f"After {i} Queries", fontsize=18, fontweight="bold")
            
            # Adjust axes labels and tick sizes
           for ax in axes.flatten():
                ax.set_xlabel(r"$\phi$", fontsize=14)
                ax.set_ylabel(r"$\psi$", fontsize=14)
                ax.tick_params(axis="both", labelsize=12)
                
           axes[1, 1].set_xlabel("Query", fontsize=14)
           axes[1, 1].set_ylabel("RMSD (kcal/mol)", fontsize=14)
           axes[1, 1].set_title("RMSD", fontsize=16)
            
            # Adjust layout
           plt.savefig(name + f"fes_after_{i}.png")
            
            # Show the combined figure
           plt.show()
            



fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)






# Plot true and predicted derivatives
for j, (data, title) in enumerate(zip(
        [x_derivative, predicted_derivatives[:, :, 0].T, y_derivative, predicted_derivatives[:, :, 1].T],
        ["True ∂F/∂x", "Predicted ∂F/∂x", "True ∂F/∂y", "Predicted ∂F/∂y"])):
    ax = axes[j // 2, j % 2]
    contour = ax.contourf(X, Y, data, levels=100, cmap="coolwarm")
    fig.colorbar(contour, ax=ax)
    ax.set_title(title)
    fig.suptitle(name + f"after {i} queries")
plt.savefig(name+"derivatives.png")


 
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Define shared color scale for the first two plots
vmin = analytical.min()
vmax = analytical.max()

# First plot: Converged Metadynamics
contour1 = axes[0, 0].contourf(X, Y, analytical.T, levels=100, cmap="viridis", vmin=vmin, vmax=vmax)
axes[0, 0].set_title("Converged Metadynamics")

# Second plot: Predicted BQ
contour2 = axes[0, 1].contourf(X, Y, bq_int.T, levels=100, cmap="viridis", vmin=vmin, vmax=vmax)
axes[0, 1].set_title("Predicted BQ")
axes[0,1].scatter(emukit_method.X[:,0], emukit_method.X[:,1], color="red")

# Shared colorbar for first two plots
cbar1 = fig.colorbar(contour1, ax=axes[0, :], shrink=0.8, location="right")
cbar1.set_label("Energy (kcal/mol)")

# Third plot: Difference (using its own colormap)
contour3 = axes[1, 0].contourf(X, Y, analytical.T - bq_int.T, levels=100, cmap="coolwarm")
axes[1, 0].set_title("Difference")
cbar2 = fig.colorbar(contour3, ax=axes[1, 0], shrink=0.8, location="right")
cbar2.set_label("Difference (kcal/mol)")

# Compute and plot RMSD
 

axes[1, 1].plot(save_queries, rmsd, marker="o", linestyle="-")
for ax in axes.flatten():
        ax.set_xlabel(r"$\phi$", fontsize=14)
        ax.set_ylabel(r"$\psi$", fontsize=14)
        ax.tick_params(axis="both", labelsize=12)
                
axes[1, 1].set_xlabel("Query")
axes[1, 1].set_ylabel("RMSD Landscape (kcal/mol)")
axes[1, 1].set_title("RMSD")
fig.suptitle(name+ f"after {i} queries")
# Adjust layout
plt.savefig(name+"fes.png")

# Show the combined figure
plt.show()

rmsd_arr= np.array(rmsd)
write_result(rmsd_arr, lengthscale, queries, weight_acq_fes,kernel_type, noise)


#%%











