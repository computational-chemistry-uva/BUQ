

import numpy as np
import GPy
from typing import Union
import matplotlib.pyplot as plt
import csv
import sys
import os
import subprocess

from scipy import optimize as scipy_optimize
from emukit.quadrature.methods import VanillaBayesianQuadrature
from emukit.model_wrappers.gpy_quadrature_wrappers import  BaseGaussianProcessGPy
from emukit.quadrature.kernels import QuadratureProductMatern52,LebesgueEmbedding
from emukit.quadrature.kernels import QuadratureRBFLebesgueMeasure
from emukit.quadrature.measures import LebesgueMeasure
from emukit.core.optimization import GradientAcquisitionOptimizer
from emukit.core.parameter_space import ParameterSpace
from emukit.quadrature.methods import VanillaBayesianQuadrature
from emukit.model_wrappers.gpy_quadrature_wrappers import   BaseGaussianProcessGPy
import emukit.quadrature.acquisitions as emu_acqui
from emukit.quadrature.interfaces import  IProductMatern52, IStandardKernel



#%% standard parameters DO NOT CHANGE

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

def get_metadynamics( derivatives=False):
    metafile = open("fes_final.dat")
    data = np.genfromtxt(metafile)
    metafile.close()
    fes = data[:,2]
    ow1=data[:,0].reshape(100,100)
    ow2=data[:,1].reshape(100,100)
    dx = data[:,3].reshape(100,100)
    dy = data[:,4].reshape(100,100)
    if derivatives:
        return ow1,ow2,dx,dy
    return (fes - np.min(fes)).reshape(100,100)




lb_1 = -np.pi 
ub_1 = np.pi
lb_2=-np.pi
ub_2= np.pi
kappa =200
measure_after_ps = 100 
ns = 0.2
nsteps = 500000 * ns 
nsteps = int(nsteps)
current_psi = 1.311767
current_phi = -1.580543
rmsd = []
save_queries= []
name = "US_on_a_grid"
analytical = get_metadynamics()
analytical=analytical.T
vmin = analytical.min()
vmax = analytical.max()
ow1,ow2, x_derivative,y_derivative = get_metadynamics(derivatives="yes")

x_grid = np.unique(ow1)
y_grid = np.unique(ow2)



X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
analytical = get_metadynamics()
analytical=analytical.T
vmin = analytical.min()
vmax = analytical.max()
ow1,ow2, x_derivative,y_derivative = get_metadynamics(derivatives="yes")
rmsd = []
queries = []
total_n_points = 10
X_flat = np.vstack([X.ravel(), Y.ravel()]).T


def integration_2D_rgrid(
        grid: np.ndarray,
        dA_grid: np.ndarray,
        integrator: str = 'simpson+mini',
        fast: bool= False) -> np.ndarray:
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

    if 'mini' in integrator:
        sys.stdout.write("+ Real Space Grid Mini ")
        sys.stdout.flush()
        # L-BFGS-B minimization of sumation of square of gradient differences
        if not fast:
            mini_result = scipy_optimize.minimize(D_tot, A_grid.ravel(), method='L-BFGS-B', options={'maxfun':np.inf, 'maxiter':np.inf, 'maxls':50, 'iprint':-1})
        if fast:
            mini_result = scipy_optimize.minimize(D_tot, A_grid.ravel(), method='L-BFGS-B', options={'maxfun':np.inf, 'maxiter':80, 'maxls':50, 'iprint':10}, callback=callback)
        if not mini_result.success:
            sys.stdout.write("\nWARNING: Minimization could not converge")
        A_grid = mini_result.x.reshape(n_jg,n_ig)

    sys.stdout.write(f"\n# Integration error:        {D_tot(A_grid.ravel()):.2f}\n\n")

    # set minimum to zero
    A_grid = A_grid - np.min(A_grid)

    # return integrated surface
    return A_grid


def get_force(phi_value, psi_value, kappa_phi=200, kappa_psi=200, measure_after_ps=1000):
    data = np.genfromtxt(f"Colvars/COLVAR_{phi_value:.3f}_{psi_value:.3f}")
    data = data[data[:, 0] > measure_after_ps]

    # Mean values
    mean_vals = np.mean(data[:, 1:5], axis=0)
    sin_phi_real, cos_phi_real, sin_psi_real, cos_psi_real = mean_vals

    sin_phi_umbrella, cos_phi_umbrella = np.sin(phi_value), np.cos(phi_value)
    sin_psi_umbrella, cos_psi_umbrella = np.sin(psi_value), np.cos(psi_value)

    # Forces along sin/cos
    force_phi_vec = np.array([sin_phi_real - sin_phi_umbrella, cos_phi_real - cos_phi_umbrella]) * kappa_phi
    force_psi_vec = np.array([sin_psi_real - sin_psi_umbrella, cos_psi_real - cos_psi_umbrella]) * kappa_psi

    # Total forces with sign
    sign_phi = -1 if np.arctan2(sin_phi_real, cos_phi_real) < np.arctan2(sin_phi_umbrella, cos_phi_umbrella) else 1
    sign_psi = -1 if np.arctan2(sin_psi_real, cos_psi_real) < np.arctan2(sin_psi_umbrella, cos_psi_umbrella) else 1

    force_phi = np.linalg.norm(force_phi_vec) * sign_phi
    force_psi = np.linalg.norm(force_psi_vec) * sign_psi

    return np.array([-force_phi, -force_psi])




def write_plumed_file(phi, psi, kappa_phi=200, kappa_psi=200, current_phi=0.0, current_psi=0.0):
    """
    Generates a PLUMED input file for torsional restraints on phi and psi angles.
    
    Args:
        phi (float): Target phi angle in radians.
        psi (float): Target psi angle in radians.
        kappa_phi (float): Force constant for phi.
        kappa_psi (float): Force constant for psi.
        current_phi (float): Current phi reference value.
        current_psi (float): Current psi reference value.
    """
    equisteps = 500
    moving_speed = 1000
    build_up_kappa_steps = 1000 + equisteps

    angles = {
        "phi": {"target": phi, "kappa": kappa_phi, "current": current_phi},
        "psi": {"target": psi, "kappa": kappa_psi, "current": current_psi}
    }

    filename = f"Colvars/plumed_{phi:.3f}_{psi:.3f}.dat"
    with open(filename, "w") as f:
        f.write("#vim:ft=plumed\n")
        f.write("MOLINFO STRUCTURE=diala.pdb\n")
        f.write("UNITS LENGTH=A TIME=ps ENERGY=kcal/mol\n")
        f.write("phi: TORSION ATOMS=@phi-2\n")
        f.write("psi: TORSION ATOMS=@psi-2\n")
        f.write("cos_phi: MATHEVAL arg=phi FUNC=cos(x) PERIODIC=NO\n")
        f.write("sin_phi: MATHEVAL arg=phi FUNC=sin(x) PERIODIC=NO\n")
        f.write("cos_psi: MATHEVAL arg=psi FUNC=cos(x) PERIODIC=NO\n")
        f.write("sin_psi: MATHEVAL arg=psi FUNC=sin(x) PERIODIC=NO\n")

        for angle_name, info in angles.items():
            distance = abs(info["current"] - info["target"])
            step_to = int(build_up_kappa_steps + distance * moving_speed)
            for trig in ["cos", "sin"]:
                target_val = np.cos(info["target"]) if trig == "cos" else np.sin(info["target"])
                current_val = np.cos(info["current"]) if trig == "cos" else np.sin(info["current"])
                f.write(
                    f"restraint_{angle_name}_{trig}: MOVINGRESTRAINT ...\n"
                    f"ARG={trig}_{angle_name}\n"
                    f"STEP0={equisteps} AT0={current_val} KAPPA0=0\n"
                    f"STEP1={build_up_kappa_steps} AT1={current_val} KAPPA1={info['kappa']}\n"
                    f"STEP2={step_to} AT2={target_val} KAPPA2={info['kappa']}\n"
                    "...\n"
                )

        f.write(f"PRINT ARG=sin_phi,cos_phi,sin_psi,cos_psi,*.* "
                f"FILE=Colvars/COLVAR_{phi:.3f}_{psi:.3f} STRIDE=100\n")
    
        


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


def write_result(rmsd_array,npointsarray):
    """
    Saves the experimental settings and RMSD results to a file.
    """
    # Automatically set the name based on lengthscale and weight_acq_fes
    name = "US_grid_points_US"
    file_path = f"{name}.txt"
    
    # Open the file for writing
    with open(file_path, "w") as f:
       # Write the header with all settings
       f.write(f"Experiment Name: {name}\n")
       f.write("\nRMSD Results:\n")

       # Save the RMSD array and npointsarray to the file in a formatted way
       for rmsd, npoints in zip(rmsd_array, npointsarray):
           f.write(f"{rmsd:.6f} {npoints}\n")
    

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

def get_n_grid_points(n):
    x_points= np.linspace(lb_1,ub_1,num=n,endpoint=False)
    y_points= np.linspace(lb_2,ub_2,num=n,endpoint=False)

    return x_points,y_points


for n_points in range(1,total_n_points+1):  
    
    sampling_points_x, sampling_points_y = get_n_grid_points(n_points)
    X_data = np.empty((0, 2))  # Empty array with shape (0,2) for positions
    force_data = np.empty((0, 2))  # Empty array with shape (0,2) for force components
    force_data = np.array(force_data).reshape(-1, 2)    
    for x_sample in sampling_points_x:
        for y_sample in sampling_points_y:
            write_plumed_file(x_sample, y_sample,kappa_phi=kappa, kappa_psi=kappa, current_phi=current_phi, current_psi=current_psi)
            run_command("rm *#*")
            command = "srun --mpi=pmix_v4 gmx_mpi mdrun -s md.tpr -plumed  Colvars/plumed_{:.3f}_{:.3f}.dat -nsteps {} -x Colvars/traj_{:.3f}_{:.3f}.xtc".format(x_sample, y_sample,nsteps,x_sample, y_sample)
            run_command(command)
            force_x, force_y = get_force(x_sample, y_sample, measure_after_ps= measure_after_ps) 
            force_xy =np.array([force_x,force_y])
            xy_new = np.array([[x_sample,y_sample]])
            X_data = np.append(X_data, xy_new, axis=0)
            force_data = np.vstack([force_data, force_xy])

            # Mirror point if x or y equals -π to cover symmetric cases
            x_mirrors = [x_sample, -x_sample] if x_sample == -np.pi else [x_sample]
            y_mirrors = [y_sample, -y_sample] if y_sample == -np.pi else [y_sample]

            for xm in x_mirrors:
                for ym in y_mirrors:
                    if xm == x_sample and ym == y_sample:
                        continue  # skip the original point, already added
                    X_data = np.append(X_data, np.array([[xm, ym]]), axis=0)
                    force_data = np.vstack([force_data, force_xy]) 
                            

    noise=0.0
    lengthscale=0.5 #also for the bayes opt kernel
    kernel1 = GPy.kern.RBF(2, lengthscale=lengthscale, variance=1, ARD=True)
    kernel2 = GPy.kern.src.static.White(2,variance = noise)
    kernel = kernel1 + kernel2
    gpy_model = GPy.models.GPRegression(X=X_data, Y=force_data, kernel=kernel)
    emukit_kernel = SumRBFWhiteGPy(gpy_model.kern)
    emukit_measure = LebesgueMeasure.from_bounds(bounds=[(lb_1, ub_1), (lb_2, ub_2)])
    emukit_qrbf = QuadratureRBFLebesgueMeasure(emukit_kernel, emukit_measure)       

   
    emukit_model = BaseGaussianProcessGPy(kern=emukit_qrbf, gpy_model=gpy_model)
    emukit_method = VanillaBayesianQuadrature(base_gp=emukit_model, X=X_data, Y=force_data)
    ivr_acquisition = emu_acqui.IntegralVarianceReduction(emukit_method)
    space = ParameterSpace(emukit_method.reasonable_box_bounds.convert_to_list_of_continuous_parameters())
    optimizer = GradientAcquisitionOptimizer(space)
    predicted_derivatives, _ = emukit_method.predict(X_flat)
    predicted_derivatives = predicted_derivatives.reshape(X.shape[0], Y.shape[1], 2)
    XY_combined = np.stack((Y,X),axis=-1)
    derivative_xy_combined = np.stack((predicted_derivatives[:, :, 1],predicted_derivatives[:, :, 0]),axis=-1)
    
    bq_int = integration_2D_rgrid(XY_combined,derivative_xy_combined, "simpson+mini")
    
    rmsd_query = np.sqrt(np.mean((analytical - bq_int) ** 2))
    rmsd.append(rmsd_query)
    queries.append(n_points*n_points)
    
     
    fig, axes = plt.subplots(2, 2, figsize=(14, 13))
    
    # Define shared color scale for the first two plots

    
    # First plot: Converged Metadynamics
    contour1 = axes[0, 0].contourf(X, Y,analytical, levels=100, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0, 0].set_title("Ground Truth Free Energy")
    
    # Second plot: Predicted BQ
    contour2 = axes[0, 1].contourf(X, Y, bq_int, levels=100, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0, 1].set_title("Prediction Using Grid ")
    axes[0,1].scatter(emukit_method.X[:,0], emukit_method.X[:,1], color="white")
    
    # Shared colorbar for first two plots
    cbar1 = fig.colorbar(contour1, ax=axes[0, :], shrink=0.8, location="right")
    cbar1.set_label(" Free Energy (kcal/mol)")
    
    # Third plot: Difference (using its own colormap)
    contour3 = axes[1, 0].contourf(X, Y, analytical - bq_int, levels=100, cmap="coolwarm")
    axes[1, 0].set_title("Difference")
    cbar2 = fig.colorbar(contour3, ax=axes[1, 0], shrink=0.8, location="right")
    cbar2.set_label("Difference (kcal/mol)")
    
    # Compute and plot RMSD
     
    for ax in axes.ravel():
        ax.set_xlabel(r"$\phi$", fontsize=14)
        ax.set_ylabel(r"$\psi$", fontsize=14)
        ax.tick_params(axis='both', labelsize=12)
    axes[1, 1].plot(queries, rmsd, marker="o", linestyle="-")
    
    axes[1, 1].set_xlabel("Query")
    axes[1, 1].set_ylabel("RMSD (kcal/mol)")
    axes[1, 1].set_title("RMSD")
    fig.suptitle(f"after {n_points*n_points} queries")

    # Adjust layout
    plt.savefig(name+f"fes_after_{n_points*n_points}.png")
    
    # Show the combined figure
    plt.show()
    
    with open(f"all_data_gridpoints_{n_points*n_points}.dat", "w") as f:
        for i in range(len(emukit_method.X)):
            f.write(f"{i+1} \t {emukit_method.X[i][0]} \t {emukit_method.X[i][1]} \t {emukit_method.Y[i][0]} \t {emukit_method.Y[i][1]}  \n")
    


rmsd_arr= np.array(rmsd)
npoints_arr = np.array(queries)
write_result(rmsd_arr, npoints_arr)

    