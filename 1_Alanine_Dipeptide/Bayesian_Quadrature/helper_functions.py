import numpy as np
from typing import Union
import sys
from emukit.quadrature.kernels import QuadratureProductMatern52,LebesgueEmbedding
from emukit.quadrature.measures import LebesgueMeasure
from emukit.quadrature.interfaces import  IProductMatern52, IStandardKernel
from scipy import optimize as scipy_optimize
import matplotlib.pyplot as plt
import subprocess
import os




# !!! in this class, there was a bug in the emukitpackage with handeling the different dimensions, I fixed it here
class QuadratureProductMatern52LebesgueMeasure(QuadratureProductMatern52, LebesgueEmbedding):
    """Product Matern52 kernel integrated over a standard Lebesgue measure.
    
    Combines the EmuKit QuadratureProductMatern52 kernel with LebesgueEmbedding
    to allow integration of functions with respect to the Lebesgue measure. Fixes
    an EmuKit bug related to handling multidimensional lengthscales.


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
        """Scale input z by kernel variance."""
        return self.variance * z

    def _get_univariate_parameters(self, dim: int) -> dict:
        lengthscales = self.lengthscales
        # Handle isotropic vs anisotropic lengthscales
        if np.ndim(lengthscales) == 0 or lengthscales.size == 1:
            ls = float(lengthscales)
        else:
            ls = float(lengthscales[dim])
        return {
            "domain": self.measure.domain.bounds[dim], #this was the line i had to change
            "lengthscale": ls,
            "normalize": self.measure.is_normalized,
        }
    def _qK_1d(self, x: np.ndarray, **parameters) -> np.ndarray:
        """
        Analytical integration of the 1D Matern52 kernel against the Lebesgue measure.
        Returns an array of integrated values at x.
        """
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
        """
        Analytical integration of the 1D kernel against itself over the domain.
        Returns a scalar.
        """
        a, b = parameters["domain"]
        lengthscale = parameters["lengthscale"]
        normalization = 1 / (b - a) if parameters["normalize"] else 1.0
        c = np.sqrt(5) * (b - a)
        bracket_term = 5 * a**2 - 10 * a * b + 5 * b**2 + 7 * c * lengthscale + 15 * lengthscale**2
        qKq = (2 * lengthscale * (8 * c - 15 * lengthscale) + 2 * np.exp(-c / lengthscale) * bracket_term) / 15
        return float(qKq) * normalization**2

    def _dqK_dx_1d(self, x: np.ndarray, **parameters) -> np.ndarray:
        """
        Analytical derivative of the 1D kernel integral with respect to x.
        Returns an array of derivatives.
        """
        a, b = parameters["domain"]
        lengthscale = parameters["lengthscale"]
        normalization = 1 / (b - a) if parameters["normalize"] else 1.0
        s5 = np.sqrt(5)
        first_exp = -np.exp(s5 * (x - b) / lengthscale) / (15 * lengthscale)
        first_term = first_exp * (15 * lengthscale - 15 * s5 * (x - b) + 25 / lengthscale * (x - b) ** 2)
        second_exp = -np.exp(s5 * (a - x) / lengthscale) / (15 * lengthscale)
        second_term = second_exp * (-15 * lengthscale + 15 * s5 * (a - x) - 25 / lengthscale * (a - x) ** 2)
        return (first_term + second_term) * normalization

#I have build my own wrappers to use a summation of RBF kernels and White kernels, and of Matern52 and white kernels
class SumRBFWhiteGPy(IStandardKernel):
    """
    Wrapper for a sum of GPy RBF and White kernels to use with EmuKit quadrature.

    Parameters
    ----------
    gpy_kernel : GPy kernel
        A kernel composed of an RBF and White component.
    """
    def __init__(self, gpy_kernel):
        
        gpy_rbf = gpy_kernel.parts[0]
        gpy_white = gpy_kernel.parts[1]
        self.gpy_rbf = gpy_rbf
        self.gpy_white = gpy_white
        self.gpy_kernel = gpy_rbf + gpy_white 

    @property
    def lengthscales(self) -> np.ndarray:
       """Return array of lengthscales (supports ARD)."""

       if self.gpy_rbf.ARD:
           return self.gpy_rbf.lengthscale.values
       return np.full((self.gpy_rbf.input_dim,), self.gpy_rbf.lengthscale[0])

    @property
    def variance(self) -> float:
        """Return variance of  kernel"""
        return self.gpy_rbf.variance.values[0] + self.gpy_white.variance.values[0]


    def K(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """Computes the full kernel matrix (RBF + White)."""
        return self.gpy_kernel.K(x1, x2)

    def dK_dx1(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """Compute derivative of kernel with respect to x1."""
        scaled_vector_diff = np.swapaxes((x1[None, :, :] - x2[:, None, :]) / self.lengthscales**2, 0, -1)
        return -self.K(x1, x2)[None, ...] * scaled_vector_diff

    def dKdiag_dx(self, x: np.ndarray) -> np.ndarray:
        """Derivative of diagonal kernel is zero."""
        return np.zeros((x.shape[1], x.shape[0]))

class SumMatern52WhiteGPy(IStandardKernel):
    """
    Wrapper for a sum of GPy Matern52 and White kernels to use with EmuKit quadrature.

    Parameters
    ----------
    gpy_kernel : GPy kernel
        A kernel composed of a Matern52 and White component.
    """

    def __init__(self, gpy_kernel):
        
        gpy_matern = gpy_kernel.parts[0]
        gpy_white = gpy_kernel.parts[1]
        self.gpy_matern = gpy_matern
        self.gpy_white = gpy_white
        self.gpy_kernel = gpy_matern + gpy_white 

    @property
    def lengthscales(self) -> np.ndarray:
       """Return array of lengthscales (supports ARD)."""
       if self.gpy_matern.ARD:
           return self.gpy_matern.lengthscale.values
       return np.full((self.gpy_matern.input_dim,), self.gpy_matern.lengthscale[0])

    @property
    def variance(self) -> float:
        """Returns the variance of the total"""
        return self.gpy_matern.variance.values[0] + self.gpy_white.variance.values[0]


    def K(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """Computes the full kernel matrix (RBF + White)."""
        return self.gpy_kernel.K(x1, x2)

    def dK_dx1(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """Compute derivative of kernel with respect to x1."""
        scaled_vector_diff = np.swapaxes((x1[None, :, :] - x2[:, None, :]) / self.lengthscales**2, 0, -1)
        return -self.K(x1, x2)[None, ...] * scaled_vector_diff


    def dKdiag_dx(self, x: np.ndarray) -> np.ndarray:
        """Derivative of diagonal kernel is zero."""
        return np.zeros((x.shape[1], x.shape[0]))







def write_result(name, rmsd_array, lengthscale, queries, weight_acq_fes, kernel_type, noise):
    """
    Saves the experimental settings and RMSD results to a text file.
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



def get_ground_truth( derivatives=False):
    """
      Load free energy landscape from file.
      
      Parameters
      ----------
      derivatives : bool
          If True, return gradients instead of free energy.
          
      Returns
      -------
      FES surface or derivatives as arrays.
    """
    metafile = open(r"simulations_essentials/fes.dat")
    data = np.genfromtxt(metafile)
    metafile.close()
    fes = data[:,2]
    phi=data[:,0].reshape(101,101)
    psi=data[:,1].reshape(101,101)
    dx = data[:,3].reshape(101,101)
    dy = data[:,4].reshape(101,101)
    if derivatives:
        return phi,psi,dx,dy
    return (fes - np.min(fes)).reshape(101,101)



def integration_2D_rgrid(
        grid: np.ndarray,
        dA_grid: np.ndarray,
        integrator: str = 'simpson+mini',
        fast: bool= False) -> np.ndarray:
    """
    Integrate 2D regular grid from its gradient.

    Parameters
    ----------
    grid : ndarray
        Grid coordinates of shape (n_j, n_i, 2)
    dA_grid : ndarray
        Gradient values of shape (n_j, n_i, 2)
    integrator : str
        Integration method: 'trapz', 'simpson', 'trapz+mini', 'simpson+mini', 'fourier'
    fast : bool
        Reduce minimization iterations if True.
    
    Returns
    -------
    A_grid : ndarray
        Integrated surface with minimum set to zero.
    """
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


def plot_acquisition_function(X,Y,ivr_plot,scaled_free_energy,together,i,name):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    # --- Plot 1: IVR ---
    contour1 = axes[0].contourf(X, Y, ivr_plot, levels=100, cmap="viridis")
    cbar1 = fig.colorbar(contour1, ax=axes[0], shrink=0.8, aspect=30)
    cbar1.set_label("IVR")
    axes[0].set_title(f"IVR Contour Plot, {i}")
    # --- Plot 2: Scaled Free Energy ---
    contour2 = axes[1].contourf(X, Y, scaled_free_energy, levels=100, cmap="viridis")
    cbar2 = fig.colorbar(contour2, ax=axes[1], shrink=0.8, aspect=30)
    cbar2.set_label("Scaled Free Energy")
    axes[1].set_title(f"Scaled Free Energy, {i}")
    # --- Plot 3: Together ---
    contour3 = axes[2].contourf(X, Y, together, levels=100, cmap="viridis")
    cbar3 = fig.colorbar(contour3, ax=axes[2], shrink=0.8, aspect=30)
    cbar3.set_label("Combined")
    axes[2].set_title(f"Combined Contour, {i}")
    plt.savefig("Plots/" +name + f"_acqui_{i}.png", dpi=300)
    plt.show()

def make_fes_figure(X,Y,analytical, bq_int, vmin,vmax,diff_min,diff_max,save_queries, rmsd, i,emukit_method ,name,reference_contour):
 
     fig, axes = plt.subplots(2, 2, figsize=(16, 14))

     # First plot: Ground Truth
     contour1 = axes[0, 0].contourf(X, Y, analytical, levels=100,
                                 cmap="viridis", vmin=vmin, vmax=vmax)
     axes[0, 0].set_title("Ground Truth Free Energy", fontsize=16)

     # Second plot: Prediction BQ
     contour2 = axes[0, 1].contourf(X, Y, bq_int, levels=100,
                                 cmap="viridis", vmin=vmin, vmax=vmax)
     axes[0, 1].set_title("Prediction using Bayesian Quadrature", fontsize=16)
     axes[0, 1].scatter(emukit_method.X[:, 0], emukit_method.X[:, 1], color="white", s=15)

     # Shared colorbar for first two
     cbar1 = fig.colorbar(contour1, ax=axes[0, :], shrink=0.8, location="right")
     cbar1.set_label("Free Energy (kcal/mol)", fontsize=14)
     cbar1.ax.tick_params(labelsize=12)

     diff = analytical - bq_int
     # Third plot: Difference
     contour3 = axes[1, 0].contourf(X, Y, diff, levels=100,
                                 cmap="coolwarm", vmin=diff_min, vmax=diff_max)
     axes[1, 0].set_title("Difference", fontsize=16)
     contour3.set_clim(diff_min, diff_max)
   
     cbar2 = fig.colorbar(reference_contour, ax=axes[1, 0], shrink=0.8, location="right")
     cbar2.set_label("Difference (kcal/mol)", fontsize=14)
     cbar2.set_ticks([-2.5, 0, 5, 10, 15, 20])
     cbar2.ax.tick_params(labelsize=12)



     # Fourth plot: RMSD
     axes[1, 1].plot(save_queries, rmsd, marker="o", linestyle="-")
     axes[1, 1].set_title("RMSD", fontsize=16)
     axes[1, 1].set_xlabel("Query", fontsize=14)
     axes[1, 1].set_ylabel("RMSD (kcal/mol)", fontsize=14)
     axes[1, 1].tick_params(axis="both", labelsize=12)

     # Make first three square
     for ax in [axes[0, 0], axes[0, 1], axes[1, 0]]:
         ax.set_aspect('equal', adjustable='box')
         ax.set_xlabel(r"$\phi$", fontsize=14)
         ax.set_ylabel(r"$\psi$", fontsize=14)
         ax.tick_params(axis="both", labelsize=12)

     # Title
     fig.suptitle(f"After {i} Queries", fontsize=18, fontweight="bold")

     plt.savefig("Plots/" +name + f"fes_after_{i}.png", dpi=150)
     plt.show()


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
        f.write("MOLINFO STRUCTURE=simulations_essentials/diala.pdb\n")
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
    "For running the MD simulation"
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

