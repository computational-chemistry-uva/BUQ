import numpy as np
import GPy
import csv
import os

from emukit.quadrature.methods import VanillaBayesianQuadrature
from emukit.model_wrappers.gpy_quadrature_wrappers import  BaseGaussianProcessGPy
from emukit.quadrature.kernels import QuadratureRBFLebesgueMeasure
from emukit.quadrature.measures import LebesgueMeasure
from emukit.core.optimization import GradientAcquisitionOptimizer
from emukit.core.parameter_space import ParameterSpace
import emukit.quadrature.acquisitions as emu_acqui
from helper_functions import QuadratureProductMatern52LebesgueMeasure, SumRBFWhiteGPy,SumMatern52WhiteGPy,write_result,get_ground_truth,integration_2D_rgrid,make_fes_figure,plot_acquisition_function, write_plumed_file, run_command
import matplotlib.pyplot as plt



def get_force(phi_value, psi_value, kappa_phi=200, kappa_psi=200, measure_after_ps=1000):
    """
    Gets the force after doing a restraint md simulation
    
    Returns array: [dF/dphi, dF/dpsi]
    """
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



# Optional: if running with Slurm arrays, uncomment
# task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))


precommand = "srun --mpi=pmix_v4 gmx_mpi mdrun -s simulations_essentials/md.tpr " #for running on a cluster
precommand = "gmx mdrun  -s simulations_essentials/md.tpr "


#This can be used to turn the parameters - see the readme for more clarification
task_id = 0 #comment this out to use the array-job functionality
with open("params.csv", "r") as f:
    reader = csv.DictReader(f)
    rows = list(reader)
row = rows[task_id]
lengthscale = (float(row["lengthscale"]))
acq_function = row["acq_function"]
queries = 3# int(row["queries"])
weight_acq_fes = float(row["weight_acq_fes"])
noise = 0.2# float(row["noise"])
kernel_type = row["kernel_type"]
full =  True #%%row["full"].strip().lower() == "true"  # Convert "True"/"False" to boolean


#Integration will be a bit faster (but less iterations, so less accurate!)
fast_integration=True
os.makedirs("Colvars", exist_ok=True) #to create a directory Colvars
os.makedirs("Plots", exist_ok=True) #to create a directory Colvars

# Set up grid and interpolate derivatives
derivatives = []  # Will store derivatives at sampled points
rmsd = []  # Track RMSD per iteration
save_queries = []  # Track query indices


analytical = get_ground_truth().T  # Reference free energy
phi_metadynamics, psi_metadynamics, x_derivative, y_derivative = get_ground_truth(derivatives=True)


lb_1, ub_1 = -np.pi, np.pi  # Grid bounds for phi
lb_2, ub_2 = -np.pi, np.pi  # Grid bounds for psi


#Simulation setup
ns = 0.2
kappa =200
measure_after_ps = 100 #this is above the maximum steering time, which is around 30 ps (2*pi * 1000 + 1500)*0.002 = 30 ps
nsteps = 500000 * ns 
nsteps = int(nsteps)  

x_grid = np.unique(phi_metadynamics)
y_grid = np.unique(psi_metadynamics)

#The values of psi and phi at the start of the md simulation, so we steer from there. 
current_psi = 1.311767 
current_phi = -1.580543

# Create meshgrid for plotting and integration
X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
XY_combined = np.stack((Y,X),axis=-1)
X_flat = np.vstack([X.ravel(), Y.ravel()]).T


weight_acq_ivr = 1.0 - weight_acq_fes  # Complementary weight for IVR acquisition
vmin, vmax = analytical.min(), analytical.max()  # For color scaling plots
diff_min, diff_max = -2.5,20  #  For color scaling plots

          

#Initial positions to sample
initial_positions_phi = np.array([ -1.507964473999999999e+00 , 8.796459429999999857e-01])
initial_positions_psi = np.array([1.193805207999999896e+00,  -8.796459429999999857e-01 ])


# Construct experiment name and save to file
name = f"AD_{kernel_type}_ls_{lengthscale}_w_{weight_acq_fes}_n{noise}_acq_{acq_function}_q_{queries}"
with open('run_name.txt', 'w') as f:
    f.write(name) #I use this for running on the cluster, making sure this run was with the right parameters



# Evaluate derivatives at initial positions
for init_phi, init_psi in zip(initial_positions_phi, initial_positions_psi):
    write_plumed_file(init_phi, init_psi, kappa_phi=kappa, kappa_psi=kappa, current_phi=current_phi, current_psi=current_psi)
    run_command("rm *#*") #remove old colvars etc
    command = precommand +f"-plumed Colvars/plumed_{init_phi:.3f}_{init_psi:.3f}.dat -nsteps {nsteps} -x Colvars/traj_{init_phi:.3f}_{init_psi:.3f}.xtc"
    run_command(command)
    derivatives.append(get_force(init_phi, init_psi, measure_after_ps=measure_after_ps))

# Stack initial positions and forces    
X_data = np.column_stack((initial_positions_phi, initial_positions_psi))
force_data = np.array(derivatives).reshape(-1, 2)


# -------------------- Kernel Setup --------------------
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


# Wrap GPy kernel in EmuKit Gaussian Process

emukit_model = BaseGaussianProcessGPy(kern=emukit_qrbf, gpy_model=gpy_model)
emukit_method = VanillaBayesianQuadrature(base_gp=emukit_model, X=X_data, Y=force_data)

# -------------------- Acquisition Function --------------------
if acq_function == "IVR":
    acquisition = emu_acqui.IntegralVarianceReduction(emukit_method)
elif acq_function == "US":
    acquisition = emu_acqui.UncertaintySampling(emukit_method)

# Set up parameter space and optimizer for acquisition
space = ParameterSpace(emukit_method.reasonable_box_bounds.convert_to_list_of_continuous_parameters())
optimizer = GradientAcquisitionOptimizer(space)


# Predict initial derivatives on grid
predicted_derivatives, _ = emukit_method.predict(X_flat)
predicted_derivatives = predicted_derivatives.reshape(X.shape[0], Y.shape[1], 2)
derivative_xy_combined = np.stack((predicted_derivatives[:, :, 1],predicted_derivatives[:, :, 0]),axis=-1)
bq_int = integration_2D_rgrid(XY_combined,derivative_xy_combined, "simpson+mini",fast=fast_integration)

      

# Save initial dataset to file
with open(name +"all_data.dat", "w") as f:
    for i in range(len(emukit_method.X)):
        f.write(f"{i+1} \t {emukit_method.X[i][0]} \t {emukit_method.X[i][1]} \t {emukit_method.Y[i][0]} \t {emukit_method.Y[i][1]}  \n")
    
    
# -------------------- Bayesian Quadrature Loop --------------------
for i in range(1,queries+1): #10
    print(f" -------------- BaysOpt loop, query {i} ---------------")
    # Evaluate IVR on grid
    
    ivr_plot = acquisition.evaluate(X_flat)
    ivr_plot = ivr_plot.reshape(X.shape) 
    ivr_plot = ivr_plot/ np.max(ivr_plot)
    scaled_free_energy = bq_int/np.max(bq_int)
    together = -weight_acq_fes* scaled_free_energy + weight_acq_ivr*ivr_plot #IVR function + sampling in the minima
    
    # Plot intermediate results if full mode enabled

    if full:
        plot_acquisition_function(X,Y,ivr_plot,scaled_free_energy,together,i,name)

    # Determine next query point
    max_index_together = np.unravel_index(np.argmax(together), together.shape) #alpha kan hier weer bij
    new_x_ivr = X[max_index_together]
    new_y_ivr = Y[max_index_together]   

    # Evaluate force at new point
    print(f"going to run a simulation at {new_x_ivr} {new_y_ivr} ")     
    write_plumed_file(new_x_ivr, new_y_ivr,kappa_phi=kappa, kappa_psi=kappa, current_phi=current_phi, current_psi=current_psi)
    run_command("rm *#*")
    command = precommand +"-plumed  Colvars/plumed_{:.3f}_{:.3f}.dat -nsteps {} -x Colvars/traj_{:.3f}_{:.3f}.xtc".format(new_x_ivr, new_y_ivr,nsteps,new_x_ivr, new_y_ivr)
    run_command(command)
    force_x, force_y = get_force(new_x_ivr, new_y_ivr,kappa_phi= kappa, kappa_psi= kappa, measure_after_ps= measure_after_ps)    
    force_xy =np.array([force_x,force_y])
    xy_new = np.array([[new_x_ivr,new_y_ivr]])
    
    # Update dataset
    X_data = np.append(X_data, xy_new, axis=0)
    force_data = np.vstack([force_data, force_xy])
    emukit_method.set_data(X_data, force_data)

    # Append new data to file
    with open(name + "all_data.dat", "a") as f:  # 'a' mode to append
        f.write(f"{i} \t {emukit_method.X[-1][0]} \t {emukit_method.X[-1][1]}   \t {emukit_method.Y[-1][0]} \t {emukit_method.Y[-1][1]} \n")

    # Predict derivatives and re-integrate surface

    predicted_derivatives, _ = emukit_method.predict(X_flat)
    predicted_derivatives = predicted_derivatives.reshape(X.shape[0], Y.shape[1], 2)
    XY_combined = np.stack((Y,X),axis=-1)
    derivative_xy_combined = np.stack((predicted_derivatives[:, :, 1],predicted_derivatives[:, :, 0]),axis=-1)
    bq_int = integration_2D_rgrid(XY_combined,derivative_xy_combined, "simpson+mini",fast=fast_integration)
    
    # Compute RMSD against analytical surface

    rmsd_query = np.sqrt(np.mean((analytical - bq_int) ** 2))
    rmsd.append(rmsd_query)
    save_queries.append(i)
    
      
    if i == 1:
        diff = analytical - bq_int
        reference_contour = plt.contourf(X, Y, diff, levels=100,
                                    cmap="coolwarm", vmin=diff_min, vmax=diff_max)
    # Plot intermediate results if full mode enabled
    if full:
       make_fes_figure(X,Y,analytical, bq_int, vmin,vmax,diff_min,diff_max,save_queries, rmsd, i,emukit_method ,name,reference_contour)


# -------------------- Final Figure --------------------
predicted_derivatives, _ = emukit_method.predict(X_flat)
predicted_derivatives = predicted_derivatives.reshape(X.shape[0], Y.shape[1], 2)
XY_combined = np.stack((Y,X),axis=-1)
derivative_xy_combined = np.stack((predicted_derivatives[:, :, 1],predicted_derivatives[:, :, 0]),axis=-1)
bq_int = integration_2D_rgrid(XY_combined,derivative_xy_combined, "simpson+mini",fast=fast_integration)


make_fes_figure(X,Y,analytical, bq_int, vmin,vmax,diff_min,diff_max,save_queries, rmsd, i,emukit_method ,name,reference_contour)
rmsd_arr= np.array(rmsd)
write_result(name,rmsd_arr, lengthscale, queries, weight_acq_fes,kernel_type, noise)













