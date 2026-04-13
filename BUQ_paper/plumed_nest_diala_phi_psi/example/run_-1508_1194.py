import numpy as np
import subprocess
import os



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

current_psi = 1.311767
current_phi = -1.580543
init_phi = -1.508
init_psi = 1.194
nsteps = 100000

write_plumed_file(init_phi, init_psi, current_phi=current_phi, current_psi=current_psi)
command = f"srun --mpi=pmix_v4 gmx_mpi mdrun -s md.tpr -plumed Colvars/plumed_{init_phi:.3f}_{init_psi:.3f}.dat -nsteps {nsteps} -x Colvars/traj_{init_phi:.3f}_{init_psi:.3f}.xtc"
run_command(command)

get_force(init_phi, init_psi)