import os
import subprocess
import numpy as np
from buq.systems import CollectiveVariableSystem


def run_command(command: str):
    """Run an MD command with the GROMACS environment set up."""
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = "/usr/local/gromacs/lib:" + env.get(
        "LD_LIBRARY_PATH", ""
    )
    env["PATH"] = "/usr/local/gromacs/bin:" + env.get(
        "PATH", "/bin:/usr/bin:/usr/local/bin"
    )

    try:
        subprocess.run(command, shell=True, check=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running command: {command}")
        print(e)


class Adipep(CollectiveVariableSystem):
    def __init__(
        self,
        measure_after_ps: float = 1000.0,
        kappa_phi: float = 200.0,
        kappa_psi: float = 200.0,
        nsteps: int = 50000,
    ):
        bounds = (-np.pi, np.pi, -np.pi, np.pi)  # inherent to this system
        super().__init__(dim=2, bounds=bounds)

        self.measure_after_ps = measure_after_ps
        self.kappa_phi = kappa_phi
        self.kappa_psi = kappa_psi
        self.nsteps = nsteps

    # -------- required interface methods --------

    def write_plumed_input(self, x: np.ndarray) -> None:
        """Create PLUMED input for CV point x = [phi, psi]."""
        phi, psi = x

        equisteps = 500
        moving_speed = 1000
        build_up_kappa_steps = 1000 + equisteps

        current_phi = 0.0
        current_psi = 0.0

        angles = {
            "phi": {"target": phi, "kappa": self.kappa_phi, "current": current_phi},
            "psi": {"target": psi, "kappa": self.kappa_psi, "current": current_psi},
        }

        os.makedirs("colvars", exist_ok=True)
        os.makedirs("plumed", exist_ok=True)
        os.makedirs("traj", exist_ok=True)


        filename = f"plumed/plumed_{phi:.3f}_{psi:.3f}.dat"

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
                    target_val = np.cos(info["target"]) if trig == "cos" else np.sin(
                        info["target"]
                    )
                    current_val = np.cos(info["current"]) if trig == "cos" else np.sin(
                        info["current"]
                    )
                    f.write(
                        f"restraint_{angle_name}_{trig}: MOVINGRESTRAINT ...\n"
                        f"ARG={trig}_{angle_name}\n"
                        f"STEP0={equisteps} AT0={current_val} KAPPA0=0\n"
                        f"STEP1={build_up_kappa_steps} AT1={current_val} KAPPA1={info['kappa']}\n"
                        f"STEP2={step_to} AT2={target_val} KAPPA2={info['kappa']}\n"
                        "...\n"
                    )

            f.write(
                f"PRINT ARG=sin_phi,cos_phi,sin_psi,cos_psi,*.* "
                f"FILE=colvars/COLVAR_{phi:.3f}_{psi:.3f} STRIDE=100\n"
            )

    def run_simulation(self, x: np.ndarray) -> None:
        """Called by BayesianQuadratureRunner: x = [phi, psi]."""
        phi, psi = x

        # clean old files
        run_command("rm *#*")

        # write PLUMED input
        self.write_plumed_input(x)

        # build and run MD command
        command = (
            "gmx mdrun -s simulations_essentials/md.tpr "
            + f"-plumed plumed/plumed_{phi:.3f}_{psi:.3f}.dat "
            + f"-nsteps {self.nsteps} "
            + f"-x traj/traj_{phi:.3f}_{psi:.3f}.xtc"
        )

        run_command(command)
        run_command("rm *#*")
        run_command("rm confout.gro ener.edr md.log state.cpt state_prev.cpt traj.trr")


    def get_force(self, x: np.ndarray) -> np.ndarray:
        """Return dF/dx for x = [phi, psi]."""
        phi_value, psi_value = x

        data = np.genfromtxt(f"colvars/COLVAR_{phi_value:.3f}_{psi_value:.3f}")
        # use only data after equilibration time
        data = data[data[:, 0] > self.measure_after_ps]

        # columns: sin_phi, cos_phi, sin_psi, cos_psi
        mean_vals = np.mean(data[:, 1:5], axis=0)
        sin_phi_real, cos_phi_real, sin_psi_real, cos_psi_real = mean_vals

        sin_phi_umb, cos_phi_umb = np.sin(phi_value), np.cos(phi_value)
        sin_psi_umb, cos_psi_umb = np.sin(psi_value), np.cos(psi_value)

        force_phi_vec = np.array([
            sin_phi_real - sin_phi_umb,
            cos_phi_real - cos_phi_umb,
        ]) * self.kappa_phi

        force_psi_vec = np.array([
            sin_psi_real - sin_psi_umb,
            cos_psi_real - cos_psi_umb,
        ]) * self.kappa_psi

        sign_phi = -1 if np.arctan2(sin_phi_real, cos_phi_real) < np.arctan2(
            sin_phi_umb, cos_phi_umb
        ) else 1
        sign_psi = -1 if np.arctan2(sin_psi_real, cos_psi_real) < np.arctan2(
            sin_psi_umb, cos_psi_umb
        ) else 1

        force_phi = np.linalg.norm(force_phi_vec) * sign_phi
        force_psi = np.linalg.norm(force_psi_vec) * sign_psi

        # return dF/dx (minus umbrella force)
        return np.array([-force_phi, -force_psi])