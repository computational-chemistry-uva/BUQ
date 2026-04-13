

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



from mace.calculators import MACECalculator
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)
# def do_simulation(d2,d1):

#     timestep = 0.5 * units.fs
#     atoms = read('p.xyz', '0')
#     potential = MACECalculator(model_paths='MACE_2_swa.model', device='cuda')

#     # pulling rc=d2-d1 does not work, C-F bond is too strong.
#     # "steer: MOVINGRESTRAINT ARG=rc STEP0=5000 AT0=2.0 KAPPA0=50000.00 STEP1=255000 AT1=-1.0",
#     # notice we start from the lowest energy, with C bound to F.
#     bias = [f"UNITS LENGTH=A TIME=ps ENERGY=kcal/mol",
#             "d1: DISTANCE ATOMS=1,4 NOPBC",
#             "d2: DISTANCE ATOMS=1,5 NOPBC",
#             "rc: COMBINE ARG=d1,d2 COEFFICIENTS=-1,1 PERIODIC=NO",
#             f"steer: MOVINGRESTRAINT ARG=d1,d2 STEP0=1000 AT0=2.64,1.84 KAPPA0=1000.0,100.0 STEP1=5000 AT1={d1},{d2}",
#             "ener: ENERGY",
#             "an: ANGLE ATOMS=1,2,4,5 NOPBC",
#             "res: RESTRAINT ARG=an AT=pi*0.5 KAPPA=400.0",
#             "an2: ANGLE ATOMS=1,5,4 NOPBC",
#             "res2: RESTRAINT ARG=an2 AT=0.0 KAPPA=400.0",
#             f"PRINT ARG=* STRIDE=100 FILE=colvars/COLVAR_{d1}_{d2}",
#             "FLUSH STRIDE=500"]

#     atoms.calc = Plumed(calc=potential,
#                         input=bias,
#                         timestep=timestep,
#                         atoms=atoms)

#     MaxwellBoltzmannDistribution(atoms, temperature_K=300)

#     dyn = Bussi(atoms, timestep, temperature_K=300, taut=100*timestep,
#             logfile=f'colvars/log_{d1}_{d2}', loginterval=500)
#     def write_frame():
#         dyn.atoms.write(f'colvars/t_{d1}_{d2}.xyz', append=True)
#     dyn.attach(write_frame, interval=500)

#     dyn.run(80000)




#do_simulation(1.84,2.64)


def do_simulation_OPES():
    timestep = 0.5 * units.fs
    atoms = read('p.xyz', '0')
    potential = MACECalculator(model_paths='MACE_2_swa.model', device='cuda')

    bias = [
        # units
   "UNITS LENGTH=A TIME=ps",
    "d1: DISTANCE ATOMS=1,4 NOPBC",
    "d2: DISTANCE ATOMS=1,5 NOPBC",

    # OPES-Metadynamics on (d1, d2) – single line
    (
        "opes: OPES_METAD_EXPLORE "
        "ARG=d1,d2 "
        "PACE=500 "
        "BARRIER=200 "
        "TEMP=300 "
        "STATE_WFILE=STATE "
        "STORE_STATES "
        "STATE_WSTRIDE=1000 "
    ),

        # keep the restraints you had (optional, but they help keep geometry sane)
        "ener: ENERGY",
        "an: ANGLE ATOMS=1,2,4,5 NOPBC",
        "res: RESTRAINT ARG=an AT=pi*0.5 KAPPA=100.0",
        "an2: ANGLE ATOMS=1,5,4 NOPBC",
        "res2: RESTRAINT ARG=an2 AT=0.0 KAPPA=100.0",
        "lwall: LOWER_WALLS ARG=d1,d2 AT=1.2,1.8 KAPPA=600.0,600.0 EXP=2,2 EPS=1,1 OFFSET=0,0 ",
        "uwall: UPPER_WALLS ARG=d1,d2 AT=2.8,3.5 KAPPA=600.0,600.0 EXP=2,2 EPS=1,1 OFFSET=0,0 ",
        # print CVs and OPES bias for later FES reconstruction
        f"PRINT STRIDE=100 FILE=COLVAR ARG=d1,d2,opes.bias",
        "FLUSH STRIDE=500",
    ]

    atoms.calc = Plumed(
        calc=potential,
        input=bias,
        timestep=timestep,
        atoms=atoms,
    )

    MaxwellBoltzmannDistribution(atoms, temperature_K=300)

    dyn = Bussi(
        atoms,
        timestep,
        temperature_K=300,
        taut=100 * timestep,
        logfile=f'log_opes',
        loginterval=500,
    )

    def write_frame():
        dyn.atoms.write(f't.xyz', append=True)

    dyn.attach(write_frame, interval=500)
    dyn.run(4000000)



do_simulation_OPES()