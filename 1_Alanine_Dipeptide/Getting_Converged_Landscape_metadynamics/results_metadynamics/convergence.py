#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 13:17:06 2025

@author: eline
"""
import matplotlib.pyplot as plt

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
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import RegularGridInterpolator
from numpy import ndarray
from scipy import optimize as scipy_optimize

import sys
import emukit.model_wrappers.gpy_quadrature_wrappers as emuwrap

import os
import subprocess
import glob




#%%%
path = "fes/"


import numpy as np
import matplotlib.pyplot as plt

def do_conf_analysis(block_size):
    num_files = 50000  # Adjust as needed
    amount_of_ns = block_size / 500
    sample_file = path + "0.dat"
    sample_data = np.genfromtxt(sample_file)[:, 2]
    fes_length = len(sample_data)

    # Initialize buffers
    rolling_buffer = np.zeros((block_size, fes_length))
    block_averaged_free_energy = np.zeros((num_files - block_size, fes_length))

    # Preload first block
    for i in range(block_size):
        try:
            rolling_buffer[i] = np.genfromtxt(path + f"{i}.dat")[:, 2]
        except Exception as e:
            print(f"Warning: Could not load {i}.dat - {e}")

    block_averaged_free_energy[0] = np.mean(rolling_buffer, axis=0)

    for i in range(1, num_files - block_size):
        try:
            new_data = np.genfromtxt(path + f"{i + block_size - 1}.dat")[:, 2]
            rolling_buffer[i % block_size] = new_data
            block_avg = np.mean(rolling_buffer, axis=0)
            block_averaged_free_energy[i] = block_avg
        except Exception as e:
            print(f"Warning: Could not load {i + block_size - 1}.dat - {e}")

        if i % 1000 == 0:
            print(f"Processed {i}/{num_files}")

    # Set reference block and align everything to its minimum
    reference_block = block_averaged_free_energy[-1]
    min_index = reference_block.argmin()
    reference_block -= reference_block[min_index]

    for i in range(num_files - block_size):
        block_averaged_free_energy[i] -= block_averaged_free_energy[i][min_index]

    rmsd = np.sqrt(np.mean((block_averaged_free_energy - reference_block) ** 2, axis=1))

    # Plot RMSD over time
    plt.figure(figsize=(10, 5))
    plt.plot(rmsd, label="RMSD vs Last Block", marker="o", markersize=2, linestyle="-")
    plt.title(f"RMSD of Block-Averaged Free Energy vs Final Block, blocksize {amount_of_ns} ns")
    plt.xlabel("dropped kernel")
    plt.ylabel("RMSD")
    plt.legend()
    plt.grid(True)
    plt.show()


    plt.suptitle(f"Final RMSD (should be 0): {rmsd[-1]}")
    plt.tight_layout()
    plt.show()

    print(f"Last block of {block_size} kernels - so {amount_of_ns} ns")

    return reference_block
    
    
block_sizes = [500,1000]
blocks = []
for i in block_sizes:
    temp =  do_conf_analysis(i)   
    blocks.append(temp)

def get_reference_block(block_size, path):
    """
    Averages the last `block_size` .dat files in `path`, assuming each file has 5 columns:
    phi, psi, fes, dF/dphi, dF/dpsi
    """
    num_files = 50000  # total number of .dat files
    start_index = num_files - block_size
    accumulated = None

    for i, file_idx in enumerate(range(start_index, num_files)):
        file_path = os.path.join(path, f"{file_idx}.dat")
        try:
            data = np.genfromtxt(file_path)[:, 0:5]  # phi, psi, fes, dF/dphi, dF/dpsi
            if accumulated is None:
                accumulated = np.zeros_like(data)
            accumulated += data
        except Exception as e:
            print(f"Warning: Could not load {file_path} - {e}")

    reference_block = accumulated / block_size

    # Normalize FES to have minimum 0
    reference_block[:, 2] -= reference_block[:, 2].min()

    return reference_block



last_block = get_reference_block(1000,"fes/")

np.savetxt("fes_final_2.dat", last_block, header="phi psi FES dF/dphi dF/dpsi")
