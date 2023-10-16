import numpy as np

from scipy.optimize import curve_fit
from scipy.stats import norm
import scipy.interpolate

import matplotlib as mpl
import matplotlib.pyplot as plt

import matplotlib as matplotlib
import h5py

from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, zoomed_inset_axes

import matplotlib.gridspec as gridspec
import matplotlib.ticker as tick

from copy import deepcopy, copy

import os
import sys
from tqdm import tqdm
import pickle


current = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current)
sys.path.append(parent_directory)

from power_spectrum import PowerSpectrum, CrossSpectrum
from map_cosmo import MapCosmo

from l2gen_argparser import parser

# Parsing parameter file object
params = parser.parse_args()

# Opening pickled astropy cosmology to use
cosmology_path = os.path.join(params.phy_cosmology_dir, params.phy_cosmology_name)
with open(cosmology_path, mode = "rb") as cosmofile:
    cosmology = pickle.load(cosmofile)

number_of_k_bin_edges = 15

# Map paths
simmap_path = os.path.join(params.map_dir, f"{params.fields[0]}_{params.map_name}.h5")
cubemap_path = os.path.join(params.map_dir, f"{params.fields[0]}_{params.map_name}_signal_tod.h5")

# Defining map objects
simmap = MapCosmo(params = params, mappath = simmap_path, cosmology = cosmology)
cubemap = MapCosmo(params = params, mappath = cubemap_path, cosmology = cosmology)


# Computing cross spectrum between map with injected signal and unfiltered binned signal 
XS_sim_cube = CrossSpectrum(simmap, cubemap)
xs_sim_cube, k_bin_centers, _ = XS_sim_cube.calculate_xs_2d(number_of_k_bin_edges = number_of_k_bin_edges)

# Computing auto-spectrum of unfiltered and binned signal
PS_cube = PowerSpectrum(cubemap)
ps_pure_signal, k_bin_centers, _ = PS_cube.calculate_ps(do_2d=True, number_of_k_bin_edges = number_of_k_bin_edges)    

# Compute transfer function
transfer_function = xs_sim_cube / ps_pure_signal

# Extracting k-bin edges and centers
k_bin_edges_par = PS_cube.k_bin_edges_par
k_bin_edges_perp = PS_cube.k_bin_edges_perp
k_perp, k_par = k_bin_centers

# Computing cross spectrum between map with injected signal and unfiltered binned signal 
xs_sim_cube_1d, k_bin_centers_1d, _ = XS_sim_cube.calculate_xs(number_of_k_bin_edges = number_of_k_bin_edges)

# Computing auto-spectrum of unfiltered and binned signal
ps_pure_signal_1d, k_bin_centers_1d, _ = PS_cube.calculate_ps(do_2d=False, number_of_k_bin_edges = number_of_k_bin_edges)    
transfer_function_1d = xs_sim_cube_1d / ps_pure_signal_1d

k_bin_edges_1d = PS_cube.k_bin_edges

# Saving transfer funtion to file
transfer_function_dir = params.transfer_function_dir
transfer_function_name = params.transfer_function_name
transfer_function_path = os.path.join(transfer_function_dir, transfer_function_name)

with h5py.File(transfer_function_path, "w") as outfile:
    outfile.create_group("cylindrically_averaged")
    outfile.create_dataset("cylindrically_averaged/k_bin_centers_perp", data = k_perp)
    outfile.create_dataset("cylindrically_averaged/k_bin_centers_par", data = k_par)
    outfile.create_dataset("cylindrically_averaged/k_bin_edges_perp", data = k_bin_edges_perp)
    outfile.create_dataset("cylindrically_averaged/k_bin_edges_par", data = k_bin_edges_par)
    outfile.create_dataset("cylindrically_averaged/transfer_function", data = transfer_function)

    outfile.create_group("spherically_averaged")
    outfile.create_dataset("spherically_averaged/transfer_function", data = transfer_function_1d)
    outfile.create_dataset("spherically_averaged/k_bin_centers", data = k_bin_centers_1d)
    outfile.create_dataset("spherically_averaged/k_bin_edges", data = k_bin_edges_1d)
    
    