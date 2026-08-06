import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from matplotlib import ticker
from matplotlib.patches import ConnectionPatch
import scipy.stats

import h5py
import matplotlib
import os
from scipy.stats import norm, chi2, ecdf
import re
from scipy.interpolate import RectBivariateSpline, CubicSpline
import copy

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

single_width = 8
double_width = 16
two_thirds_width = double_width * 2 / 3 

fontsize = 14


### Models ###
modelpath = "/mn/stornext/d16/cmbco/comap/defaults/models"
Pullen13B = {}
Pad18fduty = {}
COMAPESV = {}
L16K20 = {}
Pad18 = {}
Li16 = {}

flat_coldz = {}
flat_coldz_percentile = {}

with np.load(os.path.join(modelpath, "Pk_arrays_Pkmu_Pullen13B.npz")) as infile:
    for key, value in infile.items():
        Pullen13B[key] = value
        
        
with np.load(os.path.join(modelpath, "Pk_arrays_Pkmu_Pad18fduty.npz")) as infile:
    for key, value in infile.items():
        Pad18fduty[key] = value
        
        
with np.load(os.path.join(modelpath, "Pk_arrays_Pkmu_COMAPESV.npz")) as infile:
    for key, value in infile.items():
        COMAPESV[key] = value
        
        
with np.load(os.path.join(modelpath, "Pk_arrays_Pkmu_L16K20.npz")) as infile:
    for key, value in infile.items():
        L16K20[key] = value
        
        
with np.load(os.path.join(modelpath, "Pk_arrays_Pkmu_Pad18.npz")) as infile:
    for key, value in infile.items():
        Pad18[key] = value
        
        
with np.load(os.path.join(modelpath, "Pk_arrays_Pkmu_Li16.npz")) as infile:
    for key, value in infile.items():
        Li16[key] = value

with np.load(os.path.join(modelpath, "20210524_mcmcsummary_flat+COLDz_pspec.npz")) as infile:
    for key, value in infile.items():
        flat_coldz[key] = value

with np.load(os.path.join(modelpath, "20210524_mcmcsummary_flat+COLDz_pspec_percentiles.npz")) as infile:
    for key, value in infile.items():
        flat_coldz_percentile[key] = value

### COPSS ###
COPSS_mined_data = [
    (0.455,-265*2*np.pi**2/0.455**3,574*2*np.pi**2/0.455**3),
    (0.5728811283775609, 8409.090909090908, 18295.454545454544),
    (0.721214610451828, -9659.090909090912, 6704.545454545456),
    (0.9079554004550374, 8409.090909090908, 3579.545454545456),
    (1.143048126408596, 6477.272727272724, 2556.818181818184),
    (1.4390123332394924, 1250, 2499.999999999998),
    (1.8116091942004138, 284.09090909091174, 3238.636363636364),
    (2.297684119307595, 454.5454545454486, 5056.818181818184),
    (2.8926129260722333, 5454.545454545456, 11250.000000000002),
    (4.57, 5.57e5*2*np.pi**2/4.57**3, 3.5e5*2*np.pi**2/4.57**3),
    (5.75, -4.08e5*2*np.pi**2/5.75**3, 2.3e5*2*np.pi**2/5.75**3),
    (7.29, -6970, 17000),
(9.12, 1.42e6*2*np.pi**2/9.12**3, 8.5e5*2*np.pi**2/9.12**3),
]

COPSS_k, COPSS_Pk, COPSS_errPk = np.array(tuple(zip(*COPSS_mined_data)));
COPSS_h = 0.7;

COPSS_k *= COPSS_h;
COPSS_Pk /= COPSS_h **3;
COPSS_errPk /= COPSS_h **3;

COPSS_deltak = np.median(np.diff(np.log10(COPSS_k)))

COPSS_k_edges = np.arange(np.log10(COPSS_k.min()) - COPSS_deltak / 2, np.log10(COPSS_k.max()) + COPSS_deltak / 2, COPSS_deltak)
COPSS_k_edges = 10 ** COPSS_k_edges
_COPSS_k_edges = COPSS_k_edges.copy()
COPSS_k_edges = COPSS_k_edges[np.arange(15) != 9]



paths = {
    "Field 1 S2b": "/mn/stornext/d16/cmbco/comap/defaults/S2/paper_version/power_spectrum/average_spectra_saddlebag/co2_may22-nov23_v5_take6_n5_subtr_exper_average_fpxs.h5",
    "Field 2 S2b": "/mn/stornext/d16/cmbco/comap/defaults/S2/paper_version/power_spectrum/average_spectra_saddlebag/co7_may22-nov23_v5_take6_n5_subtr_exper_average_fpxs.h5",
    "Field 3 S2b": "/mn/stornext/d16/cmbco/comap/defaults/S2/paper_version/power_spectrum/average_spectra_saddlebag/co6_may22-nov23_v5_take6_n5_subtr_exper_average_fpxs.h5",
    "Field 1 S1+S2a": "/mn/stornext/d16/cmbco/comap/defaults/S2/paper_version/power_spectrum/average_spectra_saddlebag/co2_apr22_v5_take6_n5_subtr_exper_average_fpxs.h5",
    "Field 2 S1+S2a": "/mn/stornext/d16/cmbco/comap/defaults/S2/paper_version/power_spectrum/average_spectra_saddlebag/co7_apr22_v5_take6_n5_subtr_exper_average_fpxs.h5",
    "Field 3 S1+S2a": "/mn/stornext/d16/cmbco/comap/defaults/S2/paper_version/power_spectrum/average_spectra_saddlebag/co6_apr22_v5_take6_n5_subtr_exper_average_fpxs.h5",
}


subseason_names = ["Field 1 slow-az", "Field 2 slow-az", "Field 3 slow-az", "Field 1 fast-az", "Field 2 fast-az", "Field 3 fast-az"]

#data_selection = ""
#data_selection = "wo_0.2Mpcmask"
#data_selection = "wo_0.2Mpcmask_5perprow"

paths_s3 = {

    #### v11 va9 ###
    "Field 1 S2b+S3": "/mn/stornext/d16/cmbco/comap/data/power_spectrum/v11va9/average_spectra_saddlebag/co2_may22-mar26_v11va9_n5_subtr_exper/co2_may22-mar26_v11va9_n5_subtr_exper_average_fpxs.h5",
    "Field 2 S2b+S3": "/mn/stornext/d16/cmbco/comap/data/power_spectrum/v11va9/average_spectra_saddlebag/co7_may22-mar26_v11va9_n5_subtr_exper/co7_may22-mar26_v11va9_n5_subtr_exper_average_fpxs.h5",
    "Field 3 S2b+S3": "/mn/stornext/d16/cmbco/comap/data/power_spectrum/v11va9/average_spectra_saddlebag/co6_may22-mar26_v11va9_n5_subtr_exper/co6_may22-mar26_v11va9_n5_subtr_exper_average_fpxs.h5",

    "Field 1 S1+S2a": "/mn/stornext/d16/cmbco/comap/data/power_spectrum/v11va9/average_spectra_saddlebag/co2_apr22_v11va9_n5_subtr_exper/co2_apr22_v11va9_n5_subtr_exper_average_fpxs.h5",
    "Field 2 S1+S2a": "/mn/stornext/d16/cmbco/comap/data/power_spectrum/v11va9/average_spectra_saddlebag/co7_apr22_v11va9_n5_subtr_exper/co7_apr22_v11va9_n5_subtr_exper_average_fpxs.h5",
    "Field 3 S1+S2a": "/mn/stornext/d16/cmbco/comap/data/power_spectrum/v11va9/average_spectra_saddlebag/co6_apr22_v11va9_n5_subtr_exper/co6_apr22_v11va9_n5_subtr_exper_average_fpxs.h5",

    ### v11 va5 ###

    # "Field 1 S2b+S3": "/mn/stornext/d16/cmbco/comap/data/power_spectrum/v11va5/average_spectra_saddlebag/co2_may22-mar26_v11va5_n5_subtr_exper/co2_may22-mar26_v11va5_n5_subtr_exper_average_fpxs.h5",
    # "Field 2 S2b+S3": "/mn/stornext/d16/cmbco/comap/data/power_spectrum/v11va5/average_spectra_saddlebag/co7_may22-mar26_v11va5_n5_subtr_exper/co7_may22-mar26_v11va5_n5_subtr_exper_average_fpxs.h5",
    # "Field 3 S2b+S3": "/mn/stornext/d16/cmbco/comap/data/power_spectrum/v11va5/average_spectra_saddlebag/co6_may22-mar26_v11va5_n5_subtr_exper/co6_may22-mar26_v11va5_n5_subtr_exper_average_fpxs.h5",

    # "Field 1 S1+S2a": "/mn/stornext/d16/cmbco/comap/data/power_spectrum/v11va5/average_spectra_saddlebag/co2_apr22_v11va5_n5_subtr_exper/co2_apr22_v11va5_n5_subtr_exper_average_fpxs.h5",
    # "Field 2 S1+S2a": "/mn/stornext/d16/cmbco/comap/data/power_spectrum/v11va5/average_spectra_saddlebag/co7_apr22_v11va5_n5_subtr_exper/co7_apr22_v11va5_n5_subtr_exper_average_fpxs.h5",
    # "Field 3 S1+S2a": "/mn/stornext/d16/cmbco/comap/data/power_spectrum/v11va5/average_spectra_saddlebag/co6_apr22_v11va5_n5_subtr_exper/co6_apr22_v11va5_n5_subtr_exper_average_fpxs.h5",
    
}

paths_s3_coadd = {
    "Field 1 S1-S3": "/mn/stornext/d16/cmbco/comap/data/power_spectrum/v11va9/average_spectra_saddlebag/co2_mar26coadd_v11va9_n5_subtr_exper/co2_mar26coadd_v11va9_n5_subtr_exper_average_fpxs.h5",
    "Field 2 S1-S3": "/mn/stornext/d16/cmbco/comap/data/power_spectrum/v11va9/average_spectra_saddlebag/co7_mar26coadd_v11va9_n5_subtr_exper/co7_mar26coadd_v11va9_n5_subtr_exper_average_fpxs.h5",
    "Field 3 S1-S3": "/mn/stornext/d16/cmbco/comap/data/power_spectrum/v11va9/average_spectra_saddlebag/co6_mar26coadd_v11va9_n5_subtr_exper/co6_mar26coadd_v11va9_n5_subtr_exper_average_fpxs.h5",
}

include_coadd = True

data_selection = paths_s3["Field 1 S2b+S3"].split("/")[8][3:]
print(data_selection)

def coadd_spectra(path):
    
    xs_mean_1d = np.zeros((len(path), 14))
    xs_error_1d = np.zeros((len(path), 14))
    fields = []   
    for i, (key, f) in enumerate(path.items()):
        fields.append(key[:7])
        with h5py.File(f, "r") as infile:
            k_centers_par = infile["k_centers_par"][()]
            k_centers_perp = infile["k_centers_perp"][()]
            k_1d = infile["k_1d"][()]
            k_edges_1d = infile["k_edges_1d"][()]
            cross_variable_names = infile["cross_variable_names"][()].astype(str)
            elev_idx = np.where(cross_variable_names == "elev")[0][0]
            xs_mean_1d[i, :] = infile["xs_mean_1d"][elev_idx, :]
            xs_error_1d[i, :] = infile["xs_sigma_1d"][elev_idx, :]
    
    fields = np.array(fields)
    
    
    inv_var_1d = 1 / xs_error_1d ** 2
    xs_mean_1d_coadd = xs_mean_1d * inv_var_1d
    mask = np.isfinite(xs_mean_1d_coadd)
    inv_var_1d[~mask] = 0
    xs_mean_1d_coadd[~mask] = 0
    
    xs_error_1d_coadd = np.nansum(inv_var_1d, axis = 0)
    xs_mean_1d_coadd = np.nansum(xs_mean_1d_coadd, axis = 0) / xs_error_1d_coadd
    xs_error_1d_coadd = 1 / np.sqrt(xs_error_1d_coadd)
    
    
    inv_var_1d_co2 = 1 / xs_error_1d ** 2
    xs_mean_1d_coadd_co2 = xs_mean_1d * inv_var_1d
    mask_co2 = np.isfinite(xs_mean_1d_coadd_co2)
    inv_var_1d_co2[~mask] = 0
    xs_mean_1d_coadd_co2[~mask] = 0
    
    
    co2_mask = np.where(fields == "Field 1")[0]
    
    xs_error_1d_coadd_co2 = np.nansum(inv_var_1d_co2[co2_mask], axis = 0)
    xs_mean_1d_coadd_co2 = np.nansum(xs_mean_1d_coadd_co2[co2_mask], axis = 0) / xs_error_1d_coadd_co2
    xs_error_1d_coadd_co2 = 1 / np.sqrt(xs_error_1d_coadd_co2)
    
    
    inv_var_1d_co7 = 1 / xs_error_1d ** 2
    xs_mean_1d_coadd_co7 = xs_mean_1d * inv_var_1d
    mask_co7 = np.isfinite(xs_mean_1d_coadd_co7)
    inv_var_1d_co7[~mask] = 0
    xs_mean_1d_coadd_co7[~mask] = 0
    
    co7_mask = np.where(fields == "Field 2")[0]
    
    xs_error_1d_coadd_co7 = np.nansum(inv_var_1d_co7[co7_mask], axis = 0)
    xs_mean_1d_coadd_co7 = np.nansum(xs_mean_1d_coadd_co7[co7_mask], axis = 0) / xs_error_1d_coadd_co7
    xs_error_1d_coadd_co7 = 1 / np.sqrt(xs_error_1d_coadd_co7)
    
    inv_var_1d_co6 = 1 / xs_error_1d ** 2
    xs_mean_1d_coadd_co6 = xs_mean_1d * inv_var_1d
    mask_co6 = np.isfinite(xs_mean_1d_coadd_co6)
    inv_var_1d_co6[~mask] = 0
    xs_mean_1d_coadd_co6[~mask] = 0
    
    co6_mask = np.where(fields == "Field 3")[0]
    
    xs_error_1d_coadd_co6 = np.nansum(inv_var_1d_co6[co6_mask], axis = 0)
    xs_mean_1d_coadd_co6 = np.nansum(xs_mean_1d_coadd_co6[co6_mask], axis = 0) / xs_error_1d_coadd_co6
    xs_error_1d_coadd_co6 = 1 / np.sqrt(xs_error_1d_coadd_co6)

    xs_mean_1d_coadd_per_field, xs_error_1d_coadd_per_field = np.array([xs_mean_1d_coadd_co2, xs_mean_1d_coadd_co7, xs_mean_1d_coadd_co6]), np.array([xs_error_1d_coadd_co2, xs_error_1d_coadd_co7, xs_error_1d_coadd_co6])
    
    
    return k_1d, k_edges_1d, xs_mean_1d_coadd, xs_error_1d_coadd, xs_mean_1d, xs_error_1d, xs_mean_1d_coadd_per_field, xs_error_1d_coadd_per_field


        
### Season 2 ###
k_1d, k_edges_1d, xs_mean_1d_coadd, xs_error_1d_coadd, xs_mean_1d, xs_error_1d, xs_mean_1d_coadd_per_field, xs_error_1d_coadd_per_field = coadd_spectra(paths)

mask = np.isfinite(xs_mean_1d_coadd)

k_1d, dk_edges_1d, xs_mean_1d_coadd, xs_error_1d_coadd, xs_mean_1d, xs_error_1d, xs_mean_1d_coadd_per_field, xs_error_1d_coadd_per_field = k_1d[mask], np.diff(k_edges_1d)[mask], xs_mean_1d_coadd[mask], xs_error_1d_coadd[mask], xs_mean_1d[:, mask], xs_error_1d[:, mask], xs_mean_1d_coadd_per_field[:, mask], xs_error_1d_coadd_per_field[:, mask]


### Season 3 ###
k_1d_s3, k_edges_1d_s3, xs_mean_1d_coadd_s3, xs_error_1d_coadd_s3, xs_mean_1d_s3, xs_error_1d_s3, xs_mean_1d_coadd_per_field_s3, xs_error_1d_coadd_per_field_s3 = coadd_spectra(paths_s3)

mask_s3 = np.isfinite(xs_mean_1d_coadd_s3)

k_1d_s3, dk_edges_1d_s3, xs_mean_1d_coadd_s3, xs_error_1d_coadd_s3, xs_mean_1d_s3, xs_error_1d_s3, xs_mean_1d_coadd_per_field_s3, xs_error_1d_coadd_per_field_s3 = k_1d_s3[mask_s3], np.diff(k_edges_1d_s3)[mask_s3], xs_mean_1d_coadd_s3[mask_s3], xs_error_1d_coadd_s3[mask_s3], xs_mean_1d_s3[:, mask_s3], xs_error_1d_s3[:, mask_s3], xs_mean_1d_coadd_per_field_s3[:, mask_s3], xs_error_1d_coadd_per_field_s3[:, mask_s3]

### Season 3 map coadded ###
k_1d_s3_coadd, k_edges_1d_s3_coadd, xs_mean_1d_coadd_s3_coadd, xs_error_1d_coadd_s3_coadd, xs_mean_1d_s3_coadd, xs_error_1d_s3_coadd, xs_mean_1d_coadd_per_field_s3_coadd, xs_error_1d_coadd_per_field_s3_coadd = coadd_spectra(paths_s3_coadd)

mask_s3_coadd = np.isfinite(xs_mean_1d_coadd_s3_coadd)

k_1d_s3_coadd, dk_edges_1d_s3_coadd, xs_mean_1d_coadd_s3_coadd, xs_error_1d_coadd_s3_coadd, xs_mean_1d_s3_coadd, xs_error_1d_s3_coadd, xs_mean_1d_coadd_per_field_s3_coadd, xs_error_1d_coadd_per_field_s3_coadd = k_1d_s3_coadd[mask_s3_coadd], np.diff(k_edges_1d_s3_coadd)[mask_s3_coadd], xs_mean_1d_coadd_s3_coadd[mask_s3_coadd], xs_error_1d_coadd_s3_coadd[mask_s3_coadd], xs_mean_1d_s3_coadd[:, mask_s3_coadd], xs_error_1d_s3_coadd[:, mask_s3_coadd], xs_mean_1d_coadd_per_field_s3_coadd[:, mask_s3_coadd], xs_error_1d_coadd_per_field_s3_coadd[:, mask_s3_coadd]



with h5py.File(f"/mn/stornext/d16/cmbco/comap/nils/data/power_spectra/current_power_spectrum_results_s1-s3_v11{data_selection}.h5", "w") as infile:
#with h5py.File("/mn/stornext/d5/data/nilsoles/comap/data/power_spectrum/current_power_spectrum_results_s1-s3_v11va8.h5", "w") as infile:
    infile["co2/saddlebag-saddlebag/power_spectrum"] = xs_mean_1d_coadd_per_field_s3[0, :-1]
    infile["co2/saddlebag-saddlebag/power_spectrum_error"] = xs_error_1d_coadd_per_field_s3[0, :-1]
    
    infile["co7/saddlebag-saddlebag/power_spectrum"] = xs_mean_1d_coadd_per_field_s3[1, :-1]
    infile["co7/saddlebag-saddlebag/power_spectrum_error"] = xs_error_1d_coadd_per_field_s3[1, :-1]
    
    infile["co6/saddlebag-saddlebag/power_spectrum"] = xs_mean_1d_coadd_per_field_s3[2, :-1]
    infile["co6/saddlebag-saddlebag/power_spectrum_error"] = xs_error_1d_coadd_per_field_s3[2, :-1]
    
    infile["field_coadded/saddlebag-saddlebag/power_spectrum"] = xs_mean_1d_coadd_s3[:-1]
    infile["field_coadded/saddlebag-saddlebag/power_spectrum_error"] = xs_error_1d_coadd_s3[:-1]
    
    infile["k"] = k_1d_s3[:-1]
    infile["k_bin_widths"] = dk_edges_1d_s3[:-1]

with h5py.File(f"/mn/stornext/d16/cmbco/comap/nils/data/power_spectra/current_power_spectrum_results_s1-s2_paper_version.h5", "w") as infile:
#with h5py.File("/mn/stornext/d5/data/nilsoles/comap/data/power_spectrum/current_power_spectrum_results_s1-s3_v11va8.h5", "w") as infile:
    infile["co2/saddlebag-saddlebag/power_spectrum"] = xs_mean_1d_coadd_per_field[0, :-1]
    infile["co2/saddlebag-saddlebag/power_spectrum_error"] = xs_error_1d_coadd_per_field[0, :-1]
    
    infile["co7/saddlebag-saddlebag/power_spectrum"] = xs_mean_1d_coadd_per_field[1, :-1]
    infile["co7/saddlebag-saddlebag/power_spectrum_error"] = xs_error_1d_coadd_per_field[1, :-1]
    
    infile["co6/saddlebag-saddlebag/power_spectrum"] = xs_mean_1d_coadd_per_field[2, :-1]
    infile["co6/saddlebag-saddlebag/power_spectrum_error"] = xs_error_1d_coadd_per_field[2, :-1]
    
    infile["field_coadded/saddlebag-saddlebag/power_spectrum"] = xs_mean_1d_coadd[:-1]
    infile["field_coadded/saddlebag-saddlebag/power_spectrum_error"] = xs_error_1d_coadd[:-1]
    
    infile["k"] = k_1d[:-1]
    infile["k_bin_widths"] = dk_edges_1d[:-1]


    


### Season 1 ###

path_to_s1 = "/mn/stornext/d16/cmbco/comap/defaults/S1/power_spectrum/S1_power_spectra_2D/"
s1_names = [
    "co2_map_summer_2D_arrays.h5",
    "co7_map_summer_2D_arrays.h5",
    "co6_map_signal_new_data_2D_arrays.h5",
]


tf_full = np.loadtxt("/mn/stornext/d16/cmbco/comap/defaults/S1/transfer_functions/full_tf_CES.txt")
k_2D = np.loadtxt("/mn/stornext/d16/cmbco/comap/defaults/S1/transfer_functions/k_2D.txt")

k_perp, k_par = k_2D

#TF_func = interp2d(k_perp, k_par, tf_full)

TF_func = RectBivariateSpline(
            k_perp, 
            k_par, 
            tf_full,
            s = 0, # No smoothing when splining
            kx = 3, # Use bi-cubic spline in x-direction
            ky = 3, # Use bi-cubic spline in x-direction
            )


def sphecical_average_spectrum(ps, ps_error, k_2d, tf):
    kx, ky = k_2d
    
    k_bin_edges = np.logspace(-2.0, np.log10(1.5), len(kx) + 1)
    
    weights = (
        (tf / ps_error) ** 2
    )

    xs_mean_1d = ps.copy()
    xs_mean_1d /= tf
    xs_mean_1d *= weights

    kgrid = np.sqrt(sum(ki**2 for ki in np.meshgrid(kx, ky, indexing="ij")))

    Ck_nmodes_1d = np.histogram(
        kgrid[kgrid > 0], bins=k_bin_edges, weights=xs_mean_1d[kgrid > 0]
    )[0]
    inv_var_nmodes_1d = np.histogram(
        kgrid[kgrid > 0], bins=k_bin_edges, weights=weights[kgrid > 0]
    )[0]
    nmodes_1d = np.histogram(kgrid[kgrid > 0], bins=k_bin_edges)[0]

    k_1d = (k_bin_edges[1:] + k_bin_edges[:-1]) / 2.0

    Ck_1d = np.zeros_like(k_1d)
    rms_1d = np.zeros_like(k_1d)

    Ck_1d[np.where(nmodes_1d > 0)] = (
        Ck_nmodes_1d[np.where(nmodes_1d > 0)]
        / inv_var_nmodes_1d[np.where(nmodes_1d > 0)]
    )
    
    rms_1d[np.where(nmodes_1d > 0)] = np.sqrt(
        1 / inv_var_nmodes_1d[np.where(nmodes_1d > 0)]
    )

    return Ck_1d, rms_1d, k_1d

def read_data(path):
    data = {}
    with h5py.File(path, "r") as infile:
        for key, value in infile.items():
            data[key] = value[()]
    return data


xs_all_s1 = np.zeros(14)
xs_error_all_s1 = np.zeros(14)


for i in range(3):
    path_to_spectrum = os.path.join(path_to_s1, s1_names[i])
    data_s1 = read_data(path_to_spectrum)

    tf = TF_func(data_s1["k"][0][0], data_s1["k"][0][1])
    Ck_1d, Ck_error_1d, k_1d_s1 = sphecical_average_spectrum(data_s1["xs_mean"][1, ...], data_s1["xs_sigma"][1, ...], data_s1["k"][1, ...], tf)
    
    
    # with h5py.File(f"{s1_names[:-3]}_1d.h5", "w") as outfile:
    #     outfile.create_dataset("xs_mean", data = Ck_1d)
    #     outfile.create_dataset("xs_sigma", data = Ck_error_1d)
    
    
    
    xs_inv_var = 1 / Ck_error_1d ** 2
    xs_all_s1 += Ck_1d * xs_inv_var
    xs_error_all_s1 += xs_inv_var

xs_all_s1 = xs_all_s1 / xs_error_all_s1
xs_error_all_s1 = 1 / np.sqrt(xs_error_all_s1)


xs_all_s1 = xs_all_s1[mask]
xs_error_all_s1 = xs_error_all_s1[mask]
k_1d_s1 = k_1d_s1[mask]



### 95% ULs ###

# S2 #
x = np.linspace(-1e3, 1e3, int(1e4))
pdf_per_k = norm.pdf(x[:, None], loc = (xs_mean_1d_coadd * k_1d)[None, :], scale = (xs_error_1d_coadd * k_1d)[None, :])
cdf_per_k = norm.cdf(x[:, None], loc = (xs_mean_1d_coadd * k_1d)[None, :], scale = (xs_error_1d_coadd * k_1d)[None, :])
sf_per_k = norm.sf(x[:, None], loc = (xs_mean_1d_coadd * k_1d)[None, :], scale = (xs_error_1d_coadd * k_1d)[None, :])
sf0_per_k = norm.sf(0, loc = (xs_mean_1d_coadd * k_1d)[None, :], scale = (xs_error_1d_coadd * k_1d)[None, :])
new_prob_per_k = 0.05 * sf0_per_k
ul95_per_k = norm.ppf(1 - new_prob_per_k, loc = xs_mean_1d_coadd * k_1d, scale = xs_error_1d_coadd * k_1d)

# S1 #
x = np.linspace(-1e3, 1e3, int(1e4))
pdf_perk_s1 = norm.pdf(x[:, None], loc = (xs_all_s1 * k_1d_s1)[None, :], scale = (xs_error_all_s1 * k_1d_s1)[None, :])
cdf_perk_s1 = norm.cdf(x[:, None], loc = (xs_all_s1 * k_1d_s1)[None, :], scale = (xs_error_all_s1 * k_1d_s1)[None, :])
sf_perk_s1 = norm.sf(x[:, None], loc = (xs_all_s1 * k_1d_s1)[None, :], scale = (xs_error_all_s1 * k_1d_s1)[None, :])
sf0_perk_s1 = norm.sf(0, loc = (xs_all_s1 * k_1d_s1)[None, :], scale = (xs_error_all_s1 * k_1d_s1)[None, :])
new_prob_perk_s1 = 0.05 * sf0_perk_s1
ul95_per_k_s1 = norm.ppf(1 - new_prob_perk_s1, loc = xs_all_s1 * k_1d_s1, scale = xs_error_all_s1 * k_1d_s1)[0]


# S3 # 
x = np.linspace(-1e3, 1e3, int(1e4))
pdf_per_k_s3 = norm.pdf(x[:, None], loc = (xs_mean_1d_coadd_s3 * k_1d_s3)[None, :], scale = (xs_error_1d_coadd_s3 * k_1d_s3)[None, :])
cdf_per_k_s3 = norm.cdf(x[:, None], loc = (xs_mean_1d_coadd_s3 * k_1d_s3)[None, :], scale = (xs_error_1d_coadd_s3 * k_1d_s3)[None, :])
sf_per_k_s3 = norm.sf(x[:, None], loc = (xs_mean_1d_coadd_s3 * k_1d_s3)[None, :], scale = (xs_error_1d_coadd_s3 * k_1d_s3)[None, :])
sf0_per_k_s3 = norm.sf(0, loc = (xs_mean_1d_coadd_s3 * k_1d_s3)[None, :], scale = (xs_error_1d_coadd_s3 * k_1d_s3)[None, :])
new_prob_per_k_s3 = 0.05 * sf0_per_k_s3
ul95_per_k_s3 = norm.ppf(1 - new_prob_per_k_s3, loc = xs_mean_1d_coadd_s3 * k_1d_s3, scale = xs_error_1d_coadd_s3 * k_1d_s3)

# S3 map coadd # 
x = np.linspace(-1e3, 1e3, int(1e4))
pdf_per_k_s3_coadd = norm.pdf(x[:, None], loc = (xs_mean_1d_coadd_s3_coadd * k_1d_s3_coadd)[None, :], scale = (xs_error_1d_coadd_s3_coadd * k_1d_s3_coadd)[None, :])
cdf_per_k_s3_coadd = norm.cdf(x[:, None], loc = (xs_mean_1d_coadd_s3_coadd * k_1d_s3_coadd)[None, :], scale = (xs_error_1d_coadd_s3_coadd * k_1d_s3_coadd)[None, :])
sf_per_k_s3_coadd = norm.sf(x[:, None], loc = (xs_mean_1d_coadd_s3_coadd * k_1d_s3_coadd)[None, :], scale = (xs_error_1d_coadd_s3_coadd * k_1d_s3_coadd)[None, :])
sf0_per_k_s3_coadd = norm.sf(0, loc = (xs_mean_1d_coadd_s3_coadd * k_1d_s3_coadd)[None, :], scale = (xs_error_1d_coadd_s3_coadd * k_1d_s3_coadd)[None, :])
new_prob_per_k_s3_coadd = 0.05 * sf0_per_k_s3_coadd
ul95_per_k_s3_coadd = norm.ppf(1 - new_prob_per_k_s3_coadd, loc = xs_mean_1d_coadd_s3_coadd * k_1d_s3_coadd, scale = xs_error_1d_coadd_s3_coadd * k_1d_s3_coadd)


# COPSS #
x = np.linspace(-1e3, 1e3, int(1e4))
pdf_perk_copss = norm.pdf(x[:, None], loc = (COPSS_Pk * COPSS_k)[None, :], scale = (COPSS_errPk * COPSS_k)[None, :])
cdf_perk_copss = norm.cdf(x[:, None], loc = (COPSS_Pk * COPSS_k)[None, :], scale = (COPSS_errPk * COPSS_k)[None, :])
sf_perk_copss = norm.sf(x[:, None], loc = (COPSS_Pk * COPSS_k)[None, :], scale = (COPSS_errPk * COPSS_k)[None, :])
sf0_perk_copss = norm.sf(0, loc = (COPSS_Pk * COPSS_k)[None, :], scale = (COPSS_errPk * COPSS_k)[None, :])
new_prob_perk_copss = 0.05 * sf0_perk_copss
ul95_per_k_copss = norm.ppf(1 - new_prob_perk_copss, loc = COPSS_Pk * COPSS_k, scale = COPSS_errPk * COPSS_k)[0]




##### 95% UL plot #####
fig, ax = plt.subplots(figsize = (double_width, 6))


eb_s2 = ax.errorbar(k_1d, ul95_per_k[0], xerr = dk_edges_1d / 2, yerr = ul95_per_k[0] * 2e-1, fmt = " ", elinewidth = 7, capsize = 9, color = "plum", label = r"COMAP S2", uplims = True, alpha = 0.6)

eb_s3 = ax.errorbar(k_1d_s3, ul95_per_k_s3[0], xerr = dk_edges_1d_s3 / 2, yerr = ul95_per_k_s3[0] * 2e-1, fmt = " ", elinewidth = 7, capsize = 9, color = "gray", label =  rf"COMAP S3 (v11{data_selection})", uplims = True, alpha = 0.8)

if include_coadd:
    eb_s3_coadd = ax.errorbar(k_1d_s3_coadd, ul95_per_k_s3_coadd[0], xerr = dk_edges_1d_s3_coadd / 2, yerr = ul95_per_k_s3_coadd[0] * 2e-1, fmt = " ", elinewidth = 7, capsize = 9, color = "k", label =  rf"COMAP S3 (v11{data_selection} coadd)", uplims = True, alpha = 0.8)


eb_s1 = ax.errorbar(k_1d_s1, ul95_per_k_s1, xerr = dk_edges_1d / 2, yerr = ul95_per_k_s1 * 2e-1, fmt = " ", elinewidth = 7, capsize = 9, color = "cornflowerblue", label = r"COMAP ES", uplims = True, alpha = 0.6)

# eb_s3[-1][0].set_linestyle('--')

mask_copss = np.ones_like(ul95_per_k_copss[:8], dtype = bool)

ax.errorbar(COPSS_k[:8], ul95_per_k_copss[:8], xerr = np.diff(COPSS_k_edges)[:8] / 2, yerr = ul95_per_k_copss[:8] * 2e-1, fmt = " ", elinewidth = 7, capsize = 9, color = "orange", label = r"COPSS", uplims = True, zorder = -1, alpha = 0.6)

k_hr = np.linspace(0.01, 3, 100)


# ax.plot(k_hr, 0.5e3 * k_hr, ls = "dashdot", c = "brown", alpha = 0.5, lw = 1.5)
# ax.plot(k_hr, 1e3 * k_hr, ls = "solid", c = "brown", alpha = 0.7, lw = 3, label = "$P^\mathrm{COLDz}_\mathrm{shot}(k)$ (68% CL)")
# ax.plot(k_hr, 2e3 * k_hr, ls = "dashdot", c = "brown", alpha = 0.5, lw = 1.5)
# ax.fill_between(k_hr, 0.5e3 * k_hr, 2e3 * k_hr, ls = "dashed", color = "brown", alpha = 0.2)

ax.plot(flat_coldz["k"], flat_coldz["Pk"] * flat_coldz["k"], ls = "solid", c = "brown", alpha = 0.7, lw = 3, label = "UM+COLDz+COPSS (68% CL)")

flat_coldz_68 = flat_coldz_percentile["Pkpct"][flat_coldz_percentile["pct"] == 68][0, :]
flat_coldz_32 = flat_coldz_percentile["Pkpct"][flat_coldz_percentile["pct"] == 32][0, :]

ax.plot(flat_coldz_percentile["k"], flat_coldz_68 * flat_coldz_percentile["k"], ls = "dashed", c = "brown", alpha = 0.5, lw = 1.5)
ax.plot(flat_coldz_percentile["k"], flat_coldz_32 * flat_coldz_percentile["k"], ls = "dashed", c = "brown", alpha = 0.5, lw = 1.5)
ax.fill_between(flat_coldz_percentile["k"], flat_coldz_32 * flat_coldz_percentile["k"], flat_coldz_68 * flat_coldz_percentile["k"], ls = "dashed", color = "brown", alpha = 0.2)


h = 0.7


ln4 = ax.plot(COMAPESV["k"], np.trapezoid(COMAPESV["Pkmu_noatt"], COMAPESV["mu"]) * COMAPESV["k"], color = "m", lw = 4, label = "COMAP fiducial")
ln6, = ax.plot(L16K20["k"], np.trapezoid(L16K20["Pkmu_noatt"], L16K20["mu"]) * L16K20["k"], color = "green", lw = 4, label = "Li-Keating")


ax.grid(False)
ax.set_xscale("log")
ax.set_yscale("log")

klabels = [0.1, 0.2, 0.4, 0.8, 1.6]
ax.set_xticks(klabels)
ax.set_xticklabels(klabels, fontsize = fontsize, rotation = 0)
ax.set_xlim(0.08, 1.9)

ax.set_yticks(np.logspace(2, 5, 3))
ax.set_yticklabels([rf"$10^{np.log10(i):.0f}$" for i in np.logspace(2, 5, 3)], fontsize = fontsize, rotation = 90)
ax.set_ylim(1e2, 6e5)

ax.set_ylabel(r"$k\tilde{C}(k)$  $\mathrm{[\mu K^2 Mpc^{2}]}$", fontsize = fontsize)
ax.set_xlabel(r"$k$  $\mathrm{[Mpc^{-1}]}$", fontsize = fontsize)

handles, labels = plt.gca().get_legend_handles_labels()
print(labels)

if include_coadd:
    order = [5, 4, 3, 6, 7, 0, 1, 2]
else:
    order = [3, 4, 5, 6, 0, 1, 2]
print([labels[idx] for idx in order])

ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order], ncols = 2, frameon = False, fontsize = fontsize, loc = "upper left")

coadd_string = "_coadd" if include_coadd else ""
#ax.legend(ncols = 1, frameon = False, fontsize = fontsize, loc = "upper left", bbox_to_anchor = (0.005, 0.9), bbox_transform = ax.transAxes)
fig.savefig(f"/mn/stornext/d16/cmbco/comap/nils/figs/psx_95_ul_and_measurent_v11{data_selection}{coadd_string}.pdf", facecolor = "white", bbox_inches = "tight")
#fig.savefig(f"/mn/stornext/d16/cmbco/comap/nils/figs/psx_95_ul_and_measurent_v11{data_selection}{coadd_string}.pdf", facecolor = "white", bbox_inches = "tight")
#fig.savefig(f"/mn/stornext/d16/cmbco/comap/nils/figs/psx_95_ul_and_measurent_v11{data_selection}{coadd_string}.pdf", facecolor = "white", bbox_inches = "tight")


#### Power spectrum 1D #####
fontsize = 18
fig = plt.figure(figsize = (double_width, 9))

gs = GridSpec(3, 3, figure=fig, wspace = 0.2, hspace = 0.0)
ax = [fig.add_subplot(gs[:2,:2]), fig.add_subplot(gs[2,:2]), fig.add_subplot(gs[:2,2])]

koffset = np.linspace(-0.04, 0.04, 4)
k_hr = np.linspace(0.1, 1, 1000)

if include_coadd:
    ax[0].errorbar(k_1d_s3_coadd + k_1d_s3_coadd * koffset[0], k_1d_s3_coadd * xs_mean_1d_coadd_s3_coadd * 1e-3, k_1d_s3_coadd * xs_error_1d_coadd_s3_coadd * 1e-3, fmt = "o", label =  "COMAP S3\n"+rf"(v11{data_selection} coadd)", color = "k", markersize = 6, elinewidth = 3)

ax[0].errorbar(k_1d_s3 + k_1d_s3 * koffset[1], k_1d_s3 * xs_mean_1d_coadd_s3 * 1e-3, k_1d_s3 * xs_error_1d_coadd_s3 * 1e-3, fmt = "o", label =  "COMAP S3\n"+rf"(v11{data_selection})", color = "gray", markersize = 6, elinewidth = 3)

ax[0].errorbar(k_1d + k_1d * koffset[2], k_1d * xs_mean_1d_coadd * 1e-3, k_1d * xs_error_1d_coadd * 1e-3, fmt = "o", label = r"COMAP S2", color = "plum", markersize = 6, elinewidth = 3)

ax[0].errorbar(k_1d_s1 + k_1d_s1 * koffset[3], k_1d_s1 * xs_all_s1 * 1e-3, k_1d_s1 * xs_error_all_s1 * 1e-3, fmt = "o", label = r"COMAP ES", color = "cornflowerblue", markersize = 6, elinewidth = 3)

ax[0].errorbar(COPSS_k, COPSS_k * COPSS_Pk * 1e-3, COPSS_k * COPSS_errPk * 1e-3, fmt = "o", label = r"COPSS", color = "orange", markersize = 6, elinewidth = 3, alpha = 0.7)


ax[0].grid(False)
ax[0].set_xscale("log")
ax[0].set_ylim(-18, 18)

klabels = [0.06, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0]
ax[0].set_xticks(klabels)
ax[0].set_xticklabels(klabels, fontsize = fontsize,  visible=False)
ax[0].set_xlim(0.09, 1.4)

ax[0].tick_params("y", rotation=90, labelsize = fontsize)

ax[0].set_ylabel(r"$k\tilde{C}(k)$  $\mathrm{[10^3 \, \mu K^2 Mpc^{2}]}$", fontsize = fontsize)

k_hr = np.linspace(0.01, 3, 100)

# ax[0].plot(k_hr, 1e3 * k_hr / 1e3, ls = "solid", c = "brown", alpha = 0.7, lw = 3, label = "$P^\mathrm{COLDz}_\mathrm{shot}(k)$ (68% CL)")

ax[0].plot(flat_coldz["k"], flat_coldz["Pk"] * flat_coldz["k"] * 1e-3, ls = "solid", c = "brown", alpha = 0.7, lw = 3, label = " UM\n+COLDz\n+COPSS (68% CL)")

ln4 = ax[0].plot(COMAPESV["k"], np.trapezoid(COMAPESV["Pkmu_noatt"], COMAPESV["mu"]) * COMAPESV["k"] * 1e-3, color = "m", lw = 2, label = "COMAP fiducial")

ln6, = ax[0].plot(L16K20["k"], np.trapezoid(L16K20["Pkmu_noatt"], L16K20["mu"]) * L16K20["k"] * 1e-3, color = "green", lw = 2, label = "Li-Keating")

handles, labels = ax[0].get_legend_handles_labels()

if include_coadd:
    order = [3, 4, 5, 6, 7, 0, 1, 2]
else:
    order = [3, 4, 5, 6, 0, 1, 2]

ax[0].legend([handles[idx] for idx in order],[labels[idx] for idx in order], ncols = 2, fontsize = fontsize - 3, frameon = True, loc = "upper center", bbox_to_anchor=(1.32, -0.15), columnspacing = 1)

k_hr = np.linspace(0.1, 1, 1000)

if include_coadd:
    ax[1].errorbar(k_1d_s3_coadd + k_1d_s3_coadd * koffset[0], xs_mean_1d_coadd_s3_coadd / xs_error_1d_coadd_s3_coadd, 1, fmt = "o", label = "COMAP S3 (coadd)", color = "k", markersize = 6, elinewidth = 3)

ax[1].errorbar(k_1d_s3 + k_1d_s3 * koffset[1], xs_mean_1d_coadd_s3 / xs_error_1d_coadd_s3, 1, fmt = "o", label = "COMAP S3", color = "gray", markersize = 6, elinewidth = 3)

ax[1].errorbar(k_1d + k_1d * koffset[2], xs_mean_1d_coadd / xs_error_1d_coadd, 1, fmt = "o", label = "COMAP S2", color = "plum", markersize = 6, elinewidth = 3)

ax[1].errorbar(k_1d_s1 + k_1d_s1 * koffset[3], xs_all_s1 / xs_error_all_s1, 1, fmt = "o", label = r"COMAP ES", color = "cornflowerblue", markersize = 6, elinewidth = 3)

ax[1].errorbar(COPSS_k, COPSS_Pk / COPSS_errPk, 1, fmt = "o", label = r"COPSS", color = "orange", markersize = 6, elinewidth = 3, alpha = 0.7)


ax[0].axhline(0, color = "k", alpha = 0.5)
ax[1].axhline(0, color = "k", alpha = 0.5)

ax[1].grid(False)
ax[1].set_xscale("log")
klabels = [0.1, 0.2, 0.4, 0.8, 1.6]

ax[1].set_xticks(klabels)
ax[1].set_xticklabels(klabels, fontsize = fontsize, rotation = 0)

ax[1].set_ylim(-3, 3)
ax[1].set_xlim(0.09, 1.4)

ax[1].set_yticks(np.arange(-2, 2.1, 2))
ax[1].tick_params("y", rotation=90, labelsize = fontsize)

ax[1].set_ylabel(r"$\tilde{C} / \sigma_\tilde{C}$", fontsize = fontsize)
ax[1].set_xlabel(r"$k$  $\mathrm{[Mpc^{-1}]}$", fontsize = fontsize)




koffset = np.linspace(-0.04, 0.04, 3)
k_hr = np.linspace(0.1, 1, 1000)

if include_coadd:
    ax[2].errorbar(k_1d_s3_coadd + k_1d_s3_coadd * koffset[0], k_1d_s3_coadd * xs_mean_1d_coadd_s3_coadd * 1e-3, k_1d_s3_coadd * xs_error_1d_coadd_s3_coadd * 1e-3, fmt = "o", label =  rf"COMAP S3 v11{data_selection} (coadd)", color = "k", markersize = 6, elinewidth = 3)

ax[2].errorbar(k_1d_s3 + k_1d_s3 * koffset[1], k_1d_s3 * xs_mean_1d_coadd_s3 * 1e-3, k_1d_s3 * xs_error_1d_coadd_s3 * 1e-3, fmt = "o", label =  rf"COMAP S3 v11{data_selection}", color = "gray", markersize = 6, elinewidth = 3)

ax[2].errorbar(k_1d + k_1d * koffset[2], k_1d * xs_mean_1d_coadd * 1e-3, k_1d * xs_error_1d_coadd * 1e-3, fmt = "o", label = r"COMAP S2", color = "plum", markersize = 6, elinewidth = 3)

k_hr = np.linspace(0.01, 3, 100)

# ax[2].fill_between(k_hr, 0.5e3 * k_hr / 1e3, 2e3 * k_hr / 1e3, ls = "dashed", color = "brown", alpha = 0.2)
    
# ax[2].plot(k_hr, 0.5e3 * k_hr / 1e3, ls = "dashdot", c = "brown", alpha = 0.5, lw = 1.5)
# ax[2].plot(k_hr, 1e3 * k_hr / 1e3, ls = "solid", c = "brown", alpha = 0.7, lw = 2, label = "COLDz shot noise")
# ax[2].plot(k_hr, 2e3 * k_hr / 1e3, ls = "dashdot", c = "brown", alpha = 0.5, lw = 1.5)

ax[2].plot(flat_coldz["k"], flat_coldz["Pk"] * flat_coldz["k"] * 1e-3, ls = "solid", c = "brown", alpha = 0.7, lw = 3, label = " UM\n+COLDz\n+COPSS (68% CL)")

flat_coldz_68 = flat_coldz_percentile["Pkpct"][flat_coldz_percentile["pct"] == 68][0, :]
flat_coldz_32 = flat_coldz_percentile["Pkpct"][flat_coldz_percentile["pct"] == 32][0, :]

ax[2].plot(flat_coldz_percentile["k"], flat_coldz_68 * flat_coldz_percentile["k"]* 1e-3, ls = "dashed", c = "brown", alpha = 0.5, lw = 1.5)
ax[2].plot(flat_coldz_percentile["k"], flat_coldz_32 * flat_coldz_percentile["k"]* 1e-3, ls = "dashed", c = "brown", alpha = 0.5, lw = 1.5)
ax[2].fill_between(flat_coldz_percentile["k"], flat_coldz_32 * flat_coldz_percentile["k"]* 1e-3, flat_coldz_68 * flat_coldz_percentile["k"]* 1e-3, ls = "dashed", color = "brown", alpha = 0.2)


ax[2].grid(False)
ax[2].set_xscale("log")
ax[2].set_ylim(-3.1, 3.1)

ax[2].set_xticks(klabels, klabels, fontsize = fontsize)
ax[2].set_xlim(0.09, 1.4)
ax[2].set_xlabel(r"$k$  $\mathrm{[Mpc^{-1}]}$", fontsize = fontsize)

ax[2].tick_params("y", rotation=90, labelsize = fontsize)

ax[2].set_ylabel(r"$k\tilde{C}(k)$  $\mathrm{[10^3\,\mu K^2 Mpc^{2}]}$", fontsize = fontsize)


ln4 = ax[2].plot(COMAPESV["k"], np.trapezoid(COMAPESV["Pkmu_noatt"], COMAPESV["mu"]) * COMAPESV["k"] * 1e-3, color = "m", lw = 2, label = "COMAP fiducial")

ln6, = ax[2].plot(L16K20["k"], np.trapezoid(L16K20["Pkmu_noatt"], L16K20["mu"]) * L16K20["k"] * 1e-3, color = "green", lw = 2, label = "Li-Keating")


ax[2].axhline(0, color = "k", alpha = 0.5)

ax[2].yaxis.tick_right()
ax[2].yaxis.set_label_position("right")


con = ConnectionPatch(
    xyA = (1.4, 3),
    xyB = (0.09, 3),
    coordsA = ax[0].transData,
    coordsB = ax[2].transData,
    ec = "gray",
    fc = "gray",
    alpha = 0.5,
    linestyle = "dashed",
    lw = 2,
)
ax[0].add_artist(con)

con = ConnectionPatch(
    xyA = (1.4, -3),
    xyB = (0.09, -3),
    coordsA = ax[0].transData,
    coordsB = ax[2].transData,
    ec = "gray",
    fc = "gray",
    alpha = 0.5,
    linestyle = "dashed",
    lw = 2,
)
ax[0].add_artist(con)


ax[0].axhline(3, color = "gray", alpha = 0.5, linestyle = "dashed", lw = 2)
ax[0].axhline(-3, color = "gray", alpha = 0.5, linestyle = "dashed", lw = 2)

coadd_string = "_coadd" if include_coadd else ""

fig.savefig(f"/mn/stornext/d16/cmbco/comap/nils/figs/psx_fields_coadded_spectra_subseasons_v11{data_selection}{coadd_string}_and_s1_s2_and_copss.pdf", facecolor = "white", bbox_inches = "tight")
#fig.savefig(f"/mn/stornext/d16/cmbco/comap/nils/figs/psx_fields_coadded_spectra_subseasons_v11{data_selection}_and_s1_s2_and_copss.pdf", facecolor = "white", bbox_inches = "tight")
#fig.savefig(f"/mn/stornext/d16/cmbco/comap/nils/figs/psx_fields_coadded_spectra_subseasons_v11{data_selection}_and_s1_s2_and_copss.pdf", facecolor = "white", bbox_inches = "tight")



### Power spectrum 1D per field ###

fig = plt.figure(figsize = (single_width, 7))
gs = GridSpec(3, 1, figure=fig, wspace = 0.03, hspace = 0.0)

#ax = [fig.add_subplot(gs[i]) for i in range(2)]
ax = [fig.add_subplot(gs[:2]), fig.add_subplot(gs[-1])]

offset = np.linspace(-0.05, 0.05, 4)

k_hr = np.linspace(0.1, 1, 1000)
# ax[0].errorbar(k_1d_s1 + k_1d_s1 * koffset[0], k_1d_s1 * xs_all_s1 * 1e-3, k_1d_s1 * xs_error_all_s1 * 1e-3, fmt = "o", label = r"Ihle+2022", color = "k", markersize = 6, elinewidth = 3)
# #ax[0].errorbar(k_1d - k_1d * 0.02, k_1d * xs_coadd_1d, k_1d * xs_error_coadd_1d, fmt = "o", label = r"Combined fields feed$\times$feed", color = "m")
ax[0].errorbar(k_1d_s3 + k_1d_s3 * offset[0], k_1d_s3 * xs_mean_1d_coadd_s3 * 1e-3, k_1d * xs_error_1d_coadd_s3 * 1e-3, fmt = "o", label = r"Combined", color = "k", markersize = 6, elinewidth = 3)

ax[0].errorbar(k_1d_s3 + k_1d_s3 * offset[1], k_1d_s3 * xs_mean_1d_coadd_per_field_s3[0, :] * 1e-3, k_1d * xs_error_1d_coadd_per_field_s3[0, :] * 1e-3, fmt = "o", label = "Field 1", markersize = 6, elinewidth = 3, color = "orange", alpha = 0.6)
ax[0].errorbar(k_1d_s3 + k_1d_s3 * offset[2], k_1d_s3 * xs_mean_1d_coadd_per_field_s3[1, :] * 1e-3, k_1d_s3 * xs_error_1d_coadd_per_field_s3[1, :] * 1e-3, fmt = "o", label = "Field 2", markersize = 6, elinewidth = 3, color = "r", alpha = 0.6)
ax[0].errorbar(k_1d_s3 + k_1d_s3 * offset[3], k_1d_s3 * xs_mean_1d_coadd_per_field_s3[2, :] * 1e-3, k_1d * xs_error_1d_coadd_per_field_s3[2, :] * 1e-3, fmt = "o", label = "Field 3", markersize = 6, elinewidth = 3, color = "b", alpha = 0.6)

ax[0].grid(False)
ax[0].set_xscale("log")

klabels = [0.06, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0]
ax[0].set_xticks(klabels)
ax[0].set_xticklabels(klabels, fontsize = fontsize,  visible=False)
ax[0].set_xlim(0.09, 1.4)

ax[0].set_yticks(np.round(np.linspace(-7, 7, 5), 1)[1:])
ax[0].set_yticklabels(ax[0].get_yticklabels(), fontsize = fontsize, rotation = 90)
ax[0].set_ylim(-8, 9)


ax[0].legend(ncols = 2, fontsize = fontsize, loc = "upper center", frameon = True)#, bbox_to_anchor=(1.0, 1.0))
ax[0].set_ylabel(r"$k\tilde{C}(k)$  $\mathrm{[10^3\,\mu K^2 Mpc^{2}]}$", fontsize = fontsize)


k_hr = np.linspace(0.1, 1, 1000)
ax[1].errorbar(k_1d_s3 + k_1d_s3 * offset[0], xs_mean_1d_coadd_s3 / xs_error_1d_coadd_s3, 1, fmt = "o", label = "Combined", color = "k", markersize = 6, elinewidth = 3)

ax[1].errorbar(k_1d_s3 + k_1d_s3 * offset[1], xs_mean_1d_coadd_per_field_s3[0, :] / xs_error_1d_coadd_per_field_s3[0, :], 1, fmt = "o", label = "Field 2", markersize = 6, elinewidth = 3, color = "orange", alpha = 0.6)
ax[1].errorbar(k_1d_s3 + k_1d_s3 * offset[2], xs_mean_1d_coadd_per_field_s3[1, :] / xs_error_1d_coadd_per_field_s3[1, :], 1, fmt = "o", label = "Field 2", markersize = 6, elinewidth = 3, color = "r", alpha = 0.6)
ax[1].errorbar(k_1d_s3 + k_1d_s3 * offset[3], xs_mean_1d_coadd_per_field_s3[2, :] / xs_error_1d_coadd_per_field_s3[2, :], 1, fmt = "o", label = "Field 3", markersize = 6, elinewidth = 3, color = "b", alpha = 0.6)
    

ax[0].axhline(0, color = "k", alpha = 0.5)
ax[1].axhline(0, color = "k", alpha = 0.5)

ax[1].grid(False)
ax[1].set_xscale("log")
klabels = [0.1, 0.2, 0.4, 0.8, 1.6]
ax[1].set_xticks(klabels)
ax[1].set_xticklabels(klabels, fontsize = fontsize, rotation = 0)
ax[1].set_xlim(0.09, 1.4)

ax[1].set_ylim(-3, 4)

ax[1].set_yticks([-2, 0, 2, 4])
ax[1].set_yticklabels(ax[1].get_yticklabels(), fontsize = fontsize, rotation = 90)

ax[1].set_ylabel(r"$\tilde{C} / \sigma_\tilde{C}$", fontsize = fontsize)

ax[1].set_xlabel(r"$k$  $\mathrm{[Mpc^{-1}]}$", fontsize = fontsize)

fig.savefig(f"/mn/stornext/d16/cmbco/comap/nils/figs/psx_fields_spectra_subseasons_v11{data_selection}_signle_column.pdf", facecolor = "white", bbox_inches = "tight")

if include_coadd:
    ## Power spectrum 1D per field map coadded ###

    fig = plt.figure(figsize = (single_width, 7))
    gs = GridSpec(3, 1, figure=fig, wspace = 0.03, hspace = 0.0)

    #ax = [fig.add_subplot(gs[i]) for i in range(2)]
    ax = [fig.add_subplot(gs[:2]), fig.add_subplot(gs[-1])]

    offset = np.linspace(-0.05, 0.05, 4)

    k_hr = np.linspace(0.1, 1, 1000)
    # ax[0].errorbar(k_1d_s1 + k_1d_s1 * koffset[0], k_1d_s1 * xs_all_s1 * 1e-3, k_1d_s1 * xs_error_all_s1 * 1e-3, fmt = "o", label = r"Ihle+2022", color = "k", markersize = 6, elinewidth = 3)
    # #ax[0].errorbar(k_1d - k_1d * 0.02, k_1d * xs_coadd_1d, k_1d * xs_error_coadd_1d, fmt = "o", label = r"Combined fields feed$\times$feed", color = "m")
    ax[0].errorbar(k_1d_s3_coadd + k_1d_s3_coadd * offset[0], k_1d_s3_coadd * xs_mean_1d_coadd_s3_coadd * 1e-3, k_1d * xs_error_1d_coadd_s3_coadd * 1e-3, fmt = "o", label = r"Combined", color = "k", markersize = 6, elinewidth = 3)

    ax[0].errorbar(k_1d_s3_coadd + k_1d_s3_coadd * offset[1], k_1d_s3_coadd * xs_mean_1d_coadd_per_field_s3_coadd[0, :] * 1e-3, k_1d * xs_error_1d_coadd_per_field_s3_coadd[0, :] * 1e-3, fmt = "o", label = "Field 1", markersize = 6, elinewidth = 3, color = "orange", alpha = 0.6)
    ax[0].errorbar(k_1d_s3_coadd + k_1d_s3_coadd * offset[2], k_1d_s3_coadd * xs_mean_1d_coadd_per_field_s3_coadd[1, :] * 1e-3, k_1d_s3_coadd * xs_error_1d_coadd_per_field_s3_coadd[1, :] * 1e-3, fmt = "o", label = "Field 2", markersize = 6, elinewidth = 3, color = "r", alpha = 0.6)
    ax[0].errorbar(k_1d_s3_coadd + k_1d_s3_coadd * offset[3], k_1d_s3_coadd * xs_mean_1d_coadd_per_field_s3_coadd[2, :] * 1e-3, k_1d * xs_error_1d_coadd_per_field_s3_coadd[2, :] * 1e-3, fmt = "o", label = "Field 3", markersize = 6, elinewidth = 3, color = "b", alpha = 0.6)

    ax[0].grid(False)
    ax[0].set_xscale("log")

    klabels = [0.06, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0]
    ax[0].set_xticks(klabels)
    ax[0].set_xticklabels(klabels, fontsize = fontsize,  visible=False)
    ax[0].set_xlim(0.09, 1.4)

    ax[0].set_yticks(np.round(np.linspace(-7, 7, 5), 1)[1:])
    ax[0].set_yticklabels(ax[0].get_yticklabels(), fontsize = fontsize, rotation = 90)
    ax[0].set_ylim(-8, 9)


    ax[0].legend(ncols = 2, fontsize = fontsize, loc = "best", frameon = True)#, bbox_to_anchor=(1.0, 1.0))
    ax[0].set_ylabel(r"$k\tilde{C}(k)$  $\mathrm{[10^3\,\mu K^2 Mpc^{2}]}$", fontsize = fontsize)


    k_hr = np.linspace(0.1, 1, 1000)
    ax[1].errorbar(k_1d_s3_coadd + k_1d_s3_coadd * offset[0], xs_mean_1d_coadd_s3_coadd / xs_error_1d_coadd_s3_coadd, 1, fmt = "o", label = "Combined", color = "k", markersize = 6, elinewidth = 3)

    ax[1].errorbar(k_1d_s3_coadd + k_1d_s3_coadd * offset[1], xs_mean_1d_coadd_per_field_s3_coadd[0, :] / xs_error_1d_coadd_per_field_s3_coadd[0, :], 1, fmt = "o", label = "Field 2", markersize = 6, elinewidth = 3, color = "orange", alpha = 0.6)
    ax[1].errorbar(k_1d_s3_coadd + k_1d_s3_coadd * offset[2], xs_mean_1d_coadd_per_field_s3_coadd[1, :] / xs_error_1d_coadd_per_field_s3_coadd[1, :], 1, fmt = "o", label = "Field 2", markersize = 6, elinewidth = 3, color = "r", alpha = 0.6)
    ax[1].errorbar(k_1d_s3_coadd + k_1d_s3_coadd * offset[3], xs_mean_1d_coadd_per_field_s3_coadd[2, :] / xs_error_1d_coadd_per_field_s3_coadd[2, :], 1, fmt = "o", label = "Field 3", markersize = 6, elinewidth = 3, color = "b", alpha = 0.6)
        

    ax[0].axhline(0, color = "k", alpha = 0.5)
    ax[1].axhline(0, color = "k", alpha = 0.5)

    ax[1].grid(False)
    ax[1].set_xscale("log")
    klabels = [0.1, 0.2, 0.4, 0.8, 1.6]
    ax[1].set_xticks(klabels)
    ax[1].set_xticklabels(klabels, fontsize = fontsize, rotation = 0)
    ax[1].set_xlim(0.09, 1.4)

    ax[1].set_ylim(-3, 4)

    ax[1].set_yticks([-2, 0, 2, 4])
    ax[1].set_yticklabels(ax[1].get_yticklabels(), fontsize = fontsize, rotation = 90)

    ax[1].set_ylabel(r"$\tilde{C} / \sigma_\tilde{C}$", fontsize = fontsize)

    ax[1].set_xlabel(r"$k$  $\mathrm{[Mpc^{-1}]}$", fontsize = fontsize)

    fig.savefig(f"/mn/stornext/d16/cmbco/comap/nils/figs/psx_fields_spectra_subseasons_v11{data_selection}_coadd_signle_column.pdf", facecolor = "white", bbox_inches = "tight")



### Normalized sensitivity ###

def norm_sens(error, delta_volume):
    return error * np.sqrt(delta_volume)

vol0 = 4 * np.pi / 3 * (k_1d - dk_edges_1d / 2) ** 3
vol1 = 4 * np.pi / 3 * (k_1d + dk_edges_1d / 2) ** 3
delta_volume = vol1 - vol0

vol0_copss = 4 * np.pi / 3 * (COPSS_k_edges[:-1]) ** 3
vol1_copss = 4 * np.pi / 3 * (COPSS_k_edges[1:]) ** 3
delta_volume_copss_pts = vol1_copss - vol0_copss

normalised_sensitivity = norm_sens(xs_error_1d_coadd, delta_volume)
normalised_sensitivity_s3 = norm_sens(xs_error_1d_coadd_s3, delta_volume)
normalised_sensitivity_s3_coadd = norm_sens(xs_error_1d_coadd_s3_coadd, delta_volume)
normalised_sensitivity_s1 = norm_sens(xs_error_all_s1, delta_volume)
normalised_sensitivity_copss_pts = norm_sens(COPSS_errPk, delta_volume_copss_pts)


fig, ax = plt.subplots(figsize = (single_width, 6))
ax.plot(COPSS_k, normalised_sensitivity_copss_pts, color = "orange", label = r"COPSS", ms = 10, marker = "o", lw = 3, alpha = 0.7)
ax.plot(k_1d, normalised_sensitivity, color = "plum", label = r"COMAP S2", ms = 10, marker = "o", lw = 3, alpha = 0.7)
ax.plot(k_1d, normalised_sensitivity_s1, color = "cornflowerblue", label = r"COMAP ES", ms = 10, marker = "o", lw = 3, alpha = 0.7)
ax.plot(k_1d_s3, normalised_sensitivity_s3, color = "gray", label = rf"COMAP S3 (v11{data_selection})", ms = 10, marker = "o", lw = 3, alpha = 0.7)

if include_coadd:
    ax.plot(k_1d_s3_coadd, normalised_sensitivity_s3_coadd, color = "k", label = rf"COMAP S3 (v11{data_selection} coadd)", ms = 10, marker = "o", lw = 3, alpha = 0.7)


ax.set_yscale("log")
ax.set_xscale("log")
ax.set_ylim(2.5e2, 1e6)
#ax.set_ylim(5e2, 3e4)

handles, labels = ax.get_legend_handles_labels()
if include_coadd:
    order = [4, 3, 1, 2, 0]
else:
    order = [3, 1, 2, 0]

ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order], ncols = 1, fontsize = fontsize, frameon = False, loc = "upper left")


#klabels = np.round(np.linspace(0.1, 8, 80), 2)[::3]
klabels = [0.1, 0.2, 0.4, 0.8, 1.6]
ax.set_xticks(klabels)
ax.set_xticklabels(klabels, fontsize = fontsize, rotation = 0)
ax.tick_params("y", rotation=90, labelsize = fontsize)

ax.set_xlim(0.08, 1.9)


ax.set_xlabel(r"$k$ [$\mathrm{Mpc}^{-1}$]", fontsize = fontsize)
ax.set_ylabel(r"$\xi$ $\mathrm{[\mu K^2 Mpc^{3/2}]}$", fontsize = fontsize)
fig.savefig(f"/mn/stornext/d16/cmbco/comap/nils/figs/normalised_sensitivity_v11{data_selection}{coadd_string}.pdf", facecolor = "white", bbox_inches = "tight")

print("Finished plotting!")
