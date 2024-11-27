from __future__ import print_function
import pickle
import errno
import h5py
import numpy as np
import matplotlib.pyplot as plt
import numpy.fft as fft
import scipy.stats as stats
from scipy.optimize import curve_fit
from astropy.time import Time                                                                                                                                             
import astropy.coordinates as coord 
import astropy.units as u 
from astropy.coordinates import SkyCoord
from astropy.coordinates import solar_system_ephemeris, EarthLocation                                                                                                    
from astropy.coordinates import get_body_barycentric, get_body, get_moon
from astropy.coordinates import AltAz
from scipy.interpolate import CubicSpline
import glob
import os
import pwd
import grp
import sys
import math
import multiprocessing
import importlib
import time
import warnings
import shutil
from tqdm import trange, tqdm
from mpi4py import MPI
# from accept_params import stats_cut as accept_params
# from stats_list import stats_list
# sys.path.append("/mn/stornext/d22/cmbco/comap/jonas/pipeline/")  # TODO: Find better solution
current = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current)
sys.path.append(parent_directory)
from tools.read_runlist import read_runlist as ext_read_runlist
warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")
warnings.filterwarnings("ignore", message="invalid value encountered in power")
warnings.filterwarnings("ignore", message="invalid value encountered in double_scalars")
warnings.filterwarnings("ignore", message="invalid value encountered in multiply")
warnings.filterwarnings("ignore", message="Mean of empty slice.")
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="divide by zero encountered in true_divide")
warnings.filterwarnings("ignore", message="Covariance of the parameters could not be estimated")
warnings.filterwarnings("ignore", message="Degrees of freedom <= 0 for slice.")
os.environ["OMP_NUM_THREADS"] = "1"

class spike_data():
    def __init__(self):
        self.spike_types = ['spike', 'jump', 'anomaly', 'edge spike']
        pass

class spike_list():
    def __init__(self):
        self.spikes = []
        self.spike_types = ['spike', 'jump', 'anomaly', 'edge spike']
    
    def add(self, spike):
        self.spikes.append(spike)
    
    def addlist(self, sp_list):
        for sp in sp_list:
            self.add(sp)

    def sorted(self):
        lists = [[], [], []]
        for spike in self.spikes:
            lists[spike.type].append(spike)
        for typelist in lists:
            typelist.sort(key=lambda x: np.abs(x.amp), reverse=True)  # hat tip: https://stackoverflow.com/a/403426/5238625
        return lists

def get_spike_list(sb_mean, sd, scan_id, mjd):
    # cutoff = 0.0015 * 8.0
    my_spikes = spike_list()
    for spike_type in range(3):
        for spike in range(1000):
            sbs = sd[0, :, :, spike_type, spike]
            if np.all(sbs == 0):
                break
            max_sb = np.unravel_index(np.argmax(np.abs(sbs), axis=None), sbs.shape)
            max_ind = int(sd[1, max_sb[0], max_sb[1], spike_type, spike]) - 1
            s = spike_data()
            s.amp = sd[0, max_sb[0], max_sb[1], spike_type, spike]
            s.sbs = sbs
            s.ind = np.array((max_sb[0], max_sb[1], max_ind))  # feed, sb, ind
            s.mjd = mjd[max_ind]
            s.data = sb_mean[max_sb[0], max_sb[1], max_ind - 200:max_ind + 200]
            s.type = spike_type
            s.scanid = scan_id
            my_spikes.add(s)
    return my_spikes

def get_sid(mjd):
    return 360 * ((1.002737811 * mjd) % 1)

def make_map(ra, dec, ra_bins, dec_bins, tod, mask):
    n_freq, n_samp = tod.shape

    # print(ra_bins)
    # print(dec_bins)
    # print(ra.shape)
    # print(dec.shape)
    n_pix_ra = len(ra_bins) - 1
    n_pix_dec = len(dec_bins) - 1
    map = np.zeros((n_pix_ra, n_pix_dec, n_freq))
    nhit = np.zeros_like(map)
    for i in range(n_freq):
        if mask[i] == 1.0:
            nhit[:, :, i] = np.histogram2d(ra, dec, bins=[ra_bins, dec_bins])[0]
            where = np.where(nhit[:, :, i] > 0)
            map[:, :, i][where] = np.histogram2d(ra, dec, bins=[ra_bins, dec_bins], weights=tod[i, :])[0][where] / nhit[:, :, i][where]
    return map, nhit


def compute_power_spec3d(x, k_bin_edges, dx=1, dy=1, dz=1):
    n_x, n_y, n_z = x.shape
    Pk_3D = np.abs(fft.fftn(x)) ** 2 * dx * dy * dz / (n_x * n_y * n_z)

    kx = np.fft.fftfreq(n_x, dx) * 2 * np.pi
    ky = np.fft.fftfreq(n_y, dy) * 2 * np.pi
    kz = np.fft.fftfreq(n_z, dz) * 2 * np.pi

    kgrid = np.sqrt(sum(ki ** 2 for ki in np.meshgrid(kx, ky, kz, indexing='ij')))

    Pk_nmodes = np.histogram(kgrid[kgrid > 0], bins=k_bin_edges, weights=Pk_3D[kgrid > 0])[0]
    nmodes = np.histogram(kgrid[kgrid > 0], bins=k_bin_edges)[0]

    k = (k_bin_edges[1:] + k_bin_edges[:-1]) / 2.0
    Pk = np.zeros_like(k)
    Pk[np.where(nmodes > 0)] = Pk_nmodes[np.where(nmodes > 0)] / nmodes[np.where(nmodes > 0)]
    return Pk, k, nmodes


def compute_power_spec1d_2d(x, k_bin_edges, dx=1, dy=1, dz=1):
    n_x, n_y, n_z = x.shape
    Pk_1D = np.abs(fft.rfft(x, axis=2)) ** 2 * dz / n_z
    kz = np.fft.rfftfreq(n_z, dz) * 2 * np.pi

    Pk_1D = Pk_1D.mean((0, 1))
    Pk_1D = Pk_1D[1:]

    Pk_2D = np.abs(fft.fftn(x, axes=(0, 1))) ** 2 * (dx * dy) / (n_x * n_y)
    kx = np.fft.fftfreq(n_x, dx) * 2 * np.pi
    ky = np.fft.fftfreq(n_y, dy) * 2 * np.pi

    kgrid = np.sqrt(sum(ki ** 2 for ki in np.meshgrid(kx, ky, indexing='ij')))

    Pk_2D = Pk_2D.mean(2)

    Pk_nmodes = np.histogram(kgrid[kgrid > 0], bins=k_bin_edges, weights=Pk_2D[kgrid > 0])[0]
    nmodes = np.histogram(kgrid[kgrid > 0], bins=k_bin_edges)[0]

    k = (k_bin_edges[1:] + k_bin_edges[:-1]) / 2.0
    Pk = np.zeros_like(k)
    Pk[np.where(nmodes > 0)] = Pk_nmodes[np.where(nmodes > 0)] / nmodes[np.where(nmodes > 0)]
    return Pk_1D, Pk, k, nmodes


def get_sb_ps(ra, dec, ra_bins, dec_bins, tod, mask, sigma, d_dec, n_k=10, is_slow=False):
    if is_slow:
        with h5py.File("/mn/stornext/d5/data/nilsoles/nils/dummy_pipeline/power_spectrum/transfer_functions/tf_all_ones.h5", "r") as f:
        #with h5py.File("/mn/stornext/d5/data/nilsoles/nils/pipeline/power_spectrum/transfer_functions/tf_slow_scans.h5", "r") as f:
            tf = f["/spherically_averaged/transfer_function"][2:]
            k_tf = f["/spherically_averaged/k_bin_centers"][2:]
    else:
        with h5py.File("/mn/stornext/d5/data/nilsoles/nils/dummy_pipeline/power_spectrum/transfer_functions/tf_all_ones.h5", "r") as f:
        #with h5py.File("/mn/stornext/d5/data/nilsoles/nils/pipeline/power_spectrum/transfer_functions/tf_fast_scans.h5", "r") as f:
            tf = f["/spherically_averaged/transfer_function"][2:]
            k_tf = f["/spherically_averaged/k_bin_centers"][2:]
    tf_interp_log = CubicSpline(np.log10(k_tf), np.log10(tf))
    def tf_interp(k_tf):
        return 10**tf_interp_log(np.log10(k_tf))


    map, nhit = make_map(ra, dec, ra_bins, dec_bins, tod, mask)
    h = 0.7
    deg2Mpc = 76.22 / h
    GHz2Mpc = 699.62 / h * (1 + 2.9) ** 2 / 115

    d_th = d_dec * deg2Mpc

    dz = 32.2e-3 * GHz2Mpc

    k_bin_edges = np.logspace(-1.8, np.log10(0.5), n_k)
    where = np.where(nhit > 0)
    rms = np.zeros_like(nhit)
    rms[where] = (sigma[None, None, :]/ np.sqrt(nhit))[where]
    w = np.zeros_like(nhit)
    w[where] = 1 / rms[where] ** 2
    # w = w / np.

    Pk, k, nmodes = compute_power_spec3d(w * map, k_bin_edges, d_th, d_th, dz)
    n_sim = 100
    ps_arr = np.zeros((n_sim, n_k - 1))
    for l in range(n_sim):
        map_n = np.random.randn(*rms.shape) * rms
        ps_arr[l] = compute_power_spec3d(w * map_n, k_bin_edges, d_th, d_th, dz)[0]
    
    # transfer = 1.0 / np.exp((0.055/k) ** 2.5)  # 6.7e5 / np.exp((0.055/k) ** 2.5)#1.0 / np.exp((0.03/k) ** 2)   ######## Needs to be tested!
    transfer = tf_interp(k)
    
    ps_mean = np.mean(ps_arr, axis=0)
    ps_std = np.std(ps_arr, axis=0) / transfer
    Pk = Pk / transfer

    n_chi2 = len(k)
    chi = np.sum(((Pk - ps_mean)/ ps_std) ** 3)
    chi2 = np.sign(chi) * np.abs((np.sum(((Pk - ps_mean)/ ps_std) ** 2) - n_chi2) / np.sqrt(2 * n_chi2))
    return chi2, Pk, ps_mean, ps_std, transfer, map, rms

# From Tony Li
def ensure_dir_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


# def get_params(param_file):
#     params = {}
#     with open(param_file) as f:
#         fr = f.readlines()

#         fr = [f[:] for f in fr]

#         frs = [f.split(" = ") for f in fr]

#         for stuff in frs:
#             try:
#                 i, j = stuff
#                 params[str(i).strip()] = eval(j)
#             except ValueError:
#                 pass
#             except SyntaxError:
#                 if j == '.true.':
#                     params[str(i).strip()] = True
#                 elif j == '.false.':
#                     params[str(i).strip()] = False
#                 else:
#                     pass
#     return params

def read_runlist(params):
    fields = {}
    if rank == 0:
        scans = {}
        unique_fields = []
        runlist = ext_read_runlist(params, ignore_existing=False)
        for scan in runlist:
            field = scan[4]
            if not field in unique_fields:
                unique_fields.append(field)
        for field in unique_fields:
            fields[field] = []
            obsids_in_this_field = []
            n_scans_in_this_field = 0
            for scan in runlist:
                if scan[4] == field:
                    obsid = str(scan[0]//100)
                    if not obsid in scans:
                        obsids_in_this_field.append(obsid)
                        scans[obsid] = []
                    scans[obsid].append(f"{scan[0]:08d}")
                    n_scans_in_this_field += 1
            fields[field] = [obsids_in_this_field, scans, n_scans_in_this_field]
    fields = comm.bcast(fields, root=0)
    return fields

def read_runlist_old(params):
    filename = params['runlist']
    obsid_start = int(params['obsid_start'])
    obsid_stop = int(params['obsid_stop'])

    with open(filename) as my_file:
        lines = [line.split() for line in my_file]
    i = 0

    fields = {}
    n_fields = int(lines[i][0])
    i = i + 1
    for i_field in range(n_fields):
        obsids = []
        scans = {}
        n_scans_tot = 0
        fieldname = lines[i][0]
        n_obsids = int(lines[i][1])
        i = i + 1
        for j in range(n_obsids):
            obsid = lines[i][0]
            obsids.append(obsid)
            n_scans = int(lines[i][3])
            o_scans = []
            for k in range(1, n_scans - 1):
                if lines[i+k+1][0] != 8192:
                    if obsid_start <= int(obsid) <= obsid_stop:
                        o_scans.append(lines[i+k+1][0])
                        n_scans_tot += 1
            scans[obsid] = o_scans
            i = i + n_scans + 1 
        fields[fieldname] = [obsids, scans, n_scans_tot]
        print(fieldname, n_scans_tot)
    return fields


def insert_data_in_array(data, indata, stats_string, obsid=False):
    try:
        index = stats_list.index(stats_string)
        if obsid:
            data[:, :, :, index] = indata
        else:
            data[:, :, index] = indata
    except ValueError:
        print('Did not find statistic "' + stats_string + '" in stats list.')


def extract_data_from_array(data, stats_string):
    try:
        index = stats_list.index(stats_string)
        outdata = data[:,:,:, index]
        return outdata
    except ValueError:
        print('Did not find statistic "' + stats_string + '" in stats list.')
        return 0

def get_scan_stats(filepath, map_grid=None):
    n_stats = len(stats_list)
    try:
        with h5py.File(filepath, mode="r") as my_file:
            freq_decimation_factor = my_file["decimation_nu"][()]
    except:
        print(f"CANNOT OPEN CORRUPT OR NON-EXISTING FILE: {filepath}")
    with h5py.File(filepath, mode="r") as my_file:
        tod_ind = np.array(my_file['tod'][:])
        tod_ind[~np.isfinite(tod_ind)] = 0
        # print(np.sum(np.isfinite(tod_ind)), np.size(tod_ind), tod_ind.shape)
        n_det_ind, n_sb, n_freq, n_samp = tod_ind.shape
        freq_decimation_factor = my_file["decimation_nu"][()]
        try:
            freq_bin_sizes_lowres_MHz = (my_file["freq_bin_centers_lowres"][0,1] - my_file["freq_bin_centers_lowres"][0,0])*1000
        except:
            freq_bin_sizes_lowres_MHz = 31.25
        sb_mean_ind = np.array(my_file['sb_mean'][:])
        point_tel_ind = np.array(my_file['point_tel'][:])
        point_radec_ind = np.array(my_file['point_cel'][:])
        # mask_ind = my_file['freqmask'][:]
        # mask_full_ind = my_file['freqmask_full'][:]
        try:
            temporal_mask_ind = my_file['mask_temporal'][()]
        except:
            temporal_mask_ind = np.ones((n_det_ind, n_samp), dtype=bool)
        mask_ind = my_file['freqmask'][:]
        mask_full_ind = my_file['freqmask_full'][:]
        reason_ind = my_file['freqmask_reason'][:]
        sigma0_ind = my_file['sigma0'][()]
        # n_nan_ind = my_file['n_nan'][()]
        # n_nan_ind = my_file['n_nans'][()]

        # pixels = np.array(my_file['pixels'][:]) - 1 
        pixels = np.array(my_file['feeds'][:]) - 1 
        # pix2ind = my_file['pix2ind'][:]
        pix2ind = my_file['pix2ind_fortran'][:]
        scanid = my_file['scanid'][()]
        feat = my_file['feature'][()]
        
        
        airtemp = np.mean(my_file['hk_airtemp'][()])
        dewtemp = np.mean(my_file['hk_dewtemp'][()])
        humidity = np.mean(my_file['hk_humidity'][()])
        pressure = np.mean(my_file['hk_pressure'][()])
        rain = np.mean(my_file['hk_rain'][()])
        winddir = np.mean(my_file['hk_winddir'][()])
        windspeed = np.mean(my_file['hk_windspeed'][()])
        
        try:
            point_amp_ind = my_file['el_az_amp'][:,:,:,:2]
            # point_amp_ind = np.nanmean(my_file['el_az_stats'][()], axis=3) #### mean over chunk axis
            # 3, 19, 4, 1024 => 19, 4, 1024, 2
        except:
            point_amp_ind = np.zeros((n_det_ind, n_sb, 1024, 2))
        # try: 
        #     sd_ind = np.array(my_file['spike_data'])
        #     if (sd_ind.shape[0] == 0):
        #         sd_ind = np.zeros((3, n_det_ind, n_sb, 4, 1000))
        # except:
        sd_ind = np.zeros((3, n_det_ind, n_sb, 4, 1000))
        # use_freq_filter = my_file['use_freq_filter'][()]
        # if not use_freq_filter:
            # tod_poly_ind = my_file['tod_poly'][()]
        # try:
        if "poly_coeff" in my_file:
            tod_poly_ind = np.transpose(my_file['poly_coeff'][()], (1,2,0,3))
        else:
            tod_poly_ind = np.zeros((n_det_ind, n_sb, 2, n_samp))
        # except KeyError:
        #     tod_poly_ind = np.zeros((n_det_ind, n_sb, 2, n_samp))
        # try: 
        chi2_ind = np.array(my_file['chi2'])
        # except KeyError:
        #     chi2_ind = np.zeros_like(tod_ind[:,:,:,0])
        try:
            acc_ind = np.array(my_file['acceptrate'])
        except KeyError:
            acc_ind = np.zeros_like(tod_ind[:,:,0,0])
            print("Found no acceptrate")

        freqmask_reason = my_file['freqmask_reason'][()]
        freqmask_reason_string = my_file['freqmask_reason_string'][()]
        specific_freqmask = np.ones_like(mask_full_ind)
        for k in range(len(freqmask_reason_string)):
            _freqmask = (freqmask_reason == 2**k) == 0
            if freqmask_reason_string[k] not in ["Aliasing suppression (AB_mask)", "Aliasing suppression (leak_mask)",
                                                 "Tsys NaN or inf", "Tsys < min_tsys", "Tsys > running median max",
                                                 "Feed 20", "NaN or inf in TOD", "Marked channels"]:
                specific_freqmask *= _freqmask
        acc_ind_specific = np.mean(specific_freqmask, axis=-1)
        
        
        try:
            time = np.array(my_file['tod_time'])
        except:
            time = np.array(my_file['time'])
        mjd = time

        ampl_ind = np.zeros((4, *mask_full_ind.shape))
        pca = np.zeros((4, time.shape[-1]))
        # try:
        if "pca_comp" in my_file:
            # eigv = np.array(my_file['pca_eigv'])
            if np.array(my_file['pca_comp']).shape[0] >= 4:
                ampl_ind[:] = np.array(my_file['pca_ampl'])[:4]
                pca[:] = np.array(my_file['pca_comp'])[:4]
            else:
                ampl_ind[:np.array(my_file['pca_comp']).shape[0]] = np.array(my_file['pca_ampl'])[:np.array(my_file['pca_comp']).shape[0]]
                pca[:np.array(my_file['pca_comp']).shape[0]] = np.array(my_file['pca_comp'])[:]
        # except KeyError:
        else:
            pca = np.zeros((4, 10000))
            # eigv = np.zeros(0)
            # print('Found no pca comps', scanid)
        
        pcaf_ampl_ind = my_file['pca_feed_ampl'][:4]

        npca_ind = np.zeros((n_det_ind, n_sb))
        npcaf_ind = np.zeros((n_det_ind, n_sb))
        for feed in range(n_det_ind):
            try:
                npca_ind[feed,:] = my_file['n_pca_comp'][()]
                npcaf_ind[feed,:] = my_file['n_pca_feed_comp'][feed]
            except:
                pass

        try:
            tsys_ind = np.array(my_file['Tsys_lowres'])
        except KeyError:
            tsys_ind = np.zeros_like(tod_ind[:,:,:,0]) + 40
            # print("Found no tsys")
    # except KeyboardInterrupt:
    #     sys.exit()
    # except:
    #     print('Could not load file', filepath, 'returning nans')
    #     data = np.zeros((20, 4, n_stats), dtype=np.float32)
    #     data[:] = np.nan
    #     indices = np.zeros((20, 2, 2)).astype(int)
    #     map_list = [[None for _ in range(4)] for _ in range(20)]
        
    #     return data, [map_list, indices]

    t0 = time[0]
    time = (time - time[0]) * (24 * 60)  # minutes
 
    obsid = int(str(scanid)[:-2])

    n_freq_hr = len(mask_full_ind[0,0])
    n_det = 20

    data = np.zeros((n_det, n_sb, n_stats), dtype=np.float32)

    ## transform to full arrays with all pixels
    tod = np.zeros((n_det, n_sb, n_freq, n_samp))
    temporal_mask = np.zeros((n_det, n_samp), dtype=bool)
    mask = np.zeros((n_det, n_sb, n_freq))
    mask_full = np.zeros((n_det, n_sb, n_freq_hr))
    # n_nan = np.zeros((n_det, n_sb, n_freq_hr))
    acc = np.zeros((n_det, n_sb))
    acc_specific = np.zeros((n_det, n_sb))
    npca = np.zeros((n_det, n_sb))
    npcaf = np.zeros((n_det, n_sb))
    ampl = np.zeros((4, n_det, n_sb, n_freq_hr))
    pcaf_ampl = np.zeros((4, n_det, n_sb, n_freq_hr))
    tsys = np.zeros((n_det, n_sb, n_freq))
    chi2 = np.zeros((n_det, n_sb, n_freq))
    sd = np.zeros((3, n_det, n_sb, 4, 1000))
    sb_mean = np.zeros((n_det, n_sb, n_samp))
    reason = np.zeros((n_det, n_sb, n_freq_hr))
    sigma0 = np.zeros((n_det, n_sb, n_freq))
    point_amp = np.zeros((n_det, n_sb, n_freq_hr, 2))
    tod_poly = np.zeros((n_det, n_sb, 2, n_samp))
    point_tel = np.zeros((n_det, n_samp, 3))
    point_radec = np.zeros((n_det, n_samp, 3))

    tod[pixels] = tod_ind
    temporal_mask[pixels] = temporal_mask_ind
    mask[pixels] = mask_ind
    mask_full[pixels] = mask_full_ind
    # n_nan[pixels] = n_nan_ind
    reason[pixels] = reason_ind
    acc[pixels] = acc_ind
    acc_specific[pixels] = acc_ind_specific
    ampl[:, pixels, :, :] = ampl_ind
    pcaf_ampl[:, pixels, :, :] = pcaf_ampl_ind
    npca[pixels, :] = npca_ind
    npcaf[pixels, :] = npcaf_ind
    tsys[pixels] = tsys_ind
    chi2[pixels] = chi2_ind
    sd[:, pixels, :, :, :] = sd_ind
    sb_mean[pixels] = sb_mean_ind
    sigma0[pixels] = sigma0_ind
    point_amp[pixels] = point_amp_ind
    tod_poly[pixels] = tod_poly_ind
    point_tel[pixels,:,:2] = point_tel_ind
    point_radec[pixels,:,:2] = point_radec_ind

    az_amp = point_amp[:, :, :, 1]
    el_amp = point_amp[:, :, :, 0]


    mask_sum = np.nansum(mask_full.reshape((n_det, n_sb, n_freq, freq_decimation_factor)), axis=3)
    az_amp = az_amp * mask_full
    el_amp = el_amp * mask_full

    az_amp_lowres = np.nansum(az_amp.reshape((n_det, n_sb, n_freq, freq_decimation_factor)), axis=3) / mask_sum
    
    el_amp_lowres = np.nansum(el_amp.reshape((n_det, n_sb, n_freq, freq_decimation_factor)), axis=3) / mask_sum

    mask_sb_sum = np.nansum(mask_full, axis=2)
    where = (mask_sb_sum > 0)
    az_amp_sb = np.zeros_like(mask_sb_sum)
    el_amp_sb = np.zeros_like(mask_sb_sum)
    az_amp_sb[where] = np.nansum(az_amp, axis=2)[where] / mask_sb_sum[where]
    el_amp_sb[where] = np.nansum(el_amp, axis=2)[where] / mask_sb_sum[where]

    my_spikes = get_spike_list(sb_mean, sd, str(scanid), mjd)
    sortedlists = my_spikes.sorted()
    n_spikes = len(sortedlists[0])
    n_jumps = len(sortedlists[1])
    n_anom = len(sortedlists[2]) 

    # cutoff = 0.0015 * 8.0
    n_sigma_spikes = 5         # Get from param file   ########################
    n_spikes_sb = (np.array([s.sbs for s in sortedlists[0]]) > 0.0015 * n_sigma_spikes).sum(0)
    n_jumps_sb = (np.array([s.sbs for s in sortedlists[1]]) > 0.0015 * n_sigma_spikes).sum(0)
    n_anom_sb = (np.array([s.sbs for s in sortedlists[2]]) > 0.0015 * n_sigma_spikes).sum(0)
    
    mask_sb_sum_lowres = np.nansum(mask, axis=2)
    tsys_sb = np.nansum((tsys * mask), axis=2) / mask_sb_sum_lowres

    dt = (time[1] - time[0]) * 60  # seconds
    radiometer = 1 / np.sqrt(freq_bin_sizes_lowres_MHz * 10 ** 6 * dt)
    ampl = np.nanmean(np.abs(ampl), axis=3)
    ampl = 100 * np.sqrt(ampl ** 2 * pca.std(1)[:, None, None] ** 2 / radiometer ** 2)
    # ampl[np.where(ampl == 0)] = np.nan
    pcaf_ampl = np.nanmean(np.abs(pcaf_ampl), axis=3)
    pcaf_ampl = 100 * np.sqrt(pcaf_ampl ** 2 * pca.std(1)[:, None, None] ** 2 / radiometer ** 2)

    # Here comes the different diagnostic data that is calculated
    # Obsid
    insert_data_in_array(data, obsid, 'obsid')

    # Scanid
    insert_data_in_array(data, scanid, 'scanid')

    # MJD
    scan_mjd = 0.5 * (mjd[0] + mjd[-1]) 
    insert_data_in_array(data, scan_mjd, 'mjd')

    # night
    hours = (scan_mjd * 24 - 7) % 24
    close_to_night = np.minimum(np.abs(2.0 - hours), np.abs(26.0 - hours))
    insert_data_in_array(data, close_to_night, 'night')

    # sidereal time in degrees (up to a phase)
    insert_data_in_array(data, get_sid(scan_mjd), 'sidereal')
    
    # By default, no errors.
    insert_data_in_array(data, 0, 'acceptmod_error')

    # Mean az/el per feed
    mean_el = np.zeros((n_det, n_sb))
    mean_az = np.zeros((n_det, n_sb))

    # mean_az[:, :] = np.mean(point_tel[:, :, 0], axis=1)[:, None]
    mean_az[:, :] = np.arctan2(np.mean(np.sin(point_tel[:, :, 0] * np.pi / 180), axis=1), 
                             np.mean(np.cos(point_tel[:, :, 0] * np.pi / 180), axis=1)
                             )[:, None] * 180 / np.pi
    mean_az[:, :] = (mean_az[:, :] + 360) % 360
    mean_el[:, :] = np.mean(point_tel[:, :, 1], axis=1)[:, None]
    



    insert_data_in_array(data, mean_az, 'az')
    insert_data_in_array(data, mean_el, 'el')

    # chi2 
    chi2_sb = np.zeros((*chi2.shape[:2],))
    for ifeed in range(chi2.shape[0]):
        for isb in range(chi2.shape[1]):
            chi2_sb[ifeed,isb] = np.nansum(chi2[ifeed,isb, mask[ifeed,isb] != 0])
    n_freq_sb = np.nansum(mask, axis=2)
    wh = np.where(n_freq_sb != 0.0)
    chi2_sb[wh] = chi2_sb[wh] / np.sqrt(n_freq_sb[wh])
    wh = np.where(n_freq_sb == 0.0)
    chi2_sb[wh] = np.nan
    insert_data_in_array(data, chi2_sb, 'chi2')
 
    # acceptrate
    insert_data_in_array(data, acc, 'acceptrate')
    
    # acceptrate specific
    insert_data_in_array(data, acc_specific, 'acceptrate_specific')

    # azimuth binning
    nbins = 15                                ##### azimuth bins
    full_az_chi2 = np.zeros((n_det, n_sb))
    max_az_chi2 = np.zeros((n_det, n_sb))
    med_az_chi2 = np.zeros((n_det, n_sb))
    full_az_chi2[:] = np.nan
    max_az_chi2[:] = np.nan
    med_az_chi2[:] = np.nan
    az = point_tel[:, :, 0]
    for i in range(n_det):
        for j in range(n_sb):
            if acc[i, j]:
                freq_chi2 = np.zeros(n_freq)
                for k in range(n_freq): 
                    if mask[i, j, k]:
                        _az = az[i,temporal_mask[i]]
                        _tod = tod[i,j,k,temporal_mask[i]]
                        histsum, bins = np.histogram(_az, bins=nbins, weights=(_tod/sigma0[i,j,k]))
                        nhit = np.histogram(_az, bins=nbins)[0]
                        normhist = histsum / nhit * np.sqrt(nhit)

                        # if i == 15 and j == 3 and k == 24:
                        #     print(scanid)
                        #     plt.errorbar(bins[1:], normhist, yerr=1/np.sqrt(nhit), fmt='-o')
                        #     plt.show()
                        freq_chi2[k] = (np.sum(normhist ** 2) - nbins) / np.sqrt(2 * nbins)
                        # if freq_chi2[k] > 4.0:
                        #     file = open('diag_az_bins.txt', 'a')
                        #     print(scanid, i+1, j+1, k+1, freq_chi2[k], scan_mjd, mean_az, mean_el, 
                        #           chi2[i,j,k], chi2_sb[i,j], tsys[i,j,k], feat, az_amp_lowres[i,j,k],
                        #           az_amp_sb[i,j], np.argmax(normhist ** 2),
                        #           file=file)
                        #     file.close()        
                full_az_chi2[i, j] = np.sum(freq_chi2) / np.sqrt(np.sum(mask[i,j]))
                max_az_chi2[i, j] = np.max(freq_chi2)
                med_az_chi2[i, j] = np.median(freq_chi2)
    insert_data_in_array(data, full_az_chi2, 'az_chi2')
    insert_data_in_array(data, max_az_chi2, 'max_az_chi2')
    insert_data_in_array(data, med_az_chi2, 'med_az_chi2')

    # featurebit
    insert_data_in_array(data, feat, 'fbit')

    # az-amplitude
    insert_data_in_array(data, az_amp_sb, 'az_amp')

    # el-amplitude
    insert_data_in_array(data, el_amp_sb, 'el_amp')

    # number of spikes, jumps, and anomalies
    insert_data_in_array(data, n_spikes_sb, 'n_spikes')
    insert_data_in_array(data, n_jumps_sb, 'n_jumps')
    insert_data_in_array(data, n_anom_sb, 'n_anomalies')

    # number of nans
    where = (mask_sb_sum > 0)
    # n_nan_sb = np.zeros_like(mask_sb_sum)
    # n_nan_sb[where] = (n_nan * mask_full).sum(2)[where] / mask_sb_sum[where]

    # insert_data_in_array(data, n_nan_sb, 'n_nan')
    
    # tsys averaged over sb
    insert_data_in_array(data, tsys_sb, 'tsys')

    # pca modes 
    insert_data_in_array(data, npca, 'npca')
    insert_data_in_array(data, npcaf, 'npcaf')
    insert_data_in_array(data, ampl[0], 'pca1')
    insert_data_in_array(data, ampl[1], 'pca2')
    insert_data_in_array(data, ampl[2], 'pca3')
    insert_data_in_array(data, ampl[3], 'pca4')
    insert_data_in_array(data, pcaf_ampl[0], 'pcf1')
    insert_data_in_array(data, pcaf_ampl[1], 'pcf2')
    insert_data_in_array(data, np.sum(ampl+pcaf_ampl, axis=0), 'pcsm')

    # Observerlog blacklisting
    obslog_blacklist = np.load(observerlog_blacklist_filepath)
    blacklisted = obslog_blacklist[obsid]
    insert_data_in_array(data, blacklisted, "blacklisted")

    # weather statistic
    try:
        weather  = np.loadtxt(weather_filepath)
        weather  = weather[np.where(np.isclose(obsid, weather))[0]]
        ten_min_in_mjd = 1 / 24.0 / 6.0

        i_start  = int((mjd[0] - weather[0, 3]) // ten_min_in_mjd)
        i_end    = int((mjd[-1] - weather[0, 3]) // ten_min_in_mjd)
        
        n_chunks = len(weather[:, 2])
        i_start  = min(i_start, n_chunks - 1)
        i_end    = min(i_end, n_chunks - 1)

        forecast = max(weather[i_start, 2], weather[i_end, 2])
    except IndexError:
        # no weather data for this obsid
        # print('no weather data for obsid:', obsid)
        forecast = np.nan
        # insert_data_in_array(data, 1, 'acceptmod_error')
    insert_data_in_array(data, forecast, 'weather')

    # add kurtosis etc of data histogram
    kurtosis = np.zeros((n_det, n_sb))
    skewness = np.zeros((n_det, n_sb))

    for i in range(n_det):
        for j in range(n_sb):
            if acc[i,j]:
                where = np.where(mask[i, j] > 0.0)

                normtod = (tod[i,j]/sigma0[i,j,:,None])[where].flatten()
                normtod = normtod[normtod != 0]
                kurtosis[i,j] = stats.kurtosis(normtod)
                skewness[i,j] = stats.skew(normtod)

    insert_data_in_array(data, kurtosis, 'kurtosis')
    insert_data_in_array(data, skewness, 'skewness')

    # ps_chi2
    ra = point_radec[:, :, 0]
    dec = point_radec[:, :, 1]

    centre = [(np.max(ra[0]) + np.min(ra[0])) / 2, (np.max(dec[0]) + np.min(dec[0])) / 2]

    d_dec = 8.0 / 60 
    d_ra = d_dec / np.cos(centre[1] / 180 * np.pi) # arcmin


    n_pix = 16

    ra_bins2 = np.linspace(centre[0] - d_ra * n_pix / 2, centre[0] + d_ra * n_pix / 2, n_pix + 1)
    dec_bins2 = np.linspace(centre[1] - d_dec * n_pix / 2, centre[1] + d_dec * n_pix / 2, n_pix + 1)

    if feat == 128:
        field_centre = [np.mean(ra[0]), np.mean(dec[0])]
        ra_grid = map_grid[0] / np.cos(field_centre[1] * np.pi / 180) + field_centre[0]
        dec_grid = map_grid[1] + field_centre[1]
    else:
        ra_grid = map_grid[0]
        dec_grid = map_grid[1]
    # ra = dx / np.cos(field_centre[1] * np.pi / 180) + field_centre[0]
    # dec = dx + field_centre[1]

    # map_grid = np.array([ra, dec])
    

    indices = np.zeros((n_det, 2, 2)).astype(int)
    ps_chi2 = np.zeros((n_det, n_sb))
    ps_chi2[:] = np.nan 
    map_list = [[None for _ in range(n_sb)] for _ in range(n_det)]
    for i in range(n_det):
        indices[i, 0, :] = np.digitize((np.min(ra[i]), np.max(ra[i])), ra_grid)
        indices[i, 1, :] = np.digitize((np.min(dec[i]), np.max(dec[i])), dec_grid)
        # prevent overshooting
        indices[i, 0, 0] = max(1, indices[i, 0, 0])
        indices[i, 0, 1] = max(min(len(ra_grid) - 1, indices[i, 0, 1]), indices[i, 0, 0])
        indices[i, 1, 0] = max(1, indices[i, 1, 0])
        indices[i, 1, 1] = max(min(len(dec_grid) - 1, indices[i, 1, 1]), indices[i, 1, 0])

        ra_bins = ra_grid[indices[i, 0, 0] - 1:indices[i, 0, 1] + 1]
        dec_bins = dec_grid[indices[i, 1, 0] - 1:indices[i, 1, 1] + 1]
        # print(indices[0])
        # print(map_grid)
        # print(ra_bins)
        # print(dec_bins)
        # sys.exit()
        if (len(ra_bins) <= 1) or (len(dec_bins) <= 1):
            continue

        if (len(dec_bins) - 1 != indices[i, 1, 1] - indices[i, 1, 0] + 1):
            print(indices[i])
            print(dec_bins)
            print(len(dec_bins))
            print(dec_grid)
            print(len(dec_grid))
            print(dec[i])
            # sys.exit(1)

        for j in range(n_sb): ### should not need to be done per sideband.
            if acc[i, j]:
                map, nhit = make_map(ra[i], dec[i], ra_bins, dec_bins, tod[i, j], mask[i, j])
                where = np.where(nhit > 0)
                rms = np.zeros_like(nhit)
                rms[where] = (sigma0[i, j][None, None, :]/ np.sqrt(nhit))[where]
                # if i == 0 and j == 0:
                    # print(f"{scanid:9d}, {ra[i].min():2f}, {ra[i].max():.2f}, {dec[i].min():.2f}, {dec[i].max():.2f}, {ra_bins.shape[0]:3d}, {dec_bins.shape[0]:3d}, {ra_bins[0]:.2f}, {ra_bins[-1]:.2f}, {dec_bins[0]:.2f}, {dec_bins[-1]:.2f}, {np.nanmin(rms[rms!=0]):.5f}, {np.nanmax(nhit):.0f}")
                #print(np.nanstd((tod[i, j, :, :] / sigma0[i, j, :, None]).flatten()))
                #print(np.std(map[where] / rms[where]))
                map_list[i][j] = [map, rms]
                ps_chi2[i, j], Pk, ps_mean, ps_std, transfer, map, rms = get_sb_ps(ra[0], dec[0], ra_bins2, dec_bins2, tod[i, j], mask[i, j], sigma0[i, j], d_dec)
                # np.save(f"data_test_WN/{obsid_info.scans[l]}_ps_chi2.npy", ps_chi2)
                # np.save(f"data_test_WN/{obsid_info.scans[l]}_ps_map.npy", map)
                # np.save(f"data_test_WN/{obsid_info.scans[l]}_ps_map_rms.npy", rms)
                # np.save(f"data_test_WN/{obsid_info.scans[l]}_ps_misc.npy", np.array([n_k, d_th, dz]))

    #np.save('ps_chi2_scan', ps_chi2)
    insert_data_in_array(data, ps_chi2, 'ps_chi2')
    
    # add length of scan
    duration = (mjd[-1] - mjd[0]) * 24 * 60  # in minutes
    insert_data_in_array(data, duration, 'scan_length')
    
    # saddlebags
    saddlebags = np.zeros((n_det, n_sb))
    saddlebags[(0, 3, 4, 11, 12), :] = 1  # feeds 1, 4, 5, 13, 14
    saddlebags[(5, 13, 14, 15, 16), :] = 2  # feeds 6, 14, 15, 16, 17
    saddlebags[(1, 6, 17, 18, 19), :] = 3  # feeds 2, 7, 18, 19, (20)
    saddlebags[(2, 7, 8, 9, 10), :] = 4  # feeds 3, 8, 9, 10, 11
    insert_data_in_array(data, saddlebags, 'saddlebag')
    
    # add one over f of polyfilter components
    sigma_poly = np.zeros((n_det, n_sb, 2))
    fknee_poly = np.zeros((n_det, n_sb, 2))
    alpha_poly = np.zeros((n_det, n_sb, 2))
    sigma_poly[:] = np.nan
    fknee_poly[:] = np.nan
    alpha_poly[:] = np.nan
    for i in range(n_det):
        for j in range(n_sb):
            if acc[i, j]:
                for l in range(2):
                    sigma_poly[i,j,l], fknee_poly[i,j,l], alpha_poly[i,j,l] = get_noise_params(tod_poly[i,j,l])
                    if np.isinf(sigma_poly[i,j,l]):
                        pass
                        # print('unable to fit noise params', scanid, i, j, l)
                    elif np.isnan(sigma_poly[i,j,l]):
                        print('nan in timestream', scanid, i, j, l)

    insert_data_in_array(data, sigma_poly[:,:,0], 'sigma_poly0')
    insert_data_in_array(data, fknee_poly[:,:,0], 'fknee_poly0')
    insert_data_in_array(data, alpha_poly[:,:,0], 'alpha_poly0')
    insert_data_in_array(data, sigma_poly[:,:,1], 'sigma_poly1')
    insert_data_in_array(data, fknee_poly[:,:,1], 'fknee_poly1')
    insert_data_in_array(data, alpha_poly[:,:,1], 'alpha_poly1')

    # sb_mean 
    power_mean = np.zeros((n_det, n_sb))
    sigma_mean = np.zeros((n_det, n_sb))
    fknee_mean = np.zeros((n_det, n_sb))
    alpha_mean = np.zeros((n_det, n_sb))
    power_mean[:] = np.nan
    sigma_mean[:] = np.nan
    fknee_mean[:] = np.nan
    alpha_mean[:] = np.nan
    for i in range(n_det):
        for j in range(n_sb):
            if acc[i, j]:
                power_mean[i,j] = np.mean(sb_mean[i,j])
                sigma_mean[i,j], fknee_mean[i,j], alpha_mean[i,j] = get_noise_params(sb_mean[i,j])
                if np.isinf(sigma_mean[i,j]):
                    pass
                    # print('unable to fit noise params', scanid, i, j)
                elif np.isnan(sigma_mean[i,j]):
                    print(np.argwhere(np.isnan(sb_mean[i,j])))
                    print('nan in timestream', scanid, i, j)


    insert_data_in_array(data, power_mean[:,:], 'power_mean')
    insert_data_in_array(data, sigma_mean[:,:], 'sigma_mean')
    insert_data_in_array(data, fknee_mean[:,:], 'fknee_mean')
    insert_data_in_array(data, alpha_mean[:,:], 'alpha_mean')

    # Housekeeping data
    insert_data_in_array(data, airtemp, 'airtemp')
    insert_data_in_array(data, dewtemp, 'dewtemp')
    insert_data_in_array(data, humidity, 'humidity')
    insert_data_in_array(data, pressure, 'pressure')
    insert_data_in_array(data, rain, 'rain')
    insert_data_in_array(data, winddir, 'winddir')
    insert_data_in_array(data, windspeed, 'windspeed')


    # sun and moon position
    mean_el = mean_el[:, 0]
    mean_az = mean_az[:, 0]

    moon_dist = np.zeros((n_det, n_sb))
    moon_angle = np.zeros((n_det, n_sb))
    moon_central_sl = np.zeros((n_det, n_sb))
    moon_outer_sl = np.zeros((n_det, n_sb))

    sun_dist = np.zeros((n_det, n_sb))
    sun_angle = np.zeros((n_det, n_sb))
    sun_central_sl = np.zeros((n_det, n_sb))
    sun_outer_sl = np.zeros((n_det, n_sb))

    sun_elevation = np.zeros((n_det, n_sb))

    with solar_system_ephemeris.set('builtin'):
        loc = coord.EarthLocation(lon=-118.283 * u.deg, lat=37.2313 * u.deg)
        time = Time(scan_mjd, format='mjd')
        pole = np.array([mean_el, mean_az])
        aa = AltAz(location=loc, obstime=time)

        sun = get_body('sun', time, loc)
        cs = sun.transform_to(aa)
        sun_elevation[:, :] = cs.alt.deg

        lat, lon = move_to_frame(pole, [cs.alt.deg, cs.az.deg])
        theta_sun = 90 - lat
        phi_sun = lon

        sun_dist[:, :] = theta_sun[:, None]
        sun_angle[:, :] = phi_sun[:, None]
        sun_angle_mod90 = sun_angle % 90
        sun_central_sl = 1.0 * (sun_dist < 40) + 1.0 * (sun_dist < 30)
        cond_1 = (sun_dist > 58.0) * (sun_dist < 75.0) * (sun_angle_mod90 > 75.0)
        cond_2 = (sun_dist > 58.0) * (sun_dist < 75.0) * (sun_angle_mod90 < 15.0)
        cond_3 = (sun_dist > 63.0) * (sun_dist < 70.0) * (sun_angle_mod90 > 82.0)
        cond_4 = (sun_dist > 63.0) * (sun_dist < 70.0) * (sun_angle_mod90 < 8.0)
        sun_outer_sl = 1.0 * cond_1 + 1.0 * cond_2 + 1.0 * cond_3 + 1.0 * cond_4  # can never be more than 2.0

        moon = get_body('moon', time, loc)
        cm = moon.transform_to(aa)

        lat, lon = move_to_frame(pole, [cm.alt.deg, cm.az.deg])
        theta_moon = 90 - lat
        phi_moon = lon
        
        moon_dist[:, :] = theta_moon[:, None]
        moon_angle[:, :] = phi_moon[:, None]
        moon_angle_mod90 = moon_angle % 90
        moon_central_sl = 1.0 * (moon_dist < 40) + 1.0 * (moon_dist < 30)
        cond_1 = (moon_dist > 58.0) * (moon_dist < 75.0) * (moon_angle_mod90 > 75.0)
        cond_2 = (moon_dist > 58.0) * (moon_dist < 75.0) * (moon_angle_mod90 < 15.0)
        cond_3 = (moon_dist > 63.0) * (moon_dist < 70.0) * (moon_angle_mod90 > 82.0)
        cond_4 = (moon_dist > 63.0) * (moon_dist < 70.0) * (moon_angle_mod90 < 8.0)
        moon_outer_sl = 1.0 * cond_1 + 1.0 * cond_2 + 1.0 * cond_3 + 1.0 * cond_4  # can never be more than 2.0

    insert_data_in_array(data, moon_dist, 'moon_dist')
    insert_data_in_array(data, moon_angle, 'moon_angle')
    insert_data_in_array(data, moon_central_sl, 'moon_cent_sl')
    insert_data_in_array(data, moon_outer_sl, 'moon_outer_sl')
    insert_data_in_array(data, sun_dist, 'sun_dist')
    insert_data_in_array(data, sun_angle, 'sun_angle')
    insert_data_in_array(data, sun_central_sl, 'sun_cent_sl')
    insert_data_in_array(data, sun_outer_sl, 'sun_outer_sl')

    insert_data_in_array(data, sun_elevation, 'sun_el')

    ### perhaps a ps_xy and ps_z to distinguish frequency residuals from angular ones
    
    filename = '/mn/stornext/d22/cmbco/comap/d16/protodir/sw_complete_' + fieldname + '.h5'
    i_scan = int(str(scanid)[-2:]) - 2  # goes from 0 to n_scan
    #print(scanid, i_scan)
    n_sw = 14
    sw_array = np.zeros((n_det, n_sw, n_sb))
    try:
        with h5py.File(filename, mode="r") as my_file:
             sw = my_file['%07i/sw_stats' % obsid][()][:, i_scan]
        sw_array[:, :, :] = sw[:, :, None]
    except:
        # print('problems with standing waves')
        sw_array[:, :, :] = np.nan

    for i in range(n_sw):
        sw_str = 'sw_%02i' % (i + 1)
        insert_data_in_array(data, sw_array[:, i, :], sw_str)
   

     ######## Here you can add new statistics  ##########
   
    
    return data, [map_list, indices]

def move_to_frame(ang_cent, ang):
    # ang = {theta, phi}
    lat = ang[0] * np.pi / 180.0
    lon = ang[1] * np.pi / 180.0

    pos = np.zeros((3))
    
    pos[0] = np.cos(lat) * np.cos(lon)
    pos[1] = np.cos(lat) * np.sin(lon)
    pos[2] = np.sin(lat)

    lat_frame = ang_cent[0] * np.pi / 180.0
    lon_frame = ang_cent[1] * np.pi / 180.0

    pos_rotx = np.zeros((3, len(ang_cent[0])))
    # rotate around z-axis
    pos_rotx[0] = np.cos(lon_frame) * pos[0] + np.sin(lon_frame) * pos[1]
    pos_rotx[1] = - np.sin(lon_frame) * pos[0] + np.cos(lon_frame) * pos[1]
    pos_rotx[2] = pos[2]

    pos_roty = np.zeros_like(pos_rotx)
    # rotate around new y-axis
    pos_roty[0] = np.cos(np.pi / 2 - lat_frame) * pos_rotx[0] - np.sin(np.pi /2 - lat_frame) * pos_rotx[2]
    pos_roty[2] = np.sin(np.pi / 2 - lat_frame) * pos_rotx[0] + np.cos(np.pi /2 - lat_frame) * pos_rotx[2]
    pos_roty[1] = pos_rotx[1]

    out_ang = np.zeros((2, len(ang_cent[0])))
    out_ang[0] = np.arctan2(pos_roty[2], np.sqrt(pos_roty[0] ** 2 + pos_roty[1] ** 2)) #np.arccos(pos_roty[2])
    out_ang[1] = np.arctan2(pos_roty[1], pos_roty[0])

    return out_ang * 180 / np.pi

def pad_nans(tod):
    n_pad = 10
    nan_indices = np.argwhere(np.isnan(tod))
    
    for nan_index in nan_indices:
        start_ind = int(max([nan_index - n_pad, 0]))
        end_ind = int(min(([nan_index + n_pad, len(tod) - 1])))
        mean = np.nanmean(tod[start_ind:end_ind])
        std = np.nanstd(tod[start_ind:end_ind])
        tod[nan_index] = mean + np.random.randn() * std
    return tod


def get_noise_params(tod, samprate=50.0):
    tod = tod[:-20]
    if not np.isfinite(np.mean(tod)):
        tod = pad_nans(tod)
    dt = 1 / samprate  # seconds

    n = len(tod)
    freq = fft.rfftfreq(n, dt)
    p = np.abs(fft.rfft(tod)) ** 2 / (n)
    bins = np.logspace(-2, 1, 20)
    nmodes = np.histogram(freq, bins=bins)[0]
    bin_freqs = np.histogram(freq, bins=bins, weights=freq)[0] / nmodes
    ps = np.histogram(freq, bins=bins, weights=p)[0] / nmodes

    sigma0 = np.std(tod[1:] - tod[:-1]) / np.sqrt(2)

    def one_over_f(freq, alpha, fknee):
        return sigma0 ** 2 * (1.0 + (freq / fknee) ** alpha)

    try: 
        p0 = (-2, 5)
        
        popt, pcov = curve_fit(one_over_f, bin_freqs, ps, p0=p0, sigma=ps/np.sqrt(nmodes))
        alpha = popt[0]
        fknee = popt[1]
    except:
        try:
            a = -1  # solve for alpha
            p0 = (a, 10)
            
            popt, pcov = curve_fit(one_over_f, bin_freqs, ps, p0=p0, sigma=ps/np.sqrt(nmodes))
            alpha = popt[0]
            fknee = popt[1]
        except: 
            return np.inf, np.inf, np.inf
    #     print('unable to fit noise parameters')
    #     print(ps)
    #     # return np.nan, np.nan, np.nan
    #     p0 = (-1, 10)
    #     plt.loglog(freq, p)
    #     plt.loglog(freq, sigma0 ** 2 + 0.0 * freq)
    #     plt.loglog(bin_freqs, ps) 
    #     plt.loglog(bin_freqs, one_over_f(bin_freqs, *p0), 'g--', label='fit: alpha=%5.3f, fknee=%5.3f' % tuple(p0))
    #     plt.show()
    #     popt, pcov = curve_fit(one_over_f, bin_freqs, ps, p0=p0, sigma=ps/np.sqrt(nmodes))
    #     sys.exit()
    #     # 
    return sigma0, fknee, alpha

class ObsidData():
    def __init__(self):
        pass

def get_scan_data(params, fields, fieldname, paralellize=True):
    WORK_TAG = 1
    DIE_TAG = 2

    if rank == 0:
        l2_path = params.level2_dir
        n_freqs = params.decimation_freqs
        field = fields[fieldname]
        tot_scans = field[2]
        n_feeds = 20
        n_sb = 4
        n_stats = len(stats_list)
        
        scan_list = np.zeros((tot_scans), dtype=np.int32)
        scan_data = np.zeros((tot_scans, n_feeds, n_sb, n_stats), dtype=np.float32)

        i_scan = 0
        obsid_infos = []
        for obsid in field[0]:
            scans = field[1][obsid]
            n_scans = len(scans)
            obsid_info = ObsidData()
            obsid_info.scans = scans
            obsid_info.n_freqs = n_freqs
            obsid_info.field = fieldname
            obsid_info.l2_path = l2_path
            obsid_infos.append(obsid_info)
            i_scan += n_scans
        
        n_tasks = len(obsid_infos) 
        print(f"Total of {n_tasks} tasks (obsids).")
        i_scan = 0
        tasks_done = 0
        tasks_started = 0
        proc_order = np.arange(1, Nproc)
        np.random.shuffle(proc_order)

        progress_bar = tqdm(total=n_tasks, ncols=80)
        for iproc in range(Nproc-1):
            comm.send(obsid_infos[tasks_started], dest=proc_order[iproc], tag=WORK_TAG)
            tasks_started += 1
            if tasks_started == n_tasks:  # If we have more processes than tasks, kill the rest, and break the task-assigment loop.
                for iirank in range(iproc, Nproc-1):
                    comm.send(-1, dest=proc_order[iirank], tag=DIE_TAG)
                break
            if params.distributed_starting:
                time.sleep(0.5)
            time.sleep(0.01)

        while tasks_started < n_tasks:
            status = MPI.Status()
            scan_data_list = comm.recv(source=MPI.ANY_SOURCE, status=status)
            tasks_done += 1
            workerID = status.Get_source()
            comm.send(obsid_infos[tasks_started], dest=workerID, tag=WORK_TAG)
            tasks_started += 1
            n_scans = len(scan_data_list)
            scan_data[i_scan:i_scan+n_scans] = scan_data_list
            scan_list[i_scan:i_scan+n_scans] = [int(scaninfo[0,0,1]) for scaninfo in scan_data_list]
            i_scan += n_scans
            progress_bar.update(1)

        while tasks_done < n_tasks:
            status = MPI.Status()
            scan_data_list = comm.recv(source=MPI.ANY_SOURCE, status=status)
            tasks_done += 1
            workerID = status.Get_source()
            n_scans = len(scan_data_list)
            scan_data[i_scan:i_scan+n_scans] = scan_data_list
            scan_list[i_scan:i_scan+n_scans] = [int(scaninfo[0,0,1]) for scaninfo in scan_data_list]
            i_scan += n_scans
            progress_bar.update(1)
            workerID = status.Get_source()
            comm.send(-1, dest=workerID, tag=DIE_TAG)
            time.sleep(0.01)


        for iproc in range(1, Nproc):
            comm.send(-1, dest=iproc, tag=DIE_TAG)

        # print(scan_data.shape)
        # print(scan_data[0,0,0])
        # scan_idx_in_scanid = np.argwhere(stats_list == "scanid")[0]
        # print("Finalizing MPI section.")
        # for iscan in range(tot_scans):
        #     scan_data[iscan] = scan_data[iscan,0,0,1]
        
        # print(scan_data.shape)
        # print(scan_data[0,0,0])

    
    else:
        while True:
            status = MPI.Status()
            obsid_info = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
            # print(f"Worker {rank} recieved work with tag {status.Get_tag()}.", flush=True)
            if status.Get_tag() == DIE_TAG:
                break
            scan_data_list = get_obsid_data(obsid_info)
            comm.send(scan_data_list, dest=0)

    if rank == 0:
        sort_idxs = np.argsort(scan_list)
        scan_list = scan_list[sort_idxs]
        scan_data = scan_data[sort_idxs]
        return scan_list, scan_data
    else:
        return None, None


def get_obsid_data(obsid_info):
    scans = obsid_info.scans
    fieldname = obsid_info.field
    l2_path = obsid_info.l2_path
    n_freqs = obsid_info.n_freqs
    
    ## set up map grid
    info = patch_info[fieldname]
    if fieldname == "NCP":
        field_centre = [0, 0]
    else:
        field_centre = np.array(info[:2]).astype(float)
    map_radius = int(info[2])  # degrees
    pix_side = int(info[4]) * 4  # 8 arcmin

    dx = np.linspace(-map_radius, map_radius, map_radius * 60 // pix_side + 1)
    ra = dx / np.cos(field_centre[1] * np.pi / 180) + field_centre[0]
    dec = dx + field_centre[1]

    map_grid = np.array([ra, dec])
    # print(map_grid)

    n_scans = len(scans)
    n_stats = len(stats_list)
    n_feeds = 20
    n_sb = 4
    scan_data = np.zeros((n_scans, n_feeds, n_sb, n_stats), dtype=np.float32)
    maps = []
    i_scan = 0
    for scanid in scans:
        filepath = os.path.join(l2_path, fieldname) + '/' + fieldname + '_0' + scanid + '.h5'
        #filepath = l2_path + '/' + fieldname + '_0' + scanid + '.h5'
        data, map = get_scan_stats(filepath, map_grid)
        scan_data[i_scan] = data
        maps.append(map)
        i_scan += 1
    
    ps_s_sb_chi2, ps_s_feed_chi2, ps_s_chi2, ps_o_sb_chi2, ps_o_feed_chi2, ps_o_chi2, ps_z_s_sb_chi2, ps_xy_s_sb_chi2 = get_power_spectra(maps, map_grid, n_freqs, obsid_info)

    insert_data_in_array(scan_data, ps_s_sb_chi2, 'ps_s_sb_chi2', obsid=True)
    insert_data_in_array(scan_data, ps_s_feed_chi2, 'ps_s_feed_chi2', obsid=True)
    insert_data_in_array(scan_data, ps_s_chi2, 'ps_s_chi2', obsid=True)
    insert_data_in_array(scan_data, ps_o_sb_chi2, 'ps_o_sb_chi2', obsid=True)
    insert_data_in_array(scan_data, ps_o_feed_chi2, 'ps_o_feed_chi2', obsid=True)
    insert_data_in_array(scan_data, ps_o_chi2, 'ps_o_chi2', obsid=True)

    insert_data_in_array(scan_data, ps_z_s_sb_chi2, 'ps_z_s_sb_chi2', obsid=True)
    insert_data_in_array(scan_data, ps_xy_s_sb_chi2, 'ps_xy_s_sb_chi2', obsid=True)

    ## [map_list, indices]
    # ## map_list[i][j] = [map, rms]
    # print('exiting')
    # sys.exit()
    return scan_data


def get_power_spectra(maps, map_grid, n_freqs, obsid_info):
    if int(obsid_info.scans[0][:-2]) > 28500:
        is_slow=True
    else:
        is_slow=False
    n_feeds = 20
    n_sb = 4
    n_k = 10
    n_scans = len(maps)
    ra, dec = map_grid
    h = 0.7
    deg2Mpc = 76.22 / h
    GHz2Mpc = 699.62 / h * (1 + 2.9) ** 2 / 115

    d_dec = dec[1] - dec[0]
    d_th = d_dec * deg2Mpc

    dz = 32.2e-3 * GHz2Mpc

    ps_s_sb_chi2 = np.zeros((n_scans, n_feeds, n_sb))
    ps_s_feed_chi2 = np.zeros((n_scans, n_feeds, n_sb))
    ps_s_chi2 = np.zeros((n_scans, n_feeds, n_sb))
    # ps_s_stackp_chi2 = np.zeros((n_scans, n_feeds, n_sb))
    # ps_s_stackfp_chi2 = np.zeros((n_scans, n_feeds, n_sb))
    ps_o_sb_chi2 = np.zeros((n_scans, n_feeds, n_sb))
    ps_o_feed_chi2 = np.zeros((n_scans, n_feeds, n_sb))
    ps_o_chi2 = np.zeros((n_scans, n_feeds, n_sb))
    
    ps_xy_s_sb_chi2 = np.zeros((n_scans, n_feeds, n_sb))
    ps_z_s_sb_chi2 = np.zeros((n_scans, n_feeds, n_sb))
    
    # ps_o_stackp_chi2 = np.zeros((n_scans, n_feeds, n_sb))
    # ps_o_stackfp_chi2 = np.zeros((n_scans, n_feeds, n_sb))
    sum_obsid = np.zeros((len(ra) - 1, len(dec) - 1, n_sb, n_freqs))  # np.zeros((len(ra), len(dec), 64))
    div_obsid = np.zeros_like(sum_obsid)
    sum_sb_obsid = np.zeros((n_feeds, len(ra) - 1, len(dec) - 1, n_sb, n_freqs))  # np.zeros((len(ra), len(dec), 64))
    div_sb_obsid = np.zeros_like(sum_sb_obsid)
    ind_feed = []  # np.zeros((n_scans, n_feeds, 2, 2)).astype(int)
    accepted = np.zeros((n_scans, n_feeds, n_sb))
    for l in range(n_scans):  # need tests for if a scan is 
        map_list, indices = maps[l]
        # if l == 0:
        #     with open('scan1map.pkl','wb') as my_file:
        #         pickle.dump(maps[l],my_file)
            # np.save('scan1map.npy', maps[l])
        sum_scan = np.zeros((len(ra) - 1, len(dec) - 1, n_sb, n_freqs))  # np.zeros((len(ra), len(dec), 64))
        div_scan = np.zeros_like(sum_scan)
        ind_feed.append(indices)
        for i in range(n_feeds):
            # ra_bins = map_grid[0][indices[i, 0, 0] - 1:indices[i, 0, 1] + 1]
            # dec_bins = map_grid[1][indices[i, 1, 0] - 1:indices[i, 1, 1] + 1]
            ind = indices[i]
            # print(indices)
            map_feed = np.zeros((ind[0, 1] - ind[0, 0] + 1, ind[1, 1] - ind[1, 0] + 1, n_sb, n_freqs))  # np.zeros((len(ra), len(dec), 64))
            rms_feed = np.zeros_like(map_feed)
            for j in range(n_sb):
                if not map_list[i][j]:
                    ps_s_sb_chi2[l, i, j] = np.nan
                    ps_z_s_sb_chi2[l, i, j] = np.nan
                    ps_xy_s_sb_chi2[l, i, j] = np.nan
                else:
                    accepted[l, i, j] = 1.0
                    map, rms = map_list[i][j]   ######### ######################### flip frequencies!! ############################# #################
                    
                    
                    # print(map[:,:,10])
                    # print(rms[:,:,10])
                    ps_s_sb_chi2[l, i, j] = get_ps_chi2(map, rms, n_k, d_th, dz, is_slow=is_slow, name="s_sb")  # , Pk, ps_mean, ps_std, transfer
                    if len(params.accept_mod_debug_maps_dir) > 0:
                        np.save(f"{params.accept_mod_debug_maps_dir}/{obsid_info.scans[l]}_ps_s_sb_chi2.npy", ps_s_sb_chi2)
                        np.save(f"{params.accept_mod_debug_maps_dir}/{obsid_info.scans[l]}_ps_s_sb_map.npy", map)
                        np.save(f"{params.accept_mod_debug_maps_dir}/{obsid_info.scans[l]}_ps_s_sb_map_rms.npy", rms)
                        np.save(f"{params.accept_mod_debug_maps_dir}/{obsid_info.scans[l]}_ps_s_sb_misc.npy", np.array([n_k, d_th, dz]))

                    ps_z_s_sb_chi2[l, i, j], ps_xy_s_sb_chi2[l, i, j] = get_ps_1d2d_chi2(map, rms, n_k, d_th, dz)
                    if len(params.accept_mod_debug_maps_dir) > 0:
                        np.save(f"{params.accept_mod_debug_maps_dir}/{obsid_info.scans[l]}_ps_z_s_sb_chi2.npy", ps_z_s_sb_chi2)
                        np.save(f"{params.accept_mod_debug_maps_dir}/{obsid_info.scans[l]}_ps_xy_s_sb_chi2.npy", ps_xy_s_sb_chi2)
                        np.save(f"{params.accept_mod_debug_maps_dir}/{obsid_info.scans[l]}_ps_z_xy_s_sb_map.npy", map)
                        np.save(f"{params.accept_mod_debug_maps_dir}/{obsid_info.scans[l]}_ps_z_xy_s_sb_map_rms.npy", rms)
                        np.save(f"{params.accept_mod_debug_maps_dir}/{obsid_info.scans[l]}_ps_z_xy_s_sb_misc.npy", np.array([n_k, d_th, dz]))

                    chi2 = ps_s_sb_chi2[l, i, j]
                    # if np.isnan(chi2):
                    #     print("Nan in chi2")
                    #     print(map[np.isnan(map)])
                    #     print(rms[np.isnan(rms)])
                        # sys.exit()
                    map_feed[:, :, j, :] = map
                    rms_feed[:, :, j, :] = rms
                    
                    where = np.where(rms > 0)
                    sum_sb_obsid[i, ind[0, 0] - 1:ind[0, 1], ind[1, 0] - 1:ind[1, 1], j, :][where] += map[where] / rms[where] ** 2
                    div_sb_obsid[i, ind[0, 0] - 1:ind[0, 1], ind[1, 0] - 1:ind[1, 1], j, :][where] += 1.0 / rms[where] ** 2 
            if np.sum(accepted[l, i, :]) == 0:
                ps_s_feed_chi2[l, i, :] = np.nan
            else:
                # if ((l == 0) and (i == 0)):
                #     np.save('map_feed.npy', map_feed)
                #     np.save('rms_feed.npy', rms_feed)

                sh = map_feed.shape

                ps_s_feed_chi2[l, i, :] = get_ps_chi2(
                    map_feed.reshape((sh[0], sh[1], n_sb * n_freqs)),
                    rms_feed.reshape((sh[0], sh[1], n_sb * n_freqs)),
                    n_k, d_th, dz, is_feed=True, is_slow=is_slow, name="s_feed")
                if len(params.accept_mod_debug_maps_dir) > 0:
                    np.save(f"{params.accept_mod_debug_maps_dir}/{obsid_info.scans[l]}_ps_s_feed_chi2.npy", ps_s_feed_chi2)
                    np.save(f"{params.accept_mod_debug_maps_dir}/{obsid_info.scans[l]}_ps_s_feed_map.npy", map_feed.reshape((sh[0], sh[1], n_sb * n_freqs)))
                    np.save(f"{params.accept_mod_debug_maps_dir}/{obsid_info.scans[l]}_ps_s_feed_map_rms.npy", rms_feed.reshape((sh[0], sh[1], n_sb * n_freqs)))
                    np.save(f"{params.accept_mod_debug_maps_dir}/{obsid_info.scans[l]}_ps_s_feed_misc.npy", np.array([n_k, d_th, dz]))
                where = np.where(rms_feed > 0.0)
                sum_scan[indices[i, 0, 0] - 1:indices[i, 0, 1], indices[i, 1, 0] - 1:indices[i, 1, 1], :, :][where] += map_feed[where] / rms_feed[where] ** 2 
                div_scan[indices[i, 0, 0] - 1:indices[i, 0, 1], indices[i, 1, 0] - 1:indices[i, 1, 1], :, :][where] += 1.0 / rms_feed[where] ** 2 
        if np.sum(accepted[l, :, :].flatten()) == 0:
            ps_s_chi2[l, :, :] = np.nan
        else:        
            map_scan = np.zeros_like(sum_scan)
            rms_scan = np.zeros_like(sum_scan)
            where = np.where(div_scan > 0.0)
            map_scan[where] = sum_scan[where] / div_scan[where]
            rms_scan[where] = np.sqrt(1.0 / div_scan[where])
            sh = map_scan.shape
            map_scan = map_scan.reshape((sh[0], sh[1], n_sb * n_freqs))
            rms_scan = rms_scan.reshape((sh[0], sh[1], n_sb * n_freqs))
            indices = np.ma.masked_equal(indices, 0, copy=False).astype(int)
            min_ind = np.min(indices[:, :, 0], axis=0)  ## only use the non-masked sidebands
            max_ind = np.max(indices[:, :, 1], axis=0)
            # print(min_ind, max_ind)
            # print(map_scan.shape)
            ps_s_chi2[l, :, :] = get_ps_chi2(
                    map_scan[min_ind[0] -1:max_ind[0], min_ind[1] -1:max_ind[1]],
                    rms_scan[min_ind[0] -1:max_ind[0], min_ind[1] -1:max_ind[1]],
                    n_k, d_th, dz, is_feed=True, is_slow=is_slow, name="s")
            if len(params.accept_mod_debug_maps_dir) > 0:
                np.save(f"{params.accept_mod_debug_maps_dir}/{obsid_info.scans[l]}_ps_s_chi2.npy", ps_s_chi2)
                np.save(f"{params.accept_mod_debug_maps_dir}/{obsid_info.scans[l]}_ps_s_map.npy", map_scan[min_ind[0] -1:max_ind[0], min_ind[1] -1:max_ind[1]])
                np.save(f"{params.accept_mod_debug_maps_dir}/{obsid_info.scans[l]}_ps_s_map_rms.npy", rms_scan[min_ind[0] -1:max_ind[0], min_ind[1] -1:max_ind[1]])
                np.save(f"{params.accept_mod_debug_maps_dir}/{obsid_info.scans[l]}_ps_s_misc.npy", np.array([n_k, d_th, dz]))
            sum_obsid[where] += sum_scan[where]
            div_obsid[where] += div_scan[where]
    if np.sum(accepted[:, :, :].flatten()) == 0:
        ps_o_sb_chi2[:] = np.nan
        ps_o_feed_chi2[:] = np.nan
        ps_o_chi2[:] = np.nan
    else:
        map_obsid = np.zeros_like(sum_obsid)
        rms_obsid = np.zeros_like(sum_obsid)
        where = np.where(div_obsid > 0.0)
        map_obsid[where] = sum_obsid[where] / div_obsid[where]
        rms_obsid[where] = np.sqrt(1.0 / div_obsid[where])
        sh = map_obsid.shape
        map_obsid = map_obsid.reshape((sh[0], sh[1], n_sb * n_freqs))
        rms_obsid = rms_obsid.reshape((sh[0], sh[1], n_sb * n_freqs))

        ind_feed = np.array(ind_feed).astype(int)
        ind_feed = np.ma.masked_equal(ind_feed, 0, copy=False).astype(int)
        min_ind = np.min(ind_feed[:, :, :, 0], axis=(0, 1))
        max_ind = np.max(ind_feed[:, :, :, 1], axis=(0, 1))
        ps_o_chi2[:, :, :] = get_ps_chi2(
                    map_obsid[min_ind[0] -1:max_ind[0], min_ind[1] -1:max_ind[1]],
                    rms_obsid[min_ind[0] -1:max_ind[0], min_ind[1] -1:max_ind[1]],
                    n_k, d_th, dz, is_feed=True, is_slow=is_slow, name="o")
        if len(params.accept_mod_debug_maps_dir) > 0:
            np.save(f"{params.accept_mod_debug_maps_dir}/{obsid_info.scans[0][:-2]}_ps_o_chi2.npy", ps_o_chi2)
            np.save(f"{params.accept_mod_debug_maps_dir}/{obsid_info.scans[0][:-2]}_ps_o_map.npy", map_obsid[min_ind[0] -1:max_ind[0], min_ind[1] -1:max_ind[1]])
            np.save(f"{params.accept_mod_debug_maps_dir}/{obsid_info.scans[0][:-2]}_ps_o_map_rms.npy", rms_obsid[min_ind[0] -1:max_ind[0], min_ind[1] -1:max_ind[1]])
            np.save(f"{params.accept_mod_debug_maps_dir}/{obsid_info.scans[0][:-2]}_ps_o_misc.npy", np.array([n_k, d_th, dz]))

        # sum_sb_obsid = np.zeros((n_feeds, len(ra), len(dec), n_sb, 64)) 
        for i in range(n_feeds):
            min_ind = np.min(ind_feed[:, i, :, 0], axis=0)
            max_ind = np.max(ind_feed[:, i, :, 1], axis=0)
                
            for j in range(n_sb):
                if np.sum(accepted[:, i, j]) == 0:
                    ps_o_sb_chi2[:, i, j] = np.nan
                else: 
                    map_sb = np.zeros((len(ra), len(dec), n_freqs))
                    rms_sb = np.zeros((len(ra), len(dec), n_freqs))
                    where = np.where(div_sb_obsid[i, :, :, j, :] > 0)
                    map_sb[where] = sum_sb_obsid[i, :, :, j, :][where] / div_sb_obsid[i, :, :, j, :][where]
                    rms_sb[where] = np.sqrt(1.0 / div_sb_obsid[i, :, :, j, :][where])
                    ps_o_sb_chi2[:, i, j] = get_ps_chi2(
                        map_sb[min_ind[0] -1:max_ind[0], min_ind[1] -1:max_ind[1]],
                        rms_sb[min_ind[0] -1:max_ind[0], min_ind[1] -1:max_ind[1]],
                        n_k, d_th, dz, is_slow=is_slow, name="o_sb")
                    if len(params.accept_mod_debug_maps_dir) > 0:
                        np.save(f"{params.accept_mod_debug_maps_dir}/{obsid_info.scans[0][:-2]}_f{i}_sb{j}_ps_o_sb_chi2.npy", ps_o_sb_chi2)
                        np.save(f"{params.accept_mod_debug_maps_dir}/{obsid_info.scans[0][:-2]}_f{i}_sb{j}_ps_o_sb_map.npy", map_sb[min_ind[0] -1:max_ind[0], min_ind[1] -1:max_ind[1]])
                        np.save(f"{params.accept_mod_debug_maps_dir}/{obsid_info.scans[0][:-2]}_f{i}_sb{j}_ps_o_sb_map_rms.npy", rms_sb[min_ind[0] -1:max_ind[0], min_ind[1] -1:max_ind[1]])
                        np.save(f"{params.accept_mod_debug_maps_dir}/{obsid_info.scans[0][:-2]}_f{i}_sb{j}_ps_o_sb_misc.npy", np.array([n_k, d_th, dz]))

            if np.sum(accepted[:, i, :].flatten()) == 0:
                ps_o_feed_chi2[:, i, :] = np.nan
            else:   
                map_feed = np.zeros((len(ra), len(dec), n_sb, n_freqs)) #np.zeros((max_ind[0] - min_ind[0] + 2, max_ind[1] - min_ind[1] + 2, n_sb, 64)) #np.zeros((len(ra), len(dec), n_sb, 64))        
                rms_feed = np.zeros_like(map_feed)
                where = where = np.where(div_sb_obsid[i, :, :, :, :] > 0)
                map_feed[where] = sum_sb_obsid[i, :, :, :, :][where] / div_sb_obsid[i, :, :, :, :][where]
                rms_feed[where] = np.sqrt(1.0 / div_sb_obsid[i, :, :, :, :][where])
                map_feed = map_feed[min_ind[0] -1:max_ind[0], min_ind[1] -1:max_ind[1]]
                rms_feed = rms_feed[min_ind[0] -1:max_ind[0], min_ind[1] -1:max_ind[1]]
                sh = map_feed.shape
                ps_o_feed_chi2[:, i, :] = get_ps_chi2(
                        map_feed.reshape((sh[0], sh[1], n_sb * n_freqs)),
                        rms_feed.reshape((sh[0], sh[1], n_sb * n_freqs)),
                        n_k, d_th, dz, is_feed=True, is_slow=is_slow, name="o_feed")
                if len(params.accept_mod_debug_maps_dir) > 0:
                    np.save(f"{params.accept_mod_debug_maps_dir}/{obsid_info.scans[0][:-2]}_f{i}_ps_o_feed_chi2.npy", ps_o_feed_chi2)
                    np.save(f"{params.accept_mod_debug_maps_dir}/{obsid_info.scans[0][:-2]}_f{i}_ps_o_feed_map.npy", map_feed.reshape((sh[0], sh[1], n_sb * n_freqs)))
                    np.save(f"{params.accept_mod_debug_maps_dir}/{obsid_info.scans[0][:-2]}_f{i}_ps_o_feed_map_rms.npy", rms_feed.reshape((sh[0], sh[1], n_sb * n_freqs)))
                    np.save(f"{params.accept_mod_debug_maps_dir}/{obsid_info.scans[0][:-2]}_f{i}_ps_o_feed_misc.npy", np.array([n_k, d_th, dz]))


    return (ps_s_sb_chi2, ps_s_feed_chi2, ps_s_chi2, ps_o_sb_chi2,
            ps_o_feed_chi2, ps_o_chi2, ps_z_s_sb_chi2, ps_xy_s_sb_chi2)
                # return (ps_s_sb_chi2, ps_s_feed_chi2, ps_s_chi2, ps_s_stackp_chi2, ps_s_stackfp_chi2, ps_o_sb_chi2,
    #         ps_o_feed_chi2, ps_o_chi2, ps_o_stackp_chi2, ps_o_stackfp_chi2)

def get_ps_chi2(map, rms, n_k, d_th, dz, is_feed=False, is_slow=False, name=""):
    # if is_slow:
    #     with h5py.File("/mn/stornext/d5/data/nilsoles/nils/pipeline/power_spectrum/transfer_functions/tf_slow_scans.h5", "r") as f:
    #         tf = f["/spherically_averaged/transfer_function"][2:]
    #         k_tf = f["/spherically_averaged/k_bin_centers"][2:]
    # else:
    #     with h5py.File("/mn/stornext/d5/data/nilsoles/nils/pipeline/power_spectrum/transfer_functions/tf_fast_scans.h5", "r") as f:
    #         tf = f["/spherically_averaged/transfer_function"][2:]
    #         k_tf = f["/spherically_averaged/k_bin_centers"][2:]

    # tf_interp_log = CubicSpline(np.log10(k_tf), np.log10(tf))
    # def tf_interp(k_tf):
    #     return 10**tf_interp_log(np.log10(k_tf))

    if is_slow:
        speed="slow"
    else:
        speed="fast"
    transfer = np.load(f"{current}/TFs/{name}_tf_median_{speed}.npy")

    where = np.where(rms > 0)
    k_bin_edges = np.logspace(-1.8, np.log10(0.5), n_k)
    w = np.zeros_like(rms)
    w[where] = 1 / rms[where] ** 2

    Pk, k, nmodes = compute_power_spec3d(w * map, k_bin_edges, d_th, d_th, dz)
    # print(n_k, Pk.shape, transfer.shape)
    where = np.where((Pk > 0)*np.isfinite(transfer))

    n_sim = 100 #20
    ps_arr = np.zeros((n_sim, n_k - 1))
    for l in range(n_sim):
        map_n = np.random.randn(*rms.shape) * rms
        ps_arr[l] = compute_power_spec3d(w * map_n, k_bin_edges, d_th, d_th, dz)[0]

    # transfer = 1.0 / np.exp((0.055/k) ** 2.5)  # 6.7e5 / np.exp((0.055/k) ** 2.5)#1.0 / np.exp((0.03/k) ** 2)   ######## Needs to be tested!
    
    ps_mean = np.mean(ps_arr, axis=0)

    # if is_feed:
        # transfer4 = 1.0 / np.exp((0.050/k) ** 5.5)  + 1e-6 
        # transfer = np.array([7.08265320e-07, 1.30980902e-06, 1.87137602e-01, 4.91884922e-01, 6.48433271e-01, 8.27296733e-01, 8.85360854e-01, 8.14043197e-01, 8.03513664e-01]) #1.0 / np.exp((0.050/k) ** 5.5) + 1e-6
        # with open("feed_ps.txt", "ab") as myfile:
        #     np.savetxt(myfile, np.array([Pk, ps_mean]).T)

    ps_std = np.std(ps_arr, axis=0) / transfer
    Pk = Pk / transfer

    n_chi2 = len(Pk[where])
    if n_chi2 < 5:
        return np.nan
    chi = np.sum(((Pk[where] - ps_mean[where])/ ps_std[where]) ** 3)
    chi2 = np.sign(chi) * np.abs((np.sum(((Pk[where] - ps_mean[where])/ ps_std[where]) ** 2) - n_chi2) / np.sqrt(2 * n_chi2))

    # if (chi2 > 20.0) and is_feed:
    #     plt.errorbar(k, Pk * transfer, ps_std)
    #     plt.loglog(k, transfer * ps_mean[-1])
    #     plt.loglog(k, ps_mean * transfer)
    #     print(k, ps_std * transfer)
    #     print(chi2)
    #     plt.figure()

    #     plt.imshow(map[3, :, :] * w[3, :, :], interpolation=None)
    #     plt.show()
    return chi2 #, Pk, ps_mean, ps_std, transfer


def get_ps_1d2d_chi2(map, rms, n_k, d_th, dz, is_feed=False):

    where = np.where(rms > 0)
    k_bin_edges = np.logspace(-1.45, np.log10(0.1), n_k)
    w = np.zeros_like(rms)
    w[where] = 1 / rms[where] ** 2

    Pk_1D, Pk, k, nmodes = compute_power_spec1d_2d(w * map, k_bin_edges, d_th, d_th, dz)

    where = np.where(Pk > 0)

    n_sim = 100 #20
    ps_arr = np.zeros((n_sim, n_k - 1))
    ps_arr_1D = np.zeros((n_sim, len(Pk_1D)))
    for l in range(n_sim):
        map_n = np.random.randn(*rms.shape) * rms
        ps_arr_1D[l], ps_arr[l] = compute_power_spec1d_2d(w * map_n, k_bin_edges, d_th, d_th, dz)[:2]

    transfer = 1.0 #/ np.exp((0.055/k) ** 2.5)  # 6.7e5 / np.exp((0.055/k) ** 2.5)#1.0 / np.exp((0.03/k) ** 2)   ######## Needs to be tested!

    ps_mean = np.mean(ps_arr, axis=0)
    ps_1D_mean = np.mean(ps_arr_1D, axis=0)

    ps_std = np.std(ps_arr, axis=0) / transfer
    Pk = Pk / transfer

    ps_1D_std = np.std(ps_arr_1D, axis=0)


    n_chi2 = len(Pk[where])
    

    chi = np.sum(((Pk[where] - ps_mean[where])/ ps_std[where]) ** 3)
    chi2 = np.sign(chi) * np.abs((np.sum(((Pk[where] - ps_mean[where])/ ps_std[where]) ** 2) - n_chi2) / np.sqrt(2 * n_chi2))

    n_chi2_1D = len(Pk_1D)
    chi_1D = np.sum(((Pk_1D - ps_1D_mean)/ ps_1D_std) ** 3)

    chi2_1D = np.sign(chi_1D) * np.abs((np.sum(((Pk_1D - ps_1D_mean)/ ps_1D_std) ** 2) - n_chi2_1D) / np.sqrt(2 * n_chi2_1D))
    if n_chi2 < 2:
        return chi2_1D, np.nan
    return chi2_1D, chi2


def get_patch_info(path):
    with open(path, 'r') as f:
        lines = f.read().splitlines()
        my_list = [[x for x in line.split()] for line in lines]
        my_dict = { y[0]: y[1:7] for y in my_list[:-1] }
    return my_dict

def save_data_2_h5(params, scan_list, scan_data, fieldname, runid):
    filename = data_folder + 'scan_data_' + id_string + fieldname + '.h5'
    f1 = h5py.File(filename, 'w')
    f1.create_dataset('scan_list', data=scan_list)
    f1.create_dataset('scan_data', data=scan_data)
    dt = h5py.special_dtype(vlen=str) 
    stats_list_arr = np.array(stats_list, dtype=dt)
    f1.create_dataset('stats_list', data=stats_list_arr)
    f1.create_dataset("runID", data = runid)
    for key in vars(params):  # Writing entire parameter file to separate hdf5 group.
        if getattr(params, key) == None:  # hdf5 didn't like the None type.
            f1[f"params/{key}"] = "None"
        else:
            f1[f"params/{key}"] = getattr(params, key)
    f1.close()
    return filename


def make_accept_list(params, accept_params, scan_data):
    n_scans, n_det, n_sb, _ = scan_data.shape
    accept_list = np.ones((n_scans, n_det, n_sb), dtype=bool)
    reject_reason = np.zeros((n_scans, n_det, n_sb, 100), dtype=bool)

    temp_bool = np.zeros((n_scans, n_det, n_sb), dtype=bool)

    # decline all sidebands that are entirely masked
    acceptrate = extract_data_from_array(scan_data, 'acceptrate')

    # accept_list[:, 7, :] = False    
    
    acc = np.zeros(len(stats_list) + 1)
    acc[0] = np.nansum(acceptrate[accept_list]) / (n_scans * 19 * 4)
    print(acc[0], 'before cuts')
    for i, stat_string in enumerate(stats_list):
        stats = extract_data_from_array(scan_data, stat_string)
        cuts = accept_params.stats_cut[stat_string]
        temp_bool[:, :, :] = False
        if (not np.isnan(cuts[0])):
            accept_list[np.where(stats < cuts[0])] = False
            accept_list[np.where(np.isnan(stats))] = False
            temp_bool[np.where(stats < cuts[0])] = True
            temp_bool[np.where(np.isnan(stats))] = True
            reject_reason[:, :, :, i] = temp_bool
            # reject_reason[:, :, :, i][np.argwhere(stats < cuts[0])] = True
            # reject_reason[:, :, :, i][np.argwhere(np.isnan(stats))] = True
        if (not np.isnan(cuts[1])):
            accept_list[np.where(stats > cuts[1])] = False
            accept_list[np.where(np.isnan(stats))] = False
            temp_bool[np.where(stats > cuts[1])] = True
            temp_bool[np.where(np.isnan(stats))] = True
            reject_reason[:, :, :, i] = temp_bool
            # reject_reason[:, :, :, i][np.argwhere(stats < cuts[1])] = True
            # reject_reason[:, :, :, i][np.argwhere(np.isnan(stats))] = True
        
        acc[i+1] = np.nansum(acceptrate[accept_list]) / (n_scans * 19 * 4)
        print(acc[i+1], stat_string, cuts)

    return accept_list, reject_reason, acc


def read_jk_param(filepath):
    with open(filepath) as my_file:
        raw_lines = [line.split()[:3] for line in my_file]
        lines = [line for line in raw_lines if "$" not in line]
        n_temporal_splits = len(raw_lines) - len(lines)
        n_split = int(lines[0][0]) - n_temporal_splits
        lines = lines[2:n_split+1]
        strings = [l[0] for l in lines]
        types = [int(l[1]) for l in lines]

    return strings, types, n_split


def make_jk_list(params, accept_list, scan_list, scan_data, jk_param):
    strings, types, n_split = read_jk_param(jk_param)
    
    cutoff_list = np.zeros((n_split-1, 20, 4), dtype='f')
    
    n_scans, n_det, n_sb, _ = scan_data.shape
    jk_list = np.zeros((n_scans, n_det, n_sb), dtype=np.int64)

    if not np.any(accept_list.flatten()):
        jk_list[:] = 0 
        return jk_list, cutoff_list, strings
    
    el = extract_data_from_array(scan_data, 'el').copy()
    el[~accept_list] = np.nan
    cutoff = np.nanpercentile(el, 50.0, axis=0)
    el0_mask = el < cutoff

    if True:
        jk_list_test = jk_list.copy()
        sid = extract_data_from_array(scan_data, 'sidereal').copy()
        sid[~accept_list] = np.nan
        if fieldname == 'co2':
            cutoff = 87
        elif fieldname == 'co6':
            wh = np.where(sid < 50)
            sid[wh] += 360
            cutoff = 290
        elif fieldname == 'co7':
            cutoff = 231

        rise0_mask = sid < cutoff

        el0_rise0_mask = el0_mask & rise0_mask
        el0_rise1_mask = el0_mask & (~rise0_mask)
        el1_rise0_mask = (~el0_mask) & rise0_mask
        el1_rise1_mask = (~el0_mask) & (~rise0_mask)
        accept_list00 = accept_list.copy()
        accept_list01 = accept_list.copy()
        accept_list10 = accept_list.copy()
        accept_list11 = accept_list.copy()
        accept_list00[~el0_rise0_mask] = False
        accept_list01[~el0_rise1_mask] = False
        accept_list10[~el1_rise0_mask] = False
        accept_list11[~el1_rise1_mask] = False
        for j, string in enumerate(strings):
            if string == "elev":  # Hacky solution to have the split be evenly distributed across the elevation PSX split.
                implement_split(params, accept_list, scan_data, jk_list, cutoff_list, string, j+1)
            else:
                implement_split(params, accept_list00, scan_data, jk_list, cutoff_list, string, j+1)
                implement_split(params, accept_list01, scan_data, jk_list, cutoff_list, string, j+1)
                implement_split(params, accept_list10, scan_data, jk_list, cutoff_list, string, j+1)
                implement_split(params, accept_list11, scan_data, jk_list, cutoff_list, string, j+1)

    else:
        # The old method, only ensuring that splits were even in elev.
        jk_list0 = np.zeros((n_scans, n_det, n_sb), dtype=np.int64)
        jk_list1 = np.zeros((n_scans, n_det, n_sb), dtype=np.int64)
        scan_data0 = scan_data.copy()
        scan_data1 = scan_data.copy()
        scan_data0[~el0_mask] = np.nan
        scan_data1[el0_mask] = np.nan
        accept_list0 = accept_list.copy()
        accept_list1 = accept_list.copy()
        accept_list0[~el0_mask] = False
        accept_list1[el0_mask] = False
        for j, string in enumerate(strings):
            if string == "elev":  # Hacky solution to have the split be evenly distributed across the elevation PSX split.
                implement_split(params, accept_list, scan_data, jk_list, cutoff_list, string, j+1)
            else:
                implement_split(params, accept_list0, scan_data0, jk_list0, cutoff_list, string, j+1)
                implement_split(params, accept_list1, scan_data1, jk_list1, cutoff_list, string, j+1)
        jk_list += jk_list0 + jk_list1

    # insert 0 on rejected sidebands, add 1 on accepted 
    jk_list[np.invert(accept_list)] = 0
    jk_list[accept_list] += 1 
    return jk_list, cutoff_list, strings


def implement_split(params, accept_list, scan_data, jk_list, cutoff_list, string, n):
    
    # Generating all possible random split keys
    import itertools
    rndstrings = ["rnd","RND","Rnd","RNd","rNd","RnD","rnD","rND",]

    capital_letter_range = list(range(65, 91))
    letter_range = list(range(97, 123))
    total_range = capital_letter_range + letter_range 
    combos = list(itertools.product(rndstrings, total_range))
    subseeds = []
    allstrings = []
    for combo in combos:
        combostring, i = combo
        combostring = combostring + chr(i)
        allstrings.append(combostring)
        subseed = np.sum([ord(c) for c in combostring])
        while subseed in subseeds:
            subseed += 1
        subseeds.append(subseed)
        
    # even/odd
    if string == 'odde':
        obsid = [int(str(scanid)[:-2]) for scanid in scan_list]
        odd = np.array(obsid) % 2
        odd = np.zeros((scan_data.shape[:-1]), dtype=int) + odd[:,None,None]
        odd[~accept_list] = 0
        jk_list[:] += odd * int(2 ** n)
        cutoff_list[n-1] = 0.0 # placeholder (no real cutoff value)
    elif "rnd" in string.casefold():
        # random scan split
        seed = params.jk_rnd_split_seed
        seed += subseeds[allstrings.index(string)]
        np.random.seed(seed)
        bits = np.zeros((scan_data.shape[:-1]), dtype=np.int64)
        for ifeed in range(scan_data.shape[1]):
            for isb in range(scan_data.shape[2]):
                indices = np.arange(scan_data.shape[0], dtype=np.int64)
                indices = indices[accept_list[:,ifeed,isb]]
                np.random.shuffle(indices)
                bits[indices[:indices.shape[0]//2],ifeed,isb] = 1
        jk_list += bits * int(2 ** n)
        cutoff_list[n-1] = 0.0 # placeholder (no real cutoff value)
    elif "trtp" in string.casefold():
        # random scan split
        seed = 42
        np.random.seed(seed)
        bits = np.zeros((scan_data.shape[:-1]), dtype=np.int64)
        for ifeed in range(scan_data.shape[1]):
            for isb in range(scan_data.shape[2]):
                indices = np.arange(scan_data.shape[0], dtype=np.int64)
                indices = indices[accept_list[:,ifeed,isb]]
                np.random.shuffle(indices)
                bits[indices[:indices.shape[0]//2],ifeed,isb] = 1
        jk_list += bits * int(2 ** n)
        cutoff_list[n-1] = 0.0 # placeholder (no real cutoff value)
    elif string == 'dayn':
        # day/night split
        closetonight = extract_data_from_array(scan_data, 'night').copy()
        closetonight[~accept_list] = np.nan
        closetonight += np.random.normal(0, 1e-6, closetonight.shape)
        cutoff = np.nanpercentile(closetonight, 50.0)
        jk_list[np.where(closetonight > cutoff)] += int(2 ** n)
        cutoff_list[n-1] = cutoff
    elif string == 'half':
        # halfmission split
        mjd = extract_data_from_array(scan_data, 'mjd').copy()
        mjd[~accept_list] = np.nan
        cutoff = np.nanpercentile(mjd, 50.0, axis=0)
        jk_list[np.where(mjd > cutoff)] += int(2 ** n)
        cutoff_list[n-1] = cutoff
    elif string == 'tsys':
        # halfmission split
        tsys = extract_data_from_array(scan_data, 'tsys').copy()
        tsys[~accept_list] = np.nan 
        cutoff = np.nanpercentile(tsys, 50, axis = 0)
        jk_list[np.where(tsys > cutoff)] += int(2 ** n)
        cutoff_list[n-1] = cutoff
    elif string == 'sdlb':
        # saddlebag split
        saddlebags = extract_data_from_array(scan_data, 'saddlebag')
        jk_list[np.where(saddlebags > 2.5)] += int(2 ** n)
        cutoff_list[n-1] = 0.0 # placeholder
    elif string == 'sidr':
        # sidereal time split 
        sid = extract_data_from_array(scan_data, 'sidereal')
        cutoff = np.nanpercentile(sid[accept_list], 50.0)
        # print('Sidereal time cutoff: ', cutoff)
        jk_list[np.where(sid > cutoff)] += int(2 ** n) 
        cutoff_list[n-1] = cutoff
    elif string == 'sid1':
        # sidereal time split 
        sid = extract_data_from_array(scan_data, 'sidereal')
        sid = np.abs(sid - 160)
        cutoff = np.nanpercentile(sid[accept_list], 50.0)
        # print('Sidereal time cutoff: ', cutoff)
        jk_list[np.where(sid > cutoff)] += int(2 ** n) 
        cutoff_list[n-1] = cutoff
    elif string == 'sid2':
        # sidereal time split 
        sid = extract_data_from_array(scan_data, 'sidereal')
        sid = np.abs(sid - 200)
        cutoff = np.nanpercentile(sid[accept_list], 50.0)
        # print('Sidereal time cutoff: ', cutoff)
        jk_list[np.where(sid > cutoff)] += int(2 ** n) 
        cutoff_list[n-1] = cutoff
    elif string == 'sid3':
        # sidereal time split 
        sid = extract_data_from_array(scan_data, 'sidereal').copy()
        sid[~accept_list] = np.nan
        sid = np.abs(sid - 240)
        cutoff = np.nanpercentile(sid[accept_list], 50.0, axis=0)
        # print('Sidereal time cutoff: ', cutoff)
        jk_list[np.where(sid > cutoff)] += int(2 ** n) 
        cutoff_list[n-1] = cutoff
    elif string == 'cesc':
        fbit = extract_data_from_array(scan_data, 'fbit')
        jk_list[np.where(fbit == 32)] += int(2 ** n) 
        cutoff_list[n-1] = 0.0 # placeholder'
    elif string == 'ambt':
        # ambient temperature split 
        ambt = extract_data_from_array(scan_data, 'airtemp').copy()
        ambt[~accept_list] = np.nan
        cutoff = np.nanpercentile(ambt, 50.0, axis=0)
        jk_list[np.where(ambt > cutoff)] += int(2 ** n)
        cutoff_list[n-1] = cutoff
    elif string == 'dtmp':
        # dew temperature
        #  temperature split 
        dtmp = extract_data_from_array(scan_data, 'dewtemp').copy()
        dtmp[~accept_list] = np.nan
        cutoff = np.nanpercentile(dtmp, 50.0, axis=0)
        jk_list[np.where(dtmp > cutoff)] += int(2 ** n)
        cutoff_list[n-1] = cutoff
    elif string == 'elev':
        # elevation split 
        el = extract_data_from_array(scan_data, 'el').copy()
        el[~accept_list] = np.nan
        cutoff = np.nanpercentile(el, 50.0, axis=0)
        jk_list[np.where(el > cutoff)] += int(2 ** n) 
        cutoff_list[n-1] = cutoff
    elif string == 'wind':
        # windspeed split 
        wind = extract_data_from_array(scan_data, 'windspeed').copy()
        wind[~accept_list] = np.nan
        cutoff = np.nanpercentile(wind, 50.0, axis=0)
        jk_list[np.where(wind > cutoff)] += int(2 ** n) 
        cutoff_list[n-1] = cutoff

    elif string == 'widr':
        # north-south wind direction split 
        winddir = extract_data_from_array(scan_data, 'winddir').copy()
        winddir[~accept_list] = np.nan
        cutoff = np.nanpercentile(np.cos(np.radians(winddir)), 50.0, axis=0)
        jk_list[np.where(winddir > cutoff)] += int(2 ** n) 
        cutoff_list[n-1] = cutoff

    elif string == 'hmty':
        # humidity split 
        humidity = extract_data_from_array(scan_data, 'humidity').copy()
        humidity[~accept_list] = np.nan
        cutoff = np.nanpercentile(humidity, 50.0, axis=0)
        jk_list[np.where(humidity > cutoff)] += int(2 ** n) 
        cutoff_list[n-1] = cutoff

    elif string == 'pres':
        # pressure split 
        pressure = extract_data_from_array(scan_data, 'pressure').copy()
        pressure[~accept_list] = np.nan
        cutoff = np.nanpercentile(pressure, 50.0, axis=0)
        jk_list[np.where(pressure > cutoff)] += int(2 ** n) 
        cutoff_list[n-1] = cutoff

    elif string == 'rain':
        # rain split 
        rain = extract_data_from_array(scan_data, 'rain').copy()
        rain[~accept_list] = np.nan
        rain += np.random.normal(0, 1e-6, rain.shape)
        cutoff = np.nanpercentile(rain, 50.0, axis=0)
        jk_list[np.where(rain > cutoff)] += int(2 ** n) 
        cutoff_list[n-1] = cutoff

    elif string == 'wthr':
        # weather split 
        weather = extract_data_from_array(scan_data, 'weather').copy()
        weather[~accept_list] = np.nan
        cutoff = np.nanpercentile(weather, 50.0, axis=0)
        jk_list[np.where(weather > cutoff)] += int(2 ** n) 
        cutoff_list[n-1] = cutoff

    elif string == 'sune':
        # sun_elevation split 
        sunel = extract_data_from_array(scan_data, 'sun_el').copy()
        sunel[~accept_list] = np.nan
        cutoff = np.nanpercentile(sunel, 50.0, axis=0)
        jk_list[np.where(sunel > cutoff)] += int(2 ** n)
        cutoff_list[n-1] = cutoff

    elif string == 'snup':
        # sun_up split (sun elevation > -5 deg) 
        sunel = extract_data_from_array(scan_data, 'sun_el').copy()
        jk_list[np.where(sunel > -5.0)] += int(2 ** n) 
        cutoff_list[n-1] = 0.0 # placeholder

    elif string == 'modi':
        # distance to moon split 
        moondist = extract_data_from_array(scan_data, 'moon_dist').copy()
        moondist[~accept_list] = np.nan
        cutoff = np.nanpercentile(moondist, 50.0, axis=0)
        jk_list[np.where(moondist > cutoff)] += int(2 ** n)
        cutoff_list[n-1] = cutoff

    elif string == 'sudi':
        # distance to sun split 
        sundist = extract_data_from_array(scan_data, 'sun_dist').copy()
        sundist[~accept_list] = np.nan
        cutoff = np.nanpercentile(sundist, 50.0, axis=0)
        jk_list[np.where(sundist > cutoff)] += int(2 ** n)
        cutoff_list[n-1] = cutoff

    elif string == 'wint':
        mjd = extract_data_from_array(scan_data, 'mjd').copy()
        mjd[~accept_list] = np.nan
        mid_winter = 58863  # 15. Jan 2020
        days_since_mid_winter = (mjd - mid_winter) % 365                                                                                                                                                         
        close_to_winter = np.minimum(np.abs(days_since_mid_winter), np.abs(365 - days_since_mid_winter))
        cutoff = np.nanpercentile(close_to_winter, 50.0, axis=0)
        jk_list[np.where(close_to_winter > cutoff)] += int(2 ** n)
        cutoff_list[n-1] = cutoff

    elif string == 'rise':
        sid = extract_data_from_array(scan_data, 'sidereal')
        if fieldname == 'co2':
            cutoff = 87
        elif fieldname == 'co6':
            wh = np.where(sid > 180)
            sid[wh] -= 360
            cutoff = -75
        elif fieldname == 'co7':
            cutoff = 231
        else:
            print('Unknown field: ', fieldname, ' rising split invalid')
            cutoff = 0
        jk_list[np.where(sid < cutoff)] += int(2 ** n)
        cutoff_list[n-1] = cutoff

    elif string == 'azmp':  ## az filter amplitude
        az_amp = extract_data_from_array(scan_data, 'az_amp').copy() 
        az_amp[~accept_list] = np.nan
        cutoff = np.nanpercentile(az_amp, 50.0, axis=0)
        jk_list[np.where(az_amp > cutoff)] += int(2 ** n)
        cutoff_list[n-1] = cutoff

    elif string == 'fpol':  ## fknee of second polyfilter component
        fknee = extract_data_from_array(scan_data, 'fknee_poly1').copy() 
        fknee[~accept_list] = np.nan 
        cutoff = np.nanpercentile(fknee, 50.0, axis=0)
        jk_list[np.where(fknee > cutoff)] += int(2 ** n)
        cutoff_list[n-1] = cutoff

    elif string == 'fpoO':  ## fknee of first polyfilter component
        fknee = extract_data_from_array(scan_data, 'fknee_poly0').copy()
        fknee[~accept_list] = np.nan 
        cutoff = np.nanpercentile(fknee, 50.0, axis = 0)
        jk_list[np.where(fknee > cutoff)] += int(2 ** n)
        
        cutoff_list[n-1] = cutoff

    elif string == 'fpoI':  ## fknee of second polyfilter component
        fknee = extract_data_from_array(scan_data, 'fknee_poly1').copy()
        fknee[~accept_list] = np.nan 
        cutoff = np.nanpercentile(fknee, 50, axis = 0)
        jk_list[np.where(fknee > cutoff)] += int(2 ** n)
        cutoff_list[n-1] = cutoff

    elif string == 'apoO':  ## alpha of second polyfilter component
        alpha = extract_data_from_array(scan_data, 'alpha_poly0').copy()
        alpha[~accept_list] = np.nan
        cutoff = np.nanpercentile(alpha, 50.0, axis = 0)
        jk_list[np.where(alpha > cutoff)] += int(2 ** n)
        cutoff_list[n-1] = cutoff

    elif string == 'apoI':  ## alpha of first polyfilter component
        alpha = extract_data_from_array(scan_data, 'alpha_poly1').copy()
        alpha[~accept_list] = np.nan
        cutoff = np.nanpercentile(alpha, 50.0, axis = 0)
        jk_list[np.where(alpha > cutoff)] += int(2 ** n)
        cutoff_list[n-1] = cutoff

    elif string == 'spoO':  ## sigma of second polyfilter component
        sigma = extract_data_from_array(scan_data, 'sigma_poly0').copy()
        sigma[~accept_list] = np.nan
        cutoff = np.nanpercentile(sigma, 50.0, axis = 0)
        jk_list[np.where(sigma > cutoff)] += int(2 ** n)
        cutoff_list[n-1] = cutoff
    
    elif string == 'spoI':  ## sigma of first polyfilter component
        sigma = extract_data_from_array(scan_data, 'sigma_poly1').copy()
        sigma[~accept_list] = np.nan
        cutoff = np.nanpercentile(sigma, 50.0, axis = 0)
        jk_list[np.where(sigma > cutoff)] += int(2 ** n)
        cutoff_list[n-1] = cutoff

    elif string == 'npca':
        npca = extract_data_from_array(scan_data, 'npca').copy()
        npcaf = extract_data_from_array(scan_data, 'npcaf').copy()
        npca_both = npca + npcaf
        npca_both[~accept_list] = np.nan
        npca_both += np.random.normal(0, 1e-4, npca_both.shape[0])[:, None, None]
        cutoff = np.nanpercentile(npca_both, 50.0, axis = 0)

        jk_list[np.where(npca_both > cutoff)] += int(2 ** n)
        cutoff_list[n-1] = cutoff

    elif string == 'pcaa':
        pca_ampl_sum = extract_data_from_array(scan_data, 'pcsm').copy()
        pca_ampl_sum[~accept_list] = np.nan
        cutoff = np.nanpercentile(pca_ampl_sum, 50.0, axis = 0)
        jk_list[np.where(pca_ampl_sum > cutoff)] += int(2 ** n)
        cutoff_list[n-1] = cutoff

    elif string == 's01f':
        sigma0_1f = extract_data_from_array(scan_data, 'sigma_mean').copy()
        sigma0_1f[~accept_list] = np.nan
        cutoff = np.nanpercentile(sigma0_1f, 50.0, axis = 0)
        jk_list[np.where(sigma0_1f > cutoff)] += int(2 ** n)
        cutoff_list[n-1] = cutoff

    elif string == 'fk1f':
        fknee_1f = extract_data_from_array(scan_data, 'fknee_mean').copy()
        fknee_1f[~accept_list] = np.nan
        cutoff = np.nanpercentile(fknee_1f, 50.0, axis = 0)
        jk_list[np.where(fknee_1f > cutoff)] += int(2 ** n)
        cutoff_list[n-1] = cutoff

    elif string == 'al1f':
        alpha_1f = extract_data_from_array(scan_data, 'alpha_mean').copy()
        alpha_1f[~accept_list] = np.nan
        cutoff = np.nanpercentile(alpha_1f, 50.0, axis = 0)
        jk_list[np.where(alpha_1f > cutoff)] += int(2 ** n)
        cutoff_list[n-1] = cutoff

    elif string == 'obhf':
        # obsid = [int(str(scanid)[:-2]) for scanid in scan_list]
        scanid = np.array([int(str(scanid)[-2:]) for scanid in scan_list])
        scanid_recast = np.zeros(scan_data.shape[:-1]) + scanid[:,None,None]
        scanid_recast[~accept_list] = np.nan
        scanid_recast += np.random.normal(0, 1e-6, scanid_recast.shape)
        cutoff = np.nanpercentile(scanid_recast, 50.0, axis=0)
        jk_list[np.where(scanid_recast > cutoff)] += int(2 ** n)
        cutoff_list[n-1] = cutoff

        ######## Here you can add new jack-knives  ############
        ### elif .......:
        ###
    else:
        print('Unknown split type: ', string)
    return jk_list


def save_jk_2_h5(params, scan_list, acceptrates, accept_list, reject_reason, jk_list, cutoff_list, split_list, fieldname, runID): 
    filename = data_folder + 'jk_data_' + id_string + jk_string + fieldname + '.h5'
    f1 = h5py.File(filename, 'w')
    f1.create_dataset('scan_list', data=scan_list)
    f1.create_dataset('acceptrates', data=acceptrates)
    f1.create_dataset('accept_list', data=accept_list)
    f1.create_dataset('reject_reason', data=reject_reason)
    f1.create_dataset('jk_list', data=jk_list)
    f1.create_dataset('cutoff_list', data=cutoff_list)
    dt = h5py.special_dtype(vlen=str)
    stats_list_arr = np.array(stats_list, dtype=dt)
    f1.create_dataset('stats_list', data=stats_list_arr)
    split_list_arr = np.array(split_list, dtype=dt)
    f1.create_dataset('split_list', data=split_list_arr)
    f1.create_dataset("runID", data = runid)
    for key in vars(params):  # Writing entire parameter file to separate hdf5 group.
        if getattr(params, key) == None:  # hdf5 didn't like the None type.
            f1[f"params/{key}"] = "None"
        else:
            f1[f"params/{key}"] = getattr(params, key)
    f1.close()
    return filename

def get_max_runID(folder):
    file_list = os.listdir(folder)
    file_list = [i for i in file_list if "param" in i]
    runid_list = [int(i[:-4].split("_")[-1]) for i in file_list]
    if len(runid_list) == 0:
        maxid = 0
    else:
        maxid = np.max(np.array(runid_list))
    return maxid

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    Nproc = comm.Get_size()
    rank = comm.Get_rank()

    # sys.path.append("/mn/stornext/d22/cmbco/comap/jonas/pipeline/")  # TODO: Find better solution
    from l2gen_argparser import parser
    params = parser.parse_args()
    if not params.runlist:
        raise ValueError("A runlist must be specified in parameter file or terminal.")
    param_file = params.param
    #params = vars(params)  # Accept-mod was written with params as a dict, so we just do Namespace -> dict so I don't have to rewrite stuff.

    spec = importlib.util.spec_from_file_location(params.accept_mod_params[:-3], os.path.join(current, params.accept_mod_params))
    accept_params = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(accept_params)
    spec = importlib.util.spec_from_file_location(params.stats_list[:-3], os.path.join(current, params.stats_list))    
    stats_list = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stats_list)
    stats_list = stats_list.stats_list
    fields = read_runlist(params)
    
    patch_filepath = params.patch_definition_file
    patch_info = get_patch_info(patch_filepath)
    
    if params.jk_rnd_split_seed is not None:
        params.jk_data_string = params.jk_data_string + f"_rnd{params.jk_rnd_split_seed}"
    
    weather_filepath = params.weather_filepath
    observerlog_blacklist_filepath = params.observerlog_blacklist_filepath
    data_folder = params.accept_data_folder
    id_string = params.accept_data_id_string + '_'
    jk_string = params.jk_data_string + '_'
    if id_string == '_':
        id_string = ''
    if jk_string == '_':
        jk_string = ''
    data_from_file = params.scan_stats_from_file # False #True
    jk_param_list_file = params.jk_def_file

    scan_data_data_name_list = []
    jk_data_name_list   = []
    copy_folder = data_folder + "parameter_copies/"
    
    if not os.path.isdir(copy_folder):
        os.mkdir(copy_folder)

    runid = get_max_runID(copy_folder) + 1

    for fieldname in fields:
        if data_from_file:
            filepath = data_folder + 'scan_data_' + id_string + fieldname + '.h5'
            with h5py.File(filepath, mode="r") as my_file:
                scan_list = my_file['scan_list'][()]
                scan_data = my_file['scan_data'][()]
            scan_start = params.obsid_start * 100
            scan_stop = params.obsid_stop * 100
            scans_in_range = (scan_list <= scan_stop)*(scan_list >= scan_start)
            scan_list = scan_list[scans_in_range]
            scan_data = scan_data[scans_in_range]
        else:
            scan_list, scan_data = get_scan_data(params, fields, fieldname)

        if rank == 0:
            if not data_from_file:
                scan_data_data_name = save_data_2_h5(params, scan_list, scan_data, fieldname, runid)
                scan_data_data_name_list.append(scan_data_data_name)
            print('Saved scan data')
            accept_list, reject_reason, acc = make_accept_list(params, accept_params, scan_data)
            print('Made accept list')
            jk_list, cutoff_list, split_list = make_jk_list(params, accept_list, scan_list, scan_data, jk_param_list_file)
            print('Made jk_list')
            
          
            jk_data_name = save_jk_2_h5(params, scan_list, acc, accept_list, reject_reason, jk_list, cutoff_list, split_list, fieldname, runid)
            jk_data_name_list.append(jk_data_name)


        if rank == 0:

            accept_params_name = os.path.join(current, params.accept_mod_params)
            stats_list_name = os.path.join(current, params.stats_list)
            
            accept_params_name_raw = params.accept_mod_params[:-3]
            stats_list_name_raw = params.stats_list[:-3]

            # param_file_copy_raw = param_file.split("/")
            # param_file_copy_raw = param_file_copy_raw[-1][:-4]
            runlist_name    = params.runlist
            runlist_copy_raw    = runlist_name.split("/")[-1][:-4]
            jk_def_raw    = jk_param_list_file.split("/")[-1][:-4]

            # param_file_copy = copy_folder + param_file_copy_raw + f"_{runid:06d}" + ".txt"
            runlist_copy = copy_folder + runlist_copy_raw + f"_{runid:06d}" + ".txt"
            jk_def_copy = copy_folder + jk_def_raw + f"_{runid:06d}" + ".txt"
            accept_params_name_copy = copy_folder + accept_params_name_raw + f"_{runid:06d}" + ".py"
            stats_list_name_copy = copy_folder + stats_list_name_raw + f"_{runid:06d}" + ".py"


            # shutil.copyfile(param_file, param_file_copy)
            #shutil.copyfile(runlist_name, runlist_copy)
            #shutil.copyfile(jk_param_list_file, jk_def_copy)
            #shutil.copyfile(accept_params_name, accept_params_name_copy)
            #shutil.copyfile(stats_list_name, stats_list_name_copy)
        

                
            if params.make_accept_mod_plots:
                ### Making scan data plots ###
                print("Making scan data plots")
                plot_folder = os.path.join(data_folder, "plots", fieldname)
                if not os.path.exists(plot_folder):
                    os.mkdir(plot_folder)

                lims = {
                    "obsid" : [6000, 40000],
                    "scanid" : [600000, 4000000],
                    "mjd" : [58600, 60500],
                    "night": [0, 12],
                    "sidereal" : [0, 360],
                    "az" : [0, 360],
                    "el" : [0, 80],
                    "chi2" : [-6, 6],
                    "acceptrate" : [0, 1],
                    "acceptrate_specific" : [0, 1],
                    "az_chi2" : [-7, 7],
                    "max_az_chi2" : [-4, 10],
                    "med_az_chi2" : [-4, 5],
                    "fbit" : [0, 0],
                    "az_amp" : [-0.001, 0.001],
                    "el_amp" : [-1.5, 1.5],
                    "n_spikes" : [-1, 30],
                    "n_jumps" : [-0.1, 10.1],
                    "n_anomalies" : [-0.1, 5.1],
                    "n_nan" : [-0.1, 10.1],
                    "tsys" : [0, 120],
                    "npca" : [-0.1, 12.1],
                    "npcaf" : [-0.1, 12.1],
                    "pca1" : [0, 51],
                    "pca2" : [0, 31],
                    "pca3" : [0, 21],
                    "pca4" : [0, 11],
                    'pcf1' : [0.0, 30.0],
                    'pcf2' : [0.0, 25.0],
                    'pcsm' : [0, 60.0],
                    "weather" : [-0.05, 1.05],
                    "kurtosis" : [-0.03, 0.03],
                    "skewness" : [-0.02, 0.02],
                    "scan_length" : [0, 20],
                    "saddlebag" : [-0.1, 5.1],
                    "acceptmod_error" : [-0.1, 5.1],
                    "blacklisted" : [-0.1, 1.1],
                    "sigma_poly0" : [0, 0.0025],
                    "fknee_poly0" : [0, 31],
                    "alpha_poly0" : [-4, 2],
                    "sigma_poly1" : [0, 0.0002],
                    "fknee_poly1" : [0, 10],
                    "alpha_poly1" : [-4, 2],
                    "power_mean" : [-1e7, 1.5e8],
                    "sigma_mean" : [-100, 1100],
                    "fknee_mean" : [0, 30],
                    "alpha_mean" : [-4, 2],
                    "airtemp" : [-25, 45],
                    "dewtemp" : [-30, 60],
                    "humidity" : [-0.1, 1.1],
                    "pressure" : [790, 1010],
                    "rain" : [-0.1, 3.0],
                    "winddir" : [-1, 361],
                    "windspeed" : [-1, 21],
                    "moon_dist" : [0, 200],
                    "moon_angle" : [-200, 370],
                    "moon_cent_sl" : [-0.1, 2.1],
                    "moon_outer_sl" : [-0.1, 2.1],    
                    "sun_dist" : [0, 181],
                    "sun_angle" : [-200, 370],
                    "sun_cent_sl" : [-0.1, 1.1],
                    "sun_outer_sl" : [-0.1, 2.1],
                    "sun_el" : [-91, 91],
                    "ps_chi2" : [-20, 30],
                    "ps_s_sb_chi2" : [-20, 30],
                    "ps_s_feed_chi2" : [-20, 100],
                    "ps_s_chi2" : [-20, 200],
                    "ps_o_sb_chi2": [-20.0, 30.0],
                    "ps_o_feed_chi2": [-20.0, 80.0],
                    "ps_o_chi2": [-20.0, 160.0],
                    "ps_z_s_sb_chi2": [-60.0, 40.0],
                    "ps_xy_s_sb_chi2": [-60.0, 40.0]}
                
                import sys
                # sys.path.append("/mn/stornext/d22/cmbco/comap/protodir/accept_mod/")
                stats_cut = accept_params.stats_cut

                Nstats = len(stats_list) - 14
                
                
                def downsample(mjd, data, bin_size_days=1):
                    mjd_start = int(np.floor(mjd.min()))
                    mjd_stop = int(np.ceil(mjd.max()))
                    bins = np.arange(mjd_start, mjd_stop+1, bin_size_days)
                    Nbins = bins.shape[0]
                    data_binned = np.zeros((Nbins)) + np.nan
                    N_binned = np.zeros((Nbins))
                    for ibin in range(Nbins-1):
                        data_binned[ibin] = np.nanmean(data[(mjd >= bins[ibin])*(mjd < bins[ibin+1])])
                        N_binned[ibin] = np.sum((mjd >= bins[ibin])*(mjd < bins[ibin+1]))
                    data_binned[N_binned < bin_size_days*5*20*4] = np.nan
                    return bins, data_binned
                
                fig, ax = plt.subplots(int(math.ceil(Nstats/4)), 4, figsize=(12*3, Nstats*3/4))
                for i in range(Nstats):
                    stat = stats_list[i]
                    stat_cut = stats_cut[stat]
                    histrange = lims[stat]
                    _data = scan_data[:,:,:,i].copy()
                    _data[_data < histrange[0]] = histrange[0] + 1e-8
                    _data[_data > histrange[1]] = histrange[1] - 1e-8
                    for ifeed in range(_data.shape[1]):
                        ax[i//4, i%4].hist(_data[:,ifeed].flatten(), density=True, bins=100, range=histrange, histtype="step", lw=1.0, alpha=0.3, color="k")
                    ax[i//4, i%4].hist(_data.flatten(), density=True, bins=100, range=histrange, histtype="step", lw=2)
                    ax[i//4, i%4].axvline(stat_cut[0], ls="--", c="k")
                    ax[i//4, i%4].axvline(stat_cut[1], ls="--", c="k")
                    ax[i//4, i%4].set_title(stat)

                        
                plt.tight_layout()
                plt.savefig(os.path.join(plot_folder, params.accept_data_id_string + "_" + fieldname + f"_{params.map_name}" + "_histograms.png"), bbox_inches="tight", dpi=200)

                fig, ax = plt.subplots(int(math.ceil(Nstats/4)), 4, figsize=(12*3, Nstats*3/4))
                for i in range(Nstats):
                    stat = stats_list[i]
                    stat_cut = stats_cut[stat]
                    histrange = lims[stat]
                    _data = scan_data[:,:,:,i].flatten().copy()
                    _data[_data < histrange[0]] = histrange[0] + 1e-8
                    _data[_data > histrange[1]] = histrange[1] - 1e-8

                    bins_weeks, data_weeks = downsample(scan_data[:,:,:,2].flatten(), scan_data[:,:,:,i].flatten(), bin_size_days=7)
                    data_weeks[data_weeks < histrange[0]] = histrange[0] + 1e-8
                    data_weeks[data_weeks > histrange[1]] = histrange[1] - 1e-8

                    ax[i//4, i%4].axhline(stat_cut[0], ls="--", c="k")
                    ax[i//4, i%4].axhline(stat_cut[1], ls="--", c="k")
                    ax[i//4, i%4].scatter(scan_data[:,:,:,2].flatten()[::31], _data[::31], s=0.01, alpha=0.1)
                    ax[i//4, i%4].plot(bins_weeks, data_weeks, c="tab:orange")
                    ax[i//4, i%4].set_title(stat)
                plt.tight_layout()
                plt.savefig(os.path.join(plot_folder, params.accept_data_id_string + "_" + fieldname + f"_{params.map_name}" + "_time_plot.png"), bbox_inches="tight", dpi=200)