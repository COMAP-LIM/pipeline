import numpy as np
import matplotlib as mpl
mpl.use('Agg')  # Use a non-interactive backend suitable for scripts without display
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import sys
import numpy.fft as fft
import os
import errno
import h5py
from scipy import stats
from scipy.stats import norm
import copy

import argparse
import re 
from tqdm import tqdm
import warnings
import time as tm

from astropy.time import Time
from datetime import datetime

warnings.filterwarnings("ignore", category = RuntimeWarning) # Ignore warnings caused by mask nan/inf weights
warnings.filterwarnings("ignore", category = UserWarning)    # Ignore warning when producing log plot of emty array

# import accept_mod.stats_list as stats_list
# stats_list = stats_list.stats_list

current = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current)

sys.path.append(parent_directory)

from tools.read_runlist import read_runlist
# from tools.spikes import spike_data, spike_list, get_spike_list
from l2gen_argparser import parser


class L2plots():
    def __init__(self):
        self.outpath = "/mn/stornext/d16/www_cmb/ihle/comap/files/"
        self.input()
        self.read_paramfile()

    def input(self):
        """
        Function parsing the command line input.
        """
        params = parser.parse_args()
        self.params = params

    def read_paramfile(self):
        """
        Function reading the parameter file provided by the command line
        argument, and defining class parameters.
        """
    
        params = self.params
        self.scan_data_path = params.accept_data_folder                          # Extracting path

        self.id_string = params.accept_data_id_string                                 # Extracting path
        
        self.l2_path = params.level2_dir                  # Extracting path

        self.runlist = read_runlist(params, ignore_existing=False)
        self.scanids = np.array([scan_content[0] for scan_content in self.runlist])

        modified_scanids = (self.scanids // 100).astype(int)  # Floor divide by 100 is equivalent to dropping last two digits

        unique_ids, inverse_indices = np.unique(modified_scanids, return_inverse=True)

        # Dictionary creation
        self.obsid_dict = {}
        for i, unique_id in enumerate(unique_ids):
            mask = inverse_indices == i  # Create a mask where inverse_indices matches current index
            relevant_scans = [scan for scan, m in zip(self.runlist, mask) if m] # Filter out the relevant scanids using mask
            relevant_scanids = self.scanids[mask]
            sorted_scans = [relevant_scans[ind] for ind in np.argsort(relevant_scanids)]  # Sort the filtered scanids
            self.obsid_dict[str(unique_id)] = list(sorted_scans)  # Add to the dictionary

        self.obsIDs = list(self.obsid_dict.keys())
        self.nobsIDs = len(self.obsIDs) 
        
        self.patch_name = self.obsid_dict[str(list(self.obsid_dict.keys())[0])][0][4]
 
               
        # self.runlist_path = params.runlist                  # Extracting path
        
        # runlist_file = open(self.runlist_path, "r")         # Opening 
        # runlist = runlist_file.read()
        
        # tod_in_list = re.findall(r"\/.*?\.\w+", runlist)
        # self.tod_in_list = tod_in_list

        # patch_name = re.search(r"\s([a-zA-Z0-9]+)\s", runlist)
        # self.patch_name = str(patch_name.group(1))

        # obsIDs = re.findall(r"\s\d{6}\s", runlist)         # Regex pattern to find all obsIDs in runlist
        # self.obsIDs = [num.strip() for num in obsIDs]
        # self.nobsIDs = len(self.obsIDs)                    # Number of obsIDs in runlist
        
        # dates = re.findall(r"-(20\d{2}-\d{2}-\d{2}-\d+)\.", runlist)         # Regex pattern to find all dates in runlist
        # self.dates = [num.strip() for num in dates]

        # scans_per_obsid = re.findall(r"\s(\d+)\s20.+.hd5", runlist)
        # self.scans_per_obsid = [int(num.strip()) - 2 for num in scans_per_obsid]

        # runlist_file.close()

    def ensure_dir_exists(self, path):
        try:
            os.makedirs(path)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise

    def run(self):
        ps_chi2, ps_s_feed, ps_s_chi2, ps_o_sb, ps_o_feed, ps_o_chi2, scan_data = self.open_scan_data()
        # print(ps_chi2.shape)
        # print(ps_o_sb.shape)
        for i in range(self.nobsIDs):
            self.obsid = int(self.obsIDs[i])
            
            # self.date  = self.dates[i]
            #if int(self.obsid) != 12338:
            #    continue

            print(f"Processing obsID: {self.obsid}")
            self.scanids = [content[0] for content in self.obsid_dict[str(self.obsid)]]

            scan_idx = [self.allscanids.index(scanid) for scanid in self.scanids]
            # which_obsid = (self.allscanids // 100).astype(int)
            # mask = which_obsid == self.obsid
            # print(self.obsIDs)  # just for debugging, comment out
            # first_scanid = self.obsid + "02"
            # idx = np.argmin(np.abs(self.allscanids - int(first_scanid)))
            # start = idx
            # stop  = idx + self.scans_per_obsid[i]
            
            # self.scanids = self.allscanids[start:stop]
            # scan_numbers = ['%02i' % (i+2) for i in range(self.scans_per_obsid[i])]
            # self.scanids = [int(self.obsid + '%02i' % (i+2)) for i in range(self.scans_per_obsid[i])]
            self.n_scans = len(self.scanids)

            self.ps_chi2 = ps_chi2[scan_idx, ...]
            self.ps_s_feed = ps_s_feed[scan_idx, ...]
            self.ps_s_chi2 = ps_s_chi2[scan_idx, ...]
            self.ps_o_sb   = ps_o_sb[scan_idx[0], ...]
            self.ps_o_feed = ps_o_feed[scan_idx[0], ...]
            self.ps_o_chi2 = ps_o_chi2[scan_idx[0], ...]
            self.scan_data_obsid = scan_data[scan_idx]
            
            t0 = tm.time()
            self.plot_ps_chi2_data()
            print("PS chi2 plot:", tm.time() - t0, "s")
            
            self.get_obsid_diagnostics()  # here we also make all the scan-level plots
            
            if len(self.corrs) == 0:
                print('No working scans in obsid')
                continue

            t0 = tm.time()
            self.plot_obsid_diagnostics()
            print("obsID plot:", tm.time() - t0, "s")

    def open_scan_data(self):
        print(self.id_string, self.patch_name)
        filename = "scan_data_" + self.id_string + "_" + self.patch_name + ".h5"
        filename = self.scan_data_path + filename

        with h5py.File(filename, "r") as infile:
            data = np.array(infile['scan_data'][()])
            self.allscanids = list(infile['scan_list'][()])
            self.stats_list = list(infile['stats_list'][()].astype(str))
        index = self.stats_list.index('ps_s_sb_chi2')
        ps_chi2 = data[:,:,:, index]
        index = self.stats_list.index('ps_s_feed_chi2')
        ps_s_feed = data[:,:,:, index]
        index = self.stats_list.index('ps_s_chi2')
        ps_s_chi2 = data[:,:,:, index]

        index = self.stats_list.index('ps_o_sb_chi2')
        ps_o_sb = data[:,:,:, index]
        index = self.stats_list.index('ps_o_feed_chi2')
        ps_o_feed = data[:,:,:, index]
        index = self.stats_list.index('ps_o_chi2')
        ps_o_chi2 = data[:,:,:, index]

        return ps_chi2, ps_s_feed, ps_s_chi2, ps_o_sb, ps_o_feed, ps_o_chi2, data

    def get_corr(self):
        with h5py.File(os.path.join(f"{self.l2_path}{self.patch_name}" ,f"{self.l2name}"), mode = "r") as my_file:
            self.scan_id       = self.l2name[-12:-3]
            tod_ind       = np.array(my_file['tod'][:])
            sb_mean_ind   = np.array(my_file['sb_mean'][:])
            mask_ind      = my_file['freqmask'][:]
            mask_full_ind = my_file['freqmask_full'][:]
            reason_ind    = my_file['freqmask_reason'][:]
            pixels        = np.array(my_file['feeds'][:]) - 1 
            self.windspeed = np.mean(my_file['hk_windspeed'][()])
            self.feat = my_file['feature'][()]
            
            self.n_det_ind, self.n_sb, self.n_freq, self.n_samp = tod_ind.shape
            try: 
                n_spikes_ind      = np.array(my_file['n_spikes'][:])
                spike_data_ind    = np.array(my_file['spike_data'][:]) 
            except KeyError:
                n_spikes_ind = np.zeros((self.n_det_ind, self.n_sb))
                spike_data_ind = np.zeros((self.n_det_ind, self.n_sb, 50, 5))
                print("Found no spike_data")
            # try: 
            #     sd_ind = np.array(my_file['spike_data'])
            # except KeyError:
            #     sd_ind = np.zeros((3, self.n_det_ind, self.n_sb, 4, 1000))
            try: 
                chi2_ind = np.array(my_file['chi2'])
            except KeyError:
                chi2_ind = np.zeros_like(tod_ind[:,:,:,0])
            try:
                acc_ind = np.array(my_file['acceptrate'])
            except KeyError:
                acc_ind = np.zeros_like(tod_ind[:,:,0,0])
                print("Found no acceptrate")
            self.time = np.array(my_file['time'])
            mjd = self.time
            try:
                self.n_pca_comp = my_file['n_pca_comp'][()]
                self.pca      = np.array(my_file['pca_comp'])
                self.eigv     = np.array(my_file['pca_eigvals'])
                ampl_ind = np.array(my_file['pca_ampl'])
            except KeyError:
                self.pca = np.zeros((4, self.n_samp))
                self.eigv = np.zeros(4)
                ampl_ind = np.zeros((4, *mask_full_ind.shape))
                print('Found no pca comps')
            try:
                n_pca_feed_comp_ind = my_file['n_pca_feed_comp'][()]
                pca_feed_ind      = np.array(my_file['pca_feed_comp'])
                eigv_feed_ind     = np.array(my_file['pca_feed_eigvals'])
                ampl_feed_ind = np.array(my_file['pca_feed_ampl'])
            except KeyError:
                pca_feed_ind = np.zeros((4, self.n_det_ind, self.n_samp))
                eigv_feed_ind = np.zeros(self.n_det_ind, 4)
                ampl_feed_ind = np.zeros((4, *mask_full_ind.shape))
                print('Found no feed pca comps')
            try:
                tsys_ind = np.array(my_file['Tsys_lowres'])
            except KeyError:
                tsys_ind = np.zeros_like(tod_ind[:,:,:,0]) + 40
                print("Found no tsys")

        t0   = self.time[0]
        self.time = (self.time - self.time[0]) * (24 * 60)  # minutes
        dt = (self.time[1] - self.time[0]) * 60  # seconds
        self.radiometer = 1 / np.sqrt(31.25 * 10 ** 6 * dt)  # * 1.03

        self.n_freq_hr = len(mask_full_ind[0,0])
        self.n_det     = np.max(pixels) + 1 

        ## transform to full arrays with all pixels
        tod       = np.zeros((self.n_det, self.n_sb, self.n_freq, self.n_samp))
        mask      = np.zeros((self.n_det, self.n_sb, self.n_freq))
        mask_full = np.zeros((self.n_det, self.n_sb, self.n_freq_hr))
        acc       = np.zeros((self.n_det, self.n_sb))
        self.ampl      = np.zeros((self.n_pca_comp, self.n_det, self.n_sb, self.n_freq_hr))
        self.n_pca_feed_comp = np.zeros((self.n_det))
        self.ampl_feed = np.zeros((12, self.n_det, self.n_sb, self.n_freq_hr))
        self.pca_feed = np.zeros((12, self.n_det, self.n_samp))
        self.eigv_feed_ind = np.zeros((self.n_det, 12))
        tsys      = np.zeros((self.n_det, self.n_sb, self.n_freq))
        chi2      = np.zeros((self.n_det, self.n_sb, self.n_freq))

        self.sb_mean   = np.zeros((self.n_det, self.n_sb, self.n_samp))
        reason    = np.zeros((self.n_det, self.n_sb, self.n_freq_hr))
        self.n_spikes = np.zeros((self.n_det, self.n_sb))
        self.spike_data = np.zeros((self.n_det, self.n_sb, 50, 5))

        tod[pixels]       = tod_ind
        mask[pixels]      = mask_ind
        mask_full[pixels] = mask_full_ind
        reason[pixels]    = reason_ind
        acc[pixels]       = acc_ind
        self.ampl[:, pixels, :, :]      = ampl_ind
        self.ampl_feed[:, pixels, :, :] = ampl_feed_ind
        self.pca_feed[:, pixels, :]     = pca_feed_ind
        self.eigv_feed_ind[pixels, :]   = eigv_feed_ind
        self.n_pca_feed_comp[pixels]    = n_pca_feed_comp_ind
        tsys[pixels]           = tsys_ind
        chi2[pixels]           = chi2_ind
        self.sb_mean[pixels]        = sb_mean_ind 
        self.n_spikes[pixels] = n_spikes_ind
        self.spike_data[pixels] = spike_data_ind

        self.mask_hr   = mask_full
        acc       = acc.flatten()
        self.mask_full = mask_full.reshape((self.n_det, self.n_sb, self.n_freq, 16)).sum(3)

        tod            = tod[:, :, :, :] * mask[:, :, :, None]
        # self.tod_hist = copy.deepcopy(tod)
        tod[:, (0, 2)] = tod[:, (0, 2), ::-1]

        tod_flat = tod.reshape((self.n_det * self.n_sb * self.n_freq, self.n_samp))
        corr     = np.corrcoef(tod_flat)
        
        chi2 = chi2.flatten()
        chi2[(chi2 == 0.0)] = np.nan
        tsys = tsys.flatten()
        tsys[(tsys == 0.0)] = np.nan
        
        # my_spikes = get_spike_list(self.sb_mean, sd, self.scan_id, mjd)

        reasonarr = np.zeros(6)
        r = reason[pixels[:-1]].flatten()
        reasonarr[0] = len(r[(r == 1)])
        reasonarr[1] = len(r[(r == 2)])
        reasonarr[2] = len(np.where(np.logical_and(r>=3, r<=13))[0])
        reasonarr[3] = len(r[(r == 15)])
        reasonarr[4] = len(np.where(np.logical_and(r>=16, r<=27))[0])
        reasonarr[5] = len(np.where(np.logical_and(r>=40, r<=41))[0])
        

        reasonarr /= self.n_det_ind * self.n_sb * self.n_freq


        if (self.feat == 16):
            self.scanmode = 'circular'
        elif (self.feat == 32):
            self.scanmode = 'CES'
        elif (self.feat == 512):
            self.scanmode = 'raster'
        elif (self.feat == 32768):
            self.scanmode = 'lissajous'
        else:
            self.scanmode = ''

        return corr, 1.0 / self.n_samp, self.n_det, acc, chi2, tsys, t0, reasonarr

    def get_obsid_diagnostics(self):
        self.corrs       = []
        self.variances   = []
        self.accs        = []
        self.chi2s       = []
        self.tsyss       = []
        self.t0s         = []
        self.rs          = []

        for i in tqdm(range(self.n_scans)):
            self.l2name = f"{self.patch_name}_{self.scanids[i]:09}.h5"
            self.corr, self.var, self.n_det, self.acc, self.chi2, self.tsys, self.t0, self.reasonarr = self.get_corr()

            self.corrs.append(self.corr)
            self.variances.append(self.var)
            self.accs.append(self.acc)
            self.chi2s.append(self.chi2)
            self.tsyss.append(self.tsys)
            self.t0s.append(self.t0)
            self.rs.append(self.reasonarr)

            self.plot_scan_diagnostics()

        self.t0     = (np.mean(np.array(self.t0s)) * 24 - 7) % 24 
        self.tstart = np.min(np.array(self.t0s))
        self.corrs  = np.array(self.corrs)
        self.variances = np.array(self.variances)
        self.reason    = np.nanmean(np.array(self.rs), 0)
        self.weights   = 1 / self.variances[:, None, None] * (np.isfinite(self.corrs)) #(corrs != 0)
        
        self.W = np.sum(self.weights, 0)
        
        self.corr[(self.W > 0)] = 1 / self.W[(self.W > 0)] * np.nansum(self.corrs[:, (self.W > 0)] * self.weights[:, (self.W > 0)], 0)


    def plot_scan_spikes(self):
        def is_new_spike(spike, detector_id, sideband_id, spikes, closeness=0.2):
            for old_spike, old_detector_id, old_sideband_id in spikes:
                # If time difference is less than 0.1 and it's from the same detector and sideband, it's not a new spike
                if np.abs(spike[1] - old_spike[1]) < closeness and detector_id == old_detector_id:
                    return False
            return True

        def top_spikes(spike_data, num_spikes=9):
            # List to hold the flat data along with indices
            flat_data = []

            for detector_id in range(spike_data.shape[0]):
                for sideband_id in range(spike_data.shape[1]):
                    for spike_id in range(spike_data.shape[2]):
                        spike = spike_data[detector_id, sideband_id, spike_id]
                        flat_data.append((spike, detector_id, sideband_id))
            
            # Sort data by amplitude, descending order
            sorted_data = sorted(flat_data, key=lambda x: x[0][0], reverse=True)
            
            # Get top unique spikes
            top_spikes = []
            for spike, detector_id, sideband_id in sorted_data:
                if is_new_spike(spike, detector_id, sideband_id, top_spikes):
                    top_spikes.append((spike, detector_id, sideband_id))
                    if len(top_spikes) == num_spikes:
                        break

            return top_spikes


        def gauss(x, a, b, c, d):
            return a * np.exp(-(x - b)**2 / (2 * c**2)) + d
        

        time = self.time * 60  # seconds
        top_spikes_data = top_spikes(self.spike_data)

        fig, axes = plt.subplots(3, 3, figsize=(12, 10))
        axes = axes.flatten()
        for i in range(len(top_spikes_data)):

            feed = top_spikes_data[i][1] + 1 
            sb = top_spikes_data[i][2] + 1

            pars = top_spikes_data[i][0][:4]
            chi2 = top_spikes_data[i][0][4]
            if (sum(pars) == 0.0):
                continue

            n_time = len(time)
            n_fit = 100

            peak = np.abs(time - top_spikes_data[i][0][1]).argmin()
            earliest_ind = max(0, peak-n_fit)
            latest_ind = min(n_time, peak+n_fit)

            data = self.sb_mean[top_spikes_data[i][1], top_spikes_data[i][2]]

            data = data / data.mean() - 1
            axes[i].set_title('feed %i, sb %i' % (feed, sb))
            
            axes[i].plot(time[earliest_ind:latest_ind], gauss(time[earliest_ind:latest_ind], *pars), '--r', zorder=2, alpha=0.5)  # plot fitted gaussians
            axes[i].plot(time[max(0,earliest_ind - 150):min(n_time,latest_ind + 150)], data[max(0,earliest_ind - 150):min(n_time,latest_ind + 150)], zorder=1)

            axes[i].set_xlabel(r'$t$ [sec]')
            axes[i].set_xlim(time[earliest_ind]-2, time[latest_ind-1] + 2)
        fig.subplots_adjust(hspace=0.4)
        fig.subplots_adjust(top=0.92)
        fig.suptitle('Strongest spikes, scanid %07i' % int(self.scan_id), fontsize=15)

        save_string = 'spikes_%07i_%02i_00_AB.png' % (int(self.scan_id[:-2]), int(self.scan_id[-2:]))
        
        self.ensure_dir_exists(self.outpath + 'spikes')
        
        plt.savefig(self.outpath + 'spikes/' + save_string, bbox_inches='tight')


    def plot_scan_correlation(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        vmax = 0.1
        
        im = ax.imshow(self.corr, vmin=-vmax, vmax=vmax,  #vmin=-1, vmax=1,#vmin=-0.1, vmax=0.1,
                        extent=(0.5, self.n_det + 0.5, self.n_det + 0.5, 0.5))

        cbar = fig.colorbar(im)
        cbar.set_label('Correlation')

        new_tick_locations = np.array(range(self.n_det)) + 1

        ax.set_xticks(new_tick_locations)
        ax.set_yticks(new_tick_locations)

        xl = np.linspace(0.5, self.n_det + 0.5, self.n_det * 1 + 1)
        ax.vlines(xl, ymin=0.5, ymax = self.n_det + 0.5, linestyle='-', color='k', lw=0.05, alpha=1.0)
        ax.hlines(xl, xmin=0.5, xmax = self.n_det + 0.5, linestyle='-', color='k', lw=0.05, alpha=1.0)

        for i in range(self.n_det):
            plt.text(xl[i] + 0.7, xl[i] + 0.2, str(i+1), rotation=0, verticalalignment='center', fontsize=2)

        ax.set_xlabel('Feed')
        ax.set_ylabel('Feed')
        ax.set_title('Scan: ' + str(self.scan_id))

        # save_string = '_%s_%07i_%02i_00_AB.png' % (self.date, int(self.scan_id[:-2]), int(self.scan_id[-2:]))
        save_string = '_%07i_%02i_00_AB.png' % (int(self.scan_id[:-2]), int(self.scan_id[-2:]))
        
        self.ensure_dir_exists(self.outpath + 'corr_hr')
        self.ensure_dir_exists(self.outpath + 'corr_lr')
        
        plt.savefig(self.outpath + 'corr_hr/corr_highres' + save_string, bbox_inches='tight', dpi=800)
        plt.savefig(self.outpath + 'corr_lr/corr_lowres' + save_string, bbox_inches='tight', dpi=150) 

    def plot_scan_acc_chi2(self):
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(311)
    

        x = np.linspace(0.5, self.n_det + 0.5, len(self.acc) + 1)
        plotter = np.zeros((len(self.acc) + 1))
        plotter[:-1] = self.acc
        eff_feeds = (self.acc.sum()/4.0)

        ax.step(x, plotter, where='post', label='Effective feeds: ' + '%.2f' % eff_feeds)

        ax.vlines(np.linspace(0.5, self.n_det + 0.5, self.n_det + 1), ymin=0, ymax=1, linestyle='--', color='k', lw=0.6, alpha=0.2)
        ax.vlines(np.linspace(0.5, self.n_det + 0.5, self.n_det * 4 + 1), ymin=0, ymax=1, linestyle='--', color='k', lw=0.2, alpha=0.2)
        new_tick_locations = np.array(range(self.n_det)) + 1
        ax.set_xticks(new_tick_locations)
        ax.set_xlim(0.5, self.n_det + 0.48)
        ax.set_ylim(0.0, 1)

        ax.set_ylabel('Acceptance rate')

        plt.legend()

        ax = fig.add_subplot(312)
        ymin = -5
        ymax = 5

        x = np.linspace(0.5, self.n_det + 0.5, len(self.chi2))
        plotter = np.zeros((len(self.chi2)))
        plotter = self.chi2
        chi2mean = np.nanmean(self.chi2)

        ax.plot(x, plotter, label=r'$\langle \chi^2\rangle$: ' + '%.2f' % chi2mean)

        ax.vlines(np.linspace(0.5, self.n_det + 0.5, self.n_det + 1), ymin=ymin, ymax=ymax, linestyle='--', color='k', lw=0.6, alpha=0.2)
        ax.vlines(np.linspace(0.5, self.n_det + 0.5, self.n_det * 4 + 1), ymin=ymin, ymax=ymax, linestyle='--', color='k', lw=0.2, alpha=0.2)

        new_tick_locations = np.array(range(self.n_det)) + 1
        ax.set_xticks(new_tick_locations)

        ax.set_xlim(0.5, self.n_det + 0.48)
        ax.set_ylim(ymin, ymax)

        ax.set_ylabel(r'$\chi^2$')
        plt.legend()
        
        ax = fig.add_subplot(313)
        ymin = 20
        ymax = 100

        x = np.linspace(0.5, self.n_det + 0.5, len(self.tsys) + 1)
        plotter = np.zeros((len(self.tsys) + 1))
        plotter[:-1] = self.tsys
        tsysmean = np.nanmean(self.tsys)

        ax.plot(x, plotter, '.', markersize=1.5, label=r'$\langle T_{sys}\rangle$: ' + '%.2f' % tsysmean)

        ax.vlines(np.linspace(0.5, self.n_det + 0.5, self.n_det + 1), ymin=ymin, ymax=ymax, linestyle='--', color='k', lw=0.6, alpha=0.2)
        ax.vlines(np.linspace(0.5, self.n_det + 0.5, self.n_det * 4 + 1), ymin=ymin, ymax=ymax, linestyle='--', color='k', lw=0.2, alpha=0.2)

        new_tick_locations = np.array(range(self.n_det)) + 1
        ax.set_xticks(new_tick_locations)

        ax.set_xlim(0.5, self.n_det + 0.48)
        ax.set_ylim(ymin, ymax)

        ax.set_xlabel('Feed')
        ax.set_ylabel(r'$T_{sys}$')

        plt.legend()
        reason = self.reasonarr
        str_id = ['fixed mask', 'NaN', 'outl. freq', 'bad sb', 'corr. struct', 'edge corrs']
        
        inds = np.argsort(reason)[::-1]
        
        figstrings = [str_id[k] + ': %.2f \n' % reason[k] for k in inds]
        figstring = "".join(figstrings)


        plt.figtext(0.92, 0.7, figstring[:-2])

        # save_string = 'acc_chi2_%s_%07i_%02i_00_AB.png' % (self.date, int(self.scan_id[:-2]), int(self.scan_id[-2:]))
        save_string = 'acc_chi2_%07i_%02i_00_AB.png' % (int(self.scan_id[:-2]), int(self.scan_id[-2:]))
        
        self.ensure_dir_exists(self.outpath + 'acc_chi2')
        
        plt.savefig(self.outpath + 'acc_chi2/' + save_string, bbox_inches='tight')
        
    def plot_scan_pca_amplitudes(self):
        a2 = np.abs(self.ampl ** 2).mean((1, 2, 3))
        fig = plt.figure(figsize=(6, 12))

        for i in range(min(self.n_pca_comp, 3)):
            subplot = int(f"{611 + 2 * i}")

            var_exp = a2[i] * (self.pca[i]).std() ** 2 / self.radiometer ** 2 
            ax2 = fig.add_subplot(int(subplot))
            ax2.plot(self.time, self.pca[i], label=str(self.scan_id) + ", PCA comp. " + str(i+1))

            ax2.legend()

            ax2.set_xlim(0, self.time[-1])
            ax2.set_xlabel('time [m]')

            subplot = int(f"{611 + 2 * i + 1}")

            ax = fig.add_subplot(subplot)

            acc = self.ampl[i].flatten()
            n = len(acc)

            n_dec1 = 16
            acc = np.abs(acc.reshape((n // n_dec1, n_dec1)).mean(1))

            n_dec2 = 64
            acc = np.abs(acc.reshape((n // (n_dec1 * n_dec2), n_dec2))).mean(1)

            n = len(acc)

            acc = 100 * np.sqrt(acc ** 2 * (self.pca[i]).std() ** 2 / self.radiometer ** 2)

            x = np.linspace(0.5, self.n_det + 0.5, len(acc) + 1)
            plotter = np.zeros((len(acc) + 1))
            plotter[:-1] = acc

            ax.step(x, plotter, label='Avg std explained: %.1f %%' % (acc.mean()), where='post')

            ax.vlines(np.linspace(0.5, self.n_det + 0.5, self.n_det + 1), ymin=-1, ymax=100, linestyle='--', color='k', lw=0.6, alpha=0.2)
            ax.vlines(np.linspace(0.5, self.n_det + 0.5, self.n_det * 4 + 1), ymin=-1, ymax=100, linestyle='--', color='k', lw=0.2, alpha=0.2)

            new_tick_locations = np.array(range(self.n_det)) + 1
            ax.set_xticks(new_tick_locations)
            ax.set_xlim(0.5, self.n_det + 0.48)
            ax.set_ylim(0, 60)

            ax.set_xlabel('Feed')
            ax.set_ylabel(r'% of $1 / \sqrt{B\tau}$')

            ax.legend(loc=1)

        # save_string = 'pca_ampl_time_%s_%07i_%02i_00_AB.png' % (self.date, int(self.scan_id[:-2]), int(self.scan_id[-2:]))

        save_string = 'pca_ampl_time_%07i_%02i_00_AB.png' % (int(self.scan_id[:-2]), int(self.scan_id[-2:]))
        
        self.ensure_dir_exists(self.outpath + 'pca_ampl')        
        plt.savefig(self.outpath + "pca_ampl/" + save_string, bbox_inches='tight')
        
    def plot_scan_pca_components(self):
        ampl2 = self.ampl[:] / self.mask_hr[None, :]

        self.ampl[:, :, (0, 2)] = self.ampl[:, :, (0, 2), ::-1]
        ampl2[:, :, (0, 2)] = ampl2[:, :, (0, 2), ::-1]

        for j in range(len(self.ampl[:, 0, 0, 0])):
            fig = plt.figure(figsize=(12, 10))

            gs = fig.add_gridspec(22, 2, width_ratios=[3, 2])

            fig.suptitle('PCA, %s %s scan: ' % (self.patch_name, self.scanmode) + self.scan_id + ', mode:' + str(j+1) + ', eigv.: %.2f' % (self.eigv[j]) 
                        + ', windspeed: %.2f m/s' % (self.windspeed), fontsize=12)
            fig.subplots_adjust(top=0.915)

            maxval = 0.15 

            ax3 = fig.add_subplot(gs[:2, :])

            ax3.plot(self.time, self.pca[j]) 

            ax3.set_ylabel('pca mode')
            ax3.set_xlim(0, self.time[-1])
            ax3.set_xlabel('time [m]')
            ax3.set_yticklabels(())
            ax3.set_yticks(())
            ax3.yaxis.set_minor_formatter(mticker.NullFormatter())
            ax3.xaxis.tick_top()
            ax3.xaxis.set_label_position('top') 
            for i in range(ampl2.shape[1]):

                ax = fig.add_subplot(gs[i+2, 0])
                ax2 = fig.add_subplot(gs[i+2, 1])

                nu = np.linspace(26, 34, 4096 + 1)[:-1]
                dnu = nu[1] - nu[0]
                nu = nu + dnu / 2
                a = self.ampl[j, i, :, :].flatten()
                am = ampl2[j, i, :, :].flatten()
                ax.plot(nu, am)
                ax.yaxis.set_label_position("right")
                h = ax.set_ylabel(str(i+1))
                
                ax.set_ylim(-maxval, maxval)
                ax.set_xlim(26, 34)
                ax.set_xlabel('Frequency [GHz]')

                if i < 19:
                    ax.set_xticklabels(())
                    ax2.set_xticklabels(())
                
                f = np.abs(fft.rfftfreq(len(a), dnu * 1e3))[1:]

                ps = (np.abs(fft.rfft(a)) ** 2 / len(a))[1:]

                ax2.plot(f, ps)

                ax2.set_xscale('log')
                ax2.set_yscale('log')

                ax2.set_xlim(2e-3, 2e-2)
                ax2.set_ylim(1e-7, 1e0)
                if i == 19:
                    ax2.set_xlabel('[1/MHz]')
                if i < 19:
                    ax2.set_xticks(())

                ax2.xaxis.set_minor_formatter(mticker.NullFormatter())

            # save_string = 'pca_%02i_%s_%07i_%02i_00_AB.png' % (j + 1, self.date, int(self.scan_id[:-2]), int(self.scan_id[-2:]))
            save_string = 'pca_%02i_%07i_%02i_00_AB.png' % (j + 1, int(self.scan_id[:-2]), int(self.scan_id[-2:]))

            self.ensure_dir_exists(self.outpath + 'pca_comp')
            plt.savefig(self.outpath + "pca_comp/" + save_string, bbox_inches='tight')

    def plot_feed_pca_components(self):
        ampl2 = self.ampl_feed[:] / self.mask_hr[None, :]

        self.ampl_feed[:, :, (0, 2)] = self.ampl_feed[:, :, (0, 2), ::-1]
        ampl2[:, :, (0, 2)] = ampl2[:, :, (0, 2), ::-1]
        for i in range(ampl2.shape[1]):
        
            fig = plt.figure(figsize=(12, 10))

            gs = fig.add_gridspec(12, 2, width_ratios=[3, 2])

            fig.suptitle('Feed PCA, %s %s scan: ' % (self.patch_name, self.scanmode) + self.scan_id 
                        + ', feed %i, %i modes, windspeed: %.2f m/s' % (i+1, self.n_pca_feed_comp[i], self.windspeed), fontsize=12)
            fig.subplots_adjust(top=0.915)

            maxval = 0.15 

            for j in range(4):#len(self.ampl[:, 0, 0, 0])):
                

                ax3 = fig.add_subplot(gs[3*j:3*j+2, :])

                ax3.plot(self.time, self.pca_feed[j, i]) 

                ax3.set_ylabel('pca mode %i' % (j+1))
                ax3.set_xlim(0, self.time[-1])
                if j == 0:
                    ax3.set_xlabel('time [m]')
                    ax3.xaxis.set_label_position('top')
                else:
                    ax3.set_xticklabels(())
                ax3.xaxis.tick_top()
                ax3.set_yticklabels(())
                ax3.set_yticks(())
                ax3.yaxis.set_minor_formatter(mticker.NullFormatter())


                ax = fig.add_subplot(gs[3*j+2, 0])
                ax2 = fig.add_subplot(gs[3*j+2, 1])

                nu = np.linspace(26, 34, 4096 + 1)[:-1]
                dnu = nu[1] - nu[0]
                nu = nu + dnu / 2
                a = self.ampl_feed[j, i, :, :].flatten()
                am = ampl2[j, i, :, :].flatten()
                ax.plot(nu, am)
                ax.yaxis.set_label_position("right")
                h = ax.set_ylabel(str(i+1))
                
                ax.set_ylim(-maxval, maxval)
                ax.set_xlim(26, 34)
                ax.set_xlabel('Frequency [GHz]')

                if j < 3:
                    ax.set_xticklabels(())
                    ax2.set_xticklabels(())
                
                f = np.abs(fft.rfftfreq(len(a), dnu * 1e3))[1:]
                print(np.max(self.pca_feed[j, i]), i, j)
                if np.sum(self.pca_feed[j, i] ** 2) == 0.0:
                    ps = (np.abs(fft.rfft(a)) ** 2 / len(a))[1:]
                else:
                    ps = np.zeros_like(f) + np.nan

                ax2.plot(f, ps)

                ax2.set_xscale('log')
                ax2.set_yscale('log')

                ax2.set_xlim(2e-3, 2e-2)
                ax2.set_ylim(1e-7, 1e0)
                if j == 3:
                    ax2.set_xlabel('[1/MHz]')
                if j < 3:
                    ax2.set_xticks(())

                ax2.xaxis.set_minor_formatter(mticker.NullFormatter())

            # save_string = 'pca_%02i_%s_%07i_%02i_00_AB.png' % (j + 1, self.date, int(self.scan_id[:-2]), int(self.scan_id[-2:]))
            save_string = 'pca_feed_%07i_%02i_%02i_AB.png' % (int(self.scan_id[:-2]), int(self.scan_id[-2:]), i+1)

            self.ensure_dir_exists(self.outpath + 'pca_feed_comp')
            plt.savefig(self.outpath + "pca_feed_comp/" + save_string, bbox_inches='tight')

    def plot_scan_histogram(self):
        tsys  =self.tsys.reshape(self.n_det, self.n_sb, self.n_freq)
        tsys[np.where(tsys == 0.0)] = np.inf

        tod2 = (self.tod_hist / (tsys[:, :, :, None] * self.radiometer) * np.sqrt(self.mask_full[:, :, :, None] / 16)).flatten()  # tod.flatten() / radiometer#(tod / radiometer * np.sqrt(mask_full[:, :, :, None] / 16)).flatten()
        tod2 = tod2[np.where(np.abs(tod2) > 0)]
        tod2 = tod2[np.where(np.abs(tod2) < 20)]
        std = np.nanstd(tod2)

        fig = plt.figure(figsize=(8, 4))
        ax1 = fig.add_subplot(121)
        x = np.linspace(-4 * std, 4 * std, 300)
        ax1.plot(x, norm.pdf(x), 'g', lw=2, label=r'$\mathcal{N}(0, 1)$', zorder=1)
        ax1.plot(x, norm.pdf(
            x, scale=std), 'r', lw=2,
            label=r'$\mathcal{N}(0, \sigma_\mathrm{samp})$',
            zorder=2)
        ax1.hist(tod2, bins=x, density=True, label='All samples, ', alpha=0.8, zorder=3)

        if not np.any(np.isnan(x)) and not np.any(np.isinf(x)):
            ax1.set_xlim(x[0], x[-1])

        ax1.set_xlabel(r'$x \cdot \sqrt{B\tau}$')
        ax1.set_ylabel(r'$p(x\cdot \sqrt{B\tau})$')
        ax1.text(-3.95, .395, 'Scan: ' + str(self.scan_id), fontsize=10) 

        ax2 = fig.add_subplot(122)
        x = np.linspace(-6 * std, 6 * std, 400)
        ax2.plot(x, norm.pdf(x), 'g', lw=2, label=r'$\mathcal{N}(0, 1)$', zorder=1)
        ax2.plot(x, norm.pdf(
            x, scale=std), 'r', lw=2,
            label=r'$\mathcal{N}(0, \sigma_\mathrm{samp})$',
            zorder=2)
        ax2.hist(tod2, bins=x, density=True, label='All samples, ', alpha=0.8, zorder=3)
        ax2.set_yscale('log')
        ax2.legend()

        if not np.any(np.isnan(x)) and not np.any(np.isinf(x)):
            ax2.set_xlim(x[0], x[-1])

        ax2.set_ylim(1e-6, 1e0)
        ax2.set_xlabel(r'$x \cdot \sqrt{B\tau}$')

        save_string = 'hist_%08i.png' % (int(self.scan_id))

        self.ensure_dir_exists(self.outpath + 'hist')
        plt.savefig(self.outpath + "hist/" + save_string, bbox_inches='tight')

    def plot_scan_acc_var(self):
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(211)

        n = len(self.acc)

        x = np.linspace(0.5, self.n_det + 0.5, len(self.acc) + 1)
        plotter = np.zeros((len(self.acc) + 1))
        plotter[:-1] = self.acc  ## is this wrong???

        ax.step(x, plotter, where='post', label='Mean acceptrate: ' + str(self.acc.mean() * 20 / 19))

        ax.vlines(np.linspace(0.5, self.n_det + 0.5, self.n_det + 1), ymin=0, ymax=1, linestyle='--', color='k', lw=0.6, alpha=0.2)
        ax.vlines(np.linspace(0.5, self.n_det + 0.5, self.n_det * 4 + 1), ymin=0, ymax=1, linestyle='--', color='k', lw=0.2, alpha=0.2)

        new_tick_locations = np.array(range(self.n_det)) + 1
        ax.set_xticks(new_tick_locations)

        ax.set_xlim(0.5, self.n_det + 0.48)
        ax.set_ylim(0.75, 1)

        ax.set_ylabel('Acceptance rate')
        ax.legend(loc=3)

        tsys  =self.tsys.reshape(self.n_det, self.n_sb, self.n_freq)

        tod = self.tod_hist[:, :, :, :]
        tod = tod * np.sqrt(self.mask_full[:, :, :, None] / 16)

        std = tod.std(3)
        std[np.where(std != 0)] = std[np.where(std != 0)] / tsys[np.where(std != 0)]
        mask = np.zeros_like(std)
        mask[np.where(std != 0)] = 1.0
        mask = mask.reshape(self.n_det, self.n_sb, self.n_freq)

        mean_std = std.sum(2)
        mean_std[np.where(mean_std > 0)] = mean_std[np.where(mean_std > 0)] / mask.sum(2)[np.where(mean_std > 0)]
        mean_std = mean_std.flatten()
        mean_std[np.where(mean_std == 0)] = 1e3
        x = np.linspace(0.5, self.n_det + 0.5, len(mean_std) + 1)

        ax = fig.add_subplot(212)
        plotter = np.zeros((len(mean_std) + 1))
        plotter[:-1] = mean_std

        ax.step(x, plotter / self.radiometer, where='post', label=str(self.scan_id))

        new_tick_locations = np.array(range(self.n_det)) + 1
        ax.set_xticks(new_tick_locations)
        ax.set_xlim(0.5, self.n_det + 0.5)

        ax.vlines(np.linspace(0.5, self.n_det + 0.5, self.n_det + 1), ymin=1, ymax=2, linestyle='--', color='k', lw=0.6, alpha=0.2)
        ax.vlines(np.linspace(0.5, self.n_det + 0.5, self.n_det * self.n_sb + 1), ymin=1, ymax=2, linestyle='--', color='k', lw=0.2, alpha=0.2)

        for i in range(len(mean_std)):
            if mean_std[i] == 1e3:
                ax.axvspan(x[i], x[i + 1], alpha=1.0, color='gray', zorder=5)

        ax.set_ylim(1, 1.4)
        ax.set_xlim(0.5, self.n_det + 0.48)
        ax.legend(loc=0)
        
        ax.set_xlabel('Feed')
        ax.set_ylabel(r'$\langle\sigma_\mathrm{TOD}\rangle \cdot \sqrt{B\tau}$')
        
        save_string = 'acc_var_%08i.png' % (int(self.scan_id))

        self.ensure_dir_exists(self.outpath + 'acc_var')
        plt.savefig(self.outpath + "acc_var/" + save_string, bbox_inches='tight')

    def plot_scan_diagnostics(self): 
        t0 = tm.time()
        print(self.obsid, self.scan_id)

        self.plot_scan_spikes()
        print("Spike plot:", tm.time() - t0, "s")
        t0 = tm.time()

        self.plot_scan_correlation()
        print("Correlation plot:", tm.time() - t0, "s")
        t0 = tm.time()
        
        self.plot_scan_acc_chi2()
        print("Acc chi2 plot:", tm.time() - t0, "s")
        t0 = tm.time()
        
        self.plot_feed_pca_components()
        print("PCA feed. plot:", tm.time() - t0, "s")
        t0 = tm.time()
        # self.plot_scan_pca_amplitudes()
        # print("PCA ampl. plot:", tm.time() - t0, "s")
        # t0 = tm.time()

        self.plot_scan_pca_components()
        print("PCA comp. plot:", tm.time() - t0, "s")
        t0 = tm.time()


        # self.plot_scan_histogram()
        # print("Histogram plot:", tm.time() - t0, "s")
        # t0 = tm.time()
        
        # self.plot_scan_acc_var()
        # print("Acc var plot:", tm.time() - t0, "s")

    def plot_ps_chi2_data(self):
        ps_chi2 = self.ps_chi2 
        ps_s_feed = self.ps_s_feed 
        ps_s_chi2 = self.ps_s_chi2 
        ps_o_sb = self.ps_o_sb   
        ps_o_feed = self.ps_o_feed 
        ps_o_chi2 = self.ps_o_chi2 

        
        fig5 = plt.figure(figsize=(12, 10))  # constrained_layout=True, 

        n_cols = 2 * self.n_scans
        n_rows = 5

        widths = np.zeros(n_cols)
        heights = np.zeros(n_rows)

        heights = [10, 1, 3, 1, 1]

        for i in range(self.n_scans):
            widths[2 * i] = 3
            widths[2 * i + 1] = 1

        print("Generating ps chi2 plot:")

        spec5 = fig5.add_gridspec(ncols=n_cols, nrows=n_rows, width_ratios=widths,
                                height_ratios=heights)
        vmax = 10
        self.n_det = 20
        row = 0
        ax1 = fig5.add_subplot(spec5[row, 0])
        im = ax1.imshow(ps_chi2[0], interpolation = 'none', aspect = 'auto', vmin=-vmax, vmax=vmax, extent=(0.5, 4.5, self.n_det + 0.5, 0.5))
        new_tick_locations = np.array(range(self.n_det)) + 1
        ax1.set_yticks(new_tick_locations)
        x_tick_loc = [1, 2, 3, 4]
        x_tick_labels = ['LA', 'UA', 'LB', 'UB']
        ax1.set_xticks(x_tick_loc)
        # ax1.set_xticklabels(x_tick_labels, rotation=90)
        #ax1.title.set_text(str(self.scanids[0]))
        ax1.set_title(str(self.scanids[0]), rotation = 45)

        fig5.subplots_adjust(right=0.85)
        fig5.suptitle('Obsid:' + str(self.scanids[0])[:-2], fontsize=20)
        # cbar_ax = fig5.add_axes([0.15, 0.90, 0.70, 0.02])
        cbar_ax = fig5.add_axes([0.87, 0.1, 0.02, 0.8]) 
        cbar = fig5.colorbar(im, cax=cbar_ax)#, orientation='horizontal')
        cbar.set_label('ps_chi2')

        ax2 = fig5.add_subplot(spec5[row, 1], sharey=ax1)
        ax2.imshow(ps_s_feed[0, :], interpolation = 'none', aspect = 'auto', vmin=-vmax, vmax=vmax, extent=(0.5, 1.5, self.n_det + 0.5, 0.5))
        new_tick_locations = np.array(range(self.n_det))+1
        ax2.set_yticks(new_tick_locations)
        x_tick_loc = []
        ax2.set_xticks(x_tick_loc)
        #ax2.set_xlabel(['feed'])

        plt.setp(ax2.get_yticklabels(), visible=False)
        row = 1
        ax = fig5.add_subplot(spec5[row, 0: 2])
        # print(ps_s_chi2[6])
        ax.imshow(ps_s_chi2[0], interpolation = 'none', aspect = 'auto', vmin=-vmax, vmax=vmax)
        plt.setp(ax.get_yticklabels(), visible=False)
        plt.setp(ax.get_xticklabels(), visible=False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel('scan')
        for scan in range(1, self.n_scans):
            row = 0
            ax3 = fig5.add_subplot(spec5[row, 2 * scan], sharey=ax1)
            ax3.imshow(ps_chi2[scan], interpolation = 'none', aspect = 'auto', vmin=-vmax, vmax=vmax, extent=(0.5, 4.5, self.n_det + 0.5, 0.5))
            new_tick_locations = np.array(range(self.n_det))+1
            ax3.set_yticks(new_tick_locations)
            #ax3.title.set_text(str(self.scanids[scan]), rotation = 45)
            ax3.set_title(str(self.scanids[scan]), rotation = 45)
            x_tick_loc = [1, 2, 3, 4]
            x_tick_labels = ['LA', 'UA', 'LB', 'UB']
            ax3.set_xticks(x_tick_loc)
            # ax1.set_xticklabels(x_tick_labels, rotation=90)
            plt.setp(ax3.get_yticklabels(), visible=False)
            ax4 = fig5.add_subplot(spec5[row, 2 * scan+1], sharey=ax1)
            ax4.imshow(ps_s_feed[scan], interpolation = 'none', aspect = 'auto', vmin=-vmax, vmax=vmax, extent=(0.5, 1.5, self.n_det + 0.5, 0.5))
            new_tick_locations = np.array(range(self.n_det))+1
            ax4.set_yticks(new_tick_locations)
            x_tick_loc = []
            ax4.set_xticks(x_tick_loc)
            plt.setp(ax4.get_yticklabels(), visible=False)
            row = 1
            ax = fig5.add_subplot(spec5[row, 2 * scan:2 * scan + 2])
            ax.imshow(ps_s_chi2[scan], interpolation = 'none', aspect = 'auto', vmin=-vmax, vmax=vmax)
            plt.setp(ax.get_yticklabels(), visible=False)
            plt.setp(ax.get_xticklabels(), visible=False)
            ax.set_xticks([])
            ax.set_yticks([])
            # label = 'Width: {}\nHeight: {}'.format(1, heights[row])
            # ax.annotate(label, (0.1, 0.5), xycoords='axes fraction', va='center')
        row = 2 
        ax6 = fig5.add_subplot(spec5[row, :])
        ax6.title.set_text('Data from full obsid:')
        # print(ps_o_sb.T)
        im = ax6.imshow(ps_o_sb.T, interpolation = 'none', aspect = 'auto', vmin=-vmax, vmax=vmax, extent=(0.5, self.n_det + 0.5, 4.5, 0.5))
        new_tick_locations = np.array(range(self.n_det))+1
        ax6.set_xticks(new_tick_locations)
        y_tick_loc = [1, 2, 3, 4]
        y_tick_labels = ['LA', 'UA', 'LB', 'UB']
        ax6.set_yticks(y_tick_loc)
        # ax1.set_xticklabels(x_tick_labels, rotation=90)
        # ax1.title.set_text(str(self.scanids[0]))
        row = 3 
        ax7 = fig5.add_subplot(spec5[row, :])
        im = ax7.imshow(ps_o_feed.T, interpolation = 'none', aspect = 'auto', vmin=-vmax, vmax=vmax, extent=(0.5, self.n_det + 0.5, 1.5, 0.5))
        new_tick_locations = np.array(range(self.n_det))+1
        ax7.set_xticks(new_tick_locations)
        ax7.set_xticklabels([])
        y_tick_loc = []
        y_tick_labels = ['LA', 'UA', 'LB', 'UB']
        ax7.set_yticks(y_tick_loc)
        row = 4 
        ax8 = fig5.add_subplot(spec5[row, :])
        ax8.imshow(ps_o_chi2, interpolation = 'none', aspect = 'auto', vmin=-vmax, vmax=vmax)
        plt.setp(ax8.get_yticklabels(), visible=False)
        plt.setp(ax8.get_xticklabels(), visible=False)
        ax8.set_xticks([])
        ax8.set_yticks([])
        #plt.imshow(ps_chi2[0], interpolation='none')
        # plt.tight_layout()
        outname = f"ps_chi2_%07i.png" % int(str(self.scanids[0])[:-2])

        self.ensure_dir_exists(self.outpath + 'ps_chi2')

        plt.savefig(self.outpath + "ps_chi2/" + outname, bbox_inches = 'tight', dpi=100)
        
    def plot_obsid_correlation(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        vmax = 0.05
        im = ax.imshow(self.corr, vmin = -vmax, vmax = vmax,
                        extent = (0.5, self.n_det + 0.5, self.n_det + 0.5, 0.5))
        cbar = fig.colorbar(im)
        cbar.set_label('Correlation')
        new_tick_locations = np.array(range(self.n_det)) + 1
        ax.set_xticks(new_tick_locations)
        ax.set_yticks(new_tick_locations)
        
        xl = np.linspace(0.5, self.n_det + 0.5, self.n_det * 1 + 1)
        ax.vlines(xl, ymin = 0.5, ymax = self.n_det + 0.5, linestyle='-', color='k', lw=0.05, alpha=1.0)
        ax.hlines(xl, xmin = 0.5, xmax = self.n_det + 0.5, linestyle='-', color='k', lw=0.05, alpha=1.0)
        
        for j in range(self.n_det):
            plt.text(xl[j] + 0.7, xl[j] + 0.2, str(j+1), rotation=0, verticalalignment='center', fontsize=2)
        
        ax.set_xlabel('Feed')
        ax.set_ylabel('Feed')
        ax.set_title('Obsid: ' + str(self.obsid))

        save_string = '_%07i.png' % (int(self.obsid))
        
        self.ensure_dir_exists(self.outpath + 'corr_hr')
        self.ensure_dir_exists(self.outpath + 'corr_lr')
        
        plt.savefig(self.outpath + 'corr_hr/' + 'corr_highres' + save_string, bbox_inches='tight', dpi=800) 
        plt.savefig(self.outpath + 'corr_lr/' + 'corr_lowres' + save_string, bbox_inches='tight', dpi=150) 
        
    def plot_obsid_acc_chi2(self):
        fig = plt.figure(figsize = (6, 6))
        ax = fig.add_subplot(311)
        
        ax.set_title('Obsid: ' + str(self.obsid) + ', utc - 7: %02i [h]' % round(self.t0))

        n = len(self.acc)

        x = np.linspace(0.5, self.n_det + 0.5, len(self.acc) + 1)
        plotter = np.zeros((len(self.acc) + 1))
        plotter[:-1] = self.acc
        self.eff_feeds = (self.acc.sum()/4.0)
        
        ax.step(x, plotter, where='post', label='Effective feeds: ' + '%.2f' % self.eff_feeds)

        ax.vlines(np.linspace(0.5, self.n_det + 0.5, self.n_det + 1), ymin=0, ymax=1, linestyle='--', color='k', lw=0.6, alpha=0.2)
        ax.vlines(np.linspace(0.5, self.n_det + 0.5, self.n_det * 4 + 1), ymin=0, ymax=1, linestyle='--', color='k', lw=0.2, alpha=0.2)
        
        new_tick_locations = np.array(range(self.n_det)) + 1
        ax.set_xticks(new_tick_locations)
        ax.set_xlim(0.5, self.n_det + 0.48)
        ax.set_ylim(0.0, 1)

        ax.set_ylabel('Acceptance rate')

        plt.legend()

        chi2s = np.array(self.chi2s)
        n_samp = 1 / np.array(self.variances)
        chi2 = (np.nansum(self.chi2s * np.sqrt(2 * n_samp[:, None]), 0)) / np.sqrt(2 * np.sum(n_samp[:, None] * (np.isfinite(self.chi2s)), 0))
    
        ax = fig.add_subplot(312)#211)
        ymin = -5
        ymax = 5
        n = len(chi2)

        x = np.linspace(0.5, self.n_det + 0.5, len(chi2))
        plotter = np.zeros((len(chi2)))
        plotter = chi2
        self.chi2mean = np.nanmean(chi2)

        ax.plot(x, plotter, label=r'$\langle \chi^2\rangle$: ' + '%.2f' % self.chi2mean)

        ax.vlines(np.linspace(0.5, self.n_det + 0.5, self.n_det + 1), ymin = ymin, ymax = ymax, linestyle = '--', color = 'k', lw = 0.6, alpha = 0.2)
        ax.vlines(np.linspace(0.5, self.n_det + 0.5, self.n_det * 4 + 1), ymin = ymin, ymax = ymax, linestyle = '--', color = 'k', lw = 0.2, alpha = 0.2)

        new_tick_locations = np.array(range(self.n_det)) + 1
        ax.set_xticks(new_tick_locations)
        ax.set_xlim(0.5, self.n_det + 0.48)
        ax.set_ylim(ymin, ymax)

        ax.set_ylabel(r'$\chi^2$')

        plt.legend()

        tsys = np.nanmean(np.array(self.tsyss), 0)

        ax = fig.add_subplot(313)#211)

        ymin = 20
        ymax = 100
        n = len(tsys)

        x = np.linspace(0.5, self.n_det + 0.5, len(tsys) + 1)
        plotter = np.zeros((len(tsys) + 1))
        plotter[:-1] = tsys
        self.tsysmean = np.nanmean(tsys)

        ax.plot(x, plotter, '.', markersize = 1.5, label = r'$\langle T_{sys}\rangle$: ' + '%.2f' % self.tsysmean)

        ax.vlines(np.linspace(0.5, self.n_det + 0.5, self.n_det + 1), ymin = ymin, ymax = ymax, linestyle = '--', color = 'k', lw = 0.6, alpha = 0.2)
        ax.vlines(np.linspace(0.5, self.n_det + 0.5, self.n_det * 4 + 1), ymin = ymin, ymax = ymax, linestyle = '--', color = 'k', lw = 0.2, alpha = 0.2)

        new_tick_locations = np.array(range(self.n_det)) + 1
        ax.set_xticks(new_tick_locations)
        ax.set_xlim(0.5, self.n_det + 0.48)
        ax.set_ylim(ymin, ymax)

        ax.set_ylabel(r'$T_{sys}$')

        plt.legend()

        str_id = ['fixed mask', 'NaN', 'outl. freq', 'bad sb', 'corr. struct', 'edge corrs']
        
        inds = np.argsort(self.reason)[::-1]
        
        figstrings = [str_id[k] + ': %.2f \n' % self.reason[k] for k in inds]
        figstring = "".join(figstrings)

        plt.figtext(0.92, 0.7, figstring[:-2])
        save_string = 'acc_chi2_%07i.png' % (int(self.obsid))

        self.ensure_dir_exists(self.outpath + 'acc_chi2')
        plt.savefig(self.outpath + 'acc_chi2/' + save_string, bbox_inches='tight')
    
    def plot_obsid_spikes(self):
        sortedlists = self.my_spikes.sorted()
        self.n_spikes = len(sortedlists[0])
        self.n_jumps = len(sortedlists[1])
        self.n_anom = len(sortedlists[2]) 

        n_spike_types = 3
        n_plots = 3

        fig = plt.figure(figsize=(12, 10))
        
        fig.suptitle('Obsid: %06i, %i spikes, %i jumps and %i anomalies. Top: 3 largest spikes.\
        Middle: 3 largest jumps. Bottom: 3 largest anomalies' % (
        int(self.obsid), self.n_spikes, self.n_jumps, self.n_anom))
        
        for i in range(n_spike_types):
            for j in range(n_plots):
                try:
                    spike = sortedlists[i][j]
                    ax = fig.add_subplot(n_spike_types, n_plots, n_plots * i + j + 1)
                    ind = spike.ind
                    lab = 'Feed %i, sb %i, Ampl: %.3f' % (ind[0] + 1, ind[1] + 1, spike.amp)
                    dt = 0.02  # sampling rate
                    mjd = (spike.mjd - self.tstart) * 24 * 3600
                    time = np.linspace(mjd - 200 * dt, mjd + 199 * dt, len(spike.data))
                    ax.plot(time, spike.data, label = lab)
                    ax.set_xlabel('time [s]')
                    plt.legend()
                    if j == 0:
                        ax.set_ylabel('sb avg tod')
                except IndexError:
                    ax = fig.add_subplot(n_spike_types, n_plots, n_plots * i + j + 1)
                    pass
       
        save_string = 'spikes_%07i.png' % (int(self.obsid))

        self.ensure_dir_exists(self.outpath + 'spikes')

        plt.savefig(self.outpath + 'spikes/' + save_string, bbox_inches='tight')

        plt.close('all')


    def plot_obsid_info_table(self):
        infodir = self.outpath + 'obsid_info/'
        self.ensure_dir_exists(infodir)

        columns = ( "date", "field", "scan-mode", "n_scans", "weather", "az", "el", "ambient temp", "windspeed", "sun distance", "scan_length")  # , "eff_feeds", r"$\langle \chi^2 \rangle$", r"$\langle T_\mathrm{sys} \rangle$")
        mjd = self.tstart
        time = Time(mjd, format='mjd')
        date = time.datetime.strftime("%Y-%m-%d %H:%M")
        index = self.stats_list.index('az')
        az = np.nanmean(self.scan_data_obsid[:, 0, 0, index])
        index = self.stats_list.index('airtemp')
        ambt = np.nanmean(self.scan_data_obsid[:, 0, 0, index])
        index = self.stats_list.index('windspeed')
        windspeed = np.nanmean(self.scan_data_obsid[:, 0, 0, index])
        index = self.stats_list.index('sun_dist')
        sun_dist = np.nanmean(self.scan_data_obsid[:, 0, 0, index])
        index = self.stats_list.index('el')
        el = np.nanmean(self.scan_data_obsid[:, 0, 0, index])
        index = self.stats_list.index('weather')
        weather = np.nanmax(self.scan_data_obsid[:, 0, 0, index])*100
        index = self.stats_list.index('fbit')
        fbit = np.nanmax(self.scan_data_obsid[0, 0, 0, index])
        index = self.stats_list.index('scan_length')
        scan_length = np.nanmean(self.scan_data_obsid[0, 0, 0, index])

        if fbit == 16:
            scanmode = 'circular'
        elif fbit == 32:
            scanmode = 'CES'
        elif fbit == 512:
            scanmode = 'lissajous'
        else:
            scanmode = 'unknown'

        data = [[date, self.patch_name, scanmode, self.n_scans, '%.1f%%' % weather, '%.2f deg' % az, '%.2f deg' % el, 
                '%.1f C' % ambt, '%.2f m/s' % windspeed, '%.1f deg' % sun_dist, '%.3f min' % scan_length]]  

        # Create a new figure with size 6x6 inches
        fig, ax = plt.subplots(figsize=(12, 0.5))

        # Remove plot frame
        ax.axis('tight')
        ax.axis('off')

        # Add the table
        the_table = ax.table(cellText=data,
                            colLabels=columns,
                            loc='center')

        # Rescale the table
        the_table.scale(1, 1.2)

        save_string = 'obsid_info_%07i.png' % (int(self.obsid))

        plt.savefig(infodir + save_string, dpi=300, bbox_inches='tight')



    # def save_obsid_stats(self):
    #     statsdir = self.outpath + 'l2stats/'

    #     self.ensure_dir_exists(statsdir)

    #     fields = ['co2', 'co6', 'co7']
    #     stats = np.zeros(50)
    #     stats[0] = int(self.obsid)
    #     stats[1] = self.n_scans
    #     stats[2] = fields.index(self.patch_name)
    #     stats[3] = 0.0  # Scanning strategy 
    #     stats[4] = self.tstart
    #     stats[5] = self.eff_feeds
    #     stats[6] = self.chi2mean
    #     stats[7] = self.tsysmean
    #     stats[8] = self.n_spikes
    #     stats[9] = self.n_jumps
    #     stats[10] = self.n_anom
        
    #     save_string = 'l2_stats_%07i.txt' % (int(self.obsid))
    #     np.savetxt(statsdir + save_string, stats)

    def plot_obsid_diagnostics(self):
        self.plot_obsid_correlation()
        self.plot_obsid_acc_chi2()
        # self.plot_obsid_spikes()
        # self.save_obsid_stats()
        self.plot_obsid_info_table()

if __name__ == "__main__":
    l2plotter = L2plots()
    l2plotter.run()
