import copy
from re import search
import numpy as np
import ctypes
from scipy.fft import fft, ifft, rfft, irfft, fftfreq, rfftfreq, next_fast_len
# from pixell import fft as pfft
from scipy.optimize import curve_fit
import h5py
from tqdm import trange
import time
import logging
import matplotlib.pyplot as plt
import os
import math
from scipy.ndimage import gaussian_filter1d
from mpi4py import MPI
from sklearn.decomposition import PCA
from simpipeline.l2gen_simulation_filters import Cube2TOD, Replace_TOD_With_Signal, Replace_TOD_with_WN, Replace_TOD_with_Tsys_WN
from scipy.interpolate import interp1d

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

class Filter:
    name = ""  # Short name of filter, used for compact writes.
    name_long = ""  # Verbose, more explanatory name of filter.
    run_when_masking = False  # If set to True, this filter will be applied to a local copy of the data before masking.
    has_corr_template = False # Set to True if the template impacts the correlation matrix and must be corrected (through the "get_corr_template" function).
    # def get_corr_template(l2, C, feed, sb):
        # return np.zeros_like(C)
    def run(self, l2):
        pass


class Subtract_Start_Exponential(Filter):
    name = "start_exp"
    name_long ="subtract_start_exponential"
    
    def __init__(self, params, use_ctypes=False, omp_num_threads=2):
        self.params = params
        
    def run(self, l2):
        def exp_func(time, amp, offset):
            decay_time = self.params.start_exponential_decay_time  # Seconds
            return amp/np.exp(time/decay_time) + offset

        if True:
            l2.tofile_dict["start_exp_solutions"] = np.zeros((l2.Nfeeds, l2.Nsb, l2.Nfreqs, 4))
            time = np.linspace(0, (l2.Ntod-1)/l2.samprate, l2.Ntod)
            B = np.zeros((l2.Ntod, 4))
            B[:,0] = np.ones(l2.Ntod)
            B[:,1] = np.linspace(0, 1, l2.Ntod)
            B[:,2] = exp_func(time, 1.0, 0.0)
            B[:,3] = B[:,2]*time
            B /= np.linalg.norm(B, axis=0)
            N_inv = 1.0/np.ones((l2.Ntod))
            for ifeed in range(l2.Nfeeds):
                for isb in range(l2.Nsb):
                    for ifreq in range(l2.Nfreqs):
                        if l2.freqmask[ifeed,isb,ifreq]:
                            d = l2.tod[ifeed,isb,ifreq,:]
                            N_inv_d = ifft(N_inv*fft(d)).real
                            BT_N_inv_d = B.T.dot(N_inv_d)
                            BT_N_inv_B_inv = np.linalg.inv(B.T.dot(fft(N_inv[:,None]*ifft(B))).real)
                            a = BT_N_inv_B_inv.dot(BT_N_inv_d)
                            l2.tod[ifeed,isb,ifreq] -= B[:,2]*a[2] + B[:,3]*a[3]  # Subtract exponential part only.
                            l2.tofile_dict["start_exp_solutions"][ifeed,isb,ifreq] = a

        else:
            self.start_exp_amp = np.zeros((l2.Nfeeds, l2.Nsb, l2.Nfreqs))
            self.start_exp_offset = np.zeros((l2.Nfeeds, l2.Nsb, l2.Nfreqs))
            time = np.linspace(0, 4999/l2.samprate, 5000)
            time_long = np.linspace(0, 9999/l2.samprate, 10000)
            for ifeed in range(l2.Nfeeds):
                for isb in range(l2.Nsb):
                    for ifreq in range(l2.Nfreqs):
                        if l2.freqmask[ifeed,isb,ifreq]:
                            try:
                                (amp, offset), pcov = curve_fit(exp_func, time, l2.tod[ifeed,isb,ifreq,:5000], maxfev=20000, p0=(0, np.nanmean(l2.tod[ifeed,isb,ifreq,:5000])))
                                l2.tod[ifeed,isb,ifreq,:10000] -= exp_func(time_long, amp, 0)
                                self.start_exp_amp[ifeed,isb,ifreq] = amp
                                self.start_exp_offset[ifeed,isb,ifreq] = offset
                            except:
                                pass  # curve_fit throws an error if it doesn't converge.
            l2.tofile_dict["start_exp_amp"] = self.start_exp_amp
            l2.tofile_dict["start_exp_offset"] = self.start_exp_offset


class Normalize_Gain_Gaussian(Filter):
    name = "norm3"
    name_long = "normalization_gaussian"
    
    def __init__(self, params, use_ctypes=False, omp_num_threads=2):
        self.params = params
        self.deci_factor = 20
    
    def run(self, l2):
        self.gauss_width_lowres = (self.params.gain_norm_gauss_sigma_seconds*l2.samprate)//self.deci_factor
        N_lowres = math.ceil(l2.Ntod/self.deci_factor)
        weight_on_final_point = (l2.Ntod%self.deci_factor)/self.deci_factor

        for ifeed in range(l2.Nfeeds):
            for isb in range(l2.Nsb):
                tod_lowres = np.zeros((l2.Nfreqs, N_lowres))
                mask_lowres = np.zeros((N_lowres))
                for i in range(N_lowres):
                    tod_lowres[:,i] = np.nanmean(l2.tod[ifeed,isb,:,i*self.deci_factor:(i+1)*self.deci_factor], axis=-1)
                    mask_lowres[i] = np.nanmean(l2.mask_temporal[ifeed,i*self.deci_factor:(i+1)*self.deci_factor])
                mask_lowres[-1] *= weight_on_final_point
                tod_lowres[~np.isfinite(tod_lowres)] = 0.0

                lowpass_lowres = gaussian_filter1d(tod_lowres*mask_lowres, self.gauss_width_lowres)
                lowpass_mask_lowres = gaussian_filter1d(mask_lowres, self.gauss_width_lowres)
                lowpass_lowres /= lowpass_mask_lowres
                lowpass = np.zeros((l2.Nfreqs, l2.Ntod))
                # for i in range(N_lowres):
                #     lowpass[:,i*self.deci_factor:(i+1)*self.deci_factor] = lowpass_lowres[:,i][:,None]
                idx_lowres = np.linspace(0, l2.Ntod-1, N_lowres)
                idx_highres = np.linspace(0, l2.Ntod-1, l2.Ntod)
                lowpass_interp = interp1d(idx_lowres, lowpass_lowres)
                lowpass = lowpass_interp(idx_highres)
                
                l2.tod[ifeed,isb] /= lowpass
                l2.tod[ifeed,isb] /= np.nanmean(l2.tod[ifeed,isb], axis=-1)[:,None]
                l2.tod[ifeed,isb] -= 1.0



class Normalize_Gain_Mean(Filter):
    name = "norm2"
    name_long = "normalization2"
    
    def __init__(self, params, use_ctypes=False, omp_num_threads=2):
        pass
    
    def run(self, l2):
        for ifeed in range(l2.Nfeeds):
            for isb in range(l2.Nsb):
                l2.tod[ifeed,isb] /= np.mean(l2.tod[ifeed,isb], axis=-1)[:,None]
                l2.tod[ifeed,isb] -= 1



class Normalize_Gain(Filter):
    name = "norm"
    name_long = "normalization"
    def __init__(self, params, use_ctypes=False, omp_num_threads=2):
        self.use_ctypes = use_ctypes
        self.omp_num_threads = omp_num_threads
        self.fknee = params.gain_norm_fknee
        self.alpha = params.gain_norm_alpha
    
    def lowpass_filter(self, signal, mask, fknee=0.01, alpha=4.0, samprate=50, num_threads=2):
        """ Applies a lowpass filter to a signal, and returns the low-frequency result (same shape as input).
            Assumes signal is of shape [freq, time]. Frequencies are calculated entirely independently.
            Uses weighting at the edges to deal with periodicity requirements.
        """
        Ntod = signal.shape[-1]
        Nfreq = signal.shape[0]
        fastlen = next_fast_len(int(1.5*Ntod))

        center_slice = slice(int(0.25*Ntod), int(1.25*Ntod))
        signal_padded = np.zeros((Nfreq, fastlen))
        mask_padded = np.zeros((fastlen))
        signal_padded[:,center_slice] = signal
        mask_padded[center_slice][mask] = 1.0

        freq_padded = rfftfreq(fastlen)*samprate
        W = 1.0/(1 + (freq_padded/fknee)**alpha)

        signal_padded = irfft(rfft(signal_padded)*W)
        mask_padded = irfft(rfft(mask_padded)*W)            
        signal_padded /= mask_padded[None,:]
        return signal_padded[:,center_slice]
        

    def run(self, l2):
        if not self.use_ctypes:
            for ifeed in range(l2.Nfeeds):
                for isb in range(l2.Nsb):
                    tod_lowpass = self.lowpass_filter(l2.tod[ifeed, isb], l2.mask_temporal[ifeed], num_threads=self.omp_num_threads, fknee=self.fknee, alpha=self.alpha, samprate=l2.samprate)
                    l2.tod[ifeed,isb] = l2.tod[ifeed,isb]/tod_lowpass - 1
            del(tod_lowpass)

class Normalize_Gain_Old(Filter):
    name = "norm_old"
    name_long = "normalization_old"

    def __init__(self, params, use_ctypes=False, omp_num_threads=2):
        self.use_ctypes = use_ctypes
        self.omp_num_threads = omp_num_threads
        self.fknee = params.gain_norm_fknee
        self.alpha = params.gain_norm_alpha

    def lowpass_filter_safe(self, signal, fknee=0.01, alpha=4.0, samprate=50, num_threads=2):
        """ Applies a lowpass filter to a signal, and returns the low-frequency result.
            Uses mirrored padding of signal to deal with edge effects.
            Assumes signal is of shape [freq, time].
        """
        Nfreq, N = signal.shape
        signal_padded = np.zeros((Nfreq, 2*N))
        signal_padded[:,:N] = signal
        signal_padded[:,N:] = signal[:,::-1]

        freq_full = np.fft.fftfreq(2*N)*samprate
        W = 1.0/(1 + (freq_full/fknee)**alpha)
        return ifft(fft(signal_padded)*W).real[:,:N]


    def lowpass_filter(self, signal, fknee=0.01, alpha=4.0, samprate=50, num_threads=-1):
        Nfreq, Ntod = signal.shape
        fastlen = next_fast_len(2*Ntod)
        signal_padded = np.zeros((Nfreq, fastlen))

        # signal_padded[:,:Ntod] = signal[:]
        # signal_padded[:,-Ntod:] = signal[:,::-1]
        # signal_padded[:,Ntod:fastlen//2] = signal[:,-(fastlen//2-Ntod):][::-1]
        # signal_padded[:,fastlen//2:fastlen-Ntod] = signal[:,-(fastlen//2-Ntod):]

        signal_padded[:,:Ntod] = signal[:,:]
        signal_padded[:,Ntod:Ntod*2] = signal[:,::-1]
        signal_padded[:,Ntod*2:] = np.nanmean(signal[:,:400], axis=-1)[:,None]

        freq_full = np.fft.rfftfreq(fastlen)*samprate
        W = 1.0/(1 + (freq_full/fknee)**alpha)
        # return ifft(fft(signal_padded)*W).real[:,:Ntod]  # Now crashes with more workers, for some reason ("Resource temporarily unavailable").
        # return irfft(rfft(signal_padded, workers=num_threads)*W, n=fastlen, workers=num_threads).real[:,:Ntod]
        return irfft(rfft(signal_padded)*W, n=fastlen).real[:,:Ntod]


    def lowpass_filter_new(self, signal, fastlen, fknee=0.01, alpha=4.0, samprate=50, num_threads=-1):
        Nfreq, Ntod = signal.shape
        signal_padded = np.zeros((Nfreq, fastlen))
        x = np.linspace(0, 1, Ntod)
        c0 = np.nanmean(signal[:,:1000], axis=-1)
        c1 = np.nanmean(signal[:,-1000:], axis=-1) - c0
        lin_fit = c1[:,None]*x[None,:] + c0[:,None]
        signal_padded[:,:Ntod] = signal - lin_fit
        freq_full = np.fft.rfftfreq(fastlen)*samprate
        W = 1.0/(1 + (freq_full/fknee)**alpha)
        # return irfft(rfft(signal_padded, workers=num_threads)*W, workers=num_threads).real[:,:Ntod] + lin_fit
        return irfft(rfft(signal_padded)*W, n=Ntod).real[:,:Ntod] + lin_fit


    def run(self, l2):
        if not self.use_ctypes:
            # Ntod = l2.tod.shape[-1]
            # fft_times = np.load(os.path.join(CURRENT_DIR, "/fft_times_owl33_10runs.npy")
            # search_start = int(Ntod*1.05)
            # search_stop = int(Ntod*1.10)  # We add at most 10% to the TOD to find fastest length during FFT.
            # if search_stop < fft_times.shape[-1]:  # If search search is within the catalogue, use it. Otherwise, use scipy.
            #     fastlen = np.argmin(fft_times[search_start:search_stop]) + search_start  # Find fastest FFT time within the search length.
            # else:
            #     fastlen = next_fast_len(Ntod)
            # fastlen = next_fast_len(2*Ntod)
            # print(Ntod, fastlen)
            for feed in range(l2.Nfeeds):
                for sb in range(l2.Nsb):
                    # tod_lowpass = lowpass_filter_new(l2.tod[feed, sb], fastlen, num_threads = self.omp_num_threads, fknee=self.fknee, alpha=self.alpha)
                    tod_lowpass = self.lowpass_filter(l2.tod[feed, sb], num_threads = self.omp_num_threads, fknee=self.fknee, alpha=self.alpha)
                    # tod_lowpass = self.lowpass_filter_safe(l2.tod[feed, sb], num_threads = self.omp_num_threads, fknee=self.fknee, alpha=self.alpha)
                    l2.tod[feed,sb] = l2.tod[feed,sb]/tod_lowpass - 1
            del(tod_lowpass)
        else:
            C_LIB_PATH = os.path.join(CURRENT_DIR, "C_libs/norm/normlib.so.1")
            normlib = ctypes.cdll.LoadLibrary(C_LIB_PATH)
            float32_array2 = np.ctypeslib.ndpointer(dtype=ctypes.c_float, ndim=2, flags="contiguous")
            normlib.normalize.argtypes = [float32_array2, ctypes.c_ulong, ctypes.c_ulong]
            Nfull = next_fast_len(2*l2.Ntod)
            for feed in range(l2.Nfeeds):
                for sb in range(l2.Nsb):
                    l2_padded = np.zeros((l2.Nfreqs, Nfull), dtype=np.float32)
                    l2_padded[:,:l2.Ntod] = l2.tod[feed,sb,:,:]
                    l2_padded[:,l2.Ntod:l2.Ntod*2] = l2.tod[feed,sb,:,::-1]
                    l2_padded[:,l2.Ntod*2:] = np.nanmean(l2.tod[feed,sb,:,:400], axis=-1)[:,None]
                    normlib.normalize(l2_padded, l2.Nfreqs, Nfull)
                    l2.tod[feed,sb] = l2_padded[:,:l2.Ntod]        


class Decimation(Filter):
    name = "dec"
    name_long = "decimation"

    def __init__(self, params, omp_num_threads=2):
        self.omp_num_threads = omp_num_threads
        self.Nfreqs_lowres = params.decimation_freqs
        self.params = params

    def run(self, l2):
        # Set up the lowres frequency bins.
        self.freq_bin_centers_lowres = np.zeros((l2.Nsb, self.Nfreqs_lowres))
        self.freq_bin_edges_lowres = np.zeros((l2.Nsb, self.Nfreqs_lowres+1))
        for isb in range(l2.Nsb):
            self.freq_bin_edges_lowres[isb] = np.linspace(26+isb*2, 28+isb*2, self.Nfreqs_lowres+1)
            for ifreq in range(self.Nfreqs_lowres):
                self.freq_bin_centers_lowres[isb,ifreq] = np.mean(self.freq_bin_edges_lowres[isb, ifreq:ifreq+2])

        freq_idxs_included = np.zeros((l2.Nsb, self.Nfreqs_lowres, l2.Nfreqs), dtype=bool)  # An array saying whether each high-res frequency (dimension 3) is a part of a low-res bin (dimension 2).
        for isb in range(l2.Nsb):
            for ifreq in range(self.Nfreqs_lowres):
                # Find the indexes of all high-res frequencies that should go into this frequency bin:
                freq_idxs_included[isb,ifreq] = (l2.freq_bin_centers[isb] > self.freq_bin_edges_lowres[isb,ifreq])*(l2.freq_bin_centers[isb] < self.freq_bin_edges_lowres[isb,ifreq+1])
        N_freqs_in_bin = np.sum(freq_idxs_included, axis=-1)  # An array saying how many high-res frequencies goes into each low-res frequency.

        self.dnu = l2.Nfreqs//self.Nfreqs_lowres
        l2.Nfreqs_highres = l2.Nfreqs
        l2.Nfreqs = self.Nfreqs_lowres
        
        weight = 1.0/np.nanvar(l2.tod, axis=-1)
        
        # If processing signal only TOD sigma0 is substituted with 1s to make coadditions become arithmetic means
        # if "Replace_TOD_With_Signal" in self.params.filters:
        #     print("Overwriting weights in decimation with ones due to signal only TOD being processed!")
        #     weight = np.ones_like(weight)

        # weight[~l2.freqmask] = 0

        tod_decimated = np.zeros((l2.tod.shape[0], l2.tod.shape[1], self.Nfreqs_lowres, l2.tod.shape[3]), dtype=np.float32)

        if l2.is_sim:
            signal_tod_decimated = np.zeros((l2.tod.shape[0], l2.tod.shape[1], self.Nfreqs_lowres, l2.tod.shape[3]), dtype=np.float32)
            
        for isb in range(l2.Nsb):
            for ifreq in range(self.Nfreqs_lowres):
                tod_decimated[:,isb,ifreq,:] = np.nansum(l2.tod[:,isb,freq_idxs_included[isb,ifreq],:]*weight[:,isb,freq_idxs_included[isb,ifreq],None], axis=1)
                tod_decimated[:,isb,ifreq,:] /= np.nansum(weight[:,isb,freq_idxs_included[isb,ifreq]], axis=1)[:,None]

                if l2.is_sim:
                    signal_tod_decimated[:,isb,ifreq,:] = np.nansum(l2.signal_tod[:,isb,freq_idxs_included[isb,ifreq],:]*weight[:,isb,freq_idxs_included[isb,ifreq],None], axis=1)
                    signal_tod_decimated[:,isb,ifreq,:] /= np.nansum(weight[:,isb,freq_idxs_included[isb,ifreq]], axis=1)[:,None]

        l2.tod = tod_decimated

        if l2.is_sim:
            l2.signal_tod = signal_tod_decimated

        l2.freqmask_decimated = np.zeros((l2.Nfeeds, l2.Nsb, self.Nfreqs_lowres), dtype=bool)

        for isb in range(l2.Nsb):
            for ifreq in range(self.Nfreqs_lowres):
                l2.freqmask_decimated[:,isb,ifreq] = l2.freqmask[:,isb,freq_idxs_included[isb,ifreq]].any(axis=-1)
        l2.freqmask_decimated[~(np.isfinite(l2.tod).all(axis=-1))] = False
        tsys_decimated = np.zeros((l2.Nfeeds, l2.Nsb, self.Nfreqs_lowres))
        for isb in range(l2.Nsb):
            for ifreq in range(self.Nfreqs_lowres):
                tsys_decimated[:,isb,ifreq] = np.sqrt(self.dnu/np.nansum(l2.freqmask[:,isb,freq_idxs_included[isb,ifreq]]/l2.Tsys[:,isb,freq_idxs_included[isb,ifreq]]**2, axis=-1))

        tsys_decimated[~l2.freqmask_decimated] = np.nan
        l2.tofile_dict["freqmask"] = l2.freqmask_decimated
        l2.tofile_dict["decimation_nu"] = self.dnu
        l2.tofile_dict["Tsys_lowres"] = tsys_decimated
        l2.tofile_dict["N_freqs_in_bin"] = N_freqs_in_bin
        l2.tofile_dict["freq_bin_edges_lowres"] = self.freq_bin_edges_lowres
        l2.tofile_dict["freq_bin_centers_lowres"] = self.freq_bin_centers_lowres
        

class DecimationOld(Filter):
    name = "dec"
    name_long = "decimation"

    def __init__(self, params, omp_num_threads=2):
        self.omp_num_threads = omp_num_threads
        self.Nfreqs = params.decimation_freqs

    def run(self, l2):
        self.dnu = l2.Nfreqs//self.Nfreqs
        l2.Nfreqs = self.Nfreqs
        weight = 1.0/np.nanvar(l2.tod, axis=-1)
        weight[~l2.freqmask] = 0
        tod_decimated = np.zeros((l2.tod.shape[0], l2.tod.shape[1], self.Nfreqs, l2.tod.shape[3]), dtype=np.float32)
        for freq in range(self.Nfreqs):
            tod_decimated[:,:,freq,:] = np.nansum(l2.tod[:,:,freq*self.dnu:(freq+1)*self.dnu,:]*weight[:,:,freq*self.dnu:(freq+1)*self.dnu,None], axis=2)
            tod_decimated[:,:,freq,:] /= np.nansum(weight[:,:,freq*self.dnu:(freq+1)*self.dnu], axis=2)[:,:,None]
        l2.tod = tod_decimated
        l2.freqmask_decimated = np.zeros((l2.Nfeeds, l2.Nsb, self.Nfreqs), dtype=bool)
        for freq in range(self.Nfreqs):
            l2.freqmask_decimated[:,:,freq] = l2.freqmask[:,:,freq*self.dnu:(freq+1)*self.dnu].any(axis=-1)
        l2.freqmask_decimated[~(np.isfinite(l2.tod).all(axis=-1))] = False
        tsys_decimated = np.zeros((l2.Nfeeds, l2.Nsb, self.Nfreqs))
        for ifreq in range(self.Nfreqs):
            tsys_decimated[:,:,ifreq] = np.sqrt(self.dnu/np.nansum(l2.freqmask[:,:,self.dnu*ifreq:self.dnu*(ifreq+1)]/l2.Tsys[:,:,self.dnu*ifreq:self.dnu*(ifreq+1)]**2, axis=-1))

        tsys_decimated[~l2.freqmask_decimated] = np.nan
        l2.tofile_dict["freqmask"] = l2.freqmask_decimated
        l2.tofile_dict["decimation_nu"] = self.dnu
        l2.tofile_dict["Tsys_lowres"] = tsys_decimated



class Pointing_Template_Subtraction(Filter):
    name = "point"
    name_long = "pointing template subtraction"

    def __init__(self, params, use_ctypes=True, omp_num_threads=2):
        self.use_ctypes = use_ctypes
        self.omp_num_threads = omp_num_threads
    
    def run(self, l2):
        l2.tofile_dict["el_az_amp"] = np.zeros((l2.Nfeeds, l2.Nsb, l2.Nfreqs, 3))
        if not self.use_ctypes:
            def az_func(x, d, c):
                return d*x + c
            def az_el_func(x, g, d, c):
                return g*x[0] + d*x[1] + c
            def az_el_template(feed, g, d, c):
                return g/np.sin(l2.el[feed]*np.pi/180.0) + d*l2.az[feed] + c

            g, d, c = 0, 0, 0
            for feed in range(l2.Nfeeds):
                for sb in range(l2.Nsb):
                    for freq in range(l2.Nfreqs):
                        if l2.scantype == "ces":
                            _az = l2.az[feed][l2.mask_temporal[feed]]
                            _tod = l2.tod[feed, sb, freq][l2.mask_temporal[feed]]
                            if np.isfinite(_tod).all():
                                (d, c), _ = curve_fit(az_func, _az, _tod, (d, c))
                            else:
                                d, c = 0, 0
                        else:
                            if np.isfinite(l2.tod[feed, sb, freq]).all():
                                (g, d, c), _ = curve_fit(az_el_func, (1.0/np.sin(l2.el[feed]*np.pi/180.0), l2.az[feed]), l2.tod[feed, sb, freq], (g, d, c), sigma=l2.mask_temporal[feed])
                            else:
                                g, d, c = 0, 0, 0
                        l2.tod[feed,sb,freq] = l2.tod[feed,sb,freq] - az_el_template(feed, g, d, c)
                        l2.tofile_dict["el_az_amp"][feed,sb,freq,:] = g, d, c

        else:
            C_LIB_PATH = os.path.join(CURRENT_DIR, "C_libs/pointing/pointinglib.so.1")
            
            pointinglib = ctypes.cdll.LoadLibrary(C_LIB_PATH)
            float32_array3 = np.ctypeslib.ndpointer(dtype=ctypes.c_float, ndim=3, flags="contiguous")
            float64_array1 = np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=1, flags="contiguous")
            pointinglib.az_fit.argtypes = [float64_array1, float32_array3, ctypes.c_int, ctypes.c_int, float64_array1, float64_array1]
            pointinglib.az_el_fit.argtypes = [float64_array1, float64_array1, float32_array3, ctypes.c_int, ctypes.c_int, float64_array1, float64_array1, float64_array1]
            if l2.scantype == "ces":
                for ifeed in range(l2.Nfeeds):
                    _az = np.ascontiguousarray(l2.az[ifeed][l2.mask_temporal[ifeed]])
                    _tod = np.ascontiguousarray(l2.tod[ifeed][:,:,l2.mask_temporal[ifeed]])
                    Ntod_effective = _tod.shape[-1]
                    c_est = np.zeros(l2.Nfreqs*l2.Nsb)
                    d_est = np.zeros(l2.Nfreqs*l2.Nsb)
                    pointinglib.az_fit(_az, _tod, l2.Nfreqs*l2.Nsb, Ntod_effective, c_est, d_est)
                    c_est = c_est.reshape((l2.Nsb, l2.Nfreqs))
                    d_est = d_est.reshape((l2.Nsb, l2.Nfreqs))
                    l2.tod[ifeed] -= l2.az[ifeed][None,None,:]*d_est[:,:,None] + c_est[:,:,None]
                    l2.tofile_dict["el_az_amp"][ifeed,:,:,0] = np.zeros((l2.Nsb, l2.Nfreqs))
                    l2.tofile_dict["el_az_amp"][ifeed,:,:,1] = d_est
                    l2.tofile_dict["el_az_amp"][ifeed,:,:,2] = c_est
            else:
                for ifeed in range(l2.Nfeeds):
                    c_est = np.zeros(l2.Nfreqs*l2.Nsb)
                    d_est = np.zeros(l2.Nfreqs*l2.Nsb)
                    g_est = np.zeros(l2.Nfreqs*l2.Nsb)
                    el_term = 1.0/np.sin(l2.el[ifeed]*np.pi/180.0)
                    pointinglib.az_el_fit(l2.az[ifeed], el_term, l2.tod[ifeed], l2.Nfreqs*l2.Nsb, l2.Ntod, c_est, d_est, g_est)
                    c_est = c_est.reshape((l2.Nsb, l2.Nfreqs))
                    d_est = d_est.reshape((l2.Nsb, l2.Nfreqs))
                    g_est = g_est.reshape((l2.Nsb, l2.Nfreqs))
                    l2.tod[ifeed] -= el_term[None,None,:]*g_est[:,:,None] + l2.az[ifeed][None,None,:]*d_est[:,:,None] + c_est[:,:,None]
                    l2.tofile_dict["el_az_amp"][ifeed,:,:,0] = g_est
                    l2.tofile_dict["el_az_amp"][ifeed,:,:,1] = d_est
                    l2.tofile_dict["el_az_amp"][ifeed,:,:,2] = c_est

class Polynomial_filter(Filter):
    name = "poly"
    name_long = "polynomial filter"
    run_when_masking = True
    has_corr_template = True

    # def get_corr_template(l2, C, feed, ihalf):
    #     C_template = np.zeros_like(C)
    #     for k in range(2):
    #         isb = ihalf*2 + k
    #         b = np.zeros((l2.Nfreqs, 2))
    #         b[:,0] = np.ones(l2.Nfreqs)
    #         b[:,1] = np.linspace(-1, 1, l2.Nfreqs)
    #         b[~l2.freqmask[feed,isb],:] = 0
    #         # b[:,0] /= np.linalg.norm(b[:,0])
    #         # b[:,1] /= np.linalg.norm(b[:,1])
    #         if (b != 0).any():
    #             _C_template = -b.dot(np.linalg.inv(b.T.dot(b)).dot(b.T))
    #             _C_template[:,~l2.freqmask[feed,isb]] = 0.0
    #             _C_template[~l2.freqmask[feed,isb],:] = 0.0
    #             for i in range(l2.Nfreqs):
    #                 _C_template[i,i] = 0
    #                 if i+1 < l2.Nfreqs:
    #                     _C_template[i+1,i] = 0
    #                     _C_template[i,i+1] = 0
    #             C_template[k*l2.Nfreqs:(k+1)*l2.Nfreqs,k*l2.Nfreqs:(k+1)*l2.Nfreqs] = _C_template
    #     return C_template


    def __init__(self, params, use_ctypes=True, omp_num_threads=2):
        self.use_ctypes = use_ctypes
        self.omp_num_threads = omp_num_threads
        self.params = params

    def run(self, l2, masking_run=False):
        if masking_run and l2.params.write_C_matrix:
            l2.tofile_dict["C_template_polyfilter"] = np.zeros((l2.Nfeeds, 2, 2*l2.Nfreqs, 2*l2.Nfreqs))
        c0 = np.zeros((l2.Nfeeds, l2.Nsb, l2.Ntod)) + np.nan
        c1 = np.zeros((l2.Nfeeds, l2.Nsb, l2.Ntod)) + np.nan
        if not self.use_ctypes:
            sb_freqs = np.linspace(-1, 1, l2.Nfreqs)
            for ifeed in range(l2.Nfeeds):
                for sb in range(4):
                    for idx in range(l2.Ntod):
                        if l2.mask_temporal[ifeed,idx]:
                            # if np.isfinite(l2.tod[feed,sb,:,idx]).all():
                            tod_local = l2.tod[ifeed,sb,:,idx].copy()
                            tod_local[~np.isfinite(tod_local)] = 0  # polyfit doesn't allow nans.
                            try:
                                c1[ifeed,sb,idx], c0[ifeed,sb,idx] = np.polyfit(sb_freqs, tod_local, 1, w=l2.freqmask[ifeed,sb])
                                l2.tod[ifeed, sb, :, idx] = l2.tod[ifeed, sb, :, idx] - c1[ifeed,sb,idx]*sb_freqs - c0[ifeed,sb,idx]
                            except:
                                pass

        else:
            C_LIB_PATH = os.path.join(CURRENT_DIR, "C_libs/polyfit/polyfit.so.1")
            polylib = ctypes.cdll.LoadLibrary(C_LIB_PATH)
            float32_array2 = np.ctypeslib.ndpointer(dtype=ctypes.c_float, ndim=2, flags="contiguous")
            float32_array1 = np.ctypeslib.ndpointer(dtype=ctypes.c_float, ndim=1, flags="contiguous")
            float64_array1 = np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=1, flags="contiguous")
            polylib.polyfit.argtypes = [float32_array1, float32_array2, float64_array1, ctypes.c_int, ctypes.c_int, float64_array1, float64_array1]

            sb_freqs = np.linspace(-1, 1, l2.Nfreqs, dtype=np.float32)
            for ifeed in range(l2.Nfeeds):
                for sb in range(4):
                    tod_local = l2.tod[ifeed, sb].copy()
                    tod_local[~np.isfinite(tod_local)] = 0
                    weights = np.array(l2.freqmask[ifeed, sb], dtype=float)
                    polylib.polyfit(sb_freqs, l2.tod[ifeed, sb], weights, l2.Nfreqs, l2.Ntod, c0[ifeed,sb], c1[ifeed,sb])
                    l2.tod[ifeed, sb, :, :] = l2.tod[ifeed, sb, :, :] - c1[ifeed,sb,None,:]*sb_freqs[:,None] - c0[ifeed,sb,None,:]  # Future improvement: Move into C.
        l2.tofile_dict["poly_coeff"] = np.zeros((2, l2.Nfeeds, l2.Nsb, l2.Ntod))
        l2.tofile_dict["poly_coeff"][:] = c0, c1

        # Adjust correlation template.
        if masking_run:
            for ifeed in range(l2.Nfeeds):
                for isb in range(l2.Nsb):
                    b = np.zeros((l2.Nfreqs, 2))
                    b[:,0] = np.ones(l2.Nfreqs)
                    b[:,1] = np.linspace(-1, 1, l2.Nfreqs)
                    b[~l2.freqmask[ifeed,isb],:] = 0
                    if (b != 0).any():
                        template = -b.dot(np.linalg.inv(b.T.dot(b)).dot(b.T))
                        template[:,~l2.freqmask[ifeed,isb]] = 0.0
                        template[~l2.freqmask[ifeed,isb],:] = 0.0
                        for i in range(l2.Nfreqs):
                            template[i,i] = 0
                            if i+1 < l2.Nfreqs:
                                template[i+1,i] = 0
                                template[i,i+1] = 0
                        l2.corr_template[ifeed,l2.Nfreqs*isb:l2.Nfreqs*(isb+1),l2.Nfreqs*isb:l2.Nfreqs*(isb+1)] += template
                        if self.params.write_C_matrix:
                            l2.tofile_dict["C_template_polyfilter"][ifeed,isb//2, isb%2*l2.Nfreqs:(isb%2+1)*l2.Nfreqs,isb%2*l2.Nfreqs:(isb%2+1)*l2.Nfreqs] = template
            # print(np.sum(np.abs(l2.corr_template)))


class Frequency_filter(Filter):
    name = "freq"
    name_long = "frequency filter"
    run_when_masking = True
    has_corr_template = True

    # def get_corr_template(l2, C, feed, ihalf):
    #     C_template = np.zeros_like(C)
    #     for k in range(2):
    #         isb = ihalf*2 + k
    #         P = l2.tofile_dict["freqfilter_P_reduced"][feed,isb]
    #         F = l2.tofile_dict["freqfilter_F_reduced"][feed,isb]
    #         if not l2.params.freqfilter_use_prior:
    #             b = np.zeros((l2.Nfreqs, 3))
    #             b[:,:2] = P[:]
    #             b[:,2] = F[:,0]
    #             b[~np.isfinite(b)] = 0
    #             if (b != 0).any():
    #                 _C_template = -b.dot(np.linalg.inv(b.T.dot(b)).dot(b.T))
    #                 _C_template[:,~l2.freqmask[feed,isb]] = 0.0
    #                 _C_template[~l2.freqmask[feed,isb],:] = 0.0
    #             else:
    #                 _C_template = np.zeros((l2.Nfreqs, l2.Nfreqs))
    #         else:
    #             b = np.zeros((l2.Nfreqs, 3))
    #             b[:,:2] = P[:]
    #             b[:,2] = F[:,0]
    #             b[~np.isfinite(b)] = 0
    #             if (b != 0).any():
    #                 Lfull = -b.dot(np.linalg.inv(b.T.dot(b))).dot(b.T)
    #             else:
    #                 return 0
    #             b = P.copy()
    #             b[~np.isfinite(b)] = 0
    #             if (b != 0).any():
    #                 Lreduced = -b.dot(np.linalg.inv(b.T.dot(b))).dot(b.T)
    #             else:
    #                 return 0
    #             for ifreq in range(l2.Nfreqs):
    #                 Lfull[ifreq,ifreq] = 0
    #                 Lreduced[ifreq,ifreq] = 0
    #                 if ifreq < l2.Nfreqs-1:
    #                     Lfull[ifreq+1,ifreq] = 0
    #                     Lreduced[ifreq+1,ifreq] = 0
    #                     Lfull[ifreq,ifreq+1] = 0
    #                     Lreduced[ifreq,ifreq+1] = 0
    #             c = C[k*l2.Nfreqs:(k+1)*l2.Nfreqs,k*l2.Nfreqs:(k+1)*l2.Nfreqs].copy()
    #             c[~np.isfinite(c)] = 0.0
    #             c = c.flatten()
    #             lx = Lfull.flatten()
    #             ly = Lreduced.flatten()
    #             alpha = (c.T.dot(lx) - c.T.dot(ly) - lx.T.dot(ly) + ly.T.dot(ly))/(lx.T.dot(lx) - 2*lx.T.dot(ly) + ly.T.dot(ly))
            
    #             # print("c", np.sum(~np.isfinite(c)))
    #             # print("b", np.sum(~np.isfinite(b)))
    #             # print("lx", np.sum(~np.isfinite(lx)))
    #             # print("ly", np.sum(~np.isfinite(ly)))
    #             # print("alpha", alpha)

    #             if np.isfinite(alpha):
    #                 if alpha < 0:
    #                     alpha_safe = 0
    #                 elif alpha > 1:
    #                     alpha_safe = 1
    #                 else:
    #                     alpha_safe = alpha
    #                 _C_template = alpha_safe*Lfull + (1-alpha_safe)*Lreduced
    #                 _C_template[:,~l2.freqmask[feed,isb]] = 0.0
    #                 _C_template[~l2.freqmask[feed,isb],:] = 0.0
    #             else:
    #                 _C_template = np.zeros((l2.Nfreqs, l2.Nfreqs))
    #             l2.tofile_dict["freqfilter_prior_alpha"][feed, isb] = alpha

    #         C_template[k*l2.Nfreqs:(k+1)*l2.Nfreqs,k*l2.Nfreqs:(k+1)*l2.Nfreqs] = _C_template

    #     # print("C", np.sum(~np.isfinite(C_template)))
    #     # print(c.T.dot(lx), c.T.dot(ly), lx.T.dot(ly), ly.T.dot(ly))
    #     # print(lx.T.dot(lx), lx.T.dot(ly), ly.T.dot(ly))
    #     if not np.isfinite(C_template).all():
    #         raise ValueError("Non-finite (probably NaN) value in frequency filter corr-matrix template.")

    #     return C_template

    def __init__(self, params, omp_num_threads=2):
        self.omp_num_threads = omp_num_threads
        self.prior_file = params.freqfilter_prior_file
        self.use_prior = params.freqfilter_use_prior
        self.params = params


    def PS_1f(self, freqs, sigma0, fknee, alpha, wn=True, Wiener=False, fknee_W=0.01, alpha_W=-4.0):
        PS = np.zeros_like(freqs)
        if wn:
            PS[1:] = sigma0**2*(1 + (freqs[1:]/fknee)**alpha)
        else:
            PS[1:] = sigma0**2*((freqs[1:]/fknee)**alpha)
        if Wiener:
            PS[1:] /= 1 + (freqs[1:]/fknee_W)**alpha_W
        return PS

    
    def binned_PS(self, data, samprate, Nbins=100):
        # data - 4D TOD vector with time along the last axis.
        # Returns the log-binned frequencies and power spectrum of the data along the last axis.
        freqs = rfftfreq(data.shape[-1], 1.0/samprate)[1:]
        log_freqs = np.log10(freqs)
        PS = rfft(data).real[:,:,:,1:]**2
        freq_bins = np.linspace(np.log10(freqs[0])-1e-6, np.log10(freqs[-1])+1e-6, Nbins)
        indices = np.digitize(log_freqs, freq_bins)
        PS_binned = np.zeros((PS.shape[0], PS.shape[1], PS.shape[2], Nbins))
        PS_binned_freqs = np.zeros((Nbins))
        for i in range(Nbins):
            bin_indices = np.argwhere(indices==i)
            if len(bin_indices) > 0:
                bin_indices = bin_indices.T[0]
                PS_binned_freqs[i] = np.mean(freqs[bin_indices])
                PS_binned[:,:,:,i] = np.mean(PS[:,:,:,bin_indices],axis=-1)
        PS_binned = PS_binned[:,:,:,PS_binned_freqs != 0]
        PS_binned_freqs = PS_binned_freqs[PS_binned_freqs != 0]
        return PS_binned_freqs, PS_binned


    def gain_temp_sep(self, y, P, F, samprate, sigma0_g=None, fknee_g=None, alpha_g=None):
        Nfreqs, Ntod = y.shape
        if sigma0_g and fknee_g and alpha_g:  # Set any of the prior parameters to None or False to skip prior.
            freqs = np.fft.rfftfreq(Ntod, 1/samprate)
            freqs[0] = freqs[1]/2
            Cf = self.PS_1f(freqs, sigma0_g, fknee_g, alpha_g, wn=False, Wiener=True)
            sigma0_est = np.std(y[:,1:] - y[:,:-1], axis=1)/np.sqrt(2)
            sigma0_est = np.mean(sigma0_est[sigma0_est != 0])
            Z = np.eye(Nfreqs, Nfreqs) - P.dot(np.linalg.pinv(P.T.dot(P))).dot(P.T)
            RHS = np.fft.rfft(F.T.dot(Z).dot(y))
            z = F.T.dot(Z).dot(F)[0]
            a_bestfit_f = RHS/(z + sigma0_est**2/Cf)
            a_bestfit = np.fft.irfft(a_bestfit_f, n=Ntod)
            del(freqs, RHS, a_bestfit_f)
            m_bestfit = np.linalg.pinv(P.T.dot(P)).dot(P.T).dot(y - F*a_bestfit)
            return a_bestfit, m_bestfit, Cf, z, Z
        else:
            Z = np.eye(Nfreqs, Nfreqs) - P.dot(np.linalg.pinv(P.T.dot(P))).dot(P.T)
            RHS = F.T.dot(Z).dot(y)
            z = F.T.dot(Z).dot(F)
            a_bestfit = RHS/z
            m_bestfit = np.linalg.pinv(P.T.dot(P)).dot(P.T).dot(y - F*a_bestfit)
            return a_bestfit, m_bestfit


    def run(self, l2, masking_run=False):
        with h5py.File(self.prior_file, "r") as f:
            sigma0_prior = f["sigma0_prior"][l2.feeds-1]
            fknee_prior = f["fknee_prior"][l2.feeds-1]
            alpha_prior = f["alpha_prior"][l2.feeds-1]
        
        if self.params.freqfilter_full_feed:  # Testing of all-sb instead of per-sb for gain.
            l2.tofile_dict["freqfilter_P"] = np.zeros((l2.Nfeeds, l2.Nsb*l2.Nfreqs, 2))
            l2.tofile_dict["freqfilter_F"] = np.zeros((l2.Nfeeds, l2.Nsb*l2.Nfreqs, 1))
            l2.tofile_dict["freqfilter_m"] = np.zeros((l2.Nfeeds, 2, l2.Ntod))
            l2.tofile_dict["freqfilter_a"] = np.zeros((l2.Nfeeds, 1, l2.Ntod))
            # sb_freqs = np.linspace(-1, 1, l2.Nfreqs*l2.Nsb)
            # sb_freqs = sb_freqs.reshape((l2.Nsb, l2.Nfreqs))
            # sb_freqs[(0, 2), :] = sb_freqs[(0, 2), ::-1]
            sb_freqs = l2.freqs.copy()
            for feed in range(l2.Nfeeds):
                P = np.zeros((4, l2.Nfreqs, 2))
                F = np.zeros((4, l2.Nfreqs, 1))
                y = l2.tod[feed].copy()
                P[:,:,0] = 1/l2.Tsys[feed]
                P[:,:,1] = sb_freqs/l2.Tsys[feed]
                F[:,:,0] = 1
                y[l2.freqmask[feed] == 0] = 0
                P[l2.freqmask[feed] == 0, :] = 0
                F[l2.freqmask[feed] == 0, :] = 0

                end_cut = 100
                P_reduced = P.copy()
                F_reduced = F.copy()
                if self.params.freqfilter_exclude_ends:
                    y[:,:4] = 0
                    y[:,-end_cut:] = 0
                    P_reduced[:,:4] = 0
                    P_reduced[:,-end_cut:] = 0
                    F_reduced[:,:4] = 0
                    F_reduced[:,-end_cut:] = 0
                P_reduced[~np.isfinite(P_reduced)] = 0
                F_reduced[~np.isfinite(F_reduced)] = 0
                
                P = P.reshape((l2.Nsb*l2.Nfreqs, 2))
                F = F.reshape((l2.Nsb*l2.Nfreqs, 1))
                P_reduced = P_reduced.reshape((l2.Nsb*l2.Nfreqs, 2))
                F_reduced = F_reduced.reshape((l2.Nsb*l2.Nfreqs, 1))
                y = y.reshape((l2.Nsb*l2.Nfreqs, y.shape[-1]))

                if np.sum(P != 0) > 0 and np.sum(F != 0) > 0:
                    if self.use_prior:
                        a, m, Cf, z, Z = self.gain_temp_sep(y, P_reduced, F_reduced, l2.samprate, sigma0_prior[feed], fknee_prior[feed], alpha_prior[feed])
                    else:
                        a, m = self.gain_temp_sep(y, P_reduced, F_reduced, l2.samprate)
                else:
                    a = np.zeros((1, l2.Ntod))
                    m = np.zeros((2, l2.Ntod))

                for sb in range(l2.Nsb):
                    l2.tod[feed,sb] = l2.tod[feed,sb] - F[l2.Nfreqs*sb:l2.Nfreqs*(sb+1)].dot(a) - P[l2.Nfreqs*sb:l2.Nfreqs*(sb+1)].dot(m)

                # b = P_reduced.copy()
                # b[~np.isfinite(b)] = 0
                # l2.corr_template[feed, :, :] += -b.dot(np.linalg.inv(b.T.dot(b)).dot(b.T))

                l2.tofile_dict["freqfilter_P"][feed] = P
                l2.tofile_dict["freqfilter_F"][feed] = F
                l2.tofile_dict["freqfilter_m"][feed] = m
                l2.tofile_dict["freqfilter_a"][feed] = a

                if masking_run:
                    def var_est(fourier_func):
                        div_factor = (l2.Ntod-1) / 2.0 
                        if (l2.Ntod % 2 == 0):
                            var_est = (np.sum(fourier_func[1:-1]) + fourier_func[-1] / 2) / div_factor
                        else:
                            var_est = np.sum(fourier_func[1:]) / div_factor
                        return var_est

                    F1 = F_reduced - F
                    F0 = F_reduced                        
                    Z = np.eye(l2.Nfreqs*4, l2.Nfreqs*4) - P.dot(np.linalg.pinv(P_reduced.T.dot(P_reduced))).dot(P_reduced.T)
                    Z0 = np.eye(l2.Nfreqs*4, l2.Nfreqs*4) - P_reduced.dot(np.linalg.pinv(P_reduced.T.dot(P_reduced))).dot(P_reduced.T)
                    z0 = F_reduced.T.dot(Z0).dot(F_reduced)[0]

                    if self.use_prior:
                        sigma_1 = np.nanmean(np.std(l2.tod[feed,sb,:,1:] - l2.tod[feed,sb,:,:-1], axis=-1))/np.sqrt(2)
                        # sigma_2 = 0.005059644  # TODO: FIXX!!!
                        sigma = sigma_1
                        fourier_func = 1.0 /(z0 + sigma**2/Cf)
                        # fourier_func2 = (1.0 /(z + sigma**2/Cf)) ** 2

                        G = var_est(fourier_func)
                        # G2 = var_est(fourier_func2)
                    else:
                        G = 1.0/z0


                    U = G*F0.T.dot(Z0)
                    L = (Z - Z.dot(F0).dot(U) + F1.dot(U))
                    template = L.dot(L.T)
                    for i in range(l2.Nfreqs*4):
                        template[i,i] = 0
                        if i < l2.Nfreqs*4-1:
                            template[i+1,i] = 0
                            template[i,i+1] = 0
                    
                    l2.corr_template[feed,:,:] += template




        else:
                
            l2.tofile_dict["freqfilter_P"] = np.zeros((l2.Nfeeds, l2.Nsb, l2.Nfreqs, 2))
            l2.tofile_dict["freqfilter_F"] = np.zeros((l2.Nfeeds, l2.Nsb, l2.Nfreqs, 1))
            l2.tofile_dict["freqfilter_P_reduced"] = np.zeros((l2.Nfeeds, l2.Nsb, l2.Nfreqs, 2))
            l2.tofile_dict["freqfilter_F_reduced"] = np.zeros((l2.Nfeeds, l2.Nsb, l2.Nfreqs, 1))
            l2.tofile_dict["freqfilter_m"] = np.zeros((l2.Nfeeds, l2.Nsb, 2, l2.Ntod))
            l2.tofile_dict["freqfilter_a"] = np.zeros((l2.Nfeeds, l2.Nsb, 1, l2.Ntod))
            l2.tofile_dict["freqfilter_a_m_corr"] = np.zeros((l2.Nfeeds, l2.Nsb, 2))
            # if self.use_prior:
            #     l2.tofile_dict["freqfilter_prior_alpha"] = np.zeros((l2.Nfeeds, l2.Nsb))
            if masking_run and self.params.write_C_matrix:
                l2.tofile_dict["C_template_freqfilter"] = np.zeros((l2.Nfeeds, 2, 2*l2.Nfreqs, 2*l2.Nfreqs))

            self.tod_frequency_filtered = np.zeros_like(l2.tod)
            sb_freqs = np.linspace(-1, 1, l2.Nfreqs)
            P = np.zeros((l2.Nfreqs, 2))
            F = np.zeros((l2.Nfreqs, 1))
            for feed in range(l2.Nfeeds):
                for sb in range(l2.Nsb):
                    y = l2.tod[feed,sb].copy()
                    P[:,0] = 1/l2.Tsys[feed,sb]
                    P[:,1] = sb_freqs/l2.Tsys[feed,sb]
                    F[:,0] = 1
                    y[l2.freqmask[feed,sb] == 0] = 0
                    P[l2.freqmask[feed,sb] == 0, :] = 0
                    F[l2.freqmask[feed,sb] == 0, :] = 0
                    P[~np.isfinite(P)] = 0
                    F[~np.isfinite(F)] = 0

                    # Some cuts Håvard included:
                    P_reduced = P.copy()
                    F_reduced = F.copy()
                    end_cut = 100
                    y[:4] = 0
                    y[-end_cut:] = 0
                    P_reduced[:4] = 0
                    P_reduced[-end_cut:] = 0
                    F_reduced[:4] = 0
                    F_reduced[-end_cut:] = 0


                    if np.sum(P_reduced != 0) > 0 and np.sum(F_reduced != 0) > 0:
                        if self.use_prior:
                            a, m, Cf, z, Z = self.gain_temp_sep(y, P_reduced, F_reduced, sigma0_prior[feed], fknee_prior[feed], alpha_prior[feed])
                        else:
                            a, m = self.gain_temp_sep(y, P_reduced, F_reduced)
                    else:
                        a = np.zeros((1, l2.Ntod))
                        m = np.zeros((2, l2.Ntod))

                    l2.tod[feed,sb] = l2.tod[feed,sb] - F.dot(a) - P.dot(m)


                    # Corr template calculations:
                    if masking_run:
                        # if not self.use_prior:
                        #     if not l2.params.freqfilter_use_prior:
                        #         b = np.zeros((l2.Nfreqs, 3))
                        #         b[:,:2] = P_reduced[:]
                        #         b[:,2] = F_reduced[:,0]
                        #         b[~np.isfinite(b)] = 0
                        #         if (b != 0).any():
                        #             template = -b.dot(np.linalg.pinv(b.T.dot(b)).dot(b.T))
                        #             template[:,~l2.freqmask[feed,sb]] = 0.0
                        #             template[~l2.freqmask[feed,sb],:] = 0.0
                        #         else:
                        #             template = np.zeros((l2.Nfreqs, l2.Nfreqs))
                        #         l2.corr_template[feed,sb*l2.Nfreqs:(sb+1)*l2.Nfreqs,sb*l2.Nfreqs:(sb+1)*l2.Nfreqs] += template
                        #         if self.params.write_C_matrix:
                        #             l2.tofile_dict["C_template_freqfilter"][feed, sb//2, sb%2*l2.Nfreqs:(sb%2+1)*l2.Nfreqs,sb%2*l2.Nfreqs:(sb%2+1)*l2.Nfreqs] += template
                                
                        # else:
                        if (a != 0).any() or (m != 0).any():
                            def var_est(fourier_func):
                                div_factor = (l2.Ntod-1) / 2.0 
                                if (l2.Ntod % 2 == 0):
                                    var_est = (np.sum(fourier_func[1:-1]) + fourier_func[-1] / 2) / div_factor
                                else:
                                    var_est = np.sum(fourier_func[1:]) / div_factor
                                return var_est

                            F1 = F_reduced - F
                            F0 = F_reduced                        
                            Z = np.eye(l2.Nfreqs, l2.Nfreqs) - P.dot(np.linalg.pinv(P_reduced.T.dot(P_reduced))).dot(P_reduced.T)
                            Z0 = np.eye(l2.Nfreqs, l2.Nfreqs) - P_reduced.dot(np.linalg.pinv(P_reduced.T.dot(P_reduced))).dot(P_reduced.T)
                            z0 = F_reduced.T.dot(Z0).dot(F_reduced)[0]

                            if self.use_prior:
                                sigma_1 = np.nanmean(np.std(l2.tod[feed,sb,:,1:] - l2.tod[feed,sb,:,:-1], axis=-1))/np.sqrt(2)
                                # sigma_2 = 0.005059644  # TODO: FIXX!!!
                                sigma = sigma_1
                                fourier_func = 1.0 /(z0 + sigma**2/Cf)
                                # fourier_func2 = (1.0 /(z + sigma**2/Cf)) ** 2

                                G = var_est(fourier_func)
                                # G2 = var_est(fourier_func2)
                            else:
                                G = 1.0/z0


                            U = G*F0.T.dot(Z0)
                            L = (Z - Z.dot(F0).dot(U) + F1.dot(U))
                            template = L.dot(L.T)
                            for i in range(l2.Nfreqs):
                                template[i,i] = 0
                                if i < l2.Nfreqs-1:
                                    template[i+1,i] = 0
                                    template[i,i+1] = 0
                            
                            # template = Z - 2 * G * Z.dot(F_reduced.dot(F_reduced.T.dot(Z))) + G2 * z * Z.dot(F_reduced.dot(F_reduced.T.dot(Z)))
                            # template *= sigma ** 2
                            # variances = np.diag(template)
                            # cov = np.cov(l2.tod[feed,sb])
                            # template /= np.sqrt(np.outer(variances, variances))
                            l2.corr_template[feed,sb*l2.Nfreqs:(sb+1)*l2.Nfreqs,sb*l2.Nfreqs:(sb+1)*l2.Nfreqs] += template
                            if self.params.write_C_matrix:
                                l2.tofile_dict["C_template_freqfilter"][feed, sb//2, sb%2*l2.Nfreqs:(sb%2+1)*l2.Nfreqs,sb%2*l2.Nfreqs:(sb%2+1)*l2.Nfreqs] += template
                            

                    # l2.tofile_dict["freqfilter_prior_alpha"][feed, sb] = alpha
                    l2.tofile_dict["freqfilter_P"][feed, sb] = P
                    l2.tofile_dict["freqfilter_F"][feed, sb] = F
                    l2.tofile_dict["freqfilter_P_reduced"][feed, sb] = P_reduced
                    l2.tofile_dict["freqfilter_F_reduced"][feed, sb] = F_reduced
                    l2.tofile_dict["freqfilter_m"][feed, sb] = m
                    l2.tofile_dict["freqfilter_a"][feed, sb] = a
                    l2.tofile_dict["freqfilter_a_m_corr"][feed, sb] = np.sum((a - np.mean(a, axis=-1)[:,None])*(m - np.mean(m, axis=-1)[:,None]), axis=-1)/(m.shape[-1]*np.std(a, axis=-1)*np.std(m, axis=-1))
            PS_freqs, PS_m = self.binned_PS(l2.tofile_dict["freqfilter_m"], l2.samprate)
            PS_freqs, PS_a = self.binned_PS(l2.tofile_dict["freqfilter_a"], l2.samprate)
            l2.tofile_dict["freqfilter_PS_freqs"] = PS_freqs
            l2.tofile_dict["freqfilter_PS_m"] = PS_m
            l2.tofile_dict["freqfilter_PS_a"] = PS_a




class PCA_filter(Filter):
    name = "pca"
    name_long = "PCA filter"
    run_when_masking = True
    has_corr_template = False

    def __init__(self, params, omp_num_threads=2, use_ctypes=False):
        self.omp_num_threads = omp_num_threads
        self.max_pca_comp = params.max_pca_comp
        self.use_ctypes = use_ctypes
        self.params = params
    
    def run(self, l2, masking_run=False):
        if len(self.params.load_PCA_path) > 0:  # This parameter option indicates that we should not calculate PCA components, but gather them from the scans in this folder.
            path = os.path.join(self.params.load_PCA_path, l2.fieldname)
            filename = os.path.join(path, l2.l2_filename + ".h5")
            if not os.path.isfile(filename):
                raise ValueError(f"Specified l2 file for loading PCA components does not exist: {filename}.")
            l2.printer.debug_print(f"Using imported PCA components from {filename}.")
            with h5py.File(filename, "r") as f:
                comps = f["pca_comp"][()]
                n_pca_comp = f["n_pca_comp"][()]
                pca_eigvals = f["pca_eigvals"][()]

            pca_ampl = np.zeros((n_pca_comp, l2.Nfeeds, l2.Nsb, l2.Nfreqs))
            for ifeed in range(l2.Nfeeds):
                ak = np.nansum(l2.tod[ifeed,:,:,:,None]*np.transpose(comps, (1,0)), axis=2)
                l2.tod[ifeed] = l2.tod[ifeed] - np.nansum(ak[:,:,None,:]*np.transpose(comps, (1,0))[None,None,:,:], axis=-1)
                pca_ampl[:,ifeed] = np.transpose(ak, (2,0,1))
            
            l2.tofile_dict["pca_ampl"] = pca_ampl
            l2.tofile_dict["pca_comp"] = comps
            l2.tofile_dict["n_pca_comp"] = n_pca_comp
            l2.tofile_dict["pca_eigvals"] = pca_eigvals
            return


        if not self.use_ctypes:
            pca_ampl = np.zeros((self.max_pca_comp, l2.Nfeeds, l2.Nsb, l2.Nfreqs))
            N = l2.Nfeeds*l2.Nsb*l2.Nfreqs
            P = l2.Ntod
            M = l2.tod.reshape(N, P)
            M = M[l2.freqmask.flatten(), :]
            M[~np.isfinite(M)] = 0
            M = M[np.sum(M != 0, axis=-1) != 0]
            N_effective = M.shape[0]
            if N_effective > 20:
                pca = PCA(n_components=self.max_pca_comp, random_state=49)
                comps = pca.fit_transform(M.T)
                singular_values = pca.singular_values_
                del(pca)
            else:
                comps = np.zeros((P, self.max_pca_comp))
                singular_values = np.zeros((self.max_pca_comp))
            for i in range(self.max_pca_comp):
                comps[:,i] /= np.linalg.norm(comps[:,i])

            N_effective = np.sum(M.shape[0])
            sigma0 = np.nanmean(np.std(M[:,1:] - M[:,:-1], axis=-1)/np.sqrt(2))
            # Don't ask:
            empirical_correction_factor_for_singular_values_of_large_gaussian_noise_matrices = 1.00476 - 3.957e-03*np.log10(P)*np.log10(N_effective) + 8.757e-05*(np.log10(P)*np.log10(N_effective))**2
            lambda_threshold = self.params.pca_lambda_threshold*empirical_correction_factor_for_singular_values_of_large_gaussian_noise_matrices*(np.sqrt(N_effective) + np.sqrt(P))*sigma0
            self.n_pca_comp = np.sum(singular_values > lambda_threshold)
            if self.params.verbose:
                print(f"PCA scan {l2.scanid} num of modes: {self.n_pca_comp}")
            self.n_pca_comp = max(self.n_pca_comp, self.params.min_pca_comp)
            comps = comps
            del(M)

            for ifeed in range(l2.Nfeeds):
                ak = np.nansum(l2.tod[ifeed,:,:,:,None]*comps[:,:self.n_pca_comp], axis=2)
                l2.tod[ifeed] = l2.tod[ifeed] - np.nansum(ak[:,:,None,:]*comps[None,None,:,:self.n_pca_comp], axis=-1)
                pca_ampl[:self.n_pca_comp,ifeed] = np.transpose(ak, (2,0,1))
            l2.tofile_dict["pca_ampl"] = pca_ampl#[::-1]  # Scipy gives smallest eigenvalues first, we want largest first.
            l2.tofile_dict["pca_comp"] = np.transpose(comps, (1,0))#[::-1]
            l2.tofile_dict["n_pca_comp"] = self.n_pca_comp
            l2.tofile_dict["pca_eigvals"] = singular_values**2
            l2.tofile_dict["pca_lambda_threshold"] = lambda_threshold
            # pca_ampl = np.zeros((self.max_pca_comp, l2.Nfeeds, l2.Nsb, l2.Nfreqs))
            # pca_comp = np.zeros((self.max_pca_comp, l2.Ntod))
            # pca_eigval = np.zeros((self.max_pca_comp))
            # n = l2.Nfeeds*l2.Nsb*l2.Nfreqs
            # p = l2.Ntod
            # C_LIB_PATH = os.path.join(CURRENT_DIR, "C_libs/PCA/PCAlib.so.1")
            # PCAlib = ctypes.cdll.LoadLibrary(C_LIB_PATH)
            # float32_array2 = np.ctypeslib.ndpointer(dtype=ctypes.c_float, ndim=2, flags="contiguous")
            # float64_array1 = np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=1, flags="contiguous")
            # float32_array4 = np.ctypeslib.ndpointer(dtype=ctypes.c_float, ndim=4, flags="contiguous")
            # float64_array3 = np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=3, flags="contiguous")
            # bool_array1 = np.ctypeslib.ndpointer(dtype=ctypes.c_bool, ndim=1, flags="contiguous")
            # PCAlib.PCA.argtypes = [float32_array2, bool_array1, float64_array1, float64_array1, float64_array1, ctypes.c_long, ctypes.c_long, ctypes.c_long, ctypes.c_double]
            # PCAlib.subtract_eigcomp.argtypes = [float32_array4, float64_array1, float64_array3, ctypes.c_long, ctypes.c_long]
            # max_iter = self.params.pca_max_iter
            # err_tol = self.params.pca_error_tol
            # l2.tofile_dict["PCA_err"] = np.zeros((self.max_pca_comp, max_iter))
            # t0 = time.time()
            # # sigma0 = np.nanstd(l2.tod[:,:,:,1:] - l2.tod[:,:,:,:-1], axis=-1)/np.sqrt(2)
            # # sigma0 = np.nanmean(sigma0, where=l2.freqmask)
            # sigma0 = 1.0/np.sqrt(1.0/50*(2e9/1024))

            # n_actual = np.sum(l2.freqmask.flatten())
            # lambda_threshold = 1.0*(np.sqrt(n_actual) + np.sqrt(p))**2*sigma0**2
            # # print(time.time()-t0)
            # # print("PCA lambda_threshold:", lambda_threshold)
            # self.n_pca_comp = 0
            # for i in range(self.max_pca_comp):
            #     t0 = time.time()
            #     M = l2.tod.reshape((n, p))
            #     mask = l2.freqmask.reshape(n)
            #     # M = M[l2.freqmask.reshape(n), :]
            #     # M[~np.isfinite(M)] = 0
            #     # M = M[np.sum(M != 0, axis=-1) != 0]
            #     if M.shape[0] > 10:
            #         n_actual = M.shape[0]
            #         r = np.ones(p)
            #         r /= np.linalg.norm(r)
            #         err = np.zeros(max_iter)
            #         eigvals = np.zeros(max_iter)
            #         # print("1:", time.time() - t0)
            #         # print(M.shape)
            #         # print(n, n_actual, p)
            #         # print(np.sum(~np.isfinite(M)))
            #         # print(M.shape, mask.shape, r.shape, eigvals.shape, err.shape, n_actual, p, max_iter, err_tol)
            #         # print(M)
            #         # print(np.sum(mask != 0))
            #         t0 = time.time()
            #         PCAlib.PCA(M, mask, r, eigvals, err, n_actual, p, max_iter, err_tol)
            #         pca_eigval[i] = eigvals[-1]
            #         # print("PCA", i, eigvals[-1])
            #         # print("2:", time.time() - t0)
            #         t0 = time.time()
            #         pca_comp[i] = r
            #         # for ifeed in range(l2.Nfeeds):
            #         #     ak = np.sum(l2.tod[ifeed,:,:,:]*r, axis=-1)
            #         #     l2.tod[ifeed] = l2.tod[ifeed] - ak[:,:,None]*r[None,None,:]
            #         #     pca_ampl[i,ifeed] = ak
            #         # print("3:", time.time() - t0)
            #         ak = np.zeros((l2.Nfeeds, l2.Nsb, l2.Nfreqs))
            #         PCAlib.subtract_eigcomp(l2.tod, r, ak, l2.Nfeeds*l2.Nsb*l2.Nfreqs, l2.Ntod)
            #         pca_ampl[i] = ak
            #         l2.tofile_dict["PCA_err"][i] = err
            #         self.n_pca_comp = i+1

            #         if eigvals[-1] < lambda_threshold:
            #             pca_ampl = pca_ampl[:i+1]
            #             pca_comp = pca_comp[:i+1]
            #             pca_eigval = pca_eigval[:i+1]
            #             break

            # l2.tofile_dict["pca_ampl"] = pca_ampl
            # l2.tofile_dict["pca_comp"] = pca_comp
            # l2.tofile_dict["pca_eigval"] = pca_eigval
            # l2.tofile_dict["n_pca_comp"] = self.n_pca_comp
            # # print("num PCA comp: ", self.n_pca_comp)

class PCA_feed_filter(Filter):
    name = "pcaf"
    name_long = "PCA feed filter"
    run_when_masking = True
    has_corr_template = True

    # def get_corr_template(l2, C, feed, ihalf):
    #     pca_ampl = l2.tofile_dict["pca_feed_ampl"][:,feed]
    #     b = pca_ampl.reshape((pca_ampl.shape[0], l2.Nsb*l2.Nfreqs)).T
    #     b[~np.isfinite(b)] = 0
    #     b[~l2.freqmask[feed].flatten()] = 0.0
    #     if (b != 0).any():
    #         C_template = -b.dot(np.linalg.inv(b.T.dot(b)).dot(b.T))
    #         C_template = C_template[ihalf*l2.Nfreqs*2:(ihalf+1)*l2.Nfreqs*2, ihalf*l2.Nfreqs*2:(ihalf+1)*l2.Nfreqs*2]
    #         mask = l2.freqmask[feed,ihalf*2:(ihalf+1)*2].reshape((2*l2.Nfreqs))
    #         C_template[~mask,:] = 0.0
    #         C_template[:,~mask] = 0.0
    #     return C_template

            # b = pca_ampl[:,ifeed].reshape((self.max_pca_comp, l2.Nsb*l2.Nfreqs)).T
            # b[~np.isfinite(b)] = 0
            # for i in range(4):
            #     b[:,i] /= np.linalg.norm(b[:,i])
            # l2.corr_template[ifeed] += -b.dot(b.T)

    def __init__(self, params, omp_num_threads=2, use_ctypes=False):
        self.omp_num_threads = omp_num_threads
        self.max_pca_comp = params.max_pca_comp
        self.N_deci_freqs = 64
        # self.deci_factor = 16
        self.params = params
        self.use_ctypes = use_ctypes
    

    def run(self, l2, masking_run=False):
        if len(self.params.load_PCA_path) > 0:  # This parameter option indicates that we should not calculate PCA components, but gather them from the scans in this folder.
            l2.tofile_dict["pca_feed_ampl"] = np.zeros((self.max_pca_comp, l2.Nfeeds, l2.Nsb, l2.Nfreqs))
            l2.tofile_dict["pca_feed_comp"] = np.zeros((self.max_pca_comp, l2.Nfeeds, l2.Ntod))
            l2.tofile_dict["n_pca_feed_comp"] = np.zeros((l2.Nfeeds), dtype=int)
            l2.tofile_dict["pca_feed_eigvals"] = np.zeros((l2.Nfeeds, self.max_pca_comp))

            path = os.path.join(self.params.load_PCA_path, l2.fieldname)
            filename = os.path.join(path, l2.l2_filename + ".h5")
            if not os.path.isfile(filename):
                raise ValueError(f"Specified l2 file for loading PCAf components does not exist: {filename}.")
            l2.printer.debug_print(f"Using imported PCAf components from {filename}.")
            with h5py.File(filename, "r") as f:
                comps = f["pca_feed_comp"][()]
                n_pca_comp = f["n_pca_feed_comp"][()]
                pca_eigvals = f["pca_feed_eigvals"][()]

            for ifeed in range(l2.Nfeeds):
                comp = comps[:n_pca_comp[ifeed],ifeed]
                ak = np.nansum(l2.tod[ifeed,:,:,:,None]*comp.T, axis=2)
                l2.tod[ifeed] = l2.tod[ifeed] - np.nansum(ak[:,:,None,:]*comp.T[None,None,:,:], axis=-1)

                l2.tofile_dict["pca_feed_ampl"][:n_pca_comp[ifeed],ifeed] = np.transpose(ak, (2,0,1))

            l2.tofile_dict["pca_feed_comp"] = comps
            l2.tofile_dict["n_pca_feed_comp"] = n_pca_comp
            l2.tofile_dict["pca_feed_eigvals"] = pca_eigvals

            return

        self.deci_factor = l2.Nfreqs//self.N_deci_freqs
        # self.N_deci_freqs = l2.Nfreqs//self.deci_factor

        l2.tofile_dict["pca_feed_ampl"] = np.zeros((self.max_pca_comp, l2.Nfeeds, l2.Nsb, l2.Nfreqs))
        l2.tofile_dict["pca_feed_comp"] = np.zeros((self.max_pca_comp, l2.Nfeeds, l2.Ntod))
        l2.tofile_dict["n_pca_feed_comp"] = np.zeros((l2.Nfeeds), dtype=int)
        l2.tofile_dict["pca_feed_eigvals"] = np.zeros((l2.Nfeeds, self.max_pca_comp))
        l2.tofile_dict["pca_feed_lambda_threshold"] = np.zeros((l2.Nfeeds))
        
        N = l2.Nsb*self.N_deci_freqs
        P = l2.Ntod
        
        if masking_run and self.params.write_C_matrix:
            l2.tofile_dict["C_template_pcaf"] = np.zeros((l2.Nfeeds, 2, 2*l2.Nfreqs, 2*l2.Nfreqs))

        # weight = 1.0/np.nanvar(l2.tod, axis=-1)
        if not self.use_ctypes:
            # weight = 1.0/l2.Tsys**2
            weight = 1.0/l2.Tsys
            weight[~l2.freqmask] = 0.0
            for ifeed in range(l2.Nfeeds):
                M = np.zeros((N, P))
                for isb in range(l2.Nsb):
                    for ifreq in range(self.N_deci_freqs):
                        i = isb*self.N_deci_freqs + ifreq
                        M[i,:] = np.nansum(l2.tod[ifeed,isb,ifreq*self.deci_factor:(ifreq+1)*self.deci_factor,:]*weight[ifeed,isb,ifreq*self.deci_factor:(ifreq+1)*self.deci_factor,None], axis=0)
                        w = np.nansum(weight[ifeed,isb,ifreq*self.deci_factor:(ifreq+1)*self.deci_factor]**2, axis=0)
                        M[i,:] /= w
                        M[i,:] /= (self.deci_factor/np.sqrt(w))
                M[~np.isfinite(M)] = 0
                M = M[(M != 0).any(axis=1)]
                if M.shape[0] < 15:  # Very few channels remaining
                    continue
                pca = PCA(n_components=self.max_pca_comp, random_state=21)
                comps = pca.fit_transform(M.T)
                singular_values = pca.singular_values_
                for i in range(self.max_pca_comp):
                    comps[:,i] /= np.linalg.norm(comps[:,i])
                N_effective = M.shape[0]
                sigma0 = np.nanmean(np.nanstd(M[:,1:] - M[:,:-1], axis=-1)/np.sqrt(2))
                empirical_correction_factor_for_singular_values_of_large_gaussian_noise_matrices = 1.00476 - 3.957e-03*np.log10(P)*np.log10(N_effective) + 8.757e-05*(np.log10(P)*np.log10(N_effective))**2
                lambda_threshold = self.params.pcaf_lambda_threshold*empirical_correction_factor_for_singular_values_of_large_gaussian_noise_matrices*(np.sqrt(N_effective) + np.sqrt(P))*sigma0
                l2.tofile_dict["pca_feed_lambda_threshold"][ifeed] = lambda_threshold
                self.n_pca_comp = np.sum(singular_values > lambda_threshold)
                self.n_pca_comp = max(self.n_pca_comp, self.params.min_feed_pca_comp)
                comps = comps
                del(M, pca)
                
                ak = np.nansum(l2.tod[ifeed,:,:,:,None]*comps[:,:self.n_pca_comp], axis=2)
                l2.tod[ifeed] = l2.tod[ifeed] - np.nansum(ak[:,:,None,:]*comps[None,None,:,:self.n_pca_comp], axis=-1)
                l2.tofile_dict["pca_feed_ampl"][:self.n_pca_comp,ifeed] = np.transpose(ak, (2,0,1))
                l2.tofile_dict["pca_feed_comp"][:,ifeed] = np.transpose(comps, (1,0))
                l2.tofile_dict["n_pca_feed_comp"][ifeed] = self.n_pca_comp
                l2.tofile_dict["pca_feed_eigvals"][ifeed, :] = singular_values[:]**2
            if self.params.verbose:
                print(f"fPCA scan {l2.scanid} num of modes: {l2.tofile_dict['n_pca_feed_comp']}")
        else:
            raise ValueError("Depricated.")

            # t0 = time.time()
            # n = l2.Nsb*l2.Nfreqs
            # p = l2.Ntod
            # C_LIB_PATH = os.path.join(CURRENT_DIR, "C_libs/PCA/PCAlib.so.1")
            # PCAlib = ctypes.cdll.LoadLibrary(C_LIB_PATH)
            # float32_array2 = np.ctypeslib.ndpointer(dtype=ctypes.c_float, ndim=2, flags="contiguous")
            # float64_array1 = np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=1, flags="contiguous")
            # float32_array3 = np.ctypeslib.ndpointer(dtype=ctypes.c_float, ndim=3, flags="contiguous")
            # float64_array2 = np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=2, flags="contiguous")
            # bool_array1 = np.ctypeslib.ndpointer(dtype=ctypes.c_bool, ndim=1, flags="contiguous")
            # PCAlib.PCA.argtypes = [float32_array2, bool_array1, float64_array1, float64_array1, float64_array1, ctypes.c_long, ctypes.c_long, ctypes.c_long, ctypes.c_double]
            # PCAlib.subtract_eigcomp.argtypes = [float32_array3, float64_array1, float64_array2, ctypes.c_long, ctypes.c_long]
            # max_iter = self.params.pca_max_iter
            # err_tol = self.params.pca_error_tol
            # l2.tofile_dict["PCAf_err"] = np.zeros((self.max_pca_comp, l2.Nfeeds, max_iter))
            # pca_eigval = np.zeros((self.max_pca_comp, l2.Nfeeds))

            # # weight = 1.0/l2.Tsys**2
            # weight = 1.0/l2.Tsys
            # weight[~l2.freqmask] = 0.0
            # # print("1:", time.time() - t0); t0=time.time()
            # self.n_pca_comp = np.zeros(l2.Nfeeds, dtype=int)
            # for ifeed in range(l2.Nfeeds):
            #     for icomp in range(self.max_pca_comp):
            #         M = np.zeros((N, p), dtype=np.float32)
            #         mask = np.zeros((N), dtype=bool)
            #         for isb in range(l2.Nsb):
            #             for ifreq in range(self.N_deci_freqs):
            #                 i = isb*self.N_deci_freqs + ifreq
            #                 M[i,:] = np.nansum(l2.tod[ifeed,isb,ifreq*self.deci_factor:(ifreq+1)*self.deci_factor,:]*weight[ifeed,isb,ifreq*self.deci_factor:(ifreq+1)*self.deci_factor,None], axis=0)
            #                 w = np.nansum(weight[ifeed,isb,ifreq*self.deci_factor:(ifreq+1)*self.deci_factor]**2, axis=0)
            #                 M[i,:] /= w
            #                 M[i,:] /= (self.deci_factor/np.sqrt(w))
            #                 mask[i] = l2.freqmask[ifeed,isb,ifreq*self.deci_factor:(ifreq+1)*self.deci_factor].any()

            #         if (mask == 0).all():
            #             self.n_pca_comp[ifeed] = 0
            #             break
            #         M[~np.isfinite(M)] = 0
            #         M = M[(M != 0).any(axis=1)]

            #         # M = M[np.sum(M != 0, axis=-1) != 0]
            #         n_actual = M.shape[0]

            #         sigma0 = np.nanstd(M[:,1:] - M[:,:-1], axis=-1)/np.sqrt(2)
            #         sigma0 = np.nanmean(sigma0)
            #         lambda_threshold = 1.0*(np.sqrt(n_actual) + np.sqrt(p))**2*sigma0**2
            #         if lambda_threshold == 0:
            #             self.n_pca_comp[ifeed] = 0
            #             break

            #         # mask = np.ones(n_actual, dtype=bool)
            #         r = np.ones(p)
            #         r /= np.linalg.norm(r)
            #         err = np.zeros(max_iter)
            #         eigvals = np.zeros(max_iter)
            #         # print("2:", time.time() - t0); t0=time.time()
            #         PCAlib.PCA(M, mask, r, eigvals, err, n_actual, p, max_iter, err_tol)
            #         pca_eigval[icomp,ifeed] = eigvals[-1]
            #         # print("PCAf:", ifeed, icomp, eigvals[-1])
            #         # print("3:", time.time() - t0); t0=time.time()

            #         # ak = np.sum(l2.tod[ifeed,:,:,:]*r, axis=-1)
            #         # l2.tod[ifeed] = l2.tod[ifeed] - ak[:,:,None]*r[None,None,:]
            #         ak = np.zeros((l2.Nsb, l2.Nfreqs))
            #         PCAlib.subtract_eigcomp(l2.tod[ifeed], r, ak, l2.Nsb*l2.Nfreqs, l2.Ntod)


            #         # pca_ampl[icomp,ifeed] = ak
            #         # pca_comp[icomp,ifeed] = r
            #         l2.tofile_dict["PCAf_err"][icomp,ifeed] = err
            #         # print("4:", time.time() - t0); t0=time.time()

            #         self.n_pca_comp[ifeed] = icomp+1
            #         if eigvals[-1] < lambda_threshold:
            #             break

            # l2.tofile_dict["pca_feed_eigval"] = pca_eigval
            # l2.tofile_dict["n_pca_feed_comp"] = self.n_pca_comp
            # # print("num PCAf comp: ", self.n_pca_comp)


        if masking_run:
            for ifeed in range(l2.Nfeeds):
                b = l2.tofile_dict["pca_feed_ampl"][:,ifeed].reshape((self.max_pca_comp, l2.Nsb*l2.Nfreqs)).T
                b[~np.isfinite(b)] = 0
                b[~l2.freqmask[ifeed].flatten()] = 0.0
                if (b != 0).any():
                    template = -b.dot(np.linalg.pinv(b.T.dot(b)).dot(b.T))
                    mask = l2.freqmask[ifeed].flatten()
                    template[~mask,:] = 0.0
                    template[:,~mask] = 0.0
                    l2.corr_template[ifeed] += template
                else:
                    template = np.zeros((l2.Nfreqs*4, l2.Nfreqs*4))
                if l2.params.write_C_matrix:
                    l2.tofile_dict["C_template_pcaf"][ifeed,0] += template[l2.Nfreqs*0:l2.Nfreqs*2,l2.Nfreqs*0:l2.Nfreqs*2]
                    l2.tofile_dict["C_template_pcaf"][ifeed,1] += template[l2.Nfreqs*2:l2.Nfreqs*4,l2.Nfreqs*2:l2.Nfreqs*4]



        

class Masking(Filter):
    name = "mask"
    name_long = "masking"

    def __init__(self, params, omp_num_threads=2):
        self.omp_num_threads = omp_num_threads
        self.box_sizes = params.box_sizes
        self.stripe_sizes = params.stripe_sizes
        self.Nsigma_chi2_boxes = params.n_sigma_chi2_box
        self.Nsigma_chi2_stripes = params.n_sigma_chi2_stripe
        self.Nsigma_mean_boxes = params.n_sigma_mean_box
        self.Nsigma_prod_boxes = params.n_sigma_prod_box
        self.Nsigma_prod_stripes = params.n_sigma_prod_stripe
        self.prod_offset = params.prod_offset
        self.params = params
        self.verbose = self.params.verbose


    def run(self, l2):
        comm = MPI.COMM_WORLD
        Nranks = comm.Get_size()
        rank = comm.Get_rank()
        t0 = time.time(); pt0 = time.process_time()

        l2.printer.debug_print(f"[{rank}] [{self.name}] Starting correlation calculations and masking...")
        
        if len(self.params.load_freqmask_path) > 0:  # If not empty string, load freqmasks from this directory instead of computing it.
            path = os.path.join(self.params.load_freqmask_path, l2.fieldname)
            filename = os.path.join(path, l2.l2_filename + ".h5")
            if not os.path.isfile(filename):
                raise ValueError(f"Specified l2 file for loading freqmask does not exist: {filename}.")
            l2.printer.debug_print(f"Using imported freqmask from {filename}.")
            with h5py.File(filename, "r") as f:
                feeds = f["feeds"][()]
                feedmask = feeds != 20
                freqmask = f["freqmask_full_aftermasking"][()]
                freqmask = freqmask[feedmask, ...]

            l2.freqmask[~freqmask] = False
            l2.freqmask_reason[freqmask] += 2**l2.freqmask_counter; l2.freqmask_counter += 1
            l2.freqmask_reason_string.append("Imported freqmask")
        else:
            ### Aliasing masking ###
            if int(l2.obsid) < 28136:  # Newer obsids have different (overlapping) frequency grid which alleviates the aliasing problem.
                with h5py.File("/mn/stornext/d16/cmbco/comap/data/aux_data/aliasing_suppression.h5", "r") as f:
                    AB_mask = f["/AB_mask"][()]
                    leak_mask = f["/leak_mask"][()]
                for isb in l2.flipped_sidebands:
                    AB_mask[:,isb,:] = AB_mask[:,isb,::-1]
                    leak_mask[:,isb,:] = leak_mask[:,isb,::-1]
                if l2.Nfreqs == 512:
                    AB_mask = AB_mask[:,:,::2]
                    leak_mask = leak_mask[:,:,::2]
                l2.freqmask[AB_mask[l2.feeds-1] < self.params.aliasing_mask_dB] = False
                l2.freqmask[leak_mask[l2.feeds-1] < self.params.aliasing_mask_dB] = False
                l2.freqmask_reason[AB_mask[l2.feeds-1] < self.params.aliasing_mask_dB] += 2**l2.freqmask_counter; l2.freqmask_counter += 1
                l2.freqmask_reason_string.append("Aliasing suppression (AB_mask)")
                l2.freqmask_reason[leak_mask[l2.feeds-1] < self.params.aliasing_mask_dB] += 2**l2.freqmask_counter; l2.freqmask_counter += 1
                l2.freqmask_reason_string.append("Aliasing suppression (leak_mask)")
                l2.tofile_dict["AB_aliasing"] = AB_mask
                l2.tofile_dict["leak_aliasing"] = leak_mask



            l2_local = copy.deepcopy(l2)
            Nfreqs = l2_local.Nfreqs
            premask_tofile_dict_keys = list(l2_local.tofile_dict.keys())  # Store what keys where in the "tofile_dict" prior to premask filters.
            

            for masking_filter in l2.filter_list:  # Look through filter list and see if any filter needs to be run prior to masking.
                if masking_filter.run_when_masking:
                    if masking_filter.name == "freq":  # TODO: Needs to done differently.
                        masking_filter = Polynomial_filter
                    l2.printer.debug_print(f"[{rank}] [{self.name}] Running local {masking_filter.name_long} for masking purposes...")
                    t0 = time.time(); pt0 = time.process_time()
                    filter_local = masking_filter(self.params)
                    filter_local.run(l2_local, masking_run=True)
                    del(filter_local)
                    if self.params.write_inter_files:
                        l2_local.write_level2_data(name_extension=f"_premask_{masking_filter.name}")
                    l2.printer.debug_print(f"[{rank}] [{self.name}] Finished local/masking {masking_filter.name_long} in {time.time()-t0:.1f} s. Process time: {time.process_time()-pt0:.1f} s.")

            for key in l2_local.tofile_dict.keys():
                if not key in premask_tofile_dict_keys:  # If something new was added to the tofile_dict of l2_local during the premask filters, we need to explicitly move it to the main l2 file not to lose them.
                    l2.tofile_dict[f"premask_{key}"] = l2_local.tofile_dict[key]


            # ### Aliasing masking ###
            # if int(l2.obsid) < 28136:  # Newer obsids have different (overlapping) frequency grid which alleviates the aliasing problem.
            #     with h5py.File("/mn/stornext/d22/cmbco/comap/protodir/auxiliary/aliasing_suppression.h5", "r") as f:
            #         AB_mask = f["/AB_mask"][()]
            #         leak_mask = f["/leak_mask"][()]
            #     for isb in l2.flipped_sidebands:
            #         AB_mask[:,isb,:] = AB_mask[:,isb,::-1]
            #         leak_mask[:,isb,:] = leak_mask[:,isb,::-1]
            #     l2.freqmask[AB_mask[l2.feeds-1] < 15] = False
            #     l2.freqmask[leak_mask[l2.feeds-1] < 15] = False
            #     l2.freqmask_reason[AB_mask[l2.feeds-1] < 15] += 2**l2.freqmask_counter; l2.freqmask_counter += 1
            #     l2.freqmask_reason_string.append("Aliasing suppression (AB_mask)")
            #     l2.freqmask_reason[leak_mask[l2.feeds-1] < 15] += 2**l2.freqmask_counter; l2.freqmask_counter += 1
            #     l2.freqmask_reason_string.append("Aliasing suppression (leak_mask)")
            #     l2.tofile_dict["AB_aliasing"] = AB_mask
            #     l2.tofile_dict["leak_aliasing"] = leak_mask


            l2.tofile_dict["premask_C_lowres"] = np.zeros((l2.Nfeeds, 2, 2*l2.Nfreqs//16, 2*l2.Nfreqs//16), dtype=np.float32)
            l2.tofile_dict["C_template_lowres"] = np.zeros((l2.Nfeeds, 2, 2*l2.Nfreqs//16, 2*l2.Nfreqs//16), dtype=np.float32)
            if self.params.write_C_matrix:  # For debuging purposes only.
                l2.tofile_dict["premask_C"] = np.zeros((l2.Nfeeds, 2, l2.Nfreqs*2, l2.Nfreqs*2))
                l2.tofile_dict["C"] = np.zeros((l2.Nfeeds, 2, l2.Nfreqs*2, l2.Nfreqs*2))
                l2.tofile_dict["C_template"] = np.zeros((l2.Nfeeds, 2, l2.Nfreqs*2, l2.Nfreqs*2))
            max_corr = np.zeros((l2.Nfeeds, l2.Nsb, l2.Nfreqs))
            for ifeed in range(l2.tod.shape[0]):
                for ihalf in range(2):  # Perform seperate analysis on each half of of the frequency band.
                    tod = l2_local.tod[ifeed,ihalf*2:(ihalf+1)*2,:,:].copy()
                    tod = tod.reshape((2*tod.shape[1], tod.shape[2]))  # Merge sb dim and freq dim.
                    tod = tod[:,l2_local.mask_temporal[ifeed]]
                    Ntod = tod.shape[-1]
                    # Start from the already existing freqmasks.
                    freqmask = l2.freqmask[ifeed,ihalf*2:(ihalf+1)*2].flatten()
                    freqmask_reason = l2.freqmask_reason[ifeed,ihalf*2:(ihalf+1)*2].flatten()

                    C = np.corrcoef(tod)  # Correlation matrix.

                    l2.tofile_dict["premask_C_lowres"][ifeed, ihalf] = np.nanmean(C.reshape((2*l2.Nfreqs//16, 16, 2*l2.Nfreqs//16, 16)), axis=(1,3))

                    # Ignore masked frequencies.
                    C[~freqmask,:] = 0
                    C[:,~freqmask] = 0

                    # print("1:", C[:5,:5])
                    # Put diagonal and 1-off diagonal components to zero, to be ignored in analysis,
                    # because of high and expected correlation.
                    for i in range(Nfreqs*2):
                        C[i,i] = 0
                        if i+1 < Nfreqs*2:
                            C[i+1,i] = 0
                            C[i,i+1] = 0
                    
                    if l2.params.write_C_matrix:  # For debuging purposes only.
                        l2.tofile_dict["premask_C"][ifeed,ihalf] = C
                    
                    # corr_template = np.zeros((2*l2.Nfreqs, 2*l2.Nfreqs))  # The correlated induced by different filters, to be subtracted from the correlation matrix.
                    # for filter in l2.filter_list:
                    #     if filter.has_corr_template:
                    #         corr_from_filter = filter.get_corr_template(l2_local, C, ifeed, ihalf)
                    #         corr_from_filter[~freqmask,:] = 0.0
                    #         corr_from_filter[:,~freqmask] = 0.0
                    #         for i in range(Nfreqs*2):
                    #             corr_from_filter[i,i] = 0
                    #             if i+1 < Nfreqs*2:
                    #                 corr_from_filter[i+1,i] = 0
                    #                 corr_from_filter[i,i+1] = 0
                    #         corr_template += corr_from_filter
                            
                    #         if self.params.write_C_matrix:  # For debuging purposes only.
                    #             dictstring = f"C_template_{filter.name}"
                    #             if not dictstring in l2.tofile_dict:
                    #                 l2.tofile_dict[dictstring] = np.zeros((l2.Nfeeds, 2, l2.Nfreqs*2, l2.Nfreqs*2))
                    #             l2.tofile_dict[dictstring][ifeed,ihalf] += corr_from_filter

                    # print("2:", C[:5,:5])

                    corr_template = l2_local.corr_template[ifeed,ihalf*l2.Nfreqs*2:(ihalf+1)*l2.Nfreqs*2,ihalf*l2.Nfreqs*2:(ihalf+1)*l2.Nfreqs*2]

                    l2.tofile_dict["C_template_lowres"][ifeed, ihalf] = np.nanmean(corr_template.reshape((2*l2.Nfreqs//16, 16, 2*l2.Nfreqs//16, 16)), axis=(1,3))

                    # Ignore masked frequencies. (again, since template does not include aliasing masking, and has different mask.)
                    corr_template[~freqmask,:] = 0
                    corr_template[:,~freqmask] = 0

                    C -= corr_template

                    if l2.params.write_C_matrix:  # For debuging purposes only.
                        l2.tofile_dict["C"][ifeed,ihalf] = C
                        l2.tofile_dict["C_template"][ifeed,ihalf] = corr_template

                    # print("3:", C[:5,:5])



                    # Fill the max-correlation array, used for masking further down.
                    for isb in range(2):
                        for ifreq in range(l2.Nfreqs):
                            max_corr[ifeed, ihalf*2+isb, ifreq] = np.nanmax(np.abs(C[isb*l2.Nfreqs+ifreq]), axis=-1)


                    # chi2_matrix = np.zeros((2, 3, 2048, 2048))
                    for ibox in range(len(self.box_sizes)):
                        if l2.Nfreqs == 1024:
                            box_size = self.box_sizes[ibox]
                        else:
                            box_size = self.box_sizes[ibox]//2
                        Nsigma_chi2_box = self.Nsigma_chi2_boxes[ibox]
                        Nsigma_prod_box = self.Nsigma_prod_boxes[ibox]
                        Nsigma_mean_box = self.Nsigma_mean_boxes[ibox]
                        jump = l2.Nfreqs//64 # 16


                        for i in range((2*Nfreqs)//box_size):
                            for j in range((2*Nfreqs)//box_size):
                                if i < j:
                                    box = C[i*box_size:(i+1)*box_size, j*box_size:(j+1)*box_size]
                                    dof = np.sum(box != 0)
                                    if dof == 0:
                                        dof = 1
                                    # CHI2 MASKING
                                    corr = np.sum(box**2)*Ntod
                                    chi2 = (corr - dof)/np.sqrt(2*dof)
                                    # chi2_matrix[0, ibox, i*box_size:(i+1)*box_size, j*box_size:(j+1)*box_size] = chi2
                                    if chi2 > Nsigma_chi2_box:
                                        freqmask[i*box_size:(i+1)*box_size] = False
                                        for x in range(i*box_size, (i+1)*box_size):
                                            if freqmask_reason[x] & 2**(l2.freqmask_counter + ibox*3 + 0) == 0:
                                                freqmask_reason[x] += 2**(l2.freqmask_counter + ibox*3 + 0)
                                        freqmask[j*box_size:(j+1)*box_size] = False
                                        for x in range(j*box_size, (j+1)*box_size):
                                            if freqmask_reason[x] & 2**(l2.freqmask_counter + ibox*3 + 0) == 0:
                                                freqmask_reason[x] += 2**(l2.freqmask_counter + ibox*3 + 0)

                                    # PROD MASKING
                                    prod = np.sum(box[:-jump:2]*box[jump::2])*Ntod
                                    nprod = np.sum((box[:-jump:2]*box[jump::2]) != 0)
                                    if nprod == 0:
                                        nprod = 1
                                    if prod/nprod > Nsigma_prod_box*np.sqrt(1.0/nprod):
                                        freqmask[i*box_size:(i+1)*box_size] = False
                                        for x in range(i*box_size, (i+1)*box_size):
                                            if freqmask_reason[x] & 2**(l2.freqmask_counter + ibox*3 + 1) == 0:
                                                freqmask_reason[x] += 2**(l2.freqmask_counter + ibox*3 + 1)
                                        freqmask[j*box_size:(j+1)*box_size] = False
                                        for x in range(j*box_size, (j+1)*box_size):
                                            if freqmask_reason[x] & 2**(l2.freqmask_counter + ibox*3 + 1) == 0:
                                                freqmask_reason[x] += 2**(l2.freqmask_counter + ibox*3 + 1)

                                    # SUM MASKING
                                    box_sum = np.sum(box)*np.sqrt(Ntod)
                                    if np.abs(box_sum)/dof > Nsigma_mean_box*np.sqrt(1.0/dof):
                                        freqmask[i*box_size:(i+1)*box_size] = False
                                        for x in range(i*box_size, (i+1)*box_size):
                                            if freqmask_reason[x] & 2**(l2.freqmask_counter + ibox*3 + 2) == 0:
                                                freqmask_reason[x] += 2**(l2.freqmask_counter + ibox*3 + 2)
                                        freqmask[j*box_size:(j+1)*box_size] = False
                                        for x in range(j*box_size, (j+1)*box_size):
                                            if freqmask_reason[x] & 2**(l2.freqmask_counter + ibox*3 + 2) == 0:
                                                freqmask_reason[x] += 2**(l2.freqmask_counter + ibox*3 + 2)
                                # else:
                                    # chi2_matrix[0, ibox, i*box_size:(i+1)*box_size, j*box_size:(j+1)*box_size] = np.nan

                    for istripe in range(len(self.stripe_sizes)):
                        if l2.Nfreqs == 1024:
                            stripe_size = self.stripe_sizes[istripe]
                        else:
                            stripe_size = self.stripe_sizes[istripe]//2

                        Nsigma_chi2_stripe = self.Nsigma_chi2_stripes[istripe]
                        Nsigma_prod_stripe = self.Nsigma_prod_stripes[istripe]

                        for i in range((2*Nfreqs)//stripe_size):
                            stripe = C[i*stripe_size:(i+1)*stripe_size, :].copy()
                            for x in range(stripe.shape[0]):
                                stripe[x:,x] = 0
                            dof = np.sum(stripe != 0)
                            if dof == 0:
                                dof = 1
                            # CHI2 MASKING
                            corr = np.sum(stripe**2)*Ntod
                            chi2 = (corr - dof)/np.sqrt(2*dof)
                            # chi2_matrix[1, istripe, :, i*stripe_size:(i+1)*stripe_size] = chi2
                            if chi2 > Nsigma_chi2_stripe:
                                freqmask[i*stripe_size:(i+1)*stripe_size] = False
                                for x in range(i*stripe_size, (i+1)*stripe_size):
                                    if freqmask_reason[x] & 2**(l2.freqmask_counter + len(self.box_sizes)*3 + istripe*2 + 0) == 0:
                                        freqmask_reason[x] += 2**(l2.freqmask_counter + len(self.box_sizes)*3 + istripe*2 + 0)

                            # PROD MASKING
                            prod = np.sum(stripe[:,:-jump:2]*stripe[:,jump::2])
                            nprod = np.sum(stripe[:,:-jump:2]*stripe[:,jump::2] != 0)
                            if nprod == 0:
                                nprod = 1
                            if prod/nprod > Nsigma_prod_stripe*np.sqrt(1.0/nprod):
                                freqmask[i*stripe_size:(i+1)*stripe_size] = False
                                for x in range(i*stripe_size, (i+1)*stripe_size):
                                    if freqmask_reason[x] & 2**(l2.freqmask_counter + len(self.box_sizes)*3 + istripe*2 + 1) == 0:
                                        freqmask_reason[x] += 2**(l2.freqmask_counter + len(self.box_sizes)*3 + istripe*2 + 1)

                    l2.freqmask[ifeed,ihalf*2:(ihalf+1)*2] = freqmask.reshape((2,Nfreqs))
                    l2.freqmask_reason[ifeed,ihalf*2:(ihalf+1)*2] = freqmask_reason.reshape((2,Nfreqs))
            
            # Since we have a "feed" outer loop, we need to do this afterwards:
            for box_size in self.box_sizes:
                for method in ["chi2", "prod", "sum"]:
                    l2.freqmask_reason_string.append(f"Box {box_size} {method}")
                    l2.freqmask_counter += 1
            for stripe_size in self.stripe_sizes:
                for method in ["chi2", "prod"]:
                    l2.freqmask_reason_string.append(f"Stripe {stripe_size} {method}")
                    l2.freqmask_counter += 1


            ### Radiometer cut ###
            std_max = 1.15
            samprate = l2.samprate  # Hz. TODO: !! FIX !!
            dnu = np.abs(l2.freqs[0,1] - l2.freqs[0,0])*1e9  # GHz -> Hz
            radiometer = 1.0/np.sqrt(dnu * (1.0/samprate))
            var = np.nanvar(l2_local.tod, axis=-1)
            l2.freqmask[var > (radiometer*std_max)**2] = False
            l2.freqmask_reason[var > (radiometer*std_max)**2] += 2**l2.freqmask_counter
            l2.freqmask_counter += 1
            l2.freqmask_reason_string.append(f"Radiometer std")
            del(l2_local)

            
            ### Maxmimum freq-freq correlation masking ###
            max_corr_threshold = 0.1
            l2.freqmask[max_corr > max_corr_threshold] = False
            l2.freqmask_reason[max_corr > max_corr_threshold] += 2**l2.freqmask_counter
            l2.freqmask_counter += 1
            l2.freqmask_reason_string.append(f"Max corr")
            
            
            ### Mask channels stretches shorter or equal to 4 sequential channels.
            for ifeed in range(l2.Nfeeds):
                for isb in range(l2.Nsb):
                    stretch_start = []
                    stretch_len = []
                    i = -1
                    while i < Nfreqs-1:
                        i += 1
                        if l2.freqmask[ifeed,isb,i]:
                            stretch_start.append(i)
                            while i < Nfreqs-1:
                                i += 1
                                if not l2.freqmask[ifeed,isb,i]:
                                    stretch_len.append(i - stretch_start[-1])
                                    break
                    if len(stretch_start) > len(stretch_len):  # This happens if freqmask is True all the way to the end of the band, so the break is never reached.
                        stretch_len.append(Nfreqs - stretch_start[-1])

                    for i in range(len(stretch_len)):
                        if stretch_len[i] <= 4:
                            l2.freqmask[ifeed, isb, stretch_start[i]:stretch_start[i]+stretch_len[i]] = False
                            l2.freqmask_reason[ifeed, isb, stretch_start[i]:stretch_start[i]+stretch_len[i]] += 2**l2.freqmask_counter
            l2.freqmask_counter += 1
            l2.freqmask_reason_string.append(f"Lonely unmasked channel")

            
            ### Mask entire sideband if less than 10% of sideband is unmasked
            temp_mask = np.mean(l2.freqmask, axis=-1) > 0.1
            for ifeed in range(temp_mask.shape[0]):
                for isb in range(temp_mask.shape[1]):
                    if not temp_mask[ifeed,isb]:
                        l2.freqmask_reason[ifeed, isb, l2.freqmask[ifeed,isb]] += 2**l2.freqmask_counter  # Only add previously unmasked channels. No point in double-counting.
            l2.freqmask[~temp_mask] = False
            l2.freqmask_counter += 1
            l2.freqmask_reason_string.append(f"Less than 10% of band unmasked")
            del(temp_mask)
            
            
            
            l2.tofile_dict["freqmask_full_aftermasking"] = l2.freqmask.copy()            
            if self.params.write_C_matrix:
                outpath = os.path.join(l2.level2_dir, "plots")
                if not os.path.exists(outpath):
                    os.mkdir(outpath)
                fig, ax = plt.subplots(l2.Nfeeds, 4, figsize=(20, 100))
                for i in range(l2.Nfeeds):
                    ax[i,0].imshow(l2.tofile_dict["premask_C"][i,0], cmap="bwr", vmin=-0.01, vmax=0.01)
                    ax[i,1].imshow(l2.tofile_dict["premask_C"][i,1], cmap="bwr", vmin=-0.01, vmax=0.01)
                    ax[i,2].imshow(l2.tofile_dict["C"][i,0], cmap="bwr", vmin=-0.01, vmax=0.01)
                    ax[i,3].imshow(l2.tofile_dict["C"][i,1], cmap="bwr", vmin=-0.01, vmax=0.01)
                    ax[i,0].set_title(f"Feed {l2.feeds[i]}")
                    for j in range(4):
                        ax[i,j].grid(False)
                        # ax[i,j].axis(False)
                plt.savefig(os.path.join(outpath, f"corr_plot_{l2.scanid_str}.png"))

            
        l2.tod[~l2.freqmask] = np.nan

        if l2.is_sim:
            l2.signal_tod[~l2.freqmask] = np.nan

        l2.acceptrate = np.mean(l2.freqmask, axis=(-1))

        # Just printing stuff:
        def get_color(value):
            if value > 85:
                return "\033[96m"
            elif value > 70:
                return "\033[94m"
            elif value > 50:
                return "\033[93m"
            else:            
                return "\033[91m"
        printstring = f"[{rank}] [{l2.scanid}] [{self.name}] Acceptrate by feed and sideband:\n"
        printstring += f"           all"
        for ifeed in range(l2.Nfeeds):
            printstring += f"{l2.feeds[ifeed]:7d}"
        acc = np.sum(l2.acceptrate)/(l2.Nfeeds*l2.Nsb)*100
        printstring += f"\nall    {get_color(acc)}{acc:6.1f}%\033[0m"
        for ifeed in range(l2.Nfeeds):
            acc = np.sum(l2.acceptrate[ifeed])/(l2.Nsb)*100
            printstring += f"{get_color(acc)}{acc:6.1f}%\033[0m"
        for isb in range(l2.Nsb):
            acc = np.sum(l2.acceptrate[:,isb])/(l2.Nfeeds)*100
            printstring += f"\n  {isb}    {get_color(acc)}{acc:6.1f}%\033[0m"
            for ifeed in range(l2.Nfeeds):
                acc = np.sum(l2.acceptrate[ifeed,isb])*100
                printstring += f"{get_color(acc)}{acc:6.1f}%\033[0m"
        l2.printer.verbose_print(printstring)
        l2.printer.debug_print(f"[{rank}] [{self.name}] Finished correlation calculations and masking in {time.time()-t0:.1f} s. Process time: {time.process_time()-pt0:.1f} s.")


class Tsys_calc(Filter):
    name = "tsys"
    name_long = "Tsys calculation"
    
    def __init__(self, params, omp_num_threads=2):
        self.omp_num_threads = omp_num_threads
        self.cal_database_dir = params.cal_database_dir
        self.params = params

    def run(self, l2):
        Tcmb = 2.725
        obsid = l2.obsid_str
        t = l2.tod_times[l2.Ntod//2]  # scan center time.
        l2.Tsys = np.zeros((l2.Nfeeds, l2.Nsb, l2.Nfreqs)) + np.nan
        l2.Gain = np.zeros((l2.Nfeeds, l2.Nsb, l2.Nfreqs)) + np.nan
        Pcold = np.nanmean(l2.tod, axis=-1)
        Phot_interp = np.zeros(l2.Nfeeds)
        
        try:
            with h5py.File(os.path.join(self.cal_database_dir, obsid + ".h5"), "r") as f:
                Phot = f[f"Phot"][()]
                for isb in l2.flipped_sidebands:
                    Phot[:,isb,:] = Phot[:,isb,::-1]
                Thot = f[f"Thot"][()]
                calib_times = f[f"calib_times"][()]
                successful = f[f"successful"][()]
            if l2.Nfreqs == 512:
                Phot = np.mean(Phot.reshape((Phot.shape[0], 4, 512, 2, 2)), axis=-2)

        except Exception as e:
            Phot = np.zeros((l2.Nfeeds, l2.Nsb, l2.Nfreqs, 2)) + np.nan
            Thot = np.zeros((l2.Nfeeds, 2)) + np.nan
            calib_times = np.zeros((l2.Nfeeds, 2)) + np.nan
            successful = np.zeros((l2.Nfeeds, 2))
            l2.freqmask[:] = False
            l2.freqmask_reason[:] += 2**l2.freqmask_counter; l2.freqmask_counter += 1
            l2.freqmask_reason_string.append("Can't read or find Tsys calib file.")

        n_cal = np.zeros(l2.Nfeeds, dtype=int)
        any_calib_failed = False
        for ifeed in range(l2.Nfeeds):
            feed = l2.feeds[ifeed]
            if successful[feed-1,0] and successful[feed-1,1]:  # Both calibrations successful.
                t1 = calib_times[feed-1,0]
                t2 = calib_times[feed-1,1]
                P1, P2 = Phot[feed-1,:,:,0], Phot[feed-1,:,:,1]
                Phot_interp = (P1*(t2 - t) + P2*(t - t1))/(t2 - t1)
                T1, T2 = Thot[feed-1,0], Thot[feed-1,1]
                Thot_interp = (T1*(t2 - t) + T2*(t - t1))/(t2 - t1)
                n_cal[ifeed] = 2
            elif successful[feed-1,0]:  # Only first calibration successful: Use values from that one.
                Phot_interp = Phot[feed-1,:,:,0]
                Thot_interp = Thot[feed-1,0]
                n_cal[ifeed] = 1
            elif successful[feed-1,1]:  # Only second...
                Phot_interp = Phot[feed-1,:,:,1]
                Thot_interp = Thot[feed-1,1]
                n_cal[ifeed] = 1
            else:
                Phot_interp = np.zeros_like(Phot[feed-1,:,:,0]) + np.nan
                Thot_interp = np.zeros_like(Thot[feed-1,0]) + np.nan
                l2.freqmask[ifeed] = False
                l2.freqmask_reason[ifeed] += 2**l2.freqmask_counter
            l2.Gain[ifeed,:,:] = (Phot_interp - Pcold[ifeed])/(Thot_interp - Tcmb)
            l2.Tsys[ifeed,:,:] = (Thot_interp - Tcmb)/(Phot_interp/Pcold[ifeed] - 1)
        if any_calib_failed:  # We need to add this logic at the end, as it would be wrong if we repeated this this for every feed in the loop above.
            l2.freqmask_counter += 1
            l2.freqmask_reason_string.append("Both calibs marked as unsuccessful.")

        l2.tofile_dict["Tsys"] = l2.Tsys
        l2.tofile_dict["Gain"] = l2.Gain
        l2.tofile_dict["Pcold"] = Pcold
        l2.tofile_dict["Thot"] = Thot[l2.feeds-1]
        l2.tofile_dict["Phot"] = Phot[l2.feeds-1]
        l2.tofile_dict["Thot_times"] = calib_times[l2.feeds-1]
        l2.tofile_dict["n_cal"] = n_cal






class Calibration(Filter):
    name = "calib"
    name_long = "calibrations"
    def __init__(self, params, omp_num_threads=2):
        self.omp_num_threads = omp_num_threads
        self.max_tsys = params.max_tsys
        self.min_tsys = params.min_tsys
        self.median_cut = params.median_tsys_cut
        self.params = params

    def run(self, l2):
        l2.tod *= l2.Tsys[:,:,:,None]
        l2.freqmask[~np.isfinite(l2.Tsys)] = False
        l2.freqmask_reason[~np.isfinite(l2.Tsys)] += 2**l2.freqmask_counter; l2.freqmask_counter += 1
        l2.freqmask_reason_string.append("Tsys NaN or inf")

        l2.freqmask[l2.Tsys < self.params.min_tsys] = False
        l2.freqmask_reason[l2.Tsys < self.params.min_tsys] += 2**l2.freqmask_counter; l2.freqmask_counter += 1
        l2.freqmask_reason_string.append("Tsys < min_tsys")
        
        ### Running median tsys cuts ###
        kernel_size1 = 400
        kernel_size2 = 150
        median_cut1 = self.params.median_tsys_cut+10
        median_cut2 = self.params.median_tsys_cut

        tsys_temp = l2.Tsys.copy()
        tsys_temp[tsys_temp < self.params.min_tsys] = np.nan
        tsys_temp[tsys_temp > self.params.max_tsys] = np.nan
        running_median1 = np.zeros_like(tsys_temp)
        running_median2 = np.zeros_like(tsys_temp)
        for ifeed in range(l2.Nfeeds):
            for isb in range(l2.Nsb):
                # Iteration 1
                for i in range(l2.Nfreqs):
                    start = max(i-kernel_size1//2, 0)
                    stop = min(i+kernel_size1//2, l2.Nfreqs)
                    tsys_local = tsys_temp[ifeed,isb,start:stop]
                    if np.sum(np.isfinite(tsys_local)) > kernel_size1//5:  # If more than 20% of points remaining.
                        running_median1[ifeed,isb,i] = np.nanmedian(tsys_local)
                    else:
                        running_median1[ifeed,isb,i] = 0
                tsys_temp[ifeed, isb, tsys_temp[ifeed,isb] > (running_median1[ifeed,isb] + median_cut1)] = np.nan
                # Iteration 2
                for i in range(l2.Nfreqs):
                    start = max(i-kernel_size2//2, 0)
                    stop = min(i+kernel_size2//2, l2.Nfreqs)
                    tsys_local = tsys_temp[ifeed,isb,start:stop]
                    if np.sum(np.isfinite(tsys_local)) > kernel_size2//5:  # If more than 20% of points remaining.
                        running_median2[ifeed,isb,i] = np.nanmedian(tsys_local)
                    else:
                        running_median2[ifeed,isb,i] = 0
                tsys_temp[ifeed, isb, tsys_temp[ifeed,isb] > (running_median2[ifeed,isb] + median_cut2)] = np.nan
        tsys_median_mask = np.isfinite(tsys_temp)
        tsys_median_mask[~np.isfinite(l2.Tsys)] = True  # Channels that have already been flagged as NaN or inf, don't need to double-flag them.
        
        l2.freqmask[~tsys_median_mask] = False
        l2.freqmask_reason[~tsys_median_mask] += 2**l2.freqmask_counter; l2.freqmask_counter += 1
        l2.freqmask_reason_string.append("Tsys > running median max")
        

class Mask_Az_Edges(Filter):
    name = "az-cut"
    name_long = "Azimuth edge cut"
    def __init__(self, params, omp_num_threads=2):
        self.omp_num_threads = omp_num_threads
        self.params = params

    def run(self, l2):
        # az_mask = np.ones_like(l2.az, dtype=bool)
        for ifeed in range(l2.Nfeeds):
            # sigma0 = np.std(l2.tod[ifeed,:,:,1:] - l2.tod[ifeed,:,:,:-1], axis=-1)/np.sqrt(2)

            az_median = np.median(l2.az[ifeed])
            az_diff = np.abs(l2.az[ifeed] - az_median)
            initial_cut = np.percentile(az_diff, 80)
            edges = np.array(initial_cut < az_diff, dtype=int)
            diff = edges[1:] - edges[:-1]
            if edges[0]:
                diff[0] = 1
            if edges[-1]:
                diff[-1] = -1
            i = 0
            starts = []
            stops = []
            while i < diff.shape[0]:
                if diff[i] == 1:
                    start = i
                    while diff[i] != -1:
                        i += 1
                    stop = i
                    starts.append(start)
                    stops.append(stop)
                i += 1
            for i in range(len(starts)):
                extreme_idx = np.argmax(az_diff[starts[i]:stops[i]])
                cut_start = extreme_idx - self.params.az_edges_mask_size_before + starts[i]
                cut_stop = extreme_idx + self.params.az_edges_mask_size_after + starts[i]
                cut_start = max(0, cut_start)
                cut_stop = min(l2.Ntod, cut_stop)
                # cut_length = cut_stop - cut_start
                
                l2.mask_temporal[ifeed, cut_start:cut_stop] = False
                
            l2.tod[ifeed,:,:,~l2.mask_temporal[ifeed]] = np.nan
                # mean_start = np.mean(l2.tod[ifeed,:,:,cut_start-15:cut_start], axis=-1)
                # mean_stop = np.mean(l2.tod[ifeed,:,:,cut_stop:cut_stop+15], axis=-1)
                # mean = (mean_start + mean_stop)/2.0
                
                # for isb in range(l2.Nsb):
                #     for ifreq in range(l2.Nfreqs):
                #         l2.tod[ifeed,isb,ifreq,cut_start:cut_stop] = mean[isb,ifreq] + np.random.normal(0, sigma0[isb,ifreq], (cut_length,))
                # az_mask[ifeed,cut_start:cut_stop] = False
        # l2.tofile_dict["az_edge_mask"] = az_mask
        # l2.az_mask = az_mask



class Highpass(Filter):
    name = "highpass"
    name_long = "Highpass"
    def __init__(self, params, omp_num_threads=2):
        self.omp_num_threads = omp_num_threads
        self.params = params
        self.highpass_fknee = 0.3  # Hz
        self.highpass_alpha = 8.0

    def run(self, l2):
        data_padded = np.zeros((l2.Nfreqs,3*l2.Ntod))
        for ifeed in range(l2.Nfeeds):
            for isb in range(l2.Nsb):
                data_padded[:,l2.Ntod:2*l2.Ntod] = l2.tod[ifeed,isb]
                weights = np.linspace(1, 0, 100)
                data_padded[:,:l2.Ntod] = -l2.tod[ifeed,isb,:,::-1] + 2*np.average(l2.tod[ifeed,isb,:,:100], weights=weights, axis=-1)[:,None]
                data_padded[:,2*l2.Ntod:] = -l2.tod[ifeed,isb,:,::-1] + 2*np.average(l2.tod[ifeed,isb,:,-100:], weights=weights[::-1], axis=-1)[:,None]
                
                freq_padded = rfftfreq(3*l2.Ntod)*l2.samprate
                W = 1.0/(1 + (freq_padded/self.highpass_fknee)**self.highpass_alpha)
                lowpass = irfft(rfft(data_padded)*W)[:,l2.Ntod:2*l2.Ntod]
                
                l2.tod[ifeed,isb] -= lowpass