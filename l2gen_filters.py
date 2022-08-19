import copy
from re import search
import numpy as np
import multiprocessing as mp
import ctypes
from tqdm import trange
from scipy.fft import fft, ifft, rfft, irfft, next_fast_len
from pixell import fft as pfft
from scipy.optimize import curve_fit
import h5py
from tqdm import trange
import matplotlib.pyplot as plt
import scipy.linalg
import time
from mpi4py import MPI
from sklearn.decomposition import PCA

C_LIB_PATH = "/mn/stornext/d22/cmbco/comap/jonas/l2gen_python/C_libs/normlib.so.1"

def lowpass_filter_safe(signal, fknee=0.01, alpha=4.0, samprate=50, num_threads=2):
    """ Applies a lowpass filter to a signal, and returns the low-frequency result.
        Uses mirrored padding of signal to deal with edge effects.
        Assumes signal is of shape [freq, time].
    """
    N = signal.shape[-1]
    signal_padded = np.zeros((1024, 2*N))
    signal_padded[:,:N] = signal
    signal_padded[:,N:] = signal[:,::-1]

    freq_full = np.fft.fftfreq(2*N)*samprate
    W = 1.0/(1 + (freq_full/fknee)**alpha)
    return ifft(fft(signal_padded)*W).real[:,:N]


def lowpass_filter(signal, fastlen, fknee=0.01, alpha=4.0, samprate=50, num_threads=-1):
    tx = time.time()
    Ntod = signal.shape[-1]
    signal_padded = np.zeros((1024, fastlen))

    signal_padded[:,:Ntod] = signal[:]
    signal_padded[:,-Ntod:] = signal[:,::-1]
    signal_padded[:,Ntod:fastlen//2] = signal[:,-(fastlen//2-Ntod):][::-1]
    signal_padded[:,fastlen//2:fastlen-Ntod] = signal[:,-(fastlen//2-Ntod):]

    # signal_padded[:,:Ntod] = signal[:,:]
    # signal_padded[:,Ntod:Ntod*2] = signal[:,::-1]
    # signal_padded[:,Ntod*2:] = np.nanmean(signal[:,:400], axis=-1)[:,None]

    freq_full = np.fft.rfftfreq(fastlen)*samprate
    W = 1.0/(1 + (freq_full/fknee)**alpha)
    # return ifft(fft(signal_padded)*W).real[:,:Ntod]  # Now crashes with more workers, for some reason ("Resource temporarily unavailable").
    # return ifft(fft(signal_padded, workers=num_threads)*W, workers=num_threads).real[:,:Ntod]
    return irfft(rfft(signal_padded, workers=num_threads)*W, workers=num_threads).real[:,:Ntod]


def lowpass_filter_new(signal, fastlen, fknee=0.01, alpha=4.0, samprate=50, num_threads=-1):
    Ntod = signal.shape[-1]
    signal_padded = np.zeros((1024, fastlen))
    x = np.linspace(0, 1, Ntod)
    c0 = np.nanmean(signal[:,:1000], axis=-1)
    c1 = np.nanmean(signal[:,-1000:], axis=-1) - c0
    lin_fit = c1[:,None]*x[None,:] + c0[:,None]
    signal_padded[:,:Ntod] = signal - lin_fit
    freq_full = np.fft.rfftfreq(fastlen)*samprate
    W = 1.0/(1 + (freq_full/fknee)**alpha)
    # return irfft(rfft(signal_padded, workers=num_threads)*W, workers=num_threads).real[:,:Ntod] + lin_fit
    return irfft(rfft(signal_padded)*W).real[:,:Ntod] + lin_fit



class Filter:
    name = ""  # Short name of filter, used for compact writes.
    name_long = ""  # Verbose, more explanatory name of filter.
    depends_upon = []  # Other filters which needs to be run before this one. List of strings (the short names).
    run_when_masking = False  # If set to True, this filter will be applied to a local copy of the data before masking.



class Normalize_Gain(Filter):
    name = "norm"
    name_long = "normalization"

    def __init__(self, params, use_ctypes=False, omp_num_threads=2):
        self.use_ctypes = use_ctypes
        self.omp_num_threads = omp_num_threads
        self.fknee = params.gain_norm_fknee
        self.alpha = params.gain_norm_alpha

    def run(self, l2):
        # print("1")
        # fastlen = next_fast_len(l2.tod.shape[-1]*2)
        # print(f" --- Normalization - lastlen {fastlen} from TOD len {l2.tod.shape[-1]}.")
        # print("2")
        # normlib = ctypes.cdll.LoadLibrary(C_LIB_PATH)
        # print("3")
        # float32_array4 = np.ctypeslib.ndpointer(dtype=ctypes.c_float, ndim=4, flags="contiguous")
        # print("4")
        # normlib.normalize.argtypes = [float32_array4, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        # print("5")
        # normlib.normalize(l2.tod, *l2.tod.shape, fastlen)
        # print("6")

        # l2.tod = l2.tod.reshape((4*l2.Nfeeds, 1024, l2.Ntod))
        # with mp.Pool() as p:
        #     tod_lowpass = p.map(lowpass_filter_safe, l2.tod)
        # for i in range(l2.Nfeeds*4):
        #     l2.tod[i] = l2.tod[i]/tod_lowpass[i] - 1
        # l2.tod = l2.tod.reshape((l2.Nfeeds, 4, 1024, l2.Ntod))
        # del(tod_lowpass)

        if not self.use_ctypes:
            Ntod = l2.tod.shape[-1]
            fft_times = np.load("/mn/stornext/d22/cmbco/comap/jonas/l2gen_python/fft_times_owl33_10runs.npy")
            search_start = int(Ntod*1.05)
            search_stop = int(Ntod*1.10)  # We add at most 10% to the TOD to find fastest length during FFT.
            if search_stop < fft_times.shape[-1]:  # If search search is within the catalogue, use it. Otherwise, use scipy.
                fastlen = np.argmin(fft_times[search_start:search_stop]) + search_start  # Find fastest FFT time within the search length.
            else:
                fastlen = next_fast_len(Ntod)
            # fastlen = next_fast_len(2*Ntod)
            # print(Ntod, fastlen)
            for feed in range(l2.Nfeeds):
                for sb in range(l2.Nsb):
                    tod_lowpass = lowpass_filter_new(l2.tod[feed, sb], fastlen, num_threads = self.omp_num_threads, fknee=self.fknee, alpha=self.alpha)
                    l2.tod[feed,sb] = l2.tod[feed,sb]/tod_lowpass - 1
            del(tod_lowpass)
        else:
            C_LIB_PATH = "/mn/stornext/d22/cmbco/comap/jonas/l2gen_python/C_libs/norm/normlib.so.1"
            normlib = ctypes.cdll.LoadLibrary(C_LIB_PATH)
            float32_array2 = np.ctypeslib.ndpointer(dtype=ctypes.c_float, ndim=2, flags="contiguous")
            normlib.normalize.argtypes = [float32_array2, ctypes.c_ulong, ctypes.c_ulong]
            Nfull = next_fast_len(2*l2.Ntod)
            for feed in range(l2.Nfeeds):
                for sb in range(l2.Nsb):
                    l2_padded = np.zeros((1024, Nfull), dtype=np.float32)
                    l2_padded[:,:l2.Ntod] = l2.tod[feed,sb,:,:]
                    l2_padded[:,l2.Ntod:l2.Ntod*2] = l2.tod[feed,sb,:,::-1]
                    l2_padded[:,l2.Ntod*2:] = np.nanmean(l2.tod[feed,sb,:,:400], axis=-1)[:,None]
                    normlib.normalize(l2_padded, l2.Nfreqs, Nfull)
                    l2.tod[feed,sb] = l2_padded[:,:l2.Ntod]
        # for feed in range(l2.Nfeeds):
        #     for sb in range(4):
        #         with mp.Pool(mp_threads) as p:
        #             tod_lowpass = np.array(p.map(lowpass_filter2, l2.tod[feed, sb]))
        #             l2.tod[feed,sb] = l2.tod[feed,sb]/tod_lowpass - 1
        # del(tod_lowpass)
        


class Decimation(Filter):
    name = "dec"
    name_long = "decimation"

    def __init__(self, params, omp_num_threads=2):
        self.omp_num_threads = omp_num_threads
        self.Nfreqs = params.decimation_freqs

    def run(self, l2):
        weight = 1.0/np.nanvar(l2.tod, axis=-1)
        weight[~l2.freqmask] = 0
        tod_decimated = np.zeros((l2.tod.shape[0], l2.tod.shape[1], self.Nfreqs, l2.tod.shape[3]))
        for freq in range(64):
            tod_decimated[:,:,freq,:] = np.nansum(l2.tod[:,:,freq*16:(freq+1)*16,:]*weight[:,:,freq*16:(freq+1)*16,None], axis=2)
            tod_decimated[:,:,freq,:] /= np.nansum(weight[:,:,freq*16:(freq+1)*16], axis=2)[:,:,None]
        l2.tod = tod_decimated
        l2.Nfreqs = 64
        l2.freqmask_decimated = np.zeros((l2.Nfeeds, l2.Nsb, 64))
        for freq in range(64):
            l2.freqmask_decimated[:,:,freq] = l2.freqmask[:,:,freq*16:(freq+1)*16].all()
        l2.tofile_dict["freqmask_decimated"] = l2.freqmask_decimated
        l2.tofile_dict["decimation_nu"] = l2.Nfreqs//l2.params.decimation_freqs


class Pointing_Template_Subtraction(Filter):
    name = "point"
    name_long = "pointing template subtraction"

    def __init__(self, params, use_ctypes=True, omp_num_threads=2):
        self.use_ctypes = use_ctypes
        self.omp_num_threads = omp_num_threads
    
    def run(self, l2):
        el_az_amp = np.zeros((3, l2.Nfeeds, l2.Nsb, l2.Nfreqs))
        if not self.use_ctypes:
            def az_func(x, d, c):
                return d*x + c
            def az_el_func(x, g, d, c):
                return g*x[0] + d*x[1] + c
            def az_el_template(feed, g, d, c):
                return g/np.sin(l2.el[feed]*np.pi/180.0) + d*l2.az[feed] + c

            g, d, c = 0, 0, 0
            for feed in range(l2.Nfeeds):
                for sb in range(4):
                    for freq in range(1024):
                        if l2.scantype == "ces":
                            if np.isfinite(l2.tod[feed, sb, freq]).all():
                                (d, c), _ = curve_fit(az_func, l2.az[feed], l2.tod[feed, sb, freq], (d, c))
                            else:
                                d, c = 0, 0
                        else:
                            if np.isfinite(l2.tod[feed, sb, freq]).all():
                                (g, d, c), _ = curve_fit(az_el_func, (1.0/np.sin(l2.el[feed]*np.pi/180.0), l2.az[feed]), l2.tod[feed, sb, freq], (g, d, c))
                            else:
                                g, d, c = 0, 0, 0
                        l2.tod[feed,sb,freq] = l2.tod[feed,sb,freq] - az_el_template(feed, g, d, c)
                        el_az_amp[:,feed,sb,freq] = g, d, c

        else:
            C_LIB_PATH = "/mn/stornext/d22/cmbco/comap/jonas/l2gen_python/C_libs/pointing/pointinglib.so.1"
            pointinglib = ctypes.cdll.LoadLibrary(C_LIB_PATH)
            float32_array3 = np.ctypeslib.ndpointer(dtype=ctypes.c_float, ndim=3, flags="contiguous")
            float64_array1 = np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=1, flags="contiguous")
            pointinglib.az_fit.argtypes = [float64_array1, float32_array3, ctypes.c_int, ctypes.c_int, float64_array1, float64_array1]
            pointinglib.az_el_fit.argtypes = [float64_array1, float64_array1, float32_array3, ctypes.c_int, ctypes.c_int, float64_array1, float64_array1, float64_array1]
            if l2.scantype == "ces":
                for ifeed in range(l2.Nfeeds):
                    c_est = np.zeros(l2.Nfreqs*l2.Nsb)
                    d_est = np.zeros(l2.Nfreqs*l2.Nsb)
                    pointinglib.az_fit(l2.az[ifeed], l2.tod[ifeed], l2.Nfreqs*l2.Nsb, l2.Ntod, c_est, d_est)
                    c_est = c_est.reshape((l2.Nsb, l2.Nfreqs))
                    d_est = d_est.reshape((l2.Nsb, l2.Nfreqs))
                    l2.tod[ifeed] -= l2.az[ifeed][None,None,:]*d_est[:,:,None] + c_est[:,:,None]
                    el_az_amp[:,ifeed,:,:] = np.zeros((l2.Nsb, l2.Nfreqs)), d_est, c_est
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
                    el_az_amp[:,ifeed,:,:] = g_est, d_est, c_est

        l2.tofile_dict["el_az_amp"] = el_az_amp



class Polynomial_filter(Filter):
    name = "poly"
    name_long = "polynomial filter"
    run_when_masking = True

    def __init__(self, params, use_ctypes=True, omp_num_threads=2):
        self.use_ctypes = use_ctypes
        self.omp_num_threads = omp_num_threads

    def run(self, l2):
        c0 = np.zeros((l2.Nfeeds, l2.Nsb, l2.Ntod)) + np.nan
        c1 = np.zeros((l2.Nfeeds, l2.Nsb, l2.Ntod)) + np.nan
        if not self.use_ctypes:
            sb_freqs = np.linspace(-1, 1, 1024)
            for feed in range(l2.Nfeeds):
                for sb in range(4):
                    for idx in range(l2.Ntod):
                        # if np.isfinite(l2.tod[feed,sb,:,idx]).all():
                        tod_local = l2.tod[feed,sb,:,idx].copy()
                        tod_local[~np.isfinite(tod_local)] = 0  # polyfit doesn't allow nans.
                        try:
                            c1[feed,sb,idx], c0[feed,sb,idx] = np.polyfit(sb_freqs, tod_local, 1, w=l2.freqmask[feed,sb])
                            l2.tod[feed, sb, :, idx] = l2.tod[feed, sb, :, idx] - c1[feed,sb,idx]*sb_freqs - c0[feed,sb,idx]
                        except:
                            pass

        else:
            C_LIB_PATH = "/mn/stornext/d22/cmbco/comap/jonas/l2gen_python/C_libs/polyfit/polyfit.so.1"
            polylib = ctypes.cdll.LoadLibrary(C_LIB_PATH)
            float32_array2 = np.ctypeslib.ndpointer(dtype=ctypes.c_float, ndim=2, flags="contiguous")
            float32_array1 = np.ctypeslib.ndpointer(dtype=ctypes.c_float, ndim=1, flags="contiguous")
            float64_array1 = np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=1, flags="contiguous")
            polylib.polyfit.argtypes = [float32_array1, float32_array2, float64_array1, ctypes.c_int, ctypes.c_int, float64_array1, float64_array1]

            sb_freqs = np.linspace(-1, 1, 1024, dtype=np.float32)
            for feed in range(l2.Nfeeds):
                for sb in range(4):
                    tod_local = l2.tod[feed, sb].copy()
                    tod_local[~np.isfinite(tod_local)] = 0
                    weights = np.array(l2.freqmask[feed, sb], dtype=float)
                    polylib.polyfit(sb_freqs, l2.tod[feed, sb], weights, l2.Nfreqs, l2.Ntod, c0[feed,sb], c1[feed,sb])
                    l2.tod[feed, sb, :, :] = l2.tod[feed, sb, :, :] - c1[feed,sb,None,:]*sb_freqs[:,None] - c0[feed,sb,None,:]  # Future improvement: Move into C.
        l2.tofile_dict["poly_coeff"] = np.zeros((2, l2.Nfeeds, l2.Nsb, l2.Ntod))
        l2.tofile_dict["poly_coeff"][:] = c0, c1

        # Adjust correlation template.
        with h5py.File("/mn/stornext/d22/cmbco/comap/protodir/auxiliary/corr_template.h5", "r") as f:
            T = f["corr"][()]        
        for i in range(4):
            l2.corr_template[:,l2.Nfreqs*i:l2.Nfreqs*(i+1),l2.Nfreqs*i:l2.Nfreqs*(i+1)] += T



class Frequency_filter(Filter):
    name = "freq"
    name_long = "frequency filter"
    run_when_masking = True

    def __init__(self, params, omp_num_threads=2):
        self.omp_num_threads = omp_num_threads
        self.prior_file = params.freqfilter_prior_file

    def PS_1f(self, freqs, sigma0, fknee, alpha, wn=True, Wiener=False, fknee_W=0.01, alpha_W=-4.0):
        PS = np.zeros_like(freqs)
        if wn:
            PS[1:] = sigma0**2*(1 + (freqs[1:]/fknee)**alpha)
        else:
            PS[1:] = sigma0**2*((freqs[1:]/fknee)**alpha)
        if Wiener:
            PS[1:] /= 1 + (freqs[1:]/fknee_W)**alpha_W
        return PS

    def gain_temp_sep(self, y, P, F, sigma0_g, fknee_g, alpha_g):
        Nfreqs, Ntod = y.shape
        freqs = np.fft.rfftfreq(Ntod, 1/50.0)
        Cf = self.PS_1f(freqs, sigma0_g, fknee_g, alpha_g, wn=False, Wiener=True)
        Cf[0] = 1
        
        sigma0_est = np.std(y[:,1:] - y[:,:-1], axis=1)/np.sqrt(2)
        sigma0_est = np.mean(sigma0_est[sigma0_est != 0])
        Z = np.eye(Nfreqs, Nfreqs) - P.dot(np.linalg.inv(P.T.dot(P))).dot(P.T)
        
        RHS = np.fft.rfft(F.T.dot(Z).dot(y))
        z = F.T.dot(Z).dot(F)
        # a_bestfit_f = RHS/(z + sigma0_est**2)
        a_bestfit_f = RHS/(z + sigma0_est**2/Cf)
        a_bestfit = np.fft.irfft(a_bestfit_f, n=Ntod)
        m_bestfit = np.linalg.inv(P.T.dot(P)).dot(P.T).dot(y - F*a_bestfit)
        
        return a_bestfit, m_bestfit


    def run(self, l2):
        l2.tofile_dict["freqfilter_P"] = np.zeros((l2.Nfeeds, l2.Nsb, l2.Nfreqs, 2))
        l2.tofile_dict["freqfilter_F"] = np.zeros((l2.Nfeeds, l2.Nsb, l2.Nfreqs, 1))
        l2.tofile_dict["freqfilter_m"] = np.zeros((l2.Nfeeds, l2.Nsb, 2, l2.Ntod))
        l2.tofile_dict["freqfilter_a"] = np.zeros((l2.Nfeeds, l2.Nsb, 1, l2.Ntod))

        with h5py.File(self.prior_file, "r") as f:
            sigma0_prior = f["sigma0_prior"][l2.feeds-1]
            fknee_prior = f["fknee_prior"][l2.feeds-1]
            alpha_prior = f["alpha_prior"][l2.feeds-1]
        
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

                if np.sum(P != 0) > 0 and np.sum(F != 0) > 0:
                    a, m = self.gain_temp_sep(y, P, F, sigma0_prior[feed], fknee_prior[feed], alpha_prior[feed])
                else:
                    a = np.zeros((1, l2.Ntod))
                    m = np.zeros((2, l2.Ntod))

                l2.tod[feed,sb] = l2.tod[feed,sb] - F.dot(a) - P.dot(m)

                b = P.copy()
                b[~np.isfinite(b)] = 0
                b[:,0] /= np.linalg.norm(b[:,0])
                b[:,1] /= np.linalg.norm(b[:,1])
                l2.corr_template[feed, l2.Nfreqs*sb:l2.Nfreqs*(sb+1), l2.Nfreqs*sb:l2.Nfreqs*(sb+1)] += -b.dot(b.T)

                l2.tofile_dict["freqfilter_P"][feed, sb] = P
                l2.tofile_dict["freqfilter_F"][feed, sb] = F
                l2.tofile_dict["freqfilter_m"][feed, sb] = m
                l2.tofile_dict["freqfilter_a"][feed, sb] = a



class PCA_filter(Filter):
    name = "pca"
    name_long = "PCA filter"
    run_when_masking = True

    def __init__(self, params, omp_num_threads=2):
        self.omp_num_threads = omp_num_threads
        self.n_pca_comp = params.n_pca_comp
    
    def run(self, l2):
        pca_ampl = np.zeros((self.n_pca_comp, l2.Nfeeds, l2.Nsb, l2.Nfreqs))
        # pca_comp = np.zeros((self.N_pca_modes, l2.Ntod))
        N = l2.Nfeeds*l2.Nsb*l2.Nfreqs
        M = l2.tod.reshape(N, l2.Ntod)
        M = M[l2.freqmask.reshape(N), :]
        if M.shape[0] == 0:
            return

        # M = np.dot(M.T, M)
        # eigval, eigvec = scipy.linalg.eigh(M, subset_by_index=(l2.Ntod-self.N_pca_modes, l2.Ntod-1))
        # eigval, comps = scipy.sparse.linalg.eigsh(M, k=self.n_pca_comp, v0=np.ones(l2.Ntod)/np.sqrt(l2.Ntod))
        # ak = np.sum(l2.tod[:,:,:,:,None]*eigvec, axis=3)
        # l2.tod = l2.tod - np.sum(ak[:,:,:,None,:]*eigvec[None,None,None,:,:], axis=-1)
        pca = PCA(n_components=4, random_state=49)
        comps = pca.fit_transform(M.T)
        for i in range(self.n_pca_comp):
            comps[:,i] /= np.linalg.norm(comps[:,i])
        for ifeed in range(l2.Nfeeds):
            ak = np.sum(l2.tod[ifeed,:,:,:,None]*comps, axis=2)
            l2.tod[ifeed] = l2.tod[ifeed] - np.sum(ak[:,:,None,:]*comps[None,None,:,:], axis=-1)
            pca_ampl[:,ifeed] = np.transpose(ak, (2,0,1))
        l2.tofile_dict["pca_ampl"] = pca_ampl#[::-1]  # Scipy gives smallest eigenvalues first, we want largest first.
        l2.tofile_dict["pca_comp"] = np.transpose(comps, (1,0))#[::-1]



class PCA_feed_filter(Filter):
    name = "pcaf"
    name_long = "PCA feed filter"
    run_when_masking = True

    def __init__(self, params, omp_num_threads=2):
        self.omp_num_threads = omp_num_threads
        self.n_pca_comp = 4
        self.deci_factor = 16
    
    def run(self, l2):
        self.N_deci_freqs = l2.Nfreqs//self.deci_factor

        pca_ampl = np.zeros((self.n_pca_comp, l2.Nfeeds, l2.Nsb, l2.Nfreqs))
        pca_comp = np.zeros((self.n_pca_comp, l2.Nfeeds, l2.Ntod))
        # l2.corr_template = np.zeros((l2.Nfeeds, l2.Nsb*l2.Nfreqs, l2.Nsb*l2.Nfreqs))

        N = l2.Nsb*self.N_deci_freqs

        weight = 1.0/np.nanvar(l2.tod, axis=-1)
        weight[~l2.freqmask] = 0.0
        for ifeed in range(l2.Nfeeds):
            M = np.zeros((N, l2.Ntod))
            for isb in range(l2.Nsb):
                for ifreq in range(self.N_deci_freqs):
                    i = isb*self.N_deci_freqs + ifreq
                    M[i,:] = np.nansum(l2.tod[ifeed,isb,ifreq*self.deci_factor:(ifreq+1)*self.deci_factor,:]*weight[ifeed,isb,ifreq*self.deci_factor:(ifreq+1)*self.deci_factor,None], axis=0)
                    M[i,:] /= np.nansum(weight[ifeed,isb,ifreq*self.deci_factor:(ifreq+1)*self.deci_factor], axis=0)
            M[~np.isfinite(M)] = 0
            M = M[np.sum(M != 0, axis=-1) != 0]
            if M.shape[0] == 0:
                continue
            # M = np.dot(M.T, M)
            # eigval, comps = scipy.sparse.linalg.eigsh(M, k=self.n_pca_comp, v0=np.ones(l2.Ntod)/np.sqrt(l2.Ntod))
            pca = PCA(n_components=4, random_state=21)
            comps = pca.fit_transform(M.T)
            for i in range(self.n_pca_comp):
                comps[:,i] /= np.linalg.norm(comps[:,i])
            ak = np.sum(l2.tod[ifeed,:,:,:,None]*comps, axis=2)
            l2.tod[ifeed] = l2.tod[ifeed] - np.sum(ak[:,:,None,:]*comps[None,None,:,:], axis=-1)
            pca_ampl[:,ifeed] = np.transpose(ak, (2,0,1))
            pca_comp[:,ifeed] = np.transpose(comps, (1,0))

            b = pca_ampl[:,ifeed].reshape((self.n_pca_comp, l2.Nsb*l2.Nfreqs)).T
            b[~np.isfinite(b)] = 0
            for i in range(4):
                b[:,i] /= np.linalg.norm(b[:,i])
            l2.corr_template[ifeed] += -b.dot(b.T)

        l2.tofile_dict["pca_feed_ampl"] = pca_ampl#[::-1]  # Scipy gives smallest eigenvalues first, we want largest first.
        l2.tofile_dict["pca_feed_comp"] = pca_comp#[::-1]



"/mn/stornext/d22/cmbco/comap/protodir/auxiliary/comap_freqmask_1024channels.txt"
"/mn/stornext/d22/cmbco/comap/protodir/auxiliary/Ka_detectors.txt"
"/mn/stornext/d22/cmbco/comap/protodir/auxiliary/aliasing_suppression.h5"
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

        # self.box_sizes = [32, 128, 512]
        # self.Nsigma_chi2_boxes = [6, 6, 6]
        # self.Nsigma_prod_boxes = [6, 5, 4]
        # self.Nsigma_mean_boxes = [6, 10, 14]
        # self.stripe_sizes = [32, 128, 1024]
        # self.Nsigma_chi2_stripes = [6, 6, 6]
        # self.Nsigma_prod_stripes = [6, 5, 4]
        
    def run(self, l2):
        comm = MPI.COMM_WORLD
        Nranks = comm.Get_size()
        rank = comm.Get_rank()

        l2_local = copy.deepcopy(l2)
        
        for masking_filter in l2.filter_list:  # Look through filter list and see if any filter needs to be run prior to masking.
            if masking_filter.run_when_masking:
                # print(f"[{rank}] [{self.name}] Running local {masking_filter.name_long} for masking purposes...")
                t0 = time.time()
                filter_local = masking_filter(self.params)
                filter_local.run(l2_local)
                del(filter_local)
                if self.params.write_inter_files:
                    l2_local.write_level2_data(name_extension=f"_mask_{masking_filter.name}")
                # print(f"[{rank}] [{self.name}] Finished local/masking {masking_filter.name_long} in {time.time()-t0:.1f} s.")


        # print(f"[{rank}] [{self.name}] Running local freqfilter for masking purposes...")
        # t0 = time.time()
        # poly = Frequency_filter(self.params)
        # poly.run(l2_local)
        # del(poly)
        # l2_local.write_level2_data(name_extension=f"_mask_freqfilter")
        # print(f"[{rank}] [{self.name}] Finished local/masking freqfilter in {time.time()-t0:.1f} s.")
        
        # print(f"[{rank}] [{self.name}] Running local PCA filter for masking purposes...")
        # t0 = time.time()
        # pca = PCA_filter(self.params)
        # pca.run(l2_local)
        # del(pca)
        # l2_local.write_level2_data(name_extension=f"_mask_pca")
        # print(f"[{rank}] [{self.name}] Finished local/masking PCA filter in {time.time()-t0:.1f} s.")

        # print(f"[{rank}] [{self.name}] Running local PCA feed filter for masking purposes...")
        # t0 = time.time()
        # pcaf = PCA_feed_filter(self.params)
        # pcaf.run(l2_local)
        # del(pcaf)
        # l2_local.write_level2_data(name_extension=f"_mask_pcaf")
        # print(f"[{rank}] [{self.name}] Finished local/masking PCA feed filter in {time.time()-t0:.1f} s.")


        if self.verbose:
            print(f"[{rank}] [{self.name}] Starting correlation calculations and masking...")
        t0 = time.time()
        Nfreqs = l2_local.Nfreqs
        Ntod = l2_local.Ntod


        l2.tofile_dict["C"] = np.zeros((l2.Nfeeds, 2, l2.Nfreqs*2, l2.Nfreqs*2))
        l2.tofile_dict["C_template"] = l2_local.corr_template
        for ifeed in range(l2.tod.shape[0]):
            for ihalf in range(2):  # Perform seperate analysis on each half of of the frequency band.
                tod = l2_local.tod[ifeed,ihalf*2:(ihalf+1)*2,:,:]
                tod = tod.reshape((2*tod.shape[1], tod.shape[2]))  # Merge sb dim and freq dim.
                # Start from the already existing freqmasks.
                freqmask = l2.freqmask[ifeed,ihalf*2:(ihalf+1)*2].flatten()
                freqmask_reason = l2.freqmask_reason[ifeed,ihalf*2:(ihalf+1)*2].flatten()

                C = np.corrcoef(tod)  # Correlation matrix.

                # Put diagonal and 1-off diagonal components to zero, to be ignored in analysis,
                # because of high and expected correlation.
                for i in range(Nfreqs*2):
                    C[i,i] = 0
                    if i+1 < Nfreqs*2:
                        C[i+1,i] = 0
                        C[i,i+1] = 0
                

                C -= l2_local.corr_template[ifeed, ihalf*2048:(ihalf+1)*2048, ihalf*2048:(ihalf+1)*2048]
                # Ignore masked frequencies.
                C[~freqmask,:] = 0
                C[:,~freqmask] = 0
                l2.tofile_dict["C"][ifeed,ihalf] = C


                chi2_matrix = np.zeros((2, 3, 2048, 2048))
                for ibox in range(len(self.box_sizes)):
                    box_size = self.box_sizes[ibox]
                    Nsigma_chi2_box = self.Nsigma_chi2_boxes[ibox]
                    Nsigma_prod_box = self.Nsigma_prod_boxes[ibox]
                    Nsigma_mean_box = self.Nsigma_mean_boxes[ibox]

                    for i in range((2*Nfreqs)//box_size):
                        for j in range((2*Nfreqs)//box_size):
                            if i < j:
                                box = C[i*box_size:(i+1)*box_size, j*box_size:(j+1)*box_size]
                                dof = np.sum(box != 0)
                                # CHI2 MASKING
                                corr = np.sum(box**2)*Ntod
                                chi2 = (corr - dof)/np.sqrt(2*dof)
                                chi2_matrix[0, ibox, i*box_size:(i+1)*box_size, j*box_size:(j+1)*box_size] = chi2
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
                                jump = 16
                                prod = np.sum(box[:-jump:2]*box[jump::2])*Ntod
                                nprod = np.sum((box[:-jump:2]*box[jump::2]) != 0)
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
                            else:
                                chi2_matrix[0, ibox, i*box_size:(i+1)*box_size, j*box_size:(j+1)*box_size] = np.nan


                for istripe in range(len(self.stripe_sizes)):
                    stripe_size = self.stripe_sizes[istripe]
                    Nsigma_chi2_stripe = self.Nsigma_chi2_stripes[istripe]
                    Nsigma_prod_stripe = self.Nsigma_prod_stripes[istripe]

                    for i in range((2*Nfreqs)//stripe_size):
                        stripe = C[i*stripe_size:(i+1)*stripe_size, :]
                        dof = np.sum(stripe != 0)
                        # CHI2 MASKING
                        corr = np.sum(stripe**2)*Ntod
                        chi2 = (corr - dof)/np.sqrt(2*dof)
                        chi2_matrix[1, istripe, :, i*stripe_size:(i+1)*stripe_size] = chi2
                        if chi2 > Nsigma_chi2_stripe:
                            freqmask[i*stripe_size:(i+1)*stripe_size] = False
                            for x in range(i*stripe_size, (i+1)*stripe_size):
                                if freqmask_reason[x] & 2**(l2.freqmask_counter + len(self.box_sizes)*3 + istripe*2 + 0) == 0:
                                    freqmask_reason[x] += 2**(l2.freqmask_counter + len(self.box_sizes)*3 + istripe*2 + 0)

                        # PROD MASKING
                        prod = np.sum(stripe[:,:-jump:2]*stripe[:,jump::2])
                        nprod = np.sum(stripe[:,:-jump:2]*stripe[:,jump::2] != 0)
                        if prod/nprod > Nsigma_prod_stripe*np.sqrt(1.0/nprod):
                            freqmask[i*stripe_size:(i+1)*stripe_size] = False
                            for x in range(i*stripe_size, (i+1)*stripe_size):
                                if freqmask_reason[x] & 2**(l2.freqmask_counter + len(self.box_sizes)*3 + istripe*2 + 1) == 0:
                                    freqmask_reason[x] += 2**(l2.freqmask_counter + len(self.box_sizes)*3 + istripe*2 + 1)

                l2.freqmask[ifeed,ihalf*2:(ihalf+1)*2] = freqmask.reshape((2,Nfreqs))
                l2.freqmask_reason[ifeed,ihalf*2:(ihalf+1)*2] = freqmask_reason.reshape((2,Nfreqs))
                # fig, ax = plt.subplots(2, 3, figsize=(14, 6))
                # for i in range(2):
                #     for j in range(3):
                #         plot = ax[i,j].imshow(np.abs(chi2_matrix[i,j]), vmin=0, vmax=12)
                #         plt.colorbar(plot, ax=ax[i,j])
                # plt.tight_layout()
                # plt.savefig(f"plots/{l2.scanid}_feed{l2.feeds[ifeed]}_half{ihalf}.png", bbox_inches="tight")
                # plt.clf()
        del(l2_local)
        
        # Since we have a "feed" outer loop, we need to do this afterwards:
        for box_size in self.box_sizes:
            for method in ["chi2", "prod", "sum"]:
                l2.freqmask_reason_string.append(f"Box {box_size} {method}")
                l2.freqmask_counter += 1
        for stripe_size in self.stripe_sizes:
            for method in ["chi2", "prod"]:
                l2.freqmask_reason_string.append(f"Stripe {stripe_size} {method}")
                l2.freqmask_counter += 1
            
        l2.tod[~l2.freqmask] = np.nan
        l2.acceptrate = np.sum(l2.freqmask, axis=(-1))/l2.Nfreqs
        l2.tofile_dict["acceptrate"] = l2.acceptrate

        if self.verbose:
            printstring = f"[{rank}] [{self.name}] Acceptrate by feed:\n"
            printstring += f"    all"
            for ifeed in range(l2.Nfeeds):
                printstring += f"{l2.feeds[ifeed]:7d}"
            printstring += f"\n{np.sum(l2.freqmask)/(l2.Nfeeds*l2.Nsb*l2.Nfreqs)*100:6.1f}%"
            for ifeed in range(l2.Nfeeds):
                printstring += f"{np.sum(l2.freqmask[ifeed])/(l2.Nsb*l2.Nfreqs)*100:6.1f}%"
            print(printstring)
            print(f"[{rank}] [{self.name}] Finished correlation calculations and masking in {time.time()-t0:.1f} s.")



class Tsys_calc(Filter):
    name = "tsys"
    name_long = "Tsys calculation"
    
    def __init__(self, params, omp_num_threads=2):
        self.omp_num_threads = omp_num_threads
        self.cal_database_file = params.cal_database_file

    def run(self, l2):
        Tcmb = 2.725
        obsid = l2.obsid
        t = l2.tod_times[l2.Ntod//2]  # scan center time.
        l2.Tsys = np.zeros((l2.Nfeeds, l2.Nsb, l2.Nfreqs)) + np.nan
        Pcold = np.nanmean(l2.tod, axis=-1)
        Phot_interp = np.zeros(l2.Nfeeds)
        
        with h5py.File(self.cal_database_file, "r") as f:
            # tsys = f[f"/obsid/{obsid}/Tsys_obsidmean"][()]
            Phot = f[f"/obsid/{obsid}/Phot"][()]
            for isb in l2.flipped_sidebands:
                Phot[:,isb,:] = Phot[:,isb,::-1]
            Thot = f[f"/obsid/{obsid}/Thot"][()]
            calib_times = f[f"/obsid/{obsid}/calib_times"][()]
            successful = f[f"/obsid/{obsid}/successful"][()]

        n_cals = np.zeros(l2.Nfeeds)
        for ifeed in range(l2.Nfeeds):
            feed = l2.feeds[ifeed]
            if successful[feed-1,0] and successful[feed-1,1]:  # Both calibrations successful.
                t1 = calib_times[feed-1,0]
                t2 = calib_times[feed-1,1]
                P1, P2 = Phot[feed-1,:,:,0], Phot[feed-1,:,:,1]
                Phot_interp = (P1*(t2 - t) + P2*(t - t1))/(t2 - t1)
                T1, T2 = Thot[feed-1,0], Thot[feed-1,1]
                Thot_interp = (T1*(t2 - t) + T2*(t - t1))/(t2 - t1)
                n_cals[ifeed] = 2
            elif successful[feed-1,0]:  # Only first calibration successful: Use values from that one.
                Phot_interp = Phot[feed-1,:,:,0]
                Thot_interp = Thot[feed-1,:,:,0]
                n_cals[ifeed] = 1
            elif successful[feed-1,1]:  # Only second...
                Phot_interp = Phot[feed-1,:,:,1]
                Thot_interp = Thot[feed-1,:,:,1]
                n_cals[ifeed] = 1
            l2.Tsys[ifeed,:,:] = (Thot_interp - Tcmb)/(Phot_interp/Pcold[ifeed] - 1)

        l2.tofile_dict["Tsys"] = l2.Tsys
        l2.tofile_dict["Pcold"] = Pcold
        l2.tofile_dict["Thot"] = Thot
        l2.tofile_dict["Phot"] = Phot
        l2.tofile_dict["Thot_times"] = calib_times[l2.feeds-1]
        l2.tofile_dict["n_cals"] = n_cals



class Calibration(Filter):
    name = "calib"
    name_long = "calibrations"
    depends_upon = ["tsys"]
    def __init__(self, params, omp_num_threads=2):
        self.omp_num_threads = omp_num_threads
        self.max_tsys = params.max_tsys
        self.min_tsys = params.min_tsys
    
    def run(self, l2):
        l2.tod *= l2.Tsys[:,:,:,None]
        l2.freqmask[~np.isfinite(l2.Tsys)] = False
        l2.freqmask_reason[~np.isfinite(l2.Tsys)] = 2**l2.freqmask_counter; l2.freqmask_counter += 1
        l2.freqmask_reason_string.append("Tsys NaN or inf")

        l2.freqmask[l2.Tsys < self.min_tsys] = False
        l2.freqmask_reason[l2.Tsys < self.min_tsys] = 2**l2.freqmask_counter; l2.freqmask_counter += 1
        l2.freqmask_reason_string.append("Tsys < min_tsys")

        l2.freqmask[l2.Tsys > self.max_tsys] = False
        l2.freqmask_reason[l2.Tsys > self.max_tsys] = 2**l2.freqmask_counter; l2.freqmask_counter += 1
        l2.freqmask_reason_string.append("Tsys > max_tsys")