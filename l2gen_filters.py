import copy
import numpy as np
import multiprocessing as mp
import ctypes
from tqdm import trange
from scipy.fft import fft, ifft, next_fast_len
from scipy.optimize import curve_fit
import h5py
from tqdm import trange
import matplotlib.pyplot as plt
import scipy.linalg
import time

C_LIB_PATH = "/mn/stornext/d22/cmbco/comap/jonas/l2gen_python/C_libs/normlib.so.1"

def lowpass_filter_safe(signal, fknee=0.01, alpha=4.0, samprate=50):
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


def lowpass_filter(signal, fknee=0.01, alpha=4.0, samprate=50, num_threads=-1):
    Ntod = signal.shape[-1]
    Nfull = next_fast_len(2*Ntod)
    signal_padded = np.zeros((1024, Nfull))
    signal_padded[:,:Ntod] = signal[:,:]
    signal_padded[:,Ntod:Ntod*2] = signal[:,::-1]
    signal_padded[:,Ntod*2:] = np.nanmean(signal[:,:400], axis=-1)[:,None]

    freq_full = np.fft.fftfreq(Nfull)*samprate
    W = 1.0/(1 + (freq_full/fknee)**alpha)
    return ifft(fft(signal_padded, workers=num_threads)*W, workers=num_threads).real[:,:Ntod]
    



class Normalize_Gain:
    def __init__(self, use_ctypes=False, omp_num_threads=2):
        self.name = "norm"
        self.name_long = "normalization"
        self.use_ctypes = use_ctypes
        self.omp_num_threads = omp_num_threads

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
            for feed in range(l2.Nfeeds):
                for sb in range(l2.Nsb):
                    tod_lowpass = lowpass_filter(l2.tod[feed, sb], num_threads = self.omp_num_threads)
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
        


class Decimation:
    def __init__(self, omp_num_threads=2):
        self.name = "dec"
        self.name_long = "decimation"
        self.omp_num_threads = omp_num_threads

    def run(self, l2):
        dec_factor = 16
        Nfreqs = l2.tod.shape[2]//dec_factor
        weight = 1.0/np.nanvar(l2.tod, axis=-1)
        tod_decimated = np.zeros((l2.tod.shape[0], l2.tod.shape[1], Nfreqs, l2.tod.shape[3]))
        for freq in range(64):
            tod_decimated[:,:,freq,:] = np.nansum(l2.tod[:,:,freq*16:(freq+1)*16,:]*weight[:,:,freq*16:(freq+1)*16,None], axis=2)
            tod_decimated[:,:,freq,:] /= np.nansum(weight[:,:,freq*16:(freq+1)*16], axis=2)[:,:,None]
        l2.tod = tod_decimated


class Pointing_Template_Subtraction:
    def __init__(self, use_ctypes=True, omp_num_threads=2):
        self.name = "pointing"
        self.name_long = "pointing template subtraction"
        self.use_ctypes = use_ctypes
        self.omp_num_threads = omp_num_threads
    
    def run(self, l2):
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


class Polynomial_filter:
    def __init__(self, use_ctypes=True, omp_num_threads=2):
        self.name = "poly"
        self.name_long = "polynomial filter"
        self.use_ctypes = use_ctypes
        self.omp_num_threads = omp_num_threads

    def run(self, l2):
        if not self.use_ctypes:
            sb_freqs = np.linspace(-1, 1, 1024)
            for feed in range(l2.Nfeeds):
                for sb in range(4):
                    for idx in range(l2.Ntod):
                        # if np.isfinite(l2.tod[feed,sb,:,idx]).all():
                        tod_local = l2.tod[feed,sb,:,idx].copy()
                        tod_local[~np.isfinite(tod_local)] = 0  # polyfit doesn't allow nans.
                        try:
                            c1, c0 = np.polyfit(sb_freqs, tod_local, 1, w=l2.freqmask[feed,sb])
                            l2.tod[feed, sb, :, idx] = l2.tod[feed, sb, :, idx] - c1*sb_freqs - c0
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
            c0 = np.zeros(l2.Ntod) + np.nan
            c1 = np.zeros(l2.Ntod) + np.nan
            for feed in range(l2.Nfeeds):
                for sb in range(4):
                    tod_local = l2.tod[feed, sb].copy()
                    tod_local[~np.isfinite(tod_local)] = 0
                    weights = np.array(l2.freqmask[feed, sb], dtype=float)
                    polylib.polyfit(sb_freqs, l2.tod[feed, sb], weights, l2.Nfreqs, l2.Ntod, c0, c1)
                    for idx in range(l2.Ntod):
                        if np.isfinite(c0[idx]) and np.isfinite(c1[idx]):
                            l2.tod[feed, sb, :, idx] = l2.tod[feed, sb, :, idx] - c1[idx]*sb_freqs - c0[idx]


class PCA_filter:
    def __init__(self, N_pca_modes=4, omp_num_threads=2):
        self.name = "pca"
        self.name_long = "PCA filter"
        self.N_pca_modes = N_pca_modes
        self.omp_num_threads = omp_num_threads
    
    def run(self, l2):
        N = l2.Nfeeds*l2.Nsb*l2.Nfreqs
        M = l2.tod.reshape(N, l2.Ntod)
        M = M[l2.freqmask.reshape(N), :]
        M = np.dot(M.T, M)
        # eigval, eigvec = scipy.linalg.eigh(M, subset_by_index=(l2.Ntod-self.N_pca_modes, l2.Ntod-1))
        eigval, eigvec = scipy.sparse.linalg.eigsh(M, k=4)
        # ak = np.sum(l2.tod[:,:,:,:,None]*eigvec, axis=3)
        # l2.tod = l2.tod - np.sum(ak[:,:,:,None,:]*eigvec[None,None,None,:,:], axis=-1)
        for ifeed in range(l2.Nfeeds):
            ak = np.sum(l2.tod[ifeed,:,:,:,None]*eigvec, axis=2)
            l2.tod[ifeed] = l2.tod[ifeed] - np.sum(ak[:,:,None,:]*eigvec[None,None,:,:], axis=-1)

"/mn/stornext/d22/cmbco/comap/protodir/auxiliary/comap_freqmask_1024channels.txt"
"/mn/stornext/d22/cmbco/comap/protodir/auxiliary/Ka_detectors.txt"
"/mn/stornext/d22/cmbco/comap/protodir/auxiliary/aliasing_suppression.h5"

class Masking:
    def __init__(self, omp_num_threads=2):
        self.name = "masking"
        self.name_long = "masking"
        self.freqmask_reason_idx_start = 5
        self.box_sizes = [32, 128, 512]
        self.Nsigma_chi2_boxes = [6, 6, 6]
        self.Nsigma_prod_boxes = [6, 5, 4]
        self.Nsigma_mean_boxes = [6, 10, 14]
        self.stripe_sizes = [32, 128, 1024]
        self.Nsigma_chi2_stripes = [6, 6, 6]
        self.Nsigma_prod_stripes = [6, 5, 4]
        self.omp_num_threads = omp_num_threads
        
    def run(self, l2):
        l2_local = copy.deepcopy(l2)
        
        print(f"[{self.name}] Running local polyfilter for masking purposes...")
        t0 = time.time()
        poly = Polynomial_filter()
        poly.run(l2_local)
        del(poly)
        l2_local.write_level2_data("data/", name_extension=f"_mask_poly")
        print(f"[{self.name}] Finished polyfilter in {time.time()-t0:.1f} s.")
        
        print(f"[{self.name}] Running local PCA filter for masking purposes...")
        t0 = time.time()
        pca = PCA_filter()
        pca.run(l2_local)
        del(pca)
        l2_local.write_level2_data("data/", name_extension=f"_mask_pca")
        print(f"[{self.name}] Finished PCA filter in {time.time()-t0:.1f} s.")


        print(f"[{self.name}] Starting correlation estimation and masking...")
        t0 = time.time()
        Nfreqs = l2_local.Nfreqs
        Ntod = l2_local.Ntod
        # Load 1st order polyfilter correlation template.
        with h5py.File("/mn/stornext/d22/cmbco/comap/protodir/auxiliary/corr_template.h5", "r") as f:
            T_small = f["corr"][()]
        T = np.zeros((Nfreqs*2,Nfreqs*2))
        for i in range(2):
            T[i*1024:(i+1)*Nfreqs,i*Nfreqs:(i+1)*Nfreqs] = T_small

        for ifeed in range(l2_local.tod.shape[0]):
            for ihalf in range(2):  # Perform seperate analysis on each half of of the frequency band.
                tod = l2_local.tod[ifeed,ihalf*2:(ihalf+1)*2,:,:]
                tod = tod.reshape((2*tod.shape[1], tod.shape[2]))  # Merge sb dim and freq dim.
                # Start from the already existing freqmasks.
                freqmask = l2_local.freqmask[ifeed,ihalf*2:(ihalf+1)*2].flatten()
                freqmask_reason = l2_local.freqmask_reason[ifeed,ihalf*2:(ihalf+1)*2].flatten()

                C = np.corrcoef(tod)  # Correlation matrix.
                C -= T  # Subtract polyfilter correlation template.
                
                # Put diagonal and 1-off diagonal components to zero, to be ignored in analysis,
                # because of high and expected correlation.
                for i in range(Nfreqs*2):
                    C[i,i] = 0
                    if i+1 < Nfreqs*2:
                        C[i+1,i] = 0
                        C[i,i+1] = 0
                
                # Ignore masked frequencies.
                C[~freqmask,:] = 0
                C[:,~freqmask] = 0

                chi2_matrix = np.zeros((2, 3, 2048, 2048))
                for ibox in range(len(self.box_sizes)):
                    box_size = self.box_sizes[ibox]
                    Nsigma_chi2_box = self.Nsigma_chi2_boxes[ibox]

                    for i in range((2*Nfreqs)//box_size):
                        for j in range((2*Nfreqs)//box_size):
                            if i < j:
                                box = C[i*box_size:(i+1)*box_size, j*box_size:(j+1)*box_size]
                                dof = np.sum(box != 0)
                                corr = np.sum(box**2)*Ntod
                                chi2 = (corr - dof)/np.sqrt(2*dof)
                                chi2_matrix[0, ibox, i*box_size:(i+1)*box_size, j*box_size:(j+1)*box_size] = chi2
                                if chi2 > Nsigma_chi2_box:
                                    freqmask[i*box_size:(i+1)*box_size] = False
                                    freqmask_reason[i*box_size:(i+1)*box_size] += 2**(self.freqmask_reason_idx_start + ibox + 1)
                            else:
                                chi2_matrix[0, ibox, i*box_size:(i+1)*box_size, j*box_size:(j+1)*box_size] = np.nan


                for istripe in range(len(self.stripe_sizes)):
                    stripe_size = self.stripe_sizes[istripe]
                    Nsigma_chi2_stripe = self.Nsigma_chi2_stripes[istripe]

                    for i in range((2*Nfreqs)//stripe_size):
                        stripe = C[i*stripe_size:(i+1)*stripe_size, :]
                        dof = np.sum(stripe != 0)
                        corr = np.sum(stripe**2)*Ntod
                        chi2 = (corr - dof)/np.sqrt(2*dof)
                        chi2_matrix[1, istripe, :, i*stripe_size:(i+1)*stripe_size] = chi2
                        if chi2 > Nsigma_chi2_stripe:
                            freqmask[i*stripe_size:(i+1)*stripe_size] = False
                            freqmask_reason[i*stripe_size:(i+1)*stripe_size] += 2**(self.freqmask_reason_idx_start + len(self.box_sizes) + istripe + 1)

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
        
        l2.tod[~l2.freqmask] = np.nan
        del(l2_local)


class Calibration:
    def __init__(self, omp_num_threads=2):
        self.name = "calib"
        self.name_long = "calibrations"
        self.omp_num_threads = omp_num_threads
    
    def run(self, l2):
        obsid = l2.obsid
        with h5py.File("/mn/stornext/d22/cmbco/comap/protodir/auxiliary/level1_database.h5", "r") as f:
            tsys = f[f"/obsid/{obsid}/Tsys_obsidmean"][()]
        l2.tod *= tsys[l2.feeds-1][:,:,:,None]