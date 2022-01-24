import copy
import numpy as np
import multiprocessing as mp
import ctypes
from scipy.fftpack import fft, ifft, next_fast_len
from scipy.optimize import curve_fit

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


def lowpass_filter(signal, fknee=0.01, alpha=4.0, samprate=50):
    Ntod = signal.shape[-1]
    Nfull = next_fast_len(2*Ntod)
    signal_padded = np.zeros((1024, Nfull))
    signal_padded[:,:Ntod] = signal[:,:]
    signal_padded[:,Ntod:Ntod*2] = signal[:,::-1]
    signal_padded[:,Ntod*2:] = np.nanmean(signal[:,:400], axis=-1)[:,None]

    freq_full = np.fft.fftfreq(Nfull)*samprate
    W = 1.0/(1 + (freq_full/fknee)**alpha)
    return ifft(fft(signal_padded)*W).real[:,:Ntod]
    



class Normalize_Gain:
    def __init__(self):
        self.name = "normalization"

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

        # with mp.Pool() as p:
        #     tod_lowpass = p.map(lowpass_filter_safe, l2.tod.reshape(4*l2.Nfeeds, 1024, l2.Ntod))
        # tod_lowpass = np.array(tod_lowpass).reshape(l2.Nfeeds, 4, 1024, l2.Ntod)
        # l2.tod = l2.tod/tod_lowpass - 1
        for feed in range(l2.Nfeeds):
            for sb in range(4):
                tod_lowpass = lowpass_filter(l2.tod[feed, sb])
                l2.tod[feed,sb] = l2.tod[feed,sb]/tod_lowpass - 1
        del(tod_lowpass)


class Decimate:
    def __init__(self):
        self.name = "decimation"

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
    def __init__(self):
        self.name = "pointing template subtraction"
    
    def run(self, l2):
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
                            (g, d, c), _ = curve_fit(az_el_func, (1.0/np.sin(l2.el[feed]*np.pi/180.0), l2.az[feed]), l2.tod_norm[feed, sb, freq], (g, d, c))
                        else:
                            g, d, c = 0, 0, 0
                    l2.tod[feed,sb,freq] = l2.tod[feed,sb,freq] - az_el_template(feed, g, d, c)

class Polynomial_filter:
    def __init__(self):
        self.name = "polynomial filter"

    def run(self, l2):
        sb_freqs = np.linspace(-1, 1, 1024)
        for feed in range(l2.Nfeeds):
            for sb in range(4):
                for idx in range(l2.Ntod):
                    if np.isfinite(l2.tod[feed,sb,:,idx]).all():
                        try:
                            c1, c0 = np.polyfit(sb_freqs, l2.tod[feed,sb,:,idx], 1, w=l2.freqmask[feed,sb])
                            l2.tod[feed, sb, :, idx] = l2.tod[feed, sb, :, idx] - c1*sb_freqs - c0
                        except:
                            pass


class PCA_filter:
    def __init__(self):
        self.name = "PCA filter"
    
    def run(self, l2):
        M = l2.tod.reshape(l2.Nfeeds*l2.Nsb*l2.Nfreqs, l2.Ntod)
        M = M[l2.freqmask.reshape(l2.Nfeeds*l2.Nsb*l2.Nfreqs), :]
        M = np.dot(M.T, M)
        eigval, eigvec = np.linalg.eigh(M)
        ak = np.sum(l2.tod[:,:,:,:,None]*eigvec[:,-4:], axis=2)
        l2.tod = l2.tod - np.sum(ak[:,:,None]*eigvec[:,-4:], axis=-1)


"/mn/stornext/d22/cmbco/comap/protodir/auxiliary/comap_freqmask_1024channels.txt"
"/mn/stornext/d22/cmbco/comap/protodir/auxiliary/Ka_detectors.txt"
"/mn/stornext/d22/cmbco/comap/protodir/auxiliary/aliasing_suppression.h5"

class Masking:
    def __init__(self):
        self.name = "masking"
        
    def run(self, l2):
        l2_local = copy.deepcopy(l2)
        
        poly = Polynomial_filter()
        poly.run(l2_local)
        
        pca = PCA_filter()
        pca.run(l2_local)


