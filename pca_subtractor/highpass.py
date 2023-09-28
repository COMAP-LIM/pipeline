from scipy import linalg
import numpy as np
from typing import Tuple, Optional
import numpy.typing as ntyping
import re
import warnings
from tqdm import tqdm, trange
import os
import sys
import ctypes
from scipy.fft import ifft2
from scipy.linalg import svd

current = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current)
sys.path.append(parent_directory)

from tod2comap.COmap import COmap

# Ignore RuntimeWarning
warnings.filterwarnings("ignore", category=RuntimeWarning)


class Highpass_filter_map:
    """Class for highpass filtering COMAP maps"""

    def __init__(
        self,
        map: COmap,
        verbose: bool = False,
        Ncomp: int = 8
        # ncomps: int,
        # clean: bool = True,
        # maskrms: Optional[float] = None,
        # subtract_mean: bool = False,
        # approx_noise: bool = False,
    ):

        self.map = map
        self.verbose = verbose
        self.Ncomp = Ncomp
        # self.ncomps = ncomps
        # self.clean = clean
        # self.maskrms = maskrms
        # self.subtract_mean = subtract_mean
        # self.approx_noise = approx_noise

        # List of keys to perform PCA on (only per feed hence remove "map" and "sigma_wn")
        self.keys_to_pca = [
            key for key in map.keys if (("map" in key) and ("coadd" not in key)) and ("saddlebag" not in key) and ("pca" not in key)
        ]
        self.Npix = self.map[self.keys_to_pca[0]].shape[-1]

        self.setup()


    def setup(self):
        # Construct large mode templates.
        Ncomp = self.Ncomp

        Ts = []
        modes = []
        Ts.append(np.ones((self.Npix,self.Npix)))
        modes.append([0,0])
        for i in range(Ncomp+1):
            for j in range(Ncomp+1):
                if i > 0 or j > 0:
                    if np.sqrt(i**2 + j**2) <= Ncomp:
                        asdf = np.zeros((self.Npix,self.Npix))
                        asdf[i,j] = 1.0
                        T = ifft2(asdf).real
                        Ts.append(T)
                        T = ifft2(asdf).imag
                        Ts.append(T)
                        modes.append([i,j])
        modes = np.array(modes).T
        Ts = np.array(Ts)
        Ntemp = Ts.shape[0]
        Ts = Ts.reshape((Ntemp, self.Npix*self.Npix)).T
        for i in range(Ts.shape[-1]):
            Ts[:,i] /= np.linalg.norm(Ts[:,i])
        self.Ts = Ts



    def run(self) -> COmap:
        for key in self.keys_to_pca:
            if self.verbose:
                print(" " * 4 + "Dataset: " + f"{key}")
            rms_key = re.sub(r"map", "sigma_wn", key)
            signal_map = self.map[key].reshape((20, 4, 64, 120, 120)).copy()
            rms_map = self.map[rms_key].reshape((20, 4, 64, 120, 120)).copy()
            N_inv_all = 1.0/rms_map**2
            N_inv_all[~np.isfinite(N_inv_all)] = 0.0
            N_inv_all[~np.isfinite(signal_map)] = 0.0
            N_inv_all /= np.linalg.norm(N_inv_all, axis=(-1,-2))[:,:,:,None,None]
            signal_map[~np.isfinite(signal_map)] = 0.0

            for ifeed in trange(19):
                for isb in range(4):
                    for ifreq in range(64):
                        N_inv = N_inv_all[ifeed,isb,ifreq].reshape((120*120))
                        N_inv[~np.isfinite(N_inv)] = 0.0
                        D = (self.Ts.T*N_inv).dot(self.Ts)
                        U, S, V = svd(D)
                        S[S/S[0] < 1e-10] = np.inf
                        D_inv = V.T.dot(np.eye(S.shape[0])/S).dot(U.T)
                        
                        K = self.Ts.T*N_inv
                        a = D_inv.dot(K).dot(signal_map[ifeed,isb,ifreq].flatten())
                        template = self.Ts.dot(a).reshape((self.Npix, self.Npix))

                        signal_map[ifeed,isb,ifreq] = signal_map[ifeed,isb,ifreq].reshape((self.Npix,self.Npix)) - template
                        signal_map[ifeed,isb,ifreq,N_inv.reshape((120,120))==0] = np.nan
            signal_map[N_inv_all==0] = np.nan

            self.map[key] = signal_map.reshape((20, 4, 64, 120, 120))

        self.map["is_highpassed"] = True
        self.map["mpca_highpass_Nmodes"] = self.Ncomp
        return self.map

