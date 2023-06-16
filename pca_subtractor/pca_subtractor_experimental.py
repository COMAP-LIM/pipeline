from scipy import linalg
import numpy as np
from typing import Tuple, Optional
import numpy.typing as ntyping
import re
import warnings
from tqdm import tqdm, trange
import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current)
sys.path.append(parent_directory)

from tod2comap.COmap import COmap

# Ignore RuntimeWarning
warnings.filterwarnings("ignore", category=RuntimeWarning)


def PCA_newer(data, map_rms):
    data = data.copy().reshape((256,120*120))
    data[~np.isfinite(data)] = 0.0
    map_rms2 = map_rms.copy().reshape((256,120*120))
    map_rms2[map_rms2==0] = 1.0
    map_rms2[~np.isfinite(map_rms2)] = 1.0
    ci = np.random.normal(0, 1, 256)
    ai = np.random.normal(0, 1, (120*120))
    for i in range(600):
        ci_new = np.sum(data*ai/map_rms2**2, axis=-1)/np.nansum(ai**2/map_rms2**2, axis=-1)
        ai_new = np.sum(data*ci_new[:,None]/map_rms2**2, axis=0)/np.nansum(ci_new[:,None]**2/map_rms2**2, axis=0)

        if np.nanmean(np.abs(ai - ai_new)) < 1e-8 and np.nanmean(np.abs(ci - ci_new)) < 1e-8:
            break
        elif i == 599:
            print(f"Warning: Experimental PCA did not converge. Residuals: {np.nansum(np.abs(ai - ai_new)):.2e} & {np.nansum(np.abs(ci - ci_new)):.2e}")
            break
        # if i%10 == 0:
            # print(i, np.sum(np.abs(ai - ai_new)), np.sum(np.abs(ci - ci_new)))
            
        ai = ai_new.copy()
        ci = ci_new.copy()
    # print("--", i, np.nanmean(np.abs(ai - ai_new)), np.nanmean(np.abs(ci - ci_new)))
    return ai_new, ci_new



class PCA_SubTractor_Experimental:
    """Class for computing PCA components of COMAP maps"""

    def __init__(
        self,
        map: COmap,
        ncomps: int,
        clean: bool = True,
        maskrms: Optional[float] = None,
        verbose: bool = False,
        subtract_mean: bool = False,
        approx_noise: bool = False,
    ):

        self.map = map
        self.ncomps = ncomps
        self.verbose = verbose
        self.clean = clean
        self.maskrms = maskrms
        self.subtract_mean = subtract_mean
        self.approx_noise = approx_noise

        # List of keys to perform PCA on (only per feed hence remove "map" and "sigma_wn")
        self.keys_to_pca = [
            key for key in map.keys if (("map" in key) and ("coadd" not in key)) and ("saddlebag" not in key)
        ]


    def compute_pca(self, norm: str = "sigma_wn") -> COmap:
        for key in self.keys_to_pca:
            if self.verbose:
                print(" " * 4 + "Dataset: " + f"{key}")

            rms_key = re.sub(r"map", "sigma_wn", key)

            rms_map = self.map[rms_key].reshape((20, 4*64, 120, 120)).copy()
            signal_map = self.map[key].reshape((20, 4*64, 120, 120)).copy()

            freqvec = np.zeros((self.ncomps, 20, 256))
            angvec = np.zeros((self.ncomps, 20, 120*120))

            for ifeed in trange(20):
                for icomp in range(self.ncomps):

                    ai, ci = PCA_newer(signal_map[ifeed], rms_map[ifeed])
                    angvec[icomp, ifeed] = ai
                    freqvec[icomp, ifeed] = ci
                    reconstruction = (ci[:,None]*ai[None,:]).reshape(256,120,120)
                    signal_map[ifeed] -= reconstruction
            
            self.map[key] = signal_map.reshape((20, 4, 64, 120, 120))

            self.map[key + "_pca_freqvec"] = freqvec
            self.map[key + "_pca_angvec"] = angvec

        self.map["is_pca_subtr"] = True
        self.map["pca_norm"] = norm
        self.map["pca_approx_noise"] = False
        self.map["n_pca"] = self.ncomps

        return self.map

