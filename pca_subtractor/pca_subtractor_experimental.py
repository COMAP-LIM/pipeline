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

current = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current)
sys.path.append(parent_directory)

from tod2comap.COmap import COmap

# Ignore RuntimeWarning
warnings.filterwarnings("ignore", category=RuntimeWarning)


def PCA_newer(data, map_rms, nchannels, npix):
    data = data.copy().reshape((nchannels,npix))
    data[~np.isfinite(data)] = 0.0
    map_rms2 = map_rms.copy().reshape((nchannels,npix))
    map_rms2[map_rms2==0] = 1.0
    map_rms2[~np.isfinite(map_rms2)] = 1.0
    ci = np.random.normal(0, 1, nchannels)
    ai = np.random.normal(0, 1, (npix))
    for i in range(600):
        ci_new = np.sum(data*ai/map_rms2**2, axis=-1)/np.nansum(ai**2/map_rms2**2, axis=-1)
        ai_new = np.sum(data*ci_new[:,None]/map_rms2**2, axis=0)/np.nansum(ci_new[:,None]**2/map_rms2**2, axis=0)

        if np.nanmean(np.abs(ai - ai_new)) < 1e-8 and np.nanmean(np.abs(ci - ci_new)) < 1e-8:
            break
        elif i == 599:
            print(f"Warning: Experimental PCA did not converge. Residuals: {np.nanmean(np.abs(ai - ai_new)):.2e} & {np.nanmean(np.abs(ci - ci_new)):.2e}")
            break
        # if i%10 == 0:
            # print(i, np.sum(np.abs(ai - ai_new)), np.sum(np.abs(ci - ci_new)))
            
        ai = ai_new.copy()
        ci = ci_new.copy()
    print("--", i, np.nanmean(np.abs(ai - ai_new)), np.nanmean(np.abs(ci - ci_new)))
    return ai_new, ci_new


def PCA_experimental_ctypes(data, map_rms, nchannels, npix):
    map_signal = data.copy().reshape((nchannels, npix))
    map_signal[~np.isfinite(map_signal)] = 0.0
    inv_rms_2 = 1.0/map_rms.copy().reshape((nchannels,npix))**2
    inv_rms_2[inv_rms_2==0] = 1.0
    inv_rms_2[~np.isfinite(inv_rms_2)] = 1.0
    freqvec = np.random.normal(0, 1, nchannels)
    freqvec /= np.linalg.norm(freqvec)
    angvec = np.random.normal(0, 1, (npix))
    angvec = np.array(np.sum(map_signal, axis=0), dtype=np.float64)
    angvec /= np.linalg.norm(angvec)

    map_signal_T = np.ascontiguousarray(map_signal.T)
    inv_rms_2_T = np.ascontiguousarray(inv_rms_2.T)

    C_LIB_PATH = os.path.join(parent_directory, "C_libs/mPCA/mPCAlib.so.1")
    mPCAlib = ctypes.cdll.LoadLibrary(C_LIB_PATH)
    float32_array2 = np.ctypeslib.ndpointer(dtype=ctypes.c_float, ndim=2, flags="contiguous")
    float64_array1 = np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=1, flags="contiguous")
    mPCAlib.PCA_experimental.argtypes = [float32_array2, float32_array2, float32_array2, float32_array2, float64_array1, float64_array1, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double]
    mPCAlib.PCA_experimental(map_signal, map_signal_T, inv_rms_2, inv_rms_2_T, angvec, freqvec, nchannels, npix, 500, 1e-8)

    return angvec, freqvec


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

            mapshape = self.map[key].shape
            nfeed, nsb, nfreq, ndec, nra = mapshape
            self.nfeed, self.nsb, self.nfreq, self.ndec, self.nra = nfeed, nsb, nfreq, ndec, nra
            
            rms_map = self.map[rms_key].reshape((nfeed, nsb * nfreq, ndec, nra)).copy()
            signal_map = self.map[key].reshape((nfeed, nsb * nfreq, ndec, nra)).copy()

            freqvec = np.zeros((self.ncomps, nfeed, nsb * nfreq))
            angvec = np.zeros((self.ncomps, nfeed, ndec * nra))

            for ifeed in trange(20):
                for icomp in range(self.ncomps):
                    # ai, ci = PCA_newer(signal_map[ifeed], rms_map[ifeed])
                    ai, ci = PCA_experimental_ctypes(signal_map[ifeed], rms_map[ifeed], nsb * nfreq, ndec * nra)
                    angvec[icomp, ifeed] = ai
                    freqvec[icomp, ifeed] = ci
                    reconstruction = (ci[:,None]*ai[None,:]).reshape(nsb*nfreq, ndec, nra)
                    signal_map[ifeed] -= reconstruction
            
            self.map[key] = signal_map.reshape((nfeed, nsb, nfreq, ndec, nra))

            self.map[key + "_pca_freqvec"] = freqvec
            self.map[key + "_pca_angvec"] = angvec

            map_saddlebag_key = re.sub(r"map", "map_saddlebag", key)
            rms_saddlebag_key = re.sub(r"map", "sigma_wn_saddlebag", key)
            nhit_saddlebag_key = re.sub(r"map", "nhit_saddlebag", key)

            if "nhit" in self.map.keys:
                map_saddlebag, nhit_saddlebag, rms_saddlebag = self.get_coadded_saddlebags(key)
                self.map[nhit_saddlebag_key] = nhit_saddlebag
            else:
                map_saddlebag, rms_saddlebag = self.get_coadded_saddlebags(key)


            self.map[map_saddlebag_key] = map_saddlebag
            self.map[rms_saddlebag_key] = rms_saddlebag
            
            
            # Coadd feed map
            if "nhit" in self.map.keys:
                map_coadd, nhit_coadd, rms_coadd = self.get_coadded_feeds("map")
            else:
                map_coadd, rms_coadd = self.get_coadded_feeds("map")

            if not self.maskrms:
                # Assert if coadded rms and nhit maps are same as the ones from
                # original initialization
                #if "nhit" in self.map.keys:
                #    assert np.allclose(nhit_coadd, self.map["nhit_coadd"])

                rms_coadd[rms_coadd == 0] = np.inf

                #assert np.allclose(rms_coadd, self.map["sigma_wn_coadd"])

            rms_coadd[np.isinf(rms_coadd)] = np.nan

            self.map["map_coadd"] = map_coadd

            if "nhit" in self.map.keys:
                self.map["nhit_coadd"] = nhit_coadd

            self.map["sigma_wn_coadd"] = rms_coadd
            

        self.map["is_pca_subtr"] = True
        self.map["pca_norm"] = norm
        self.map["pca_approx_noise"] = False
        self.map["n_pca"] = self.ncomps

        return self.map


    def get_coadded_saddlebags(self, key) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Coadd feed maps into saddlebag maps

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple of coadded saddlebag map, hit map and rms map.
        """
        # Make keys for rms and nhit datasets that corresponds to map dataset
        rms_key = re.sub(r"map", "sigma_wn", key)
        nhit_key = re.sub(r"map", "nhit", key)

        # Get feed map data
        inmap = self.map[key]
        inrms = self.map[rms_key]
        if nhit_key in self.map.keys:
            innhit = self.map[nhit_key]

        # Define inverse variance
        inv_var = 1 / inrms**2


        # Mask zero hit regions
        mask = ~np.isfinite(inv_var)
        inv_var[mask] = 0


        # Define feed-coadded map
        map_saddlebag = np.zeros((self.map.saddlebag_feeds.shape[0], *self.map["map_coadd"].shape))

        sigma_wn_saddlebag = np.zeros((self.map.saddlebag_feeds.shape[0],  *self.map["sigma_wn_coadd"].shape))
        
        if nhit_key in self.map.keys:
            nhit_saddlebag = np.zeros((self.map.saddlebag_feeds.shape[0], *self.map["nhit_coadd"].shape), dtype = np.int32)

        for i in range(self.map.saddlebag_feeds.shape[0]):
            # Current saddle bag feed indices
            feeds_in_saddlebag = self.map.saddlebag_feeds[i] - 1

            weights = 1 / inrms[feeds_in_saddlebag, ...] ** 2 
            data = inmap[feeds_in_saddlebag, ...] * weights
            inv_var = np.nansum(weights, axis = 0)

            map_saddlebag[i] = np.nansum(data, axis = 0) / inv_var
            sigma_wn_saddlebag[i] = 1 / np.sqrt(inv_var)
            if nhit_key in self.map.keys:
                nhit_saddlebag[i] = np.sum(innhit[feeds_in_saddlebag, ...], axis = 0)
        
        where = ~np.isfinite(sigma_wn_saddlebag)
        map_saddlebag[where] = np.nan
        sigma_wn_saddlebag[where] = np.nan
    
        if nhit_key in self.map.keys:
            return (map_saddlebag, nhit_saddlebag, sigma_wn_saddlebag)
        else:
            return (map_saddlebag, sigma_wn_saddlebag)

    def get_coadded_feeds(self, key) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Coadd feed maps into one single map

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple of coadded map, hit map and rms map.
        """
        # Make keys for rms and nhit datasets that corresponds to map dataset
        rms_key = re.sub(r"map", "sigma_wn", key)
        nhit_key = re.sub(r"map", "nhit", key)

        # Get feed map data
        inmap = self.map[key]
        inrms = self.map[rms_key]
        if nhit_key in self.map.keys:
            innhit = self.map[nhit_key]

        # Define inverse variance
        inv_var = 1 / inrms**2


        # Mask zero hit regions
        mask = ~np.isfinite(inv_var)
        inv_var[mask] = 0


        # Define feed-coadded map
        map_coadd = inmap.copy()
        map_coadd[mask] = 0

        # Coadding feed maps
        map_coadd = map_coadd * inv_var
        map_coadd = map_coadd.sum(0)
        map_coadd = map_coadd / inv_var.sum(0)

        # Coadd nhit and rms feed maps
        if nhit_key in self.map.keys:
            nhit_coadd = innhit.sum(0).astype(np.float32)

        rms_coadd = 1 / np.sqrt(inv_var.sum(0))

        # Mask regions with no hits in coadded feed maps
        mask_coadd = ~np.isfinite(rms_coadd)
        map_coadd[mask_coadd] = np.nan
        rms_coadd[mask_coadd] = np.nan
        if nhit_key in self.map.keys:
            nhit_coadd[mask_coadd] = np.nan
            return (map_coadd, nhit_coadd, rms_coadd)
        else:
            return (map_coadd, rms_coadd)
        
