from scipy import linalg

import numpy as np
from typing import Tuple, Optional
import numpy.typing as ntyping
import re
import warnings
from tqdm import tqdm


import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current)
sys.path.append(parent_directory)

from tod2comap.COmap import COmap

# Ignore RuntimeWarning
warnings.filterwarnings("ignore", category=RuntimeWarning)


class PCA_SubTractor:
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
        """Initializing class instance

        Args:
            map (Map): COMAP map object to compute PCA compoents of.
            ncomp (int): Number of PCA components to compute/subtract.
            verbose (bool, optional): Boolean specifying whether to run in verbose mode.
            maskrms (float, optional): RMS value beyond which to mask maps (in muK).
            clean (bool, optional): Boolean specifying whether to clean subtract PCA modes.
        """

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

    def get_svd_basis(
        self, data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Method for finding the frequency-transverse PCA basis for input data.

        Args:
            data (np.ndarray): Data from which to compute PCA basis.
                               Must be of shape (sidebands, frequencies, RA, Dec)

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple containing frequency eigen-vector,
                                                       angular eigen-vector and singular values
                                                       making up PCA decomposition.
        """

        # Get length of array data's dimensions
        nfeed, nsb, nfreq, nra, ndec = data.shape

        # Total number of PCA components
        ncomp = nsb * nfreq

        # Define empty buffers
        all_freqvec = np.zeros((nfeed, self.ncomps, nsb * nfreq))
        all_angvec = np.zeros((nfeed, self.ncomps, nra * ndec))
        all_singular_values = np.zeros((nfeed,  nsb * nfreq))

        # Perform PCA decomposition of data per feed
        for feed in tqdm(range(nfeed)):
        # for feed in [7]:
            feed_data = data[feed, :]
            feed_data = feed_data.reshape(nsb * nfreq, nra * ndec)
            

            # freq_vec, singular_values, ang_vec = np.linalg.svd(
            freq_vec, singular_values, ang_vec = np.linalg.svd(
                feed_data, full_matrices=False
            )

            # feed_data = feed_data.reshape(nsb * nfreq, nra * ndec)
            # freq_vec, singular_values, ang_vec = sparse.linalg.svds(
            #     feed_data, k=self.ncomps
            # )

            # Fill buffers
            all_freqvec[feed, :self.ncomps, :] = (freq_vec.T)[:self.ncomps, :]
            all_angvec[feed, :self.ncomps, :] = ang_vec[:self.ncomps, :]
            all_singular_values[feed, :] = singular_values

        # Return eigenvectors and singular values of PCA
        return (
            all_freqvec.reshape(nfeed, self.ncomps, nsb, nfreq),
            all_angvec.reshape(nfeed, self.ncomps, nra, ndec),
            all_singular_values,
        )

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

    def define_normalization_exponent(self, norm: str = "sigma_wn"):
        """Methode that maps name of normalization to exponent number

        Args:
            norm (str, optional): String that specifies what RMS normalization to apply to map prior to SVD. Defaults to "sigma_wn".
                        Can take values;
                         - 'approx' - for normalizing map by first PCA mode of RMS-map
                         - 'sigma_wn' - for normalizing map by RMS-map
                         - 'var' - for normalizing map by variance-map
        """
        if norm == "sigma_wn":
            # rms normalized PCA
            self.norm_exponent = 1
        elif norm == "var":
            # variance normalized PCA
            self.norm_exponent = 2
        elif norm == "three":
            # rms^3 normalized PCA
            self.norm_exponent = 3
        elif norm == "weightless":
            # non-normalized PCA
            self.norm_exponent = 0

    def normalize_data(self, key: str, norm: str) -> ntyping.ArrayLike:
        """_summary_

        Args:
            key (str): key to map dataset to perform PCA on
            norm (str): String that specifies what RMS normalization to apply to map prior to SVD. Defaults to "sigma_wn".
                        Can take values;
                         - 'approx' - for normalizing map by first PCA mode of RMS-map
                         - 'rms' - for normalizing map by RMS-map
                         - 'var' - for normalizing map by variance-map

        Raises:
            ValueError: norm must be either of "approx", "sigma_wn", "var".

        Returns:
            np.ndarray: Normalized map dataset
        """

        # Making sure normalization is valid
        if norm not in ["approx", "sigma_wn", "var", "three", "weightless"]:
            message = 'Make sure that normalization argument norm is either of the three "approx", "sigma_wn", "var", "three", "weightless".'
            raise ValueError(message)

        # Make key for rms dataset that corresponds to map dataset
        rms_key = re.sub(r"map", "sigma_wn", key)

        if self.approx_noise:
            norm_data = self.map[key] * (self.weights ** self.norm_exponent)
        else:
            norm_data = self.map[key] / (self.map[rms_key] ** self.norm_exponent)

        if self.subtract_mean:
            print("Subtracting line-of-sign mean:")
            norm_data -= np.nanmean(norm_data, axis=(1, 2))[:, None, None, :, :]

        # Remove NaN values from indata
        norm_data = np.where(np.isfinite(norm_data), norm_data, 0)

        return norm_data

    def reconstruct_modes(self, key: str) -> np.ndarray:
        """Method to compute PCA reconstructions of computed PCA modes for given dataset key.

        Args:
            key (str): Dataset key for which to clean out self.ncomps PCA modes.

        Returns:
            np.ndarray: PCA map dataset reconstruction
        """

        # Getting SVD basis
        freqvec = self.map[key + "_pca_freqvec"][:, : self.ncomps, ...]
        angvec = self.map[key + "_pca_angvec"][:, : self.ncomps, ...]
        singular_values = self.map[key + "_pca_sing_val"][:, : self.ncomps]

        # Make key for rms dataset that corresponds to map dataset
        rms_key = re.sub(r"map", "sigma_wn", key)

        # Input rms map from key
        inrms = self.map[rms_key]

        # Perform outer product from basis vecotrs
        map_reconstruction = angvec[:, :, None, None, :, :]
        map_reconstruction = (
            map_reconstruction
            * freqvec[
                :,
                :,
                :,
                :,
                None,
                None,
            ]
        )
        map_reconstruction = (
            map_reconstruction * singular_values[:, :, None, None, None, None]
        )


        all_reconstructed_modes = map_reconstruction.copy().transpose(1, 0, 2, 3, 4, 5)
        map_reconstruction = np.sum(map_reconstruction, axis=1)

        # Get back original units by undoing normalization
        if self.approx_noise:
            map_reconstruction = map_reconstruction / self.weights
            all_reconstructed_modes = all_reconstructed_modes / self.weights
        else:
            map_reconstruction = map_reconstruction * (inrms**self.norm_exponent)
            all_reconstructed_modes = all_reconstructed_modes * (inrms[None, ...] ** self.norm_exponent)

        map_reconstruction = np.where(
            map_reconstruction != 0, map_reconstruction, np.nan
        )

        all_reconstructed_modes = np.where(
            all_reconstructed_modes != 0, all_reconstructed_modes, np.nan
        )

        return map_reconstruction, all_reconstructed_modes

    def compute_pca(self, norm: str = "sigma_wn") -> COmap:
        """Method to compute PCA of map datasets

        Args:
            norm (str, optional): String that specifies what RMS normalization to apply to map prior to SVD. Defaults to "sigma_wn".
                        Can take values;
                         - 'approx' - for normalizing map by first PCA mode of RMS-map
                         - 'sigma_wn' - for normalizing map by RMS-map
                         - 'var' - for normalizing map by variance-map
        Returns:
            COmap: Map object with PCA modes saved. If self.clean the map object
                   is also PCA subtracted.
        """
        if self.verbose:
            print(f"Computing {norm} normalized PCA of:")

        self.define_normalization_exponent(norm)

        # Empty dict to save pca reconstructions
        self.map.pca_reconstruction = {}
        self.map.full_pca_reconstruction = {}

        # Compute PCA of all feed-map datasets
        for key in self.keys_to_pca:
            
            # if "rndD" not in key or ("rndE" not in key or "rndF" not in key):
            #     continue
            # if not "RNDC1elev1" in key:
            #     continue
            
            # print(" " * 4 + "Dataset: " + f"{key}")
            
            if self.verbose:
                print(" " * 4 + "Dataset: " + f"{key}")

            if self.maskrms is not None:
                # Masking high-noise regions
                maskrms = self.maskrms

                if self.verbose:
                    print(
                        f"Masking all sigma_wn > {maskrms} times the mean bottom 100 noise on each (feed, frequency):"
                    )

                # Make keys for rms and nhit datasets that corresponds to map dataset
                rms_key = re.sub(r"map", "sigma_wn", key)
                nhit_key = re.sub(r"map", "nhit", key)


                feed_noise_freq_coadded = 1 / self.map[rms_key] ** 2
                mask = ~np.isfinite(feed_noise_freq_coadded)
                feed_noise_freq_coadded[mask] = 0

                feed_noise_freq_coadded = 1 / np.sqrt(np.sum(feed_noise_freq_coadded, axis = (1, 2)))

                nfeed, _, _, nra, ndec = self.map[rms_key].shape
                sorted_rms = feed_noise_freq_coadded.reshape(nfeed, nra * ndec)

                bottom100_idx = np.argpartition(sorted_rms, 100, axis=-1)[..., :100]
                bottom100 = np.take_along_axis(sorted_rms, bottom100_idx, axis=-1)

                mean_bottom100_rms = np.nanmean(bottom100, axis=-1) 

                noise_lim = self.maskrms * mean_bottom100_rms 

                mask = feed_noise_freq_coadded[:, None, None, :] * np.ones_like(self.map[rms_key]) > noise_lim[:, None, None, None, None] 

                self.map[key][mask] = np.nan

                # self.map[key] = np.where(
                #     self.map[rms_key] < noise_lim[..., None, None],
                #     self.map[key],
                #     np.nan,
                # )

                if nhit_key in self.map.keys:
                    # self.map[nhit_key] = np.where(
                    #     self.map[rms_key] < noise_lim[..., None, None],
                    #     self.map[nhit_key],
                    #     np.nan,
                    # )
                    self.map[nhit_key] = self.map[nhit_key].astype(np.float32)
                    self.map[nhit_key][mask] = np.nan


                # self.map[rms_key] = np.where(
                #     self.map[rms_key] < noise_lim[..., None, None],
                #     self.map[rms_key],
                #     np.nan,
                # )
                self.map[rms_key][mask] = np.nan

            if self.approx_noise:
                self.weights = self.approximate_sigma_wn(self.map[rms_key])
                self.weights = 1 / self.weights
                self.weights[~np.isfinite(self.weights)] = 0

            # Normalize data
            indata = self.normalize_data(key, norm)#.astype(np.float64)

            try:
                # Compute SVD basis of indata
                freqvec, angvec, singular_values = self.get_svd_basis(indata)
            except:
                print(f"SVD crash on key {key} :(")
                np.save(f"debug_svd_data_{key.split('/')[-1]}", indata)
                exit(1)
                
            # Save computed PCA components
            self.map[key + "_pca_freqvec"] = freqvec
            self.map[key + "_pca_angvec"] = angvec
            self.map[key + "_pca_sing_val"] = singular_values

            if self.clean:
                # Clean data
                map_reconstruction, full_map_reconstruction = self.reconstruct_modes(key)
                self.map[key] -= map_reconstruction
                self.map.pca_reconstruction[f"{key}"] = map_reconstruction
                self.map.full_pca_reconstruction[f"{key}"] = full_map_reconstruction
            
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


        if self.clean:
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

        # Assigning parameter specifying that map object is PCA subtracted
        # and what norm was used
        self.map["is_pca_subtr"] = True
        self.map["pca_norm"] = norm
        self.map["pca_approx_noise"] = self.approx_noise
        self.map["n_pca"] = self.ncomps

        # Return copy of input map object
        return self.map

    def approximate_sigma_wn(self, rms):
        """Method that computes a PCA approximation of the noise level use as weights on dataset when computing map PCA."""

        # rms = np.where(np.isfinite(rms), rms**self.norm_exponent, 0)
        # freqvec, angvec, singular_values = self.get_svd_basis(rms)

        # # We only want the dominant mode for this approximation
        # freqvec = freqvec[:, 0, ...]
        # angvec = angvec[:, 0, ...]
        # singular_values = singular_values[:, 0, ...]

        # # Perform outer product from basis vecotrs
        # rms_reconstruction = angvec[:, None, None, :, :]
        # rms_reconstruction = (
        #     rms_reconstruction
        #     * freqvec[
        #         :,
        #         :,
        #         :,
        #         None,
        #         None,
        #     ]
        # )
        # rms_reconstruction = (
        #     rms_reconstruction * singular_values[:, None, None, None, None]
        # )

        inv_var = 1 / rms ** 2
        mask = ~np.isfinite(inv_var)
        inv_var[mask] = 0.0 

        inv_var_per_freq = inv_var.sum((3, 4))
        inv_var_per_pix = inv_var.sum((1, 2))

        rms_per_freq = np.sqrt(np.prod(inv_var.shape[3:]) / inv_var_per_freq)
        rms_per_pix = np.sqrt(np.prod(inv_var.shape[1:3]) / inv_var_per_pix)

        rms_reconstruction = np.sqrt(rms_per_freq[:, :, :, None, None] * rms_per_pix[:, None, None, :, :])


        return rms_reconstruction

    def overwrite_maps_with_reconstruction(self, component = None):
        """Method that will overwrite, i.e. fill in, map data with PCA reconstruction"""

        if component is None:
            for key, value in self.map.pca_reconstruction.items():
                self.map[key] = value

            recon_map = self.map.pca_reconstruction["map"].copy()
            recon_sigma_wn = self.map["sigma_wn"].copy()

            inv_var = 1 / recon_sigma_wn ** 2
            mask = ~np.isfinite(inv_var)
            inv_var[mask] = 0
            recon_coadd = recon_map.copy() * inv_var
            recon_coadd[mask] = 0
            recon_coadd = recon_coadd.sum(0)

            inv_var = inv_var.sum(0)

            recon_coadd /= inv_var

            self.map["map_coadd"] = recon_coadd.copy()
            
            self.map["is_pca_subtr"] = False
            self.map["is_pca_recon"] = True
            self.map["pca_component"] = -1

        else:
            for key, value in self.map.full_pca_reconstruction.items():
                self.map[key] = value[component]

            recon_map = self.map.full_pca_reconstruction["map"][component].copy()
            recon_sigma_wn = self.map["sigma_wn"].copy()

            inv_var = 1 / recon_sigma_wn ** 2
            mask = ~np.isfinite(inv_var)
            inv_var[mask] = 0
            recon_coadd = recon_map.copy() * inv_var
            recon_coadd[mask] = 0
            recon_coadd = recon_coadd.sum(0)

            inv_var = inv_var.sum(0)

            recon_coadd /= inv_var


            self.map["map_coadd"] = recon_coadd.copy()
            
            self.map["is_pca_subtr"] = False
            self.map["is_pca_recon"] = True
            self.map["pca_component"] = component
        