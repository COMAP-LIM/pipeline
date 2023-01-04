from map_object import COmap
from scipy import linalg

import numpy as np
from typing import Tuple, Optional
import numpy.typing as ntyping
import re
import warnings
from tqdm import tqdm

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

        # List of keys to perform PCA on (only per feed hence remove "map" and "rms")
        self.keys_to_pca = [
            key for key in map.keys if ("map" in key) and ("coadd" not in key)
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
        all_freqvec = np.zeros((nfeed, ncomp, nsb * nfreq))
        all_angvec = np.zeros((nfeed, ncomp, nra * ndec))
        all_singular_values = np.zeros((nfeed, ncomp))

        # Perform PCA decomposition of data per feed
        for feed in tqdm(range(nfeed)):
            feed_data = data[feed, :]
            feed_data = feed_data.reshape(nsb * nfreq, nra * ndec)
            freq_vec, singular_values, ang_vec = linalg.svd(
                feed_data, full_matrices=False
            )

            # feed_data = feed_data.reshape(nsb * nfreq, nra * ndec)
            # freq_vec, singular_values, ang_vec = sparse.linalg.svds(
            #     feed_data, k=self.ncomps
            # )

            # Fill buffers
            all_freqvec[feed, :, :] = freq_vec.T
            all_angvec[feed, :, :] = ang_vec
            all_singular_values[feed, :] = singular_values

        # Return eigenvectors and singular values of PCA
        return (
            all_freqvec.reshape(nfeed, ncomp, nsb, nfreq),
            all_angvec.reshape(nfeed, ncomp, nra, ndec),
            all_singular_values,
        )

    def get_coadded_feeds(self, key) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Coadd feed maps into one single map

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple of coadded map, hit map and rms map.
        """
        # Make keys for rms and nhit datasets that corresponds to map dataset
        rms_key = re.sub(r"map", "rms", key)
        nhit_key = re.sub(r"map", "nhit", key)

        # Get feed map data
        inmap = self.map[key]
        inrms = self.map[rms_key]
        innhit = self.map[nhit_key]

        # Define inverse variance
        inv_var = 1 / inrms**2

        # Mask zero hit regions
        mask = ~np.isfinite(inv_var)
        inv_var[mask] = 0

        # Define feed-coadded map
        map_coadd = inmap.copy()

        # Coadding feed maps
        map_coadd = map_coadd * inv_var
        map_coadd = map_coadd.sum(0)
        map_coadd = map_coadd / inv_var.sum(0)

        # Coadd nhit and rms feed maps
        nhit_coadd = innhit.sum(0).astype(np.float32)
        rms_coadd = 1 / np.sqrt(inv_var.sum(0))

        # Mask regions with no hits in coadded feed maps
        mask_coadd = ~np.isfinite(rms_coadd)
        map_coadd[mask_coadd] = 0
        nhit_coadd[mask_coadd] = 0
        rms_coadd[mask_coadd] = 0
        return (map_coadd, nhit_coadd, rms_coadd)

    def normalize_data(self, key: str, norm: str) -> ntyping.ArrayLike:
        """_summary_

        Args:
            key (str): key to map dataset to perform PCA on
            norm (str): String that specifies what RMS normalization to apply to map prior to SVD. Defaults to "rms".
                        Can take values;
                         - 'approx' - for normalizing map by first PCA mode of RMS-map
                         - 'rms' - for normalizing map by RMS-map
                         - 'var' - for normalizing map by variance-map

        Raises:
            ValueError: norm must be either of "approx", "rms", "var".

        Returns:
            np.ndarray: Normalized map dataset
        """

        # Making sure normalization is valid
        if norm not in ["approx", "rms", "var"]:
            message = 'Make sure that normalization argument norm is either of the three "approx", "rms", "var".'
            raise ValueError(message)

        # Make key for rms dataset that corresponds to map dataset
        rms_key = re.sub(r"map", "rms", key)

        if norm == "rms":
            # rms normalized PCA
            self.norm_exponent = 1
        elif norm == "var":
            # variance normalized PCA
            self.norm_exponent = 2
        else:
            # rms approximation normalized PCA
            return NotImplemented

        norm_data = self.map[key] / self.map[rms_key] ** self.norm_exponent

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
        rms_key = re.sub(r"map", "rms", key)

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
        map_reconstruction = np.sum(map_reconstruction, axis=1)

        # Get back original units by undoing normalization
        map_reconstruction = map_reconstruction * inrms**self.norm_exponent

        return map_reconstruction

    def compute_pca(self, norm: str = "rms") -> COmap:
        """Method to compute PCA of map datasets

        Args:
            norm (str, optional): String that specifies what RMS normalization to apply to map prior to SVD. Defaults to "rms".
                        Can take values;
                         - 'approx' - for normalizing map by first PCA mode of RMS-map
                         - 'rms' - for normalizing map by RMS-map
                         - 'var' - for normalizing map by variance-map
        Returns:
            COmap: Map object with PCA modes saved. If self.clean the map object
                   is also PCA subtracted.
        """
        if self.verbose:
            print(f"Computing {norm} normalized PCA of:")

        # Compute PCA of all feed-map datasets
        for key in self.keys_to_pca:
            if self.verbose:
                print(" " * 4 + "Dataset: " + f"{key}")

            if self.maskrms:
                # Masking high-noise regions
                maskrms = self.maskrms
                # Make keys for rms and nhit datasets that corresponds to map dataset
                rms_key = re.sub(r"map", "rms", key)
                nhit_key = re.sub(r"map", "nhit", key)

                self.map[key] = np.where(
                    1e6 * self.map[rms_key] < maskrms, self.map[key], 0
                )
                self.map[nhit_key] = np.where(
                    1e6 * self.map[rms_key] < maskrms, self.map[nhit_key], 0
                )
                self.map[rms_key] = np.where(
                    1e6 * self.map[rms_key] < maskrms, self.map[rms_key], 0
                )

            # Normalize data
            indata = self.normalize_data(key, norm)

            # Compute SVD basis of indata
            freqvec, angvec, singular_values = self.get_svd_basis(indata)

            # Save computed PCA components
            self.map[key + "_pca_freqvec"] = freqvec
            self.map[key + "_pca_angvec"] = angvec
            self.map[key + "_pca_sing_val"] = singular_values

            if self.clean:
                # Clean data
                map_reconstruction = self.reconstruct_modes(key)
                self.map[key] -= map_reconstruction

        if self.clean:
            # Coadd feed map
            map_coadd, nhit_coadd, rms_coadd = self.get_coadded_feeds("map")

            if not self.maskrms:
                # Assert if coadded rms and nhit maps are same as the ones from
                # original initialization
                assert np.allclose(nhit_coadd, self.map["nhit_coadd"])
                assert np.allclose(rms_coadd, self.map["rms_coadd"])

            self.map["map_coadd"] = map_coadd
            self.map["nhit_coadd"] = nhit_coadd
            self.map["rms_coadd"] = rms_coadd

        # Assigning parameter specifying that map object is PCA subtracted
        # and what norm was used
        self.map["is_pca_subtr"] = True
        self.map["pca_norm"] = norm
        self.map["n_pca"] = self.ncomps

        # Return copy of input map object
        return self.map
