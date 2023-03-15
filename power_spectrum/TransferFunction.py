from __future__ import annotations
from typing import Optional
import h5py
import numpy as np
import numpy.typing as npt
from pixell import enmap
from dataclasses import dataclass, field
from power_spectrum import PowerSpectrum
from map_cosmo import MapCosmo


@dataclass
class TransferFunction:
    """COMAP transfer fuunction data class"""

    # Path to save/read in transfer function with
    path: str = field(default_factory=str)

    # List of length 3;
    #   [0] contains path to simulation only map,
    #   [1] contains path to map with injected signal and
    #   [2] contains path to map with only noise (pure COMAP data without injected signal).
    mappaths: list[str] = field(default_factory=list[str])

    # Internal data dictionary used to store private data
    _data: dict[str, npt.ArrayLike] = field(default_factory=dict)

    def compute_transfer_function(
        self,
        do_2d: bool = True,
        do_1d: bool = True,
        feed: Optional[int] = None,
        split: Optional[str] = None,
    ):
        """Method for computing the transfer function of the spherically or cylidrically averaged power spectra

        Args:
            do_2d (bool, optional): If True a cylidrically averaged power spectrum
            is used to get transfer function in parallel/perpendicular k-space.
            If False a spherically averaged power spectrum is used to estimate
            transfer function. Defaults to True.

            feed (int): which feed to use for estimating transfer function. Default None
            results in coadded map being used.

            split (str): split key (corresponding to HDF5 dataset key) specifying which split map to use for estimating transfer functions. Default None results in not using any splits .
        """

        if not do_1d and not do_2d:
            raise ValueError(
                "Make sure to specify either cylindrically or spherically averaged transfer function."
            )

        if len(self.mappaths) != 3:
            raise ValueError("Make sure to only provide three map paths. ")

        signal_map_path, injected_with_signal_map_path, noise_map_path = self.mappaths

        # Map object with only pure signal
        signal_map = MapCosmo(signal_map_path, feed=feed, split=split)

        # Power spectrum object with only spure signal
        P_signal = PowerSpectrum(signal_map)

        # Map object with signal injected and pipeline processed data
        injected_with_signal_map = MapCosmo(injected_with_signal_map_path, feed=feed, split=split)

        # Power spectrum object with signal injected and pipeline processed data
        P_injected_with_signal_map = PowerSpectrum(injected_with_signal_map)

        # Map object with pipeline processed noisy data
        noisemap = MapCosmo(noise_map_path, feed=feed, split=split)

        # Power spectrum object with pipeline processed noisy data
        P_noise = PowerSpectrum(noisemap)

        which_averaging = []
        if do_1d:
            which_averaging.append(False)
        if do_2d:
            which_averaging.append(True)

        for avg in which_averaging:
            (
                ps_pure_signal,
                k_bin_centers,
                _,
            ) = P_signal.calculate_ps(do_2d=avg)

            ps_injected_with_signal, _, _ = P_injected_with_signal_map.calculate_ps(do_2d=avg)

            ps_pure_noise, _, _ = P_noise.calculate_ps(do_2d=avg)

            # Transfer function computed by using the power spectra
            transfer_function = (ps_injected_with_signal - ps_pure_noise) / ps_pure_signal

            # Save as class attributes
            if avg:
                self.ps_pure_signal_2D = ps_pure_signal
                self.ps_injected_with_signal_2D = ps_injected_with_signal
                self.ps_pure_noise_2D = ps_pure_noise
                self.transfer_function_2D = transfer_function
                self.k_bin_centers_par_2D = k_bin_centers[0]
                self.k_bin_centers_perp_2D = k_bin_centers[1]
                self.k_bin_edges_par_2D = P_signal.k_bin_edges_par
                self.k_bin_edges_perp_2D = P_signal.k_bin_edges_perp
            else:
                self.ps_pure_signal_1D = ps_pure_signal
                self.ps_injected_with_signal_1D = ps_injected_with_signal
                self.ps_pure_noise_1D = ps_pure_noise
                self.transfer_function_1D = transfer_function

                self.k_bin_centers_1D = k_bin_centers
                self.k_bin_edges_1D = P_signal.k_bin_edges

    def write(self, outpath: Optional[str] = None):
        """Method that writes transfer function to HDF5 file

        Args:
            outpath (Optional[str], optional): Output path that to where file should be saved. Defaults to None.
        """

        if outpath is None:
            outpath = self.path

        print("Saving transfer function at: ", outpath)

        with h5py.File(outpath, "w") as outfile:
            if "transfer_function_2D" in self.__dict__.keys():
                cyl_avg = "cylindrically_averaged"
                # Save transfer function from cylindrically averaged power spectrum
                outfile[f"{cyl_avg}/transfer_function_2D"] = self.transfer_function_2D

                outfile[f"{cyl_avg}/k_bin_edges_par_2D"] = self.k_bin_edges_par_2D
                outfile[f"{cyl_avg}/k_bin_edges_perp_2D"] = self.k_bin_edges_perp_2D

                outfile[f"{cyl_avg}/k_bin_centers_par_2D"] = self.k_bin_centers_par_2D
                outfile[f"{cyl_avg}/k_bin_centers_perp_2D"] = self.k_bin_centers_perp_2D

                # Save power spectra going in to the transfer functions
                # Cylindrically averaged power spectra
                outfile[
                    f"{cyl_avg}/power_spectrum/pure_signal_2D"
                ] = self.ps_pure_signal_2D
                outfile[
                    f"{cyl_avg}/power_spectrum/pure_noise_2D"
                ] = self.ps_pure_noise_2D
                outfile[
                    f"{cyl_avg}/power_spectrum/injected_with_signal_2D"
                ] = self.ps_injected_with_signal_2D

            if "transfer_function_1D" in self.__dict__.keys():
                sph_avg = "spherically_averaged"
                # Save transfer function from spherically averaged power spectrum
                outfile[f"{sph_avg}/transfer_function_1D"] = self.transfer_function_1D
                outfile[f"{sph_avg}/k_bin_edges_1D"] = self.k_bin_edges_1D

                outfile[f"{sph_avg}/k_bin_centers_1D"] = self.k_bin_centers_1D

                # Save power spectra going in to the transfer functions
                # Spherically averaged power spectra
                outfile[
                    f"{sph_avg}/power_spectrum/pure_signal_1D"
                ] = self.ps_pure_signal_1D
                outfile[
                    f"{sph_avg}/power_spectrum/pure_noise_1D"
                ] = self.ps_pure_noise_1D
                outfile[
                    f"{sph_avg}/power_spectrum/injected_with_signal_1D"
                ] = self.ps_injected_with_signal_1D

    def read(self, inpath: Optional[str] = None):
        """Method that reads transfer function from HDF5 file

        Args:
            inpath (Optional[str], optional): Input path that to file should be used to read in data. Defaults to None.
        """

        if inpath is None:
            inpath = self.path

        with h5py.File(inpath, "r") as infile:
            if "transfer_function_2D" in self.__dict__.keys():
                cyl_avg = "cylindrically_averaged"
                # Read transfer function from cylindrically averaged power spectrum
                self.transfer_function_2D = infile[f"{cyl_avg}/transfer_function_2D"][
                    ()
                ]

                self.k_bin_edges_par_2D = infile[f"{cyl_avg}/k_bin_edges_par_2D"][()]
                self.k_bin_edges_perp_2D = infile[f"{cyl_avg}/k_bin_edges_perp_2D"][()]

                self.k_bin_centers_par_2D = infile[f"{cyl_avg}/k_bin_centers_par_2D"][
                    ()
                ]
                self.k_bin_centers_perp_2D = infile[f"{cyl_avg}/k_bin_centers_perp_2D"][
                    ()
                ]
                # Read power spectra going in to the transfer functions

                # Cylindrically averaged power spectra
                self.ps_pure_signal_2D = infile[
                    f"{cyl_avg}/power_spectrum/pure_signal_2D"
                ][()]
                self.ps_pure_noise_2D = infile[
                    f"{cyl_avg}/power_spectrum/pure_noise_2D"
                ][()]
                self.ps_injected_with_signal_2D = infile[
                    f"{cyl_avg}/power_spectrum/injected_with_signal_2D"
                ][()]

            if "transfer_function_1D" in self.__dict__.keys():
                sph_avg = "spherically_averaged"

                # Read transfer function from spherically averaged power spectrum
                self.transfer_function_1D = infile[f"{sph_avg}/transfer_function_1D"][
                    ()
                ]
                self.k_bin_edges_1D = infile[f"{sph_avg}/k_bin_edges_1D"][()]

                self.k_bin_centers_1D = infile[f"{sph_avg}/k_bin_centers_1D"][()]

                # Read power spectra going in to the transfer functions
                # Spherically averaged power spectra
                self.ps_pure_signal_1D = infile[
                    f"{sph_avg}/power_spectrum/pure_signal_1D"
                ][()]
                self.ps_pure_noise_1D = infile[
                    f"{sph_avg}/power_spectrum/pure_noise_1D"
                ][()]
                self.ps_injected_with_signal_1D = infile[
                    f"{sph_avg}/power_spectrum/injected_with_signal_1D"
                ][()]

    def __sub__(self, other: TransferFunction):
        """Magic method to compute arithmetic difference between two transfer function objects.

        NOTE: that the power spectra from self will be kept
        and that difference is only computed between transfer function attributes.

        Args:
            other (TransferFunction): Transfer function object that is subtracted from self

        Returns:
            _type_: Returning new instance of TransferFunction class
            containing difference between transfer functions
        """
        difference = self.__class__()

        if (
            "transfer_function_1D" in self.__dict__.keys()
            and "transfer_function_1D" in other.__dict__.keys()
        ):
            difference.transfer_function_1D = (
                self.transfer_function_1D - other.transfer_function_1D
            )

        if (
            "transfer_function_2D" in self.__dict__.keys()
            and "transfer_function_2D" in other.__dict__.keys()
        ):
            difference.transfer_function_2D = (
                self.transfer_function_2D - other.transfer_function_2D
            )

        return difference

    def __mul__(self, other: TransferFunction):
        """Magic method to compute product between two transfer function objects.

        NOTE: that the power spectra from self will be kept
        and that product is only computed between transfer function attributes.

        Args:
            other (TransferFunction): Transfer function object that is subtracted from self

        Returns:
            _type_: Returning new instance of TransferFunction class
            containing product of transfer functions
        """
        product = self.__class__()

        if (
            "transfer_function_1D" in self.__dict__.keys()
            and "transfer_function_1D" in other.__dict__.keys()
        ):
            product.transfer_function_1D = (
                self.transfer_function_1D * other.transfer_function_1D
            )

        if (
            "transfer_function_2D" in self.__dict__.keys()
            and "transfer_function_2D" in other.__dict__.keys()
        ):
            product.transfer_function_2D = (
                self.transfer_function_2D * other.transfer_function_2D
            )

        return product

    def __truediv__(self, other: TransferFunction):
        """Magic method to compute divition between two transfer function objects.

        NOTE: that the power spectra from self will be kept
        and that divition is only computed between transfer function attributes.

        Args:
            other (TransferFunction): Transfer function object that is subtracted from self

        Returns:
            _type_: Returning new instance of TransferFunction class
            containing divition of transfer functions
        """
        divition = self.__class__()

        if (
            "transfer_function_1D" in self.__dict__.keys()
            and "transfer_function_1D" in other.__dict__.keys()
        ):
            divition.transfer_function_1D = (
                self.transfer_function_1D / other.transfer_function_1D
            )

        if (
            "transfer_function_2D" in self.__dict__.keys()
            and "transfer_function_2D" in other.__dict__.keys()
        ):
            divition.transfer_function_2D = (
                self.transfer_function_2D / other.transfer_function_2D
            )

        return divition


if __name__ == "__main__":

    simpath = "/mn/stornext/d22/cmbco/comap/protodir/maps/co2_python_freq_april2022_small_sim_simcube.h5"
    mappath = "/mn/stornext/d22/cmbco/comap/protodir/maps/co2_python_freq_april2022_small_sim.h5"
    noisepath = (
        "/mn/stornext/d22/cmbco/comap/protodir/maps/co2_python_freq_april2022_small.h5"
    )

    # Plot PS and TF:
    outputplot = ""

    mappaths = [simpath, mappath, noisepath]
    outname = outputplot

    tf = TransferFunction(path=outname, mappaths=mappaths)
    tf.compute_transfer_function()

    simpath = "/mn/stornext/d22/cmbco/comap/protodir/maps/co2_python_poly_april2022_small_sim_simcube.h5"
    mappath = "/mn/stornext/d22/cmbco/comap/protodir/maps/co2_python_poly_april2022_small_sim.h5"
    noisepath = (
        "/mn/stornext/d22/cmbco/comap/protodir/maps/co2_python_poly_april2022_small.h5"
    )
    mappaths = [simpath, mappath, noisepath]

    tf2 = TransferFunction(path=outname, mappaths=mappaths)
    tf2.compute_transfer_function()

    # tf.write("test_tf.h5")

    # print("saved tf")

    # tf.read("test_tf.h5")
    # print("read tf")

    tf3 = tf2 - tf
    tf4 = tf2 * tf
    tf5 = tf2 / tf

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 3, figsize=(10, 10))

    ax[0, 0].imshow(
        tf.transfer_function_2D, cmap="CMRmap", vmin=0, vmax=1, origin="lower"
    )
    ax[0, 1].imshow(
        tf2.transfer_function_2D, cmap="CMRmap", vmin=0, vmax=1, origin="lower"
    )

    ax[1, 0].imshow(
        tf3.transfer_function_2D, cmap="RdBu_r", vmin=-1, vmax=1, origin="lower"
    )
    ax[1, 1].imshow(
        tf4.transfer_function_2D, cmap="CMRmap", vmin=0, vmax=1, origin="lower"
    )
    ax[1, 2].imshow(
        tf5.transfer_function_2D, cmap="CMRmap", vmin=0, vmax=1, origin="lower"
    )

    plt.show()
