import argparse


import os
import sys


from pca_subtractor import PCA_SubTractor


current = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current)
sys.path.append(parent_directory)

from tod2comap.COmap import COmap


def main():
    """
    Parsing the command line input.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inname", type=str, help="""Path to input map.""")

    parser.add_argument(
        "-o", "--outname", type=str, help="""Name of output map.""", default=None
    )

    parser.add_argument(
        "-r",
        "--rmsnorm",
        type=str,
        help="""Which normalistion to use before PCA decomposition.
        Choose between "approx", "sigma_wn" or "var". Default is "sigma_wn".""",
        default="sigma_wn",
    )

    parser.add_argument(
        "-a",
        "--approx_noise",
        help="""Whether to approximate noise weights by PCA to conserve outer product""",
        action="store_true",
    )

    parser.add_argument(
        "-m",
        "--maskrms",
        type=float,
        help="""Factor of mean of bottom 100 sigma_wn of each feed and frequency beyond which map is masked.""",
        default=None,
    )

    parser.add_argument(
        "-n",
        "--ncomps",
        type=int,
        help="""How many PCA modes to subtract from input map. Default is 5. """,
        default=5,
    )

    parser.add_argument(
        "-v",
        "--verbose",
        help="""Whether to run in verbose mode or not. Default is False""",
        action="store_true",
    )

    parser.add_argument(
        "-s",
        "--subtract_mean",
        help="""Whether to subtract line-of-sight mean per pixel prior to PCA. Default is False""",
        action="store_true",
    )

    parser.add_argument(
        "--save_reconstruction",
        help="""Whether to save PCA reconstrucion in its own map file(s). Default is False""",
        action="store_true",
    )

    # Defining argument object
    args = parser.parse_args()

    # Check if any arguments are missing or of invalid formate
    try:
        inpath = args.inname
    except:
        message = """Input map path is missing"""
        raise NameError(message)

    try:
        outpath = args.outname
    except:
        message = """Output map path is missing"""
        raise NameError(message)

    try:
        rmsnorm = args.rmsnorm
        assert rmsnorm in ["approx", "sigma_wn", "var", "three", "weightless"]
    except:
        message = """Please choose a normalisation to apply prior to PCA;  -n approx, -n rms or -n var."""
        raise NameError(message)

    try:
        ncomps = int(args.ncomps)
    except:
        message = (
            """Number of subtracted PCA modes invalid or missing. Must be integer."""
        )
        raise NameError(message)

    is_verbose = args.verbose

    subtract_mean = args.subtract_mean
    approx_noise = args.approx_noise

    maskrms = args.maskrms

    save_reconstruction = args.save_reconstruction

    # Define map object to process
    mymap = COmap(path=inpath)

    # Reading in map object from file
    mymap.read_map()

    # Define PCA subtractor object
    pca_sub = PCA_SubTractor(
        map=mymap,
        ncomps=ncomps,
        maskrms=maskrms,
        verbose=is_verbose,
        subtract_mean=subtract_mean,
        approx_noise=approx_noise,
    )

    # PCA mode subtracted cleaned map object
    mymap_clean = pca_sub.compute_pca(norm=rmsnorm)

    # Writing cleaned map data to file
    mymap_clean.write_map(outpath=outpath, save_fits=True, save_hdf5=True)

    if save_reconstruction:
        pca_sub.overwrite_maps_with_reconstruction()

        if outpath is None or len(outpath) == 0:
            outpath = mymap_clean.path.split(".h5")[0] + "_pca_reconstruction.h5"
        else:
            outpath = outpath.split(".h5")[0] + "_pca_reconstruction.h5"

        mymap_clean.write_map(outpath=outpath, save_fits=False, save_hdf5=True)


if __name__ == "__main__":
    main()
