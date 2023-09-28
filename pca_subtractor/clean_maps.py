import argparse


import os
import sys


from pca_subtractor import PCA_SubTractor
from pca_subtractor_experimental import PCA_SubTractor_Experimental
from highpass import Highpass_filter_map


current = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current)
sys.path.append(parent_directory)

from tod2comap.COmap import COmap


def main():
    
    # Defining argument object
    from l2gen_argparser import parser
    args = parser.parse_args()


    # Check if any arguments are missing or of invalid formate
    if len(args.fields) > 1:
        raise ValueError("More than 1 field not currently supported by the mPCA.")
    else:
        field = args.fields[0]

    if not args.mpca_inname is None:
        inpath = args.mpca_inname
    else:
        if not args.map_name is None:
            inpath = os.path.join(args.map_dir, field + "_" + args.map_name + ".h5")
        else:
            raise NameError("Missing input map name.")

    outpath = args.mpca_outname

    try:
        rmsnorm = args.mpca_rmsnorm
        assert rmsnorm in ["approx", "sigma_wn", "var", "three", "weightless", "exper"]
    except:
        message = """Please choose a normalisation to apply prior to PCA;  -n approx, -n sigma_wn or -n var."""
        raise NameError(message)

    ncomps = int(args.mpca_ncomps)

    is_verbose = args.mpca_verbose

    if is_verbose:
        print(f"Subtracting n={ncomps} PCA modes from maps:")

    subtract_mean = args.mpca_subtract_mean
    approx_noise = args.mpca_approx_noise

    maskrms = args.mpca_maskrms

    save_reconstruction = args.mpca_save_reconstruction

    # Define map object to process
    mymap = COmap(path=inpath)

    # Reading in map object from file
    mymap.read_map()

    # Define PCA subtractor object
    if rmsnorm == "exper":
        pca_sub = PCA_SubTractor_Experimental(
            map=mymap,
            ncomps=ncomps,
            maskrms=maskrms,
            verbose=is_verbose,
            subtract_mean=subtract_mean,
            approx_noise=approx_noise,
        )
    else:
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

    # Highpass map
    if args.mpca_highpass:
        highpass = Highpass_filter_map(
            map=mymap_clean,
            verbose=is_verbose,
            Ncomp=args.mpca_highpass_Nmodes)
        mymap_clean = highpass.run()

    # Writing cleaned map data to file
    mymap_clean.write_map(outpath=outpath, save_fits=False, save_hdf5=True)

    if save_reconstruction:
        pca_sub.overwrite_maps_with_reconstruction()

        if approx_noise:
            approx = "_approx"
        else:
            approx = ""

        if outpath is None or len(outpath) == 0:
            new_outpath = mymap_clean.path.split(".h5")[0] + f"_pca_reconstruction_n{ncomps}{approx}_{rmsnorm}.h5"
        else:
            new_outpath = outpath.split(".h5")[0] + f"_pca_reconstruction_n{ncomps}{approx}_{rmsnorm}.h5"

        mymap_clean.write_map(outpath=new_outpath, save_fits=False, save_hdf5=True)

        # for i in range(ncomps):
        #     pca_sub.overwrite_maps_with_reconstruction(component=i)

        #     if outpath is None or len(outpath) == 0:
        #         new_outpath = mymap_clean.path.split(".h5")[0] + f"_pca_reconstruction_n{ncomps}_comp{i}.h5"
        #     else:
        #         new_outpath = outpath.split(".h5")[0] + f"_pca_reconstruction_n{ncomps}_comp{i}.h5"

        #     mymap_clean.write_map(outpath=new_outpath, save_fits=False, save_hdf5=True)

if __name__ == "__main__":
    main()
