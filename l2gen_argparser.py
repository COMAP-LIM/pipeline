import argparse


class LoadFromFile(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        with values as f:
            # parse arguments in the file and store them in the target namespace
            parser.parse_args(f.read().split(), namespace)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


parser = argparse.ArgumentParser()

parser.add_argument("-v", "--verbose", action="count", help="Enable verbose printing.")

### Parameter file and runlist
parser.add_argument(
    "-p",
    "--param",
    type=open,
    action=LoadFromFile,
    help="Path to parameter file. File should have argparse syntax, and overwrites any value listed here.",
)
parser.add_argument("--runlist", type=str, help="(REQUIRED) Path to runlist.")
parser.add_argument(
    "-f",
    "--filters",
    type=str,
    nargs="+",
    default=[
        "Tsys_calc",
        "Normalize_Gain",
        "Pointing_Template_Subtraction",
        "Masking",
        "Frequency_filter",
        "PCA_filter",
        "PCA_feed_filter",
        "Calibration",
        "Decimation",
    ],
    help="Name of each filter, in order, to include in the l2gen run.",
)
parser.add_argument(
    "--obsid_start", type=int, default=0, help="Earliest obsid to include."
)
parser.add_argument(
    "--obsid_stop", type=int, default=9999999, help="Last obsid to include."
)

### Paths and files
parser.add_argument(
    "--level1_dir",
    type=str,
    default="/mn/stornext/d22/cmbco/comap/protodir/level1/",
    help="Path to level1 files.",
)
parser.add_argument(
    "--level2_dir",
    type=str,
    default="/mn/stornext/d22/cmbco/comap/protodir/level2/Ka/",
    help="Location of level2 files (made by l2gen).",
)
parser.add_argument(
    "--log_dir",
    type=str,
    default="/mn/stornext/d22/cmbco/comap/jonas/l2gen_python/logs/",
    help="Path to outputed logs.",
)
parser.add_argument(
    "--map_dir",
    type=str,
    default="/mn/stornext/d22/cmbco/comap/protodir/maps/",
    help="Location of map files (made by tod2comap).",
)

parser.add_argument(
    "--map_name",
    type=str,
    default=None,
    help="Specific name of map file made by tod2comap. Full name will be '[FIELDNAME]_[map_name].h5'",
)

parser.add_argument(
    "--cal_database_file",
    type=str,
    default="/mn/stornext/d22/cmbco/comap/protodir/auxiliary/level1_database.h5",
    help="Location of calibration hdf5 database.",
)

parser.add_argument(
    "--scantypes",
    type=int,
    default=[16,],
    nargs="+",
    help="Scan types to include in run.",
)

parser.add_argument(
    "--write_inter_files",
    type=str2bool,
    default=False,
    help="Write intermediate level2 files after each filter.",
)

parser.add_argument(
    "--distributed_starting",
    type=str2bool,
    default=False,
    help="Include a 30 seconds delay in between starting mpi processes, for better initial workload distribution.",
)


###### FILTER SETTINGS ######
### Gain normalization filter
parser.add_argument(
    "--gain_norm_fknee",
    type=float,
    default=0.01,
    help="(norm) Knee freq of gain normalization.",
)
parser.add_argument(
    "--gain_norm_alpha",
    type=float,
    default=4.0,
    help="(norm) PS slope of gain normalization.",
)

### Polynomial filter
# parser.add_argument("--polyorder",          type=int,   default=1,      help="(unused)(poly) Order of the frequency polynomial to be subtracted.")

### Frequency filter
parser.add_argument(
    "--freqfilter_use_prior",
    type=str2bool,
    default=True,
    help="(freq) Whether to use a prior on the gain term in the frequency filter.",
)
parser.add_argument(
    "--freqfilter_prior_file",
    type=str,
    default="/mn/stornext/d22/cmbco/comap/protodir/auxiliary/Cf_prior.h5",
    help="(freq) Location of hdf5 file which contains sigma0, fknee and alpha for the freqfilter PS prior.",
)

parser.add_argument(
    "--freqfilter_exclude_ends",
    type=str2bool,
    default=True,
    help="(freq) Exclude the first 4 and the last 100 frequency channels from freqfilter fits.",
)


### PCA filter
parser.add_argument(
    "--n_pca_comp",
    type=int,
    default=4,
    help="(pca) Number of feed-global PCA components to be subtracted.",
)

### PCA feed filter
# parser.add_argument("--n_feed_pca_comp",    type=int,   default=4,      help="[feedpca] Number of per-feed PCA components to be subtracted.")

### Masking
parser.add_argument(
    "--box_sizes",
    type=int,
    default=[32, 128, 512],
    nargs="+",
    help="(mask) Size of masking boxes.",
)
parser.add_argument(
    "--stripe_sizes",
    type=int,
    default=[32, 128, 1024],
    nargs="+",
    help="(mask) Size of masking stripes.",
)
parser.add_argument(
    "--n_sigma_chi2_box",
    type=float,
    default=[6.0, 6.0, 6.0],
    nargs="+",
    help="(mask) Sigma tolerance of chi2 box cuts.",
)
parser.add_argument(
    "--n_sigma_chi2_stripe",
    type=float,
    default=[6.0, 6.0, 6.0],
    nargs="+",
    help="(mask) Sigma tolerance of chi2 stripe cuts.",
)
parser.add_argument(
    "--n_sigma_mean_box",
    type=float,
    default=[6.0, 10.0, 14.0],
    nargs="+",
    help="(mask) Sigma tolerance of mean box cuts.",
)
parser.add_argument(
    "--n_sigma_prod_box",
    type=float,
    default=[6.0, 5.0, 4.0],
    nargs="+",
    help="(mask) Sigma tolerance of product box cuts.",
)
parser.add_argument(
    "--n_sigma_prod_stripe",
    type=float,
    default=[6.0, 5.0, 4.0],
    nargs="+",
    help="(mask) Sigma tolerance of product stripe cuts.",
)
parser.add_argument(
    "--prod_offset",
    type=int,
    default=16,
    help="(mask) Offset length in box and stripe product test.",
)
parser.add_argument(
    "--write_C_matrix",
    type=str2bool,
    default=False,
    help="(mask) Whether to write corr-matrix (and template) to file. Warning: It's big, do not use for large runs.",
)

### Decimation
parser.add_argument(
    "--decimation_freqs",
    type=int,
    default=64,
    help="(dec) Number of frequencies to decimate each sideband into, from the original 1024.",
)

### Tsys/Calibration
parser.add_argument(
    "--max_tsys", type=float, default=100.0, help="(cal) Max tsys. Mask above this."
)
parser.add_argument(
    "--min_tsys", type=float, default=0.0, help="(cal) Min tsys. Mask below this."
)



###### ACCEPT MOD ######
### Please do set ###
parser.add_argument("--accept_data_id_string", type=str, default="")
parser.add_argument("--jk_data_string", type=str, default="")
parser.add_argument("--scan_stats_from_file", type=str2bool, default=False)
parser.add_argument("--jk_def_file", type=str, default="/mn/stornext/d22/cmbco/comap/protodir/auxiliary/jk_lists/jk_list_only_elev.txt")
parser.add_argument("--show_accept_plot", type=str2bool, default=True)

### Defaults ###
parser.add_argument("--stats_list", type=str, default="stats_list.py")
parser.add_argument("--accept_param_folder", type=str, default="/mn/stornext/d22/cmbco/comap/protodir/accept_mod/")
parser.add_argument("--accept_mod_params", type=str, default="accept_params.py")
parser.add_argument("--patch_definition_file", type=str, default="/mn/stornext/d22/cmbco/comap/protodir/auxiliary/patches_celestial.txt")
parser.add_argument("--weather_filepath", type=str, default="/mn/stornext/d22/cmbco/comap/protodir/auxiliary/weather_list.txt")
parser.add_argument("--accept_data_folder", type=str, default="/mn/stornext/d22/cmbco/comap/protodir/auxiliary/scan_data/")


###### MAPMAKER ######
### Field information and grid resolutions
parser.add_argument("--grid_size", type=int, default=(120, 120))
parser.add_argument("--grid_res", type=float, default=[2 / 60, 2 / 60])  # in deg
parser.add_argument(
    "--field_center",
    type=float,
    default={"co2": [25.435, 0.000], "co6": [226.00, 55.00], "co7": [170.00, 52.50]},
)  # in deg

parser.add_argument(
    "--make_nhit",
    action="store_true",
    help="If flag is provided hit maps are made. By default no hit maps are made.",
)

parser.add_argument(
    "--split",
    action="store_true",
    help="If flag is provided split maps are computed.",
)

parser.add_argument(
    "--split_def",
    type=str,
    default="/mn/stornext/d22/cmbco/comap/protodir/auxiliary/jk_lists/jk_list_only_elev.txt",
    help="Split definition file in which split bit order and names are defined.",
)

parser.add_argument(
    "--accept_dir",
    type=str,
    default="/mn/stornext/d22/cmbco/comap/protodir/auxiliary/scan_data/",
    help="Direcotry to which accept mod data file are saved",
)

parser.add_argument(
    "--scan_data",
    type=str,
    help="Name of accept mod generated scan_data file.",
)

parser.add_argument(
    "--split_data",
    type=str,
    help="Name of accept mod generated jk_data file.",
)
