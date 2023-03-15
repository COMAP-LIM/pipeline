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

parser.add_argument(
    "--time_start_cut",
    type=int,
    default=0,
    help="Time, in seconds, to cut at the beginning of each scan.",
)
parser.add_argument(
    "--time_stop_cut",
    type=int,
    default=0,
    help="Time, in seconds, to cut at the beginning of each scan.",
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
    default=[
        16,
    ],
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

parser.add_argument(
    "--use_l2_compression",
    type=str2bool,
    default=True,
    help="Use hdf5 GZIP compression when writing the level2 tod data to file."
)


###### FILTER SETTINGS ######
### Start-of-scan exponential subtraction filter
parser.add_argument(
    "--start_exponential_decay_time",
    type=float,
    default=19.2,
    help="(start_exp) Decay time ('mean lifetime') of exponential fitted and subtracted at the start of scans."
)


### Azimuth edge masking filter
parser.add_argument(
    "--az_edges_mask_size_before",
    type=int,
    default=25,
    help="(az-mask) How many TOD time samples to mask at the azimuth extremes, before turnaround."
)
parser.add_argument(
    "--az_edges_mask_size_after",
    type=int,
    default=25,
    help="(az-mask) How many TOD time samples to mask at the azimuth extremes, after turnaround."
)

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

parser.add_argument(
    "--gain_norm_gauss_sigma_seconds",
    type=int,
    default=16,
    
)

### Polynomial filter
# parser.add_argument("--polyorder",          type=int,   default=1,      help="(unused)(poly) Order of the frequency polynomial to be subtracted.")

### Frequency filter
parser.add_argument(
    "--freqfilter_use_prior",
    type=str2bool,
    default=False,
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

parser.add_argument(
    "--freqfilter_full_feed",
    type=str2bool,
    default=False,
    help="(freq) Fit the templates across the entire feed, as opposed to for each sideband.",
)


### PCA filter
parser.add_argument(
    "--max_pca_comp",
    type=int,
    default=12,
    help="(pca) Number of feed-global PCA components to be subtracted.",
)

parser.add_argument(
    "--pca_max_iter",
    type=int,
    default=20,
    help="(pca) Max number of power iterations used to solve for PCA.",
)

parser.add_argument(
    "--pca_error_tol",
    type=float,
    default=1e-12,
    help="(pca) Error toleranse (|r - s/lamb|/n) when using power iterations to solve for PCA.",
)


### PCA feed filter
# parser.add_argument("--n_feed_pca_comp",    type=int,   default=4,      help="[feedpca] Number of per-feed PCA components to be subtracted.")

### Masking
parser.add_argument(
    "--load_freqmask_path",
    type=str,
    default="",
    help="(mask) Path to level2 files from which to load freqmasks instead of computing masks. If empty, will compute masking as usual.",
)
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
    "--max_tsys", type=float, default=75.0, help="(cal) Max tsys. Mask above this."
)
parser.add_argument(
    "--min_tsys", type=float, default=25.0, help="(cal) Min tsys. Mask below this."
)
parser.add_argument(
    "--median_tsys_cut",
    type=float,
    default=5.0,
    help="(cal) How many Kelvin above the running median Tsys value to mask.",
)


###### ACCEPT MOD ######
### Please do set ###
parser.add_argument("--accept_data_id_string", type=str, default="")
parser.add_argument("--jk_data_string", type=str, default="")
parser.add_argument("--scan_stats_from_file", type=str2bool, default=False)
parser.add_argument(
    "--jk_def_file",
    type=str,
    default="/mn/stornext/d22/cmbco/comap/protodir/auxiliary/jk_lists/jk_list_only_elev.txt",
)
parser.add_argument("--show_accept_plot", type=str2bool, default=True)

### Defaults ###
parser.add_argument("--stats_list", type=str, default="stats_list.py")
parser.add_argument(
    "--accept_param_folder",
    type=str,
    default="/mn/stornext/d22/cmbco/comap/protodir/accept_mod/",
)
parser.add_argument("--accept_mod_params", type=str, default="accept_params.py")
parser.add_argument(
    "--patch_definition_file",
    type=str,
    default="/mn/stornext/d22/cmbco/comap/protodir/auxiliary/patches_celestial.txt",
)
parser.add_argument(
    "--weather_filepath",
    type=str,
    default="/mn/stornext/d22/cmbco/comap/protodir/auxiliary/weather_list.txt",
)
parser.add_argument(
    "--accept_data_folder",
    type=str,
    default="/mn/stornext/d22/cmbco/comap/protodir/auxiliary/scan_data/",
)


###### MAPMAKER ######
### Field information and grid resolutions
parser.add_argument(
    "--res_factor",
    type=float,
    default=1,
    help="(tod2comap) Resolution factor. Default value 1 gives 2 arcmin pixels; value 2 gives 1 arcmin pixels; value 0.5 gives 4 arcmin pixels; etc. Can also be used to upgrade/downgrade simulation cubes",
)

parser.add_argument(
    "--make_no_nhit",
    action="store_false",
    help="(tod2comap) If flag is provided hit maps are made. By default hit maps are made.",
)

parser.add_argument(
    "--split",
    action="store_true",
    help="(tod2comap) If flag is provided split maps are computed.",
)

parser.add_argument(
    "--temporal_mask",
    action="store_true",
    help="(tod2comap) If flag is provided temporal masking, excluding turn around points in azimuth and pathologically large fluctuations in elevation.",
)

parser.add_argument(
    "--az_mask_percentile",
    type=float,
    default=90,
    help="(tod2comap) Number of datapoints to cut at start of scan when creating runlist. Must be between 0 and 100",
)

parser.add_argument(
    "--el_mask_cut",
    type=float,
    default=2.5e-3,
    help="(tod2comap) Mask all points in time where elevation is larger og smaller respectively than median(elevation) ± el_mask_cut. Value must be in degrees.",
)

parser.add_argument(
    "--no_hdf5",
    action="store_true",
    help="(tod2comap) If flag is provided no maps are saved as HDF5 file.",
)

parser.add_argument(
    "--no_fits",
    action="store_true",
    help="(tod2comap) If flag is provided no maps are saved as fits files.",
)

parser.add_argument(
    "--horizontal",
    action="store_true",
    help="(tod2comap) If flag is provided, maps in horizontal coordinates are made.",
)

parser.add_argument(
    "--directional",
    action="store_true",
    help="(tod2comap) If flag is provided, maps for right and left moving azimuth.",
)

parser.add_argument(
    "--override_accept",
    action="store_true",
    help="(tod2comap) If flag is provided, accept list masking is ignored.",
)

parser.add_argument(
    "--drop_first_scans",
    action="store_true",
    help="(tod2comap) If flag is provided, all scans with id ...02 are discarded.",
)

parser.add_argument(
    "--temporal_chunking",
    type=int,
    default=0,
    help="(tod2comap) Number of obsIDs to chunck in temporal chunking runs. If default 0 is used, no temporal chunking is performed.",
)

#### Scan detect stuff ####
parser.add_argument(
    "--scandetect_cut_start",
    type=int,
    default=10,
    help="(scan_detect) Number of datapoints to cut at start of scan when creating runlist.",
)

parser.add_argument(
    "--scandetect_cut_end",
    type=int,
    default=50,
    help="(scan_detect) Number of datapoints to cut at end of scan when creating runlist.",
)

parser.add_argument(
    "--scandetect_minimum_scan_length",
    type=float,
    default=2.0,
    help="(scan_detect) Minimum allowed length of scans, in minutes, when creating runlist.",
)


parser.add_argument(
    "--ces_only",
    type=str2bool,
    default=True,
    help="(scan_detect) Use only CES scans when creating runlist.",
)

###### Cross-Spectrum stuff ######
parser.add_argument(
    "--tf_cutoff",
    type=float,
    default=0.2,
    help="(comap2fpxs) Value of transfer function above which to compute chi2 in feed-feed pseudo cross-spectra.",
)

parser.add_argument(
    "--from_file",
    action="store_true",
    help="(comap2fpxs) If flag is provided already computed spectra are read from file.",
)

parser.add_argument(
    "--power_spectrum_dir",
    type=str,
    default="/mn/stornext/d22/cmbco/comap/protodir/power_spectrum/fpxs/",
    help="(comap2fpxs) Path to directory where cross spectrum data is saved.",
)

###### Make Signal Cube ######
parser.add_argument(
    "--model_name",
    type=str,
    default="",
    help="(make_cube) Name of model to use. By defualt 'power_cov' is used.",
)

parser.add_argument(
    "--exp_params",
    type=str,
    default="experimental_parameters_sim2tod.py",
    help="File name (without path) to experimental parameters to use to make simulation cube.",
)

###### Signal Injection ######
parser.add_argument(
    "--signal_path",
    type=str,
    default=None,
    help="Complete path to signal cube to use for singal injection. Should be HDF5 file.",
)

parser.add_argument(
    "--boost_factor",
    type=float,
    default=1,
    help="Factor to multiply with simulation to be injected into TOD in simulation pipeline.",
)

parser.add_argument(
    "--populate_cube",
    action="store_true",
    help="(tod2comap/signal injection) If flag is provided the simulation (only) cube needed to compute TF, with same sigma_wn and footprint as map with signal injected data, is produced.",
)

parser.add_argument(
    "--seed",
    type=int,
    default=None,
    help="Seed to use when producing simulation cubes.",
)


parser.add_argument(
    "--transfer_function_dir",
    type=str,
    default=None,
    help="Path transfer function directory.",
)

parser.add_argument(
    "--transfer_function_name",
    type=str,
    default=None,
    help="(run_tod2tf) Specific name of transfer function file made by run_tod2tf. Full name will be '[FIELDNAME]_[transfer_function_name].h5'",
)


parser.add_argument(
    "--main_dir_l2",
    type=str,
    default="",
    help="(run_tod2tf) Complete path to pure data level 2 files. By default string is empty. If default is used a new level 2 set is generated.",
)

parser.add_argument(
    "--sim_dir_l2",
    type=str,
    default="",
    help="(run_tod2tf) Complete path to signal injected data level 2 files. By default string is empty. If default is used a new signal injected level 2 set is generated.",
)

parser.add_argument(
    "--main_dir_map",
    type=str,
    default="",
    help="(run_tod2tf) Complete path to pure data map file. By default string is empty. If default is used a new pure data map is generated.",
)

parser.add_argument(
    "--sim_dir_map",
    type=str,
    default="",
    help="(run_tod2tf) Complete path to signal injected data map file. By default string is empty. If default is used a new signal injected map and populated cube are generated.",
)
