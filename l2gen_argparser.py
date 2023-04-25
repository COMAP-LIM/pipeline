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
    "--fields",
    type=str,
    default=["co2", "co6", "co7"],
    nargs="+",
    help="List of fields to include in run."
)
parser.add_argument(
    "--obsid_start",
    type=int,
    default=0,
    help="Earliest obsid to include."
)
parser.add_argument(
    "--obsid_stop",
    type=int,
    default=9999999,
    help="Last obsid to include."
)
parser.add_argument(
    "--runlist_split_in_n",
    type=int,
    default=1,
    help="Split the runlist into this number of random sub-parts. See '--runlist_split_num_i' for which of the parts this run contains. The split is deterministic for identical runlist."
)
parser.add_argument(
    "--runlist_split_num_i",
    type=int,
    default=0,
    help="Which of the N sub-runlists this run contains (zero-indexed), where N is specified by the '--runlist_split_in_n' argument."
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
    default=1,
    help="Time, in seconds, to cut at the end of each scan.",
)
parser.add_argument(
    "--min_allowed_scan_length",
    type=float,
    default=120.0,
    help="Minimum allowed length of scans, in seconds (applied when reading runlist).",
)
parser.add_argument(
    "--sbA_num_masked_channels",
    type=int,
    default=179,
    help="How many channels on each edge of sisdeband A to mask (they behave weirdly).",
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
    "--allowed_scantypes",
    type=int,
    default=[
        32,
    ],
    nargs="+",
    help="Scan types to include in run.",
)
parser.add_argument(
    "--included_feeds",
    type=int,
    default=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19],
    nargs="+",
    help="List of feeds to include."
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
    "--aliasing_mask_dB",
    type=float,
    default=15,
    help="(mask) Aliasing suppression threshold. Frequency channels below this is masked.",
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
    "--make_nhit",
    type=str2bool,
    default=True,
    help="(tod2comap) If True, hit maps are made. Default True.",
)

parser.add_argument(
    "--split",
    type=str2bool,
    default=True,
    help="(tod2comap) Parameter determains whether to perfor splits or not. Default True.",
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


parser.add_argument(
    "--t2m_rms_mask_factor",
    type=float,
    default=8,
    help="(tod2comap) If not negative, this parameter will mask high noise regions in map datasets. All regions with a higher noise than the bottom-100 freq-coadded noise per feed and split times the parameter factor will be masked out.",
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


parser.add_argument(
    "--split_base_number",
    type=int,
    default=2,
    help="(comap2fpxs) Base number to use for splitting the data. Only 2 supported for now. The data is split into N parts for each split variable.",
)

parser.add_argument(
    "--psx_map_names",
    type=str,
    nargs="+",
    default = [],
    help="(comap2fpxs) List of map filenames (not absolute path) to use for cross-field null tests.",
)

parser.add_argument(
    "--psx_null_cross_field",
    type=str2bool,
    default=False,
    help="(comap2fpxs) If True compute cross-field feed-feed pseudo cross-spectrum null tests. Default False."
)


parser.add_argument(
    "--psx_null_diffmap",
    type=str2bool,
    default=False,
    help="(comap2fpxs) If True compute difference map feed-feed pseudo cross-spectrum null tests. Default False."
)

parser.add_argument(
    "--psx_noise_sim_number",
    type=int,
    default=50,
    help="(comap2fpxs) Number of noise simulations to run to get power spectrum error bars.",
)

parser.add_argument(
    "--psx_number_of_k_bins",
    type=int,
    default=14,
    help="(comap2fpxs) Number of k bins (centers) to use for computing binned power spectra.",
)

parser.add_argument(
    "--psx_chi2_cut_limit",
    type=float,
    default=5,
    help="(comap2fpxs) Which chi2 to use when cutting bad cross-spectra.",
)


###### Physics ######
parser.add_argument(
    "--phy_center_redshift",
    type=float,
    default=2.9,
    help="(physics) Central redshift of CO maps. Default 2.9.",
)

parser.add_argument(
    "--phy_cosmology_dir",
    type=str,
    default="./cosmologies/",
    help="(physics) Path to pickled astropy cosmologies to be used. Defaults to repository directory containing defalt COMAP cosmology.",
)

parser.add_argument(
    "--phy_cosmology_name",
    type=str,
    default="default_comap_cosmology.pkl",
    help="(physics) Pickled astopy cosmology to be used. Defaults to cosmology used in Li et al. 2016 and Ihle et al. 2019.",
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

parser.add_argument(
    "--sim_verbose",
    type=str2bool,
    default=True,
    help="(SimGenerator) Verbose output while generating simulation cubes."
)

parser.add_argument(
    "--sim_halo_catalogue_file",
    type=str,
    default='/mn/stornext/d22/cmbco/comap/delaney/limlam_mocker/catalogues/COMAP_z2.39-3.44_1140Mpc_seed_13579.npz',
    help="(SimGenerator) Path to peak-patch simulation catalogue. Default seed 13579."
)

parser.add_argument(
    "--sim_mass_cutoff",
    type=float,
    default=500000000000.,
    help="(SimGenerator) Maximum DM mass to include in the simulated cube (in M_sun). Default 5e11 M_sun."
)

parser.add_argument(
    "--sim_min_mass",
    type=float,
    default=10000000000.,
    help="(SimGenerator) Minimum DM mass to include in the simulated cube (in M_sun). Default 1e10 M_sun."
)

parser.add_argument(
    "--sim_model",
    type=str,
    default="fiuducial",
    help="(SimGenerator) Name of model to use for halo CO luminosities. By default 'fiuducial' is used."
)

parser.add_argument(
    "--sim_model_coeffs",
    type=list,
    default=None,
    help="(SimGenerator) Adjusted model coefficients. If 'None', default values for the model are used."
)
parser.add_argument(
    "--sim_catalog_model",
    type=str,
    default='default',
    help="(SimGenerator) Name of the function used to model emission of the other tracer. Defaults to 'default' function in generate_luminosities.py."
)

parser.add_argument(
    "--sim_catalog_coeffs",
    type=str,
    default=None,
    help="(SimGenerator) Coefficients used for the emission modeling function. Defaults to None."
)

parser.add_argument(
    "--sim_catdex",
    type=float,
    default=0.5,
    help="(SimGenerator) Size of the artificial scatter in the tracer luminosities. Defaults to 0.5."
)

parser.add_argument(
    "--sim_codex",
    type=float,
    default=0.42,
    help="(SimGenerator) Size of the artificial scatter in the tracer luminosities. Defaults to 0.42 (chung22 fiducial value)."
)

parser.add_argument(
    "--sim_rho",
    type=float,
    default=0.8,
    help="(SimGenerator) Correlation between CO and tracer luminosities (-1, 1). Default to 0.8."
)

parser.add_argument(
    "--sim_lum_uncert_seed",
    type=int,
    default=12345,
    help="(SimGenerator) Seed for the RNG determining scatter in the halo luminosities. Default 12345."
)

parser.add_argument(
    "--sim_save_scatterless_lums",
    type=str2bool,
    default=True,
    help="(SimGenerator) Boolean: whether to keep the luminosity values calculated before scatter added. Defaults to True."
)

parser.add_argument(
    "--sim_cosmology",
    type=str,
    default='comap',
    help="(SimGenerator) The cosmological parameters to use in generating the simulations. Defaults to the values used in Li 2016."
)

parser.add_argument(
    "--sim_units",
    type=str,
    default='temperature',
    help="(SimGenerator) The brightness units used by the simulations. Defaults to 'intensity'."
)

parser.add_argument(
    "--sim_nmaps",
    type=int,
    default=1024,
    help="(SimGenerator) Number of frequency channels to include in the (final) simulation cube. Default 1024."
)

parser.add_argument(
    "--sim_npix_x",
    type=int,
    default=120,
    help="(SimGenerator) Number of pixels to include in the RA axis of the (final) simulation cube. Default 120."
)

parser.add_argument(
    "--sim_npix_y",
    type=int,
    default=120,
    help="(SimGenerator) Number of pixels to include in the Dec axis of the (final) simulation cube. Default 120."
)

parser.add_argument(
    "--sim_fov_x",
    type=float,
    default=4.,
    help="(SimGenerator) Size in the RA axis of the simulation cube (in degrees). Default 4.0 degrees."
)

parser.add_argument(
    "--sim_fov_y",
    type=float,
    default=4.,
    help="(SimGenerator) Size in the Dec axis of the simulation cube (in degrees). Default 4.0 degrees."
)

parser.add_argument(
    "--sim_nu_f",
    type=float,
    default=26.,
    help="(SimGenerator) Minimum frequency to include in the map (in GHz). Default 26.0."
)

parser.add_argument(
    "--sim_nu_i",
    type=float,
    default=34.,
    help="(SimGenerator) Maximum frequency to include in the map (in GHz). Default 34.0."
)

parser.add_argument(
    "--sim_nu_rest",
    type=float,
    default=115.27,
    help="(SimGenerator) Rest-frequency of the spectral line being modeled (in GHz). Default CO(1-0): 115.27 GHz."
)

parser.add_argument(
    "--sim_xrefine",
    type=int,
    default=5,
    help="(SimGenerator) Factor by which to oversample the angular axes of the simulations. Defaults to 5."
)

parser.add_argument(
    "--sim_freqrefine",
    type=int,
    default=5,
    help="(SimGenerator) Factor by which to oversample the frequency axes of the simulations. Defaults to 5."
)

parser.add_argument(
    "--sim_beambroaden",
    type=str2bool,
    default=True,
    help="(SimGenerator) Whether to smooth the angular axes by a 4.5' Gaussian approximation of the COMAP primary beam. Defaults to True."
)

parser.add_argument(
    "--sim_beamkernel",
    type=str,
    default=None,
    help="(SimGenerator) Convolution kernel approximating the primary beam. If none, use Gaussian with 4.5' FWHM."
)

parser.add_argument(
    "--sim_freqbroaden",
    type=str2bool,
    default=True,
    help="(SimGenerator) Whether to simulate astrophysical line broadening. Defaults to True."
)

parser.add_argument(
    "--sim_bincount",
    type=int,
    default=5,
    help="(SimGenerator) Number of mass bins to split simulated halos into before line broadening. Defaults to 5."
)

parser.add_argument(
    "--sim_fwhmfunction",
    type=str,
    default=None,
    help="(SimGenerator) Function used to calculate FWHMa for halos. If none (default), vvirsini is used."
)

parser.add_argument(
    "--sim_velocity_attr",
    type=str,
    default='vvirincli',
    help="(SimGenerator) Which type of per-halo velocity (stored as an attribute) to use when broadening. Default 'vvirincli'."
)

parser.add_argument(
    "--sim_lazyfilter",
    type=str2bool,
    default=True,
    help="(SimGenerator) Faster FFT when binning after line broadening. Defaults to True."
)

parser.add_argument(
    "--sim_output_dir",
    type=str,
    default='./simulations',
    help="(SimGenerator) Path to directory in which to store all the output simulation files." #*****
)

parser.add_argument(
    "--sim_map_output_file_name",
    type=str,
    default='sim_map.npz',
    help="(SimGenerator) File name for the final simulated map (.npz). Default sim_map.npz" #*****
)

parser.add_argument(
    "--sim_cat_output_file_name",
    type=str,
    default='sim_cat.npz',
    help="(SimGenerator) File name for the final simulated catalogue (.npz). Default sim_cat.npz" #*****
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
