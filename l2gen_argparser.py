import argparse

class LoadFromFile(argparse.Action):
    def __call__ (self, parser, namespace, values, option_string = None):
        with values as f:
            # parse arguments in the file and store them in the target namespace
            parser.parse_args(f.read().split(), namespace)


parser = argparse.ArgumentParser()


### Parameter file and runlist
parser.add_argument("--param_file",         type=open, action=LoadFromFile, help="Path to parameter file. File should have argparse syntax, and overwrites any value listed here.")
parser.add_argument("--runlist",            type=str,                       help="(REQUIRED) Path to runlist.")


### Paths and files
parser.add_argument("--level1_dir",         type=str,   default="/mn/stornext/d22/cmbco/comap/protodir/level1/",    help="Path to level1 files.")
parser.add_argument("--level2_dir",         type=str,   default="/mn/stornext/d22/cmbco/comap/protodir/level2/Ka/", help="Location of level2 files (made by l2gen).")
# parser.add_argument("--map_dir",            type=str,   default="/mn/stornext/d22/cmbco/comap/protodir/maps/",      help="(unused) Location of map files (made by tod2comap).")
parser.add_argument("--cal_database_file",  type=str,   default="/mn/stornext/d22/cmbco/comap/protodir/auxiliary/level1_database.h5", help="Location of calibration hdf5 database.")

parser.add_argument("--write_inter_files",  type=bool,  default=False,  help="Write intermediate level2 files after each filter.")


###### FILTER SETTINGS ######
### Gain normalization filter
parser.add_argument("--gain_norm_fknee",    type=float, default=0.01,   help="(norm) Knee freq of gain normalization.")
parser.add_argument("--gain_norm_alpha",    type=float, default=4.0,    help="(norm) PS slope of gain normalization.")

### Polynomial filter
# parser.add_argument("--polyorder",          type=int,   default=1,      help="(unused)(poly) Order of the frequency polynomial to be subtracted.")

### Frequency filter
parser.add_argument("--freqfilter_prior_file", type=str,   default="/mn/stornext/d22/cmbco/comap/protodir/auxiliary/Cf_prior.h5", help="(freq) Location of hdf5 file which contains sigma0, fknee and alpha for the freqfilter PS prior.")

### PCA filter
parser.add_argument("--n_pca_comp",         type=int,   default=4,      help="(pca) Number of feed-global PCA components to be subtracted.")

### PCA feed filter
# parser.add_argument("--n_feed_pca_comp",    type=int,   default=4,      help="[feedpca] Number of per-feed PCA components to be subtracted.")

### Masking
parser.add_argument("--box_sizes",          type=int,   default=[32, 128, 512],     nargs="+",  help="(mask) Size of masking boxes.")
parser.add_argument("--stripe_sizes",       type=int,   default=[32, 128, 1024],    nargs="+",  help="(mask) Size of masking stripes.")
parser.add_argument("--n_sigma_chi2_box",   type=float, default=[6.0, 6.0, 6.0],    nargs="+",  help="(mask) Sigma tolerance of chi2 box cuts.")
parser.add_argument("--n_sigma_chi2_stripe",type=float, default=[6.0, 6.0, 6.0],    nargs="+",  help="(mask) Sigma tolerance of chi2 stripe cuts.")
parser.add_argument("--n_sigma_mean_box",   type=float, default=[6.0, 10.0, 14.0],  nargs="+",  help="(mask) Sigma tolerance of mean box cuts.")
parser.add_argument("--n_sigma_prod_box",   type=float, default=[6.0, 5.0, 4.0],    nargs="+",  help="(mask) Sigma tolerance of product box cuts.")
parser.add_argument("--n_sigma_prod_stripe",type=float, default=[6.0, 5.0, 4.0],    nargs="+",  help="(mask) Sigma tolerance of product stripe cuts.")
parser.add_argument("--prod_offset",        type=int,   default=16, help="(mask) Offset length in box and stripe product test.")

### Decimation
parser.add_argument("--decimation_freqs",   type=int,   default=64,     help="(dec) Number of frequencies to decimate each sideband into, from the original 1024.")

### Tsys/Calibration
parser.add_argument("--max_tsys",           type=float, default=100.0,  help="(cal) Max tsys. Mask above this.")
parser.add_argument("--min_tsys",           type=float, default=0.0,    help="(cal) Min tsys. Mask below this.")