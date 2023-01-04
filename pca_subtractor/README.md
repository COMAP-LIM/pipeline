# PCA subtractor tool
The PCA subtractor tool will fit and subtract the `n` leading PCA modes for all feed maps. 

## Usage

```
usage: clean_maps.py [-h] [-i INNAME] [-o OUTNAME] [-r RMSNORM] [-m MASKRMS] [-n NCOMPS] [-v]

optional arguments:
  -h, --help            show this help message and exit
  -i INNAME, --inname INNAME
                        Path to input map.
  -o OUTNAME, --outname OUTNAME
                        Name of output map.
  -r RMSNORM, --rmsnorm RMSNORM
                        Which normalistion to use before PCA decomposition. Choose between "approx", "rms" or "var". Default is "rms".
  -m MASKRMS, --maskrms MASKRMS
                        Value of RMS value (in muK) beyond which to mask maps prior to coadding together feed maps.
  -n NCOMPS, --ncomps NCOMPS
                        How many PCA modes to subtract from input map. Default is 5.
  -v, --verbose         Whether to run in verbose mode or not. Default is True
  ```