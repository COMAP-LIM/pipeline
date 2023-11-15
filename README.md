# 1. Usage

## 1.1 Command-line usage and argparse
l2gen.py uses argparse for arguments (run `python3 l2gen.py --help` for all options, or have a look in the l2gen_argparse.py file).
Example usage:
```
python l2gen.py --obsid_start 10000 --runlist my_runlist.txt --write_inter_files True
```

However, it's easier to add these parameters to a *parameter file*, using the following syntax:
```
python l2gen.py --params param_file.txt
```
Example of a file containing the same three parameters as above (sadly no comments allowed):
```
--obsid_start 10000
--runlist my_runlist.txt
--write_inter_files True
```


## 1.2 MPI
l2gen.py support MPI, in the excepted way:
```
mpirun -n 16 python l2gen.py --param param_file.txt
```
The memory requirement is about 2-3 times the size of a scan, per process.

To run on multiple nodes, use machinefiles, in the usual way:
```
mpirun --machinefile machinefile.txt python l2gen.py --param param_file.txt
```
machinefile.txt looking like:
```
owl18:16
owl19:16
owl20:16
```


# 2. Code and Development
## 2.1 Code overview
* **l2gen.py** - The main code. Contains the class `l2gen_runner`, which reads the parameters and runlist, spawns MPI processes, and creates a series of instances of the `l2gen` class, each processing a single scan.
* **l2gen_argparse.py** - File containing all argparsing. Imported by the `l2gen_runner` class.
* **l2gen_l2class** - Contains the `level2_file` class, which is the data object holding all level2 data. It also reads level1 files from disk, and writes level2 files to disk.
* **l2gen_filters.py** - The file where all the filters used by the `l2gen` class are stored. These filters take a `level2_file` class object as input, and modifies it in different ways.


## 2.2 Creating a new filter
...

## 2.3 Generating a simulation cube
```
import l2_simulations as sims
from l2gen_argparser import parser

# read in default parameters
params = parser.parse_args()
# probably want to change:
params.sim_npix_x = 120 # number of pixels in the output map, in the angular x direction (RA)
params.sim_npix_y = 120 # number of pixels in the output map, in the angular y direction (Dec)
params.sim_nmaps = 1024 # number of frequency channels in the output map
params.sim_output_dir = './simulations' # directory in which to save the output simulation files

# set up generator object
simgen = sims.SimGenerator(params)

# run
simgen.run()
```


# 3. TODO
- [ ] **Signal injection simulation filter**: Filter which reads a simulation cube, and insertes it into the TOD.
- [ ] **PCA optimizations**: Currently using SKLearn PCA filter, which isn't parallelized. Want a Ctypes module which calculates and removes a single PCA component.
- [ ] **Adaptive number of PCA modes**: Instead of a static number of PCA modes, calculate and remove modes until some criteria is reached (eigenvalue threshold, probably).
- [ ] **Finish frequency filter**: The implemented frequency filter does not seem to be working optimally, and some debuging is required.
