# Bash script which runs l2gen, accept_mod, mapmaker, cross-spectrum, map-pca, and then cross-spectrum on the pca subtracted maps.
# Takes two arguments, a machinefile and a parameter file.
# Use example:
# bash bash_pipeline_runner.sh machinefile_owls.txt /mn/stornext/d22/cmbco/comap/protodir/params/param_co7_apr22_v2.txt
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 mpirun --machinefile $1 python3 -u -m mpi4py l2gen.py -p $2 &&\
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 mpirun --machinefile $1 python3 -u -m mpi4py accept_mod/accept_mod.py -p $2 &&\
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 mpirun --machinefile $1 python3 -u -m mpi4py tod2comap/tod2comap.py -p $2 &&\
python3 -u pca_subtractor/clean_maps.py -p $2 &&\
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 mpirun --machinefile $1 python3 -u -m mpi4py power_spectrum/comap2fpxs.py -p $2 &&\
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 mpirun --machinefile $1 python3 -u -m mpi4py power_spectrum/comap2fpxs.py -p $2 --psx_map_name_postfix _n5_subtr_sigma_wn