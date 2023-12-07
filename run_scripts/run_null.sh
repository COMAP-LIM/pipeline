#!/bin/bash

for script in accept_mod/accept_mod.py tod2comap/tod2comap.py;
do 
    for param in param_co7_apr22_v4_very_large_rnd_no_rain.txt param_co7_apr22_v4_no_rain.txt param_co7_apr22_v4_null_no_rain.txt;
    do
        export OMP_NUM_THREADS=10; mpirun --machinefile machinefile_owl_25_26_27.txt python -u -m mpi4py ${script} -p /mn/stornext/d16/cmbco/comap/src/params/${param}
    done 
done


for script in pca_subtractor/clean_maps.py;
do 
    for param in param_co7_apr22_v4_very_large_rnd_no_rain.txt param_co7_apr22_v4_no_rain.txt param_co7_apr22_v4_null_no_rain.txt;
    do
        export OMP_NUM_THREADS=1; mpirun -n 1 python -u -m mpi4py ${script} -p /mn/stornext/d16/cmbco/comap/src/params/${param}
    done 
done

for script in power_spectrum/comap2fpxs.py;
do 
    for param in param_co7_apr22_v4_very_large_rnd_no_rain.txt param_co7_apr22_v4_no_rain.txt param_co7_apr22_v4_null_no_rain.txt;
    do
        export OMP_NUM_THREADS=1; mpirun -n 72 python -u -m mpi4py ${script} -p /mn/stornext/d16/cmbco/comap/src/params/${param}
    done 
done