#!/bin/bash

Help()
{   
    # Helper function printing usefull information about runscript's usage
    
    echo -e "\n======================================================================================================================================================"
    echo "Usage: $ ./run_tod2tf.sh [parameters/flags]" 
    echo -e "======================================================================================================================================================\n"
    echo "(-N | --MPIl2gen) Is command line argument which specifies the number of MPI processes to use for l2gen."
    echo "(-n | --MPItod2comap) Is command line argument which specifies the number of MPI processes to use for tod2comap."
    echo "(-M | --OMPl2gen) Is command line argument which specifies the number of Open MP threads to use for l2gen."
    echo "(-m | --OMPtod2comap) Is command line argument which specifies the number of Open MP threads to use for tod2comap."
    echo "(-p | --params) Specify path to params.txt to use for l2gen and tod2 WITH signal injection run."
    echo -e "(-P | --paramsl2) Specify path to params.txt to use for l2gen and tod2 WITHOUT signal injection run. \n If this is not given the simulation run will assume that the path for loading frequency mask is provided in simulation run parameter file."
    echo "(-e | --end2end) Produce power spectra and transfer functions."
    echo "(-a | --accept_mod) Run accept mod after making new level 2 dataset."
    echo "(-h | --Help) Help flag will print this message."
    echo -e "\n======================================================================================================================================================"
    echo -e "$ Run example: ./run_tod2tf.sh --MPIl2gen 4 --MPItod2comap 5 --OMPtod2comap 7 --params my_sim_run_params.txt --paramsl2 my_l2_run_params.txt"
    echo -e "======================================================================================================================================================\n"
}

# Pipeline root direc_roottory
pipeline_root_dir="$(cd ../;pwd)"

# Path to accept_mod.py 
accept_mod_path="/mn/stornext/d22/cmbco/comap/protodir/accept_mod"

# Path to cross-spectrum code
fpxs_path="/mn/stornext/d22/cmbco/comap/nils/COMAP_XS"

# Defualt use ony 1 process
MPI_NUM_PROC_L2GEN=1 
MPI_NUM_PROC_TOD2COMAP=1 

OMP_NUM_PROC_L2GEN=1 
OMP_NUM_PROC_TOD2COMAP=1 

# By default no new level 2 dataset (w/o signal) will be produces as we usually have these laying around
new_raw_data_run=false

# Run accept mod
run_accept=false

# Whether to run end-to-end test or not
run_end2end=false

# Process command line input
# Process command line input
ARGS=$(getopt -a --options N:n:M:m:p:P:F:eah --long "MPIl2gen:,MPItod2comap:,OMPl2gen:,OMPtod2comap:,paramsl2:,params:,machinefile:,e2e,accept_mod,help" -- "$@")
eval set -- "$ARGS"

while true; do
    case "$1" in 
        -N|--MPIl2gen) 
        MPI_NUM_PROC_L2GEN=$2
        shift;;

        -n|--MPItod2comap) 
        MPI_NUM_PROC_TOD2COMAP=$2
        shift;;

        -M|--OMPl2gen) 
        OMP_NUM_PROC_L2GEN=$2
        shift;;

        -m|--OMPtod2comap) 
        OMP_NUM_PROC_TOD2COMAP=$2
        shift;;

        -p|--params) 
        PARAMS=$2
        shift;;

        -P|--paramsl2) 
        PARAMSL2=$2
        new_raw_data_run=true
        shift;;

        -F|machinefile)
        MACHINEFILE=$1
        shift;;

        -e|--e2e) 
        run_end2end=true
        shift;;

        -a|--accept_mod) 
        run_accept=true
        shift;;

        -h|--help) 
        Help
        shift;;

        --)
        break;;
        
        *)
        shift;;
    esac
done

# If to produce new level 2 dataset to use as base for simulation run. NOTE: this needs seperate parameter file input.
if [[ "$new_raw_data_run" == "true" ]]
then
    export OMP_NUM_THREADS=$OMP_NUM_PROC_L2GEN;mpirun -bind-to none --machinefile $pipeline_root_dir/machinefile_18_19_20_21.txt -n $MPI_NUM_PROC_L2GEN  python $pipeline_root_dir/l2gen.py -p $PARAMSL2
    
    if [[ "$run_accept" == "true" ]]
    then
        python $accept_mod_path/accept_mod_python_old.py -p $PARAMSL2
    fi

    export OMP_NUM_THREADS=$OMP_NUM_PROC_TOD2COMAP;mpirun -bind-to none -n $MPI_NUM_PROC_TOD2COMAP  python $pipeline_root_dir/tod2comap/tod2comap.py -p $PARAMSL2
    
    if [[ "$run_end2end" == "true" ]]
    then
        # Compute feed-feed-cross-spectra for noise data only
        python $fpxs_path/xs_script.py -p $PARAMSL2
    fi
fi

# Perform simulation run
export OMP_NUM_THREADS=$OMP_NUM_PROC_L2GEN;mpirun -bind-to none --machinefile $pipeline_root_dir/machinefile_18_19_20_21.txt -n $MPI_NUM_PROC_L2GEN  python $pipeline_root_dir/l2gen.py -p $PARAMS

# Run mapmaker and save path to simcube map from mapmaker print-out to variable 
print_out="$(export OMP_NUM_THREADS=$OMP_NUM_PROC_TOD2COMAP;mpirun -bind-to none -n $MPI_NUM_PROC_TOD2COMAP  python $pipeline_root_dir/tod2comap/tod2comap.py -p $PARAMS)" 


if [[ "$run_end2end" == "true" ]]
then   
    cube_map_name="$(echo $print_out | grep -oP '[^\s]*simcube\.h5')"

    # Get filename and directory from cube map path
    cube_map_dir="$(dirname $cube_map_name)"
    cube_map_name="$(basename $cube_map_name)"

    # Remove "coX" and ".h5" from cube map name
    cube_map_name="$(echo $cube_map_name | cut -c 5-)"
    cube_map_name="$(echo $cube_map_name | cut -d "." -f 1)"

    # Compute feed-feed-cross-spectra for signal injected data
    python $fpxs_path/xs_script.py -p $PARAMS  

    # Compute feed-feed-cross-spectra for signal only data
    python $fpxs_path/xs_script.py -p $PARAMS --map_name $cube_map_name --map_dir $cube_map_dir

    # Compute transfer function from auto-spectra
    python $pipeline_root_dir/power_spectrum/comap2tf.py --param $PARAMSL2 --params_sim $PARAMS
    
fi

