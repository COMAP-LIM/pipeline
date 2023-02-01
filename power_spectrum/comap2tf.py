from transfer_function_utils import (
    get_1D_TF,
    get_2D_TF,
    PS_plotter,
    plot_two_TFs_and_diff,
)


# simpath = "/mn/stornext/d22/cmbco/comap/protodir/maps/co2_python_freq_april2022_small_sim_simcube.h5"
# mappath = (
#     "/mn/stornext/d22/cmbco/comap/protodir/maps/co2_python_freq_april2022_small_sim.h5"
# )
# noisepath = (
#     "/mn/stornext/d22/cmbco/comap/protodir/maps/co2_python_freq_april2022_small.h5"
# )

# # Plot PS and TF:
# outputplot = "/mn/stornext/d22/cmbco/comap/nils/COMAP_general/figs/tf_python_freqfilter_april2022_small.pdf"


# PS_plotter(simpath, mappath, noisepath, outname=outputplot)

simpath_1 = "/mn/stornext/d22/cmbco/comap/protodir/maps/co2_python_poly_april2022_small_sim_simcube.h5"
mappath_1 = (
    "/mn/stornext/d22/cmbco/comap/protodir/maps/co2_python_poly_april2022_small_sim.h5"
)
noisepath_1 = (
    "/mn/stornext/d22/cmbco/comap/protodir/maps/co2_python_poly_april2022_small.h5"
)

simpath_2 = "/mn/stornext/d22/cmbco/comap/protodir/maps/co2_python_freq_april2022_small_sim_simcube.h5"
mappath_2 = (
    "/mn/stornext/d22/cmbco/comap/protodir/maps/co2_python_freq_april2022_small_sim.h5"
)
noisepath_2 = (
    "/mn/stornext/d22/cmbco/comap/protodir/maps/co2_python_freq_april2022_small.h5"
)

# Plot PS and TF:
outputplot = "/mn/stornext/d22/cmbco/comap/nils/COMAP_general/figs/tf_python_poly_freqfilter_diff_april2022_small.pdf"


plot_two_TFs_and_diff(
    simpath_1, mappath_1, noisepath_1, simpath_2, mappath_2, noisepath_2, outputplot
)

"""
# Generate HDF5 file with TF(k) in 1D:

TF, k = get_1D_TF(simpath, mappath, noisepath)

dfile = h5py.File("TF_1d.h5", "a")

dfile.create_dataset("TF", data = TF)
dfile.create_dataset("k", data = k)
dfile.close()

# Generate HDF5 file with TF(k) in 1D:

TF, k = get_2D_TF(simpath, mappath, noisepath)

dfile = h5py.File("TF_2d.h5", "a")

dfile.create_dataset("TF", data = TF)
dfile.create_dataset("k", data = k)
dfile.close()
"""
