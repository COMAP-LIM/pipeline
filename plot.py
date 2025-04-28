import warnings
import h5py 
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import rfft, irfft


def waterfall(signal, Ntime,  title, save, ylabel = "# Frequency sample", xlabel = "# Time sample", shape = 4*128, colormap = "RdBu_r", minmax = False):#, vmin = '-np.nanpercentile(np.abs(signal.flatten()), 98)', vmax = 'np.nanpercentile(np.abs(signal.flatten()), 98)'):
    
    # print(f'Signal shape is {signal.shape}, reshaping to shape {shape} for Ntime {Ntime} ')
    
    fig, ax = plt.subplots(figsize = (8, 5))
    ax.set_title(title, pad = 20)

    
    img = ax.imshow(
        (signal).reshape(shape, Ntime),
        cmap = colormap,
        interpolation = "none",
        aspect = "auto",
        vmin = (np.min(signal)),
        vmax = (np.max(signal)),
        # vmin = np.nanpercentile(np.abs(signal.flatten()), 2),
        # vmax = np.nanpercentile(np.abs(signal.flatten()), 98),
        # vmin = -np.nanpercentile(np.abs(signal.flatten()), 98),
        # vmax = np.nanpercentile(np.abs(signal.flatten()), 98),
        )
    cbar = plt.colorbar(img, ax=ax)
    cbar.set_label("K", fontsize=11, rotation=90, labelpad=13)
    # ax.axvline(x=10000, color='black', linestyle='-', linewidth=2)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    plt.savefig(save)
    
  
    if minmax:
        # return img.get_array(),[ -np.nanpercentile(np.abs(signal.flatten()), 98), np.nanpercentile(np.abs(signal.flatten()), 98) ]
        return img.get_array(),[ -np.abs(np.max(signal.flatten())),np.abs(np.max(signal.flatten())) ]
    return img.get_array()


def map_colormap(data, title, save, ylabel = 'Pixel Y', xlabel = 'Pixel X', Ndim = 120, colormap = "seismic", cmap_title ='Signal intensity'):

    if data.size != Ndim * Ndim:
        raise ValueError(f"Input data cannot be reshaped to ({Ndim}, {Ndim}). Check the size of the data.")
 

    im = plt.imshow(data.reshape(Ndim, Ndim), cmap = colormap, interpolation='none', vmin=np.min(data), vmax=np.max(data))
    cbar = plt.colorbar(im)
    cbar.set_label(cmap_title)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(save)

    return im


import matplotlib.pyplot as plt
import numpy as np

def subplot_waterfalls(images, titles, minmax=None, ylabel="# Frequency sample", xlabel="# Time sample", colormap="RdBu_r", save=None, n_cols=4, n_rows=1):
    """
    Creates a subplot from pre-generated images by the waterfall function.

    """
    n_plots = len(images)

    # if n_plots < 4:
    #     n_cols = n_plots
    #     n_rows = 1
    
    # else:
    #     n_cols =  (n_plots + 1) // 2
    #     n_rows = (n_plots + 1) // 2

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axs = np.atleast_1d(axs).flatten()  # Flatten for easy indexing

    for i, (img_data, title) in enumerate(zip(images, titles)):
        
        if minmax[i]:
            vmin, vmax = minmax[i] 
        else:
            vmin = None
            vmax = None

        # Plot each image in the respective subplot
        img = axs[i].imshow(img_data, cmap=colormap, aspect="auto", vmin=vmin, vmax=vmax, interpolation = "none")
        axs[i].set_title(title)

        # Same xlabel and ylabel for all subplots
        axs[i].set_xlabel(xlabel)
        axs[i].set_ylabel(ylabel)
        
        # Add color bar for each subplot
        cbar = fig.colorbar(img, ax=axs[i], fraction=0.046, pad=0.04)
        cbar.set_label('K')


    # Turn off extra plots  # *** This doesnt work yet
    if len(axs)>len(titles):
        for j in range(len(axs)-len(titles)):
            axs[j].axis('off')



    # Adjust layout for tight spacing
    plt.tight_layout()
    
    # Save figure if a path is provided
    if save:
        plt.savefig(save)

    if save == None:
        plt.show()
        


