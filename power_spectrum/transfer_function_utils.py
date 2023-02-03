import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
import matplotlib.colors as colors
import matplotlib as mpl

import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current)

sys.path.append(parent_directory)
sys.path.append(os.path.join(parent_directory, "power_spectrum"))

import map_cosmo as map_cosmo
import power_spectrum as power_spectrum

from sklearn.decomposition import PCA


fonts = {
    "font.family": "serif",
    "axes.labelsize": 20,
    "font.size": 20,
    "legend.fontsize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
}
plt.rcParams.update(fonts)


def plotPS(path, outname=None):
    # Raw cube
    # Raw cube

    fonts = {
        "font.family": "serif",
        "axes.labelsize": 14,
        "font.size": 14,
        "legend.fontsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    }
    plt.rcParams.update(fonts)

    inmap = map_cosmo.MapCosmo(path)

    P_in = power_spectrum.PowerSpectrum(inmap)

    P_in2d, k_in, nmodes_in = P_in.calculate_ps(do_2d=True)

    fig0, ax0 = plt.subplots(figsize=(5, 5))
    cmap = "CMRmap"

    img0 = ax0.imshow(
        np.log10(P_in2d),
        interpolation="none",
        origin="lower",
        extent=[0, 1, 0, 1],
        cmap=cmap,
    )  # ,
    # vmin = 4, vmax = 12, rasterized = True)

    # fig_list = [fig0, fig1, fig2, fig3]
    # ax_list  = [ax0, ax1, ax2, ax3]
    # img_list  = [[img0, img1], [img2, img3]]

    def log2lin(x, k_edges):
        loglen = np.log10(k_edges[-1]) - np.log10(k_edges[0])
        logx = np.log10(x) - np.log10(k_edges[0])
        return logx / loglen

    # ax.set_xscale('log')
    minorticks = [
        0.0002,
        0.0003,
        0.0004,
        0.0005,
        0.0006,
        0.0007,
        0.0008,
        0.0009,
        0.002,
        0.003,
        0.004,
        0.005,
        0.006,
        0.007,
        0.008,
        0.009,
        0.02,
        0.03,
        0.04,
        0.05,
        0.06,
        0.07,
        0.08,
        0.09,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        20.0,
        30.0,
        40.0,
        50.0,
        60.0,
        70.0,
        80.0,
        90.0,
        200.0,
        300.0,
        400.0,
        500.0,
        600.0,
        700.0,
        800.0,
        900.0,
    ]

    majorticks = [1.0e-04, 1.0e-03, 1.0e-02, 1.0e-01, 1.0e00, 1.0e01, 1.0e02]
    majorlabels = [
        "$10^{-4}$",
        "$10^{-3}$",
        "$10^{-2}$",
        "$10^{-1}$",
        "$10^{0}$",
        "$10^{1}$",
        "$10^{2}$",
    ]

    xbins = P_in.k_bin_edges_par

    ticklist_x = log2lin(minorticks, xbins)
    majorlist_x = log2lin(majorticks, xbins)

    ybins = P_in.k_bin_edges_perp

    ticklist_y = log2lin(minorticks, ybins)
    majorlist_y = log2lin(majorticks, ybins)

    # plt.imshow(np.log10(nmodes), interpolation='none', origin='lower')

    divider = make_axes_locatable(ax0)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig0.colorbar(img0, ax=ax0, cax=cax)
    cbar.set_label(
        r"$\log_{10}(\tilde{P}_{\parallel, \bot}(k))$ [$\mu$K${}^2$ (Mpc)${}^3$]"
    )

    ax0.set_xticks(ticklist_x, minor=True)
    ax0.set_xticks(majorlist_x, minor=False)
    ax0.set_xticklabels(majorlabels, minor=False)
    ax0.set_yticks(ticklist_y, minor=True)
    ax0.set_yticks(majorlist_y, minor=False)
    ax0.set_yticklabels(majorlabels, minor=False)

    ax0.set_xlim(0, 1)
    ax0.set_ylim(0, 1)

    ax0.set_xlabel(r"$k_{\parallel}$ [Mpc$^{-1}$]")
    ax0.set_xlabel(r"$k_{\parallel}$ [Mpc$^{-1}$]")

    ax0.set_ylabel(r"$k_{\bot}$ [Mpc$^{-1}$]")
    ax0.set_ylabel(r"$k_{\bot}$ [Mpc$^{-1}$]")

    # plt.savefig('ps_par_vs_perp_nmodes.png')
    # plt.savefig('ps_par_vs_perp_l2.png')
    # ax0[1, 1].set_title(r"$\frac{P_{data}}{P_{sim}}$")
    if outname != None:
        plt.savefig(outname, bbox_inches="tight")


def PS_plotter(simpath, l2mappath, noisepath, outname=None):
    # Raw cube
    # Raw cube

    fonts = {
        "font.family": "serif",
        "axes.labelsize": 14,
        "font.size": 14,
        "legend.fontsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    }
    plt.rcParams.update(fonts)

    inmap = map_cosmo.MapCosmo(simpath)

    P_in = power_spectrum.PowerSpectrum(inmap)

    P_in2d, k_in, nmodes_in = P_in.calculate_ps(do_2d=True)

    # L2 w/ sim map
    outmap = map_cosmo.MapCosmo(l2mappath)

    P_out = power_spectrum.PowerSpectrum(outmap)

    P_out2d, k_out, nmodes_out = P_out.calculate_ps(do_2d=True)

    # L2 wo/ sim map
    noisemap = map_cosmo.MapCosmo(noisepath)

    P_noise = power_spectrum.PowerSpectrum(noisemap)

    P_noise2d, k_noise, nmodes_noise = P_noise.calculate_ps(do_2d=True)

    # my_ps.make_h5(outname = "/mn/stornext/d16/cmbco/comap/nils/COMAP_general/data/maps/PS/out.h5")
    Transfunc = (P_out2d - P_noise2d) / P_in2d

    fig0, ax0 = plt.subplots(2, 2, figsize=(9, 7), sharex=True, sharey=True)
    cmap = "CMRmap"

    img0 = ax0[0, 0].imshow(
        np.log10(P_in2d),
        interpolation="none",
        origin="lower",
        extent=[0, 1, 0, 1],
        cmap=cmap,
        vmin=4,
        vmax=12,
        rasterized=True,
    )

    img1 = ax0[0, 1].imshow(
        np.log10(P_out2d),
        interpolation="none",
        origin="lower",
        cmap=cmap,
        extent=[0, 1, 0, 1],
        vmin=4,
        vmax=12,
        rasterized=True,
    )

    # img1 = ax0[0, 1].imshow(np.log10(P_out2d), interpolation='none', origin='lower',
    #                        extent=[0, 1, 0, 1], vmin = 4, vmax = 12)

    img2 = ax0[1, 0].imshow(
        np.log10(P_noise2d),
        interpolation="none",
        origin="lower",
        extent=[0, 1, 0, 1],
        cmap=cmap,
        vmin=4,
        vmax=12,
        rasterized=True,
    )

    # Transfunc = np.where(Transfunc <= 1, Transfunc, np.nan)

    img3 = ax0[1, 1].imshow(
        Transfunc,
        interpolation="none",
        origin="lower",
        extent=[0, 1, 0, 1],
        cmap=cmap,
        vmin=0,
        vmax=1.05,
        rasterized=True,
    )

    # fig_list = [fig0, fig1, fig2, fig3]
    # ax_list  = [ax0, ax1, ax2, ax3]
    img_list = [[img0, img1], [img2, img3]]

    def log2lin(x, k_edges):
        loglen = np.log10(k_edges[-1]) - np.log10(k_edges[0])
        logx = np.log10(x) - np.log10(k_edges[0])
        return logx / loglen

    # ax.set_xscale('log')
    minorticks = [
        0.0002,
        0.0003,
        0.0004,
        0.0005,
        0.0006,
        0.0007,
        0.0008,
        0.0009,
        0.002,
        0.003,
        0.004,
        0.005,
        0.006,
        0.007,
        0.008,
        0.009,
        0.02,
        0.03,
        0.04,
        0.05,
        0.06,
        0.07,
        0.08,
        0.09,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        20.0,
        30.0,
        40.0,
        50.0,
        60.0,
        70.0,
        80.0,
        90.0,
        200.0,
        300.0,
        400.0,
        500.0,
        600.0,
        700.0,
        800.0,
        900.0,
    ]

    majorticks = [1.0e-04, 1.0e-03, 1.0e-02, 1.0e-01, 1.0e00, 1.0e01, 1.0e02]
    majorlabels = [
        "$10^{-4}$",
        "$10^{-3}$",
        "$10^{-2}$",
        "$10^{-1}$",
        "$10^{0}$",
        "$10^{1}$",
        "$10^{2}$",
    ]

    xbins = P_in.k_bin_edges_par

    ticklist_x = log2lin(minorticks, xbins)
    majorlist_x = log2lin(majorticks, xbins)

    ybins = P_in.k_bin_edges_perp

    ticklist_y = log2lin(minorticks, ybins)
    majorlist_y = log2lin(majorticks, ybins)

    # plt.imshow(np.log10(nmodes), interpolation='none', origin='lower')

    divider = make_axes_locatable(ax0[0, 0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig0.colorbar(img0, ax=ax0[0, 0], cax=cax)
    cbar.set_label(
        r"$\log_{10}(\tilde{P}_{\parallel, \bot}(k))$ [$\mu$K${}^2$ (Mpc)${}^3$]"
    )

    divider = make_axes_locatable(ax0[0, 1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig0.colorbar(img1, ax=ax0[0, 1], cax=cax)
    cbar.set_label(
        r"$\log_{10}(\tilde{P}_{\parallel, \bot}(k))$ [$\mu$K${}^2$ (Mpc)${}^3$]"
    )

    divider = make_axes_locatable(ax0[1, 0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig0.colorbar(img2, ax=ax0[1, 0], cax=cax)
    cbar.set_label(
        r"$\log_{10}(\tilde{P}_{\parallel, \bot}(k))$ [$\mu$K${}^2$ (Mpc)${}^3$]"
    )

    divider = make_axes_locatable(ax0[1, 1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig0.colorbar(img3, ax=ax0[1, 1], cax=cax)
    cbar.set_label(r"$\tilde{T}_{\parallel, \bot}(k))$")
    # cbar.set_label(r'$\log_{10}(T)$')

    for i in range(2):
        for j in range(2):
            ax0[i, j].set_xticks(ticklist_x, minor=True)
            ax0[i, j].set_xticks(majorlist_x, minor=False)
            ax0[i, j].set_xticklabels(majorlabels, minor=False)
            ax0[i, j].set_yticks(ticklist_y, minor=True)
            ax0[i, j].set_yticks(majorlist_y, minor=False)
            ax0[i, j].set_yticklabels(majorlabels, minor=False)

            ax0[i, j].set_xlim(0, 1)
            ax0[i, j].set_ylim(0, 1)

    ax0[1, 0].set_xlabel(r"$k_{\parallel}$ [Mpc$^{-1}$]")
    ax0[1, 1].set_xlabel(r"$k_{\parallel}$ [Mpc$^{-1}$]")

    ax0[0, 0].set_ylabel(r"$k_{\bot}$ [Mpc$^{-1}$]")
    ax0[1, 0].set_ylabel(r"$k_{\bot}$ [Mpc$^{-1}$]")

    # plt.savefig('ps_par_vs_perp_nmodes.png')
    # plt.savefig('ps_par_vs_perp_l2.png')
    ax0[0, 0].set_title(r"$\tilde{P}_\mathrm{sim}$", fontsize=14)
    ax0[0, 1].set_title(r"$\tilde{P}_\mathrm{TOD+sim}$", fontsize=14)
    # ax0[0, 1].set_title(r"$P_{data}$")
    ax0[1, 0].set_title(r"$\tilde{P}_\mathrm{TOD}$", fontsize=14)
    ax0[1, 1].set_title(
        r"$\frac{\tilde{P}_\mathrm{TOD+sim} - \tilde{P}_\mathrm{TOD}}{\tilde{P}_\mathrm{sim}}$",
        fontsize=16,
    )
    # ax0[1, 1].set_title(r"$\frac{P_{data}}{P_{sim}}$")
    fig0.tight_layout(h_pad=0.1)
    if outname != None:
        plt.savefig(outname)


def plot_8_TFs(simpaths, mappaths, noisepaths, titles, outname=None):
    # Raw cube
    # Raw cube

    fonts = {
        "font.family": "serif",
        "axes.labelsize": 25,
        "font.size": 25,
        "legend.fontsize": 25,
        "xtick.labelsize": 25,
        "ytick.labelsize": 25,
    }

    plt.rcParams.update(fonts)

    def log2lin(x, k_edges):
        loglen = np.log10(k_edges[-1]) - np.log10(k_edges[0])
        logx = np.log10(x) - np.log10(k_edges[0])
        return logx / loglen

    minorticks = [
        0.0002,
        0.0003,
        0.0004,
        0.0005,
        0.0006,
        0.0007,
        0.0008,
        0.0009,
        0.002,
        0.003,
        0.004,
        0.005,
        0.006,
        0.007,
        0.008,
        0.009,
        0.02,
        0.03,
        0.04,
        0.05,
        0.06,
        0.07,
        0.08,
        0.09,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        20.0,
        30.0,
        40.0,
        50.0,
        60.0,
        70.0,
        80.0,
        90.0,
        200.0,
        300.0,
        400.0,
        500.0,
        600.0,
        700.0,
        800.0,
        900.0,
    ]

    majorticks = [1.0e-04, 1.0e-03, 1.0e-02, 1.0e-01, 1.0e00, 1.0e01, 1.0e02]
    majorlabels = [
        "$10^{-4}$",
        "$10^{-3}$",
        "$10^{-2}$",
        "$10^{-1}$",
        "$10^{0}$",
        "$10^{1}$",
        "$10^{2}$",
    ]

    fig0, ax0 = plt.subplots(4, 2, figsize=(16, 16 * 1.5), sharex=True, sharey=True)
    cmap = "CMRmap"
    cbars = []
    for i in range(4):
        for j in range(2):
            inmap = map_cosmo.MapCosmo(simpaths[i * 2 + j])

            P_in = power_spectrum.PowerSpectrum(inmap)

            P_in2d, k_in, nmodes_in = P_in.calculate_ps(do_2d=True)

            # L2 w/ sim map
            outmap = map_cosmo.MapCosmo(mappaths[i * 2 + j])

            P_out = power_spectrum.PowerSpectrum(outmap)

            P_out2d, k_out, nmodes_out = P_out.calculate_ps(do_2d=True)

            # L2 wo/ sim map
            noisemap = map_cosmo.MapCosmo(noisepaths[i * 2 + j])

            P_noise = power_spectrum.PowerSpectrum(noisemap)

            P_noise2d, k_noise, nmodes_noise = P_noise.calculate_ps(do_2d=True)

            # my_ps.make_h5(outname = "/mn/stornext/d16/cmbco/comap/nils/COMAP_general/data/maps/PS/out.h5")
            Transfunc = (P_out2d - P_noise2d) / P_in2d

            img0 = ax0[i, j].imshow(
                Transfunc,
                interpolation="none",
                origin="lower",
                extent=[0, 1, 0, 1],
                cmap=cmap,
                vmin=0,
                vmax=1.0,
                rasterized=True,
            )

            xbins = P_in.k_bin_edges_par

            ticklist_x = log2lin(minorticks, xbins)
            majorlist_x = log2lin(majorticks, xbins)

            ybins = P_in.k_bin_edges_perp

            ticklist_y = log2lin(minorticks, ybins)
            majorlist_y = log2lin(majorticks, ybins)

            ax0[i, j].set_xticks(ticklist_x, minor=True)
            ax0[i, j].set_xticks(majorlist_x, minor=False)
            ax0[i, j].set_xticklabels(majorlabels, minor=False)
            ax0[i, j].set_yticks(ticklist_y, minor=True)
            ax0[i, j].set_yticks(majorlist_y, minor=False)
            ax0[i, j].set_yticklabels(majorlabels, minor=False)

            ax0[i, j].set_xlim(0, 1)
            ax0[i, j].set_ylim(0, 1)

            ax0[i, j].set_title(titles[i * 2 + j], fontsize=23)

            divider = make_axes_locatable(ax0[i, j])
            cax = divider.append_axes("right", size="5%", pad=0.2)
            cbar = fig0.colorbar(img0, ax=ax0[i, j], cax=cax)
            cbars.append(cbar)
            cbar.set_label(r"$\tilde{T}_{\parallel, \bot}(k))$")

    cbars[0].remove()
    cbars[2].remove()
    cbars[4].remove()
    cbars[6].remove()

    ax0[3, 0].set_xlabel(r"$k_{\parallel}$ [Mpc$^{-1}$]")
    ax0[3, 1].set_xlabel(r"$k_{\parallel}$ [Mpc$^{-1}$]")

    ax0[0, 0].set_ylabel(r"$k_{\bot}$ [Mpc$^{-1}$]")
    ax0[1, 0].set_ylabel(r"$k_{\bot}$ [Mpc$^{-1}$]")
    ax0[2, 0].set_ylabel(r"$k_{\bot}$ [Mpc$^{-1}$]")
    ax0[3, 0].set_ylabel(r"$k_{\bot}$ [Mpc$^{-1}$]")
    """
    ax0[0, 0].text(0.12, 0.12, r"$\mathrm{\bf{(a)}}$", color = "w", fontsize = 25)
    ax0[0, 1].text(0.12, 0.12, r"$\mathrm{\bf{(b)}}$", color = "w", fontsize = 25)
    
    ax0[1, 0].text(0.12, 0.12, r"$\mathrm{\bf{(c)}}$", color = "w", fontsize = 25)
    ax0[1, 1].text(0.12, 0.12, r"$\mathrm{\bf{(d)}}$", color = "w", fontsize = 25)
    
    ax0[2, 0].text(0.12, 0.12, r"$\mathrm{\bf{(e)}}$", color = "w", fontsize = 25)
    ax0[2, 1].text(0.12, 0.12, r"$\mathrm{\bf{(f)}}$", color = "w", fontsize = 25)
    
    ax0[3, 0].text(0.12, 0.12, r"$\mathrm{\bf{(g)}}$", color = "w", fontsize = 25)
    ax0[3, 1].text(0.12, 0.12, r"$\mathrm{\bf{(h)}}$", color = "w", fontsize = 25)
    """

    fig0.tight_layout()
    if outname != None:
        plt.savefig(outname)


def plot_8_TFs_diffs(simpaths, mappaths, noisepaths, titles, clims, outname=None):
    # Raw cube
    # Raw cube

    fonts = {
        "font.family": "serif",
        "axes.labelsize": 25,
        "font.size": 25,
        "legend.fontsize": 25,
        "xtick.labelsize": 25,
        "ytick.labelsize": 25,
    }

    plt.rcParams.update(fonts)

    def log2lin(x, k_edges):
        loglen = np.log10(k_edges[-1]) - np.log10(k_edges[0])
        logx = np.log10(x) - np.log10(k_edges[0])
        return logx / loglen

    minorticks = [
        0.0002,
        0.0003,
        0.0004,
        0.0005,
        0.0006,
        0.0007,
        0.0008,
        0.0009,
        0.002,
        0.003,
        0.004,
        0.005,
        0.006,
        0.007,
        0.008,
        0.009,
        0.02,
        0.03,
        0.04,
        0.05,
        0.06,
        0.07,
        0.08,
        0.09,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        20.0,
        30.0,
        40.0,
        50.0,
        60.0,
        70.0,
        80.0,
        90.0,
        200.0,
        300.0,
        400.0,
        500.0,
        600.0,
        700.0,
        800.0,
        900.0,
    ]

    majorticks = [1.0e-04, 1.0e-03, 1.0e-02, 1.0e-01, 1.0e00, 1.0e01, 1.0e02]
    majorlabels = [
        "$10^{-4}$",
        "$10^{-3}$",
        "$10^{-2}$",
        "$10^{-1}$",
        "$10^{0}$",
        "$10^{1}$",
        "$10^{2}$",
    ]

    fig0, ax0 = plt.subplots(4, 2, figsize=(16, 16 * 1.5), sharex=True, sharey=True)
    cmap1 = "CMRmap"
    cmap2 = "RdBu_r"
    cbars = []
    TFs = []

    for i in range(4):
        for j in range(2):
            inmap = map_cosmo.MapCosmo(simpaths[i * 2 + j])

            P_in = power_spectrum.PowerSpectrum(inmap)

            P_in2d, k_in, nmodes_in = P_in.calculate_ps(do_2d=True)

            # L2 w/ sim map
            outmap = map_cosmo.MapCosmo(mappaths[i * 2 + j])

            P_out = power_spectrum.PowerSpectrum(outmap)

            P_out2d, k_out, nmodes_out = P_out.calculate_ps(do_2d=True)

            # L2 wo/ sim map
            noisemap = map_cosmo.MapCosmo(noisepaths[i * 2 + j])

            P_noise = power_spectrum.PowerSpectrum(noisemap)

            P_noise2d, k_noise, nmodes_noise = P_noise.calculate_ps(do_2d=True)

            # my_ps.make_h5(outname = "/mn/stornext/d16/cmbco/comap/nils/COMAP_general/data/maps/PS/out.h5")
            Transfunc = (P_out2d - P_noise2d) / P_in2d
            TFs.append(Transfunc)

            img0 = ax0[i, j].imshow(
                TFs[i * 2 + j] - TFs[0],
                interpolation="none",
                origin="lower",
                extent=[0, 1, 0, 1],
                cmap=cmap2,
                vmin=-clims[i * 2 + j],
                vmax=clims[i * 2 + j],
                rasterized=True,
            )

            xbins = P_in.k_bin_edges_par

            ticklist_x = log2lin(minorticks, xbins)
            majorlist_x = log2lin(majorticks, xbins)

            ybins = P_in.k_bin_edges_perp

            ticklist_y = log2lin(minorticks, ybins)
            majorlist_y = log2lin(majorticks, ybins)

            ax0[i, j].set_xticks(ticklist_x, minor=True)
            ax0[i, j].set_xticks(majorlist_x, minor=False)
            ax0[i, j].set_xticklabels(majorlabels, minor=False)
            ax0[i, j].set_yticks(ticklist_y, minor=True)
            ax0[i, j].set_yticks(majorlist_y, minor=False)
            ax0[i, j].set_yticklabels(majorlabels, minor=False)

            ax0[i, j].set_xlim(0, 1)
            ax0[i, j].set_ylim(0, 1)

            ax0[i, j].set_title(titles[i * 2 + j], fontsize=23)

            divider = make_axes_locatable(ax0[i, j])
            cax = divider.append_axes("right", size="5%", pad=0.2)
            cbar = fig0.colorbar(img0, ax=ax0[i, j], cax=cax)
            cbars.append(cbar)
            cbar.set_label(r"$\Delta\tilde{T}_{\parallel, \bot}(k))$")

    """cbars[0].remove()
    cbars[2].remove()
    cbars[4].remove()
    cbars[6].remove()"""

    ax0[3, 0].set_xlabel(r"$k_{\parallel}$ [Mpc$^{-1}$]")
    ax0[3, 1].set_xlabel(r"$k_{\parallel}$ [Mpc$^{-1}$]")

    ax0[0, 0].set_ylabel(r"$k_{\bot}$ [Mpc$^{-1}$]")
    ax0[1, 0].set_ylabel(r"$k_{\bot}$ [Mpc$^{-1}$]")
    ax0[2, 0].set_ylabel(r"$k_{\bot}$ [Mpc$^{-1}$]")
    ax0[3, 0].set_ylabel(r"$k_{\bot}$ [Mpc$^{-1}$]")
    """
    ax0[0, 0].text(0.12, 0.12, r"$\mathrm{\bf{(a)}}$", color = "w", fontsize = 25)
    ax0[0, 1].text(0.12, 0.12, r"$\mathrm{\bf{(b)}}$", color = "w", fontsize = 25)
    
    ax0[1, 0].text(0.12, 0.12, r"$\mathrm{\bf{(c)}}$", color = "w", fontsize = 25)
    ax0[1, 1].text(0.12, 0.12, r"$\mathrm{\bf{(d)}}$", color = "w", fontsize = 25)
    
    ax0[2, 0].text(0.12, 0.12, r"$\mathrm{\bf{(e)}}$", color = "w", fontsize = 25)
    ax0[2, 1].text(0.12, 0.12, r"$\mathrm{\bf{(f)}}$", color = "w", fontsize = 25)
    
    ax0[3, 0].text(0.12, 0.12, r"$\mathrm{\bf{(g)}}$", color = "w", fontsize = 25)
    ax0[3, 1].text(0.12, 0.12, r"$\mathrm{\bf{(h)}}$", color = "w", fontsize = 25)
    """
    cbars[0].remove()
    img0 = ax0[0, 0].imshow(
        TFs[0],
        interpolation="none",
        origin="lower",
        extent=[0, 1, 0, 1],
        cmap=cmap1,
        vmin=0,
        vmax=1,
        rasterized=True,
    )

    divider = make_axes_locatable(ax0[0, 0])
    cax = divider.append_axes("right", size="5%", pad=0.2)
    cbar = fig0.colorbar(img0, ax=ax0[0, 0], cax=cax)
    cbars.append(cbar)
    cbar.set_label(r"$\tilde{T}_{\parallel, \bot}(k))$")

    fig0.tight_layout()
    if outname != None:
        plt.savefig(outname)


def get_TF_diff(
    simpath_1,
    l2mappath_1,
    noisepath_1,
    simpath_2,
    l2mappath_2,
    noisepath_2,
    name1,
    name2,
    outname=None,
):
    # -------- First file --------
    # Raw cube
    inmap_1 = map_cosmo.MapCosmo(simpath_1)

    P_in_1 = power_spectrum.PowerSpectrum(inmap_1)

    P_in2d_1, k_in_1, nmodes_in_1 = P_in_1.calculate_ps(do_2d=True)

    # L2 w/ sim map
    outmap_1 = map_cosmo.MapCosmo(l2mappath_1)

    P_out_1 = power_spectrum.PowerSpectrum(outmap_1)

    P_out2d_1, k_out_1, nmodes_out_1 = P_out_1.calculate_ps(do_2d=True)

    # L2 wo/ sim map
    noisemap_1 = map_cosmo.MapCosmo(noisepath_1)

    P_noise_1 = power_spectrum.PowerSpectrum(noisemap_1)

    P_noise2d_1, k_noise_1, nmodes_noise_1 = P_noise_1.calculate_ps(do_2d=True)

    # my_ps.make_h5(outname = "/mn/stornext/d16/cmbco/comap/nils/COMAP_general/data/maps/PS/out.h5")
    Transfunc_1 = (P_out2d_1 - P_noise2d_1) / P_in2d_1

    # -------- Second file --------

    inmap_2 = map_cosmo.MapCosmo(simpath_2)

    P_in_2 = power_spectrum.PowerSpectrum(inmap_2)

    P_in2d_2, k_in_2, nmodes_in_2 = P_in_2.calculate_ps(do_2d=True)

    # L2 w/ sim map
    outmap_2 = map_cosmo.MapCosmo(l2mappath_2)

    P_out_2 = power_spectrum.PowerSpectrum(outmap_2)

    P_out2d_2, k_out_2, nmodes_out_2 = P_out_2.calculate_ps(do_2d=True)

    # L2 wo/ sim map
    noisemap_2 = map_cosmo.MapCosmo(noisepath_2)

    P_noise_2 = power_spectrum.PowerSpectrum(noisemap_1)

    P_noise2d_2, k_noise_2, nmodes_noise_2 = P_noise_2.calculate_ps(do_2d=True)

    # my_ps.make_h5(outname = "/mn/stornext/d16/cmbco/comap/nils/COMAP_general/data/maps/PS/out.h5")
    Transfunc_2 = (P_out2d_2 - P_noise2d_2) / P_in2d_2

    TF_diff = Transfunc_1 - Transfunc_2

    fig0, ax0 = plt.subplots()
    cmap = "CMRmap"

    img0 = ax0.imshow(
        TF_diff,
        interpolation="none",
        origin="lower",
        extent=[0, 1, 0, 1],
        cmap=cmap,
        vmin=0,
        vmax=1.0,
    )

    def log2lin(x, k_edges):
        loglen = np.log10(k_edges[-1]) - np.log10(k_edges[0])
        logx = np.log10(x) - np.log10(k_edges[0])
        return logx / loglen

    # ax.set_xscale('log')
    minorticks = [
        0.0002,
        0.0003,
        0.0004,
        0.0005,
        0.0006,
        0.0007,
        0.0008,
        0.0009,
        0.002,
        0.003,
        0.004,
        0.005,
        0.006,
        0.007,
        0.008,
        0.009,
        0.02,
        0.03,
        0.04,
        0.05,
        0.06,
        0.07,
        0.08,
        0.09,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        20.0,
        30.0,
        40.0,
        50.0,
        60.0,
        70.0,
        80.0,
        90.0,
        200.0,
        300.0,
        400.0,
        500.0,
        600.0,
        700.0,
        800.0,
        900.0,
    ]

    majorticks = [1.0e-04, 1.0e-03, 1.0e-02, 1.0e-01, 1.0e00, 1.0e01, 1.0e02]
    majorlabels = [
        "$10^{-4}$",
        "$10^{-3}$",
        "$10^{-2}$",
        "$10^{-1}$",
        "$10^{0}$",
        "$10^{1}$",
        "$10^{2}$",
    ]

    xbins = P_in_1.k_bin_edges_par

    ticklist_x = log2lin(minorticks, xbins)
    majorlist_x = log2lin(majorticks, xbins)

    ybins = P_in_1.k_bin_edges_perp

    ticklist_y = log2lin(minorticks, ybins)
    majorlist_y = log2lin(majorticks, ybins)

    # plt.imshow(np.log10(nmodes), interpolation='none', origin='lower')

    divider = make_axes_locatable(ax0)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig0.colorbar(img0, ax=ax0, cax=cax)
    cbar.set_label(r"$\Delta T$")
    # cbar.set_label(r'$\log_{10}(T)$')

    ax0.set_xticks(ticklist_x, minor=True)
    ax0.set_xticks(majorlist_x, minor=False)
    ax0.set_xticklabels(majorlabels, minor=False)
    ax0.set_yticks(ticklist_y, minor=True)
    ax0.set_yticks(majorlist_y, minor=False)
    ax0.set_yticklabels(majorlabels, minor=False)

    ax0.set_xlabel(r"$k_{\parallel}$ [Mpc$^{-1}$]")
    ax0.set_ylabel(r"$k_{\bot}$ [Mpc$^{-1}$]")
    ax0.set_xlim(0, 1)
    ax0.set_ylim(0, 1)

    # plt.savefig('ps_par_vs_perp_nmodes.png')
    # plt.savefig('ps_par_vs_perp_l2.png')
    ax0.set_title(rf"$T_\mathrm{{{name1}}} - T_\mathrm{{{name2}}}$", fontsize=12)
    # ax0[1, 1].set_title(r"$\frac{P_{data}}{P_{sim}}$")
    fig0.tight_layout()
    if outname != None:
        plt.savefig(outname)


def plot_single_TF(simpath_1, l2mappath_1, noisepath_1, outname=None):
    # -------- First file --------
    # Raw cube
    inmap_1 = map_cosmo.MapCosmo(simpath_1)

    P_in_1 = power_spectrum.PowerSpectrum(inmap_1)

    P_in2d_1, k_in_1, nmodes_in_1 = P_in_1.calculate_ps(do_2d=True)

    # L2 w/ sim map
    outmap_1 = map_cosmo.MapCosmo(l2mappath_1)

    P_out_1 = power_spectrum.PowerSpectrum(outmap_1)

    P_out2d_1, k_out_1, nmodes_out_1 = P_out_1.calculate_ps(do_2d=True)

    # L2 wo/ sim map
    noisemap_1 = map_cosmo.MapCosmo(noisepath_1)

    P_noise_1 = power_spectrum.PowerSpectrum(noisemap_1)

    P_noise2d_1, k_noise_1, nmodes_noise_1 = P_noise_1.calculate_ps(do_2d=True)

    # my_ps.make_h5(outname = "/mn/stornext/d16/cmbco/comap/nils/COMAP_general/data/maps/PS/out.h5")
    TF = (P_out2d_1 - P_noise2d_1) / P_in2d_1

    fig0, ax0 = plt.subplots()
    cmap = "CMRmap"

    img0 = ax0.imshow(
        TF,
        interpolation="none",
        origin="lower",
        extent=[0, 1, 0, 1],
        cmap=cmap,
        vmin=0,
        vmax=1.0,
    )

    def log2lin(x, k_edges):
        loglen = np.log10(k_edges[-1]) - np.log10(k_edges[0])
        logx = np.log10(x) - np.log10(k_edges[0])
        return logx / loglen

    # ax.set_xscale('log')
    minorticks = [
        0.0002,
        0.0003,
        0.0004,
        0.0005,
        0.0006,
        0.0007,
        0.0008,
        0.0009,
        0.002,
        0.003,
        0.004,
        0.005,
        0.006,
        0.007,
        0.008,
        0.009,
        0.02,
        0.03,
        0.04,
        0.05,
        0.06,
        0.07,
        0.08,
        0.09,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        20.0,
        30.0,
        40.0,
        50.0,
        60.0,
        70.0,
        80.0,
        90.0,
        200.0,
        300.0,
        400.0,
        500.0,
        600.0,
        700.0,
        800.0,
        900.0,
    ]

    majorticks = [1.0e-04, 1.0e-03, 1.0e-02, 1.0e-01, 1.0e00, 1.0e01, 1.0e02]
    majorlabels = [
        "$10^{-4}$",
        "$10^{-3}$",
        "$10^{-2}$",
        "$10^{-1}$",
        "$10^{0}$",
        "$10^{1}$",
        "$10^{2}$",
    ]

    xbins = P_in_1.k_bin_edges_par

    ticklist_x = log2lin(minorticks, xbins)
    majorlist_x = log2lin(majorticks, xbins)

    ybins = P_in_1.k_bin_edges_perp

    ticklist_y = log2lin(minorticks, ybins)
    majorlist_y = log2lin(majorticks, ybins)

    # plt.imshow(np.log10(nmodes), interpolation='none', origin='lower')

    divider = make_axes_locatable(ax0)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig0.colorbar(img0, ax=ax0, cax=cax)
    cbar.set_label(r"$\tilde{T}_{\parallel, \bot}(k)$")
    # cbar.set_label(r'$\log_{10}(T)$')

    ax0.set_xticks(ticklist_x, minor=True)
    ax0.set_xticks(majorlist_x, minor=False)
    ax0.set_xticklabels(majorlabels, minor=False)
    ax0.set_yticks(ticklist_y, minor=True)
    ax0.set_yticks(majorlist_y, minor=False)
    ax0.set_yticklabels(majorlabels, minor=False)

    ax0.set_xlabel(r"$k_{\parallel}$ [Mpc$^{-1}$]")
    ax0.set_ylabel(r"$k_{\bot}$ [Mpc$^{-1}$]")
    ax0.set_xlim(0, 1)
    ax0.set_ylim(0, 1)

    # plt.savefig('ps_par_vs_perp_nmodes.png')
    # plt.savefig('ps_par_vs_perp_l2.png')
    # ax0[1, 1].set_title(r"$\frac{P_{data}}{P_{sim}}$")
    fig0.tight_layout()
    if outname != None:
        plt.savefig(outname, bbox_inches="tight")


def get_2D_TF(simpath, mappath, noisepath):
    # Raw cube
    inmap = map_cosmo.MapCosmo(simpath)

    P_in = power_spectrum.PowerSpectrum(inmap)

    ps_in, k_in, nmodes_in = P_in.calculate_ps(do_2d=True)

    # L2 w/ sim map
    outmap = map_cosmo.MapCosmo(mappath)

    P_out = power_spectrum.PowerSpectrum(outmap)

    ps_out, k_out, nmodes_out = P_out.calculate_ps(do_2d=True)

    # L2 wo/ sim map
    noisemap = map_cosmo.MapCosmo(noisepath)

    P_noise = power_spectrum.PowerSpectrum(noisemap)

    ps_noise, k_noise, nmodes_noise = P_noise.calculate_ps(do_2d=True)

    TF = (ps_out - ps_noise) / ps_in
    k = k_in

    return TF, k


def get_1D_TF(simpath, mappath, noisepath):
    # Raw cube
    inmap = map_cosmo.MapCosmo(simpath)

    P_in = power_spectrum.PowerSpectrum(inmap)

    ps_in, k_in, nmodes_in = P_in.calculate_ps(do_2d=False)

    # L2 w/ sim map
    outmap = map_cosmo.MapCosmo(mappath)

    P_out = power_spectrum.PowerSpectrum(outmap)

    ps_out, k_out, nmodes_out = P_out.calculate_ps(do_2d=False)

    # L2 wo/ sim map
    noisemap = map_cosmo.MapCosmo(noisepath)

    P_noise = power_spectrum.PowerSpectrum(noisemap)

    ps_noise, k_noise, nmodes_noise = P_noise.calculate_ps(do_2d=False)

    TF = (ps_out - ps_noise) / ps_in
    k = k_in

    return TF, k


def get_2D_PS(simpath, mappath, noisepath):
    # Raw cube
    inmap = map_cosmo.MapCosmo(simpath)

    P_in = power_spectrum.PowerSpectrum(inmap)

    ps_in, k_in, nmodes_in = P_in.calculate_ps(do_2d=True)

    # L2 w/ sim map
    outmap = map_cosmo.MapCosmo(mappath)

    P_out = power_spectrum.PowerSpectrum(outmap)

    ps_out, k_out, nmodes_out = P_out.calculate_ps(do_2d=True)

    # L2 wo/ sim map
    noisemap = map_cosmo.MapCosmo(noisepath)

    P_noise = power_spectrum.PowerSpectrum(noisemap)

    ps_noise, k_noise, nmodes_noise = P_noise.calculate_ps(do_2d=True)
    print(np.log10(ps_noise[0, 0]))

    k = k_in

    return ps_in, ps_out, ps_noise, k


def get_1D_PS(path):
    # Raw cube
    inmap = map_cosmo.MapCosmo(path)

    P_in = power_spectrum.PowerSpectrum(inmap)

    ps_in, k_in, nmodes_in = P_in.calculate_ps(do_2d=False)

    return ps_in, k_in


def plot_two_TFs_and_diff(
    simpath_1,
    l2mappath_1,
    noisepath_1,
    simpath_2,
    l2mappath_2,
    noisepath_2,
    outname=None,
):
    # -------- First file --------
    # Raw cube
    inmap_1 = map_cosmo.MapCosmo(simpath_1)

    P_in_1 = power_spectrum.PowerSpectrum(inmap_1)

    P_in2d_1, k_in_1, nmodes_in_1 = P_in_1.calculate_ps(do_2d=True)

    # L2 w/ sim map
    outmap_1 = map_cosmo.MapCosmo(l2mappath_1)

    P_out_1 = power_spectrum.PowerSpectrum(outmap_1)

    P_out2d_1, k_out_1, nmodes_out_1 = P_out_1.calculate_ps(do_2d=True)

    # L2 wo/ sim map
    noisemap_1 = map_cosmo.MapCosmo(noisepath_1)

    P_noise_1 = power_spectrum.PowerSpectrum(noisemap_1)

    P_noise2d_1, k_noise_1, nmodes_noise_1 = P_noise_1.calculate_ps(do_2d=True)

    # my_ps.make_h5(outname = "/mn/stornext/d16/cmbco/comap/nils/COMAP_general/data/maps/PS/out.h5")
    Transfunc_1 = (P_out2d_1 - P_noise2d_1) / P_in2d_1

    # -------- Second file --------

    inmap_2 = map_cosmo.MapCosmo(simpath_2)

    P_in_2 = power_spectrum.PowerSpectrum(inmap_2)

    P_in2d_2, k_in_2, nmodes_in_2 = P_in_2.calculate_ps(do_2d=True)

    # L2 w/ sim map
    outmap_2 = map_cosmo.MapCosmo(l2mappath_2)

    P_out_2 = power_spectrum.PowerSpectrum(outmap_2)

    P_out2d_2, k_out_2, nmodes_out_2 = P_out_2.calculate_ps(do_2d=True)

    # L2 wo/ sim map
    noisemap_2 = map_cosmo.MapCosmo(noisepath_2)

    P_noise_2 = power_spectrum.PowerSpectrum(noisemap_2)

    P_noise2d_2, k_noise_2, nmodes_noise_2 = P_noise_2.calculate_ps(do_2d=True)

    # my_ps.make_h5(outname = "/mn/stornext/d16/cmbco/comap/nils/COMAP_general/data/maps/PS/out.h5")
    Transfunc_2 = (P_out2d_2 - P_noise2d_2) / P_in2d_2

    TF_diff = Transfunc_2 - Transfunc_1

    fig0, ax0 = plt.subplots(1, 3, figsize=(20, 20))
    cmap = "CMRmap"
    # cmap = "magma"
    cmap2 = "RdBu_r"
    # cmap2 = "bwr"

    img0 = ax0[0].imshow(
        Transfunc_1,
        interpolation="none",
        origin="lower",
        extent=[0, 1, 0, 1],
        cmap=cmap,
        vmin=0,
        vmax=1,
        rasterized=True,
    )
    img1 = ax0[1].imshow(
        Transfunc_2,
        interpolation="none",
        origin="lower",
        extent=[0, 1, 0, 1],
        cmap=cmap,
        vmin=0,
        vmax=1,
        rasterized=True,
    )
    # vmin = 0, vmax = 5.0, rasterized=True)
    img2 = ax0[2].imshow(
        TF_diff,
        interpolation="none",
        origin="lower",
        extent=[0, 1, 0, 1],
        cmap=cmap2,
        # vmin = 0.8, vmax = 2.0, rasterized=True)
        vmin=-0.25,
        vmax=0.25,
        rasterized=True,
    )
    # norm=colors.SymLogNorm(linthresh=0.01, linscale=0.01,
    #                          vmin=-30, vmax=-1e-2, base=10))

    def log2lin(x, k_edges):
        loglen = np.log10(k_edges[-1]) - np.log10(k_edges[0])
        logx = np.log10(x) - np.log10(k_edges[0])
        return logx / loglen

    # ax.set_xscale('log')
    minorticks = [
        0.0002,
        0.0003,
        0.0004,
        0.0005,
        0.0006,
        0.0007,
        0.0008,
        0.0009,
        0.002,
        0.003,
        0.004,
        0.005,
        0.006,
        0.007,
        0.008,
        0.009,
        0.02,
        0.03,
        0.04,
        0.05,
        0.06,
        0.07,
        0.08,
        0.09,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        20.0,
        30.0,
        40.0,
        50.0,
        60.0,
        70.0,
        80.0,
        90.0,
        200.0,
        300.0,
        400.0,
        500.0,
        600.0,
        700.0,
        800.0,
        900.0,
    ]

    majorticks = [1.0e-04, 1.0e-03, 1.0e-02, 1.0e-01, 1.0e00, 1.0e01, 1.0e02]
    majorlabels = [
        "$10^{-4}$",
        "$10^{-3}$",
        "$10^{-2}$",
        "$10^{-1}$",
        "$10^{0}$",
        "$10^{1}$",
        "$10^{2}$",
    ]

    xbins = P_in_1.k_bin_edges_par

    ticklist_x = log2lin(minorticks, xbins)
    majorlist_x = log2lin(majorticks, xbins)

    ybins = P_in_1.k_bin_edges_perp

    ticklist_y = log2lin(minorticks, ybins)
    majorlist_y = log2lin(majorticks, ybins)

    divider0 = make_axes_locatable(ax0[0])
    divider1 = make_axes_locatable(ax0[1])
    divider2 = make_axes_locatable(ax0[2])
    cax0 = divider0.append_axes("bottom", size="5%", pad=0.9)
    cax1 = divider1.append_axes("bottom", size="5%", pad=0.9)
    cax2 = divider2.append_axes("bottom", size="5%", pad=0.9)
    cbar0 = fig0.colorbar(img0, ax=ax0[0], cax=cax0, orientation="horizontal")
    cbar1 = fig0.colorbar(img1, ax=ax0[1], cax=cax1, orientation="horizontal")
    cbar2 = fig0.colorbar(img2, ax=ax0[2], cax=cax2, orientation="horizontal")
    cbar0.set_label(r"$\tilde{T}_{\parallel, \bot}(k)$")
    cbar1.set_label(r"$\tilde{T}_{\parallel, \bot}(k)$")
    # cbar2.set_label(r'$\tilde{T}_{\parallel, \bot}(k)^\mathrm{(b)} / \tilde{T}_{\parallel, \bot}(k)^\mathrm{(b)}$')
    cbar2.set_label(r"$\Delta \tilde{T}_{\parallel, \bot}(k)$")

    for i in range(len(ax0)):
        ax0[i].set_xticks(ticklist_x, minor=True)
        ax0[i].set_xticks(majorlist_x, minor=False)
        ax0[i].set_xticklabels(majorlabels, minor=False)
        ax0[i].set_yticks(ticklist_y, minor=True)
        ax0[i].set_yticks(majorlist_y, minor=False)
        ax0[i].set_yticklabels(majorlabels, minor=False, rotation=90)

        ax0[i].set_xlabel(r"$k_{\parallel}$ [Mpc$^{-1}$]")
        ax0[i].set_ylabel(r"$k_{\bot}$ [Mpc$^{-1}$]", rotation=90)
        ax0[i].set_xlim(0, 1)
        ax0[i].set_ylim(0, 1)

    # plt.savefig('ps_par_vs_perp_nmodes.png')
    # plt.savefig('ps_par_vs_perp_l2.png')
    # ax0[0].set_title(r"$\mathrm{\bf{(a)}}$")
    # ax0[1].set_title(r"$\mathrm{\bf{(b)}}$")
    # ax0[2].set_title(r"$\mathrm{\bf{(c)}}$")
    ax0[0].text(0.07, 0.12, r"$\mathrm{\bf{(a)}}$", color="gray", fontsize=16)
    ax0[1].text(0.07, 0.12, r"$\mathrm{\bf{(b)}}$", color="gray", fontsize=16)
    ax0[2].text(0.07, 0.12, r"$\mathrm{\bf{(c)}}$", color="gray", fontsize=16)

    fig0.tight_layout(pad=0)
    if outname != None:
        # plt.savefig(outname, bbox_inches = "tight", pad_inches = 0.0787402, dpi = 1000)
        plt.savefig(outname, bbox_inches="tight")


def plot_three_TFs(
    simpath_1,
    l2mappath_1,
    noisepath_1,
    simpath_2,
    l2mappath_2,
    noisepath_2,
    simpath_3,
    l2mappath_3,
    noisepath_3,
    outname=None,
):
    # -------- First file --------
    # Raw cube
    inmap_1 = map_cosmo.MapCosmo(simpath_1)

    P_in_1 = power_spectrum.PowerSpectrum(inmap_1)

    P_in2d_1, k_in_1, nmodes_in_1 = P_in_1.calculate_ps(do_2d=True)

    # L2 w/ sim map
    outmap_1 = map_cosmo.MapCosmo(l2mappath_1)

    P_out_1 = power_spectrum.PowerSpectrum(outmap_1)

    P_out2d_1, k_out_1, nmodes_out_1 = P_out_1.calculate_ps(do_2d=True)

    # L2 wo/ sim map
    noisemap_1 = map_cosmo.MapCosmo(noisepath_1)

    P_noise_1 = power_spectrum.PowerSpectrum(noisemap_1)

    P_noise2d_1, k_noise_1, nmodes_noise_1 = P_noise_1.calculate_ps(do_2d=True)

    # my_ps.make_h5(outname = "/mn/stornext/d16/cmbco/comap/nils/COMAP_general/data/maps/PS/out.h5")
    Transfunc_1 = (P_out2d_1 - P_noise2d_1) / P_in2d_1

    # -------- Second file --------

    inmap_2 = map_cosmo.MapCosmo(simpath_2)

    P_in_2 = power_spectrum.PowerSpectrum(inmap_2)

    P_in2d_2, k_in_2, nmodes_in_2 = P_in_2.calculate_ps(do_2d=True)

    # L2 w/ sim map
    outmap_2 = map_cosmo.MapCosmo(l2mappath_2)

    P_out_2 = power_spectrum.PowerSpectrum(outmap_2)

    P_out2d_2, k_out_2, nmodes_out_2 = P_out_2.calculate_ps(do_2d=True)

    # L2 wo/ sim map
    noisemap_2 = map_cosmo.MapCosmo(noisepath_2)

    P_noise_2 = power_spectrum.PowerSpectrum(noisemap_2)

    P_noise2d_2, k_noise_2, nmodes_noise_2 = P_noise_2.calculate_ps(do_2d=True)

    Transfunc_2 = (P_out2d_2 - P_noise2d_2) / P_in2d_2

    inmap_3 = map_cosmo.MapCosmo(simpath_3)

    P_in_3 = power_spectrum.PowerSpectrum(inmap_3)

    P_in2d_3, k_in_3, nmodes_in_3 = P_in_3.calculate_ps(do_2d=True)

    # L2 w/ sim map
    outmap_3 = map_cosmo.MapCosmo(l2mappath_3)

    P_out_3 = power_spectrum.PowerSpectrum(outmap_3)

    P_out2d_3, k_out_3, nmodes_out_3 = P_out_3.calculate_ps(do_2d=True)

    # L2 wo/ sim map
    noisemap_3 = map_cosmo.MapCosmo(noisepath_3)

    P_noise_3 = power_spectrum.PowerSpectrum(noisemap_3)

    P_noise2d_3, k_noise_3, nmodes_noise_3 = P_noise_3.calculate_ps(do_2d=True)

    Transfunc_3 = (P_out2d_3 - P_noise2d_3) / P_in2d_3

    fig0, ax0 = plt.subplots(1, 3, figsize=(16, 16 * 2 / 3), sharey=True)
    # cmap = "CMRmap"
    cmap = "RdBu_r"
    # cmap = "magma"

    img0 = ax0[0].imshow(
        Transfunc_1,
        interpolation="none",
        origin="lower",
        extent=[0, 1, 0, 1],
        cmap=cmap,
        vmin=0,
        vmax=1,
        rasterized=True,
    )
    img1 = ax0[1].imshow(
        Transfunc_2,
        interpolation="none",
        origin="lower",
        extent=[0, 1, 0, 1],
        cmap=cmap,
        vmin=0,
        vmax=1,
        rasterized=True,
    )
    # vmin = 0, vmax = 5.0, rasterized=True)
    img2 = ax0[2].imshow(
        Transfunc_3,
        interpolation="none",
        origin="lower",
        extent=[0, 1, 0, 1],
        cmap=cmap,
        vmin=0,
        vmax=1,
        rasterized=True,
    )
    # vmin = 0, vmax = 5.0, rasterized=True)

    def log2lin(x, k_edges):
        loglen = np.log10(k_edges[-1]) - np.log10(k_edges[0])
        logx = np.log10(x) - np.log10(k_edges[0])
        return logx / loglen

    # ax.set_xscale('log')
    minorticks = [
        0.0002,
        0.0003,
        0.0004,
        0.0005,
        0.0006,
        0.0007,
        0.0008,
        0.0009,
        0.002,
        0.003,
        0.004,
        0.005,
        0.006,
        0.007,
        0.008,
        0.009,
        0.02,
        0.03,
        0.04,
        0.05,
        0.06,
        0.07,
        0.08,
        0.09,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        20.0,
        30.0,
        40.0,
        50.0,
        60.0,
        70.0,
        80.0,
        90.0,
        200.0,
        300.0,
        400.0,
        500.0,
        600.0,
        700.0,
        800.0,
        900.0,
    ]

    majorticks = [1.0e-04, 1.0e-03, 1.0e-02, 1.0e-01, 1.0e00, 1.0e01, 1.0e02]
    majorlabels = [
        "$10^{-4}$",
        "$10^{-3}$",
        "$10^{-2}$",
        "$10^{-1}$",
        "$10^{0}$",
        "$10^{1}$",
        "$10^{2}$",
    ]

    xbins = P_in_1.k_bin_edges_par

    ticklist_x = log2lin(minorticks, xbins)
    majorlist_x = log2lin(majorticks, xbins)

    ybins = P_in_1.k_bin_edges_perp

    ticklist_y = log2lin(minorticks, ybins)
    majorlist_y = log2lin(majorticks, ybins)

    divider0 = make_axes_locatable(ax0[0])
    divider1 = make_axes_locatable(ax0[1])
    divider2 = make_axes_locatable(ax0[2])
    cax0 = divider0.append_axes("bottom", size="5%", pad=1)
    cax1 = divider1.append_axes("bottom", size="5%", pad=1)
    cax2 = divider2.append_axes("bottom", size="5%", pad=1)
    cbar0 = fig0.colorbar(img0, ax=ax0[0], cax=cax0, orientation="horizontal")
    cbar1 = fig0.colorbar(img1, ax=ax0[1], cax=cax1, orientation="horizontal")
    cbar2 = fig0.colorbar(img2, ax=ax0[2], cax=cax2, orientation="horizontal")
    cbar0.set_label(r"$\Delta \tilde{T}_{\parallel, \bot}(k)$")
    cbar1.set_label(r"$\Delta \tilde{T}_{\parallel, \bot}(k)$")
    cbar2.set_label(r"$\Delta \tilde{T}_{\parallel, \bot}(k)$")

    # cbar0.remove()
    # cbar1.remove()

    for i in range(len(ax0)):
        ax0[i].set_xticks(ticklist_x, minor=True)
        ax0[i].set_xticks(majorlist_x, minor=False)
        ax0[i].set_xticklabels(majorlabels, minor=False)
        ax0[i].set_yticks(ticklist_y, minor=True)
        ax0[i].set_yticks(majorlist_y, minor=False)
        ax0[i].set_yticklabels(majorlabels, minor=False, rotation=90)

        ax0[i].set_xlabel(r"$k_{\parallel}$ [Mpc$^{-1}$]")
        ax0[i].set_xlim(0, 1)
        ax0[i].set_ylim(0, 1)

    ax0[0].set_ylabel(r"$k_{\bot}$ [Mpc$^{-1}$]", rotation=90)

    # plt.savefig('ps_par_vs_perp_nmodes.png')
    # plt.savefig('ps_par_vs_perp_l2.png')
    # ax0[0].set_title(r"$\mathrm{\bf{(a)}}$")
    # ax0[1].set_title(r"$\mathrm{\bf{(b)}}$")
    # ax0[2].set_title(r"$\mathrm{\bf{(c)}}$")
    ax0[0].text(0.07, 0.85, r"$\mathrm{\bf{(a)}}$", color="gray", fontsize=22)
    ax0[1].text(0.07, 0.85, r"$\mathrm{\bf{(b)}}$", color="gray", fontsize=22)
    ax0[2].text(0.07, 0.85, r"$\mathrm{\bf{(c)}}$", color="gray", fontsize=22)

    fig0.tight_layout(h_pad=2)
    if outname != None:
        # plt.savefig(outname, bbox_inches = "tight", pad_inches = 0.0787402, dpi = 1000)
        plt.savefig(outname, bbox_inches="tight")


def plot_three_PS(noisepath_1, noisepath_2, noisepath_3, outname=None):

    noisemap_1 = map_cosmo.MapCosmo(noisepath_1)

    P_noise_1 = power_spectrum.PowerSpectrum(noisemap_1)

    P_noise2d_1, k_noise_1, nmodes_noise_1 = P_noise_1.calculate_ps(do_2d=True)

    noisemap_2 = map_cosmo.MapCosmo(noisepath_2)

    P_noise_2 = power_spectrum.PowerSpectrum(noisemap_2)

    P_noise2d_2, k_noise_2, nmodes_noise_2 = P_noise_2.calculate_ps(do_2d=True)

    noisemap_3 = map_cosmo.MapCosmo(noisepath_3)

    P_noise_3 = power_spectrum.PowerSpectrum(noisemap_3)

    P_noise2d_3, k_noise_3, nmodes_noise_3 = P_noise_3.calculate_ps(do_2d=True)

    fig0, ax0 = plt.subplots(1, 3, figsize=(16, 16 * 2 / 3), sharey=True)
    cmap = "CMRmap"
    # cmap = "magma"

    img0 = ax0[0].imshow(
        P_noise2d_1,
        interpolation="none",
        origin="lower",
        extent=[0, 1, 0, 1],
        cmap=cmap,
        vmin=1e6,
        vmax=4e7,
        rasterized=True,
    )
    img1 = ax0[1].imshow(
        P_noise2d_2,
        interpolation="none",
        origin="lower",
        extent=[0, 1, 0, 1],
        cmap=cmap,
        vmin=1e6,
        vmax=4e7,
        rasterized=True,
    )
    # vmin = 0, vmax = 5.0, rasterized=True)
    img2 = ax0[2].imshow(
        P_noise2d_3,
        interpolation="none",
        origin="lower",
        extent=[0, 1, 0, 1],
        cmap=cmap,
        vmin=1e6,
        vmax=4e7,
        rasterized=True,
    )
    # vmin = 0, vmax = 5.0, rasterized=True)

    def log2lin(x, k_edges):
        loglen = np.log10(k_edges[-1]) - np.log10(k_edges[0])
        logx = np.log10(x) - np.log10(k_edges[0])
        return logx / loglen

    # ax.set_xscale('log')
    minorticks = [
        0.0002,
        0.0003,
        0.0004,
        0.0005,
        0.0006,
        0.0007,
        0.0008,
        0.0009,
        0.002,
        0.003,
        0.004,
        0.005,
        0.006,
        0.007,
        0.008,
        0.009,
        0.02,
        0.03,
        0.04,
        0.05,
        0.06,
        0.07,
        0.08,
        0.09,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        20.0,
        30.0,
        40.0,
        50.0,
        60.0,
        70.0,
        80.0,
        90.0,
        200.0,
        300.0,
        400.0,
        500.0,
        600.0,
        700.0,
        800.0,
        900.0,
    ]

    majorticks = [1.0e-04, 1.0e-03, 1.0e-02, 1.0e-01, 1.0e00, 1.0e01, 1.0e02]
    majorlabels = [
        "$10^{-4}$",
        "$10^{-3}$",
        "$10^{-2}$",
        "$10^{-1}$",
        "$10^{0}$",
        "$10^{1}$",
        "$10^{2}$",
    ]

    xbins = P_noise_1.k_bin_edges_par

    ticklist_x = log2lin(minorticks, xbins)
    majorlist_x = log2lin(majorticks, xbins)

    ybins = P_noise_1.k_bin_edges_perp

    ticklist_y = log2lin(minorticks, ybins)
    majorlist_y = log2lin(majorticks, ybins)

    divider0 = make_axes_locatable(ax0[0])
    divider1 = make_axes_locatable(ax0[1])
    divider2 = make_axes_locatable(ax0[2])
    cax0 = divider0.append_axes("right", size="5%", pad=0.2)
    cax1 = divider1.append_axes("right", size="5%", pad=0.2)
    cax2 = divider2.append_axes("right", size="5%", pad=0.2)
    cbar0 = fig0.colorbar(img0, ax=ax0[0], cax=cax0)  # , orientation = "horizontal")
    cbar1 = fig0.colorbar(img1, ax=ax0[1], cax=cax1)  # , orientation = "horizontal")
    cbar2 = fig0.colorbar(img2, ax=ax0[2], cax=cax2)  # , orientation = "horizontal")
    cbar0.set_label(
        r"$\log_{10}(\tilde{P}_{\parallel, \bot}(k))$ [$\mu$K${}^2$ (Mpc)${}^3$]"
    )
    cbar1.set_label(
        r"$\log_{10}(\tilde{P}_{\parallel, \bot}(k))$ [$\mu$K${}^2$ (Mpc)${}^3$]"
    )
    cbar2.set_label(
        r"$\log_{10}(\tilde{P}_{\parallel, \bot}(k))$ [$\mu$K${}^2$ (Mpc)${}^3$]"
    )

    cbar0.remove()
    cbar1.remove()

    for i in range(len(ax0)):
        ax0[i].set_xticks(ticklist_x, minor=True)
        ax0[i].set_xticks(majorlist_x, minor=False)
        ax0[i].set_xticklabels(majorlabels, minor=False)
        ax0[i].set_yticks(ticklist_y, minor=True)
        ax0[i].set_yticks(majorlist_y, minor=False)
        ax0[i].set_yticklabels(majorlabels, minor=False, rotation=90)

        ax0[i].set_xlabel(r"$k_{\parallel}$ [Mpc$^{-1}$]")
        ax0[i].set_xlim(0, 1)
        ax0[i].set_ylim(0, 1)

    ax0[0].set_ylabel(r"$k_{\bot}$ [Mpc$^{-1}$]", rotation=90)

    # plt.savefig('ps_par_vs_perp_nmodes.png')
    # plt.savefig('ps_par_vs_perp_l2.png')
    # ax0[0].set_title(r"$\mathrm{\bf{(a)}}$")
    # ax0[1].set_title(r"$\mathrm{\bf{(b)}}$")
    # ax0[2].set_title(r"$\mathrm{\bf{(c)}}$")
    ax0[0].text(0.07, 0.88, r"$\mathrm{\bf{(a)}}$", color="gray", fontsize=22)
    ax0[1].text(0.07, 0.88, r"$\mathrm{\bf{(b)}}$", color="gray", fontsize=22)
    ax0[2].text(0.07, 0.88, r"$\mathrm{\bf{(c)}}$", color="gray", fontsize=22)

    fig0.tight_layout(pad=0)
    if outname != None:
        # plt.savefig(outname, bbox_inches = "tight", pad_inches = 0.0787402, dpi = 1000)
        plt.savefig(outname, bbox_inches="tight")


def plot_two_PS_and_diff(path1, path2, outname=None):
    # -------- First file --------
    map1 = map_cosmo.MapCosmo(path1)

    P1 = power_spectrum.PowerSpectrum(map1)

    P_2d_1, k1, nmodes1 = P1.calculate_ps(do_2d=True)

    # -------- Second file --------
    map2 = map_cosmo.MapCosmo(path2)

    P2 = power_spectrum.PowerSpectrum(map2)

    P_2d_2, k2, nmodes2 = P2.calculate_ps(do_2d=True)
    print(np.log10(P_2d_1[0, 0]), np.log10(P_2d_2[0, 0]))

    PS_diff = P_2d_2 / P_2d_1

    fig0, ax0 = plt.subplots(1, 3, figsize=(20, 20))
    cmap = "CMRmap"
    # cmap = "magma"
    # cmap2 = "RdBu"
    cmap2 = "bwr"

    img0 = ax0[0].imshow(
        np.log10(P_2d_1),
        interpolation="none",
        origin="lower",
        extent=[0, 1, 0, 1],
        cmap=cmap,
        vmin=None,
        vmax=None,
        rasterized=True,
    )
    img1 = ax0[1].imshow(
        np.log10(P_2d_2),
        interpolation="none",
        origin="lower",
        extent=[0, 1, 0, 1],
        cmap=cmap,
        vmin=None,
        vmax=None,
        rasterized=True,
    )
    # vmin = 0, vmax = 5.0, rasterized=True)
    img2 = ax0[2].imshow(
        PS_diff,
        interpolation="none",
        origin="lower",
        extent=[0, 1, 0, 1],
        cmap=cmap,
        vmin=0,
        vmax=3,
        rasterized=True,
    )
    # norm=colors.SymLogNorm(linthresh=0.01, linscale=0.01,
    #                          vmin=-30, vmax=-1e-2, base=10))

    def log2lin(x, k_edges):
        loglen = np.log10(k_edges[-1]) - np.log10(k_edges[0])
        logx = np.log10(x) - np.log10(k_edges[0])
        return logx / loglen

    # ax.set_xscale('log')
    minorticks = [
        0.0002,
        0.0003,
        0.0004,
        0.0005,
        0.0006,
        0.0007,
        0.0008,
        0.0009,
        0.002,
        0.003,
        0.004,
        0.005,
        0.006,
        0.007,
        0.008,
        0.009,
        0.02,
        0.03,
        0.04,
        0.05,
        0.06,
        0.07,
        0.08,
        0.09,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        20.0,
        30.0,
        40.0,
        50.0,
        60.0,
        70.0,
        80.0,
        90.0,
        200.0,
        300.0,
        400.0,
        500.0,
        600.0,
        700.0,
        800.0,
        900.0,
    ]

    majorticks = [1.0e-04, 1.0e-03, 1.0e-02, 1.0e-01, 1.0e00, 1.0e01, 1.0e02]
    majorlabels = [
        "$10^{-4}$",
        "$10^{-3}$",
        "$10^{-2}$",
        "$10^{-1}$",
        "$10^{0}$",
        "$10^{1}$",
        "$10^{2}$",
    ]

    xbins = P1.k_bin_edges_par

    ticklist_x = log2lin(minorticks, xbins)
    majorlist_x = log2lin(majorticks, xbins)

    ybins = P1.k_bin_edges_perp

    ticklist_y = log2lin(minorticks, ybins)
    majorlist_y = log2lin(majorticks, ybins)

    divider0 = make_axes_locatable(ax0[0])
    divider1 = make_axes_locatable(ax0[1])
    divider2 = make_axes_locatable(ax0[2])
    cax0 = divider0.append_axes("bottom", size="5%", pad=0.9)
    cax1 = divider1.append_axes("bottom", size="5%", pad=0.9)
    cax2 = divider2.append_axes("bottom", size="5%", pad=0.9)
    cbar0 = fig0.colorbar(img0, ax=ax0[0], cax=cax0, orientation="horizontal")
    cbar1 = fig0.colorbar(img1, ax=ax0[1], cax=cax1, orientation="horizontal")
    cbar2 = fig0.colorbar(img2, ax=ax0[2], cax=cax2, orientation="horizontal")
    cbar0.set_label(
        r"$\log_{10}(\tilde{P}_{\parallel, \bot}(k))$ [$\mu$K${}^2$ (Mpc)${}^3$]"
    )
    cbar1.set_label(
        r"$\log_{10}(\tilde{P}_{\parallel, \bot}(k))$ [$\mu$K${}^2$ (Mpc)${}^3$]"
    )
    cbar2.set_label(
        r"$\tilde{P}_{\parallel, \bot}^{(b)}(k) / \tilde{P}_{\parallel, \bot}^{(a)}(k)$"
    )

    for i in range(len(ax0)):
        ax0[i].set_xticks(ticklist_x, minor=True)
        ax0[i].set_xticks(majorlist_x, minor=False)
        ax0[i].set_xticklabels(majorlabels, minor=False)
        ax0[i].set_yticks(ticklist_y, minor=True)
        ax0[i].set_yticks(majorlist_y, minor=False)
        ax0[i].set_yticklabels(majorlabels, minor=False, rotation=90)

        ax0[i].set_xlabel(r"$k_{\parallel}$ [Mpc$^{-1}$]")
        ax0[i].set_ylabel(r"$k_{\bot}$ [Mpc$^{-1}$]", rotation=90)
        ax0[i].set_xlim(0, 1)
        ax0[i].set_ylim(0, 1)

    # plt.savefig('ps_par_vs_perp_nmodes.png')
    # plt.savefig('ps_par_vs_perp_l2.png')
    # ax0[0].set_title(r"$\mathrm{\bf{(a)}}$")
    # ax0[1].set_title(r"$\mathrm{\bf{(b)}}$")
    # ax0[2].set_title(r"$\mathrm{\bf{(c)}}$")
    ax0[0].text(0.07, 0.88, r"$\mathrm{\bf{(a)}}$", color="w")
    ax0[1].text(0.07, 0.88, r"$\mathrm{\bf{(b)}}$", color="w")
    ax0[2].text(0.07, 0.88, r"$\mathrm{\bf{(c)}}$", color="w")

    fig0.tight_layout(pad=0)
    if outname != None:
        # plt.savefig(outname, bbox_inches = "tight", pad_inches = 0.0787402, dpi = 1000)
        plt.savefig(outname, bbox_inches="tight")


def get_2d_noise_TF(noisepath, outname=None):

    noisemap = map_cosmo.MapCosmo(noisepath)

    P_noise = power_spectrum.PowerSpectrum(noisemap)

    P_noise2d, k_noise, nmodes_noise = P_noise.calculate_ps(do_2d=True)

    # my_ps.make_h5(outname = "/mn/stornext/d16/cmbco/comap/nils/COMAP_general/data/maps/PS/out.h5")
    noise_TF = P_noise2d / np.nanmax(P_noise2d)

    fig0, ax0 = plt.subplots()
    cmap = "CMRmap"

    img0 = ax0.imshow(
        noise_TF,
        interpolation="none",
        origin="lower",
        extent=[0, 1, 0, 1],
        cmap=cmap,
        vmin=0,
        vmax=1.0,
    )

    def log2lin(x, k_edges):
        loglen = np.log10(k_edges[-1]) - np.log10(k_edges[0])
        logx = np.log10(x) - np.log10(k_edges[0])
        return logx / loglen

    # ax.set_xscale('log')
    minorticks = [
        0.0002,
        0.0003,
        0.0004,
        0.0005,
        0.0006,
        0.0007,
        0.0008,
        0.0009,
        0.002,
        0.003,
        0.004,
        0.005,
        0.006,
        0.007,
        0.008,
        0.009,
        0.02,
        0.03,
        0.04,
        0.05,
        0.06,
        0.07,
        0.08,
        0.09,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        20.0,
        30.0,
        40.0,
        50.0,
        60.0,
        70.0,
        80.0,
        90.0,
        200.0,
        300.0,
        400.0,
        500.0,
        600.0,
        700.0,
        800.0,
        900.0,
    ]

    majorticks = [1.0e-04, 1.0e-03, 1.0e-02, 1.0e-01, 1.0e00, 1.0e01, 1.0e02]
    majorlabels = [
        "$10^{-4}$",
        "$10^{-3}$",
        "$10^{-2}$",
        "$10^{-1}$",
        "$10^{0}$",
        "$10^{1}$",
        "$10^{2}$",
    ]

    xbins = P_noise.k_bin_edges_par

    ticklist_x = log2lin(minorticks, xbins)
    majorlist_x = log2lin(majorticks, xbins)

    ybins = P_noise.k_bin_edges_perp

    ticklist_y = log2lin(minorticks, ybins)
    majorlist_y = log2lin(majorticks, ybins)

    # plt.imshow(np.log10(nmodes), interpolation='none', origin='lower')

    divider = make_axes_locatable(ax0)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig0.colorbar(img0, ax=ax0, cax=cax)
    cbar.set_label(r"$P_\mathrm{noise} / \mathrm{max}(P_\mathrm{noise})$")

    ax0.set_xticks(ticklist_x, minor=True)
    ax0.set_xticks(majorlist_x, minor=False)
    ax0.set_xticklabels(majorlabels, minor=False)
    ax0.set_yticks(ticklist_y, minor=True)
    ax0.set_yticks(majorlist_y, minor=False)
    ax0.set_yticklabels(majorlabels, minor=False)

    ax0.set_xlabel(r"$k_{\parallel}$ [Mpc$^{-1}$]")
    ax0.set_ylabel(r"$k_{\bot}$ [Mpc$^{-1}$]")
    ax0.set_xlim(0, 1)
    ax0.set_ylim(0, 1)

    # plt.savefig('ps_par_vs_perp_nmodes.png')
    # plt.savefig('ps_par_vs_perp_l2.png')
    # ax0[1, 1].set_title(r"$\frac{P_{data}}{P_{sim}}$")
    fig0.tight_layout()
    if outname != None:
        plt.savefig(outname)


def plot_1D_TF(noisepath, outname=None):

    noisemap = map_cosmo.MapCosmo(noisepath)

    P_noise = power_spectrum.PowerSpectrum(noisemap)

    P_noise1d, k, nmodes_noise = P_noise.calculate_ps(do_2d=False)

    TF_noise = P_noise1d / np.nanmax(P_noise1d)
    fonts = {
        "font.family": "serif",
        "axes.labelsize": 14,
        "font.size": 14,
        "legend.fontsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    }
    plt.rcParams.update(fonts)

    fig, ax = plt.subplots(figsize=(6.55, 6))

    ax.plot(
        k, TF_noise, label=r"$\frac{P_\mathrm{noise}}{\mathrm{max}(P_\mathrm{noise})}$"
    )

    ax.legend(loc=0)
    ax.set_xlabel(r"$k$ [Mpc$^{-1}$]")
    ax.set_ylabel(r"$T(k)$")
    ax.set_xscale("log")
    ax.set_xlim(1e-2, 1e0)
    ax.set_ylim(0, 1.0)
    ax.grid()
    fig.tight_layout()
    if outname != None:
        # plt.savefig(outname, bbox_inches = "tight", pad_inches = 0.0787402, dpi = 1000)
        plt.savefig(outname, bbox_inches="tight")


def plot_TF_mean(simpaths, mappaths, noisemappaths, outname1=None, outname2=None):

    n_paths = len(simpaths)
    TFs = []
    TFs_1D = []
    for i in range(n_paths):

        # -------- First file --------
        # Raw cube
        inmap = map_cosmo.MapCosmo(simpaths[i])

        P_in = power_spectrum.PowerSpectrum(inmap)

        P_in2d, k_in, nmodes_in = P_in.calculate_ps(do_2d=True)

        # L2 w/ sim map
        outmap = map_cosmo.MapCosmo(mappaths[i])

        P_out = power_spectrum.PowerSpectrum(outmap)

        P_out2d, k_out, nmodes_out = P_out.calculate_ps(do_2d=True)

        # L2 wo/ sim map
        noisemap = map_cosmo.MapCosmo(noisemappaths[i])

        P_noise = power_spectrum.PowerSpectrum(noisemap)

        P_noise2d, k_noise, nmodes_noise = P_noise.calculate_ps(do_2d=True)

        # my_ps.make_h5(outname = "/mn/stornext/d16/cmbco/comap/nils/COMAP_general/data/maps/PS/out.h5")
        TF = (P_out2d - P_noise2d) / P_in2d
        TFs.append(TF)

        TF_1D, k_1D = get_1D_TF(simpaths[i], mappaths[i], noisemappaths[i])
        TFs_1D.append(TF_1D)

    TFs = np.array(TFs)
    TF_mean = np.mean(TFs, axis=0)

    TFs_1D = np.array(TFs_1D)
    TF_1D_mean = np.mean(TFs_1D, axis=0)

    fig0, ax0 = plt.subplots(2, 3, figsize=(16, 16 * 2 / 3), sharex=True, sharey=True)
    cmap = "CMRmap"
    # cmap = "magma"
    # cmap2 = "RdBu"
    # cmap2 = "bwr"
    cmap2 = "RdBu_r"

    """
    img0 = ax0[0, 0].imshow(TF_mean, interpolation='none', origin='lower', extent=[0, 1, 0, 1], cmap = cmap,
                    vmin = 0, vmax = 1, rasterized=True)
    img1 = ax0[1, 0].imshow(TF_mean, interpolation='none', origin='lower', extent=[0, 1, 0, 1], cmap = cmap,
                    vmin = 0, vmax = 1, rasterized=True)
    img2 = ax0[2, 0].imshow(TF_mean, interpolation='none', origin='lower', extent=[0, 1, 0, 1], cmap = cmap,
                    vmin = 0, vmax = 1, rasterized=True)
    """
    img3 = ax0[0, 0].imshow(
        TFs[0],
        interpolation="none",
        origin="lower",
        extent=[0, 1, 0, 1],
        cmap=cmap,
        vmin=0,
        vmax=1,
        rasterized=True,
    )
    img4 = ax0[0, 1].imshow(
        TFs[1],
        interpolation="none",
        origin="lower",
        extent=[0, 1, 0, 1],
        cmap=cmap,
        vmin=0,
        vmax=1,
        rasterized=True,
    )
    # vmin = 0, vmax = 5.0, rasterized=True)
    img5 = ax0[0, 2].imshow(
        TFs[2],
        interpolation="none",
        origin="lower",
        extent=[0, 1, 0, 1],
        cmap=cmap,
        vmin=0,
        vmax=1,
        rasterized=True,
    )
    # norm=colors.SymLogNorm(linthresh=0.01, linscale=0.01,
    #                          vmin=-30, vmax=-1e-2, base=10))

    img6 = ax0[1, 0].imshow(
        TF_mean - TFs[0],
        interpolation="none",
        origin="lower",
        extent=[0, 1, 0, 1],
        cmap=cmap2,
        vmin=-0.25,
        vmax=0.25,
        rasterized=True,
    )
    img7 = ax0[1, 1].imshow(
        TF_mean - TFs[1],
        interpolation="none",
        origin="lower",
        extent=[0, 1, 0, 1],
        cmap=cmap2,
        vmin=-0.25,
        vmax=0.25,
        rasterized=True,
    )
    img8 = ax0[1, 2].imshow(
        TF_mean - TFs[2],
        interpolation="none",
        origin="lower",
        extent=[0, 1, 0, 1],
        cmap=cmap2,
        vmin=-0.25,
        vmax=0.25,
        rasterized=True,
    )

    def log2lin(x, k_edges):
        loglen = np.log10(k_edges[-1]) - np.log10(k_edges[0])
        logx = np.log10(x) - np.log10(k_edges[0])
        return logx / loglen

    # ax.set_xscale('log')
    minorticks = [
        0.0002,
        0.0003,
        0.0004,
        0.0005,
        0.0006,
        0.0007,
        0.0008,
        0.0009,
        0.002,
        0.003,
        0.004,
        0.005,
        0.006,
        0.007,
        0.008,
        0.009,
        0.02,
        0.03,
        0.04,
        0.05,
        0.06,
        0.07,
        0.08,
        0.09,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        20.0,
        30.0,
        40.0,
        50.0,
        60.0,
        70.0,
        80.0,
        90.0,
        200.0,
        300.0,
        400.0,
        500.0,
        600.0,
        700.0,
        800.0,
        900.0,
    ]

    majorticks = [1.0e-04, 1.0e-03, 1.0e-02, 1.0e-01, 1.0e00, 1.0e01, 1.0e02]
    majorlabels = [
        "$10^{-4}$",
        "$10^{-3}$",
        "$10^{-2}$",
        "$10^{-1}$",
        "$10^{0}$",
        "$10^{1}$",
        "$10^{2}$",
    ]

    xbins = P_in.k_bin_edges_par

    ticklist_x = log2lin(minorticks, xbins)
    majorlist_x = log2lin(majorticks, xbins)

    ybins = P_in.k_bin_edges_perp

    ticklist_y = log2lin(minorticks, ybins)
    majorlist_y = log2lin(majorticks, ybins)

    """
    divider0 = make_axes_locatable(ax0[0, 0])
    divider1 = make_axes_locatable(ax0[1, 0])
    divider2 = make_axes_locatable(ax0[2, 0])
    """
    divider3 = make_axes_locatable(ax0[0, 0])
    divider4 = make_axes_locatable(ax0[0, 1])
    divider5 = make_axes_locatable(ax0[0, 2])

    divider6 = make_axes_locatable(ax0[1, 0])
    divider7 = make_axes_locatable(ax0[1, 1])
    divider8 = make_axes_locatable(ax0[1, 2])

    """
    cax0 = divider0.append_axes("right", size = "5%", pad = 0.5)
    cax1 = divider1.append_axes("right", size = "5%", pad = 0.5)
    cax2 = divider2.append_axes("right", size = "5%", pad = 0.5)
    """
    cax3 = divider3.append_axes("right", size="5%", pad=0.2)
    cax4 = divider4.append_axes("right", size="5%", pad=0.2)
    cax5 = divider5.append_axes("right", size="5%", pad=0.2)

    cax6 = divider6.append_axes("right", size="5%", pad=0.2)
    cax7 = divider7.append_axes("right", size="5%", pad=0.2)
    cax8 = divider8.append_axes("right", size="5%", pad=0.2)
    """
    cbar0 = fig0.colorbar(img0, ax = ax0[0, 0], cax = cax0, orientation = "horizontal")
    cbar1 = fig0.colorbar(img1, ax = ax0[1, 0], cax = cax1, orientation = "horizontal")
    cbar2 = fig0.colorbar(img2, ax = ax0[2, 0], cax = cax2, orientation = "horizontal")
    """
    cbar3 = fig0.colorbar(img3, ax=ax0[0, 0], cax=cax3)  # , orientation = "horizontal")
    cbar4 = fig0.colorbar(img4, ax=ax0[0, 1], cax=cax4)  # , orientation = "horizontal")
    cbar5 = fig0.colorbar(img5, ax=ax0[0, 2], cax=cax5)  # , orientation = "horizontal")

    cbar6 = fig0.colorbar(img6, ax=ax0[1, 0], cax=cax6)  # , orientation = "horizontal")
    cbar7 = fig0.colorbar(img7, ax=ax0[1, 1], cax=cax7)  # , orientation = "horizontal")
    cbar8 = fig0.colorbar(img8, ax=ax0[1, 2], cax=cax8)  # , orientation = "horizontal")

    cbar3.remove()
    cbar4.remove()
    cbar6.remove()
    cbar7.remove()

    """
    cbar0.set_label(r'$\langle T \rangle$')
    cbar1.set_label(r'$\langle T \rangle$')
    cbar2.set_label(r'$\langle T \rangle$')
    """
    # cbar3.set_label(r'$T$')
    # cbar4.set_label(r'$T$')
    cbar5.set_label(r"$\tilde{T}_{\parallel, \bot}(k)$")

    # cbar6.set_label(r'$\Delta T$')
    # cbar7.set_label(r'$\Delta T$')
    cbar8.set_label(r"$(\langle\tilde{T}\rangle - \tilde{T})_{\parallel, \bot}(k)$")

    for i in range(2):
        for j in range(3):
            ax0[i, j].set_xticks(ticklist_x, minor=True)
            ax0[i, j].set_xticks(majorlist_x, minor=False)
            ax0[i, j].set_xticklabels(majorlabels, minor=False)
            ax0[i, j].set_yticks(ticklist_y, minor=True)
            ax0[i, j].set_yticks(majorlist_y, minor=False)
            ax0[i, j].set_yticklabels(majorlabels, minor=False, rotation=90)

            ax0[i, j].set_xlim(0, 1)
            ax0[i, j].set_ylim(0, 1)

    ax0[1, 0].set_xlabel(r"$k_{\parallel}$ [Mpc$^{-1}$]")
    ax0[1, 1].set_xlabel(r"$k_{\parallel}$ [Mpc$^{-1}$]")
    ax0[1, 2].set_xlabel(r"$k_{\parallel}$ [Mpc$^{-1}$]")

    ax0[0, 0].set_ylabel(r"$k_{\bot}$ [Mpc$^{-1}$]", rotation=90)
    ax0[1, 0].set_ylabel(r"$k_{\bot}$ [Mpc$^{-1}$]", rotation=90)

    ax0[0, 0].text(0.07, 0.12, r"$\mathrm{\bf{(a)}}$", color="gray", fontsize=20)
    ax0[0, 1].text(0.07, 0.12, r"$\mathrm{\bf{(b)}}$", color="gray", fontsize=20)
    ax0[0, 2].text(0.07, 0.12, r"$\mathrm{\bf{(c)}}$", color="gray", fontsize=20)

    ax0[1, 0].text(0.07, 0.12, r"$\mathrm{\bf{(d)}}$", color="gray", fontsize=20)
    ax0[1, 1].text(0.07, 0.12, r"$\mathrm{\bf{(e)}}$", color="gray", fontsize=20)
    ax0[1, 2].text(0.07, 0.12, r"$\mathrm{\bf{(f)}}$", color="gray", fontsize=20)

    fig0.tight_layout(pad=0)
    if outname1 != None:
        # plt.savefig(outname, bbox_inches = "tight", pad_inches = 0.0787402, dpi = 1000)
        plt.savefig(outname1, bbox_inches="tight")

    fonts = {
        "font.family": "serif",
        "axes.labelsize": 22,
        "font.size": 22,
        "legend.fontsize": 22,
        "xtick.labelsize": 22,
        "ytick.labelsize": 22,
    }
    plt.rcParams.update(fonts)

    fig1, ax1 = plt.subplots(1, 2, figsize=(16, 9))

    img = ax1[0].imshow(
        TF_mean,
        interpolation="none",
        origin="lower",
        extent=[0, 1, 0, 1],
        cmap=cmap,
        vmin=0,
        vmax=1,
        rasterized=True,
    )
    ax1[1].plot(k_1D, TF_1D_mean, linewidth=3)

    divider = make_axes_locatable(ax1[0])
    cax = divider.append_axes("bottom", size="5%", pad="15%")
    cbar = fig1.colorbar(img, ax=ax1[0], cax=cax, orientation="horizontal")
    cbar.set_label(r"$\langle \tilde{T}_{\parallel, \bot}(k) \rangle$")

    ax1[0].set_xticks(ticklist_x, minor=True)
    ax1[0].set_xticks(majorlist_x, minor=False)
    ax1[0].set_xticklabels(majorlabels, minor=False)
    ax1[0].set_yticks(ticklist_y, minor=True)
    ax1[0].set_yticks(majorlist_y, minor=False)
    ax1[0].set_yticklabels(majorlabels, minor=False, rotation=90)

    ax1[0].set_xlim(0, 1)
    ax1[0].set_ylim(0, 1)

    ax1[0].set_xlabel(r"$k_{\parallel}$ [Mpc$^{-1}$]")
    ax1[0].set_ylabel(r"$k_{\bot}$ [Mpc$^{-1}$]", rotation=90)

    ax1[1].set_xlabel(r"$k$ [Mpc$^{-1}$]")
    ax1[1].set_ylabel(r"$k$ [Mpc$^{-1}$]", rotation=90)
    ax1[1].set_xscale("log")
    ax1[1].set_ylim(0, 1)
    ax1[1].grid(alpha=0.5)

    fig1.tight_layout(pad=0)
    if outname2 != None:
        # plt.savefig(outname, bbox_inches = "tight", pad_inches = 0.0787402, dpi = 1000)
        plt.savefig(outname2, bbox_inches="tight")
