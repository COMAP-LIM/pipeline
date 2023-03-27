from __future__ import print_function
import time
import datetime
import os
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from astropy.convolution import convolve, Gaussian2DKernel
import copy


class SimMap():

    def __init__(self, params):
        """
        Adds input parameters to be kept by the map class and gets map details
        """

        self.nmaps  = int(params.nmaps)
        self.fov_x  = float(params.fov_x)
        self.fov_y  = float(params.fov_y)
        self.npix_x = int(params.npix_x)
        self.npix_y = int(params.npix_y)
        self.nu_i   = float(params.nu_i)
        self.nu_f   = float(params.nu_f)
        self.nu_rest= float(params.nu_rest)
        self.z_i    = self.nu_rest/self.nu_i - 1
        self.z_f    = self.nu_rest/self.nu_f - 1

        # get arrays describing the final intensity map to be output
        # map sky angle dimension
        self.pix_size_x = self.fov_x/self.npix_x
        self.pix_size_y = self.fov_y/self.npix_y

        # pixel size to convert to brightness temp
        self.Ompix = (self.pix_size_x*np.pi/180)*(self.pix_size_y*np.pi/180)

        self.pix_binedges_x = np.linspace(-self.fov_x/2,self.fov_x/2,self.npix_x+1)
        self.pix_binedges_y = np.linspace(-self.fov_y/2,self.fov_y/2,self.npix_y+1)

        self.pix_bincents_x =  0.5*(self.pix_binedges_x[1:] + self.pix_binedges_x[:-1])
        self.pix_bincents_y =  0.5*(self.pix_binedges_y[1:] + self.pix_binedges_y[:-1])

        # map frequency dimension
        # use linspace to ensure nmaps channels
        self.nu_binedges = np.linspace(self.nu_i,self.nu_f,self.nmaps+1)
        self.dnu         = np.abs(np.mean(np.diff(self.nu_binedges)))
        self.nu_bincents = self.nu_binedges[:-1] - self.dnu/2


    def copy(self):
        return copy.deepcopy(self)


    def mockmapmaker(self, halos, params):
        """
        wrapper function for all forms of mock mapmaking from the generated
        luminosity catalog
        """

        ### Calculate line freq from redshift
        halos.nu  = self.nu_rest/(halos.redshift+1)

        # Transform from Luminosity to Temperature (uK)
        # ... or to flux density (Jy/sr)
        if (params.units=='intensity'):
            if params.verbose: print('\n\tcalculating halo intensities')
            halos.Tco = I_line(halos, self)
        elif (params.units=='temperature'):
            if params.verbose: print('\n\tcalculating halo temperatures')
            halos.Tco = T_line(halos, self)
        else:
            if params.verbose: print('\n\tdefaulting to halo temperatures')
            halos.Tco = T_line(halos, self)

        # BINS HALOS SPECTRALLY BY VVIR, MAKES MAPS OF EACH SUBSET, SMOOTHS THEM, AND
        # THEN COMBINES
        if ~params.freqbroaden or 1==params.bincount:
            subsets = [halos]
        else:
            binattr_val = getattr(halos, params.binattr)
            attr_ranges = np.linspace(min(binattr_val)*(1-1e-16),max(binattr_val), params.bincount+1)
            subsets = [halos.attrcut_subset(halos, params.binattr, v1, v2)
                            for v1,v2 in zip(attr_ranges[:-1], attr_ranges[1:])]

        # SET UP FINER BINNING IN RA, DEC, FREQUENCY
        # (if oversampling isn't necessary these will just be equal to the regular sampling)
        bins3D_fine = [np.linspace(min(self.pix_binedges_x),
                                   max(self.pix_binedges_x),
                                   len(self.pix_binedges_x[1:])*params.xrefine+1),
                       np.linspace(min(self.pix_binedges_y),
                                   max(self.pix_binedges_y),
                                   len(self.pix_binedges_y[1:])*params.xrefine+1),
                       np.linspace(min(self.nu_binedges),
                                   max(self.nu_binedges),
                                   len(self.nu_binedges[1:])*params.freqrefine+1)]

        dx_fine = np.mean(np.diff(bins3D_fine[0]))
        dy_fine = np.mean(np.diff(bins3D_fine[1]))
        dnu_fine = np.mean(np.diff(bins3D_fine[-1]))

        maps = np.zeros((len(self.pix_bincents_x)*params.xrefine,
                         len(self.pix_bincents_y)*params.xrefine,
                         len(self.nu_bincents)))

        if params.freqbroaden:

            try:
                velocities = getattr(halos, 'vbroaden')
            except AttributeError:
                halos.get_velocities(params)

            # pull the relevant parameters out of the params object
            # number of velocity bins to use
            bincount = params.bincount
            # function to turn halo attributes into a line width (in observed freq space)
            # default is vmax/c times observed frequency (if set to None)
            fwhmfunc = params.fwhmfunction
            # number of bins by which to oversample in frequency
            freqrefine = params.freqrefine
            # function used to broaden halo emission based on linewidth
            filterfunc = params.filterfunc
            # if true, will do a fast convolution thing (??)
            lazyfilter = params.lazyfilter


            # bin in RA, DEC, NU_obs
            if fwhmfunc is None:
                # a fwhmfunc is needed to turn halo attributes into a line width
                #   (in observed frequency space)
                # default fwhmfunc based on halos is vmax/c times observed frequency
                fwhmfunc = lambda h:h.nu*h.vbroaden/299792.458

            # for each velocity bin, convolve all halos with a kernel of the same width and then add onto the map
            for i,sub in enumerate(subsets):
                if sub.nhalo < 1: continue;
                maps_fine = np.histogramdd( np.c_[sub.ra, sub.dec, sub.nu],
                                              bins    = bins3D_fine,
                                              weights = sub.Tco )[0]
                if callable(fwhmfunc):
                    sigma = 0.4246609*np.nanmedian(fwhmfunc(sub)) # in freq units (GHz)
                else: # hope it's a number
                    sigma = 0.4246609*fwhmfunc # in freq units (GHz)
                if sigma > 0:
                    if lazyfilter:
                        if lazyfilter=='rehist':
                            # uses fast_histogram assuming map bins are evenly spaced
                            filteridx = fast_histogram.histogram2d(sub.ra,sub.dec,
                                                        (self.npix_x, self.npix_y),
                                                        ((-self.fov_x/2, self.fov_x/2),
                                                         (-self.fov_y/2, self.fov_y/2))) > 0
                        else:
                            filteridx = np.where(np.any(maps_fine, axis=-1))
                        maps_fine[filteridx] = filterfunc(maps_fine[filteridx], sigma/dnu_fine)
                    else:
                        maps_fine = filterfunc(maps_fine, sigma/dnu_fine)
                # collapse back down to the end-stage frequency sampling
                maps += np.sum(maps_fine.reshape((maps_fine.shape[0], maps_fine.shape[1], -1, freqrefine)), axis=-1)
                if params.verbose:
                    print('\n\tsubset {} / {} complete'.format(i,len(subsets)))
            if (params.units=='intensity'):
                maps /= self.Ompix
            # flip back frequency bins and store in object
            self.map = maps[:,:,::-1]

        else:
            # bin in RA, DEC, NU_obs
            if params.verbose: print('\n\tBinning halos into map')
            maps, edges = np.histogramdd( np.c_[halos.ra.flatten(), halos.dec.flatten(), halos.nu.flatten()],
                                          bins    = bins3D_fine,
                                          weights = halos.Tco )
            if (params.units=='intensity'):
                maps /= self.Ompix
            # flip back frequency bins
            self.map = maps[:,:,::-1]

        if params.beambroaden:
            # smooth by the primary beam

            # come up with a convolution kernel approximating the beam if one isn't already passed
            if not params.beamkernel:
                # number of refined pixels corresponding to the fwhm in arcminutes
                std = 4.5 / (2*np.sqrt(2*np.log(2))) / 60 # standard deviation in degrees
                std_pix = std / dx_fine

                beamkernel = Gaussian2DKernel(std_pix)

            if params.verbose:
                print('\nsmoothing by synthesized beam: {} channels total'.format(maps.shape[-1]))

            smoothsimlist = []
            for i in range(maps.shape[-1]):
                smoothsimlist.append(convolve(maps[:,:,i], beamkernel))
                if params.verbose:
                    if i%100 == 0:
                        print('\n\t done {} of {} channels'.format(i, maps.shape[-1]))

            maps_sm_fine = np.stack(smoothsimlist, axis=-1)
            print(maps_sm_fine.shape)

            # rebin
            mapssm = np.sum(maps_sm_fine.reshape((params.npix_x, params.xrefine,
                                                  params.npix_y, params.xrefine, -1)), axis=(1,3))

            if (params.units=='intensity'):
                mapssm/= self.Ompix
            # flip back frequency bins
            self.map = mapssm[:,:,::-1]


    def write(self, params):
        """
        save 3D data cube in .npz format, including map header information
        """
        if params.verbose: print('\n\tSaving Map Data Cube to\n\t\t', params.map_output_file)
        np.savez(params.map_output_file,
                 fov_x=self.fov_x, fov_y=self.fov_y,
                 pix_size_x=self.pix_size_x, pix_size_y=self.pix_size_y,
                 npix_x=self.npix_x, npix_y=self.npix_y,
                 map_pixel_ra    = self.pix_bincents_x,
                 map_pixel_dec   = self.pix_bincents_y,
                 map_frequencies = self.nu_bincents,
                 map_cube        = self.map)

        return



""" UNIT CONVERSIONS """
def I_line(halos, map):
    '''
     calculates I_line = L_line/4/pi/D_L^2/dnu
     output units of Jy/sr
     assumes L_line in units of L_sun, dnu in GHz

     then 1 L_sun/Mpc**2/GHz = 4.0204e-2 Jy/sr
    '''
    convfac = 4.0204e-2 # Jy/sr per Lsol/Mpc/Mpc/GHz
    Ico     = convfac * halos.Lco/4/np.pi/halos.chi**2/(1+halos.redshift)**2/map.dnu

    return Ico


def T_line(halos, map):
    """
    The line Temperature in Rayleigh-Jeans limit
    T_line = c^2/2/kb/nuobs^2 * I_line

     where the Intensity I_line = L_line/4/pi/D_L^2/dnu
        D_L = D_p*(1+z), I_line units of L_sun/Mpc^2/Hz

     T_line units of [L_sun/Mpc^2/GHz] * [(km/s)^2 / (J/K) / (GHz) ^2] * 1/sr
        = [ 3.48e26 W/Mpc^2/GHz ] * [ 6.50966e21 s^2/K/kg ]
        = 2.63083e-6 K = 2.63083 muK
    """
    convfac = 2.63083
    Tco     = 1./2*convfac/halos.nu**2 * halos.Lco/4/np.pi/halos.chi**2/(1+halos.redshift)**2/map.dnu/map.Ompix

    return Tco
