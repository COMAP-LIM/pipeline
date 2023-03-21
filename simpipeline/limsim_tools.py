from __future__ import print_function
import time
import datetime
import os
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import copy

class empty_table():
    """
    simple Class creating an empty table
    used for halo catalogue and map instances
    """
    def __init__(self):
        pass

    def copy(self):
        """@brief Creates a copy of the table."""
        return copy.copy(self)

def write_time(string_in):
    """
    write time info in as nicely formatted string
    """
    fmt       = '%H:%M:%S on %m/%d/%Y'
    timestamp = datetime.datetime.now().strftime(fmt)
    bar = 72*'-'
    print( '\n\n'+bar )
    print( string_in )
    print( 'Time:      '+timestamp )
    print( bar+'\n' )

    return

def timeme(method):
    """
    writes the time it takes to run a function
    To use, pput above a function definition. eg:
    @timeme
    def Lco_to_map(halos,map):
    """
    def wrapper(*args, **kw):
        startTime = int(round(time.time()))
        result = method(*args, **kw)
        endTime = int(round(time.time()))

        print('  ',endTime - startTime,'sec')
        return result

    return wrapper

""" DOPPLER CONVERSIONS """
def freq_to_z(nuem, nuobs):
    """
    returns a redshift given an observed and emitted frequency
    """
    zval = (nuem - nuobs) / nuobs
    return zval

def nuem_to_nuobs(nuem, z):
    """
    returns the frequency at which an emitted line at a given redshift would be
    observed
    """
    nuobs = nuem / (1 + z)
    return nuobs

def nuobs_to_nuem(nuobs, z):
    """
    returns the frequency at which an observed line at a given redshift would have
    been emitted
    """
    nuem = nuobs * (1 + z)
    return nuem



# Cosmology Functions
# Explicitily defined here instead of using something like astropy
# in order for ease of use on any machine
def hubble(z,h,omegam):
    """
    H(z) in units of km/s
    """
    return h*100*np.sqrt(omegam*(1+z)**3+1-omegam)

def drdz(z,h,omegam):
    return 299792.458 / hubble(z,h,omegam)

def chi_to_redshift(chi, cosmo):
    """
    Transform from redshift to comoving distance
    Agrees with NED cosmology to 0.01% - http://www.astro.ucla.edu/~wright/CosmoCalc.html
    """
    zinterp = np.linspace(0,4,10000)
    dz      = zinterp[1]-zinterp[0]

    chiinterp  = np.cumsum( drdz(zinterp,cosmo.h,cosmo.Omega_M) * dz)
    chiinterp -= chiinterp[0]
    z_of_chi   = sp.interpolate.interp1d(chiinterp,zinterp)

    return z_of_chi(chi)

def redshift_to_chi(z, cosmo):
    """
    Transform from comoving distance to redshift
    Agrees with NED cosmology to 0.01% - http://www.astro.ucla.edu/~wright/CosmoCalc.html
    """
    zinterp = np.linspace(0,4,10000)
    dz      = zinterp[1]-zinterp[0]

    chiinterp  = np.cumsum( drdz(zinterp,cosmo.h,cosmo.Omega_M) * dz)
    chiinterp -= chiinterp[0]
    chi_of_z   = sp.interpolate.interp1d(zinterp,chiinterp)

    return chi_of_z(z)


def plot_results(mapinst,k,Pk,Pk_sampleerr,params):
    """
    Plot central frequency map and or powerspectrum
    """
    if params.verbose: print("\n\tPlotting results")

    ### Plot central frequency map
    plt.rcParams['font.size'] = 16
    if params.plot_cube:
        plt.figure().set_tight_layout(True)
        im = plt.imshow(np.log10(mapinst.maps[:,:,params.nmaps//2]+1e-6), extent=[-mapinst.fov_x/2,mapinst.fov_x/2,-mapinst.fov_y/2,mapinst.fov_y/2],vmin=-1,vmax=2)
        plt.colorbar(im,label=r'$log_{10}\ T_b\ [\mu K]$')
        plt.xlabel('degrees',fontsize=20)
        plt.ylabel('degrees',fontsize=20)
        plt.title('simulated map at {0:.3f} GHz'.format(mapinst.nu_bincents[params.nmaps//2]),fontsize=24)
        plt.savefig(params.plot_cube_file)

    if params.plot_pspec:
        plt.figure().set_tight_layout(True)
        plt.errorbar(k,k**3*Pk/(2*np.pi**2),k**3*Pk_sampleerr/(2*np.pi**2),
                     lw=3,capsize=0)
        plt.gca().set_xscale('log')
        plt.gca().set_yscale('log')
        plt.grid(True)
        plt.xlabel('k [1/Mpc]',fontsize=18)
        plt.ylabel('$\\Delta^2(k)$ [$\\mu$K$^2$]',fontsize=18)
        plt.title('simulated line power spectrum',fontsize=24)
        plt.savefig(params.plot_pspec_file)

    if params.plot_cube or params.plot_pspec:
        plt.show()
        backpath = os.path.normpath(os.getcwd() + os.sep + os.pardir)
        os.chdir(backpath)

    return
