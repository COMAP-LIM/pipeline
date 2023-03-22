from __future__ import absolute_import, print_function
import numpy as np
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import astropy.constants as const
import copy
# from  .limsim_tools import *
from  limsim_tools import *

class HaloCatalog():
    """
    designer class for holding a catalogue of simulated halos
    """

    def __init__(self, params, inputfile=None, load_all=False):
        if inputfile:
            self.load(inputfile, params)
            self.cull(params)
        else:
            pass

    def copy(self):
        return copy.deepcopy(self)

    def load(self, filein, params):

        """
        Load peak patch halo catalogue into halos class and cosmology into cosmo class
        taken from George Stein's limlam_mocker package: https://github.com/georgestein/limlam_mocker

        Returns
        -------
        None
        """

        halo_info  = np.load(filein, allow_pickle=True)
        if params.verbose: print("\thalo catalogue contains:\n\t\t", halo_info.files)

        #get cosmology from halo catalogue
        params_dict    = halo_info['cosmo_header'][()]
        Omega_M  = params_dict.get('Omega_M')
        Omega_B  = params_dict.get('Omega_B')
        hvalue        = params_dict.get('h'      )

        self.cosmo = FlatLambdaCDM(H0=100*hvalue * u.km/(u.Mpc*u.s), Om0 = Omega_M, Ob0 = Omega_B)

        self.Omega_L  = params_dict.get('Omega_L')
        self.ns       = params_dict.get('ns'     )
        self.sigma8   = params_dict.get('sigma8' )

        cen_x_fov  = params_dict.get('cen_x_fov', 0.) #if the halo catalogue is not centered along the z axis
        cen_y_fov  = params_dict.get('cen_y_fov', 0.) #if the halo catalogue is not centered along the z axis

        self.M          = halo_info['M']     # halo mass in Msun
        self.x_pos      = halo_info['x']     # halo x position in comoving Mpc
        self.y_pos      = halo_info['y']     # halo y position in comoving Mpc
        self.z_pos      = halo_info['z']     # halo z position in comoving Mpc
        self.vx         = halo_info['vx']    # halo x velocity in km/s
        self.vy         = halo_info['vy']    # halo y velocity in km/s
        self.vz         = halo_info['vz']    # halo z velocity in km/s
        self.redshift   = halo_info['zhalo'] # observed redshift incl velocities
        self.zformation = halo_info['zform'] # formation redshift of halo

        self.nhalo = len(self.M)

        self.chi        = np.sqrt(self.x_pos**2+self.y_pos**2+self.z_pos**2)
        self.ra         = np.arctan2(-self.x_pos,self.z_pos)*180./np.pi - cen_x_fov
        self.dec        = np.arcsin(  self.y_pos/self.chi  )*180./np.pi - cen_y_fov

        assert np.max(self.M) < 1.e17,             "Halos seem too massive"
        assert np.max(self.redshift) < 4.,         "need to change max redshift interpolation in tools.py"

        if params.verbose: print('\n\t%d halos loaded' % self.nhalo)

    def cull(self, params):
        """
        initial halo cut to get rid of all the really irrelevant ones (out of the redshift range,
        below the minimum mass, etc.)
        """

        # convert the limits in frequency to limits in redshift
        params.z_i = freq_to_z(params.nu_rest, params.nu_i)
        params.z_f = freq_to_z(params.nu_rest, params.nu_f)

        # check that the limits are the right way round
        if params.z_i > params.z_f:
            tz = params.z_i
            params.z_i = params.z_f
            params.z_f = tz

        # relevant conditions:
        goodidx = (self.M > params.min_mass) * \
                  (self.redshift >= params.z_i) * \
                  (self.redshift <= params.z_f) * \
                  (np.abs(self.ra) <= params.fov_x/2) * \
                  (np.abs(self.dec) <= params.fov_y/2)

        goodidx = np.where(goodidx)[0]

        self.indexcut(goodidx, in_place=True)

        if params.verbose: print('\n\t%d halos remain after mass/map cut' % self.nhalo)

        # sort halos by mass, so fluctuations in luminosity
        # are the same with any given mass cut
        sortidx = np.argsort(self.M)[::-1]
        self.indexcut(sortidx, in_place=True)

    def get_velocities(self, params):
        """
        assigns each halo a rotation velocity based on its DM mass
        adds the velocity value to self.vbroaden and also returns it
        uses params.velocity_attr:
        if 'vvirincli', calculates the virial velocity and muliplies it by a
           sin(i), i a randomly generated inclination angle, to simulate the
           effects of inclination on line broadening
        if 'vvir', just calculates the virial velocity
        """
        vvir = lambda M,z:35*(M*self.cosmo.H(z).value/1e10)**(1/3) # km/s

        if params.velocity_attr == 'vvirincli':
            # Calculate doppler parameters
            self.sin_i = np.sqrt(1-np.random.uniform(size=self.nhalo)**2)
            self.vvir = vvir(self.M, self.redshift) / 2 #****
            self.vbroaden = self.vvir*self.sin_i/0.866

        elif params.velocity_attr == 'vvir':
            # Calculate doppler parameters
            self.sin_i = np.sqrt(1-np.random.uniform(size=self.nhalo)**2)
            self.vbroaden = vvir(self.M, self.redshift)

        return self.vbroaden




    #### FUNCTIONS TO SLICE THE HALO CATALOGUE IN SOME WAY
    def indexcut(self, idx, in_place=False):
        """
        crops the halo catalogue to only include halos included in the passed index
        array.
        """
        # assert np.max(idx) <= self.nhalo,   "Too many indices"

        if not in_place:
            # new halos object to hold the cut catalogue
            subset = HaloCatalog()

            # copy all the arrays over, indexing as you go
            for i in dir(self):
                if i[0]=='_': continue
                try:
                    setattr(subset, i, getattr(self,i)[idx])
                except TypeError:
                    pass
            subset.nhalo = len(subset.M)

        else:

            # replace all the arrays with an indexed version
            for i in dir(self):
                if i[0]=='_': continue
                try:
                    setattr(self, i, getattr(self,i)[idx])
                except TypeError:
                    pass
                self.nhalo = len(self.M)

        if not in_place:
            return subset


    def attrcut_subset(self, attr, minval, maxval, in_place=False):
        """
        crops the halo catalogue to only include desired halos, based on some arbitrary
        attribute attr. will include haloes with attr from minval to maxval.
        """

        keepidx = np.where(np.logical_and(getattr(self,attr) > minval,
                                          getattr(self,attr) <= maxval))[0]

        if not in_place:
            # new halos object to hold the cut catalogue
            subset = HaloCatalog()

            # copy all the arrays over, indexing as you go
            for i in dir(self):
                if i[0]=='_': continue
                try:
                    setattr(subset, i, getattr(self,i)[keepidx])
                except TypeError:
                    pass
            nhalo = len(subset.M)
            subset.nhalo = nhalo

        else:

            # replace all the arrays with an indexed version
            for i in dir(self):
                if i[0]=='_': continue
                try:
                    setattr(self, i, getattr(self,i)[keepidx])
                except TypeError:
                    pass
                nhalo = len(self.M)
                self.nhalo = nhalo

        if params.verbose: print('\n\t%d halos remain after attribute cut' % nhalo)

        if not in_place:
            return subset


    def masscut_subset(self, min_mass, max_mass, in_place=False):
        """
        cut on mass specifically (for convenience)
        """
        if in_place:
            self.attrcut_subset('M', min_mass, max_mass, in_place=True)
        else:
            return self.attrcut_subset('M', min_mass, max_mass)

    def vmaxcut_subset(self, min_vmax, max_vmax, in_place=False):
        """
        cut on vmax specifically (for convenience)
        """
        if in_place:
            self.attrcut_subset('vmax', min_vmax, max_vmax, in_place=True)
        else:
            return self.attrcut_subset('vmax', min_vmax, max_vmax)


    def write_cat(self, params, trim=None, writeall=False):
        """
        save halos as npz catalog
        """
        if params.verbose: print('\n\tSaving Halo Catalogue to\n\t\t', params.cat_output_file)
        if trim:
            i = trim
        else:
            i = -1

        try:
            velocities = self.vbroaden[:i]
        except AttributeError:
            velocities = np.ones(len(self.dec[:i])) * -99

        if writeall:
            np.savez(params.cat_output_file,
                     dec=self.dec[:i], nhalo=len(self.dec[:i]),
                     nu=self.nu[:i], ra=self.ra[:i],
                     z=self.redshift[:i], vx=self.vx[:i],
                     vz    = self.vz[:i],
                     x_pos   = self.x_pos[:i],
                     y_pos = self.y_pos[:i],
                     z_pos        = self.z_pos[:i],
                     zformation=self.zformation[:i],
                     Lco = self.Lco[:i],
                     Lcat = self.Lcat[:i],
                     vhalo = velocities,
                     M = self.M[:i])
        else:
            np.savez(params.cat_output_file,
                     dec = self.dec[:i], ra = self.ra[:i],
                     z = self.redshift[:i],
                     Lco = self.Lco[:i],
                     Lcat = self.Lcat[:i],
                     vhalo = velocities,
                     M = self.M[:i])

        return
