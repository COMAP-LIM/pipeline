import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.interpolate
import sys
import os
# from .limsim_tools import *
from limsim_tools import *
# from .load_halos import *

sfr_interp_tab = None

@timeme
def Mhalo_to_Ls(halos, params):
    """
    wrapper function to calculate the CO luminosities and (if required) tracer
    luminosities for the full halo catalogue. returns a halo catalogue object
    with two new values for each halo: Lco (the CO luminosity) and Lcat (the
    luminosity of the catalogue tracer). pulls relavent model parameters out
    of the params object
    """

    # if no random number generator seed is set, give it one
    try:
        seed = params.seed
    except AttributeError:
        params.seed = 12345

    try:
        scatterless = params.save_scatterless_lums
    except AttributeError:
        params.save_scatterless_lums = None

    # CO luminosities without the lognormal scatter
    halos.Lco, params = Mhalo_to_Lco(halos, params)

    # catalogue luminosities
    if params.catalog_model:
        halos.Lcat, params = Mhalo_to_Lcatalog(halos, params)

        # for testing--save luminosity values directly from the model (no scatter)
        if params.save_scatterless_lums:
            halos.scatterless_Lco = copy.deepcopy(halos.Lco)
            halos.scatterless_Lcat = copy.deepcopy(halos.Lcat)

        # joint scatter
        halos = add_co_tracer_dependant_scatter(halos, params.rho, params.codex, params.catdex, params.seed)

    else:
        if params.save_scatterless_lums:
            halos.scatterless_Lco = copy.deepcopy(halos.Lco)

        # co-only scatter
        halos.Lco = add_log_normal_scatter(halos.Lco, params.codex, params.seed)



def Mhalo_to_Lco(halos, params):
    """
    General function to get L_co(M_halo) given a certain model <model>
    if adding your own model follow this structure,
    and simply specify the model to use in the parameter file
    will output halo luminosities in **L_sun**

    Parameters
    ----------
    halos : class
        Contains all halo information (position, redshift, etc..)
    params: class
        must contain 'model' (str pointing to one of the CO painting models) and
        'coeffs': either none (use default coeffs for model) or an array with the
        input model coefficients
    """
    dict = {'Li':          Mhalo_to_Lco_Li,
            'Li_sc':       Mhalo_to_Lco_Li_sigmasc,
            'Padmanabhan': Mhalo_to_Lco_Padmanabhan,
            'fiuducial':   Mhalo_to_Lco_fiuducial,
            'Yang':        Mhalo_to_Lco_Yang,
            'arbitrary':   Mhalo_to_Lco_arbitrary,
            }

    model = params.model

    if model in dict.keys():
        return dict[model](halos, params)

    else:
        sys.exit('\n\n\tYour model, '+model+', does not seem to exist\n\t\tPlease check src/halos_to_luminosity.py to add it\n\n')


def Mhalo_to_Lco_Li(halos, params):
    """
    halo mass to SFR to L_CO
    following the Tony li 2016 model
    arXiv 1503.08833
    """

    try:
        coeffs = params.coeffs
    except AttributeError:
        coeffs = None

    if coeffs is None:
        # Power law parameters from paper
        log_delta_mf,alpha,beta,sigma_sfr,sigma_lco = (
            0.0, 1.37,-1.74, 0.3, 0.3)
    else:
        log_delta_mf,alpha,beta,sigma_sfr,sigma_lco = coeffs;
    delta_mf = 10**log_delta_mf;

    # Get Star formation rate
    if not hasattr(halos,'sfr'):
        halos.sfr = Mhalo_to_sfr_Behroozi(halos, sigma_sfr);

    # infrared luminosity
    lir      = halos.sfr * 1e10 / delta_mf
    alphainv = 1./alpha
    # Lco' (observers units)
    Lcop     = lir**alphainv * 10**(-beta * alphainv)
    # Lco in L_sun
    Lco      =  4.9e-5 * Lcop
    Lco      = add_log_normal_scatter(Lco, sigma_lco, 2)

    if params.verbose: print('\n\tMhalo to Lco calculated')

    params.codex = sigma_lco

    return Lco, params

def Mhalo_to_Lco_Li_sigmasc(halos, params):
    """
    halo mass to SFR to L_CO
    following the Tony li 2016 model
    arXiv 1503.08833

    DD 2022 - updated to include a single lognormal scatter coeff
    """

    try:
        coeffs = params.coeffs
    except AttributeError:
        coeffs = None

    if coeffs is None:
        # Power law parameters from paper
        log_delta_mf,alpha,beta,sigma_sc = (
            0.0, 1.37,-1.74, 0.3)
    else:
        log_delta_mf,alpha,beta,sigma_sc = coeffs;
    delta_mf = 10**log_delta_mf;

    params.codex = sigma_sc

    # Get Star formation rate
    if not hasattr(halos,'sfr'):
        halos.sfr = Mhalo_to_sfr_Behroozi(halos, sigma_sc);

    # infrared luminosity
    lir      = halos.sfr * 1e10 / delta_mf
    alphainv = 1./alpha
    # Lco' (observers units)
    Lcop     = lir**alphainv * 10**(-beta * alphainv)
    # Lco in L_sun
    Lco      =  4.9e-5 * Lcop

    if params.verbose: print('\n\tMhalo to Lco calculated')

    return Lco, params

def Mhalo_to_Lco_Padmanabhan(halos, params):
    """
    halo mass to L_CO
    following the Padmanabhan 2017 model
    arXiv 1706.01471
    """

    try:
        coeffs = params.coeffs
    except AttributeError:
        coeffs = None

    if coeffs is None:
        m10,m11,n10,n11,b10,b11,y10,y11 = (
            4.17e12,-1.17,0.0033,0.04,0.95,0.48,0.66,-0.33)
    else:
        m10,m11,n10,n11,b10,b11,y10,y11 = coeffs

    z  = halos.redshift
    hm = halos.M

    m1 = 10**(np.log10(m10)+m11*z/(z+1))
    n  = n10 + n11 * z/(z+1)
    b  = b10 + b11 * z/(z+1)
    y  = y10 + y11 * z/(z+1)

    Lprime = 2 * n * hm / ( (hm/m1)**(-b) + (hm/m1)**y )
    Lco    = 4.9e-5 * Lprime

    params.codex = -1

    return Lco

def Mhalo_to_Lco_fiuducial(halos, params):
    """
    DD 2022, based on Chung+2022 fiuducial model
    arXiv 2111.05931
    """

    try:
        coeffs = params.coeffs
    except AttributeError:
        coeffs = None

    if coeffs is None:
        # default to UM+COLDz+COPSS model from Chung+22
        coeffs = (
            -2.85, -0.42, 10.63, 12.3, 0.42)
        halos.model_coeffs = coeffs
        A, B, logC, logM, sigma = coeffs
    else:
        A,B,logC,logM,sigma = coeffs
        halos.model_coeffs = coeffs

    Mh = halos.M

    C = 10**logC
    M = 10**logM

    Lprime = C / ((Mh/M)**A + (Mh/M)**B)
    Lco = 4.9e-5 * Lprime

    params.codex = sigma

    return Lco, params

def Mhalo_to_Lco_Yang(halos, params):
    """
    DD 2022, SAM from Breysse+2022/Yang+2021
    arXiv 2111.05933/2108.07716
    Not set up for anything other than CO(1-0) at COMAP redshifts currently
    becasue the model is a pretty complicated function of redshift
    for other models edit function directly with parameters from Yang+22
    """

    try:
        coeffs = params.coeffs
    except AttributeError:
        coeffs = None


    if coeffs is not None:
        print('The function is only set up for CO(1-0), 1<z<4')
        return 0

    z = halos.redshift
    Mh = halos.M

    # Lco function
    logM1 = 12.13 - 0.1678*z
    logN = -6.855 + 0.2366*z - 0.05013*z**2
    alpha = 1.642 + 0.1663*z - 0.03238*z**2
    beta = 1.77*np.exp(-1/2.72) - 0.00827

    M1 = 10**logM1
    N = 10**logN

    Lco = 2*N * Mh / ((Mh/M1)**(-alpha) + (Mh/M1)**(-beta))

    # fduty function
    logM2 = 11.73 + 0.6634*z
    gamma = 1.37 - 0.190*z + 0.0215*z**2

    M2 = 10**logM2

    fduty = 1 / (1 + (Mh/M2)**gamma)

    Lco = Lco * fduty

    # scatter
    sigmaco = 0.357 - 0.0701*z + 0.00621*z**2

    params.codex = sigmaco

    return Lco


def Mhalo_to_Lco_arbitrary(halos, params):
    """
    halo mass to L_CO
    allows for utterly arbitrary models!
    coeffs:
        coeffs[0] is a function that takes halos as its only argument
        coeffs[1] is a boolean: do we need to calculate sfr or not?
        coeffs[2] is optional sigma_sfr
        coeffs[3] is optional argument that must almost never be invoked
        alternatively, if coeffs is callable, then assume we calculate sfr
            default sigma_sfr is 0.3 dex
    if sfr is calculated, it is stored as a halos attribute
    """

    try:
        coeffs = params.coeffs
    except AttributeError:
        coeffs = None

    sigma_sfr = 0.3
    bad_extrapolation = False
    if callable(coeffs):
        sfr_calc = True
        lco_func = coeffs
    else:
        lco_func, sfr_calc = coeffs[:2]
        if len(coeffs)>2:
            sigma_sfr = coeffs[2]
        if len(coeffs)>3:
            bad_extrapolation = coeffs[3]
    if sfr_calc:
        halos.sfr = Mhalo_to_sfr_Behroozi(halos, sigma_sfr, bad_extrapolation)

    params.codex = sigma_sfr

    return lco_func(halos), params

def Mhalo_to_sfr_Behroozi(halos, sigma_sfr, bad_extrapolation=False):
    global sfr_interp_tab
    if sfr_interp_tab is None:
        sfr_interp_tab = get_sfr_table(bad_extrapolation)
    sfr = sfr_interp_tab.ev(np.log10(halos.M), np.log10(halos.redshift+1))
    # sfr = add_log_normal_scatter(sfr, sigma_sfr, 1)
    return sfr

def get_sfr_table(bad_extrapolation=False):
    """
    LOAD SFR TABLE from Behroozi+13a,b
    Columns are: z+1, logmass, logsfr, logstellarmass
    Intermediate processing of tabulated data
    with option to extrapolate to unphysical masses
    """

    tablepath = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    tablepath+= '/tables/sfr_behroozi_release.dat'
    dat_zp1, dat_logm, dat_logsfr, _ = np.loadtxt(tablepath, unpack=True)

    dat_logzp1 = np.log10(dat_zp1)
    dat_sfr    = 10.**dat_logsfr

    # Reshape arrays
    dat_logzp1  = np.unique(dat_logzp1)    # log(z), 1D
    dat_logm    = np.unique(dat_logm)    # log(Mhalo), 1D
    dat_sfr     = np.reshape(dat_sfr, (dat_logm.size, dat_logzp1.size))
    dat_logsfr  = np.reshape(dat_logsfr, dat_sfr.shape)

    # optional extrapolation to masses excluded in Behroozi+13
    if bad_extrapolation:
        from scipy.interpolate import SmoothBivariateSpline
        dat_logzp1_,dat_logm_ = np.meshgrid(dat_logzp1,dat_logm)
        badspl = SmoothBivariateSpline(dat_logzp1_[-1000<(dat_logsfr)],dat_logm_[-1000<(dat_logsfr)],dat_logsfr[-1000<(dat_logsfr)],kx=4,ky=4)
        dat_sfr[dat_logsfr==-1000.] = 10**badspl(dat_logzp1,dat_logm).T[dat_logsfr==-1000.]

    # Get interpolated SFR value(s)
    sfr_interp_tab = sp.interpolate.RectBivariateSpline(
                            dat_logm, dat_logzp1, dat_sfr,
                            kx=1, ky=1)
    return sfr_interp_tab


"""
 functions to get the tracer luminosity (not the CO luminosity) based on the halo mass
"""


@timeme
def Mhalo_to_Lcatalog(halos, params):
    """
    General function to get L_catalog(M_halo) given a certain model <model>
    if adding your own model follow this structure,
    and simply specify the model to use in the parameter file
    will output halo luminosities in **L_sun**

    Parameters
    ----------
    halos : class
        Contains all halo information (position, redshift, etc..)
    model : str
        Model to use, specified in the parameter file
    coeffs :
        None for default coeffs
    """

    model = params.catalog_model

    dict = {'default':          Mhalo_to_Lcatalog_test1,
            'test2':          Mhalo_to_Lcatalog_test2
            }

    if model in dict.keys():
        return dict[model](halos, params)

    else:
        sys.exit('\n\n\tYour model, '+model+', does not seem to exist\n\t\tPlease check src/halos_to_luminosity.py to add it\n\n')


def Mhalo_to_Lcatalog_test1(halos, params):
    """
    test model for assigning lums of an arbitrary tracer to halos based on M_halo
    """

    try:
        coeffs = params.coeffs
    except AttributeError:
        coeffs = None


    if coeffs is None:
        # default to scaled version of UM+COLDz+COPSS model from Chung+22 ***
        coeffs = (
            -2, -0.5, 11, 13, 0.5)
        halos.model_coeffs = coeffs
        A, B, logC, logM, sigma = coeffs
    else:
        A,B,logC,logM,sigma = coeffs
        halos.model_coeffs = coeffs

    Mh = halos.M

    C = 10**logC
    M = 10**logM

    Lprime = C / ((Mh/M)**A + (Mh/M)**B)
    Lcatalog = 4.9e-5 * Lprime

    params.catdex = sigma

    return Lcatalog, params

def Mhalo_to_Lcatalog_test2(halos, params):
    """
    test model for assigning lums of an arbitrary tracer to halos based on M_halo
    """

    try:
        coeffs = params.coeffs
    except AttributeError:
        coeffs = None


    if coeffs is None:
        # default to wildly different version of UM+COLDz+COPSS model from Chung+22 ***
        coeffs = (
            0.5, 2, 11, 12, 0.5)
        halos.model_coeffs = coeffs
        A, B, logC, logM, sigma = coeffs
    else:
        A,B,logC,logM,sigma = coeffs
        halos.model_coeffs = coeffs

    Mh = halos.M

    C = 10**logC
    M = 10**logM

    Lprime = C / ((Mh/M)**A + (Mh/M)**B)
    Lcatalog = 4.9e-5 * Lprime

    params.catdex = sigma

    return Lcatalog


def add_log_normal_scatter(data,dex,seed):
    """
    Return array x, randomly scattered by a log-normal distribution with sigma=dexscatter.
    [via @tonyyli - https://github.com/dongwooc/imapper2]
    Note: scatter maintains mean in linear space (not log space).
    """
    if np.any(dex<=0):
        return data
    # Calculate random scalings
    sigma       = dex * 2.302585 # Stdev in log space (DIFFERENT from stdev in linear space), note: ln(10)=2.302585
    mu          = -0.5*sigma**2

    # Set standard seed so changing minimum mass cut
    # does not change the high mass halos
    np.random.seed(seed*13579)
    randscaling = np.random.lognormal(mu, sigma, data.shape)
    xscattered  = np.where(data > 0, data*randscaling, data)

    return xscattered


def add_co_tracer_dependant_scatter(halos, rho, codex, catdex, seed):
    if np.any(np.logical_or(codex <= 0, catdex <= 0)):
        print('passed a negative dex value. not scattering')
        return halos

    # set up a numpy random number generator
    scalerng = np.random.default_rng(seed=seed)

    # parameters for the CO distribution
    sigmaco = codex * 2.30285 # stdev in log space
    muco = -0.5*sigmaco**2

    # parameters for the catalogue tracer distribution
    sigmatr = catdex * 2.30285
    mutr = -0.5*sigmatr**2

    # mean and convariance matrix for the joint distribution
    mean = [0,0]
    cov = [[sigmaco**2, sigmaco*sigmatr*rho],
           [sigmaco*sigmatr*rho, sigmatr**2]]

    # LINEAR normal scalings for co and the halo tracer
    coscale, trscale = scalerng.multivariate_normal(mean, cov, size=len(halos.Lco)).T

    # change those into lognormal scalings (output of this would be the same as pulling from
    # np.random.lognormal for a single variable)
    logscaleco = np.exp(coscale*sigmaco + muco)
    logscaletr = np.exp(trscale*sigmatr + mutr)

    # slap scalings onto existing catalogue and co luminosities
    halos.Lco = halos.Lco*logscaleco
    halos.Lcat = halos.Lcat*logscaletr

    return halos
