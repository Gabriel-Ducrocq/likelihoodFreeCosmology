import healpy as hp
from classy import Class
import numpy as np


cosmo = Class()

LENSING = 'yes'
OUTPUT_CLASS = 'tCl pCl lCl'
observations = None

NSIDE = 256 # NSIDE for generating the pixel grid over the sphere.
Npix = 12 * NSIDE ** 2 # Number of pixels
L_MAX_SCALARS=int(2*NSIDE)
COSMO_PARAMS_NAMES = ["n_s", "omega_b", "omega_cdm", "100*theta_s", "ln10^{10}A_s", "tau_reio"] # Parameters names
COSMO_PARAMS_MEAN_PRIOR = np.array([0.9665, 0.02242, 0.11933, 1.04101, 3.047, 0.0561]) # Prior mean
COSMO_PARAMS_SIGMA_PRIOR = np.array([0.0038, 0.00014, 0.00091, 0.00029, 0.014, 0.0071]) # Prior std

def generate_cls(theta, pol = True):
    """
    generates the power spectrum corresponding the input cosmological parameters.
    :param theta: array of float, 6 cosmological parameters.
    :param pol: boolean, whether to compute polarization power spectra.
    :return: arrays of float, size L_max +1, of the power spectra.
    """
    params = {'output': OUTPUT_CLASS,
              "modes":"s,t",
              "r":0.001,
              'l_max_scalars': L_MAX_SCALARS,
              'lensing': LENSING}
    d = {name:val for name, val in zip(COSMO_PARAMS_NAMES, theta)}
    params.update(d)
    cosmo.set(params)
    cosmo.compute()
    cls = cosmo.lensed_cl(L_MAX_SCALARS)
    # 10^12 parce que les cls sont exprimés en kelvin carré, du coup ça donne une stdd en 10^6
    # 10^12 because cls are expressed in kelvin squared, so it gives a stdd in 10^6
    cls_tt = cls["tt"]*2.7255e6**2
    if not pol:
        cosmo.struct_cleanup()
        cosmo.empty()
        return cls_tt
    else:
        cls_ee = cls["ee"]*2.7255e6**2
        cls_bb = cls["bb"]*2.7255e6**2
        cls_te = cls["te"]*2.7255e6**2
        cosmo.struct_cleanup()
        cosmo.empty()
        return [cls_tt, cls_ee, cls_bb, cls_te]

theta = COSMO_PARAMS_MEAN_PRIOR + np.random.normal(size=6)*COSMO_PARAMS_SIGMA_PRIOR #simulate from the prior
pow_spec = generate_cls(theta) #getting the corresponding temperature  polarization power spectra.
alms = hp.synalm(pow_spec, lmax=L_MAX_SCALARS, new=True) #sampling T, E and B skymap expressed in spherical harmonics.
cls_tt_hat, cls_ee_hat, cls_bb_hat, cls_te_hat = hp.alm2cl(alms, lmax=L_MAX_SCALARS) #Computing the "observed" power spectra from the skymap alms.
