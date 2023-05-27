import healpy as hp
import numpy as np
from classy import Class
from time import time

COSMO_PARAMS_NAMES = ["n_s", "omega_b", "omega_cdm", "100*theta_s", "ln10^{10}A_s", "tau_reio"] # Parameters names
COSMO_PARAMS_MEAN_PRIOR = np.array([0.9665, 0.02242, 0.11933, 1.04101, 3.047, 0.0561]) # Prior mean
COSMO_PARAMS_SIGMA_PRIOR = np.array([0.0038, 0.00014, 0.00091, 0.00029, 0.014, 0.0071]) # Prior std


cosmo = Class()

def generate_cls(theta, pol = False):
    """
    generates the power spectrum corresponding the input cosmological parameters.
    :param theta: array of float, 6 cosmological parameters.
    :param pol: boolean, whether to compute polarization power spectra.
    :return: arrays of float, size L_max +1, of the power spectra.
    """
    params = {'output': 'tCl pCl lCl',
              'l_max_scalars': 2500,
              'lensing': 'yes'}
    d = {name:val for name, val in zip(COSMO_PARAMS_NAMES, theta)}
    params.update(d)
    cosmo.set(params)
    cosmo.compute()
    cls = cosmo.lensed_cl(2500)
    # 10^12 parce que les cls sont exprimés en kelvin carre, du coup ca donne une stdd en 10^6
    cls_tt = cls["tt"] * 2.7255e6 ** 2
    if not pol:
        cosmo.struct_cleanup()
        cosmo.empty()
        return cls_tt
    else:
        cls_ee = cls["ee"] * 2.7255e6 ** 2
        cls_bb = cls["bb"] * 2.7255e6 ** 2
        cls_te = cls["te"] * 2.7255e6 ** 2
        cosmo.struct_cleanup()
        cosmo.empty()
        return cls_tt, cls_ee, cls_bb, cls_te


all_cls_tt = []
all_cls_ee = []
all_cls_bb = []
all_cls_te = []
all_true_cls_tt = []
all_true_cls_ee = []
all_true_cls_bb = []
all_true_cls_te = []
all_theta = []
start = time()
theta_proposal = np.load("data/mixture_round0.npy")
##4365 is a problem so I skipped 4364 and 4365
for n in range(6434, 22001):
    if n % 1 == 0:
        print(n/22000)

    #theta = COSMO_PARAMS_MEAN_PRIOR  + COSMO_PARAMS_SIGMA_PRIOR*np.random.normal(size=6)

    theta = theta_proposal[n]
    cls_tt, cls_ee, cls_bb, cls_te = generate_cls(theta, True)
    alm_T, alm_E, alm_B = hp.synalm([cls_tt, cls_ee, cls_bb, cls_te, np.zeros(cls_te.shape), np.zeros(cls_te.shape)], lmax= 2500, new=True)
    cls_tt_hat, cls_ee_hat, cls_bb_hat, cls_te_hat, _, _ = hp.alm2cl([alm_T, alm_E, alm_B], lmax=2500)

    all_true_cls_tt.append(cls_tt)
    all_true_cls_ee.append(cls_ee)
    all_true_cls_bb.append(cls_bb)
    all_true_cls_te.append(cls_te)

    all_cls_tt.append(cls_tt_hat)
    all_cls_ee.append(cls_ee_hat)
    all_cls_bb.append(cls_bb_hat)
    all_cls_te.append(cls_te_hat)
    all_theta.append(theta)

    if n % 100 == 0:
        all_cls_tt_array = np.array(all_cls_tt)
        all_cls_ee_array = np.array(all_cls_ee)
        all_cls_bb_array = np.array(all_cls_bb)
        all_cls_te_array = np.array(all_cls_te)
        all_theta_array = np.array(all_theta)

        all_true_cls_tt_array = np.array(all_true_cls_tt)
        all_true_cls_ee_array = np.array(all_true_cls_ee)
        all_true_cls_bb_array = np.array(all_true_cls_bb)
        all_true_cls_te_array = np.array(all_true_cls_te)

        np.save("/home/gabdu45/likelihoodFreeCosmology/data/polarizationGaussianRound1/all_cls_tt4.npy", all_true_cls_tt_array)

        np.save("/home/gabdu45/likelihoodFreeCosmology/data/polarizationGaussianRound1/all_cls_ee4.npy", all_true_cls_ee_array)

        np.save("/home/gabdu45/likelihoodFreeCosmology/data/polarizationGaussianRound1/all_cls_bb4.npy", all_true_cls_bb_array)

        np.save("/home/gabdu45/likelihoodFreeCosmology/data/polarizationGaussianRound1/all_cls_te4.npy", all_true_cls_te_array)

        np.save("/home/gabdu45/likelihoodFreeCosmology/data/polarizationGaussianRound1/all_cls_tt_hat4.npy", all_cls_tt_array)

        np.save("/home/gabdu45/likelihoodFreeCosmology/data/polarizationGaussianRound1/all_cls_ee_hat4.npy", all_cls_ee_array)
        np.save("/home/gabdu45/likelihoodFreeCosmology/data/polarizationGaussianRound1/all_cls_bb_hat4.npy", all_cls_bb_array)
        np.save("/home/gabdu45/likelihoodFreeCosmology/data/polarizationGaussianRound1/all_cls_te_hat4.npy", all_cls_te_array)
        np.save("/home/gabdu45/likelihoodFreeCosmology/data/polarizationGaussianRound1/all_theta4.npy", all_theta_array)




