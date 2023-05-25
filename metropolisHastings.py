import numpy as np
from numba import prange, njit
import utils_mh
import time


all_theta = []
all_cls_tt = []
all_cls_ee = []
all_cls_bb = []
all_cls_te = []
for i in range(1, 4):
    if i == 1:
        all_theta.append(np.load("data/polarizationGaussianPrior/all_theta.npy"))
        all_cls_tt.append(np.load("data/polarizationGaussianPrior/all_cls_tt_hat.npy"))
        all_cls_ee.append(np.load("data/polarizationGaussianPrior/all_cls_ee_hat.npy"))
        all_cls_bb.append(np.load("data/polarizationGaussianPrior/all_cls_bb_hat.npy"))
        all_cls_te.append(np.load("data/polarizationGaussianPrior/all_cls_te_hat.npy"))
        #print(np.load("data/polarizationGaussianPrior/all_theta.npy").shape)
    else:
        all_theta.append(np.load("data/polarizationGaussianPrior/all_theta" + str(i) + ".npy"))
        all_cls_tt.append(np.load("data/polarizationGaussianPrior/all_cls_tt_hat" + str(i) + ".npy"))
        all_cls_ee.append(np.load("data/polarizationGaussianPrior/all_cls_ee_hat" + str(i) + ".npy"))
        all_cls_bb.append(np.load("data/polarizationGaussianPrior/all_cls_bb_hat" + str(i) + ".npy"))
        all_cls_te.append(np.load("data/polarizationGaussianPrior/all_cls_te_hat" + str(i) + ".npy"))
        #print(np.load("data/polarizationGaussianPrior/all_theta" + str(i) + ".npy").shape)

all_cls_tt = np.vstack(all_cls_tt)
all_cls_ee = np.vstack(all_cls_ee)
all_cls_bb = np.vstack(all_cls_bb)
all_cls_te = np.vstack(all_cls_te)
all_theta = np.vstack(all_theta)

all_theta = all_theta[:, :]
all_cls_tt = all_cls_tt[:, 2:]
all_cls_ee = all_cls_ee[:, 2:]
all_cls_bb = all_cls_bb[:, 2:]
all_cls_te = all_cls_te[:, 2:]

observed_cls_tt = all_cls_tt[19369:][100]
observed_cls_ee = all_cls_ee[19369:][100]
observed_cls_bb = all_cls_bb[19369:][100]
observed_cls_te = all_cls_te[19369:][100]

true_theta = all_theta[19369:][100]

observed_cls = np.zeros((2499, 3,3))
observed_cls[:, 0, 0] = observed_cls_tt
observed_cls[:, 1, 1] = observed_cls_ee
observed_cls[:, 2, 2] = observed_cls_bb
observed_cls[:, 1, 0] = observed_cls_te
observed_cls[:, 0, 1] = observed_cls_te


print("observed CLS shape:", observed_cls.shape)

COSMO_PARAMS_NAMES = ["n_s", "omega_b", "omega_cdm", "100*theta_s", "ln10^{10}A_s", "tau_reio"] # Parameters names
COSMO_PARAMS_MEAN_PRIOR = np.array([0.9665, 0.02242, 0.11933, 1.04101, 3.047, 0.0561]) # Prior mean
COSMO_PARAMS_SIGMA_PRIOR = np.array([0.0038, 0.00014, 0.00091, 0.00029, 0.014, 0.0071]) # Prior std
proposal_std = COSMO_PARAMS_SIGMA_PRIOR*0.1
prior_std = COSMO_PARAMS_SIGMA_PRIOR
prior_mean = COSMO_PARAMS_MEAN_PRIOR

@njit()
def compute_log_likelihood(cls_hat, cls_true, cls_true_inv):
    """

    :param cls_hat: matrices 3x3 of observed power spectrum
    :param cls_true: matrices 3x3 of true power spectrum, obtained with class
    :param cls_true_inv: matrix 3x3 of inverted power spectrum
    :return: log likelhood
    """
    ##Adding 2 because we don't count monopole and dipole, so  in fact l = 0 is l= 2
    l = cls_true.shape[0] + 2
    log_lik_ell = np.zeros(l)
    for m in prange(2, l):
        ##We use a -2 offset because in the cls_hat and cls_true l=2 is in fact at index 0
        log_lik_ell[m] = -((2*m+1)/2) * utils_mh.compute_trace(utils_mh.matrix_product(cls_hat[m-2], cls_true_inv[m-2])) \
                         - ((2*m+1)/2) * np.log(utils_mh.compute_3x3_det(cls_true[m-2]))
        #print("Log lik ell", str(m), log_lik_ell[m])

    return np.sum(log_lik_ell)


def compute_log_prior(theta):
    return -0.5*np.sum((theta - prior_mean)**2/prior_std**2)


def compute_log_ratio(theta_new, cls_true_new, cls_true_inv_new, theta, cls_true, cls_true_inv, cls_hat):
    log_r = compute_log_likelihood(cls_hat, cls_true_new, cls_true_inv_new)+ compute_log_prior(theta_new) \
        - compute_log_likelihood(cls_hat, cls_true, cls_true_inv) - compute_log_prior(theta)

    print("Log ratio:", log_r)
    return log_r

def propose_theta(theta_old):
    theta_new = np.random.normal(size = 6)*proposal_std + theta_old
    return theta_new


def metropolis(theta_init, cls_hat, n_iter=5000, lmax=2500, pol=True):
    all_theta = []
    all_theta.append(theta_init)
    theta = theta_init
    #If we target a conditional
    #theta[1:] = true_theta[1:]
    cls_true = utils_mh.generate_matrix_cls(theta, pol=pol)
    inv_cls_true = utils_mh.invert_all_matrices(cls_true)
    start = time.time()
    for l in range(1,n_iter+1):
        if (l+1)%100==0:
            end = time.time()
            print("Iteration", l)
            print("Duration:", end - start)
            start = time.time()

        theta_new = propose_theta(theta)
        #If we target a condtional
        #theta_new[1:] = true_theta[1:]
        cls_true_new = utils_mh.generate_matrix_cls(theta_new, pol=pol)
        inv_cls_true_new = utils_mh.invert_all_matrices(cls_true_new)

        log_ratio = compute_log_ratio(theta_new, cls_true_new, inv_cls_true_new, theta, cls_true, inv_cls_true, cls_hat)
        if np.log(np.random.uniform()) < log_ratio:
            theta = theta_new.copy()
            cls_true = cls_true_new.copy()
            inv_cls_true = inv_cls_true_new.copy()

        all_theta.append(theta.copy())
        np.save("trace_plot.npy",np.array(all_theta))

    return np.array(all_theta)


if __name__== "__main__":
    #theta_init = np.array([0.9700805, 0.02216023, 0.12027733, 1.04093185, 3.04730308,
    #       0.05417271])
    metropolis(true_theta, observed_cls)









