import utils_mh
import numpy as np
from numba import prange, njit


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

observed_cls = np.zeros((2499, 3,3))
observed_cls[:, 0, 0] = observed_cls_tt
observed_cls[:, 1, 1] = observed_cls_ee
observed_cls[:, 2, 2] = observed_cls_bb
observed_cls[:, 1, 0] = observed_cls_te
observed_cls[:, 0, 1] = observed_cls_te

true_theta = all_theta[19369:][100]
mu = 0.96993044
std = 3.05267139e-04
right_part = np.linspace(mu, mu+2*std, 100)
left_part = np.linspace(mu, mu-2*std, 100)


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
        log_lik_ell[m] = -((2*m+1)/2) * utils_mh.compute_trace(utils_mh.matrix_product(cls_hat[m-2], cls_true_inv[m-2])) - ((2*m+1)/2) * np.log(utils_mh.compute_3x3_det(cls_true[m-2]))
        #print("Log lik ell", str(m), log_lik_ell[m])

    return np.sum(log_lik_ell)


def compute_log_lik_conditional(param):
    theta_input = np.concatenate([np.array([param]), true_theta[1:]])
    cls_true = utils_mh.generate_matrix_cls(theta_input, pol=True)
    true_inv = utils_mh.invert_all_matrices(cls_true)
    log_lik = compute_log_likelihood(observed_cls, cls_true, true_inv)
    return log_lik


all_right_log_lik = []
all_left_log_lik = []
for ind in range(1000):
    print(ind)
    right_param = right_part[ind]
    right_log_lik = compute_log_lik_conditional(right_param)
    all_right_log_lik.append(right_log_lik)
    right_log_lik_array = np.array(all_right_log_lik)
    np.save("right_log_lik", right_log_lik_array)

    left_param = left_part[ind]
    left_log_lik = compute_log_lik_conditional(left_param)
    all_left_log_lik.append(left_log_lik)
    left_log_lik_array = np.array(all_left_log_lik)
    np.save("left_log_lik", left_log_lik_array)



