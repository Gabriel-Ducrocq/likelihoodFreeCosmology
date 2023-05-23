import os
from classy import Class
from torch import nn
import torch
import math
from numba import njit, prange
import numpy as np
from time import time

cosmo = Class()

LENSING = 'yes'
OUTPUT_CLASS = 'tCl pCl lCl'
observations = None

NSIDE = 256 # NSIDE for generating the pixel grid over the sphere.
Npix = 12 * NSIDE ** 2 # Number of pixels
L_MAX_SCALARS=int(2*NSIDE)
COSMO_PARAMS_NAMES = ["n_s", "omega_b", "omega_cdm", "100*theta_s", "ln10^{10}A_s", "tau_reio"] # Parameters names

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        """
        Get the positional encoding of times t
        :param time: torch.tensor (N_batch,) of float, corresponding to the sampling times
        :return: torch.tensor (N_batch, dim), the positionql encodings.
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings




@njit()
def invert_2x2(mat):
    det = mat[0, 0]*mat[1,1] - mat[0, 1]*mat[1,0]
    inv = np.zeros((2,2))
    inv[0,0] = mat[1,1]
    inv[1,1] = mat[0,0]
    inv[0,1] = -mat[0,1]
    inv[1,0] = -mat[1,0]
    return inv/det


@njit()
def invert_3x3(mat):
    inv = np.zeros((3,3))
    m = invert_2x2(mat[:2, :2])
    inv[:2, :2] = m
    inv[2,2] = 1/mat[2,2]
    return inv

@njit(parallel=True)
def invert_all_matrices(matrices):
    l = matrices.shape[0]
    inverse_matrices = np.zeros((l, 3, 3))
    for i in prange(l):
        m = matrices[i]
        inverse_matrices[i] = invert_3x3(m)

    return inverse_matrices


@njit()
def matrix_product(mat1, mat2):
    prod = np.zeros((3,3))
    prod[2,2] = mat1[2,2]*mat2[2,2]
    prod[0,0] = mat1[0,0]*mat2[0,0] + mat1[0,1]*mat2[1,0]
    prod[1,1] = mat1[1, 0] * mat2[0, 1] + mat1[1, 1] * mat2[1, 1]
    prod[0, 1] = mat1[0, 0] * mat2[0, 1] + mat1[0, 1] * mat2[1, 1]
    prod[1, 0] = mat1[1, 0] * mat2[0, 0] + mat1[1, 1] * mat2[1, 0]
    return prod


@njit()
def compute_trace(mat):
    return mat[0, 0] + mat[1, 1] + mat[2,2]


@njit()
def compute_3x3_det(mat):
    det_2x2 = mat[0, 0] * mat[1, 1] - mat[0, 1] * mat[1, 0]
    return det_2x2*mat[2,2]


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
        return cls_tt, cls_ee, cls_bb, cls_te


def generate_matrix_cls(theta, pol = True):
    cls_tt, cls_ee, cls_bb, cls_te = generate_cls(theta, pol=pol)
    cls_true = np.zeros((L_MAX_SCALARS, 3, 3))
    cls_true[:, 0,0] = cls_tt
    cls_true[:, 1, 1] = cls_ee
    cls_true[:, 2, 2] = cls_bb
    cls_true[:, 0, 1] = cls_te
    cls_true[:, 1, 0] = cls_te

    return cls_true[2:]


if __name__== "__main__":
    a = np.random.normal(size=(3,3))
    a[2, 0] = a[2, 1] = a[0, 2] = a[1, 2] = 0

    b = np.random.normal(size=(3,3))
    b[2, 0] = b[2, 1] = b[0, 2] = b[1, 2] = 0

    prod = np.dot(a,b)
    prod_numba = matrix_product(a,b)
    print("Checking ,atrix product")
    print(prod - prod_numba)

    tr_numba = compute_trace(a)
    tr = np.trace(a)
    print("Checking trace")
    print(tr_numba - tr)

    m = np.dot(a, a.T) + np.eye(3)
    inv_numba = invert_3x3(m)
    inv = np.linalg.inv(m)
    print("Checking inversion 3x3")
    print(m)
    print(inv - inv_numba)

    for_inv = []
    for i in range(512):
        a = np.random.normal(size=(3, 3))
        a[2, 0] = a[2, 1] = a[0, 2] = a[1, 2] = 0
        m = np.dot(a, a.T) + np.eye(3)
        for_inv.append(m[None, :, :])

    all_mat = np.vstack(for_inv)
    start = time()
    inv_numba = invert_all_matrices(all_mat)
    end = time()
    print("Duration numba:", end - start)

    inverted = []
    start = time()
    for i in range(512):
        inverted.append(np.linalg.inv(all_mat[i, :, :])[None, :, :])

    end = time()
    print("Duration:", end - start)
    inverted = np.vstack(inverted)
    print(inv_numba - inverted)





