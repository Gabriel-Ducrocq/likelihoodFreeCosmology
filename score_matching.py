import torch as torch
import numpy as np


def f(theta):
    return theta**2

theta = np.random.uniform(size=100000)
data = np.random.normal(size=100000) + f(theta)
