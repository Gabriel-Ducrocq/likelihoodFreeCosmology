import matplotlib.pyplot as plt
import numpy as np
import torch as torch
from utils import SinusoidalPositionEmbeddings


class BrownianBridge:

    def __init__(self, dimension, a=3, b=4, T0 = 0, TN= 1, use_unet=False):
        """
        Brownian Bridge diffusion object, with drift sigma_t.
        :param dimension: integer, dimension of the state space
        :param sigma_t: real valued function taking time as input, drift function
        :param int_sigma_sq_t: real valued function, taking integral bounds as input, integrting the square of sigma
        :param T0: integer, starting time.
        :param TN: integer, ending time
        """

        self.a = a
        self.b = b
        self.dimension = dimension
        self.T0 = T0
        self.TN = TN
        self.use_unet = use_unet
        self.pos_embbed = SinusoidalPositionEmbeddings(9996)


    def sigma_t(self, t):
        return self.a*np.exp(-self.b*t)

    def int_sigma_sq_t(self, t0, t):
        return - self.a**2/(2*self.b) * (np.exp(-2*self.b*t) - np.exp(-2*self.b*t0))

    def get_means(self, t, x0, xT,  t0=0, T=1):
        """
        get the conditionnal mean of the Brownian bridge
        :param t: torch tensor(N_batch, 1), times at which we sample
        :param x0: torch tensor (N_batch, self.dimension), starting values of the bridge.
        :param xT: torch tensor (N_batch, self.dimension), ending values of the bridge.
        :param t0: torch tensor(N_batch, 1), times of the starting value
        :param T: torch tensor(N_batch, 1), times of the ending values
        :return: torch tensor (N_batch, self.dimension), conditional means at tme t
        """
        return (self.int_sigma_sq_t(t0, t)/self.int_sigma_sq_t(t0, T))*(xT - x0) + x0

    def get_variances(self, t, t0=0, T=1):
        """
        get the conditionnal variance of the Brownian bridge, where the variance sigma_t is supposed to be proportional
        to the identity matric
        :param t: float, time at which we sample
        :param t0: float, time of the starting value
        :param T: float, time of the ending value
        :return: torch tensor (N_batch, 1), conditional variances at time t
        """
        ### Change from T to t in the first term of the product
        return self.int_sigma_sq_t(t0, t)*(1- self.int_sigma_sq_t(t0, t)/self.int_sigma_sq_t(t0, T))

    def sample(self, t, x0, xT,  t0=0, T=1):
        """
        get the conditionnal mean of the Brownian bridge
        :param t: torch tensor(N_batch, 1), times at which we sample
        :param x0: torch tensor (N_batch, self.dimension), starting values of the bridge.
        :param xT: torch tensor (N_batch, self.dimension), ending values of the bridge.
        :param t0: torch tensor(N_batch, 1), times of the starting value
        :param T: torch tensor(N_batch, 1), times of the ending values
        :return: torch tensor (N_batch, self.dimension), sampled conditional value marginal.
        """
        means = self.get_means(t, x0, xT, t0, T)
        variances = self.get_variances(t, t0, T)
        return torch.randn_like(means)*torch.sqrt(variances) + means


    def compute_drift_maruyama(self, x_t, t, tau, network, t0=0, Unet=False, observation=None):
        """
        Computing the drift part of the SDE
        :param x_t:
        :param time:
        :param network:
        :Unet: Boolean, if True, set the input in the Unet format.
        :param observation: if not None, means we are doing conditional inference
        :return:
        """
        #input = torch.concat([x_t, t], dim=-1)
        with torch.no_grad():
            if self.use_unet:
                input = torch.reshape(x_t, (x_t.shape[0], 28, 28))
                input = input.to(dtype=torch.float32)
                approximate_expectation = network.forward(input[:, None, :, :], t[:, 0])
            else:
                batch_size = x_t.shape[0]
                t = t.repeat(batch_size, 1)
                if observation is not None:
                    #input = torch.concat([observation + t, x_t], dim=-1)
                    time_embedd = self.pos_embbed(t)[:, 0, :]
                    approximate_expectation = network.forward(observation + time_embedd, x_t)
                else:
                    input = torch.concat([x_t, t], dim=-1)
                    approximate_expectation = network.forward(input)


        #plt.imshow(approximate_expectation[0, 0].detach().numpy(), cmap="gray")
        #plt.show()
        approximate_expectation = torch.reshape(approximate_expectation, (x_t.shape[0], self.dimension))

        drift = (approximate_expectation - x_t)/(self.int_sigma_sq_t(t0, tau) - self.int_sigma_sq_t(t0, t)) * self.sigma_t(t)**2
        return drift

    def euler_maruyama(self, x_0, times, tau, network, observation=None):
        """

        :param x_0: torch.tensor(1, dim_process), starting point of the Euler-Maruyama scheme
        :param observed_data: torch.tensor(1, dim_data), observed data which defines the posterior distribution
        :param times: torch.tensor(N_times, 1), time discretization of the Eular-Maruyama scheme.
        :param tau: float, time horizon
        :param network: torch network, approximating the expectation
        :return: torch.tensor(1, dim_process), point approximately simulated according to the posterior distribution.
        """
        x_t = x_0
        t = torch.zeros((1,1), dtype=torch.float32)
        trajectories = [0]
        batch_size = x_0.shape[0]
        for i, t_new in enumerate(times):
            if i%10 == 0:
                print(i)

            drift = self.compute_drift_maruyama(x_t=x_t, t=t, tau=tau, network=network, observation=observation)
            ##Check transposition here
            x_t_new = x_t + drift * (t_new - t) + np.sqrt((t_new - t)) * torch.randn((batch_size, self.dimension))*self.sigma_t(t)
            x_t = x_t_new
            t = t_new

        return np.array(trajectories), x_t