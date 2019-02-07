import torch
import numpy as np
from torch.autograd import grad, backward
from torch.nn import ModuleList
from Net import *
from torch.autograd import Variable
import torch.nn.functional as F
import copy


"""
Proposal p(x)
target g(x)

1) thetap = theta + dtheta where dtheta ~ N(0, sigma)
2) compute rho = g(theta|X)/g(theta|X)
3) Do the decision 


https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm

"""

class BNN_MCMC(torch.nn.Module):
    def __init__(self, sigma=1):
        super().__init__()
        self.sigma = sigma

    def sample_n(self, n, X, y):
        self.n = n
        count = 0
        nn = SingleWeightNeuralNet(input_dim=1, output_dim=1, bias=False)
        samples = []

        while count < self.n:
            nn_prop = self.propose(nn)
            accept_prob = self.compute_accept_probability(nn, nn_prop, X, y)

            if accept_prob >= 1:
                samples.append(nn_prop)
                count += 1
                nn = nn_prop
            else:
                if np.random.random() < accept_prob:
                    samples.append(nn_prop)
                    count += 1
                    nn = nn_prop

        return samples

    def propose(self, nn):
        """

        :param nn:
        :return:

        theta_prop = theta + d_theta
        where
        d_theta ~ N(0, sigma)
        """
        proposed = SingleWeightNeuralNet(input_dim=1, output_dim=1, bias=False)
        for i, param in enumerate(proposed.nn_params):
            w = nn.nn_params[i].weight.data
            dw = torch.randn([1])
            param.weight.data = w + dw

        return proposed

    def compute_log_posterior(self, nn, X, y):
        yhat = nn.forward(X)
        log_ll = -0.5 * torch.sum((y - yhat)**2)/ self.sigma
        log_prior = 0.
        for i, param in enumerate(nn.nn_params):
            w = param.weight.data
            log_prior += -1/2 * torch.sum(w**2)/self.sigma
        return log_ll + log_prior

    def compute_accept_probability(self, nn, nn_prop, X, y):
        log_post_cur = self.compute_log_posterior(nn, X, y)
        log_post_prop = self.compute_log_posterior(nn_prop, X, y)
        A = torch.exp(log_post_prop - log_post_cur)
        return A



def do_experiment():
    N = 100
    eps = 1
    x_train_N = np.linspace(-3, 3, N)
    y_train_N = 1 * x_train_N + np.random.normal(0, 0.1, size=(N,))

    X = torch.FloatTensor(np.expand_dims(x_train_N, axis=1))
    y = torch.FloatTensor(np.expand_dims(y_train_N, axis=1))

    bnn_mcmc = BNN_MCMC(sigma=1)

    samples = bnn_mcmc.sample_n(1500, X, y)


    error = 0
    num = 0
    for i in range(int(len(samples)*14/15), len(samples)):
        sample = samples[i]

        yhat = sample.forward(X)
        error += torch.mean((yhat - y)**2)

        w1 = sample.nn_params[0].weight.data
        w2 = sample.nn_params[1].weight.data
        print(w1.numpy()[0], w2.numpy()[0])
        num += 1

    print("RMSE = ", torch.sqrt(error/num))


do_experiment()





