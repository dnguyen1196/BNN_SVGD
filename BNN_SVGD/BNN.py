import torch
from torch.autograd import grad, backward
from torch.nn import ModuleList
from .Net import BasicNet
from torch.autograd import Variable
import torch.nn.functional as F
import copy

class BNN_SVGD(torch.nn.Module):
    def __init__(self, x_dim, y_dim, num_networks=16, network_structure=[32]):
        super().__init__()
        self.num_nn = num_networks
        self.nn_arch = network_structure
        self.nns = ModuleList()

        # Initialize all the neural networks
        for _ in range(num_networks):
            zi = BasicNet(x_dim, y_dim, network_structure)
            for i, layer in enumerate(zi.nn_params):
                layer.weight.requires_grad_(True)
                layer.bias.requires_grad_(True)

            self.nns.append(zi)

    def optimize(self, X_train, y_train):
        """
        :param X_train:
        :param y_train:

        ---
        :return:

        Perform the following update
        z_i <- z_i + lr * phi(z_i)

        where phi(z_i) = 1/n sum_z( k(z_i, z) * d/dz_i log p(z_i|X, Y) + d/dzi k(z_i, z))

        -> Need the pair wise kernelized discrepancies
        -> Need the derivative of the kernelized discrepancies
        -> Need the derivative for each NN
        """
        # Compute all the phi(zi)
        max_iterations = 2000
        eps = 0.1
        for iteration in range(max_iterations):
            self.do_one_iteration(X_train, y_train, eps)

            if iteration in [0, 1, 10, 25, 50] or iteration % 100 == 0:
                mse = self.evaluate(X_train, y_train)
                print("iter: ", iteration, " - mse: ", mse)

    def do_one_iteration(self, X_train, y_train, eps):
        phi_zi_arr = list()
        # Compute all the phi_zi
        for i, zi in enumerate(self.nns):
            phi_zi = self.phi_zi_compute(zi, X_train, y_train)
            phi_zi_arr.append(phi_zi)

        # Update the zi
        for i, zi in enumerate(self.nns):
            update = self.nn_scale(phi_zi_arr[i], eps)
            self.update_zi(zi, update)

    def phi_zi_compute(self, zi, X, y):
        """
        :param zi:
        :return:
        """
        # Compute phi_zi
        #
        if self.num_nn == 1:
            # TODO: MAP estimate when num_nn = 1
            return
        else:
            phi_zi = None

            for i in range(1, len(self.nns)):
                z = self.nns[i]
                kd    = self.pair_wise_kernel_discrepancy_compute(zi, z)
                dlogp = self.derivative_log_posterior_compute(zi, X, y)
                dkd   = self.derivative_kernel_discrepancy_compute(zi, z)
                term1 = self.nn_scale(dlogp, kd)
                if phi_zi is None:
                    phi_zi = self.two_nns_add(term1, dkd)
                    phi_zi = self.nn_scale(phi_zi, 1./self.num_nn)
                else:
                    update = self.two_nns_add(term1, dkd)
                    update = self.nn_scale(update, 1./self.num_nn)
                    phi_zi = self.two_nns_add(phi_zi, update)

            return phi_zi


    def pair_wise_kernel_discrepancy_compute(self, z1, z2):
        """
        Compute the pair wise kernel discepancy between z1 and z2
        :param z1: a Basic Neural net
        :param z2: a Basic Neural net
        :return:

        The kernelized discrepancy between two neural networks
        z1 and z2 must have the same architecture
        """
        sigma  = 1. # parameter of RBF kernel
        norm_diff = 0.
        for i in range(len(z1.nn_params)):
            norm_diff += torch.sum(-(z1.nn_params[i].weight - z2.nn_params[i].weight)**2/(2*sigma))
            norm_diff += torch.sum(-(z1.nn_params[i].bias - z2.nn_params[i].bias)**2/(2*sigma))
        kd = torch.exp(norm_diff)
        return kd

    def derivative_kernel_discrepancy_compute(self, zi, z):
        """
        Compute the derivative pf the kernelized discrepancy with respect to neural
        network denoted as zi
        :param zi:
        :param z:
        :return:

        Calls function to compute pair wise kernel discrepancy
        Then use ptorch autograd to compute gradient wrt zi
        """
        zi.zero_grad()
        kd = self.pair_wise_kernel_discrepancy_compute(zi, z)
        derivative_kd = grad(kd, zi.parameters(), allow_unused=True) # Compute the gradient of k(zi, z) wrt zi
        return derivative_kd

    def derivative_log_posterior_compute(self, zi, X, y):
        """
        Compute the derivative of the posterior with respect to neural network zi
        :param zi:
        :param X
        :param y
        ----

        :return:

        p(z|X, y) ~ p(y|X, z)p(z)
        lg p(z|X,y) ~ lg p(y|X,z) + lg p(z)
        d/dz lg p(z|X,y) = d/dz lg p(y|X,z) + d/dz lg p(z)
        """
        zi.zero_grad()
        # Compute the log likelihood and the
        log_likelihood = self.log_likelihood_compute(zi, X, y)
        derivative_lg_likelihood = grad(log_likelihood, zi.parameters())

        # nn_params = list(zi.nn_params.parameters())
        # log_prior = self.log_prior_compute_params(nn_params)
        # derivative_lg_prior = grad(log_prior, nn_params)
        log_prior = self.log_prior_compute(zi)
        derivative_lg_prior = grad(log_prior, zi.parameters())

        derivative_lg_posterior = self.two_nns_add(derivative_lg_prior, derivative_lg_prior)
        return derivative_lg_posterior

    def two_nns_add(self, z1, z2):
        z = list()
        for i in range(len(z1)):
            res = z1[i] + z2[i]
            z.append(res)
        return z

    def nn_scale(self, z, m):
        for i in range(len(z)):
            z[i] *= m
        return z

    def update_zi(self, zi, update):
        for i in range(len(update)):
            update_layer = update[i]
            nn_layer = zi.nn_params[int(i/2)]
            if i % 2 == 0:
                nn_layer.weight.data.add_(update_layer)
            else:
                nn_layer.bias.data.add_(update_layer)

    def log_prior_compute_params(self, nn_params):
        """
        Compute derivative of prior with respect to neural network zi
        :param nn_params:
        :return:
        """
        lp = 0.
        sigma = 1
        for i in range(len(nn_params)):
            wi = Variable(zi.nn_params[i].weight.data)
            bi = Variable(zi.nn_params[i].bias.data)
            lp += -0.5 * torch.sum((nn_params[i].data**2))/sigma
            lp += -0.5 * torch.sum((nn_params[i].data**2))/sigma
        return lp

    def log_prior_compute(self, zi):
        """
        Compute derivative of prior with respect to neural network zi
        :param zi:
        :return:
        """
        lp = 0.
        sigma = 1
        for i in range(len(zi.nn_params)):
            # wi = zi.nn_params[i].weight
            # print(wi.requires_grad)
            # bi = zi.nn_params[i].bias
            lp += -0.5 * torch.sum((zi.nn_params[i].weight**2))/sigma
            lp += -0.5 * torch.sum((zi.nn_params[i].bias**2))/sigma
        return lp

    def log_likelihood_compute(self, zi, X, y):
        """
        Compute the derivative of the log likelihood with respect to neural network zi
        :param zi:
        :return:
        """
        sigma = 1.
        yhat = zi.forward(X)
        ll = -0.5 * torch.sum((y - yhat)**2)/sigma
        return  ll

    def evaluate(self, X_train, y_train):
        y_hat = torch.ones(y_train.size())
        for i, zi in enumerate(self.nns):
            y_hat += zi.forward(X_train) * 1/self.num_nn
        return F.mse_loss(y_hat, y_train)