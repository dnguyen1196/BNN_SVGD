import torch
from torch.autograd import grad, backward
from torch.nn import ModuleList
from .Net import *
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import copy

"""
Base SVGD class
FC_SVGD and CovNet_SVGD (below) extend on this base SVGD class
"""
class BNN_SVGD(torch.nn.Module):
    def __init__(self, ll_sigma=1, p_sigma=1, rbf_sigma=1):
        """

        :param ll_sigma:
        :param p_sigma:
        :param rbf_sigma:
        """
        super().__init__()
        self.ll_sigma = ll_sigma
        self.p_sigma = p_sigma
        self.rbf_sigma = rbf_sigma

    def loss(self, X, y):
        """
        :param X: training data
        :param y: training labels
        -----
        :return: SVGD batch loss

        Use trick to implicitly let pytorch do the heavy work of gradient computatin and update

        phi(zi) = 1/n  sum [ k(zi, zj) d/dzi log p(zi|X, y) + d/dzi k(zi, zj) ]

        So we create a loss function that is
        L(zi) = 1/n  sum [ k(zi, zj) log p(zi|X, y) + k(zi, zj) ]
        And detach the gradient from the first k(zi, zj) term
        """
        loss = 0.
        for i, zi in enumerate(self.nns):
            # for zj in self.nns:
            for j in range(0, len(self.nns)):
                zj = self.nns[j]

                log_ll = self.log_likelihood_compute(zj, X, y)
                log_prior = self.log_prior_compute(zj)
                kernel_term1 = self.pair_wise_kernel_discrepancy_compute(zj, zi)
                kernel_term1.detach()  # Detach from gradient graph

                kernel_term2 = self.pair_wise_kernel_discrepancy_compute(zj, zi)

                loss += 1/(2 * self.num_nn) * kernel_term2
                loss += 1 / self.num_nn * (kernel_term1 * (log_prior + log_ll))

        return -loss

    def step(self, X, y, step_size=0.1):
        """

        :param X:
        :param y:
        :param step_size:
        :return:
        """
        self.svgd_optimizer.zero_grad()
        for i, zi in enumerate(self.nns):
            log_ll = -self.log_likelihood_compute(zi, X, y)
            log_prior = -self.log_prior_compute(zi)
            log_post  = log_ll + log_prior
            log_post.backward()
            # log_ll.backward()
            # y_pred = torch.squeeze(zi.forward(X))
            # mse = torch.sum((y_pred - torch.squeeze(y))**2)
            # mse.backward()

        self.svgd_optimizer.step()

    def step_svgd(self, X, y, step_size=0.1):
        """

        :param X:
        :param y:
        :param step_size:

        Perform one step of svgd update
        :return:
        """
        N = len(self.nns) # Number of particles
        # Compute all the discrepancies and the derivative of the discrepancy with respect
        # to the particles
        discrepancy_matrix = torch.zeros(N,N)
        gradient_discrepancy_matrix = [[None for _ in range(N)] for _ in range(N)]

        for i in range(N):
            for j in range(i, N):
                self.svgd_optimizer.zero_grad()

                zi = self.nns[i]
                zj = self.nns[j]
                kd = self.pair_wise_kernel_discrepancy_compute(zi, zj)

                kd.backward()

                discrepancy_matrix[i, j] = kd.detach()
                discrepancy_matrix[j, i] = kd.detach()

                gradient_discrepancy_matrix[i][j] = self.copy_gradient(zj)
                gradient_discrepancy_matrix[j][i] = self.copy_gradient(zi)

        # Compute the derivative of log posterior of all the neural networks
        derivative_log_posterior = [None for _ in range(N)]
        self.svgd_optimizer.zero_grad()

        for i in range(N):
            z = self.nns[i]
            log_lik       = self.log_likelihood_compute(z, X, y)
            log_prior     = self.log_prior_compute(z)
            log_posterior = log_lik + log_prior
            log_posterior.backward()
            grad_z        = self.copy_gradient(z)
            derivative_log_posterior[i] = grad_z

        self.svgd_optimizer.zero_grad()
        # Apply svgd update
        for i in range(N):
            self.nns[i] = self.apply_svgd_update(i, discrepancy_matrix, gradient_discrepancy_matrix, \
                                                 derivative_log_posterior, step_size)

    def apply_svgd_update(self, i, discrepancy_matrix, gradient_discrepancy_matrix, derivative_log_posterior, step_size):
        """

        :param i:
        :param discrepancy_matrix:
        :param gradient_discrepancy_matrix:
        :param derivative_log_posterior:

        :return:
        """
        zi = self.nns[i] # Updating zi
        N = len(self.nns)

        for j in range(N): # Loop through all particles O(N^2)

            kd = discrepancy_matrix[i, j]                     # kernel discrepancy k(zi, zj)
            dlog_post = derivative_log_posterior[j]           # Derivative of log posterior with respect to zj
            derivative_kd = gradient_discrepancy_matrix[i][j] # Derivative of kd(zi, zj) with respect to zj

            # Apply update: zi = zi + eps * phi(zi)
            self.apply_phi(zi, kd, dlog_post, derivative_kd, step_size)

        return zi

    def apply_phi(self, zi, kd, dlog_post, derivative_kd, step_size):
        """

        :param zi:
        :param kd:
        :param dlog_post:
        :param derivative_kd:
        :param step_size:

        phi(zi) = 1/n sum_zj [k(zi, zj) d_zj log p(zj) + d_zj k(zi, zj)]

        :return:
        """
        # Apply update to z
        N = len(self.nns)

        for k, param in enumerate(zi.nn_params):
            if self.bias:  # If there is a bias term
                param.weight.data += 1 / N * step_size * (kd * dlog_post[k * 2] + derivative_kd[k * 2])
                param.bias.data += 1 / N * step_size * (kd * dlog_post[k * 2 + 1] + derivative_kd[k * 2 + 1])
            else:  # If there is no bias term
                param.weight.data += 1 / N * step_size * (kd * dlog_post[k] + derivative_kd[k])

    def copy_gradient(self, z):
        """
        :param z:

        Copy the gradient from particle z
        :return:
        """
        gradient = []

        for i, param in enumerate(z.nn_params):
            gradient.append(copy.deepcopy(param.weight.grad))
            if self.bias:
                gradient.append(copy.deepcopy(param.bias.grad))

        return gradient

    def pair_wise_kernel_discrepancy_compute(self, z1, z2):
        """
        :param z1: a Neural net
        :param z2: a Neural net
        :return:

        Compute the pair wise kernel discepancy between z1 and z2

        The kernelized discrepancy between two neural networks
        z1 and z2 must have the same architecture
        """
        norm_diff = 0.
        for i in range(len(z1.nn_params)):
            norm_diff += torch.sum(-(z1.nn_params[i].weight - z2.nn_params[i].weight) ** 2 / (2 * self.rbf_sigma**2))

            if self.bias:
                norm_diff += torch.sum(-(z1.nn_params[i].bias - z2.nn_params[i].bias) ** 2 / (2 * self.rbf_sigma**2))

        kd = torch.exp(norm_diff)
        return kd

    def energies_compute(self, X_train, y_train):
        """
        :param X_train:
        :param y_train:
        :return:

        Compute the energies of the BNNs, which is essentially negative log posterior
        """
        energies = np.zeros((self.num_nn,))
        for i, zi in enumerate(self.nns):
            energies[i] = self.calc_potential_energy(zi, X_train, y_train)
        return energies

    def calc_potential_energy(self, bnn, X_train, y_train):
        """

        :param bnn:
        :param X_train:
        :param y_train:
        :return:

        The potential energy of the system

        Potential energy = -log(posterior) ~~ C * ( log p(y|z) + log p(z) )
        """
        log_prior = self.log_prior_compute(bnn)
        log_likelihood = self.log_likelihood_compute(bnn, X_train, y_train)
        return -log_prior - log_likelihood

    def log_prior_compute(self, zi):
        """
        Compute derivative of prior with respect to neural network zi
        :param zi: a neural network
        :return:
        """
        lp = 0.
        sigma = 1
        for i in range(len(zi.nn_params)):
            lp += -0.5 * torch.sum((zi.nn_params[i].weight ** 2)) / self.p_sigma**2
            if self.bias:
                lp += -0.5 * torch.sum((zi.nn_params[i].bias ** 2)) / self.p_sigma**2
        return lp

    def log_likelihood_compute(self, zi, X, y):
        """
        Compute the derivative of the log likelihood with respect to neural network zi
        :param zi:
        :return:
        """
        sigma = 1.
        yhat = zi.forward(X)
        ll = -0.5 * torch.sum((torch.squeeze(y) - torch.squeeze(yhat)) ** 2) / self.ll_sigma**2
        return ll

    def evaluate(self, X_train, y_train):
        """
        :param X_train:
        :param y_train:
        :return:
        Return the current MSE given training data and labels
        """
        y_hat = torch.zeros(y_train.size())
        for i, zi in enumerate(self.nns):
            y_hat += zi.forward(X_train) / self.num_nn
        return torch.mean((y_hat - y_train) ** 2)

    def predict(self, X_test):
        """
        :param X_test:
        :return: prediction array, float array of size
        (num_neural_networks,)
        """
        ys_prediction = list()

        for i in range(len(self.nns)):
            zi = self.nns[i]
            ys_prediction.append(torch.squeeze(zi.forward(X_test)))

        return ys_prediction

    def predict_average(self, X_test):
        """
        :param X_test:
        :return: Similar to predict but returns the
        average of predictions from all neural networks
        """
        ys_prediction = self.predict(X_test)
        y_pred = torch.zeros(ys_prediction[0].shape)
        for i, v in enumerate(ys_prediction):
            y_pred += v
        y_pred = y_pred / len(ys_prediction)
        return y_pred


"""
SVGD class for fully connected neural networks
"""
class FC_SVGD(BNN_SVGD):
    def __init__(self, x_dim, y_dim, num_networks=16, network_structure=[32], ll_sigma=1, p_sigma=1, rbf_sigma=1, step_size=0.01):
        """

        :param x_dim: dimension of input
        :param y_dim: dimension of output
        :param num_networks: number of neural networks
        :param network_structure: hidden layer structure
        :param ll_sigma:
        :param p_sigma:
        :param rbf_sigma:
        """
        super(FC_SVGD, self).__init__(ll_sigma, p_sigma, rbf_sigma)

        self.num_nn = num_networks
        self.nn_arch = network_structure
        self.nns = ModuleList()
        self.x_dim = x_dim
        self.y_dim = y_dim

        # Initialize all the neural networks
        for _ in range(num_networks):
            zi = FullyConnectedNet(x_dim, y_dim, network_structure)
            self.nns.append(zi)

        self.step_size = step_size
        self.svgd_optimizer = optim.SGD(self.parameters(), lr=self.step_size)


"""
SVGD class for convolution neural network
Used for testing on CIFAR-10 and MNIST
"""
class CovNet_SVGD(BNN_SVGD):
    def __init__(self, image_set, num_networks=10, step_size=0.01):
        """

        :param image_set:
        :param num_networks:
        """
        super(CovNet_SVGD, self).__init__()
        self.num_nn = num_networks
        self.nns = ModuleList()
        self.image_set = image_set
        self.bias = True

        # Initialize all the neural networks
        for _ in range(num_networks):
            if image_set == "MNIST":
                zi = MnistCovNet()
            elif image_set == "CIFAR-10":
                zi = Cifar10LeNet()
            self.nns.append(zi)

        self.ll_sigma  = 1
        self.p_sigma   = 1
        self.rbf_sigma = 1
        self.step_size = step_size
        self.svgd_optimizer = optim.SGD(self.parameters(), lr=self.step_size)

    def pair_wise_kernel_discrepancy_compute(self, z1, z2):
        """
        Compute the pair wise kernel discepancy between z1 and z2
        :param z1: a Basic Neural net
        :param z2: a Basic Neural net
        :return:

        The kernelized discrepancy between two neural networks
        z1 and z2 must have the same architecture
        
        MNIST net
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)
        """
        norm_diff = 0.

        if self.image_set == "MNIST":
            norm_diff += torch.sum(-(z1.fc1.weight - z2.fc1.weight) ** 2 / (2 * self.rbf_sigma**2))
            norm_diff += torch.sum(-(z1.fc2.weight - z2.fc2.weight) ** 2 / (2 * self.rbf_sigma**2))
            norm_diff += torch.sum(-(z1.fc1.bias   - z2.fc1.bias) ** 2 / (2 * self.rbf_sigma**2))
            norm_diff += torch.sum(-(z1.fc2.bias   - z2.fc2.bias) ** 2 / (2 * self.rbf_sigma**2))
        elif self.image_set == "CIFAR-10":
            norm_diff += torch.sum(-(z1.fc1.weight - z2.fc1.weight) ** 2 / (2 * self.rbf_sigma**2))
            norm_diff += torch.sum(-(z1.fc1.bias   - z2.fc1.bias) ** 2 / (2 * self.rbf_sigma**2))
            norm_diff += torch.sum(-(z1.fc2.weight - z2.fc2.weight) ** 2 / (2 * self.rbf_sigma**2))
            norm_diff += torch.sum(-(z1.fc2.bias   - z2.fc2.bias) ** 2 / (2 * self.rbf_sigma**2))
            norm_diff += torch.sum(-(z1.fc3.weight - z2.fc3.weight) ** 2 / (2 * self.rbf_sigma**2))
            norm_diff += torch.sum(-(z1.fc3.bias   - z2.fc3.bias) ** 2 / (2 * self.rbf_sigma**2))

        kd = torch.exp(norm_diff)
        return kd


    def log_prior_compute(self, bnn):
        """
        :param bnn:
        :return:
        
        MNIST net
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)
        """
        lp = 0.
        if self.image_set == "MNIST":
            for layer in [bnn.fc1, bnn.fc2]:
                lp += -0.5 * torch.sum((layer.weight**2))/self.p_sigma**2
                lp += -0.5 * torch.sum((layer.bias**2))/self.p_sigma**2
        elif self.image_set == "CIFAR-10":
            for layer in [bnn.fc1, bnn.fc2, bnn.fc3]:
                lp += -0.5 * torch.sum((layer.weight**2))/self.p_sigma**2
                lp += -0.5 * torch.sum((layer.bias**2))/self.p_sigma**2

        return lp


"""
Simple SVGD class for experimentation
"""
class SVGD_simple(BNN_SVGD):
    def __init__(self, x_dim, y_dim, num_networks=16, network_structure=[32], ll_sigma=1, p_sigma=1, rbf_sigma=1, step_size=0.01):
        """

        :param x_dim: dimension of input
        :param y_dim: dimension of output
        :param num_networks: number of neural networks (or particles)
        :param network_structure: hidden layer structure
        :param ll_sigma: standard deviation of likelihood term
        :param p_sigma:  standard deviation of prior term
        :param rbf_sigma: rbf length scale
        """
        super(SVGD_simple, self).__init__(ll_sigma, p_sigma, rbf_sigma)

        self.num_nn = num_networks
        self.nn_arch = network_structure
        self.nns = ModuleList()
        self.x_dim = x_dim
        self.y_dim = y_dim

        # Initialize all the neural networks, note that we use SingleWeightNeuralNet for experimentation
        # purpose only
        self.bias = False
        for _ in range(num_networks):
            zi = SingleWeightNeuralNet(x_dim, y_dim)
            self.nns.append(zi)

        self.step_size = step_size
        self.svgd_optimizer = optim.SGD(self.parameters(), lr=self.step_size)
