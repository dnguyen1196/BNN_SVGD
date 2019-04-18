import torch
from torch.autograd import grad, backward
from torch.nn import ModuleList
from .Net import *
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from BNN_SVGD.HMC_BNN import HMC_sampler, SG_HMC_BNN
import time
import copy
from BNN_SVGD.SVGD_BNN import *
import copy

"""
SVGD-Naive stochastic HMC hybrid
"""
class SVGD_naive_SHMC_hybrid(BNN_SVGD):
    def __init__(self, x_dim, y_dim, num_networks=16, network_structure=[32], ll_sigma=1, p_sigma=1, rbf_sigma=1,\
                 svgd_step_size=0.01, hmc_step_size=0.01, hmc_n_leapfrog_steps=10):
        """

        :param x_dim:
        :param y_dim:
        :param num_networks:
        :param network_structure:
        :param ll_sigma:
        :param p_sigma:
        :param rbf_sigma:
        :param svgd_step_size:
        :param hmc_step_size:
        :param hmc_n_leapfrog_steps:
        """
        super(SVGD_naive_SHMC_hybrid, self).__init__(ll_sigma, p_sigma, rbf_sigma)

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

        self.svgd_step_size = svgd_step_size
        self.hmc_step_size  = hmc_step_size
        self.hmc_n_leapfrog_steps = hmc_n_leapfrog_steps
        self.svgd_optimizer = optim.SGD(self.parameters(), lr=self.svgd_step_size)

    def sample_stochastic_hmc(self, zi, X, y):
        """

        :param zi:
        :param X:
        :param y:
        :return:
        """
        def calc_likelihood(bnn, X, y):
            yhat = bnn.forward(X)
            ll = -0.5 * torch.sum((torch.squeeze(y) - torch.squeeze(yhat)) ** 2) / self.ll_sigma**2
            return ll

        def calc_potential_energy(bnn, X, y):
            log_prior = 0.
            for i, layer in enumerate(bnn.nn_params):
                log_prior += -0.5 * torch.sum(layer.weight**2 / self.p_sigma**2)
                if bnn.bias:
                    log_prior += -0.5* torch.sum(layer.bias**2 / self.p_sigma**2)

            log_ll = calc_likelihood(bnn, X, y)
            potential = -log_prior - log_ll
            return potential

        def calc_kinetic_energy(momentum):
            kinetic = torch.sum(momentum**2)
            return kinetic

        def calc_grad_potential_energy(bnn, X, y):
            potential = calc_potential_energy(bnn, X, y)
            # Compute the gradient with respect to the parameters
            grad = []
            grad_nn = torch.autograd.grad(outputs=potential, inputs=bnn.parameters(), retain_graph=True)
            for i, layer in enumerate(grad_nn):
                grad.append(layer)
            return grad

        def generate_rand_momentum(bnn):
            momentum = []
            for i, layer in enumerate(bnn.nn_params):
                shape = layer.weight.shape
                momentum.append(np.random.normal(0, 1, size=shape))
                if layer.bias:
                    shape = layer.bias.shape
                    momentum.append(np.random.normal(0, 1, size=shape))

            return torch.FloatTensor(momentum)

        def update_momentum(momentum, dm, stepsize):
            for i, m in enumerate(dm):
                momentum[i] += m * stepsize
            return momentum

        def update_params(bnn, d_bnn, stepsize):
            for i, layer in enumerate(bnn.nn_params):
                if not bnn.bias:
                    layer.weight.data = layer.weight.data + d_bnn[i] * stepsize
                else:
                    layer.weight.data = layer.weight.data + d_bnn[i*2] * stepsize
                    layer.weight.data = layer.weight.data + d_bnn[i * 2 + 1] * stepsize
            return bnn

        # Run 1 iteration of hmc on each chain
        prop_bnn, accepted = hmc_sampler.sample_hmc(init_bnn=zi, n_leapfrog_steps=self.hmc_n_leapfrog_steps,
                                                    step_size=self.hmc_step_size,
                                                    calc_potential_energy=calc_potential_energy,
                                                    calc_kinetic_energy=calc_kinetic_energy,
                                                    calc_grad_potential_energy=calc_grad_potential_energy,
                                                    generate_rand_momentum=generate_rand_momentum,
                                                    update_params=update_params,
                                                    update_momentum=update_momentum,
                                                    X_batch=X, y_batch=y)
        return prop_bnn, accepted

    def fit(self, train_loader, num_iterations=1000, svgd_iteration=20, hmc_iteration=20):
        """
        :param train_loader: an iterable, streaming data in batches
        :param num_iterations: number of total iterations
        :param svgd_iteration: number of svgd iterations per svgd turn
        :param hmc_iteration:  number of hmc iterations per hmc turn
        :return:
        """
        # Start with adagrad
        self.hmc_optimizer = optim.SGD(self.parameters(), lr=1)

        svgd = True # start with svgd
        count = 1
        iteration = 1
        start = time.time()
        self.positions_over_time = []
        self.hmc_sampled_bnn     = []

        # Parameters for HMC
        n_leapfrog_steps = 10
        step_size = 0.001

        while iteration < num_iterations + 1:
            # Get the next batch
            X_batch, y_batch = next(train_loader)
            X = torch.FloatTensor(X_batch)
            y = torch.FloatTensor(y_batch)

            if svgd:
                # Do 1 step of svgd
                self.step_svgd(X, y, step_size=self.svgd_step_size)
            else:
                # Run HMC sampler
                self.hmc_optimizer.zero_grad()
                for i, zi in enumerate(self.nns):
                    prop_bnn, accepted = self.sample_stochastic_hmc(zi, X, y)
                    if accepted:
                        self.nns[i] = prop_bnn
                        self.hmc_sampled_bnn.append(prop_bnn)

            # Occasionally report on svgd-loss and mean squared error
            if iteration % 100 == 0 or iteration in [1]:
                preds = self.predict_average(X)
                error = torch.mean((torch.squeeze(preds) - torch.squeeze(y)) ** 2)
                print("iteration: ", iteration, " time: ",time.time() - start ,
                      " batch MSE: ", error.detach().numpy())

            # Keeping track of the positions over time, also specify if the position is during SVGD or HMC iterations
            curr_position = []
            for nnid in range(len(self.nns)):
                weight1 = self.nns[nnid].nn_params[0].weight.detach().numpy()[0]
                weight2 = self.nns[nnid].nn_params[1].weight.detach().numpy()[0]
                curr_position.append([weight1[0], weight2[0]])
            self.positions_over_time.append((curr_position, svgd))

            # Switch between svgd update and hmc update
            if (count % svgd_iteration == 0 and svgd) or (count % hmc_iteration == 0 and not svgd):
                svgd = not svgd
                count = 1

            count += 1
            iteration += 1

"""
SVGD-SG_HMC hybrid
SVGD - stochastic gradient HMC hybrid, theoretically improved over the naive stochastic HMC
"""
class SVGD_SGHMC_hybrid(SVGD_naive_SHMC_hybrid):
    def __init__(self, x_dim, y_dim, num_networks=16, network_structure=[32], ll_sigma=1, p_sigma=1, rbf_sigma=1, \
                 svgd_step_size=0.01, hmc_step_size=0.01, hmc_n_leapfrog_steps=10, momentum=0.999):
        """

        :param x_dim:
        :param y_dim:
        :param num_networks:
        :param network_structure:
        :param ll_sigma:
        :param p_sigma:
        :param rbf_sigma:
        :param svgd_step_size:
        :param hmc_step_size:
        :param hmc_n_leapfrog_steps:
        :param momentum:
        """
        super(SVGD_SGHMC_hybrid, self).__init__(x_dim, y_dim, num_networks, \
                                                network_structure, ll_sigma, p_sigma, rbf_sigma, svgd_step_size, \
                                                hmc_step_size, hmc_n_leapfrog_steps)
        self.momentum = momentum

    def sample_stochastic_hmc(self, zi, X, y):
        """
        NOTE: this overrides sample_stochastic_hmc function in SVGD_naiveHMC_hybrid

        :param init_bnn:
        :param n_leapfrog_steps:
        :param step_size:
        :param X:
        :param y:
        :return:
        """
        def calc_likelihood(bnn, X, y):
            yhat = bnn.forward(X)
            ll = -0.5 * torch.sum((torch.squeeze(y) - torch.squeeze(yhat)) ** 2) / self.ll_sigma**2
            return ll

        def calc_potential_energy(bnn, X, y):
            log_prior = 0.
            for i, layer in enumerate(bnn.nn_params):
                log_prior += -0.5 * torch.sum(layer.weight**2 / self.p_sigma**2)
                if bnn.bias:
                    log_prior += -0.5* torch.sum(layer.bias**2 / self.p_sigma**2)

            log_ll = calc_likelihood(bnn, X, y)
            potential = -log_prior - log_ll
            return potential

        # Use SGD with momentum
        optimizer = optim.SGD(zi.parameters(), lr=self.hmc_step_size, momentum=self.momentum)

        for i in range(self.hmc_n_leapfrog_steps):
            optimizer.zero_grad()
            energy = calc_potential_energy(zi, X, y)
            energy.backward()
            optimizer.step()

        return copy.deepcopy(zi), True