import torch
from torch.autograd import grad, backward
from torch.nn import ModuleList
from .Net import *
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from BNN_SVGD.HMC_BNN import HMC_sampler
import time
import copy
from BNN_SVGD.SVGD_BNN import *
import copy


class SVGD_HMC_hybrid(BNN_SVGD):
    def __init__(self, x_dim, y_dim, num_networks=16, network_structure=[32], ll_sigma=1, p_sigma=1, rbf_sigma=1):
        super(SVGD_HMC_hybrid, self).__init__(ll_sigma, p_sigma, rbf_sigma)

        self.num_nn = num_networks
        self.nn_arch = network_structure
        self.nns = ModuleList()
        self.x_dim = x_dim
        self.y_dim = y_dim

        # Initialize all the neural networks
        self.bias = False
        for _ in range(num_networks):
            zi = SingleWeightNeuralNet(x_dim, y_dim)
            self.nns.append(zi)

    def fit(self, train_loader, num_iterations=1000, svgd_iteration=20, hmc_iteration=20):
        # svgd_optimizer = optim.Adagrad(self.parameters(), lr=1)
        svgd_optimizer = optim.SGD(self.parameters(), lr=0.01)

        hmc_optimizer = optim.SGD(self.parameters(), lr=1)

        svgd = True
        count = 1
        iteration = 1
        hmc_sampler = HMC_sampler()
        start = time.time()

        positions_over_time = []

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

        n_leapfrog_steps = 25
        step_size = 0.001

        while iteration < num_iterations + 1:

            X_batch, y_batch = next(train_loader)
            X = torch.FloatTensor(X_batch)
            y = torch.FloatTensor(y_batch)

            if svgd:
                # run svgd
                svgd_optimizer.zero_grad()
                svgd_loss_batch = self.loss(X, y)
                svgd_loss_batch.backward()
                svgd_optimizer.step()

            else:
                # Run HMC sampler
                hmc_optimizer.zero_grad()

                for i, zi in enumerate(self.nns):
                    prop_bnn, accepted = hmc_sampler.sample_hmc(init_bnn=zi, n_leapfrog_steps=n_leapfrog_steps,
                                                                step_size=step_size,
                                                                calc_potential_energy=calc_potential_energy,
                                                                calc_kinetic_energy=calc_kinetic_energy,
                                                                calc_grad_potential_energy=calc_grad_potential_energy,
                                                                generate_rand_momentum=generate_rand_momentum,
                                                                update_params=update_params,
                                                                update_momentum=update_momentum,
                                                                X_batch=X, y_batch=y)

                    # TODO: should we always update to the new proposed bnn?
                    if accepted:
                        self.nns[i] = prop_bnn

                hmc_optimizer.zero_grad()

            # Keeping track of the positions over time, also make sure to be clear
            # if the current position is during svgd or hmc iteration
            curr_position = []
            for nnid in range(len(self.nns)):
                weight1 = self.nns[nnid].nn_params[0].weight.detach().numpy()[0]
                weight2 = self.nns[nnid].nn_params[1].weight.detach().numpy()[0]
                curr_position.append([weight1[0], weight2[0]])
            positions_over_time.append((curr_position, svgd))

            # Occasionally report on svgd-loss and mean squared error
            if iteration % 50 == 0 or iteration in [1]:
                preds = self.predict_average(X)
                error = torch.mean((torch.squeeze(preds) - torch.squeeze(y)) ** 2)
                svgd_loss_batch = self.loss(X, y)
                print("iteration: ", iteration, " time: ",time.time() - start ,
                      " MSE: ", error.detach().numpy(), " svgd-loss: ", svgd_loss_batch.detach().numpy())

            # Switch between svgd update and hmc update
            if (count % svgd_iteration == 0 and svgd) or (count % hmc_iteration == 0 and not svgd):
                svgd = not svgd
                # # NOTE: is the reason why during hybrid optimization, svgd updates make little progress? Yes, seems like
                # # it because if I leave this line (meaning it resets every time i switch back to svgd, then
                # # svgd updates moves now
                # if svgd:
                #     optimizer = optim.Adagrad(self.parameters(), lr=1)

                count = 1

            count += 1
            iteration += 1

        return positions_over_time


