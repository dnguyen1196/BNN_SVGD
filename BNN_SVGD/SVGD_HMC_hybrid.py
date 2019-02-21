import torch
from torch.autograd import grad, backward
from torch.nn import ModuleList
from .Net import *
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import time
import copy
from BNN_SVGD.SVGD_BNN import *
import copy


class HMC_sampler(nn.Module):
    def __init__(self):
        super(HMC_sampler, self).__init__()

    def make_proposal_via_leapfrog_steps(self, cur_bnn, cur_momentum, n_leapfrog_steps=25, step_size=0.001,
                                         calc_grad_potential_energy=None, update_params=None, update_momentum=None):
        """

        :param cur_bnn:
        :param cur_momentum:
        :param n_leapfrog_steps:
        :param step_size:
        :param calc_grad_potential_energy:
        :param update_params:
        :param update_momentum:
        :return:
        """
        assert (calc_grad_potential_energy)
        assert (update_params)
        assert (update_momentum)

        prop_bnn = copy.deepcopy(cur_bnn)
        prop_momentum = cur_momentum

        # prop_momentum_vec -= step_size * calc_grad_potential_energy(prop_bnn_params)/2.0
        grad_potential = calc_grad_potential_energy(prop_bnn)
        prop_momentum = update_momentum(prop_momentum, grad_potential, -step_size / 2.0)

        # This will use the grad of potential energy (use provided function)
        for step_id in range(n_leapfrog_steps):
            # This will use the grad of kinetic energy (has simple closed form)

            if step_id < (n_leapfrog_steps - 1):
                prop_bnn = update_params(prop_bnn, prop_momentum, -step_size)
                prop_momentum = update_momentum(prop_momentum, calc_grad_potential_energy(prop_bnn), -step_size)
            else:
                prop_bnn = update_params(prop_bnn, prop_momentum, step_size)
                prop_momentum = update_momentum(prop_momentum, calc_grad_potential_energy(prop_bnn), -step_size / 2.0)

        prop_momentum = update_momentum(prop_momentum, prop_momentum, -2.0) # Equivalent to flipping sign
        return prop_bnn, prop_momentum

    def sample_hmc(self, n_leapfrog_steps=15, step_size=0.0001, init_bnn=None, calc_potential_energy=None,
                   calc_kinetic_energy=None,calc_grad_potential_energy=None, generate_rand_momentum=None, update_params=None, update_momentum=None):
        """

        :param n_leapfrog_steps:
        :param step_size:
        :param init_bnn:
        :param calc_potential_energy:
        :param calc_kinetic_energy:
        :param calc_grad_potential_energy:
        :param generate_rand_momentum:
        :param update_params:
        :param update_momentum:

        :return:
        """

        # Make sure that these functions are not None
        assert (calc_potential_energy)
        assert (calc_kinetic_energy)
        assert (calc_grad_potential_energy)
        assert (init_bnn)
        assert (generate_rand_momentum)
        assert (update_params)
        assert (update_momentum)

        # Start by generating a random momentum
        cur_momentum = generate_rand_momentum(init_bnn)
        cur_potential = calc_potential_energy(init_bnn)
        cur_kinetic = calc_kinetic_energy(cur_momentum)

        # Create PROPOSED configuration
        proposed_bnn, proposed_momentum = self.make_proposal_via_leapfrog_steps(
            init_bnn, cur_momentum, n_leapfrog_steps=n_leapfrog_steps,step_size=step_size,
            calc_grad_potential_energy=calc_grad_potential_energy, update_params=update_params, update_momentum=update_momentum)

        proposed_potential = calc_potential_energy(proposed_bnn)
        proposed_kinetic = calc_kinetic_energy(proposed_momentum)

        accept_proba = np.minimum(1, np.exp( -proposed_potential.detach().numpy() - proposed_kinetic.detach().numpy()
                                             + cur_kinetic.detach().numpy() + cur_potential.detach().numpy()))

        # Draw random value from (0,1) to determine if we accept or not
        if np.random.rand() < accept_proba:
            # If here, we accepted the proposal
            return  proposed_bnn

        return init_bnn



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

    def fit(self, train_loader, num_iterations=1000, svgd_iteration=10, hmc_iteration=20):
        optimizer = optim.Adagrad(self.parameters(), lr=1)
        svgd = True
        count = 0
        iteration = 0
        hmc_sampler = HMC_sampler()
        start = time.time()

        positions_over_time = []


        def calc_prior(bnn):
            log_prior = 0.
            for i, layer in enumerate(bnn.nn_params):
                log_prior += -0.5 * layer.weight**2 / self.p_sigma
                if bnn.bias:
                    log_prior += -0.5 * layer.bias**2 / self.p_sigma
            return log_prior

        def calc_likelihood(bnn):
            yhat = bnn.forward(X)
            ll = -0.5 * torch.sum((y - torch.squeeze(yhat)) ** 2) / self.ll_sigma
            return ll

        def calc_potential_energy(bnn):
            log_prior = calc_prior(bnn)
            log_likelihood = calc_likelihood(bnn)
            return log_prior + log_likelihood

        def calc_kinetic_energy(momentum):
            kinetic = torch.sum(momentum**2)
            return kinetic

        def calc_grad_potential_energy(bnn):
            optimizer.zero_grad()
            potential = calc_potential_energy(bnn)
            potential.backward()
            grad = []
            for i, layer in enumerate(bnn.nn_params):
                grad.append(layer.weight.grad)
                if bnn.bias:
                    grad.append(layer.bias.grad)
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

        while iteration + 1 < num_iterations:
            optimizer.zero_grad()

            X_batch, y_batch = next(train_loader)
            X = torch.FloatTensor(X_batch)
            y = torch.FloatTensor(y_batch)

            if svgd:
                # run svgd
                svgd_loss_batch = self.loss(X, y)
                svgd_loss_batch.backward()
                optimizer.step()

            else:
                # Run HMC sampler
                for i, zi in enumerate(self.nns):
                    self.nns[i] = hmc_sampler.sample_hmc(init_bnn=zi,
                                                         calc_potential_energy=calc_potential_energy,
                                                         calc_kinetic_energy=calc_kinetic_energy,
                                                         calc_grad_potential_energy=calc_grad_potential_energy,
                                                         generate_rand_momentum=generate_rand_momentum,
                                                         update_params=update_params, update_momentum=update_momentum)

            # Keeping track of the positions over time, also make sure to be clear
            # if the current position is during svgd or hmc iteration
            curr_position = []
            for nnid in range(len(self.nns)):
                weight1 = self.nns[nnid].nn_params[0].weight.detach().numpy()[0]
                weight2 = self.nns[nnid].nn_params[1].weight.detach().numpy()[0]
                curr_position.append([weight1[0], weight2[0]])
            positions_over_time.append((curr_position, svgd))

            if iteration % 50 == 0:
                preds = self.predict_average(X)
                error = torch.mean((torch.squeeze(preds) - torch.squeeze(y)) ** 2)
                svgd_loss_batch = self.loss(X, y)
                print("iteration: ", iteration, " time: ",time.time() - start ,
                      " MSE: ", error.detach().numpy(), " svgd-loss: ", svgd_loss_batch.detach().numpy())

            count += 1
            if (count % svgd_iteration == 0 and svgd) or (count % hmc_iteration == 0 and not svgd):
                svgd = not svgd
                count = 0

            iteration += 1

        return positions_over_time


