import torch.optim as optim
import time
from BNN_SVGD.SVGD_BNN import *
import copy
import math


class HMC_sampler(nn.Module):
    def __init__(self):
        return

    def make_proposal_via_leapfrog_steps(self, cur_bnn, cur_momentum, n_leapfrog_steps, step_size,
                                         calc_grad_potential_energy=None, update_params=None, update_momentum=None,
                                         X_batch=None, y_batch=None):
        """
        :param cur_bnn: current neural network
        :param cur_momentum: current momentum (same structure as the neural network)
        :param n_leapfrog_steps: number of leap frog steps
        :param step_size: step size
        :param calc_grad_potential_energy: function to compute the gradient of the potential energy
        :param update_params: function to do update params
        :param update_momentum: function to update momentum
        :return:
        """
        assert (calc_grad_potential_energy)
        assert (update_params)
        assert (update_momentum)
        assert (X_batch is not None)
        assert (y_batch is not None)

        prop_bnn = copy.deepcopy(cur_bnn)
        prop_momentum = copy.deepcopy(cur_momentum)

        # Compute the current potential energy
        grad_potential = calc_grad_potential_energy(prop_bnn, X_batch, y_batch)

        # Take a half step
        prop_momentum = update_momentum(prop_momentum, grad_potential, -step_size / 2.0)

        # This will use the grad of potential energy (use provided function)
        for step_id in range(n_leapfrog_steps):
            # This will use the grad of kinetic energy (has simple closed form)
            prop_bnn = update_params(prop_bnn, prop_momentum, step_size)
            if step_id < (n_leapfrog_steps - 1):
                prop_momentum = update_momentum(prop_momentum, calc_grad_potential_energy(prop_bnn, X_batch, y_batch), -step_size)
            else:
                # At the last step, take a half step
                prop_momentum = update_momentum(prop_momentum, calc_grad_potential_energy(prop_bnn, X_batch, y_batch), -step_size / 2.0)

        # Flip the sign of the momentum to make the proposal symmetric
        prop_momentum = update_momentum(prop_momentum, prop_momentum, -2.0) # flipping the sign of momentum
        return prop_bnn, prop_momentum

    def sample_hmc(self, n_leapfrog_steps=100, step_size=1e-3, init_bnn=None, calc_potential_energy=None,
                   calc_kinetic_energy=None,calc_grad_potential_energy=None, generate_rand_momentum=None,
                   update_params=None, update_momentum=None, X_batch=None, y_batch=None):
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
        assert (X_batch is not None)
        assert (y_batch is not None)

        # Start by generating a random momentum
        cur_momentum = generate_rand_momentum(init_bnn)
        cur_potential = calc_potential_energy(init_bnn, X_batch, y_batch)
        cur_kinetic = calc_kinetic_energy(cur_momentum)

        # Create proposed configuration
        proposed_bnn, proposed_momentum = self.make_proposal_via_leapfrog_steps(
            cur_bnn=init_bnn, cur_momentum=cur_momentum, n_leapfrog_steps=n_leapfrog_steps,step_size=step_size,
            calc_grad_potential_energy=calc_grad_potential_energy, update_params=update_params, update_momentum=update_momentum,
            X_batch=X_batch, y_batch=y_batch)

        proposed_potential = calc_potential_energy(proposed_bnn, X_batch, y_batch)
        proposed_kinetic = calc_kinetic_energy(proposed_momentum)

        accept_proba = np.minimum(1, np.exp( -proposed_potential.detach().numpy() - proposed_kinetic.detach().numpy()
                                             + cur_kinetic.detach().numpy() + cur_potential.detach().numpy()))

        # Draw random value from (0,1) to determine if we accept or not
        if np.random.rand() < accept_proba:
            return proposed_bnn, True

        return proposed_bnn, False


class HMC_BNN(torch.nn.Module):
    def __init__(self, x_dim, y_dim, num_networks=16, network_structure=[32], ll_sigma=1, p_sigma=1, rbf_sigma=1):
        """
        :param x_dim:
        :param y_dim:
        :param num_networks:
        :param network_structure:
        :param ll_sigma:
        :param p_sigma:
        :param rbf_sigma:
        """
        super(HMC_BNN, self).__init__()

        self.num_nn = num_networks
        self.nn_arch = network_structure
        self.nns = ModuleList()
        self.x_dim = x_dim
        self.y_dim = y_dim

        # Initialize all the neural networks
        for _ in range(num_networks):
            zi = SingleWeightNeuralNet(x_dim, y_dim)
            self.nns.append(zi)

    def fit(self, train_loader, num_iterations=1000):
        optimizer = optim.Adagrad(self.parameters(), lr=1)
        hmc_sampler = HMC_sampler()
        start = time.time()

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

        sampled_bnn = []
        energies_list = []
        acceptances = 0
        n_leapfrog_steps = 25
        step_size = 0.001

        for iteration in range(num_iterations+1):
            optimizer.zero_grad()

            X_batch, y_batch = next(train_loader)
            X = torch.FloatTensor(X_batch)
            y = torch.FloatTensor(y_batch)

            # Run HMC sampler, running multiple chains at the same time
            for i, zi in enumerate(self.nns):
                prop_bnn, accepted = hmc_sampler.sample_hmc(init_bnn=zi, n_leapfrog_steps=n_leapfrog_steps,
                                         step_size=step_size,
                                         calc_potential_energy=calc_potential_energy,
                                         calc_kinetic_energy=calc_kinetic_energy,
                                         calc_grad_potential_energy=calc_grad_potential_energy,
                                         generate_rand_momentum=generate_rand_momentum,
                                         update_params=update_params, update_momentum=update_momentum,
                                         X_batch=X, y_batch=y)

                if accepted:
                    sampled_bnn.append(prop_bnn)
                    energies_list.append(calc_potential_energy(prop_bnn, X, y))
                    acceptances += 1

                self.nns[i] = prop_bnn

        return sampled_bnn, energies_list


