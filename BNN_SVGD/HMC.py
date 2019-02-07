import autograd.numpy as ag_np
import autograd
import matplotlib.pyplot as plt
import numpy as np
import copy
import math
import time

# ag_np.random.seed(42)
mean = 0
variance = 1.0
sigma = 0.1

N = 100
eps = 1
x_train_N = np.linspace(-3, 3, N)
y_train_N = 1 * x_train_N + np.random.normal(0, 0.1, size=(N,))


def relu(x):
    # return ag_np.maximum(x, ag_np.zeros(x.shape))
    return x


def update(target, source, step_size, addition):
    t = copy.deepcopy(target)
    for i in range(len(t)):
        if addition:
            t[i]['w'] += step_size * source[i]['w']
            # t[i]['b'] += step_size * source[i]['b']
        else:
            t[i]['w'] -= step_size * source[i]['w']
            # t[i]['b'] -= step_size * source[i]['b']
    return t


def make_nn_params_as_list_of_dicts(
        n_hiddens_per_layer_list=[1],
        n_dims_input=1,
        n_dims_output=1,
        weight_fill_func=ag_np.random.normal,
        bias_fill_func=ag_np.random.normal):
    nn_param_list = []
    n_hiddens_per_layer_list = [n_dims_input] + n_hiddens_per_layer_list + [n_dims_output]

    # Given full network size list is [a, b, c, d, e]
    # For loop should loop over (a,b) , (b,c) , (c,d) , (d,e)
    for n_in, n_out in zip(n_hiddens_per_layer_list[:-1], n_hiddens_per_layer_list[1:]):
        nn_param_list.append(
            dict(
                w=weight_fill_func(mean, variance, (n_in, n_out)),
                # b=bias_fill_func(mean, variance, (n_out,)),
            ))
    return nn_param_list


def pretty_print_nn_param_list(nn_param_list_of_dict):
    """ Create pretty display of the parameters at each layer
    """
    for ll, layer_dict in enumerate(nn_param_list_of_dict):
        print("Layer %d" % ll)
        print("  w | size %9s | %s" % (layer_dict['w'].shape, layer_dict['w'].flatten()))
        print("  b | size %9s | %s" % (layer_dict['b'].shape, layer_dict['b'].flatten()))


def predict_y_given_x_with_NN(x=None, nn_param_list=None, activation_func=ag_np.tanh):
    """ Predict y value given x value via feed-forward neural net

    Args
    ----
    x : array_like, n_examples x n_input_dims

    Returns
    -------
    y : array_like, n_examples
    """
    for layer_id, layer_dict in enumerate(nn_param_list):
        if layer_id == 0:
            if x.ndim > 1:
                in_arr = x
            else:
                if x.size == nn_param_list[0]['w'].shape[0]:
                    in_arr = x[ag_np.newaxis, :]
                else:
                    in_arr = x[:, ag_np.newaxis]
        else:
            in_arr = activation_func(out_arr)
        out_arr = ag_np.dot(in_arr, layer_dict['w']) # + layer_dict['b']
    return ag_np.squeeze(out_arr)


def calc_potential_energy(cur_bnn_params):
    potential = 0

    for i in range(len(cur_bnn_params)):
        for j in range(cur_bnn_params[i]['w'].shape[0]):
            for k in range(cur_bnn_params[i]['w'].shape[1]):
                potential += ag_np.log(1.0 / math.sqrt(2 * math.pi)) - 0.5 * ag_np.power(cur_bnn_params[i]['w'][j, k],
                                                                                         2)
        # for p in range(cur_bnn_params[i]['b'].shape[0]):
        #     potential += ag_np.log(1.0 / math.sqrt(2 * math.pi)) - 0.5 * ag_np.power(cur_bnn_params[i]['b'][p], 2)


    y_pred = predict_y_given_x_with_NN(x=x_train_N, nn_param_list=cur_bnn_params)
    y_pred = y_pred - y_train_N

    for i in range(y_pred.shape[0]):
        potential += ag_np.log(1.0 / math.sqrt(2 * math.pi) / sigma) - 0.5 / (sigma * sigma) * ag_np.power(y_pred[i], 2)
    return -potential


grad_u = autograd.grad(calc_potential_energy)


def calc_kinetic_energy(cur_momentum_vec):
    kinetic = 0
    for i in range(len(cur_momentum_vec)):
        for j in range(cur_momentum_vec[i]['w'].shape[0]):
            for k in range(cur_momentum_vec[i]['w'].shape[1]):
                kinetic += ag_np.power(cur_momentum_vec[i]['w'][j, k], 2) / 2
        # for p in range(cur_momentum_vec[i]['b'].shape[0]):
        #     kinetic += ag_np.power(cur_momentum_vec[i]['b'][p], 2) / 2
    return kinetic


def make_proposal_via_leapfrog_steps(
        cur_bnn_params, cur_momentum_vec,
        n_leapfrog_steps=1,
        step_size=1.0,
        calc_grad_potential_energy=None):
    """ Construct one HMC proposal via leapfrog integration

    Returns
    -------
    prop_bnn_params : same type/size as cur_bnn_params
    prop_momentum_vec : same type/size as cur_momentum_vec

    """
    # Initialize proposed variables as copies of current values
    prop_bnn_params = copy.deepcopy(cur_bnn_params)
    prop_momentum_vec = copy.deepcopy(cur_momentum_vec)

    # prop_momentum_vec -= step_size * calc_grad_potential_energy(prop_bnn_params)/2.0
    prop_momentum_vec = copy.deepcopy(
        update(prop_momentum_vec, calc_grad_potential_energy(prop_bnn_params), step_size / 2.0, False))

    # This will use the grad of potential energy (use provided function)

    for step_id in range(n_leapfrog_steps):
        # This will use the grad of kinetic energy (has simple closed form)

        if step_id < (n_leapfrog_steps - 1):
            prop_bnn_params = copy.deepcopy(update(prop_bnn_params, prop_momentum_vec, step_size, True))
            prop_momentum_vec = copy.deepcopy(
                update(prop_momentum_vec, calc_grad_potential_energy(prop_bnn_params), step_size, False))
        else:
            prop_bnn_params = copy.deepcopy(update(prop_bnn_params, prop_momentum_vec, step_size, True))
            prop_momentum_vec = copy.deepcopy(
                update(prop_momentum_vec, calc_grad_potential_energy(prop_bnn_params), step_size / 2.0, False))

    prop_momentum_vec = copy.deepcopy(update(prop_momentum_vec, prop_momentum_vec, 2.0, False))
    return prop_bnn_params, prop_momentum_vec


def run_HMC_sampler(
        init_bnn_params=None,
        n_hmc_iters=10,
        n_leapfrog_steps=1,
        step_size=1.0,
        random_seed=42,
        calc_potential_energy=calc_potential_energy,
        calc_kinetic_energy=None,
        calc_grad_potential_energy=None,):
    """ Run HMC sampler for many iterations (many proposals)

    Returns
    -------
    bnn_samples : list
        List of samples of NN parameters produced by HMC
        Can be viewed as 'approximate' posterior samples if chain runs to convergence.
    info : dict
        Tracks energy values at each iteration and other diagnostics.

    References
    ----------
    See Neal's pseudocode algorithm for a single HMC proposal + acceptance:
    https://arxiv.org/pdf/1206.1901.pdf#page=14

    This function repeats many HMC proposal steps.
    """
    # Create random-number-generator with specific seed for reproducibility
    start_time_sec = time.time()
    prng = np.random.RandomState(int(random_seed))

    # Set initial bnn params
    cur_bnn_params = init_bnn_params
    cur_potential_energy = calc_potential_energy(cur_bnn_params)
    # pretty_print_nn_param_list(cur_bnn_params)

    bnn_samples = list()
    potential_energy_list = list()

    n_accept = 0
    for t in range(n_hmc_iters):
        # Draw momentum for CURRENT configuration
        cur_momentum_vec = make_nn_params_as_list_of_dicts(n_hiddens_per_layer_list=[1])
        cur_potential_energy = calc_potential_energy(cur_bnn_params)
        cur_kinetic_energy = calc_kinetic_energy(cur_momentum_vec)

        # Create PROPOSED configuration
        prop_bnn_params, prop_momentum_vec = make_proposal_via_leapfrog_steps(
            cur_bnn_params, cur_momentum_vec,
            n_leapfrog_steps=n_leapfrog_steps,
            step_size=step_size,
            calc_grad_potential_energy=calc_grad_potential_energy)

        prop_potential_energy = calc_potential_energy(prop_bnn_params)

        prop_kinetic_energy = calc_kinetic_energy(prop_momentum_vec)

        accept_proba = np.minimum(1, np.exp(
            -prop_potential_energy - prop_kinetic_energy + cur_kinetic_energy + cur_potential_energy))

        # Draw random value from (0,1) to determine if we accept or not
        if prng.rand() < accept_proba:
            # If here, we accepted the proposal
            n_accept += 1
            cur_bnn_params = prop_bnn_params
            cur_potential_energy = calc_potential_energy(cur_bnn_params)

        # Update list of samples from "posterior"
        bnn_samples.append(cur_bnn_params)
        potential_energy_list.append(cur_potential_energy)

        # Print some diagnostics every 50 iters
        if t < 5 or ((t + 1) % 100 == 0) or (t + 1) == n_hmc_iters:
            accept_rate = float(n_accept) / float(t + 1)
            print("iter %6d/%d after %7.1f sec | accept_rate %.3f" % (
                t + 1, n_hmc_iters, time.time() - start_time_sec, accept_rate))

    return (
        bnn_samples, potential_energy_list,
        dict(
            n_accept=n_accept,
            n_hmc_iters=n_hmc_iters,
            accept_rate=accept_rate),
    )




def do_synthetic_experiment():
    arch = [1]

    nn_params = make_nn_params_as_list_of_dicts(n_hiddens_per_layer_list=arch)

    chain = [(25, 1e-2), (25, 0.005), (40, 1e-2)]
    bnn_all_samples = []

    bnn_samples, potential_list, dict_other = run_HMC_sampler(init_bnn_params=nn_params, n_hmc_iters=300,
                                                              n_leapfrog_steps=20, step_size=0.001, random_seed=3100,
                                                              calc_potential_energy=calc_potential_energy,
                                                              calc_kinetic_energy=calc_kinetic_energy,
                                                              calc_grad_potential_energy=grad_u)

    num_samples = len(bnn_samples)

    error = 0
    num = 0
    for i in range(int(num_samples*0/10), num_samples):
        nn = bnn_samples[i]
        print(nn[0]["w"], nn[1]["w"])

        num += 1


do_synthetic_experiment()