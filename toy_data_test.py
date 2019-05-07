import argparse
import numpy as np
from BNN_SVGD.SVGD_HMC_hybrid import SVGD_SGHMC_hybrid
from BNN_SVGD.SVGD_BNN import SVGD_simple
from BNN_SVGD.HMC_BNN import SG_HMC_BNN
from utils.probability import generate_toy_data
import torch
import os
import matplotlib.pyplot as plt
from utils.MiniBatch import MiniBatch, CyclicMiniBatch
from matplotlib.animation import FuncAnimation
from utils.probability import estimate_jensen_shannon_divergence_from_numerical_distribution

plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
np.random.seed(42)

parser = argparse.ArgumentParser(description="SVGD toy data test_ensemble")
parser.add_argument('--type', type=str, help='Type of test_ensemble', choices=["SVGD", "hybrid", "hmc"])
parser.add_argument('--outdir', type=str, help="Output directory", required=True)
args = parser.parse_args()

outdir = args.outdir

# Generate toy data
x_N_small, y_N_small   = generate_toy_data(N=3)
x_N_medium, y_N_medium = generate_toy_data(N=25)
x_N_big, y_N_big       = generate_toy_data(N=100)

# Get thet type of test_ensemble
test_type = args.type

x_dim = 1
y_dim = 1
num_networks = 100
network_structure = []


def run_svgd_model(model, train_loader, num_epochs):
    positions_over_time = []
    for epoch in range(num_epochs):
        for X_train, y_train in train_loader:
            data = torch.FloatTensor(X_train)
            target = torch.FloatTensor(y_train)
            model.step_svgd(data, target, step_size)

            curr_position = []
            for nnid in range(len(model.nns)):
                weight1 = model.nns[nnid].nn_params[0].weight.detach().numpy()[0]
                weight2 = model.nns[nnid].nn_params[1].weight.detach().numpy()[0]
                curr_position.append([weight1[0], weight2[0]])
            positions_over_time.append((curr_position, True))
    return positions_over_time


def plot_particle_positions(outdir, particles, num_networks, rbf, jsd):
    plt.scatter(particles[:, 0], particles[:, 1])
    title = "{} particles, rbf length scale = {}".format(num_networks, rbf)
    print("For {} particles, rbf = {}, jsd = {}".format(num_networks, rbf, jsd))
    plt.title(title)
    plt.ylim([-15, 15])
    plt.xlim([-15, 15])
    savefile = os.path.join(outdir, "C={}_rbf={}_jsd={}.png".format(num_networks, rbf, jsd))
    plt.savefig(savefile)
    plt.close()


if test_type == "SVGD":
    """
    - Effect of length scale on the final distribution
    - Verify and observe the effect of increasing number of particles
    - low data, medium data and high data (full batch vs mini-batch)
    """

    num_networks = 20
    batch_size   = 100
    p_sigma = 1
    l_sigma = 1
    num_epochs = 100
    step_size = 0.01

    # Experiment on length scale of rbf kernel
    for rbf in [0.9, 1, 10, 50, 100]:
        # Initialize train loader
        train_loader = MiniBatch(xs=x_N_big, ys=y_N_big, batch_size=batch_size)
        model = SVGD_simple(x_dim, y_dim, num_networks, network_structure, l_sigma, p_sigma, rbf)
        positions_over_time = run_svgd_model(model, train_loader, num_epochs)

        # After training
        # Compute the jensen shannon divergence
        print(positions_over_time[-1][0])
        jsd = estimate_jensen_shannon_divergence_from_numerical_distribution(np.array(positions_over_time[-1][0]),
                                                                             x_N=x_N_big, y_N=y_N_big, plot=False)
        jsd = np.around(jsd, decimals=4)

        particles = np.array(positions_over_time[-1][0])
        plt.scatter(particles[:, 0], particles[:, 1])
        title = "{} particles, rbf length scale = {}".format(num_networks, rbf)
        print("For {} particles, rbf = {}, jsd = {}".format(num_networks, rbf, jsd))
        plt.title(title)
        plt.ylim([-15, 15])
        plt.xlim([-15, 15])
        savefile = os.path.join(outdir, "C={}_rbf={}_jsd={}.png".format(num_networks, rbf, jsd))
        plt.savefig(savefile)
        plt.close()

    # for num_networks in [1, 10, 50, 100]:
    #     # Initialize train loader
    #     train_loader = MiniBatch(xs=x_N_medium, ys=y_N_medium, batch_size=batch_size)
    #     model = SVGD_simple(x_dim, y_dim, num_networks, network_structure, l_sigma, p_sigma, rbf)
    #     num_epochs = max(50, num_epochs * 2)
    #     positions_over_time = run_svgd_model(model, train_loader, num_epochs)
    #
    #     # After training
    #     # Compute the jensen shannon divergence
    #     jsd = estimate_jensen_shannon_divergence_from_numerical_distribution(np.array(positions_over_time[-1][0]),
    #                                                                          x_N=x_N_medium, y_N=y_N_medium, plot=False)
    #     jsd = np.around(jsd, decimals=4)
    #
    #     particles = np.array(positions_over_time[-1][0])
    #     plt.scatter(particles[:, 0], particles[:, 1])
    #     title = "{} particles, rbf length scale = {}".format(num_networks, rbf)
    #     plt.title(title)
    #     savefile = os.path.join(outdir, "C={}_rbf={}_jsd={}.png".format(num_networks, rbf, jsd))
    #     plt.savefig(savefile)
    #     plt.close()

    # Test on small data
    # Test on medium data
    # Test on big data

elif test_type == "hybrid":
    """
    - Pattern of particle movement
    - png showing final particle location
    - low data, medium and high data (full batch and mini-batch)
    """
    num_networks = 20
    batch_size   = 100
    p_sigma = 1
    l_sigma = 1
    num_epochs = 200
    step_size = 0.01
    rbf = 1

    # First train on small toy data set
    # Small data set has N = 3 data points
    train_loader = CyclicMiniBatch(xs=x_N_small, ys=y_N_small, batch_size=10)
    model = SVGD_SGHMC_hybrid(x_dim, y_dim, num_networks, network_structure, l_sigma, p_sigma, rbf, svgd_step_size=0.01, \
                              momentum=0.999, hmc_n_leapfrog_steps=10, hmc_step_size=0.001)
    positions_over_time = run_svgd_model(model, train_loader, num_epochs)

    jsd = estimate_jensen_shannon_divergence_from_numerical_distribution(np.array(positions_over_time[-1][0]),
                                                                         x_N=x_N_small, y_N=y_N_small, plot=False)
    jsd = np.around(jsd, decimals=4)
    plot_particle_positions(outdir, positions_over_time[-1][0], num_networks, rbf, jsd)

    # Train on medium data set
    # Medium data set has N = 25 data points
    train_loader = CyclicMiniBatch(xs=x_N_medium, ys=y_N_medium, batch_size=25)
    model = SVGD_SGHMC_hybrid(x_dim, y_dim, num_networks, network_structure, l_sigma, p_sigma, rbf, svgd_step_size=0.01, \
                              momentum=0.999, hmc_n_leapfrog_steps=10, hmc_step_size=0.001)
    positions_over_time = run_svgd_model(model, train_loader, num_epochs)

    jsd = estimate_jensen_shannon_divergence_from_numerical_distribution(np.array(positions_over_time[-1][0]),
                                                                         x_N=x_N_medium, y_N=y_N_medium, plot=False)
    jsd = np.around(jsd, decimals=4)
    plot_particle_positions(outdir, positions_over_time[-1][0], num_networks, rbf, jsd)

    # Train on big toy data set
    # 'Big' dataset has N = 100 data points
    # Full batch
    train_loader = CyclicMiniBatch(xs=x_N_big, ys=y_N_big, batch_size=100)
    model = SVGD_SGHMC_hybrid(x_dim, y_dim, num_networks, network_structure, l_sigma, p_sigma, rbf, svgd_step_size=0.01, \
                              momentum=0.999, hmc_n_leapfrog_steps=10, hmc_step_size=0.001)
    positions_over_time = run_svgd_model(model, train_loader, num_epochs)

    # Mini-batch
    train_loader = CyclicMiniBatch(xs=x_N_big, ys=y_N_big, batch_size=10)
    model = SVGD_SGHMC_hybrid(x_dim, y_dim, num_networks, network_structure, l_sigma, p_sigma, rbf, svgd_step_size=0.01, \
                              momentum=0.999, hmc_n_leapfrog_steps=10, hmc_step_size=0.001)
    positions_over_time = run_svgd_model(model, train_loader, num_epochs)


else:
    """
    - low data, medium and high data (full batch and mini-batch)
    """

    pass


