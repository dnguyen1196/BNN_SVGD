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

parser = argparse.ArgumentParser(description="SVGD toy data test")
parser.add_argument('--type', type=str, help='Type of test', choices=["SVGD", "hybrid", "hmc"])
parser.add_argument('--outdir', type=str, help="Output directory", required=True)
args = parser.parse_args()

outdir = args.outdir

# Generate toy data
x_N_small, y_N_small   = generate_toy_data(N=3)
x_N_medium, y_N_medium = generate_toy_data(N=25)
x_N_big, y_N_big       = generate_toy_data(N=100)

# Get thet type of test
test_type = args.type

x_dim = 1
y_dim = 1
network_structure = []


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


def track_position_over_time(positions_over_time, N, num_networks, n_svgd, n_hmc, num_iters ,jsd):
    # Initialize the figure
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)

    def update(i):
        plt.clf()
        pos, svgd = positions_over_time[i]

        position = np.array(pos)
        xpos = position[:, 0]
        ypos = position[:, 1]

        label = 'Epoch {0}'.format(i)
        color = "b" if svgd else "r"
        plt.scatter(xpos, ypos, c=color)

        plt.ylim(-15, 15)
        plt.xlim(-15, 15)
        plt.xlabel(label)

    anim = FuncAnimation(fig, update, frames=np.arange(0, len(positions_over_time)), interval=100)
    filename="./particle_N={}_C={}_nsvgd={}_nhmc={}_total={}_jsd={}.gif".format\
                    (N, num_networks, n_svgd, n_hmc, num_iters , np.around(jsd, 4))
    print(filename)

    anim.save(filename, dpi=80, writer='imagemagick')


# Test on small data
# Test on medium data
# Test on big data


num_networks = 50
batch_size   = 100
p_sigma = 1
l_sigma = 1
num_epochs = 100
step_size = 0.001
rbf = .1


x_N = x_N_medium
y_N = y_N_medium

# Initialize train loader
train_loader = CyclicMiniBatch(xs=x_N, ys=y_N, batch_size=batch_size)
model = SVGD_SGHMC_hybrid(x_dim, y_dim, num_networks, network_structure, l_sigma, p_sigma, rbf)

n_svgd = 300
n_hmc = 200
num_iters = 500
model.fit(train_loader=train_loader, num_iterations=num_iters, svgd_iteration=n_svgd, hmc_iteration=n_hmc)

positions_over_time = model.positions_over_time



# After training Compute the jensen shannon divergence
jsd = estimate_jensen_shannon_divergence_from_numerical_distribution(np.array(positions_over_time[-1][0]),
                                                                     x_N=x_N, y_N=y_N, plot=False)
jsd = np.around(jsd, decimals=4)

particles = np.array(positions_over_time[-1][0])
plt.scatter(particles[:, 0], particles[:, 1])
title = "{} particles".format(num_networks, rbf)
plt.title(title)
print("Hybrid algorithm, C = {}, rbf = {}, jsd = {}".format(num_networks, rbf, jsd))
plt.ylim([-5, 5])
plt.xlim([-5, 5])
savefile = os.path.join(outdir, "Hybrid_C={}_rbf={}_jsd={}.png".format(num_networks, rbf, jsd))
plt.savefig(savefile)
plt.close()

