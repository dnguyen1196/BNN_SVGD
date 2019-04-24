import argparse
import numpy as np
from BNN_SVGD.SVGD_HMC_hybrid import SVGD_SGHMC_hybrid
from BNN_SVGD.SVGD_BNN import SVGD_simple
from BNN_SVGD.HMC_BNN import SG_HMC_BNN
from utils.probability import generate_toy_data
from utils.visualization import plot_weight_distribution_hmc
import torch
import os
import matplotlib.pyplot as plt
from utils.MiniBatch import MiniBatch, CyclicMiniBatch
from matplotlib.animation import FuncAnimation
from utils.probability import estimate_jensen_shannon_divergence_from_numerical_distribution

plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
np.random.seed(42)

parser = argparse.ArgumentParser(description="Hybrid toy data test")
parser.add_argument('--outdir', type=str, help="Output directory", required=True)
args = parser.parse_args()

outdir = args.outdir

# Generate toy data
x_N_small, y_N_small   = generate_toy_data(N=3)
x_N_medium, y_N_medium = generate_toy_data(N=25)
x_N_big, y_N_big       = generate_toy_data(N=100)


x_dim = 1
y_dim = 1
network_structure = []


def plot_particle_positions(outdir, particles, num_networks, rbf, jsd):
    plt.scatter(particles[:, 0], particles[:, 1])
    print("For {} particles, rbf = {}, jsd = {}".format(num_networks, rbf, jsd))
    plt.ylim([-4, 4])
    plt.xlim([-4, 4])
    savefile = os.path.join(outdir, "Hybrid_C={}_rbf={}_jsd={}.png".format(num_networks, rbf, jsd))
    plt.savefig(savefile)
    plt.close()


def track_position_over_time(outdir, positions_over_time, N, num_networks, n_svgd, n_hmc, num_iters ,jsd):
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

        plt.ylim(-4, 4)
        plt.xlim(-4, 4)
        plt.xlabel(label)

    anim = FuncAnimation(fig, update, frames=np.arange(0, len(positions_over_time)), interval=100)
    filename="hybrid_particle_N={}_C={}_nsvgd={}_nhmc={}_total={}_jsd={}.gif".format\
                    (N, num_networks, n_svgd, n_hmc, num_iters , np.around(jsd, 4))

    print("Saving to file ", os.path.join(outdir, filename))
    anim.save(filename, dpi=80, writer='imagemagick')


# Test on small data
# Test on medium data
# Test on big data

num_networks = 25
p_sigma = 1
l_sigma = 1
step_size = 0.001
rbf = .1


# x_N = x_N_medium
# y_N = y_N_medium
# N = 25
# batch_size = 25


x_N = x_N_big
y_N = y_N_big
N = 100
batch_size = 10

# for x_N, y_N, N in [(x_N_small, y_N_small, 3), (x_N_medium, y_N_medium, 25), (x_N_big, y_N_big, 100)]:

n_svgd = 1001
n_hmc = 100
num_iters = 1000

# Initialize train loader 
# NOTE: it has to be Cyclic
train_loader = CyclicMiniBatch(xs=x_N, ys=y_N, batch_size=batch_size)
model = SVGD_SGHMC_hybrid(x_dim, y_dim, num_networks, network_structure, l_sigma, p_sigma, rbf, \
                          svgd_step_size=step_size, hmc_step_size=0.001, hmc_n_leapfrog_steps=15)

model.fit(train_loader=train_loader, num_iterations=num_iters, svgd_iteration=n_svgd, hmc_iteration=n_hmc)

positions_over_time = model.positions_over_time
sampled_bnn  = model.hmc_sampled_bnn

particles = plot_weight_distribution_hmc(sampled_bnn, show=False)

# After training Compute the jensen shannon divergence
jsd = estimate_jensen_shannon_divergence_from_numerical_distribution(particles,
                                                                     x_N=x_N, y_N=y_N, plot=False)
jsd = np.around(jsd, decimals=4)

plot_particle_positions(outdir, particles, num_networks, rbf, jsd)
# track_position_over_time(outdir, positions_over_time, N, num_networks, n_svgd, n_hmc, num_iters ,jsd)


