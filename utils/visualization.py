import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.stats
from matplotlib.animation import FuncAnimation
import seaborn as sns
from matplotlib import colors, ticker

def plot_weight_distribution_hmc(sampled_bnn, filename=None, show=True):
    distribution = []

    for bnn in sampled_bnn:
        weight1 = bnn.nn_params[0].weight.detach().numpy()[0][0]
        weight2 = bnn.nn_params[1].weight.detach().numpy()[0][0]
        distribution.append([weight1, weight2])

    distribution = np.array(distribution)

    # Plot the "correct" distribution
    refx_neg = np.linspace(-10, -0.1, 19)
    refx_pos = np.linspace(0.1, 10, 19)
    refy_neg = 1. / refx_neg
    refy_pos = 1. / refx_pos

    plt.plot(refx_neg, refy_neg, "r")
    plt.plot(refx_pos, refy_pos, "r")
    plt.scatter(distribution[:, 0], distribution[:, 1])

    if filename:
        plt.savefig(filename)
    if show:
        plt.show()

    plt.close()

    return distribution


def track_position_over_time(sampled_bnn):
    # Initialize the figure
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)

    def update(i):
        plt.clf()
        label = 'Sample num {0}'.format(i)
        # Update the line and the axes (with a new xlabel). Return a tuple of
        # "artists" that have to be redrawn for this frame.
        bnn = sampled_bnn[i]

        w1 = bnn.nn_params[0].weight.detach().numpy()[0][0]
        w2 = bnn.nn_params[1].weight.detach().numpy()[0][0]
        plt.scatter(w1, w2)
        plt.ylim(-10, 10)
        plt.xlim(-10, 10)
        plt.xlabel(label)

    # FuncAnimation will call the 'update' function for each frame; here
    # animating over 10 frames, with an interval of 200ms between frames.
    anim = FuncAnimation(fig, update, frames=np.arange(0, len(sampled_bnn)), interval=100)
    anim.save('./position-over-time-svgd.gif', dpi=80, writer='imagemagick')
    plt.show()


def generate_observed_data(N=5, true_sigma_y=0.333):

    ## Assume given a regular grid of 1D x values between -1 and 1
    x_N = np.linspace(-1, 1, N)

    ## Draw real-valued labels y from a Normal with mean x and variance sigma_lik
    y_N = np.random.normal(x_N, true_sigma_y)

    return x_N, y_N



