import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.stats
from matplotlib.animation import FuncAnimation
import seaborn as sns
from matplotlib import colors, ticker

def plot_weight_distribution(sampled_bnn, data, target):
    distribution = []
    mse_arr = []

    for bnn in sampled_bnn:
        yhat = bnn.forward(data)
        mse  = torch.mean((torch.squeeze(yhat) - torch.squeeze(target))** 2).detach().numpy()
        if np.isnan(mse):
            continue
        mse_arr.append(mse)
        weight1 = bnn.nn_params[0].weight.detach().numpy()[0][0]
        weight2 = bnn.nn_params[1].weight.detach().numpy()[0][0]
        distribution.append([weight1, weight2])

    distribution = np.array(distribution)
    print("Average mse: ", np.mean(mse_arr))

    # Plot the "correct" distribution
    refx_neg = np.linspace(-10, -0.1, 19)
    refx_pos = np.linspace(0.1, 10, 19)
    refy_neg = 1. / refx_neg
    refy_pos = 1. / refx_pos

    plt.plot(refx_neg, refy_neg, "r")
    plt.plot(refx_pos, refy_pos, "r")

    plt.scatter(distribution[:, 0], distribution[:, 1])
    plt.show()

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

# # Generate data
# N = 100
# x_N, y_N = generate_observed_data(N=100, true_sigma_y=0.333)
#
# plt.plot(x_N, y_N, 'k.')
# plt.xlim([-1.5, 1.5])
# plt.ylim([-1.5, 1.5])
# plt.show()
#
# sigma_prior = 1.0
# sigma_y = 0.3
# a_min = -10
# a_max = +10
#
# N = x_N.size
# G = 100
# H = 103
#
# a_grid = np.linspace(a_min, a_max, G)
# b_grid = np.linspace(a_min, a_max, H)
# a_GH, b_GH = np.meshgrid(a_grid, b_grid)
# G, H = a_GH.shape
# ab_M2 = np.vstack([a_GH.flatten(), b_GH.flatten()]).T
#
#
# # Compute log prior
# log_pdf_prior_M = scipy.stats.multivariate_normal.logpdf(
#     ab_M2, np.zeros(2), sigma_prior*np.eye(2))
# log_pdf_prior_GH = log_pdf_prior_M.reshape(a_GH.shape)
#
# # Compute log likelihood
# mean_MN = np.prod(ab_M2, axis=1, keepdims=1) * x_N[np.newaxis,:]
# log_pdf_lik_M = np.zeros(mean_MN.shape[0])
# for mm in range(ab_M2.shape[0]):
#     a = ab_M2[mm, 0]
#     b = ab_M2[mm, 1]
#     log_ll = scipy.stats.norm.logpdf(y_N, a * b * x_N, sigma_y)
#     log_pdf_lik_M[mm] = np.sum(log_ll)
#
# log_pdf_lik_GH = log_pdf_lik_M.reshape((G, H))
#
#
# # Unnormalized log posterior
# log_pdf_post_GH = log_pdf_lik_GH + log_pdf_prior_GH
#
# # Function to do contour plots
# def make_log_pdf_contour_plot(a_GH, b_GH, log_pdf_GH, vmin=-10000):
#     M = log_pdf_GH.max() + 1e-5
#     log_pdf_GH = log_pdf_GH - M
#     print(log_pdf_GH.min())
#     print(log_pdf_GH.max())
#     #lev_exp = np.arange(
#     #    np.floor(np.log10(log_pdf_GH.min())-1),
#     #    np.ceil(np.log10(log_pdf_GH.max())+1))
#     #levs = np.power(10, lev_exp)
#     cs = plt.contourf(
#         a_GH, b_GH, log_pdf_GH, vmin=vmin, vmax=0, levels=np.linspace(vmin, 0, 25))
#     plt.colorbar()
#     plt.show()
#
#
# # Make contour plots for prior
# make_log_pdf_contour_plot(a_GH, b_GH, log_pdf_prior_GH, -100)
#
# # For likelihood
# make_log_pdf_contour_plot(a_GH, b_GH, log_pdf_lik_GH, -100)
#
# # For posterior
# make_log_pdf_contour_plot(a_GH, b_GH, log_pdf_post_GH, -100)
