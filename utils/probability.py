import numpy as np
from scipy.stats import entropy
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
from numpy.random import multivariate_normal
import matplotlib.pyplot as plt
import scipy
import re


def estimate_KL_divergence_discrete(Xs1, Xs2, h=0.2, xlimit=[-4, 4], ylimit=[-4, 4]):
    '''

    :param Xs1:
    :param Xs2:
    :param h:
    :param xlimit:
    :param ylimit:
    :return:
    '''

    # Create kernel density estimator
    kde1 = KernelDensity(kernel='gaussian', bandwidth=h).fit(Xs1)
    kde2 = KernelDensity(kernel='gaussian', bandwidth=h).fit(Xs2)

    # Create mesh grid
    x_grid_N = 100
    x_grid = np.linspace(xlimit[0], xlimit[1], x_grid_N)
    y_grid_N = 100
    y_grid = np.linspace(ylimit[0], ylimit[1], y_grid_N)

    x_GH, y_GH = np.meshgrid(x_grid, y_grid)

    xy_grid = np.vstack([x_GH.flatten(), y_GH.flatten()]).T

    score_1 = kde1.score_samples(xy_grid)
    score_2 = kde2.score_samples(xy_grid)

    p1 = np.exp(score_1)
    p2 = np.exp(score_2)

    kl_1_2 = entropy(p1, p2)
    return kl_1_2


def generate_observed_data(N=5, true_sigma_y=1):
    """

    :param N:
    :param true_sigma_y:
    :return:
    """
    ## Assume given a regular grid of 1D x values between -1 and 1
    x_N = np.linspace(-3, 3, N)

    ## Draw real-valued labels y from a Normal with mean x and variance sigma_lik
    y_N = np.random.normal(x_N, true_sigma_y)

    return x_N, y_N


# Function to do contour plots
def make_log_pdf_contour_plot(a_GH, b_GH, log_pdf_GH, vmin=-10000):
    """

    :param a_GH:
    :param b_GH:
    :param log_pdf_GH:
    :param vmin:
    :return:
    """
    M = log_pdf_GH.max() + 1e-5
    log_pdf_GH = log_pdf_GH - M
    cs = plt.contourf(
        a_GH, b_GH, log_pdf_GH, vmin=vmin, vmax=0, levels=np.linspace(vmin, 0, 25))
    plt.colorbar()
    plt.show()


def estimate_kl_divergence_discrete_true_posterior(particles, x_N, y_N, h=0.2, xlimit=[-4, 4], ylimit=[-4, 4], grid_N=100):
    """

    :param particles:
    :param x_N:
    :param y_N:
    :param h:
    :param xlimit:
    :param ylimit:
    :param grid_N:
    :return:
    """

    kde1 = KernelDensity(kernel='gaussian', bandwidth=h).fit(particles)

    # Create mesh grid
    x_grid_N = grid_N
    x_grid = np.linspace(xlimit[0], xlimit[1], x_grid_N)
    y_grid_N = grid_N
    y_grid = np.linspace(ylimit[0], ylimit[1], y_grid_N)

    x_GH, y_GH = np.meshgrid(x_grid, y_grid)

    xy_grid = np.vstack([x_GH.flatten(), y_GH.flatten()]).T

    score_1 = kde1.score_samples(xy_grid)
    p1 = np.exp(score_1)

    plt.plot(x_N, y_N, 'k.')
    plt.xlim([-4, 4])
    plt.ylim([-4, 4])
    plt.show()

    sigma_prior = 1
    sigma_y = 1

    # Compute log prior, this is straight forward
    log_pdf_prior_M = scipy.stats.multivariate_normal.logpdf(
        xy_grid, np.zeros(2), sigma_prior * np.eye(2))
    log_pdf_prior_GH = log_pdf_prior_M.reshape((grid_N, grid_N))

    # Plot contour plots for prior
    make_log_pdf_contour_plot(x_grid, y_grid, log_pdf_prior_GH, -100)

    # Compute log likelihood
    log_pdf_lik_M = np.zeros(xy_grid.shape[0])

    for mm in range(xy_grid.shape[0]):
        a = xy_grid[mm, 0]
        b = xy_grid[mm, 1]
        log_ll = scipy.stats.norm.logpdf(y_N, a * b * x_N, sigma_y)
        log_pdf_lik_M[mm] = np.sum(log_ll)

    log_pdf_lik_GH = log_pdf_lik_M.reshape((grid_N, grid_N))

    # Plot contour for likelihood
    make_log_pdf_contour_plot(x_grid, y_grid, log_pdf_lik_GH, -100)

    # Compute unnormalized log posterior
    log_pdf_post_GH = log_pdf_lik_GH + log_pdf_prior_GH
    log_pdf_post_vector = log_pdf_post_GH.flatten()

    # Compute Posterior
    pdf_post_vector = np.exp(log_pdf_post_vector)
    pdf_post_vector = pdf_post_vector/np.sum(pdf_post_vector)

    # Plot contour plots for posterior
    make_log_pdf_contour_plot(x_grid, y_grid, log_pdf_post_GH, -100)

    return entropy(pdf_post_vector, p1)