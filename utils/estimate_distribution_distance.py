import numpy as np
from scipy.stats import entropy
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
import scipy


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


def make_log_pdf_contour_plot(a_GH, b_GH, log_pdf_GH, name, vmin=-10000):
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
    plt.title(name)
    plt.show()


def estimate_jensen_shannon_divergence_from_numerical_distribution(particles, x_N, y_N, h=0.2, xlimit=[-4, 4], ylimit=[-4, 4], grid_N=100, plot=True):
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

    # Fit the particles
    kde1 = KernelDensity(kernel='gaussian', bandwidth=h).fit(particles)

    # Create mesh grid
    x_grid_N = grid_N
    x_grid = np.linspace(xlimit[0], xlimit[1], x_grid_N)
    y_grid_N = grid_N
    y_grid = np.linspace(ylimit[0], ylimit[1], y_grid_N)

    x_GH, y_GH = np.meshgrid(x_grid, y_grid)
    xy_grid = np.vstack([x_GH.flatten(), y_GH.flatten()]).T
    log_pdf_kde = kde1.score_samples(xy_grid)
    pdf_kde = np.exp(log_pdf_kde)

    sigma_prior = 1
    sigma_y = 1

    # Compute log prior, this is straight forward
    log_pdf_prior_M = scipy.stats.multivariate_normal.logpdf(
        xy_grid, np.zeros(2), sigma_prior * np.eye(2))
    log_pdf_prior_GH = log_pdf_prior_M.reshape((grid_N, grid_N))


    # Compute log likelihood
    log_pdf_lik_M = np.zeros(xy_grid.shape[0])

    for mm in range(xy_grid.shape[0]):
        a = xy_grid[mm, 0]
        b = xy_grid[mm, 1]
        log_ll = scipy.stats.norm.logpdf(y_N, a * b * x_N, sigma_y)
        log_pdf_lik_M[mm] = np.sum(log_ll)

    log_pdf_lik_GH = log_pdf_lik_M.reshape((grid_N, grid_N))

    # Compute unnormalized log posterior
    log_pdf_post_GH = log_pdf_lik_GH + log_pdf_prior_GH
    log_pdf_post_vector = log_pdf_post_GH.flatten()

    # Compute Posterior
    pdf_post_vector = np.exp(log_pdf_post_vector)
    pdf_post_vector = pdf_post_vector/np.sum(pdf_post_vector)

    # Compute jensen-shannon divergence
    # JSD(q|p) = 0.5 * KL(q|m) + 0.5 * KL(p|m) where m = 0.5*(p+q)
    q = 0.5 * (pdf_post_vector + pdf_kde)
    jsd = 0.5 * entropy(pdf_post_vector, q) + 0.5 * entropy(pdf_kde, q)

    if plot:
        # Plot data
        plt.plot(x_N, y_N, 'k.')
        plt.xlim([-4, 4])
        plt.ylim([-4, 4])
        plt.title("Data distribution")
        plt.show()
        # Plot contour plots for prior
        make_log_pdf_contour_plot(x_grid, y_grid, log_pdf_prior_GH, "Prior distribution", -100)
        # Plot contour for likelihood
        make_log_pdf_contour_plot(x_grid, y_grid, log_pdf_lik_GH, "likelihood", -100)
        # Plot contour plots for posterior
        make_log_pdf_contour_plot(x_grid, y_grid, log_pdf_post_GH, "posterior", -100)
        # Plot contour plots for KDE
        x_particle = particles[:, 0]
        y_particle = particles[:, 1]
        plt.scatter(x_particle, y_particle)
        plt.title("Particles")
        plt.show()
        make_log_pdf_contour_plot(x_grid, y_grid, np.reshape(log_pdf_kde, (grid_N, grid_N)), "kde(particles), jsd = {}".format(jsd), -100)

    return jsd



# Estimate which h length scale is 'best'
# Generate data

y_sigma = 1

N = 100

x_N = np.linspace(-3, 3, N)
y_N = x_N + np.random.normal(0, y_sigma, size=(N,))

# The idea is that
# If we have a distribution P which we think is close to the true distribution
# And we generate M samples from P, as M -> infinity, JSD(particles|true distribution) goes to 0
# And we pick a length scale h that satisfies this


# What does the posterior distribution looks like
# a, b ~ N(0, 1)
# y ~ N(x * a * b, y_sigma)
# what is P(a, b), if we know P(a)P(b)P(y|x, a, b), we can use MCMC-metropolis hastings
# So I can do MCMC-MH

def sample_mcmc_mh(x_N, y_N, at, bt, y_sigma, p_sigma):
    a_prop = np.random.normal(at, 1)
    b_prop = np.random.normal(bt, 1)

    lik_prop = np.sum(scipy.stats.norm.logpdf(y_N, a_prop * b_prop * x_N, y_sigma))
    prior_prop = scipy.stats.norm.logpdf(0, a_prop, p_sigma) + scipy.stats.norm.logpdf(0, b_prop, p_sigma)

    lik_t    = np.sum(scipy.stats.norm.logpdf(y_N, at * bt * x_N, y_sigma))
    prior_t  = scipy.stats.norm.logpdf(0, at, p_sigma) + scipy.stats.norm.logpdf(0, bt, p_sigma)

    a = np.exp(lik_prop + prior_prop - lik_t - prior_t)

    if np.random.rand() < a:
        return a_prop, b_prop, True
    return at, bt, False


# The simplest is to re-construct the experiment, but that is not the posterior

p_sigma = 1

def generate_particles(M, h, plot=False):
    acceptances = 0
    particles = []
    at = 1.5
    bt = 1.5

    while acceptances < int(M/2):
        at, bt, accepted = sample_mcmc_mh(x_N, y_N, at, bt, y_sigma, p_sigma)
        if accepted:
            particles.append([at, bt])
            acceptances += 1
        # else:
        #     at = np.random.normal(0, 1)
        #     bt = np.random.normal(0, 1)

    at = -1.5
    bt = -1.5
    while acceptances < M:
        at, bt, accepted = sample_mcmc_mh(x_N, y_N, at, bt, y_sigma, p_sigma)
        if accepted:
            particles.append([at, bt])
            acceptances += 1

    particles = np.array(particles)
    jsd = estimate_jensen_shannon_divergence_from_numerical_distribution(particles, x_N, y_N, h=h, plot=plot)
    print("With h = %.3f, M = %d, jsd = %.4f" % (h, M, jsd))


h = 0.1

for h in [0.1, 0.05, 0.025, 0.02, 0.015, 0.01, 0.005, 0.001]:
    generate_particles(M=5000, h=h, plot=False)