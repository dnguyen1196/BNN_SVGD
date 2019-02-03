import theano.tensor as T
import theano
import numpy as np
from scipy.spatial.distance import pdist, squareform
import random
import time

'''
    Sample code to reproduce our results for the Bayesian neural network example.
    Our settings are almost the same as Hernandez-Lobato and Adams (ICML15) https://jmhldotorg.files.wordpress.com/2015/05/pbp-icml2015.pdf
    Our implementation is also based on their Python code.

    p(y | W, X, \gamma) = \prod_i^N  N(y_i | f(x_i; W), \gamma^{-1})
    p(W | \lambda) = \prod_i N(w_i | 0, \lambda^{-1})
    p(\gamma) = Gamma(\gamma | a0, b0)
    p(\lambda) = Gamma(\lambda | a0, b0)

    The posterior distribution is as follows:
    p(W, \gamma, \lambda) = p(y | W, X, \gamma) p(W | \lambda) p(\gamma) p(\lambda) 
    To avoid negative values of \gamma and \lambda, we update loggamma and loglambda instead.

    Copyright (c) 2016,  Qiang Liu & Dilin Wang
    All rights reserved.
'''


class svgd_bayesnn:
    '''
        We define a one-hidden-layer-neural-network specifically. We leave extension of deep neural network as our future work.

        Input
            -- X_train: training dataset, features
            -- y_train: training labels
            -- batch_size: sub-sampling batch size
            -- max_iter: maximum iterations for the training procedure
            -- M: number of particles are used to fit the posterior distribution
            -- n_hidden: number of hidden units
            -- a0, b0: hyper-parameters of Gamma distribution
            -- master_stepsize, auto_corr: parameters of adgrad
    '''

    def __init__(self, X_train, y_train, X_test, y_test, batch_size=100, max_iter=1000, M=20, n_hidden=50, a0=1, b0=0.1,
                 master_stepsize=1e-3, auto_corr=0.9):
        self.n_hidden = n_hidden
        self.d = X_train.shape[1]  # number of data, dimension

        print("Dimension d:", self.d)
        self.M = M

        num_vars = self.d * n_hidden + 1 + 2  # w1: d*n_hidden; b1: 1; 2 variances
        self.theta = np.zeros([self.M, num_vars])  # particles, will be initialized later

        '''
            We keep the last 10% (maximum 500) of training data points for model developing
        '''
        size_dev = min(int(np.round(0.1 * X_train.shape[0])), 500)
        X_dev, y_dev = X_train[-size_dev:], y_train[-size_dev:]
        X_train, y_train = X_train[:-size_dev], y_train[:-size_dev]

        '''
            Theano symbolic variables
            Define the neural network here
        '''
        X = T.matrix('X')  # Feature matrix
        y = T.vector('y')  # labels

        w_1 = T.matrix('w_1')  # weights between input layer and hidden layer
        b_1 = T.vector('b_1')  # bias vector of hidden layer

        N = T.scalar('N')  # number of observations

        log_gamma = T.scalar('log_gamma')  # variances related parameters
        log_lambda = T.scalar('log_lambda')

        ###
        prediction_1_layer = T.dot(X, w_1) + b_1

        ''' define the log posterior distribution '''
        # TODO: replace prediction
        log_lik_data = -0.5 * X.shape[0] * (T.log(2 * np.pi) - log_gamma) - (T.exp(log_gamma) / 2) * T.sum(
            T.power(prediction_1_layer - y, 2))

        log_prior_data = (a0 - 1) * log_gamma - b0 * T.exp(log_gamma) + log_gamma

        log_prior_w_1_layer = -0.5 * (num_vars - 2) * (T.log(2 * np.pi) - log_lambda) - (T.exp(log_lambda) / 2) * (
            (w_1 ** 2).sum() +  (b_1 ** 2).sum()) + (a0 - 1) * log_lambda - b0 * T.exp(log_lambda) + log_lambda


        # sub-sampling mini-batches of data, where (X, y) is the batch data, and N is the number of whole observations
        # TODO: replace with log_prior_w
        log_posterior = (log_lik_data * N / X.shape[0] + log_prior_data + log_prior_w_1_layer)

        dw_1, db_1, d_log_gamma, d_log_lambda = T.grad(log_posterior, [w_1, b_1, log_gamma, log_lambda], disconnected_inputs="ignore")

        # automatic gradient
        logp_gradient = theano.function(
            inputs=[X, y, w_1, b_1, log_gamma, log_lambda, N],
            outputs=[dw_1, db_1, d_log_gamma, d_log_lambda]
        )

        # prediction function
        # TODO: reaplce with prediction
        self.nn_predict = theano.function(inputs=[X, w_1, b_1], outputs=prediction_1_layer, on_unused_input='ignore')

        '''
            Training with SVGD
        '''
        # normalization

        # X_train, y_train = self.normalization(X_train, y_train) # TODO; uncomment

        N0 = X_train.shape[0]  # number of observations

        ''' initializing all particles '''
        for i in range(self.M):
            w1, b1, loggamma, loglambda = self.init_weights(a0, b0)
            # use better initialization for gamma
            ridx = np.random.choice(range(X_train.shape[0]), \
                                    np.min([X_train.shape[0], 1000]), replace=False)

            y_hat = self.nn_predict(X_train[ridx, :], w1, b1)

            loggamma = -np.log(np.mean(np.power(y_hat - y_train[ridx], 2)))
            self.theta[i, :] = self.pack_weights(w1, b1, loggamma, loglambda)

        grad_theta = np.zeros([self.M, num_vars])  # gradient
        # adagrad with momentum
        fudge_factor = 1e-6
        historical_grad = 0
        for iter in range(max_iter):
            # sub-sampling
            batch = [i % N0 for i in range(iter * batch_size, (iter + 1) * batch_size)]
            for i in range(self.M):
                w1, b1, loggamma, loglambda = self.unpack_weights(self.theta[i, :])
                dw1, db1, dloggamma, dloglambda = logp_gradient(X_train[batch, :], y_train[batch], w1, b1, loggamma, loglambda, N0)
                grad_theta[i, :] = self.pack_weights(dw1, db1, dloggamma, dloglambda)

            # calculating the kernel matrix
            kxy, dxkxy = self.svgd_kernel(h=-1)
            grad_theta = (np.matmul(kxy, grad_theta) + dxkxy) / self.M  # \Phi(x)

            # adagrad
            if iter == 0:
                historical_grad = historical_grad + np.multiply(grad_theta, grad_theta)
            else:
                historical_grad = auto_corr * historical_grad + (1 - auto_corr) * np.multiply(grad_theta, grad_theta)
            adj_grad = np.divide(grad_theta, fudge_factor + np.sqrt(historical_grad))
            self.theta = self.theta + master_stepsize * adj_grad


            if iter % 100 == 0:
                svgd_rmse, svgd_ll = self.evaluation(X_test, y_test)
                print('iter = ', iter, " rsme: ", svgd_rmse, " logll: ", svgd_ll)

        '''
            Model selection by using a development set
        '''
        X_dev = self.normalization(X_dev)
        for i in range(self.M):
            w1, b1, loggamma, loglambda = self.unpack_weights(self.theta[i, :])
            pred_y_dev = self.nn_predict(X_dev, w1, b1) * self.std_y_train + self.mean_y_train

            # likelihood
            def f_log_lik(loggamma):
                return np.sum(np.log(np.sqrt(np.exp(loggamma)) / np.sqrt(2 * np.pi) * np.exp(
                    -1 * (np.power(pred_y_dev - y_dev, 2) / 2) * np.exp(loggamma))))

            # The higher probability is better
            lik1 = f_log_lik(loggamma)
            # one heuristic setting
            loggamma = -np.log(np.mean(np.power(pred_y_dev - y_dev, 2)))
            lik2 = f_log_lik(loggamma)
            if lik2 > lik1:
                self.theta[i, -2] = loggamma  # update loggamma

    def normalization(self, X, y=None):
        X = (X - np.full(X.shape, self.mean_X_train)) / \
            np.full(X.shape, self.std_X_train)

        if y is not None:
            y = (y - self.mean_y_train) / self.std_y_train
            return (X, y)
        else:
            return X

    '''
        Initialize all particles
    '''

    def init_weights(self, a0, b0):
        w1 = 1.0 / np.sqrt(self.d + 1) * np.random.randn(self.d, self.n_hidden)
        b1 = np.zeros((self.n_hidden,))
        loggamma = np.log(np.random.gamma(a0, b0))
        loglambda = np.log(np.random.gamma(a0, b0))
        return (w1, b1, loggamma, loglambda)

    '''
        Calculate kernel matrix and its gradient: K, \nabla_x k
    '''

    def svgd_kernel(self, h=-1):
        sq_dist = pdist(self.theta)
        pairwise_dists = squareform(sq_dist) ** 2
        if h < 0:  # if h < 0, using median trick
            h = np.median(pairwise_dists)
            h = np.sqrt(0.5 * h / np.log(self.theta.shape[0] + 1))

        # compute the rbf kernel

        Kxy = np.exp(-pairwise_dists / h ** 2 / 2)

        dxkxy = -np.matmul(Kxy, self.theta)
        sumkxy = np.sum(Kxy, axis=1)
        for i in range(self.theta.shape[1]):
            dxkxy[:, i] = dxkxy[:, i] + np.multiply(self.theta[:, i], sumkxy)
        dxkxy = dxkxy / (h ** 2)
        return (Kxy, dxkxy)

    '''
        Pack all parameters in our model
    '''

    def pack_weights(self, w1, b1, loggamma, loglambda):
        params = np.concatenate([w1.flatten(), b1, [loggamma], [loglambda]])
        return params

    '''
        Unpack all parameters in our model
    '''

    def unpack_weights(self, z):
        w = z
        w1 = np.reshape(w[:self.d * self.n_hidden], [self.d, self.n_hidden])
        b1 = w[self.d * self.n_hidden:(self.d + 1) * self.n_hidden]

        # the last two parameters are log variance
        loggamma, loglambda = w[-2], w[-1]

        return (w1, b1,loggamma, loglambda)

    '''
        Evaluating testing rmse and log-likelihood, which is the same as in PBP 
        Input:
            -- X_test: unnormalized testing feature set
            -- y_test: unnormalized testing labels
    '''

    def evaluation(self, X_test, y_test):
        # normalization
        # X_test = self.normalization(X_test) # TODO: uncomment

        # average over the output
        pred_y_test = np.zeros([self.M, len(y_test)])
        prob = np.zeros([self.M, len(y_test)])

        '''
            Since we have M particles, we use a Bayesian view to calculate rmse and log-likelihood!!!
        '''
        for i in range(self.M):
            w1, b1, loggamma, loglambda = self.unpack_weights(self.theta[i, :])
            pred_y_test[i, :] = self.nn_predict(X_test, w1, b1) * self.std_y_train + self.mean_y_train
            prob[i, :] = np.sqrt(np.exp(loggamma)) / np.sqrt(2 * np.pi) * np.exp(
                -1 * (np.power(pred_y_test[i, :] - y_test, 2) / 2) * np.exp(loggamma))

        pred = np.mean(pred_y_test, axis=0)
        # evaluation
        svgd_rmse = np.sqrt(np.mean((pred - y_test) ** 2))
        svgd_ll = np.mean(np.log(np.mean(prob, axis=0)))

        return (svgd_rmse, svgd_ll)


class MultiModalDistribution():
    def __init__(self, m, covs, ps):
        D = np.size(m[0], axis=0)

        for v in m:  # make sure that all vectors have dimension D
            assert (np.shape(v) == (D,))

        for c in covs:  # make sure that all covariance matrices are D x D
            if D == 1:
                assert (np.shape(c) == (1,))
            else:
                assert (np.shape(c) == (D, D))

        assert (np.sum(ps) == 1.)  # Assert that probabilities sum to 1.

        self.means = m
        self.covs = covs
        self.probs = ps
        self.D = D  # dimension of variables

    def sample(self, N):
        cid = np.random.multinomial(1, self.probs, size=N)
        samples = []

        for ids in cid:
            idx = np.argmax(ids)

            m = self.means[idx]
            S = self.covs[idx]
            if self.D == 1:
                samples.append(np.random.normal(m, S))
            else:
                samples.append(np.random.multivariate_normal(m, S))

        return np.asarray(samples)


def generate_data(probs, means, covs, N=200):
    probs = np.asarray(probs)
    dist = MultiModalDistribution(means, covs, probs)
    D = dist.D

    if D == 1:
        # Xs = np.linspace(1, 5, N)
        Xs = np.ones((N,))
    else:
        Xs = np.random.multivariate_normal(np.asarray([0, 0]), np.asarray([[1, 0], [0, 1]]), size=N)

    theta = dist.sample(N)
    ys = np.zeros((N,))

    for i in range(N):
        if D == 1:
            x = Xs[i]
            z = theta[i]
        else:
            x = Xs[i, :]
            z = theta[i, :]
        ys[i] = np.dot(x, z)

    return Xs, ys, theta


if __name__ == '__main__':
    print('Theano', theano.version.version)  # our implementation is based on theano 0.8.2
    np.random.seed(1)

    # Please make sure that the last column is the label and the other columns are features
    probs = [0.5, 0.5]
    means = [np.asarray([1]), np.asarray([2])]
    covs = [np.asarray([0.1]), np.asarray([0.1])]
    X_input, y_input, z = generate_data(probs, means, covs, N=200)

    ''' build the training and testing data set'''
    train_ratio = 0.9  # We create the train and test sets with 90% and 10% of the data
    permutation = np.arange(X_input.shape[0])
    random.shuffle(permutation)

    size_train = int(np.round(X_input.shape[0] * train_ratio))
    index_train = permutation[0: size_train]
    index_test = permutation[size_train:]

    D = 1
    if D == 1:
        X_train = np.take(X_input, index_train)
        X_train = np.expand_dims(X_train, axis=1)
    else:
        X_train = np.take(X_input, index_train, axis=0)
    y_train = np.take(y_input, index_train)
    # y_train = np.expand_dims(y_train, axis=1)

    if D == 1:
        X_test = np.take(X_input, index_test)
        X_test = np.expand_dims(X_test, axis=1)
    else:
        X_test = np.take(X_input, index_test, axis=0)
    y_test = np.take(y_input, index_test)
    # y_test = np.expand_dims(y_test, axis=1)


    start = time.time()
    ''' Training Bayesian neural network with SVGD '''
    batch_size, n_hidden, max_iter = 32, 1, 5000  # max_iter is a trade-off between running time and performance
    svgd = svgd_bayesnn(X_train, y_train, X_test, y_test, batch_size=batch_size, n_hidden=n_hidden, max_iter=max_iter)
    svgd_time = time.time() - start
    svgd_rmse, svgd_ll = svgd.evaluation(X_test, y_test)
    print('Final SVGD', svgd_rmse, svgd_ll, svgd_time)
