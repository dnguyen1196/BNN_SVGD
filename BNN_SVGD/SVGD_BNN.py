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
import lasagne


input_var = T.tensor4('X')
target_var = T.ivector('y')

network = lasagne.layers.InputLayer((), input_var)