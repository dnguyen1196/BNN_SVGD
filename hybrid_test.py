import numpy as np
import sys
import os
from BNN_SVGD.SVGD_HMC_hybrid import *
import torch.optim as optim
import torch
import matplotlib.pyplot as plt


# Generate data
N = 100
eps = 1
Xs = np.linspace(-3, 3, N)
ys = 1 * Xs + np.random.normal(0, 1, size=(N,))

Xs = np.expand_dims(Xs, axis=1)
ys = np.expand_dims(ys, axis=1)

data = torch.FloatTensor(Xs)
target = torch.FloatTensor(ys)

# Initialize the model
x_dim = 1
y_dim = 1
num_networks = 20
network_structure = []

l = 1
p = 1
rbf = 1

# Fit
model = SVGD_HMC_hybrid(x_dim, y_dim, num_networks, network_structure, l, p, rbf)
model.fit(data, target, num_iterations=500, svgd_iteration=600, hmc_iteration=1)


for nnid in range(num_networks):
    weight1 = model.nns[nnid].nn_params[0].weight.detach().numpy()[0]
    weight2 = model.nns[nnid].nn_params[1].weight.detach().numpy()[0]

    print(weight1, weight2)