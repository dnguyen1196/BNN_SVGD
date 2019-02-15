from __future__ import absolute_import
import numpy as np
import sys
import os
from BNN_SVGD.SVGD_BNN import *
import torch.optim as optim
import torch
import matplotlib.pyplot as plt


"""
Generate synthetic data
"""
N = 100
eps = 1
Xs = np.linspace(-3, 3, N)
ys = 1 * Xs + np.random.normal(0, 1, size=(N,))


"""
Iterator class to do mini-batching
"""

class MiniBatch:
    def __init__(self, xs, ys, batch_size):
        self.Xs = xs
        self.ys = ys
        self.batch_size = batch_size
        self.it = 0
        self.L = np.size(ys, axis=0)

    def __iter__(self):
        return self

    def __next__(self):
        if self.it >= self.L:
            self.it = 0
            raise StopIteration
        else:
            res = None
            if self.it + self.batch_size < len(self.ys):
                res = self.Xs[self.it: self.L, :], self.ys[self.it: self.L]
            else:
                res = self.Xs[self.it: self.it + self.batch_size, :], self.ys[self.it: self.it + self.batch_size]
            self.it += self.batch_size
            return res


# Data preparation to create test/train set
N = np.size(Xs, axis=0)
train_ratio = 0.8
N_train = int(N * train_ratio)
D = 1

indices = np.arange(N)
np.random.shuffle(indices)
train_indices = indices[:N_train]
test_indices = indices[N_train:]

if D == 1:
    Xs_train = np.take(Xs, train_indices)
    Xs_train = np.expand_dims(Xs_train, axis=1)
else:
    Xs_train = np.take(Xs, train_indices, axis=0)
ys_train = np.take(ys, train_indices)

if D == 1:
    Xs_test = np.take(Xs, test_indices)
    Xs_test = np.expand_dims(Xs_test, axis=1)
else:
    Xs_test = np.take(Xs, test_indices, axis=0)
ys_test = np.take(ys, test_indices)

# Initialize Mini-batch
batch_size = 100
loader = MiniBatch(Xs_train, ys_train, batch_size)
test_loader = MiniBatch(Xs_test, ys_test, batch_size)


# train and test functions
def train(epoch, model, optimizer):
    total_loss = 0.
    data = torch.FloatTensor(Xs_train)
    target = torch.FloatTensor(ys_train)
    optimizer.zero_grad()
    loss = model.loss(data, target)
    # print(loss)
    total_loss += loss
    loss.backward()
    optimizer.step()


def test(epoch, model):
    total_loss = 0.
    error = 0.
    num_pt = 0
    with torch.no_grad():
        data = torch.FloatTensor(Xs_train)
        target = torch.FloatTensor(ys_train)

        loss = model.loss(data, target)
        total_loss += loss

        preds = model.predict_average(data)
        error += torch.sum((preds - torch.squeeze(target)) ** 2)
        num_pt += target.shape[0]

    error = error / num_pt
    print("Epoch {} => SVGD loss = {}, rmse = {}".format(epoch, total_loss, torch.sqrt(error)))


"""

Initialize the model and the optimizer

Model
y ~ N(a b x, eps)
a ~ N(0,1)
b ~ N(0,1)

Then the posterior p(a, b|X, y) will be multimodal

"""

num_networks = 20
x_dim = 1
y_dim = 1
network_structure = []  # Hidden layer


n_epochs = 500

# Prints the weights of the neural networks
ll_sigmas = [0.01] #, 0.1, 1, 2]
p_sigmas = [0.01] #, 0.1, 1, 2]
rbf_sigmas = [0.01] # , 0.1, 1, 2]

l = 1
p = 1
rbf = 1

model = FC_SVGD(x_dim, y_dim, num_networks, network_structure, l, p, rbf)
optimizer = optim.Adagrad(model.parameters(), lr=1)

for epoch in range(n_epochs):
    train(epoch, model, optimizer)
    if epoch % 100 == 0:
        test(epoch, model)

for nnid in range(num_networks):
    weight1 = model.nns[nnid].nn_params[0].weight.detach().numpy()[0]
    weight2 = model.nns[nnid].nn_params[1].weight.detach().numpy()[0]

    print(weight1, weight2)

# Try all combinations
# for l in ll_sigmas:
#     for p in p_sigmas:
#         for rbf in rbf_sigmas:
#             model = FC_SVGD(x_dim, y_dim, num_networks, network_structure, l, p, rbf)
#             optimizer = optim.Adagrad(model.parameters(), lr=1)
#
#             for epoch in range(n_epochs):
#                 train(epoch, model, optimizer)
#                 if epoch % 100 == 0:
#                     test(epoch, model)
#
#             weights_array = []
#             for nnid in range(num_networks):
#                 weight1 = model.nns[nnid].nn_params[0].weight.detach().numpy()[0]
#                 weight2 = model.nns[nnid].nn_params[1].weight.detach().numpy()[0]
#
#                 weights_array.append([weight1[0], weight2[0]])
#
#             weights_array = np.asarray(weights_array)
#
#             title = "ll_sigma={},prior_sigma={},rbf_sigma={}.png".format(l, p, rbf)
#
#             filename = "experiments/synthetic/SVGD/" + title
#             plt.scatter(weights_array[:, 0], weights_array[:, 1])
#             plt.ylim((-15, 15))
#             plt.xlim((-15, 15))
#             plt.savefig(filename)
#             plt.close()