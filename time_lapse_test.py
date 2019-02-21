from __future__ import absolute_import
import numpy as np
import sys
import os
from BNN_SVGD.SVGD_BNN import *
import torch.optim as optim
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# np.random.seed(42)
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
        error += torch.sum((torch.squeeze(preds) - torch.squeeze(target)) ** 2)
        num_pt += target.shape[0]

    error = error / num_pt
    print("Epoch {} => SVGD loss = {}, mse = {}".format(epoch, total_loss, error))


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

#
l = .1
p = .1
rbf = .1

model = FC_SVGD(x_dim, y_dim, num_networks, network_structure, l, p, rbf)
optimizer = optim.Adagrad(model.parameters(), lr=1)

#
capture_frame       = [0, 1,2,5,10, 50, 100, 200, 350, 500]
positions_over_time = []

for epoch in range(n_epochs+1):
	train(epoch, model, optimizer)
	# Keep track of the weights after each epoch

	if epoch in capture_frame:
		position = []

		for nnid in range(num_networks):
			weight1 = model.nns[nnid].nn_params[0].weight.detach().numpy()[0]
			weight2 = model.nns[nnid].nn_params[1].weight.detach().numpy()[0]
			position.append([weight1[0], weight2[0]])

		positions_over_time.append(position)

	if epoch % 100 == 0:
		test(epoch, model)

positions_over_time = np.array(positions_over_time)

# Initialize the figure
fig, ax = plt.subplots()
fig.set_tight_layout(True)

def update(i):
	plt.clf()
	label = 'Epoch {0}'.format(capture_frame[i])
	# Update the line and the axes (with a new xlabel). Return a tuple of
	# "artists" that have to be redrawn for this frame.
	position = positions_over_time[i]
	plt.scatter(position[:, 0], position[:, 1])
	plt.ylim(-15, 15)
	plt.xlim(-15, 15)

	plt.xlabel(label)
	# return ax


# FuncAnimation will call the 'update' function for each frame; here
# animating over 10 frames, with an interval of 200ms between frames.
anim = FuncAnimation(fig, update, frames=np.arange(0, len(capture_frame)), interval=500)
anim.save('position-over-time-svgd.gif', dpi=80, writer='imagemagick')
plt.show()