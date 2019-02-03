from __future__ import absolute_import
import numpy as np
import sys
import os
from BNN_SVGD.BNN import *
import torch.optim as optim
import torch

"""
BernoulliGaussian distribution class
e.g
>>> dist = BernoulliGaussian(means, covs, class_probabilities)
>>> samples = dist.sample(N)
"""
class BernoulliGaussian():
	def __init__(self, m, covs, ps):
		D = np.size(m[0], axis=0)

		for v in m: # make sure that all vectors have dimension D
			assert(np.shape(v) == (D,))

		for c in covs: # make sure that all covariance matrices are D x D
			if D == 1:
				assert(np.shape(c) == (1,))
			else:
				assert(np.shape(c) == (D, D))

		assert(np.sum(ps) == 1.) # Assert that probabilities sum to 1.

		self.means = m
		self.covs = covs
		self.probs = ps
		self.D = D # dimension of variables

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


"""
Generate synthetic data
"""
def generate_data(probs, means, covs, N = 200):
	probs = np.asarray(probs)
	dist = BernoulliGaussian(means, covs, probs)
	D = dist.D

	if D == 1:
		# Xs = np.linspace(1, 5, N)
		Xs = np.ones((N,))
	else:
		Xs = np.random.multivariate_normal(np.asarray([0,0]), np.asarray([[1,0],[0,1]]), size=N)
	
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
	    		res = self.Xs[self.it : self.L, :], self.ys[self.it : self.L]
	    	else:
	    		res = self.Xs[self.it : self.it + self.batch_size, :],  self.ys[self.it : self.it + self.batch_size]
	    	self.it += self.batch_size
	    	return res



# First generate data according to Bernoulli-Gaussian distribution
probs = [0.5, 0.5]
means = [np.asarray([1]), np.asarray([2])]
covs  = [np.asarray([0.1]), np.asarray([0.1])]
Xs, ys, zs = generate_data(probs, means, covs, N=200)


#Initialize the model and the optimizer
num_networks = 20
x_dim = 1
y_dim = 1
network_structure = [] # Hidden layer
model = FC_SVGD(x_dim, y_dim, num_networks, network_structure)
optimizer = optim.Adagrad(model.parameters(), lr=1)


# Data preparation to create test/train set
N = np.size(Xs, axis=0)
train_ratio = 0.8
N_train = int(N * train_ratio)
D = x_dim

indices = np.arange(N)
np.random.shuffle(indices)
train_indices = indices[:N_train]
test_indices  = indices[N_train:]

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
batch_size = 32
loader = MiniBatch(Xs_train, ys_train, batch_size)
test_loader = MiniBatch(Xs_test, ys_test, batch_size)


# train and test functions
def train(epoch, model, optimizer, loader):
	total_loss = 0.
	for batch, (x_batch, y_batch) in enumerate(loader):
		data = torch.FloatTensor(x_batch)
		target = torch.FloatTensor(y_batch)

		loss = model.loss(data, target)
		# print(loss)
		total_loss += loss
		loss.backward()
		optimizer.step()

def test(epoch, model, loader):
	total_loss = 0.
	error = 0.
	num_pt = 0
	with torch.no_grad():
		for batch, (x_batch, y_batch) in enumerate(loader):
			data = torch.FloatTensor(x_batch)
			target = torch.FloatTensor(y_batch)
			
			loss = model.loss(data, target)
			total_loss += loss

			preds = model.predict_average(data)
			error += torch.sum((preds - target)**2)
			num_pt += target.shape[0]

	error = error/ num_pt
	print("Epoch {} => SVGD loss = {}, rmse = {}".format(epoch, total_loss/(batch+1), torch.sqrt(error)))

n_epochs = 1000
test(-1, model, test_loader)
for epoch in range(n_epochs):
	train(epoch, model, optimizer, loader)
	if epoch % 100 == 0:
		test(epoch, model, test_loader)

# Prints the weights of the neural networks
for nnid in range(num_networks):
	weight = model.nns[nnid].nn_params[0].weight.detach().numpy()[0]
	bias = ""
	if model.bias:
		bias   = model.nns[nnid].nn_params[0].bias.detach().numpy()[0]
	print(weight, "$", bias)

