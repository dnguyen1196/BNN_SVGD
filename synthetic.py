from __future__ import absolute_import
import numpy as np
import sys
import os
from BNN_SVGD.BNN import *
import torch.optim as optim
import torch


class MultiModalDistribution():
	def __init__(self, m, covs, ps):
		D = np.size(m[0], axis=0)

		for v in m: # make sure that all vectors have dimension D
			assert(np.shape(v) == (D,))

		for c in covs: # make sure that all covariance matrices are D x D
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
			samples.append(np.random.multivariate_normal(m, S))

		return np.asarray(samples)


def generate_data():
	means = [np.asarray([0,0]), np.asarray([1,1])]
	covs  = [np.asarray([[.1,0],[0,.1]]), np.asarray([[.1,0],[0,.1]])]
	probs = np.asarray([0.8, 0.2])
	dist = MultiModalDistribution(means, covs, probs)

	N = 100

	Xs = np.random.multivariate_normal(np.asarray([0,0]), np.asarray([[1,0],[0,1]]), size=N)
	theta = dist.sample(N)

	ys = np.zeros((N,))

	for i in range(N):
		x = Xs[i, :]
		z = theta[i, :]
		ys[i] = np.dot(x, z)

	return Xs, ys, theta
	

class SyntheticLoader:
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


Xs, ys, zs = generate_data()

N = np.size(Xs, axis=0)
train_ratio = 0.8

N_train = int(N * train_ratio)

Xs_train = Xs[:N_train, :]
ys_train = ys[:N_train]

Xs_test = Xs[N_train : , :]
ys_test = ys[N_train : ]


"""
Initialize the model and the optimizer
"""
num_networks = 2
x_dim = 2
y_dim = 1
model = FC_SVGD(2, 1, num_networks, network_structure=[16])

# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adagrad(model.parameters(), lr=0.001)

batch_size = 10
loader = SyntheticLoader(Xs_train, ys_train, batch_size)
test_loader = SyntheticLoader(Xs_test, ys_test, batch_size)

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

	# print ("epoch {} => avg-loss = {}".format(epoch, total_loss/batch))


def test(epoch, model, loader):
	total_loss = 0.
	error = 0.
	with torch.no_grad():
		for batch, (x_batch, y_batch) in enumerate(loader):
			data = torch.FloatTensor(x_batch)
			target = torch.FloatTensor(y_batch)
			
			loss = model.loss(data, target)
			total_loss += loss

			preds = model.predict_average(data)
			error += torch.sum((preds - target)**2)/ target.shape[0]

	print("Epoch {} => loss = {}, rsme = {}".format(epoch, total_loss/batch, torch.sqrt(error)))

n_epochs = 100
for epoch in range(n_epochs):
	train(epoch, model, optimizer, loader)
	if epoch % 10 == 0:
		test(epoch, model, test_loader)


