import numpy as np
from BNN_SVGD.HMC_BNN import HMC_BNN
import torch
import matplotlib.pyplot as plt
from utils.MiniBatch import MiniBatch, CyclicMiniBatch
from matplotlib.animation import FuncAnimation

np.random.seed(42)

# Generate data
N = 100
eps = 1
Xs = np.linspace(-3, 3, N)
ys = 1 * Xs + np.random.normal(0, 0.1, size=(N,))

Xs = np.expand_dims(Xs, axis=1)
ys = np.expand_dims(ys, axis=1)

data = torch.FloatTensor(Xs)
target = torch.FloatTensor(ys)

# Initialize the model
x_dim = 1
y_dim = 1
num_networks = 20
network_structure = []

l = .1
p = .1
rbf = 1

# Fit

train_loader = CyclicMiniBatch(xs=Xs, ys=ys, batch_size=100)

model = HMC_BNN(x_dim, y_dim, num_networks, network_structure, l, p, rbf)
sampled_bnn = model.fit(train_loader=train_loader, num_iterations=500)

distribution = []
mse_arr = []

for bnn in sampled_bnn:
    yhat = bnn.forward(data)
    mse_arr.append(torch.mean((yhat-target)**2).detach().numpy())
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

