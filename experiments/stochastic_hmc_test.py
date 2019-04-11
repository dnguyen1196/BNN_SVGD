import numpy as np
from BNN_SVGD.HMC_BNN import SG_HMC_BNN
import torch
import matplotlib.pyplot as plt
from utils.MiniBatch import CyclicMiniBatch
from matplotlib.animation import FuncAnimation
from utils.visualization import plot_weight_distribution, track_position_over_time
from utils.probability import estimate_kl_divergence_discrete_true_posterior

# Random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

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
batch_size = 10  # Batch size (100 is full batch, otherwise mini-batch)
train_loader = CyclicMiniBatch(xs=Xs, ys=ys, batch_size=batch_size)

model = SG_HMC_BNN(x_dim, y_dim, num_networks, network_structure, l, p, rbf)

sampled_bnn = model.fit(train_loader=train_loader, num_iterations=100, n_leapfrog_steps=20, step_size=0.001, momentum=0.9925)

# track_position_over_time(sampled_bnn)

distribution = plot_weight_distribution(sampled_bnn, data, target)

kl = estimate_kl_divergence_discrete_true_posterior(distribution, Xs, ys)


print("KL(estimated true posterior | KDE(stochastic HMC)) = ", kl)