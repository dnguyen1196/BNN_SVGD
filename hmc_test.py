import numpy as np
from BNN_SVGD.HMC_BNN import HMC_BNN
import torch
import matplotlib.pyplot as plt
from utils.MiniBatch import MiniBatch, CyclicMiniBatch
from matplotlib.animation import FuncAnimation

# np.random.seed(42)
# torch.manual_seed(42)

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

l = 1 #0.3162
p = 1
rbf = 1

# Fit
batch_size = 100
train_loader = CyclicMiniBatch(xs=Xs, ys=ys, batch_size=batch_size)

model = HMC_BNN(x_dim, y_dim, num_networks, network_structure, l, p, rbf)
sampled_bnn, energies_list = model.fit(train_loader=train_loader, num_iterations=100)

def plot_energies(energies_list):
    energies_numpy = [energy.detach().numpy() for energy in energies_list]
    plt.plot(energies_numpy)

def plot_weight_distribution(sampled_bnn):
    distribution = []
    mse_arr = []

    for bnn in sampled_bnn:
        yhat = bnn.forward(data)
        mse  = torch.mean((torch.squeeze(yhat) - torch.squeeze(target))** 2).detach().numpy()
        mse_arr.append(mse)
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

def track_position_over_time(sampled_bnn):
    # Initialize the figure
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)

    def update(i):
        plt.clf()
        label = 'Sample num {0}'.format(i)
        # Update the line and the axes (with a new xlabel). Return a tuple of
        # "artists" that have to be redrawn for this frame.
        bnn = sampled_bnn[i]

        w1 = bnn.nn_params[0].weight.detach().numpy()[0][0]
        w2 = bnn.nn_params[1].weight.detach().numpy()[0][0]
        plt.scatter(w1, w2)
        plt.ylim(-10, 10)
        plt.xlim(-10, 10)
        plt.xlabel(label)

    # FuncAnimation will call the 'update' function for each frame; here
    # animating over 10 frames, with an interval of 200ms between frames.
    anim = FuncAnimation(fig, update, frames=np.arange(0, len(sampled_bnn)), interval=500)
    # anim.save('position-over-time-svgd.gif', dpi=80, writer='imagemagick')
    plt.show()

plot_energies(energies_list)

track_position_over_time(sampled_bnn)

plot_weight_distribution(sampled_bnn)