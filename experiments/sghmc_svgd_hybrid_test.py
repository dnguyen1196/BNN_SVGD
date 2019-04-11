import numpy as np
from BNN_SVGD.SVGD_HMC_hybrid import SVGD_SGHMC_hybrid
import torch
import matplotlib.pyplot as plt
from utils.MiniBatch import MiniBatch, CyclicMiniBatch
from matplotlib.animation import FuncAnimation
from utils.probability import estimate_kl_divergence_discrete_true_posterior

plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

np.random.seed(42)

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
batch_size = 100
train_loader = CyclicMiniBatch(xs=Xs, ys=ys, batch_size=batch_size)

model = SVGD_SGHMC_hybrid(x_dim, y_dim, num_networks, network_structure, l, p, rbf)
positions_over_time = model.fit(train_loader=train_loader, num_iterations=100, svgd_iteration=200, hmc_iteration=10)


# Initialize the figure
fig, ax = plt.subplots()
fig.set_tight_layout(True)

def update(i):
    plt.clf()
    pos, svgd = positions_over_time[i]

    position = np.array(pos)
    xpos = position[:, 0]
    ypos = position[:, 1]

    label = 'Epoch {0}'.format(i)
    color = "b" if svgd else "r"
    plt.scatter(xpos, ypos, c=color)

    plt.ylim(-15, 15)
    plt.xlim(-15, 15)
    plt.xlabel(label)

anim = FuncAnimation(fig, update, frames=np.arange(0, len(positions_over_time)), interval=100)
anim.save('position-over-time-hybrid.gif', dpi=80, writer='imagemagick')
plt.show()



# Get the final particle positions and estimate KL(true posterior | KDE)
particle_positions = []
for nnid in range(len(model.nns)):
    weight1 = model.nns[nnid].nn_params[0].weight.detach().numpy()[0]
    weight2 = model.nns[nnid].nn_params[1].weight.detach().numpy()[0]
    particle_positions.append([weight1[0], weight2[0]])
particle_positions = np.array(particle_positions)

kl = estimate_kl_divergence_discrete_true_posterior(particle_positions)

print("KL(estimated true posterior | KDE(hybrid)) = ", kl)