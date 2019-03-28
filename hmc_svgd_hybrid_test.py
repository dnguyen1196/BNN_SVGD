import numpy as np
from BNN_SVGD.SVGD_HMC_hybrid import SVGD_HMC_hybrid
import torch
import matplotlib.pyplot as plt
from utils.MiniBatch import MiniBatch, CyclicMiniBatch
from matplotlib.animation import FuncAnimation

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
batch_size = 10
train_loader = CyclicMiniBatch(xs=Xs, ys=ys, batch_size=batch_size)

model = SVGD_HMC_hybrid(x_dim, y_dim, num_networks, network_structure, l, p, rbf)
positions_over_time = model.fit(train_loader=train_loader, num_iterations=200, svgd_iteration=30, hmc_iteration=200)

grad_norms = model.avg_grad_norms

iterations = [grad[1] for grad in grad_norms]
avg_norms  = [grad[0] for grad in grad_norms]

pieces = []
prev = 0
cur_piece = []

for i, iter_num in enumerate(iterations):
    if iter_num == prev + 1:
        cur_piece.append((iter_num, avg_norms[i]))
    else:
        pieces.append(cur_piece)
        cur_piece = []
    pieces.append(cur_piece)
    prev = iter_num

for i, piece in enumerate(pieces):
    plt.plot([x[0] for x in piece], [x[1] for x in piece])

plt.show()

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