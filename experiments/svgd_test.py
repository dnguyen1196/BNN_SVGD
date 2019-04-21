import numpy as np
from BNN_SVGD.SVGD_BNN import SVGD_simple
import torch
import matplotlib.pyplot as plt
from utils.MiniBatch import CyclicMiniBatch, MiniBatch
from matplotlib.animation import FuncAnimation
from utils.probability import estimate_jensen_shannon_divergence_from_numerical_distribution

plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

np.random.seed(42)

# Generate data
N = 10
eps = 1
Xs = np.linspace(-3, 3, N)
ys = 1 * Xs + np.random.normal(0, 1, size=(N,))

Xs = np.expand_dims(Xs, axis=1)
ys = np.expand_dims(ys, axis=1)

# Initialize the model
x_dim = 1
y_dim = 1
num_networks = 40
network_structure = []

l = 1
p = 1
rbf = 0.1

# Fit
batch_size = 10
train_loader = MiniBatch(xs=Xs, ys=ys, batch_size=batch_size)

num_epochs = 50
step_size  = 0.01

X = torch.FloatTensor(Xs)
y = torch.FloatTensor(ys)

n_retries = 5

h = 0.02

for num_networks in [1]:
    jsd_array = []
    num_epochs = max(50, num_networks * 2)
    # Do some sort of average jsd
    for i in range(n_retries):
        positions_over_time = []

        model = SVGD_simple(x_dim, y_dim, num_networks, network_structure, l, p, rbf)

        for epoch in range(num_epochs):
            for X_train, y_train in train_loader:
                data = torch.FloatTensor(X_train)
                target = torch.FloatTensor(y_train)
                model.step_svgd(data, target, step_size)

                curr_position = []
                for nnid in range(len(model.nns)):
                    weight1 = model.nns[nnid].nn_params[0].weight.detach().numpy()[0]
                    weight2 = model.nns[nnid].nn_params[1].weight.detach().numpy()[0]

                    curr_position.append([weight1[0], weight2[0]])
                positions_over_time.append((curr_position, True))

            y_pred = model.predict_average(X)
            mse    = torch.mean((torch.squeeze(y_pred) - torch.squeeze(y))**2)

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

        jsd = estimate_jensen_shannon_divergence_from_numerical_distribution(np.array(positions_over_time[-1][0]),x_N=Xs, y_N=ys,h=h,plot=True)
        jsd = np.around(jsd, decimals=4)

        jsd_array.append(jsd)

    print("With C = {} neural networks, N = {}, rbf = {}, h = {}, average jsd = {}, std-jsd = {}".format\
              (num_networks, N, rbf, h, np.mean(jsd_array), np.std(jsd_array)))

    # Producing gifs that track the particles' motion
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    anim = FuncAnimation(fig, update, frames=np.arange(0, len(positions_over_time)), interval=100)
    anim.save('SVGD_N={}_C={}_rbf={}_particles_jsd={}_h={}.gif'.format\
                  (N, num_networks, rbf, np.mean(jsd_array), h), dpi=80, writer='imagemagick')
    # plt.show()
