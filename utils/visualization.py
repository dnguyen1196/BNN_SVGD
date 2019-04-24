import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.stats
from matplotlib.animation import FuncAnimation
import seaborn as sns
from matplotlib import colors, ticker

def plot_weight_distribution_hmc(sampled_bnn, filename=None, show=True):
    distribution = []

    for bnn in sampled_bnn:
        weight1 = bnn.nn_params[0].weight.detach().numpy()[0][0]
        weight2 = bnn.nn_params[1].weight.detach().numpy()[0][0]
        distribution.append([weight1, weight2])

    distribution = np.array(distribution)

    # Plot the "correct" distribution
    refx_neg = np.linspace(-10, -0.1, 19)
    refx_pos = np.linspace(0.1, 10, 19)
    refy_neg = 1. / refx_neg
    refy_pos = 1. / refx_pos

    plt.plot(refx_neg, refy_neg, "r")
    plt.plot(refx_pos, refy_pos, "r")
    plt.ylim(-4, 4)
    plt.xlim(-4, 4)
    plt.scatter(distribution[:, 0], distribution[:, 1])

    if filename:
        plt.savefig(filename)
    if show:
        plt.show()

    plt.close()

    return distribution


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
        plt.ylim(-4, 4)
        plt.xlim(-4, 4)
        plt.xlabel(label)

    # FuncAnimation will call the 'update' function for each frame; here
    # animating over 10 frames, with an interval of 200ms between frames.
    anim = FuncAnimation(fig, update, frames=np.arange(0, len(sampled_bnn)), interval=100)
    anim.save('./position-over-time-svgd.gif', dpi=80, writer='imagemagick')
    plt.show()


def generate_observed_data(N=5, true_sigma_y=0.333):

    ## Assume given a regular grid of 1D x values between -1 and 1
    x_N = np.linspace(-1, 1, N)

    ## Draw real-valued labels y from a Normal with mean x and variance sigma_lik
    y_N = np.random.normal(x_N, true_sigma_y)

    return x_N, y_N


import os
import sys
import time
import math

import torch.nn as nn
import torch.nn.init as init


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


