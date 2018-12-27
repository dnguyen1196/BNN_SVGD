from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import os
import argparse

from BNN_SVGD.BNN import *
from torch.autograd import Variable
import numpy as np
import torch

from torchvision import datasets, transforms


device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=128, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=128, shuffle=False)



def train(epoch, optimizer, model, train_loader, device):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)


        optimizer.zero_grad()
        loss = model.loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


        break


model = CovNet_SVGD(image_set="MNIST", num_networks=1)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, nesterov=False)


# for batch_idx, (data, target) in enumerate(train_loader):
#     X, y = data.to(device), target.to(device)
#     preds = model.predict(X)

#     print(preds)
#     break

# for epoch in range(start_epoch, start_epoch+200):
#     train(epoch, optimizer, model, train_loader, device)

#     break
