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
batch_size = 128
num_classes = 10

# Data
print('==> Preparing data..')

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=False)



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

num_networks = 10
model = CovNet_SVGD(image_set="MNIST", num_networks=num_networks)
# optimizer = optim.Adagrad(model.parameters(), lr=1)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)




count = 1

def train(model, optimizer, train_loader, device):
    total_loss = 0.
    num_batches = 0
    count = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        X, y = data.to(device), target.to(device)

        y_onehot = torch.FloatTensor(y.shape[0], num_classes)
        y_onehot.zero_()
        y_onehot.scatter_(1, y.view(-1,1), 1.)

        # print(X.shape)
        # print(y.shape)
        optimizer.zero_grad()

        loss = model.loss(X, y_onehot)
        loss.backward()
        outputs = model.predict_average(X)

        optimizer.step()

        total_loss += loss
        num_batches += 1

        _, predicted = outputs.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()

        count += 1
        # if loss > 10000:
        #     print(loss)

    return total_loss/num_batches, correct/total


def test(model, optimizer, train_loader, device):
    total_loss = 0.
    num_batches = 0
    count = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(train_loader):
            X, y = data.to(device), target.to(device)

            y_onehot = torch.FloatTensor(y.shape[0], num_classes)
            y_onehot.zero_()
            y_onehot.scatter_(1, y.view(-1,1), 1.)

            loss = model.loss(X, y_onehot)
            outputs = model.predict_average(X)

            total_loss += loss
            num_batches += 1

            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

            count += 1
            # if loss > 10000:
            #     print(loss)

    return total_loss/num_batches, correct/total


avg_loss_test, accuracy_test = test(model, optimizer, test_loader, device)
print("Init: test-avg-loss = {}, test-accuracy = {}"\
        .format(avg_loss_test, accuracy_test))

for epoch in range(200):
    avg_loss_train, accuracy_train = train(model, optimizer, train_loader, device)
    # break
    avg_loss_test, accuracy_test = test(model, optimizer, test_loader, device)
    print("epoch {}, train-avg-loss = {}, train-accuracy = {}, test-avg-loss = {}, test-accuracy = {}"\
        .format(epoch, avg_loss_train, accuracy_train, avg_loss_test, accuracy_test))
