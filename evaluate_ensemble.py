import dill as pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from BNN_SVGD.SVGD_BNN import CovNet_SVGD

import torchvision
from torchvision import datasets, transforms

import os
import argparse
import time

import numpy as np

#from utils.visualization import progress_bar

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


# Load the ensemble
model_nums = range(1, 11)
ensemble   = []
for num in model_nums:
    filename = "model_num_{}.t7".format(num)
    checkpoint = torch.load(os.path.join("./cifar10-pytorch-checkpoint/",filename ))
    net = LeNet()
    accuraccy =checkpoint['acc']
    net.load_state_dict(checkpoint['net'])
    ensemble.append(net)


testloader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=128, shuffle=True)


def test_ensemble(ensemble):
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            outputs = ensemble[0](inputs) / len(ensemble)

            for i in range(2, len(ensemble)):
                outputs += 1/ len(ensemble) * ensemble[i](inputs)

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    # Save checkpoint.
    acc = 100.*correct/total
    print("ensemble accuracy on test set = {}".format(acc))

test_ensemble(ensemble)


svgd_model_file = "./cifar10-cnn-svgd-checkpoint/data_CIFAR-10-epochs_100-numnns_10-model.pkl"

model = None
with open(svgd_model_file, 'rb') as in_strm:
    model = pickle.load(in_strm)

def evaluate_test_set(model, test_loader):
    num_batches = 0
    count = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            X, y = data, target

            y_onehot = torch.FloatTensor(y.shape[0], 10)
            y_onehot.zero_()
            y_onehot.scatter_(1, y.view(-1,1), 1.)

            num_batches += 1

            outputs = model.predict_average(X)
            pred = outputs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += y.size(0)

            count += 1

    return correct/total


accuraccy = evaluate_test_set(model, testloader)
print("SVGD accuracy =", accuraccy)