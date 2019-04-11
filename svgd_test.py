from __future__ import print_function

import dill as pickle
import torch.optim as optim
from BNN_SVGD.SVGD_BNN import CovNet_SVGD
import os
import torch
import argparse

from torchvision import datasets, transforms

parser = argparse.ArgumentParser(description="Stochastic SVGD test")

parser.add_argument('--batch_size', type=int, help='Batch size', default=128)
parser.add_argument('--num_epochs', type=int, help='Number of epochs', default=50)
parser.add_argument('--num_nns', type=int, help='Number of neural networks', default=20)
parser.add_argument('--dataset', type=str, help='Dataset', choices=['MNIST', 'CIFAR-10'], default='CIFAR-10')
parser.add_argument('--outdir', type=str, help='Output directory', default='./')

args = parser.parse_args()

device = 'cpu'
best_acc = 0  # best evaluate_test_set accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
batch_size = args.batch_size
num_classes = 10
num_epochs   = args.num_epochs
num_networks = args.num_nns
dataset = args.dataset
outdir = args.outdir

# Data
print('==> Preparing data..')

if dataset == 'CIFAR-10':
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=False)

else:
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




model = CovNet_SVGD(image_set=dataset, num_networks=num_networks)

optimizer = optim.SGD(model.parameters(), lr=0.001)

count = 1

def train_model(model, optimizer, train_loader, device):
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


def evaluate_test_set(model, train_loader, device):
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

    return total_loss/num_batches, correct/total


avg_loss_test, accuracy_test = evaluate_test_set(model, test_loader, device)
print("Init: test set avg batch loss = {}, test set accuracy = {}".format(avg_loss_test, accuracy_test))

for epoch in range(num_epochs):
    avg_loss_train, accuracy_train = train_model(model, optimizer, train_loader, device)
    avg_loss_test, accuracy_test = evaluate_test_set(model, test_loader, device)
    print("epoch {}, train_model-avg-loss = {}, train_model-accuracy = {}, evaluate_test_set-avg-loss = {}, evaluate_test_set-accuracy = {}"\
        .format(epoch, avg_loss_train, accuracy_train, avg_loss_test, accuracy_test))


model_file = "data_{}-epochs_{}-numnns_{}-model.pkl".format(dataset, num_epochs, num_networks)
save_dir   = os.path.join(outdir, model_file)

with open(save_dir, "wb") as f:
    pickle.dump(model, f)