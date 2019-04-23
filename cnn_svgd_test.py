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
parser.add_argument('--num_nns', type=int, help='Number of neural networks', default=10)
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

print("SVGD for CNN, dataset %s, number of epochs %d, number of neural networks %d" %(dataset, num_epochs, num_networks))

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

count = 1

def train_model(epoch, model, train_loader, device):
    total_loss = 0.
    num_batches = 0
    count = 0
    correct = 0
    total = 0

    lr0  = 1
    epochs_drop = 1
    drop = 0.9

    lr = lr0 * drop**int(epoch / epochs_drop)

    # lr = max(lr, 0.009)

    print("Epoch %d, lr = %.4f" %(epoch, lr))

    for batch_idx, (data, target) in enumerate(train_loader):
        # print("batch_idx", batch_idx)
        X, y = data.to(device), target.to(device)

        model.step_svgd(X, y, step_size=lr)

        num_batches += 1

        # Do prediction
        y_onehot = torch.FloatTensor(y.shape[0], num_classes)
        y_onehot.zero_()
        y_onehot.scatter_(1, y.view(-1,1), 1.)

        outputs = model.predict_average(X)

        pred = outputs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += y.size(0)

        count += 1

    return correct/total


def evaluate_test_set(epoch, model, train_loader, device):
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

            num_batches += 1

            outputs = model.predict_average(X)
            pred = outputs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += y.size(0)

            count += 1

    return correct/total

accuracy_test = evaluate_test_set(-1, model, test_loader, device)
print("Init: test set accuracy = {}".format(accuracy_test))

for epoch in range(num_epochs):
    accuracy_train = train_model(epoch, model, train_loader, device)
    accuracy_test = evaluate_test_set(epoch, model, test_loader, device)
    print("epoch {}, train_model-accuracy = {}, evaluate_test_set-accuracy = {}"\
        .format(epoch, accuracy_train, accuracy_test))


model_file = "data_{}-epochs_{}-numnns_{}-model.pkl".format(dataset, num_epochs, num_networks)
save_dir   = os.path.join(outdir, model_file)

with open(save_dir, "wb") as f:
    pickle.dump(model, f)
