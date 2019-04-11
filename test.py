from __future__ import print_function

import dill as pickle
import torch.optim as optim
from BNN_SVGD.SVGD_BNN import CovNet_SVGD
import torch

from torchvision import datasets, transforms


# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
best_acc = 0  # best evaluate_test_set accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
batch_size = 128
num_classes = 10
log_interval = 100
num_epochs   = 0

# Data
print('==> Preparing data..')

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



model = None
with open("cifar-10-model.pkl", "rb") as f:
    model = pickle.load(f)


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
print("Init: evaluate_test_set-avg-loss = {}, evaluate_test_set-accuracy = {}".format(avg_loss_test, accuracy_test))
