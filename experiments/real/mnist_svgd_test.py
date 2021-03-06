from __future__ import print_function


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
num_epochs   = 200

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



num_networks = 1
model = CovNet_SVGD(image_set="MNIST", num_networks=num_networks)
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
print("Init: evaluate_test_set-avg-loss = {}, evaluate_test_set-accuracy = {}"\
        .format(avg_loss_test, accuracy_test))

for epoch in range(num_epochs):
    avg_loss_train, accuracy_train = train_model(model, optimizer, train_loader, device)
    avg_loss_test, accuracy_test = evaluate_test_set(model, test_loader, device)
    print("epoch {}, train_model-avg-loss = {}, train_model-accuracy = {}, evaluate_test_set-avg-loss = {}, evaluate_test_set-accuracy = {}"\
        .format(epoch, avg_loss_train, accuracy_train, avg_loss_test, accuracy_test))
