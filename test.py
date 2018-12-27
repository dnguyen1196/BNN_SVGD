from BNN_SVGD.BNN import BNN_SVGD
from torch.autograd import Variable
import numpy as np
import torch
from torchvision import datasets, transforms


x_train_N = torch.FloatTensor([[-3], [-2], [-1], [1], [2], [3]])
y_train_N = torch.FloatTensor([[0], [-1], [3], [3], [-1], [0]])

# svgd_bnn = BNN_SVGD(x_dim=1, y_dim=1, num_networks=4, network_structure=[32, 32])
# svgd_bnn.optimize(x_train_N, y_train_N)


train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)

# sgd = torch.optim.SGD
# sgd.step()
G = 20
x_test_N = np.linspace(-3, 3, G)