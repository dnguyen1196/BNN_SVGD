from BNN_SVGD.BNN import BNN_SVGD
from torch.autograd import Variable
import numpy as np
import torch

x_train_N = torch.FloatTensor([[-3], [-2], [-1], [1], [2], [3]])
y_train_N = torch.FloatTensor([[0], [-1], [3], [3], [-1], [0]])

svgd_bnn = BNN_SVGD(x_dim=1, y_dim=1, num_networks=4, network_structure=[32, 32])
svgd_bnn.optimize(x_train_N, y_train_N)


# sgd = torch.optim.SGD
# sgd.step()
G = 20
x_test_N = np.linspace(-3, 3, G)