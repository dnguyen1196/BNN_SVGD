import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList
from torch.distributions.normal import Normal
import numpy as np

"""
Simple neural network to do experimentations (not used often)
Refer to single weight neural net below
"""
class SimpleNeuralNet(nn.Module):
    def __init__(self, input_dim, output_dim, structure=[32], bias=False):
        super(SimpleNeuralNet, self).__init__()
        self.bias = bias
        self.n_dims_data = input_dim
        nn_layer_size = (
            [input_dim] + structure + [output_dim]
        )
        self.n_layers = len(nn_layer_size) - 1
        # Create the encoder, layer by layer
        self.activation_funcs = list() # Activation function
        self.nn_params = nn.ModuleList()

        for layer_id, (n_in, n_out) in enumerate(zip(
                nn_layer_size[:-1], nn_layer_size[1:])):

            hidden_layer = nn.Linear(n_in, n_out, bias=bias)
            self.nn_params.append(hidden_layer)

            # Initialize the weights of the neural networks with the same bernoulli-gaussian distribution
            probs = [0.5, 0.5]
            mode = np.argmax(np.random.multinomial(1, probs, size=1))
            if mode == 0:
                dist = Normal(1, 0.1)
            else:
                dist = Normal(2, 0.1)

            w = dist.sample((1,))
            hidden_layer.weight.data.fill_(w[0])

            self.activation_funcs.append(F.relu) # rectified linear activation

        # Last activation function is the identity function
        self.activation_funcs[-1] = lambda a: a

    def forward(self, x):
        result = x
        for ll in range(self.n_layers):
            layer_transform = self.nn_params[ll]
            activation = self.activation_funcs[ll]
            result = activation(layer_transform(result))
        return result

"""
Single weight neural network
- A two layer neural network where each layer is just a scalar weight
without any bias
For experiment purpose
"""
class SingleWeightNeuralNet(nn.Module):
    def __init__(self, input_dim, output_dim, bias=False):
        super(SingleWeightNeuralNet, self).__init__()
        self.bias = bias
        self.n_dims_data = input_dim

        # Create the encoder, layer by layer
        self.activation_funcs = list() # Activation function
        self.nn_params = nn.ModuleList()
        self.n_layers  = 2

        for i in range(self.n_layers): # Just 1 hidden layer
            hidden_layer = nn.Linear(in_features=1, out_features=1, bias=bias)
            self.nn_params.append(hidden_layer)
            self.activation_funcs.append(lambda a: a)


    def forward(self, x):
        result = x
        for ll in range(self.n_layers):
            layer_transform = self.nn_params[ll]
            activation = self.activation_funcs[ll]
            result = activation(layer_transform(result))
        return result


"""
################################################################################# 

                                DEEP NEURAL NETWORKS

#################################################################################
"""



"""
Fully connected neural network
"""
class FullyConnectedNet(nn.Module):
    def __init__(self, input_dim, output_dim, structure=[32], bias=True):
        super(FullyConnectedNet, self).__init__()
        self.bias = bias
        self.n_dims_data = input_dim
        nn_layer_size = (
            [input_dim] + structure + [output_dim]
        )
        self.n_layers = len(nn_layer_size) - 1
        # Create the encoder, layer by layer
        self.activation_funcs = list() # Activation function
        self.nn_params = nn.ModuleList()

        for layer_id, (n_in, n_out) in enumerate(zip(
                nn_layer_size[:-1], nn_layer_size[1:])):

            hidden_layer = nn.Linear(n_in, n_out, bias=bias)
            self.nn_params.append(hidden_layer)
            self.activation_funcs.append(F.relu) # rectified linear activation

        # Last activation function is the identity
        self.activation_funcs[-1] = lambda a: a

    def forward(self, x):
        result = x
        for ll in range(self.n_layers):
            layer_transform = self.nn_params[ll]
            activation = self.activation_funcs[ll]
            result = activation(layer_transform(result))
        return result



"""
LeNET
Note that this architecture is specific to CIFAR-10 datasets
"""
class Cifar10LeNet(nn.Module):
    def __init__(self):
        super(Cifar10LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)
        self.drop_out_p = 0.1

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        # out = F.dropout(out, p=self.drop_out_p)
        out = F.relu(self.fc2(out))
        # out = F.dropout(out, p=self.drop_out_p)
        out = self.fc3(out)
        return out


"""
MnistNet
"""

class MnistCovNet(nn.Module):
    def __init__(self):
        super(MnistCovNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x


"""
Generic convolution neural network
"""
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()

        # Original image is 28 x 28
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            # Use 16 filters, which gives (28 x 28 x 16)
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)) # Max pooling, then we get (14, 14, 16)

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            # This gets us (14, 14, 32)
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)) # This gets us (7,7,32)

        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(7 * 7 * 32, 512)
        self.fc2 = nn.Linear(512, num_classes)


    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1) # Before feeding into the nn layer, squeeze
        # out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

