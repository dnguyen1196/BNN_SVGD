import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList

class BasicNet(nn.Module):
    def __init__(self, input_dim, output_dim, structure=[32]):
        super(BasicNet, self).__init__()
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
            self.nn_params.append(nn.Linear(n_in, n_out))
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

