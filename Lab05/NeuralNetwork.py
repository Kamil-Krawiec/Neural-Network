import torch.nn as nn


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size):
        super(NeuralNetwork, self).__init__()
        layers = []
        prev_layer_size = input_size
        for i in range(len(hidden_layers)):
            layers.append(nn.Linear(prev_layer_size, hidden_layers[i]))
            layers.append(nn.ReLU())
            prev_layer_size = hidden_layers[i]

        layers.append(nn.Linear(prev_layer_size, output_size))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)

