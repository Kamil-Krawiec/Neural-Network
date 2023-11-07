import torch
from torch import nn

class TorchMLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(TorchMLPClassifier, self).__init__()
        self.input_size = input_size
        self.hidden_layers = nn.ModuleList()
        for hidden_size in hidden_sizes:
            self.hidden_layers.append(nn.Linear(input_size, hidden_size))
            self.hidden_layers.append(nn.ReLU())  # Add ReLU activation between layers
            input_size = hidden_size
        self.output = nn.Linear(input_size, output_size)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output(x)
        return x
