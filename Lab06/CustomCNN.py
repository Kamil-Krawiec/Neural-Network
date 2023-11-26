from torch import nn
from torch.nn import LazyLinear


class CustomCNN(nn.Module):
    def __init__(self, out_channels, kernel_size, pool_size):
        super(CustomCNN, self).__init__()

        # Convolutional layer with adjustable parameters
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, out_channels, kernel_size=kernel_size, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=pool_size, stride=2)
        )

        # Flatten and fully connected layers
        self.fc_block = nn.Sequential(
            nn.Flatten(),
            LazyLinear(out_channels * 14 * 14),
            nn.LeakyReLU(),
            LazyLinear(128),
            nn.LeakyReLU(),
            LazyLinear(10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = self.fc_block(x)
        return x
