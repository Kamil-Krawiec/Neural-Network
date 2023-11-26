import random

import numpy as np
from torch.utils.data import Subset
from torchvision import transforms, datasets
from torchvision.transforms import GaussianBlur

from CustomCNN import CustomCNN
from Functions import show_learning_curve
from Lab06.functions import *


def main():
    default_batch_size = 100

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])

    transform_blur = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        GaussianBlur(kernel_size=9, sigma=(1.5, 5.0))
    ])

    # how to get train and test data
    train_data = datasets.FashionMNIST('path', download=True, train=True, transform=transform)
    test_data = datasets.FashionMNIST('path', download=True, train=False, transform=transform)

    # use only 5% of train data
    train_list = list(range(len(train_data)))
    random.shuffle(train_list)
    train_20_percent = train_list[:len(train_list) // 5]

    test_list = list(range(len(test_data)))
    random.shuffle(test_list)
    test_20_percent = test_list[:len(test_list) // 5]

    train_subset = Subset(train_data, train_20_percent)
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=default_batch_size, shuffle=True)

    test_subset = Subset(test_data, test_20_percent)
    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=default_batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()

    epochs = 50

    out_channels = 16
    kernel_size = 3
    pool_size = 2
    dummy_tensor = torch.rand(1, 1, 28, 28)

    cnn_model = CustomCNN(out_channels, kernel_size, pool_size)
    cnn_model(dummy_tensor)

    cnn_model.apply(init_normal)

    print(cnn_model)
    training_losses, test_losses = train(
        cnn_model,
        criterion,
        train_loader,
        test_loader,
        epochs=epochs)
    evaluate_model(cnn_model, test_loader)

    name = f"ChanN_{out_channels}_kernelSize_{kernel_size}_PoolSize_{pool_size}_BlurTest_{False}_BlurTrain_{False}"

    show_learning_curve(train_cost_history=training_losses, test_cost_history=test_losses, name=name)


if __name__ == '__main__':
    main()
