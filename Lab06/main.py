import random

import numpy as np
from torch.utils.data import Subset
from torchvision import transforms, datasets
from torchvision.transforms import GaussianBlur

from CustomCNN import CustomCNN
from Functions import show_learning_curve
from Lab06.functions import *


def run_tests():
    default_batch_size = 100

    # Parametry do przetestowania
    out_channels_values = [32,126]
    kernel_size_values = [10,15]
    pool_size_values = [20,10]

    for out_channels in out_channels_values:
        for kernel_size in kernel_size_values:
            for pool_size in pool_size_values:
                random.seed(seed)
                np.random.seed(seed)

                transform_blur = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,)),
                    GaussianBlur(kernel_size=9, sigma=(1.5, 5.0))
                ])

                transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

                train_data = datasets.FashionMNIST('path', download=True, train=True, transform=transform_blur)
                test_data = datasets.FashionMNIST('path', download=True, train=False, transform=transform)

                train_list = list(range(len(train_data)))
                random.shuffle(train_list)
                train_20_percent = train_list[:len(train_list) // 5]

                test_list = list(range(len(test_data)))
                random.shuffle(test_list)
                test_20_percent = test_list[:len(test_list) // 5]

                train_subset = Subset(train_data, train_20_percent)
                train_loader = torch.utils.data.DataLoader(train_subset, batch_size=default_batch_size,
                                                           shuffle=True)

                test_subset = Subset(test_data, test_20_percent)
                test_loader = torch.utils.data.DataLoader(test_subset, batch_size=default_batch_size, shuffle=True)

                criterion = nn.CrossEntropyLoss()

                epochs = 20

                dummy_tensor = torch.rand(1, 1, 28, 28)

                cnn_model = CustomCNN(out_channels, kernel_size, pool_size)
                cnn_model(dummy_tensor)

                cnn_model.apply(init_normal)

                training_losses, test_losses = train(
                    cnn_model,
                    criterion,
                    train_loader,
                    test_loader,
                    epochs=epochs)

                acc, prec, f1, rec = evaluate_model(cnn_model, test_loader)

                name = f"ChanN_{out_channels}_kernelSize_{kernel_size}_PoolSize_{pool_size}_BlurTest_{False}_BlurTrain_{True}"
                show_learning_curve(train_cost_history=training_losses, test_cost_history=test_losses, name=name,
                                    acc=acc, rec=rec, prec=prec, f1=f1)


if __name__ == '__main__':
    run_tests()
