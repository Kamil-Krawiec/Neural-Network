import torch
import torch.nn.init as init
import torchvision
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pandas as pd

from Functions import *
from NeuralNetwork import NeuralNetwork

seed = 1


def init_normal(m):
    if isinstance(m, nn.Linear):
        torch.manual_seed(seed)
        init.xavier_normal_(m.weight)
        init.constant_(m.bias, 0)


def predict(model, data_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy


def prepare_data(train_size_percentage, data, add_noise_train=False, add_noise_test=False, noise_std=0.1,
                 batch_size=16):
    total_size = len(data)
    train_size = int(train_size_percentage * total_size)
    test_size = total_size - train_size

    train_data, test_data = random_split(data, [train_size, test_size])

    if add_noise_train:
        train_data = [(add_noise(inputs, std=noise_std), labels) for inputs, labels in train_data]

    if add_noise_test:
        test_data = [(add_noise(inputs, std=noise_std), labels) for inputs, labels in test_data]

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def add_noise(data_to_noise, mean=0, std=0.1):
    noise = torch.randn_like(data_to_noise) * std + mean
    return data_to_noise + noise


def train(model, criterion, data_loader, test_loader, epochs):
    # reset model
    model.apply(init_normal)
    optimizer = optim.Adam(model.parameters())
    torch.manual_seed(seed)

    training_loss = []
    test_loss_list = []
    for epoch in range(epochs):
        running_loss = 0
        test_loss = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.view(images.shape[0], -1)
                logits = model(images)
                loss_test = criterion(logits, labels)
                test_loss += loss_test.item()

        for images, labels in data_loader:
            images = images.view(images.shape[0], -1)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        running_loss = running_loss / len(data_loader)
        test_loss = test_loss / len(test_loader)
        training_loss.append(running_loss)
        test_loss_list.append(test_loss)

        print(f"Epoch {epoch}/{epochs} - Train Loss: {running_loss:.4f}, Test Loss: {test_loss:.4f}")
    return training_loss, test_loss_list


def main():
    results_df = pd.DataFrame(
        columns=['hidden_size', 'train_size', 'batch_size', 'test_noise', 'train_noise', 'accuracy',
                 'train_size_percentage','epochs'])

    transform = transforms.Compose([transforms.ToTensor()])

    data = torchvision.datasets.FashionMNIST('path', download=True, transform=transform)

    # Parametry modelu
    input_size = data[0][0].shape[1] * data[0][0].shape[2]
    output_size = len(set(data.targets))
    hidden_layers = [10]
    batch_sizes = [512, 1024]
    epochs = 15

    for batch_size in batch_sizes:
        model = NeuralNetwork(input_size, hidden_layers, output_size)
        criterion = nn.CrossEntropyLoss()
        for train_size_percentage in [0.2]:
            for add_noise_train, add_noise_test in [(False, False), (True, True)]:
                train_loader, test_loader = prepare_data(train_size_percentage, data,
                                                         add_noise_train=add_noise_train,
                                                         add_noise_test=add_noise_test, batch_size=batch_size)

                cost_history, test_cost_list = train(model=model,
                                                     data_loader=train_loader,
                                                     criterion=criterion,
                                                     epochs=epochs,
                                                     test_loader=test_loader
                                                     )

                acc = predict(model, test_loader)
                name = f'ls_{hidden_layers}_ts_{train_size_percentage}' \
                       f'_bs_{batch_size}_acc_{round(acc, 2)}_noise_train_{add_noise_train}_noise_test_{add_noise_test}'
                show_learning_curve(cost_history, test_cost_list, name=name)
                print(name)

                new_row_data = {
                    'hidden_size': hidden_layers,
                    'train_size': train_size_percentage,
                    'batch_size': batch_size,
                    'test_noise': add_noise_test,
                    'train_noise': add_noise_train,
                    'accuracy': acc,
                    'train_size_percentage': train_size_percentage,
                    'epochs': epochs,
                }

                # Create a new DataFrame from the dictionary
                new_row_df = pd.DataFrame([new_row_data])

                # Concatenate the new DataFrame with the existing results_df
                results_df = pd.concat([results_df, new_row_df], ignore_index=True)

                results_df.to_csv(f'../media/Lab05_files/results.csv', index=False)


if __name__ == '__main__':
    main()
