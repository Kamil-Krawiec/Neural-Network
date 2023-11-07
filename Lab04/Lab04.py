import numpy as np
import pandas as pd
from Functions import *
from sklearn.model_selection import train_test_split
from Lab03.main import make_dataset_more_balanced
from TorchMLPClassifier import TorchMLPClassifier
import torch
from torch import nn

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def predict(model, X, Y):
    with torch.no_grad():
        y_pred = model(X)
        y_pred = torch.round(torch.sigmoid(y_pred))
        acc = accuracy_score(Y, y_pred)
        f1 = f1_score(Y, y_pred, average='macro')  # Specify the 'average' parameter
        precision = precision_score(Y, y_pred, average='macro', zero_division=1)  # Specify the 'average' parameter
        recall = recall_score(Y, y_pred, average='macro')  # Specify the 'average' parameter

    return acc, f1, precision, recall


def train_model(model, optimizer, X_train, y_train, X_test, y_test, num_iterations=100, batch_size=30, seed=1,
                criterion=None):
    torch.manual_seed(seed)

    cost_history = []
    test_cost_list = []
    scores_history = []

    num_batches = len(X_train) // batch_size

    X_train_tensor = torch.from_numpy(X_train.values).float()
    y_train_tensor = torch.from_numpy(y_train).float()
    n_samples = X_train_tensor.shape[0]

    for i in range(num_iterations):
        random_order = torch.randperm(n_samples)
        X_shuffled = X_train_tensor[random_order]
        y_shuffled = y_train_tensor[random_order]

        for batch in range(num_batches):
            start = batch * batch_size
            end = (batch + 1) * batch_size
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            optimizer.zero_grad()
            y_pred = model.forward(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            cost_history.append(loss.item())

        if (i + 1) % (num_iterations // 20) == 0:
            scores_history.append(predict(model, X_test, y_test))

        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs.squeeze(), y_test)
            test_cost_list.append(test_loss.item())

    return cost_history, test_cost_list, scores_history


if __name__ == "__main__":
    X = pd.read_csv('../Dataset/X.csv')
    y = pd.read_csv('../Dataset/y.csv')
    y = y.map(make_dataset_more_balanced)

    y = y.values
    y = np.squeeze(y)

    y = pd.get_dummies(y).values.astype('int64')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    y_test = torch.from_numpy(y_test).float()
    X_test = torch.from_numpy(X_test.values).float()

    hidden_sizes = [128, 64]
    num_iterations = 100
    batch_sizes = [32, 64]
    learning_rates = [0.001, 0.01, 0.1]

    optimizers = [
        ('SGD', torch.optim.SGD),# SGD (Stochastic Gradient Descent)
        ('Adam', torch.optim.Adam), # Adam (Adaptive Moment Estimation):
        ('RMSProp', torch.optim.RMSprop)# RMSProp (Root Mean Square Propagation)
    ]

    for optimizer_name, optimizer_type in optimizers:
        for batch_size in batch_sizes:
            for learning_rate in learning_rates:
                name = f'op_{optimizer_name}_bs_{batch_size}_lr_{learning_rate}'

                model = TorchMLPClassifier(input_size=X_train.shape[1], hidden_sizes=hidden_sizes, output_size=3)
                optimizer = optimizer_type(model.parameters(), lr=learning_rate)
                criterion = nn.BCEWithLogitsLoss()  # Specify your loss criterion here

                cost, test_cost, scores_history = train_model(model, optimizer, X_train, y_train, X_test, y_test,
                                                              num_iterations=num_iterations, batch_size=batch_size,
                                                              criterion=criterion)

                show_learning_curve(test_cost, name=name+"_test")
                show_learning_curve(cost, name=name+"_train")
                acc, f1, precision, recall = predict(model, X_test, y_test)
                show_metrics(acc, f1, precision, recall, name=name+"_metrics", scores_history=scores_history)
