import time

import torch
from sklearn.metrics import precision_score, f1_score, recall_score
from torch import nn
from torch import optim
from torch.autograd import Variable

seed = 42


def train(model, criterion, data_loader, test_loader, epochs):
    model.apply(init_normal)
    optimizer = optim.Adam(model.parameters())
    torch.manual_seed(seed)

    start_timestamp = time.time()
    training_loss = []
    test_loss_list = []

    for epoch in range(epochs):
        running_loss = 0
        test_loss = 0
        with torch.no_grad():

            for images, labels in test_loader:
                images = Variable(images.view(-1, 1, 28, 28))
                labels = Variable(labels)

                # forward pass
                logits = model(images)
                loss_test = criterion(logits, labels)
                test_loss += loss_test.item()

        for images, labels in data_loader:
            images = Variable(images.view(-1, 1, 28, 28))
            labels = Variable(labels)
            logits = model(images)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        running_loss = running_loss / len(data_loader)
        test_loss = test_loss / len(test_loader)
        training_loss.append(running_loss)
        test_loss_list.append(test_loss)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs} - Train Loss: {running_loss:.4f}, Test Loss: {test_loss:.4f}")

    print(f"\nTraining Time (in seconds) = {(time.time() - start_timestamp):.2f}")
    return training_loss, test_loss_list


def init_normal(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.manual_seed(seed)
        if hasattr(m, 'weight'):
            nn.init.xavier_normal_(m.weight)
        if hasattr(m, 'bias'):
            nn.init.constant_(m.bias, 0)


def evaluate_model(model, test_loader):
    model.eval()
    total = 0
    correct = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.view(-1, 1, 28, 28)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.numpy())
            all_predictions.extend(predicted.numpy())

    accuracy = correct / total
    precision = precision_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')

    return accuracy, precision, f1, recall
