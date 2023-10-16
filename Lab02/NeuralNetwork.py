import random

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit

from Lab02.Functions import compute_metrics


class NeuralNetwork:
    def __init__(self, size, num_of_iterations=1000, learning_rate=0.03, name='basic data'):
        self.W = np.random.rand(size, 1)
        self.b = np.random.rand(1)
        self.name = name
        self.num_iterations = num_of_iterations
        self.learning_rate = learning_rate
        self.cost_history = []
        self.scores_history = []

    def sigmoid(self, n):
        return expit(n)

    def p(self, x):
        argument = np.dot(x, self.W) + self.b
        return self.sigmoid(argument)

    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return compute_metrics(y_test, y_pred)

    def predict(self, X_test):
        return (self.p(X_test) > 0.5).astype(int)

    def cross_entropy_loss(self, y, y_pred):
        epsilon = 1e-15
        loss = y * np.log(y_pred + epsilon) + (1 - y) * np.log(1 - y_pred + epsilon)
        return -np.sum(loss)

    def compute_gradient(self, X_train, y_train):
        y_pred = self.p(X_train)
        dz = y_pred - y_train
        dw = np.dot(X_train.T, dz)
        db = np.sum(dz)
        return dw, db

    def fit_model_convergence(self, X_train, y_train, learning_rate, convergence_threshold):
        prev_cost = float('inf')
        i = 0
        while True:
            dw, db = self.compute_gradient(X_train, y_train)

            # Aktualizacja wag i bias zgodnie z gradientem i współczynnikiem uczenia
            self.W -= learning_rate * dw
            self.b -= learning_rate * db

            # Oblicz funkcję kosztu
            cost = self.cross_entropy_loss(y_train, self.p(X_train))

            # Weryfikacja zbieżności
            if i % 10 == 0:
                self.cost_history.append(cost)
                self.scores_history.append(compute_metrics(y_train, self.p(X_train)))
                print(f"Iteration {i}: Cost = {cost}")

            # Sprawdzenie warunku zbieżności
            if abs(prev_cost - cost) < convergence_threshold:
                print(f"Convergence reached with cost difference {abs(prev_cost - cost)}")
                break

            i += 1
            prev_cost = cost

    def fit(self, X_train, y_train):
        for i in range(self.num_iterations):

            # Przemieszaj dane uczące
            permutation = np.random.permutation(X_train.shape[0])
            X_train_shuffled = X_train[permutation]
            y_train_shuffled = y_train[permutation]

            dw, db = self.compute_gradient(X_train_shuffled, y_train_shuffled)

            # Aktualizacja wag i bias zgodnie z gradientem i współczynnikiem uczenia
            self.W -= self.learning_rate * dw
            self.b -= self.learning_rate * db

            # funkcja kosztu
            cost = self.cross_entropy_loss(y_train, self.p(X_train))

            if i % (self.num_iterations // 10) == 0:
                self.cost_history.append(cost)
                self.scores_history.append(compute_metrics(y_train, self.p(X_train)))

    def fit_batches(self, X_train, y_train, batch_size):
        num_batches = len(X_train) // batch_size

        for i in range(self.num_iterations):

            # Przemieszaj dane uczące
            permutation = np.random.permutation(X_train.shape[0])
            X_train_shuffled = X_train[permutation]
            y_train_shuffled = y_train[permutation]

            for batch in range(num_batches):
                start = batch * batch_size
                end = (batch + 1) * batch_size

                X_batch = X_train_shuffled[start:end]
                y_batch = y_train_shuffled[start:end]

                dw, db = self.compute_gradient(X_batch, y_batch)

                # Aktualizacja wag i bias zgodnie z gradientem i współczynnikiem uczenia
                self.W -= self.learning_rate * dw
                self.b -= self.learning_rate * db

            # funkcja kosztu
            cost = self.cross_entropy_loss(y_train, self.p(X_train))

            if i % (self.num_iterations // 10) == 0:
                self.cost_history.append(cost)
                self.scores_history.append(compute_metrics(y_train, self.p(X_train)))

    def show_learning_curve(self):
        plt.plot(range(len(self.cost_history)), self.cost_history)
        plt.xlabel("Iterations")
        plt.ylabel("Cost")
        plt.title("Training Cost for " + self.name)
        plt.show()

    def show_metrics(self, test_accuracy, test_precision, test_recall, test_f1):
        accuracy_history, precision_history, recall_history, f1_history = zip(*self.scores_history)
        iterations = range(len(self.scores_history))
        plt.plot(iterations, accuracy_history, color='r', label="Accuracy")
        plt.plot(iterations, precision_history, color='g', label="Precision")
        plt.plot(iterations, recall_history, color='b', label="Recall")
        plt.plot(iterations, f1_history, color='m', label="F1-Score")
        plt.axhline(y=test_accuracy, color='r', linestyle='--', label="Test Accuracy")
        plt.axhline(y=test_precision, color='g', linestyle='--', label="Test Precision")
        plt.axhline(y=test_recall, color='b', linestyle='--', label="Test Recall")
        plt.axhline(y=test_f1, color='m', linestyle='--', label="Test F1-Score")
        plt.xlabel("Iterations")
        plt.ylabel("Metric Value")
        plt.title("Training Metrics for " + self.name)
        plt.legend()
        plt.show()
