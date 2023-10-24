import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit
from Lab02.Functions import compute_metrics_one_hot


class MLPClassifier:
    def __init__(self, input_size, hidden_size, output_size, num_iterations=1000, learning_rate=0.03):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate

        # Inicjalizacja wag i biasów
        self.W1 = np.random.rand(input_size, hidden_size)
        self.b1 = np.random.rand(1, hidden_size)
        self.W2 = np.random.rand(hidden_size, output_size)
        self.b2 = np.random.rand(1, output_size)

    def sigmoid(self, x):
        return expit(x)

    def sigmoid_derivative(self,x):
        fx = self.sigmoid(x)
        return fx * (1 - fx)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def compute_loss(self, y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.sum(y_true * np.log(y_pred)) / len(y_true)
        return loss

    def forward_propagation(self, X):
        # Oblicz wyjście z warstwy ukrytej
        z1 = np.dot(X, self.W1) + self.b1
        a1 = self.sigmoid(z1)

        # Oblicz wyjście z warstwy wyjściowej
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = self.softmax(z2)

        return a1, a2

    def backward_propagation(self, a1, a2, y):
        delta2 = y - a2

        # Oblicz błędy dla jednostek ukrytych z wykorzystaniem sigmoid_derivative
        delta1 = (delta2.dot(self.W2.T)) * self.sigmoid_derivative(a1)

        return delta1, delta2

    def update_weights(self, X, a1, delta1, delta2):

        self.W2 += a1.T.dot(delta2) * self.learning_rate
        self.b2 += np.sum(delta2, axis=0, keepdims=True) * self.learning_rate

        self.W1 += X.T.dot(delta1) * self.learning_rate
        self.b1 += np.sum(delta1, axis=0, keepdims=True) * self.learning_rate

    def fit(self, X, y):
        for iteration in range(self.num_iterations):

            a1, a2 = self.forward_propagation(X)

            delta1, delta2 = self.backward_propagation(a1, a2, y)

            self.update_weights(X, a1, delta1, delta2)

            loss = self.compute_loss(y, a2)
            if iteration % (self.num_iterations // 10) == 0:
                print(f"Iteration {iteration}: Loss {loss:.4f}")

    def predict(self, X):
        a1, a2 = self.forward_propagation(X)
        return np.argmax(a2, axis=1)

    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        # Zamień y_pred z listy klas na macierz one-hot
        num_samples = len(y_pred)
        num_classes = len(y_test[0])
        y_pred_one_hot = np.zeros((num_samples, num_classes))
        y_pred_one_hot[np.arange(num_samples), y_pred] = 1

        return compute_metrics_one_hot(y_test, y_pred_one_hot)