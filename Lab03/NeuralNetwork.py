import os

import numpy as np
from matplotlib import pyplot as plt
from scipy.special import expit

from Lab02.Functions import compute_metrics_one_hot


class MLPClassifier:
    def __init__(self, layer_sizes, num_iterations=1000, learning_rate=0.03, name='Basic'):
        self.layer_sizes = layer_sizes
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.weights = [np.random.rand(layer_sizes[i], layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)]
        self.biases = [np.random.rand(1, size) for size in layer_sizes[1:]]
        self.scores_history = []
        self.cost_history = []
        self.name = name

    def sigmoid(self, x):
        return expit(x)

    def sigmoid_derivative(self, x):
        fx = self.sigmoid(x)
        return fx * (1 - fx)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def compute_loss(self, y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.sum(y_true * np.log(y_pred))
        return loss

    '''
    Przy przejściu w przód, musimy zapisywać wartość wektora x na potrzeby
    późniejszego przejścia po operacjach wstecz
    
    Jest to robione za pomoca tablicy activations ktora zapisuje wyjscia z poszczegolnych warstw
    '''

    def forward_propagation(self, X):
        activations = [X]

        for i in range(len(self.layer_sizes) - 1):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            a = self.sigmoid(z) if i < len(self.layer_sizes) - 2 else self.softmax(z)
            activations.append(a)

        return activations

    '''
    Przy przejściu w przód, musimy zapisywać wartość wektora x na potrzeby
    późniejszego przejścia po operacjach wstecz

    Jest to robione za pomoca tablicy activations ktora zapisuje wyjscia z poszczegolnych warstw
    '''

    def backward_propagation(self, activations, y):
        num_layers = len(self.layer_sizes)
        deltas = [None] * (num_layers - 1)
        deltas[-1] = y - activations[-1]

        for i in range(num_layers - 2, 0, -1):
            deltas[i - 1] = (deltas[i].dot(self.weights[i].T)) * self.sigmoid_derivative(activations[i])

        return deltas

    def update_weights(self, activations, deltas):
        for i in range(len(self.layer_sizes) - 1):
            self.weights[i] += np.dot(activations[i].T, deltas[i]) * self.learning_rate
            self.biases[i] += np.sum(deltas[i], axis=0, keepdims=True) * self.learning_rate

    '''
    W obrębie backward, jeżeli korzystamy z uczonych parametrów modelu,
    powinniśmy wyliczyć i zapisać gradient – pochodne cząstkowe po tych
    parametrach
    
    Funkcja update_weights liczy i zapisuje gradient 
    '''

    def fit_batches(self, X, y, batch_size=16):
        num_batches = len(X) // batch_size

        for i in range(self.num_iterations):
            # Przemieszaj dane uczące
            permutation = np.random.permutation(X.shape[0])
            X_shuffled = X[permutation]
            y_shuffled = y[permutation]

            for batch in range(num_batches):
                start = batch * batch_size
                end = (batch + 1) * batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                activations = self.forward_propagation(X_batch)

                deltas = self.backward_propagation(activations, y_batch)

                self.update_weights(activations, deltas)

            output_layer = self.forward_propagation(X_shuffled)[-1]
            loss = self.compute_loss(y_shuffled,output_layer)
            if i % (self.num_iterations // 10) == 0:
                self.cost_history.append(loss)
                self.scores_history.append(self.score(X_shuffled, y_shuffled))

    def fit(self, X, y):
        for iteration in range(self.num_iterations):

            activations = self.forward_propagation(X)

            deltas = self.backward_propagation(activations, y)

            self.update_weights(activations, deltas)

            loss = self.compute_loss(y, activations[-1])
            if iteration % (self.num_iterations // 10) == 0:
                self.cost_history.append(loss)
                self.scores_history.append(self.score(X, y))

    def show_learning_curve(self):
        plt.plot(range(len(self.cost_history)), self.cost_history)
        plt.xlabel("Iterations")
        plt.ylabel("Cost")
        plt.title("Training Cost for " + self.name)
        self.save_chart()
        plt.show()

    def predict(self, X):
        activations = self.forward_propagation(X)
        return np.argmax(activations[-1], axis=1)

    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        num_samples = len(y_pred)
        num_classes = self.layer_sizes[-1]
        y_pred_one_hot = np.zeros((num_samples, num_classes))
        y_pred_one_hot[np.arange(num_samples), y_pred] = 1

        return compute_metrics_one_hot(y_test, y_pred_one_hot)

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
        self.save_chart()
        plt.show()

    def save_chart(self):
        plot_name = f"../media/Lab03_files/{self.name.split(' ')[0]}"
        index = 1

        while os.path.exists(f"{plot_name}_{index}.png"):
            index += 1

        new_name = f"{plot_name}_{index}.png"
        plt.savefig(new_name)
