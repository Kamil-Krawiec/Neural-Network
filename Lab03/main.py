import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from Lab02.Functions import normalize_data
from Lab03.NeuralNetwork import MLPClassifier


def make_dataset_more_balanced(x):
    if x in (1, 2):
        return 1
    elif x in (3, 4):
        return 2
    else:
        return x


if __name__ == "__main__":
    X = pd.read_csv('../Dataset/X.csv')
    y = pd.read_csv('../Dataset/y.csv')
    y = y.map(make_dataset_more_balanced)
    y_value_counts = y.value_counts()
    y_value_percentages = (y_value_counts / len(y)) * 100

    y = y.values
    y = np.squeeze(y)  # Flatten y to a 1D array

    y_one_hot = pd.get_dummies(y).values.astype('int64')

    X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

    hidden_layers = (16,)

    model = MLPClassifier(layer_sizes=[X_train.shape[1], *hidden_layers, y_train.shape[1]], num_iterations=2000,
                          learning_rate=0.001,name = 'Basic MLP')

    X_train, X_test = normalize_data(X_train=X_train, X_test=X_test)

    model.fit(X_train, y_train)
    model.show_metrics(*model.score(X_test, y_test))
    model.show_learning_curve()

    model_batch = MLPClassifier(layer_sizes=[X_train.shape[1], *hidden_layers, y_train.shape[1]], num_iterations=2000,
                          learning_rate=0.001,name = 'Batch MLP')

    model_batch.fit_batches(X_train, y_train)
    model_batch.show_metrics(*model.score(X_test, y_test))
    model_batch.show_learning_curve()
