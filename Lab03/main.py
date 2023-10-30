import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from Functions import *
from Lab02.Distribution import *
from Lab03.MLPClassifier import MLPClassifier


def make_dataset_more_balanced(x):
    if x in (1, 2):
        return 1
    elif x in (3, 4):
        return 2
    else:
        return x


def test_X_distribution(distribution=Distribution.NONE):
    learning_rate = 0.001
    num_of_iterations = 1000
    batch_size = 64
    hidden_layers = (1000,100,10)

    if distribution == Distribution.NONE:
        with_X_method(X, y,
                      basic_data,
                      learning_rate=learning_rate,
                      num_of_iterations=num_of_iterations,
                      batch_size=batch_size,
                      hidden_layers=hidden_layers)

    elif distribution == Distribution.DISCRETE:
        with_X_method(X, y,
                      discretize_data,
                      learning_rate=learning_rate,
                      num_of_iterations=num_of_iterations,
                      batch_size=batch_size,
                      hidden_layers=hidden_layers)

    elif distribution == Distribution.NORMAL:
        with_X_method(X, y,
                      normalize_data,
                      learning_rate=learning_rate,
                      num_of_iterations=num_of_iterations,
                      batch_size=batch_size,
                      hidden_layers=hidden_layers)


def with_X_method(X, y, method, learning_rate=0.001, num_of_iterations=2000, hidden_layers=(16,), batch_size=8):
    method_name = method.__name__ + f' batch size = {batch_size}, learning rate = {learning_rate}, hidden layers = {hidden_layers}'

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train, X_test = method(X_train=X_train, X_test=X_test)

    # Ustaw hiperparametry i parametry
    neural_network = MLPClassifier(layer_sizes=[X_train.shape[1],
                                                *hidden_layers,
                                                y_train.shape[1]],
                                   num_iterations=num_of_iterations,
                                   learning_rate=learning_rate,
                                   name=method_name,
                                   weights_initializer=initialize_weights_xavier)

    neural_network.fit_batches(X_train, y_train, batch_size)

    neural_network.show_metrics(*neural_network.score(X_test, y_test))
    neural_network.show_learning_curve()


if __name__ == "__main__":
    X = pd.read_csv('../Dataset/X.csv')
    y = pd.read_csv('../Dataset/y.csv')
    y = y.map(make_dataset_more_balanced)
    y_value_counts = y.value_counts()
    y_value_percentages = (y_value_counts / len(y)) * 100

    y = y.values
    y = np.squeeze(y)

    y = pd.get_dummies(y).values.astype('int64')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    test_X_distribution(Distribution.NORMAL)

    # test_X_distribution(Distribution.DISCRETE)
    # test_X_distribution(Distribution.NONE)
