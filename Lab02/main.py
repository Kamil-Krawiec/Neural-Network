import pandas as pd
from sklearn.model_selection import train_test_split

from Lab02.Distribution import *
from Functions import normalize_data, discretize_data, basic_data
from Lab02.NeuralNetwork import NeuralNetwork


def test_X_distribution(distribution=Distribution.NONE):
    if distribution == Distribution.NONE:
        learning_rate_basic_without_b = 0.1
        learning_rate_basic_with_b = 0.001
        num_of_iterations_basic = 400
        batch_size = 100

        with_X_method(X, y,
                      basic_data,
                      learning_rate=learning_rate_basic_without_b,
                      num_of_iterations=num_of_iterations_basic)
        with_X_method(X, y,
                      basic_data,
                      learning_rate=learning_rate_basic_with_b,
                      num_of_iterations=num_of_iterations_basic,
                      fit_with_batches=1,
                      batch_size=batch_size)
    elif distribution == Distribution.DISCRETE:
        learning_rate_discrete_without_b = 0.0005
        learning_rate_discrete_with_b = 0.0005
        batch_size = 64
        num_of_iterations_discretization = 60

        with_X_method(X, y,
                      discretize_data,
                      learning_rate=learning_rate_discrete_without_b,
                      num_of_iterations=num_of_iterations_discretization)
        with_X_method(X, y,
                      discretize_data,
                      learning_rate=learning_rate_discrete_with_b,
                      num_of_iterations=num_of_iterations_discretization,
                      fit_with_batches=1,
                      batch_size=batch_size)
    elif distribution == Distribution.NORMAL:
        learning_rate_normalization_without_b = 0.001
        learning_rate_normalization_with_b = 0.001
        num_of_iterations_normalization = 200
        batch_size= 128
        with_X_method(X, y,
                      normalize_data,
                      learning_rate=learning_rate_normalization_without_b,
                      num_of_iterations=num_of_iterations_normalization)
        with_X_method(X, y,
                      normalize_data,
                      learning_rate=learning_rate_normalization_with_b,
                      num_of_iterations=num_of_iterations_normalization,
                      fit_with_batches=1,
                      batch_size= batch_size)


def with_X_method(X, y, method, learning_rate=0.01, num_of_iterations=1000, fit_with_batches=False, batch_size=8):
    method_name = method.__name__
    if fit_with_batches:
        method_name += f' with batch size {batch_size}'
    else:
        method_name += ' without batches'

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train, X_test = method(X_train=X_train, X_test=X_test)

    # Ustaw hiperparametry i parametry
    neural_network = NeuralNetwork(size=X_train.shape[1],
                                   name=method_name,
                                   learning_rate=learning_rate,
                                   num_of_iterations=num_of_iterations
                                   )
    if fit_with_batches:
        neural_network.fit_batches(X_train, y_train, batch_size)
    else:
        neural_network.fit(X_train, y_train)
    neural_network.show_learning_curve()

    # Oblicz metryki
    neural_network.show_metrics(*neural_network.score(X_test, y_test))


if __name__ == "__main__":
    X = pd.read_csv('../Dataset/X.csv')
    y = pd.read_csv('../Dataset/y.csv')
    y = y.map(lambda x: 1 if x in (1, 2, 3, 4) else 0).values
    test_X_distribution(Distribution.NORMAL)
    test_X_distribution(Distribution.DISCRETE)
    test_X_distribution(Distribution.NONE)


