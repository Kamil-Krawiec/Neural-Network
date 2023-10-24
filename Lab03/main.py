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
    y= y.map(make_dataset_more_balanced)
    y_value_counts = y.value_counts()
    y_value_percentages = (y_value_counts / len(y)) * 100

    y = y.values
    y = np.squeeze(y)  # Flatten y to a 1D array

    y_one_hot = pd.get_dummies(y).values.astype('int64')

    X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)
    model = MLPClassifier(input_size=X_train.shape[1], hidden_size=10, output_size=y_train.shape[1], num_iterations=1000,
                          learning_rate=0.01)

    X_train, X_test = normalize_data(X_train=X_train, X_test=X_test)
    # Rozpocznij proces uczenia
    model.fit(X_train, y_train)
    print(model.score(X_test,y_test))

