import os
import textwrap

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer


def initialize_weights_normal(shape, n_inputs):
    return np.random.normal(loc=0.0, scale=0.1, size=shape)


def initialize_weights_xavier(shape, n_inputs):
    stddev = np.sqrt(1 / n_inputs)
    return np.random.normal(loc=0.0, scale=stddev, size=shape)


def initialize_weights_he(shape, n_inputs):
    stddev = np.sqrt(2 / n_inputs)
    return np.random.normal(loc=0.0, scale=stddev, size=shape)


def compute_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, (y_pred > 0.5).astype(int))
    precision = precision_score(y_true, (y_pred > 0.5).astype(int), zero_division=1)
    recall = recall_score(y_true, (y_pred > 0.5).astype(int))
    f1 = f1_score(y_true, (y_pred > 0.5).astype(int))
    return accuracy, precision, recall, f1


def compute_metrics_one_hot(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    # Ustaw zero_division=0 w precision_score
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    return accuracy, precision, recall, f1


def normalize_data(X_train, X_test):
    # Inicjalizacja obiektu do normalizacji
    scaler = StandardScaler()

    # Zastosuj normalizację na danych treningowych i testowych
    X_train_normalized = scaler.fit_transform(X_train)
    X_test_normalized = scaler.transform(X_test)

    return X_train_normalized, X_test_normalized


def discretize_data(X_train, X_test, n_bins=10, strategy='uniform'):
    # Initialize the KBinsDiscretizer object
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy, subsample=None)

    X_train_discretized = discretizer.fit_transform(X_train)

    X_test_discretized = discretizer.transform(X_test)

    return X_train_discretized, X_test_discretized


def basic_data(X_train, X_test):
    return X_train.values, X_test.values


def show_metrics(test_accuracy, test_precision, test_recall, test_f1, name='', scores_history=()):
    accuracy_history, precision_history, recall_history, f1_history = zip(*scores_history)
    iterations = range(len(scores_history))
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
    max_title_width = 40
    title = "Training Metrics for " + name
    wrapped_title = textwrap.fill(title, max_title_width)
    plt.title(wrapped_title)
    plt.legend(loc='lower right')
    plt.subplots_adjust(top=0.85)
    save_chart_lab05(name)
    plt.show()


def smooth_curve(points, factor=0.95):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points



def show_learning_curve(cost_history, name=''):
    plt.figure(figsize=(10, 6))

    # Plot the original cost history in blue
    plt.plot(range(len(cost_history)), cost_history, color='grey', label='Original Curve')

    # Plot the smoothed cost history in red with orange line
    smoothed_cost_history = smooth_curve(cost_history)
    plt.plot(range(len(smoothed_cost_history)), smoothed_cost_history, color='orange', label='Smoothed Curve')

    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    max_title_width = 40
    title = "Learning Curve " + name
    wrapped_title = textwrap.fill(title, max_title_width)
    plt.title(wrapped_title)
    plt.legend()

    save_chart(name)
    plt.show()

def show_learning_curve(train_cost_history, test_cost_history, name='',acc=0,prec=0,rec=0,f1=0):
    plt.figure(figsize=(15, 10))

    # Plot the original training cost history in blue
    plt.plot(range(len(train_cost_history)), train_cost_history, color='blue', label='Training Curve')

    # Plot the original testing cost history in red
    plt.plot(range(len(test_cost_history)), test_cost_history, color='red', label='Testing Curve')

    plt.xlabel("Iterations")
    plt.ylabel("Cost")

    max_title_width = 40
    title = "Learning and Testing Curve " + name
    wrapped_title = textwrap.fill(title, max_title_width)
    plt.title(wrapped_title)
    plt.legend()

    # Add text below the chart
    # Add table below the chart
    table_data = [['Accuracy', round(acc, 2)],
                  ['Recall', round(rec, 2)],
                  ['F1 Score', round(f1, 2)],
                  ['Precision', round(prec, 2)]]

    plt.table(cellText=table_data, loc='bottom', colLabels=['Metric', 'Value'], cellLoc='center', bbox=[0, -0.3, 1, 0.2],fontsize=12)
    plt.subplots_adjust(bottom=0.3)

    # Save and display the chart
    save_chart(name)
    plt.show()

def save_chart(name):
    # Split the name into parts
    parts = name.split('_')


    # Construct the directory structure

    save_directory = f"../media/Lab07_files/"

    # Check if the directory exists, if not, create it
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Move the saved chart to the appropriate directory
    new_name = os.path.join(save_directory, f"{name}.jpg")
    plt.savefig(new_name)

