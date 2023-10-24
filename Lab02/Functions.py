from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer


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
