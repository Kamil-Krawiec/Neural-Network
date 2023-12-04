import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from functions import *

# Load IMDB dataset
(X, Y), _ = imdb.load_data(
    path="imdb.npz",
    num_words=1000,
    skip_top=0,
    maxlen=None,
    seed=113,
    start_char=1,
    oov_char=2,
    index_from=3)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=113)

# Preprocess the data
x_train = pad_sequences(x_train, maxlen=max_len)
x_test = pad_sequences(x_test, maxlen=max_len)

# Train and evaluate RNN model
rnn_model = build_rnn_model()
train_and_evaluate(rnn_model, x_train, y_train, x_test, y_test)

# Train and evaluate LSTM model
lstm_model = build_lstm_model()
train_and_evaluate(lstm_model, x_train, y_train, x_test, y_test)
