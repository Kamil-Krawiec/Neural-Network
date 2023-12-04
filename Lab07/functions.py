from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, Dense
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
from Functions import save_chart

max_len = 100
recurrence_layer_size = 246


def build_rnn_model():
    model = Sequential()
    model.add(Embedding(input_dim=10000, output_dim=32, input_length=max_len))
    model.add(SimpleRNN(units=recurrence_layer_size, activation='tanh'))
    model.add(Dense(units=1, activation='sigmoid'))
    return model


def build_lstm_model():
    model = Sequential()
    model.add(Embedding(input_dim=10000, output_dim=32, input_length=max_len))
    model.add(LSTM(units=recurrence_layer_size, activation='tanh'))
    model.add(Dense(units=1, activation='sigmoid'))
    return model


# Train and evaluate models
def train_and_evaluate(model, x_train, y_train, x_test, y_test):
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=20, batch_size=512, validation_split=0.1,verbose=0)
    loss, accuracy = model.evaluate(x_test, y_test,verbose=0)
    names = [(layer.name, type(layer).__name__) for layer in model.layers]
    name = f"model-{names[1][1]}|max_len-{max_len}|r_layer_size-{recurrence_layer_size}|activation-tanh"
    plot_history(history, loss, accuracy,name)


def plot_history(history, test_loss, test_accuracy, name):
    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 6))
    plt.suptitle(name, fontsize=16)

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.axhline(test_accuracy, color='r', linestyle='--', label='Test Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation', 'Test'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.axhline(test_loss, color='r', linestyle='--', label='Test Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation', 'Test'], loc='upper left')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust subplot layout and add space for suptitle
    save_chart(name)
    plt.show()


