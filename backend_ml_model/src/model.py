import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, LSTM, Dense, Bidirectional, Dropout
from typing import Tuple

def build_lstm_model(input_shape: Tuple[int, int], num_classes: int = 10) -> tf.keras.Model:
    """
    Builds and compiles a Keras CNN+LSTM model for sequence classification.
    
    Uses convolutional layers to extract features from the sequence,
    followed by LSTM for temporal relationships, and dense for classification.

    Args:
        input_shape: A tuple representing the shape of one input sequence
                     (e.g., (sequence_length, num_features)).
        num_classes: The number of output classes (e.g., 10 for digits 0-9).

    Returns:
        A compiled tf.keras.Model instance.
    """
    model = Sequential(name="AirWriting_CNN_LSTM_Classifier")

    # --- CNN Feature Extraction Layers ---
    # Using Conv1D - good for finding patterns across the time sequence dimension
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape, name="conv1d_1"))
    model.add(MaxPooling1D(pool_size=2, name="maxpool1d_1"))
    model.add(Dropout(0.3, name="dropout_1")) # Added Dropout

    model.add(Conv1D(filters=128, kernel_size=3, activation='relu', name="conv1d_2"))
    model.add(MaxPooling1D(pool_size=2, name="maxpool1d_2"))
    model.add(Dropout(0.3, name="dropout_2")) # Added Dropout

    # --- LSTM Sequence Processing Layer ---
    # Feed the output of CNN features into LSTM
    # `return_sequences=False` as it's the last RNN layer before Dense
    model.add(LSTM(64, name="lstm_layer")) # Or use Bidirectional(LSTM(64)) based on BiLSTM findings
    model.add(Dropout(0.3, name="dropout_3")) # Added Dropout

    # --- Output Layer ---
    model.add(Dense(num_classes, activation='softmax', name="output_probabilities"))

    # --- Compile ---
    # Optimizer and loss remain the same for classification
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model 