import keras
import numpy as np

def create_LSTM_NN(input_size,interior_layer_dimensions, output_size):
    model = keras.models.Sequential()

    for i in range(len(interior_layer_dimensions)):

        if i == 0:
            input_dimension = input_size
            output_dimension = interior_layer_dimensions[i]
        else:
            input_dimension = interior_layer_dimensions[i-1]
            output_dimension = interior_layer_dimensions[i]

        LSTM_layer = keras.layers.LSTMCell(output_dimension,)
