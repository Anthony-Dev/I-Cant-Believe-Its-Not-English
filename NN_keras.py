import keras
import numpy as np

def create_LSTM_NN(input_shape,batch_size,embedding_vocabulary_size,embedding_layer_size,interior_layer_dimensions, output_size):
    model = keras.models.Sequential()

    embedding_layer = keras.layers.Embedding(embedding_vocabulary_size,embedding_layer_size,input_shape=input_shape,batch_size=batch_size)

    model.add(embedding_layer)

    for i in range(len(interior_layer_dimensions)):
        if i == len(interior_layer_dimensions) - 1:
            LSTM_layer = keras.layers.LSTM(interior_layer_dimensions[i],stateful=True,dropout=0.5)
        else:
            LSTM_layer = keras.layers.LSTM(interior_layer_dimensions[i],return_sequences=True,stateful=True,dropout=0.5)

        model.add(LSTM_layer)

    final_layer = keras.layers.Dense(output_size,activation='softmax')

    model.add(final_layer)

    model.compile(loss='categorical_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])

    return model

data_dim = 16
timesteps = 8
num_classes = 10
batch_size = 32
vocabulary_size = 100

LSTM = create_LSTM_NN((timesteps,data_dim),batch_size,vocabulary_size,5,[5,5,5],num_classes)
print(LSTM.summary())

# Generate dummy training data
x_train = np.random.random((batch_size * 10, timesteps, data_dim))
y_train = np.random.random((batch_size * 10, num_classes))

# Generate dummy validation data
x_val = np.random.random((batch_size * 3, timesteps, data_dim))
y_val = np.random.random((batch_size * 3, num_classes))

LSTM.fit(x_train, y_train,
          batch_size=batch_size, epochs=5, shuffle=False,
          validation_data=(x_val, y_val))
