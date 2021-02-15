import tensorflow as tf
import rl

def create_model(input_shape, output_size):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(2*output_size, activation='relu'),
        tf.keras.layers.Dense(output_size, activation='linear')
    ])

    print(model.summary())

    return model

