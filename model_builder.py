import tensorflow as tf
from keras.layers.fake_approx_convolutional import FakeApproxConv2D

def build_model(arch):
    """Build model with exact Conv2D layers"""
    layers = []

    for i in range(arch['num_conv_layers']):
        layers.append(tf.keras.layers.Conv2D(
            filters=arch['filters'][i],
            kernel_size=(arch['kernels'][i], arch['kernels'][i]),
            activation='relu',
            padding='same'
        ))
        layers.append(tf.keras.layers.MaxPooling2D())

    layers.append(tf.keras.layers.Flatten())
    layers.append(tf.keras.layers.Dense(arch['dense_units'], activation='relu'))
    layers.append(tf.keras.layers.Dense(10, activation='softmax'))

    return tf.keras.Sequential(layers)

def build_approx_model(arch):
    """Build model with approximate Conv2D layers"""
    layers = []

    for i in range(arch['num_conv_layers']):
        layers.append(FakeApproxConv2D(
            filters=arch['filters'][i],
            kernel_size=(arch['kernels'][i], arch['kernels'][i]),
            activation='relu',
            mul_map_file=arch['mul_map_file'],
            padding='same'
        ))
        layers.append(tf.keras.layers.MaxPooling2D())

    layers.append(tf.keras.layers.Flatten())
    layers.append(tf.keras.layers.Dense(arch['dense_units'], activation='relu'))
    layers.append(tf.keras.layers.Dense(10, activation='softmax'))

    return tf.keras.Sequential(layers)
