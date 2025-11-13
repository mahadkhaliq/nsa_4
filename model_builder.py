import tensorflow as tf
from keras.layers.fake_approx_convolutional import FakeApproxConv2D

def build_model(arch):
    """Build model with exact Conv2D layers"""
    layers = []

    use_batch_norm = arch.get('use_batch_norm', False)

    for i in range(arch['num_conv_layers']):
        layers.append(tf.keras.layers.Conv2D(
            filters=arch['filters'][i],
            kernel_size=(arch['kernels'][i], arch['kernels'][i]),
            activation=None if use_batch_norm else 'relu',
            padding='same'
        ))

        if use_batch_norm:
            layers.append(tf.keras.layers.BatchNormalization())

        layers.append(tf.keras.layers.Activation('relu'))
        layers.append(tf.keras.layers.MaxPooling2D())

    layers.append(tf.keras.layers.Flatten())
    layers.append(tf.keras.layers.Dense(arch['dense_units'], activation='relu'))
    layers.append(tf.keras.layers.Dense(10, activation='softmax'))

    return tf.keras.Sequential(layers)

def build_approx_model(arch):
    """Build model with approximate Conv2D layers"""
    layers = []

    use_batch_norm = arch.get('use_batch_norm', False)

    for i in range(arch['num_conv_layers']):
        # Each layer uses its own multiplier
        mul_map = arch['mul_map_files'][i]
        layers.append(FakeApproxConv2D(
            filters=arch['filters'][i],
            kernel_size=(arch['kernels'][i], arch['kernels'][i]),
            activation=None if use_batch_norm else 'relu',
            mul_map_file=mul_map,
            padding='same'
        ))

        if use_batch_norm:
            layers.append(tf.keras.layers.BatchNormalization())

        layers.append(tf.keras.layers.Activation('relu'))
        layers.append(tf.keras.layers.MaxPooling2D())

    layers.append(tf.keras.layers.Flatten())
    layers.append(tf.keras.layers.Dense(arch['dense_units'], activation='relu'))
    layers.append(tf.keras.layers.Dense(10, activation='softmax'))

    return tf.keras.Sequential(layers)
