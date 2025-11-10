import tensorflow as tf
import random
import os
from keras.layers.fake_approx_convolutional import FakeApproxConv2D

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

# Load CIFAR-10
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')

# NAS search space
MUL_MAP_PATH = './multipliers/'
SEARCH_SPACE = {
    'num_conv_layers': [2, 3, 4],
    'filters': [16, 32, 64, 128],
    'kernel_sizes': [3, 5],
    'dense_units': [64, 128, 256],
    'mul_map_files': [MUL_MAP_PATH + 'mul8u_1JFF.bin', MUL_MAP_PATH + 'mul8u_2KV7.bin']
}

def sample_architecture():
    num_layers = random.choice(SEARCH_SPACE['num_conv_layers'])
    filters = [random.choice(SEARCH_SPACE['filters']) for _ in range(num_layers)]
    kernels = [random.choice(SEARCH_SPACE['kernel_sizes']) for _ in range(num_layers)]
    dense = random.choice(SEARCH_SPACE['dense_units'])
    mul_map = random.choice(SEARCH_SPACE['mul_map_files'])

    return {
        'num_conv_layers': num_layers,
        'filters': filters,
        'kernels': kernels,
        'dense_units': dense,
        'mul_map_file': mul_map
    }

def build_model(arch):
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

def train_and_evaluate(arch, epochs=10):
    # Train with exact multipliers
    model = build_model(arch)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, validation_split=0.1, epochs=epochs, verbose=0)

    # Save weights
    weights_file = 'temp_weights.h5'
    model.save_weights(weights_file)

    # Evaluate with exact
    exact_score = model.evaluate(x_test, y_test, verbose=0)

    # Evaluate with approximate multipliers if specified
    approx_score = None
    if arch['mul_map_file']:
        approx_model = build_approx_model(arch)
        approx_model.build(input_shape=(None, 32, 32, 3))
        approx_model.load_weights(weights_file)
        approx_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        approx_score = approx_model.evaluate(x_test, y_test, verbose=0)

    os.remove(weights_file)

    return exact_score[1], approx_score[1] if approx_score else None

def random_search(num_trials=10, epochs=10):
    results = []

    for i in range(num_trials):
        arch = sample_architecture()
        print(f"\nTrial {i+1}/{num_trials}")
        print(f"Architecture: {arch}")

        exact_acc, approx_acc = train_and_evaluate(arch, epochs)

        result = {
            'arch': arch,
            'exact_accuracy': exact_acc,
            'approx_accuracy': approx_acc
        }
        results.append(result)

        print(f"Exact accuracy: {exact_acc:.4f}")
        if approx_acc:
            print(f"Approx accuracy: {approx_acc:.4f}")
            print(f"Accuracy drop: {exact_acc - approx_acc:.4f}")

    # Find best
    best = max(results, key=lambda x: x['approx_accuracy'] if x['approx_accuracy'] else x['exact_accuracy'])
    print(f"\n{'='*60}")
    print("Best architecture:")
    print(best)

    return results

if __name__ == '__main__':
    results = random_search(num_trials=5, epochs=5)
