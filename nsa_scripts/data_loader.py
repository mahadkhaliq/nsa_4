import tensorflow as tf

def load_cifar10():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    y_train = y_train.astype('float32')
    y_test = y_test.astype('float32')
    return (x_train, y_train), (x_test, y_test)

def load_mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255
    y_train = y_train.astype('float32')
    y_test = y_test.astype('float32')
    return (x_train, y_train), (x_test, y_test)

def load_fashion_mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255
    y_train = y_train.astype('float32')
    y_test = y_test.astype('float32')
    return (x_train, y_train), (x_test, y_test)

def load_dataset(dataset_name):
    if dataset_name == 'cifar10':
        return load_cifar10()
    elif dataset_name == 'mnist':
        return load_mnist()
    elif dataset_name == 'fashionmnist':
        return load_fashion_mnist()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
