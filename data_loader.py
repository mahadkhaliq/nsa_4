import tensorflow as tf

def load_cifar10():
    """Load and preprocess CIFAR-10 dataset"""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    y_train = y_train.astype('float32')
    y_test = y_test.astype('float32')
    return (x_train, y_train), (x_test, y_test)

def load_mnist():
    """Load and preprocess MNIST dataset"""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255
    y_train = y_train.astype('float32')
    y_test = y_test.astype('float32')
    return (x_train, y_train), (x_test, y_test)

def load_fashion_mnist():
    """Load and preprocess FashionMNIST dataset"""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255
    y_train = y_train.astype('float32')
    y_test = y_test.astype('float32')
    return (x_train, y_train), (x_test, y_test)

def load_imagenette(data_dir='./data/imagenette2-320', img_size=224):
    """Load and preprocess Imagenette dataset

    Imagenette is a subset of 10 easily classified classes from ImageNet.
    Classes: tench, English springer, cassette player, chain saw, church,
             French horn, garbage truck, gas pump, golf ball, parachute

    Dataset size: ~13k training images, ~500 test images per class
    Resolution: 224×224×3 (RGB)

    Args:
        data_dir: Path to imagenette2-320 dataset directory
                  Download from: https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz
        img_size: Target image size (default 224 for ImageNet-style)

    Returns:
        (x_train, y_train), (x_test, y_test): Preprocessed dataset
    """
    import os
    import numpy as np
    from tensorflow.keras.preprocessing import image_dataset_from_directory

    # Check if dataset exists
    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"Imagenette dataset not found at {data_dir}\n"
            f"Please download from: https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz\n"
            f"Extract to: {data_dir}"
        )

    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    # Load datasets using Keras image_dataset_from_directory
    train_dataset = image_dataset_from_directory(
        train_dir,
        labels='inferred',
        label_mode='int',
        image_size=(img_size, img_size),
        batch_size=None,  # Load all at once
        shuffle=False
    )

    val_dataset = image_dataset_from_directory(
        val_dir,
        labels='inferred',
        label_mode='int',
        image_size=(img_size, img_size),
        batch_size=None,
        shuffle=False
    )

    # Convert to numpy arrays and normalize
    x_train = []
    y_train = []
    for image, label in train_dataset:
        x_train.append(image.numpy())
        y_train.append(label.numpy())

    x_test = []
    y_test = []
    for image, label in val_dataset:
        x_test.append(image.numpy())
        y_test.append(label.numpy())

    x_train = np.array(x_train).astype('float32') / 255.0
    y_train = np.array(y_train).astype('float32')
    x_test = np.array(x_test).astype('float32') / 255.0
    y_test = np.array(y_test).astype('float32')

    print(f"Imagenette loaded: Train={x_train.shape}, Test={x_test.shape}")

    return (x_train, y_train), (x_test, y_test)

def load_dataset(dataset_name):
    """Load dataset by name"""
    if dataset_name == 'cifar10':
        return load_cifar10()
    elif dataset_name == 'mnist':
        return load_mnist()
    elif dataset_name == 'fashionmnist':
        return load_fashion_mnist()
    elif dataset_name == 'imagenette':
        return load_imagenette()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
