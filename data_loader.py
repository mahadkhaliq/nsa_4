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
    from PIL import Image

    # Check if dataset exists
    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"Imagenette dataset not found at {data_dir}\n"
            f"Please download from: https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz\n"
            f"Extract to: {data_dir}"
        )

    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    def load_images_from_directory(directory, img_size):
        """Load all images from directory structure"""
        images = []
        labels = []

        # Get sorted class names (subdirectories)
        class_names = sorted([d for d in os.listdir(directory)
                            if os.path.isdir(os.path.join(directory, d))])

        print(f"Loading from {directory}...")
        print(f"Found {len(class_names)} classes: {class_names}")

        for class_idx, class_name in enumerate(class_names):
            class_dir = os.path.join(directory, class_name)
            image_files = [f for f in os.listdir(class_dir)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            print(f"  Class {class_idx} ({class_name}): {len(image_files)} images")

            for img_file in image_files:
                img_path = os.path.join(class_dir, img_file)
                try:
                    # Load and resize image
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize((img_size, img_size), Image.BILINEAR)
                    img_array = np.array(img, dtype='float32')

                    images.append(img_array)
                    labels.append(class_idx)
                except Exception as e:
                    print(f"    Warning: Failed to load {img_path}: {e}")
                    continue

        return np.array(images), np.array(labels)

    # Load training and validation data
    x_train, y_train = load_images_from_directory(train_dir, img_size)
    x_test, y_test = load_images_from_directory(val_dir, img_size)

    # Normalize to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    y_train = y_train.astype('float32')
    y_test = y_test.astype('float32')

    print(f"\nImagenette loaded successfully!")
    print(f"  Training: {x_train.shape} images, {y_train.shape} labels")
    print(f"  Validation: {x_test.shape} images, {y_test.shape} labels")

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
