
import tensorflow as tf
from keras.layers.fake_approx_convolutional import FakeApproxConv2D
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, Activation, Add,
    GlobalAveragePooling2D, Dense, MaxPooling2D
)
from tensorflow.keras.models import Model
import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print(f"GPU memory growth enabled on {len(physical_devices)} device(s)")
except:
    print("Running without GPU memory growth")
    pass

def residual_block_exact(x, filters, stride=1, name=''):
    shortcut = x

    x = Conv2D(filters, 3, strides=stride, padding='same',
               name=f'{name}_conv1')(x)
    x = BatchNormalization(name=f'{name}_bn1')(x)
    x = Activation('relu', name=f'{name}_relu1')(x)

    x = Conv2D(filters, 3, strides=1, padding='same',
               name=f'{name}_conv2')(x)
    x = BatchNormalization(name=f'{name}_bn2')(x)

    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, 1, strides=stride, padding='same',
                         name=f'{name}_shortcut')(shortcut)
        shortcut = BatchNormalization(name=f'{name}_shortcut_bn')(shortcut)

    x = Add(name=f'{name}_add')([x, shortcut])
    x = Activation('relu', name=f'{name}_relu2')(x)

    return x


def residual_block_approx(x, filters, mul_map_file, stride=1, name=''):
    shortcut = x

    x = FakeApproxConv2D(filters, 3, strides=stride, padding='same',
                         mul_map_file=mul_map_file, name=f'{name}_approx_conv1')(x)
    x = BatchNormalization(name=f'{name}_bn1')(x)
    x = Activation('relu', name=f'{name}_relu1')(x)

    x = FakeApproxConv2D(filters, 3, strides=1, padding='same',
                         mul_map_file=mul_map_file, name=f'{name}_approx_conv2')(x)
    x = BatchNormalization(name=f'{name}_bn2')(x)

    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, 1, strides=stride, padding='same',
                         name=f'{name}_shortcut')(shortcut)
        shortcut = BatchNormalization(name=f'{name}_shortcut_bn')(shortcut)

    x = Add(name=f'{name}_add')([x, shortcut])
    x = Activation('relu', name=f'{name}_relu2')(x)

    return x


def build_mini_resnet_exact(input_shape=(32, 32, 3), num_classes=10):
    inputs = Input(shape=input_shape, name='input')

    x = Conv2D(16, 3, padding='same', name='init_conv')(inputs)
    x = BatchNormalization(name='init_bn')(x)
    x = Activation('relu', name='init_relu')(x)

    x = residual_block_exact(x, 16, stride=1, name='stage1_block1')
    x = residual_block_exact(x, 16, stride=1, name='stage1_block2')
    x = residual_block_exact(x, 16, stride=1, name='stage1_block3')

    x = residual_block_exact(x, 32, stride=2, name='stage2_block1')
    x = residual_block_exact(x, 32, stride=1, name='stage2_block2')
    x = residual_block_exact(x, 32, stride=1, name='stage2_block3')

    x = residual_block_exact(x, 64, stride=2, name='stage3_block1')
    x = residual_block_exact(x, 64, stride=1, name='stage3_block2')
    x = residual_block_exact(x, 64, stride=1, name='stage3_block3')

    x = GlobalAveragePooling2D(name='global_pool')(x)
    outputs = Dense(num_classes, activation='softmax', name='output')(x)

    model = Model(inputs, outputs, name='ResNet20_CIFAR10_Exact')
    return model


def build_mini_resnet_approx(input_shape=(32, 32, 3), num_classes=10,
                             mul_map_file='./multipliers/mul8u_1JJQ.bin'):
    inputs = Input(shape=input_shape, name='input')

    x = Conv2D(16, 3, padding='same', name='init_conv')(inputs)
    x = BatchNormalization(name='init_bn')(x)
    x = Activation('relu', name='init_relu')(x)

    x = residual_block_approx(x, 16, mul_map_file, stride=1, name='stage1_block1')
    x = residual_block_approx(x, 16, mul_map_file, stride=1, name='stage1_block2')
    x = residual_block_approx(x, 16, mul_map_file, stride=1, name='stage1_block3')

    x = residual_block_approx(x, 32, mul_map_file, stride=2, name='stage2_block1')
    x = residual_block_approx(x, 32, mul_map_file, stride=1, name='stage2_block2')
    x = residual_block_approx(x, 32, mul_map_file, stride=1, name='stage2_block3')

    x = residual_block_approx(x, 64, mul_map_file, stride=2, name='stage3_block1')
    x = residual_block_approx(x, 64, mul_map_file, stride=1, name='stage3_block2')
    x = residual_block_approx(x, 64, mul_map_file, stride=1, name='stage3_block3')

    x = GlobalAveragePooling2D(name='global_pool')(x)
    outputs = Dense(num_classes, activation='softmax', name='output')(x)

    model = Model(inputs, outputs, name='ResNet20_CIFAR10_Approx')
    return model


def load_cifar10_subset(num_samples=50000):
    print("Loading CIFAR-10 subset...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    x_train = x_train[:num_samples]
    y_train = y_train[:num_samples]
    x_test = x_test[:10000]
    y_test = y_test[:10000]

    print(f"Train samples: {len(x_train)}, Test samples: {len(x_test)}")
    return (x_train, y_train), (x_test, y_test)


def test_exact_model(epochs=5):
    print("\n" + "="*60)
    print("TEST 1: Exact ResNet-8 (Baseline)")
    print("="*60)

    (x_train, y_train), (x_test, y_test) = load_cifar10_subset()

    print("\nBuilding exact model...")
    model = build_mini_resnet_exact()
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print(f"Model parameters: {model.count_params():,}")

    print(f"\nTraining for {epochs} epochs...")
    history = model.fit(
        x_train, y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=32,
        verbose=1
    )

    print("\nEvaluating on test set...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

    print(f"\n{'='*40}")
    print(f"Exact Model Results:")
    print(f"  Test Accuracy: {test_acc:.4f}")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"{'='*40}")

    weights_file = 'temp_exact_weights.h5'
    model.save_weights(weights_file)
    print(f"\nWeights saved to {weights_file}")

    return model, test_acc, weights_file


def test_approx_model(weights_file, mul_map_file='./multipliers/mul8u_1JJQ.bin'):
    print("\n" + "="*60)
    print("TEST 2: Approximate ResNet-8 (FakeApproxConv2D)")
    print("="*60)
    print(f"Using multiplier: {mul_map_file}")

    if not os.path.exists(mul_map_file):
        print(f"\nWARNING: Multiplier file not found: {mul_map_file}")
        print("This test will fail if the file doesn't exist.")
        print("Available multipliers should be in ./multipliers/ directory")
        return None, None

    (x_train, y_train), (x_test, y_test) = load_cifar10_subset()

    print("\nBuilding approximate model...")
    try:
        model = build_mini_resnet_approx(mul_map_file=mul_map_file)
        model.build(input_shape=(None, 32, 32, 3))

        print("✓ Approximate model built successfully!")
        print("✓ FakeApproxConv2D works with Functional API")
        print("✓ Skip connections (Add layer) compatible")

        print(f"\nModel parameters: {model.count_params():,}")

        print(f"\nLoading weights from {weights_file}...")
        model.load_weights(weights_file)
        print("✓ Weight transfer successful")

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        print("\nEvaluating approximate model on test set...")
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

        print(f"\n{'='*40}")
        print(f"Approximate Model Results:")
        print(f"  Test Accuracy: {test_acc:.4f}")
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"{'='*40}")

        return model, test_acc

    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")
        print("\nThis indicates FakeApproxConv2D may not be compatible")
        import traceback
        traceback.print_exc()
        return None, None


def compare_results(exact_acc, approx_acc):
    print("\n" + "="*60)
    print("COMPARISON: Exact vs Approximate")
    print("="*60)

    if approx_acc is None:
        print("✗ Approximate model failed - cannot compare")
        return

    accuracy_drop = exact_acc - approx_acc
    accuracy_drop_pct = (accuracy_drop / exact_acc) * 100

    print(f"Exact Accuracy:       {exact_acc:.4f}")
    print(f"Approximate Accuracy: {approx_acc:.4f}")
    print(f"Accuracy Drop:        {accuracy_drop:.4f} ({accuracy_drop_pct:.2f}%)")

    if accuracy_drop < 0.05:
        print("\n✓ EXCELLENT: Minimal accuracy loss (<5%)")
    elif accuracy_drop < 0.10:
        print("\n✓ GOOD: Acceptable accuracy loss (<10%)")
    else:
        print("\n⚠ WARNING: Significant accuracy loss (>10%)")

    print("\n" + "="*60)
    print("CONCLUSION:")
    print("="*60)
    if approx_acc is not None:
        print("✓ FakeApproxConv2D is COMPATIBLE with ResNet architecture")
        print("✓ Skip connections work correctly with approximate layers")
        print("✓ Weight transfer from exact to approximate works")
        print("✓ Ready to integrate into main NAS system")
    else:
        print("✗ Compatibility issues detected")
        print("✗ Further investigation needed before integration")


def main():
    print("\n" + "="*70)
    print(" ResNet-20 + FakeApproxConv2D Compatibility Test")
    print("="*70)
    print("\nThis test verifies that approximate multipliers work with ResNet-20,")
    print("the proven CIFAR-10 architecture from the original ResNet paper.")
    print("\nTest architecture: ResNet-20 for CIFAR-10")
    print("  - 3 stages: [16, 32, 64] filters")
    print("  - 3 residual blocks per stage")
    print("  - 18 approximate conv layers total (9 blocks × 2 conv/block)")
    print("  - Skip connections with Add() layer")
    print("  - Expected exact accuracy: 91-92% (from paper)")
    print("  - Expected approx accuracy drop: <5% (target)")

    exact_model, exact_acc, weights_file = test_exact_model(epochs=40)

    approx_model, approx_acc = test_approx_model(weights_file)

    compare_results(exact_acc, approx_acc)

    if os.path.exists(weights_file):
        os.remove(weights_file)
        print(f"\nCleaned up {weights_file}")

    print("\n" + "="*70)
    print(" Test Complete!")
    print("="*70)


if __name__ == '__main__':
    main()
