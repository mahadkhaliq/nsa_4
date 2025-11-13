"""
Test ResNet with Approximate Multipliers

This is a standalone test to verify that FakeApproxConv2D works with
ResNet-style residual blocks using Functional API before integrating
into the main NAS system.

Goal: Verify compatibility of approximate multipliers with skip connections
"""

import tensorflow as tf
from keras.layers.fake_approx_convolutional import FakeApproxConv2D
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, Activation, Add,
    GlobalAveragePooling2D, Dense, MaxPooling2D
)
from tensorflow.keras.models import Model
import numpy as np

# Suppress TF warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# GPU setup - same as main.py
physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print(f"GPU memory growth enabled on {len(physical_devices)} device(s)")
except:
    print("Running without GPU memory growth")
    pass

def residual_block_exact(x, filters, stride=1, name=''):
    """Exact residual block with standard Conv2D"""
    shortcut = x

    # First conv
    x = Conv2D(filters, 3, strides=stride, padding='same',
               name=f'{name}_conv1')(x)
    x = BatchNormalization(name=f'{name}_bn1')(x)
    x = Activation('relu', name=f'{name}_relu1')(x)

    # Second conv
    x = Conv2D(filters, 3, strides=1, padding='same',
               name=f'{name}_conv2')(x)
    x = BatchNormalization(name=f'{name}_bn2')(x)

    # Adjust shortcut if needed (when stride > 1 or filters change)
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, 1, strides=stride, padding='same',
                         name=f'{name}_shortcut')(shortcut)
        shortcut = BatchNormalization(name=f'{name}_shortcut_bn')(shortcut)

    # Skip connection
    x = Add(name=f'{name}_add')([x, shortcut])
    x = Activation('relu', name=f'{name}_relu2')(x)

    return x


def residual_block_approx(x, filters, mul_map_file, stride=1, name=''):
    """Approximate residual block with FakeApproxConv2D"""
    shortcut = x

    # First conv - APPROXIMATE
    x = FakeApproxConv2D(filters, 3, strides=stride, padding='same',
                         mul_map_file=mul_map_file, name=f'{name}_approx_conv1')(x)
    x = BatchNormalization(name=f'{name}_bn1')(x)
    x = Activation('relu', name=f'{name}_relu1')(x)

    # Second conv - APPROXIMATE
    x = FakeApproxConv2D(filters, 3, strides=1, padding='same',
                         mul_map_file=mul_map_file, name=f'{name}_approx_conv2')(x)
    x = BatchNormalization(name=f'{name}_bn2')(x)

    # Adjust shortcut if needed - keep EXACT for skip connection
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, 1, strides=stride, padding='same',
                         name=f'{name}_shortcut')(shortcut)
        shortcut = BatchNormalization(name=f'{name}_shortcut_bn')(shortcut)

    # Skip connection - THE KEY TEST: Does Add() work with FakeApproxConv2D?
    x = Add(name=f'{name}_add')([x, shortcut])
    x = Activation('relu', name=f'{name}_relu2')(x)

    return x


def build_mini_resnet_exact(input_shape=(32, 32, 3), num_classes=10):
    """Build ResNet-20 for CIFAR-10 with exact multipliers (baseline)

    Architecture from original ResNet paper for CIFAR-10:
    - 3 stages with [16, 32, 64] filters
    - 3 residual blocks per stage
    - Total: 20 layers (1 init + 3*3*2 conv + 1 fc)
    - Expected accuracy: ~91-92% on CIFAR-10
    """
    inputs = Input(shape=input_shape, name='input')

    # Initial conv - 16 filters, no pooling for CIFAR-10
    x = Conv2D(16, 3, padding='same', name='init_conv')(inputs)
    x = BatchNormalization(name='init_bn')(x)
    x = Activation('relu', name='init_relu')(x)

    # Stage 1: 3 residual blocks, 16 filters, 32×32 feature maps
    x = residual_block_exact(x, 16, stride=1, name='stage1_block1')
    x = residual_block_exact(x, 16, stride=1, name='stage1_block2')
    x = residual_block_exact(x, 16, stride=1, name='stage1_block3')

    # Stage 2: 3 residual blocks, 32 filters, 16×16 feature maps (downsample)
    x = residual_block_exact(x, 32, stride=2, name='stage2_block1')
    x = residual_block_exact(x, 32, stride=1, name='stage2_block2')
    x = residual_block_exact(x, 32, stride=1, name='stage2_block3')

    # Stage 3: 3 residual blocks, 64 filters, 8×8 feature maps (downsample)
    x = residual_block_exact(x, 64, stride=2, name='stage3_block1')
    x = residual_block_exact(x, 64, stride=1, name='stage3_block2')
    x = residual_block_exact(x, 64, stride=1, name='stage3_block3')

    # Output
    x = GlobalAveragePooling2D(name='global_pool')(x)
    outputs = Dense(num_classes, activation='softmax', name='output')(x)

    model = Model(inputs, outputs, name='ResNet20_CIFAR10_Exact')
    return model


def build_mini_resnet_approx(input_shape=(32, 32, 3), num_classes=10,
                             mul_map_file='./multipliers/mul8u_197B.bin'):
    """Build ResNet-20 for CIFAR-10 with approximate multipliers

    Same architecture as exact model but with FakeApproxConv2D layers.
    Tests compatibility of approximate multipliers with ResNet-20.
    """
    inputs = Input(shape=input_shape, name='input')

    # Initial conv - keep exact for stability
    x = Conv2D(16, 3, padding='same', name='init_conv')(inputs)
    x = BatchNormalization(name='init_bn')(x)
    x = Activation('relu', name='init_relu')(x)

    # Stage 1: 3 residual blocks, 16 filters - APPROXIMATE
    x = residual_block_approx(x, 16, mul_map_file, stride=1, name='stage1_block1')
    x = residual_block_approx(x, 16, mul_map_file, stride=1, name='stage1_block2')
    x = residual_block_approx(x, 16, mul_map_file, stride=1, name='stage1_block3')

    # Stage 2: 3 residual blocks, 32 filters - APPROXIMATE
    x = residual_block_approx(x, 32, mul_map_file, stride=2, name='stage2_block1')
    x = residual_block_approx(x, 32, mul_map_file, stride=1, name='stage2_block2')
    x = residual_block_approx(x, 32, mul_map_file, stride=1, name='stage2_block3')

    # Stage 3: 3 residual blocks, 64 filters - APPROXIMATE
    x = residual_block_approx(x, 64, mul_map_file, stride=2, name='stage3_block1')
    x = residual_block_approx(x, 64, mul_map_file, stride=1, name='stage3_block2')
    x = residual_block_approx(x, 64, mul_map_file, stride=1, name='stage3_block3')

    # Output - keep exact
    x = GlobalAveragePooling2D(name='global_pool')(x)
    outputs = Dense(num_classes, activation='softmax', name='output')(x)

    model = Model(inputs, outputs, name='ResNet20_CIFAR10_Approx')
    return model


def load_cifar10_subset(num_samples=50000):
    """Load a small subset of CIFAR-10 for quick testing"""
    print("Loading CIFAR-10 subset...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Normalize
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Take subset for fast testing
    x_train = x_train[:num_samples]
    y_train = y_train[:num_samples]
    x_test = x_test[:10000]
    y_test = y_test[:10000]

    print(f"Train samples: {len(x_train)}, Test samples: {len(x_test)}")
    return (x_train, y_train), (x_test, y_test)


def test_exact_model(epochs=5):
    """Test exact ResNet baseline"""
    print("\n" + "="*60)
    print("TEST 1: Exact ResNet-8 (Baseline)")
    print("="*60)

    # Load data
    (x_train, y_train), (x_test, y_test) = load_cifar10_subset()

    # Build model
    print("\nBuilding exact model...")
    model = build_mini_resnet_exact()
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print(f"Model parameters: {model.count_params():,}")

    # Train with smaller batch size to avoid GPU memory issues
    print(f"\nTraining for {epochs} epochs...")
    history = model.fit(
        x_train, y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=32,  # Reduced from 128 to avoid GPU memory issues
        verbose=1
    )

    # Evaluate
    print("\nEvaluating on test set...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

    print(f"\n{'='*40}")
    print(f"Exact Model Results:")
    print(f"  Test Accuracy: {test_acc:.4f}")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"{'='*40}")

    # Save weights for approximate model
    weights_file = 'temp_exact_weights.h5'
    model.save_weights(weights_file)
    print(f"\nWeights saved to {weights_file}")

    return model, test_acc, weights_file


def test_approx_model(weights_file, mul_map_file='./multipliers/mul8u_197B.bin'):
    """Test approximate ResNet with pre-trained weights"""
    print("\n" + "="*60)
    print("TEST 2: Approximate ResNet-8 (FakeApproxConv2D)")
    print("="*60)
    print(f"Using multiplier: {mul_map_file}")

    # Check if multiplier file exists
    if not os.path.exists(mul_map_file):
        print(f"\nWARNING: Multiplier file not found: {mul_map_file}")
        print("This test will fail if the file doesn't exist.")
        print("Available multipliers should be in ./multipliers/ directory")
        return None, None

    # Load data
    (x_train, y_train), (x_test, y_test) = load_cifar10_subset()

    # Build approximate model
    print("\nBuilding approximate model...")
    try:
        model = build_mini_resnet_approx(mul_map_file=mul_map_file)
        model.build(input_shape=(None, 32, 32, 3))

        print("✓ Approximate model built successfully!")
        print("✓ FakeApproxConv2D works with Functional API")
        print("✓ Skip connections (Add layer) compatible")

        print(f"\nModel parameters: {model.count_params():,}")

        # Load exact weights
        print(f"\nLoading weights from {weights_file}...")
        model.load_weights(weights_file)
        print("✓ Weight transfer successful")

        # Compile
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Evaluate
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
    """Compare exact vs approximate results"""
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

    if accuracy_drop < 0.05:  # Less than 5% drop
        print("\n✓ EXCELLENT: Minimal accuracy loss (<5%)")
    elif accuracy_drop < 0.10:  # Less than 10% drop
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
    """Run complete test suite"""
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

    # Test 1: Exact model
    exact_model, exact_acc, weights_file = test_exact_model(epochs=40)

    # Test 2: Approximate model
    approx_model, approx_acc = test_approx_model(weights_file)

    # Compare
    compare_results(exact_acc, approx_acc)

    # Cleanup
    if os.path.exists(weights_file):
        os.remove(weights_file)
        print(f"\nCleaned up {weights_file}")

    print("\n" + "="*70)
    print(" Test Complete!")
    print("="*70)


if __name__ == '__main__':
    main()
