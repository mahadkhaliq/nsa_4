import tensorflow as tf
from keras.layers.fake_approx_convolutional import FakeApproxConv2D
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, Activation, Add,
    GlobalAveragePooling2D, Dense
)
from tensorflow.keras.models import Model

# ============================================================================
# Simple CNN Architecture (Original)
# ============================================================================

def build_model(arch):
    """Build CNN model with exact Conv2D layers"""
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
    """Build CNN model with approximate Conv2D layers"""
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


# ============================================================================
# ResNet-20 Architecture (from approxAI paper)
# ============================================================================

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

    # Adjust shortcut if needed
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

    # Adjust shortcut if needed - keep EXACT for stability
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, 1, strides=stride, padding='same',
                         name=f'{name}_shortcut')(shortcut)
        shortcut = BatchNormalization(name=f'{name}_shortcut_bn')(shortcut)

    # Skip connection
    x = Add(name=f'{name}_add')([x, shortcut])
    x = Activation('relu', name=f'{name}_relu2')(x)

    return x


def build_resnet20_exact(input_shape=(32, 32, 3), num_classes=10):
    """Build ResNet-20 for CIFAR-10 with exact multipliers

    Architecture from approxAI paper and original ResNet paper:
    - 3 stages with [16, 32, 64] filters
    - 3 residual blocks per stage
    - Total: 20 layers (1 init + 3*3*2 conv + 1 fc)
    - Expected accuracy: ~91-92% on CIFAR-10
    """
    inputs = Input(shape=input_shape, name='input')

    # Initial conv - 16 filters
    x = Conv2D(16, 3, padding='same', name='init_conv')(inputs)
    x = BatchNormalization(name='init_bn')(x)
    x = Activation('relu', name='init_relu')(x)

    # Stage 1: 16 filters, 32×32 feature maps
    x = residual_block_exact(x, 16, stride=1, name='stage1_block1')
    x = residual_block_exact(x, 16, stride=1, name='stage1_block2')
    x = residual_block_exact(x, 16, stride=1, name='stage1_block3')

    # Stage 2: 32 filters, 16×16 feature maps
    x = residual_block_exact(x, 32, stride=2, name='stage2_block1')
    x = residual_block_exact(x, 32, stride=1, name='stage2_block2')
    x = residual_block_exact(x, 32, stride=1, name='stage2_block3')

    # Stage 3: 64 filters, 8×8 feature maps
    x = residual_block_exact(x, 64, stride=2, name='stage3_block1')
    x = residual_block_exact(x, 64, stride=1, name='stage3_block2')
    x = residual_block_exact(x, 64, stride=1, name='stage3_block3')

    # Output
    x = GlobalAveragePooling2D(name='global_pool')(x)
    outputs = Dense(num_classes, activation='softmax', name='output')(x)

    model = Model(inputs, outputs, name='ResNet20_CIFAR10_Exact')
    return model


def build_resnet20_approx(mul_map_files, input_shape=(32, 32, 3), num_classes=10):
    """Build ResNet-20 for CIFAR-10 with approximate multipliers

    Uses heterogeneous multipliers (different per stage) as in approxAI paper.

    Args:
        mul_map_files: List of 3 multiplier files for [stage1, stage2, stage3]
                      If fewer than 3, reuses the first multiplier
    """
    inputs = Input(shape=input_shape, name='input')

    # Initial conv - keep exact for stability
    x = Conv2D(16, 3, padding='same', name='init_conv')(inputs)
    x = BatchNormalization(name='init_bn')(x)
    x = Activation('relu', name='init_relu')(x)

    # Stage 1: 16 filters - use mul_map_files[0]
    mul_stage1 = mul_map_files[0] if len(mul_map_files) > 0 else mul_map_files[0]
    x = residual_block_approx(x, 16, mul_stage1, stride=1, name='stage1_block1')
    x = residual_block_approx(x, 16, mul_stage1, stride=1, name='stage1_block2')
    x = residual_block_approx(x, 16, mul_stage1, stride=1, name='stage1_block3')

    # Stage 2: 32 filters - use mul_map_files[1]
    mul_stage2 = mul_map_files[1] if len(mul_map_files) > 1 else mul_map_files[0]
    x = residual_block_approx(x, 32, mul_stage2, stride=2, name='stage2_block1')
    x = residual_block_approx(x, 32, mul_stage2, stride=1, name='stage2_block2')
    x = residual_block_approx(x, 32, mul_stage2, stride=1, name='stage2_block3')

    # Stage 3: 64 filters - use mul_map_files[2]
    mul_stage3 = mul_map_files[2] if len(mul_map_files) > 2 else mul_map_files[0]
    x = residual_block_approx(x, 64, mul_stage3, stride=2, name='stage3_block1')
    x = residual_block_approx(x, 64, mul_stage3, stride=1, name='stage3_block2')
    x = residual_block_approx(x, 64, mul_stage3, stride=1, name='stage3_block3')

    # Output - keep exact
    x = GlobalAveragePooling2D(name='global_pool')(x)
    outputs = Dense(num_classes, activation='softmax', name='output')(x)

    model = Model(inputs, outputs, name='ResNet20_CIFAR10_Approx')
    return model
