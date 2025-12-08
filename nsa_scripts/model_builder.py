import tensorflow as tf
from keras.layers.fake_approx_convolutional import FakeApproxConv2D
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, Activation, Add,
    GlobalAveragePooling2D, Dense
)
from tensorflow.keras.models import Model


def build_model(arch):
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
    layers = []

    use_batch_norm = arch.get('use_batch_norm', False)

    for i in range(arch['num_conv_layers']):
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


def build_resnet20_exact(arch=None, input_shape=(32, 32, 3), num_classes=10):
    if arch is None:
        num_stages = 3
        blocks_per_stage = [3, 3, 3]
        filters_per_stage = [16, 32, 64]
    else:
        num_stages = arch['num_stages']
        blocks_per_stage_raw = arch['blocks_per_stage']
        if isinstance(blocks_per_stage_raw, int):
            blocks_per_stage = [blocks_per_stage_raw] * num_stages
        else:
            blocks_per_stage = blocks_per_stage_raw
        filters_per_stage = arch['filters_per_stage']

    inputs = Input(shape=input_shape, name='input')

    x = Conv2D(filters_per_stage[0], 3, padding='same', name='init_conv')(inputs)
    x = BatchNormalization(name='init_bn')(x)
    x = Activation('relu', name='init_relu')(x)

    total_conv_layers = 0
    for stage_idx in range(num_stages):
        filters = filters_per_stage[stage_idx]
        num_blocks = blocks_per_stage[stage_idx]
        stride = 2 if stage_idx > 0 else 1

        for block_idx in range(num_blocks):
            block_stride = stride if block_idx == 0 else 1
            x = residual_block_exact(
                x, filters, stride=block_stride,
                name=f'stage{stage_idx+1}_block{block_idx+1}'
            )
            total_conv_layers += 2

    x = GlobalAveragePooling2D(name='global_pool')(x)
    outputs = Dense(num_classes, activation='softmax', name='output')(x)

    total_layers = 1 + total_conv_layers + 1
    model = Model(inputs, outputs, name=f'ResNet{total_layers}_CIFAR10_Exact')
    return model


def build_resnet20_approx(arch, input_shape=(32, 32, 3), num_classes=10):
    num_stages = arch['num_stages']
    blocks_per_stage_raw = arch['blocks_per_stage']
    if isinstance(blocks_per_stage_raw, int):
        blocks_per_stage = [blocks_per_stage_raw] * num_stages
    else:
        blocks_per_stage = blocks_per_stage_raw
    filters_per_stage = arch['filters_per_stage']
    mul_map_files = arch['mul_map_files']

    inputs = Input(shape=input_shape, name='input')

    x = Conv2D(filters_per_stage[0], 3, padding='same', name='init_conv')(inputs)
    x = BatchNormalization(name='init_bn')(x)
    x = Activation('relu', name='init_relu')(x)

    total_conv_layers = 0
    for stage_idx in range(num_stages):
        filters = filters_per_stage[stage_idx]
        mul_map = mul_map_files[stage_idx]
        num_blocks = blocks_per_stage[stage_idx]
        stride = 2 if stage_idx > 0 else 1

        for block_idx in range(num_blocks):
            block_stride = stride if block_idx == 0 else 1
            x = residual_block_approx(
                x, filters, mul_map, stride=block_stride,
                name=f'stage{stage_idx+1}_block{block_idx+1}'
            )
            total_conv_layers += 2

    x = GlobalAveragePooling2D(name='global_pool')(x)
    outputs = Dense(num_classes, activation='softmax', name='output')(x)

    total_layers = 1 + total_conv_layers + 1
    model = Model(inputs, outputs, name=f'ResNet{total_layers}_CIFAR10_Approx')
    return model
