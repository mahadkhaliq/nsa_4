import os
import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model_builder import (
    build_model, build_approx_model,
    build_resnet20_exact, build_resnet20_approx
)
from energy_calculator import estimate_network_energy
from stl_monitor import evaluate_stl

def lr_schedule(epoch):
    if epoch < 40:
        return 0.001
    elif epoch < 60:
        return 0.0001
    else:
        return 0.00001

def train_and_evaluate(arch, x_train, y_train, x_test, y_test, epochs=10, use_stl=False,
                      quality_constraint=0.70, energy_constraint=50.0, use_resnet=False,
                      input_shape=(32, 32, 3), batch_size=256):

    if use_resnet:
        model = build_resnet20_exact(arch, input_shape=input_shape)
    else:
        model = build_model(arch)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )

    val_samples = int(len(x_train) * 0.1)
    x_val = x_train[-val_samples:]
    y_val = y_train[-val_samples:]
    x_train_split = x_train[:-val_samples]
    y_train_split = y_train[:-val_samples]

    callbacks = [LearningRateScheduler(lr_schedule)]

    history = model.fit(
        datagen.flow(x_train_split, y_train_split, batch_size=batch_size),
        validation_data=(x_val, y_val),
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
        steps_per_epoch=len(x_train_split) // batch_size,
        workers=4,
        use_multiprocessing=False
    )

    weights_file = 'temp_weights.h5'
    model.save_weights(weights_file)

    exact_score = model.evaluate(x_test, y_test, verbose=0, batch_size=batch_size)

    approx_score = None
    energy = None
    energy_per_layer = None
    stl_robustness = None

    if arch['mul_map_files']:
        if use_resnet:
            approx_model = build_resnet20_approx(arch, input_shape=input_shape)
        else:
            approx_model = build_approx_model(arch)

        approx_model.build(input_shape=(None,) + input_shape)
        approx_model.load_weights(weights_file)
        approx_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        approx_score = approx_model.evaluate(x_test, y_test, verbose=0, batch_size=batch_size)

        input_size = input_shape[0]
        energy, energy_per_layer = estimate_network_energy(arch, input_size=input_size)

        if use_stl:
            stl_robustness = evaluate_stl(approx_score[1], energy,
                                         quality_constraint, energy_constraint)

    os.remove(weights_file)

    return {
        'exact_accuracy': exact_score[1],
        'approx_accuracy': approx_score[1] if approx_score else None,
        'energy': energy,
        'energy_per_layer': energy_per_layer,
        'stl_robustness': stl_robustness,
        'history': history.history
    }
