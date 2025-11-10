import os
from model_builder import build_model, build_approx_model
from energy_calculator import estimate_network_energy
from stl_monitor import evaluate_stl

def train_and_evaluate(arch, x_train, y_train, x_test, y_test, epochs=10, use_stl=False):
    """Train with exact multipliers, evaluate with both exact and approximate"""

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
    energy = None
    stl_robustness = None

    if arch['mul_map_file']:
        approx_model = build_approx_model(arch)
        approx_model.build(input_shape=(None, 32, 32, 3))
        approx_model.load_weights(weights_file)
        approx_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        approx_score = approx_model.evaluate(x_test, y_test, verbose=0)

        # Calculate energy
        energy = estimate_network_energy(arch)

        # Evaluate STL if requested
        if use_stl:
            stl_robustness = evaluate_stl(approx_score[1], energy)

    os.remove(weights_file)

    return {
        'exact_accuracy': exact_score[1],
        'approx_accuracy': approx_score[1] if approx_score else None,
        'energy': energy,
        'stl_robustness': stl_robustness
    }
