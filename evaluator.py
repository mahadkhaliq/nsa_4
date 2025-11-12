import os
from model_builder import build_model, build_approx_model
from energy_calculator import estimate_network_energy
from stl_monitor import evaluate_stl

def train_and_evaluate(arch, x_train, y_train, x_test, y_test, epochs=10, use_stl=False,
                      quality_constraint=0.70, energy_constraint=50.0):
    """Train with exact multipliers, evaluate with both exact and approximate

    Args:
        arch: Architecture dictionary
        x_train, y_train: Training data
        x_test, y_test: Test data
        epochs: Training epochs
        use_stl: Enable STL monitoring with approxAI constraints
        quality_constraint: Qc - minimum accuracy (approxAI)
        energy_constraint: Ec - maximum energy in mJ (approxAI)
    """

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
    energy_per_layer = None
    stl_robustness = None

    if arch['mul_map_files']:
        approx_model = build_approx_model(arch)
        approx_model.build(input_shape=(None, 32, 32, 3))
        approx_model.load_weights(weights_file)
        approx_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        approx_score = approx_model.evaluate(x_test, y_test, verbose=0)

        # Calculate energy with per-layer breakdown
        energy, energy_per_layer = estimate_network_energy(arch)

        # Evaluate STL if requested (approxAI constraints: Qc and Ec)
        if use_stl:
            stl_robustness = evaluate_stl(approx_score[1], energy,
                                         quality_constraint, energy_constraint)

    os.remove(weights_file)

    return {
        'exact_accuracy': exact_score[1],
        'approx_accuracy': approx_score[1] if approx_score else None,
        'energy': energy,
        'energy_per_layer': energy_per_layer,
        'stl_robustness': stl_robustness
    }
