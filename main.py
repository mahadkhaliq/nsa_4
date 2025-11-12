import tensorflow as tf
from nas_search import random_search, grid_search
from evaluator import train_and_evaluate
from data_loader import load_dataset
from stl_monitor import check_pareto_optimal

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

# Load dataset
(x_train, y_train), (x_test, y_test) = load_dataset('cifar10')

# Search space
MUL_MAP_PATH = './multipliers/'
SEARCH_SPACE = {
    'num_conv_layers': [2, 3, 4],
    'filters': [16, 32, 64, 128],
    'kernel_sizes': [3, 5],
    'dense_units': [64, 128, 256],
    'mul_map_files': [
        MUL_MAP_PATH + 'mul8u_17C8.bin',   # 0.104 mW - low power
        MUL_MAP_PATH + 'mul8u_197B.bin',   # 0.206 mW - medium
        MUL_MAP_PATH + 'mul8u_0AB.bin',    # 0.302 mW - medium-high
        MUL_MAP_PATH + 'mul8u_1JJQ.bin',   # 0.391 mW - higher accuracy
    ]
}

def run_nas(search_algo='random', num_trials=5, epochs=5, use_stl=False,
            quality_constraint=0.70, energy_constraint=50.0):
    """Run NAS with specified search algorithm

    Args:
        search_algo: 'random' or 'grid'
        num_trials: Number of architectures to evaluate
        epochs: Training epochs per architecture
        use_stl: Enable STL monitoring (approxAI constraints)
        quality_constraint: Qc - minimum accuracy threshold (approxAI)
        energy_constraint: Ec - maximum energy in mJ (approxAI)
    """

    # Get architectures to evaluate
    if search_algo == 'random':
        architectures = random_search(SEARCH_SPACE, num_trials)
    elif search_algo == 'grid':
        architectures = grid_search(SEARCH_SPACE, max_trials=num_trials)
    else:
        raise ValueError(f"Unknown search algorithm: {search_algo}")

    results = []

    for i, arch in enumerate(architectures):
        print(f"\nTrial {i+1}/{len(architectures)}")
        print(f"Architecture: {arch}")

        result = train_and_evaluate(arch, x_train, y_train, x_test, y_test, epochs,
                                   use_stl, quality_constraint, energy_constraint)
        result['arch'] = arch
        results.append(result)

        print(f"Exact accuracy: {result['exact_accuracy']:.4f}")
        if result['approx_accuracy']:
            print(f"Approx accuracy: {result['approx_accuracy']:.4f}")
            print(f"Accuracy drop: {result['exact_accuracy'] - result['approx_accuracy']:.4f}")
        if result['energy']:
            print(f"Total energy: {result['energy']:.4f} mJ")
        if result['energy_per_layer']:
            print("Energy per layer:")
            for layer_info in result['energy_per_layer']:
                print(f"  Layer {layer_info['layer']}: {layer_info['multiplier']} - {layer_info['energy']:.4f} mJ (power: {layer_info['power']:.3f} mW)")
        if result['stl_robustness'] is not None:
            status = "SATISFIED" if result['stl_robustness'] > 0 else "VIOLATED"
            print(f"STL robustness: {result['stl_robustness']:.4f} ({status})")

    # Find best by accuracy
    best = max(results, key=lambda x: x['approx_accuracy'] if x['approx_accuracy'] else x['exact_accuracy'])

    # Find Pareto-optimal solutions (approxAI methodology)
    pareto_indices = check_pareto_optimal(results)

    print(f"\n{'='*60}")
    print("Best architecture by accuracy:")
    print(f"  Accuracy: {best['approx_accuracy']:.4f}, Energy: {best['energy']:.4f} mJ")
    print(f"  Architecture: {best['arch']}")

    if pareto_indices:
        print(f"\n{'='*60}")
        print(f"Pareto-optimal architectures ({len(pareto_indices)} found):")
        print("(approxAI: No architecture dominates these in both accuracy AND energy)")
        for idx in pareto_indices:
            r = results[idx]
            print(f"\n  Trial {idx+1}:")
            print(f"    Accuracy: {r['approx_accuracy']:.4f}, Energy: {r['energy']:.4f} mJ")
            if r['stl_robustness'] is not None:
                status = "SATISFIED" if r['stl_robustness'] > 0 else "VIOLATED"
                print(f"    STL (Qc={quality_constraint}, Ec={energy_constraint}): {status}")

    return results

if __name__ == '__main__':
    # Run with STL monitoring using approxAI constraints
    # Qc = 0.70 (70% minimum accuracy)
    # Ec = 50.0 mJ (maximum energy)
    results = run_nas(search_algo='random', num_trials=30, epochs=20, use_stl=True,
                     quality_constraint=0.70, energy_constraint=50.0)
