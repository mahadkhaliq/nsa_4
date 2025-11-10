import tensorflow as tf
from nas_search import random_search, grid_search
from evaluator import train_and_evaluate
from data_loader import load_dataset

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
        MUL_MAP_PATH + 'mul8u_125K.bin',
        MUL_MAP_PATH + 'mul8u_1JFF.bin',
        MUL_MAP_PATH + 'mul8u_2AC.bin',
        MUL_MAP_PATH + 'mul8u_17C8.bin'
    ]
}

def run_nas(search_algo='random', num_trials=5, epochs=5, use_stl=False):
    """Run NAS with specified search algorithm"""

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

        result = train_and_evaluate(arch, x_train, y_train, x_test, y_test, epochs, use_stl)
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
            print(f"STL robustness: {result['stl_robustness']:.4f}")

    # Find best
    best = max(results, key=lambda x: x['approx_accuracy'] if x['approx_accuracy'] else x['exact_accuracy'])
    print(f"\n{'='*60}")
    print("Best architecture:")
    print(best)

    return results

if __name__ == '__main__':
    # Run with STL monitoring
    results = run_nas(search_algo='random', num_trials=5, epochs=5, use_stl=True)
