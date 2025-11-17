import tensorflow as tf
from nas_search import random_search, grid_search
from evaluator import train_and_evaluate
from data_loader import load_dataset
from stl_monitor import check_pareto_optimal
from bayesian_nas import bayesian_search

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

# Load dataset
(x_train, y_train), (x_test, y_test) = load_dataset('cifar10')

# Search space
MUL_MAP_PATH = './multipliers/'

# CNN Search Space (for simple CNN architecture)
SEARCH_SPACE_CNN = {
    'num_conv_layers': [2, 3, 4],
    'filters': [16, 32, 64, 128],
    'kernel_sizes': [3, 5],
    'dense_units': [64, 128, 256],
    'mul_map_files': [
        MUL_MAP_PATH + 'mul8u_197B.bin',   # 0.206 mW - medium balance
        MUL_MAP_PATH + 'mul8u_1JJQ.bin',   # 0.391 mW - best performing
        MUL_MAP_PATH + 'mul8u_0AB.bin',    # 0.302 mW - medium-high
    ],
    'use_batch_norm': [True, False]
}

# ResNet Search Space (NAS searches architecture + multipliers)
SEARCH_SPACE_RESNET = {
    'num_stages': [2, 3],  # 2 or 3 stages
    'blocks_per_stage': [2, 3, 4],  # blocks per stage
    'base_filters': [16, 32],  # starting filters (doubles each stage)
    'mul_map_files': [
        MUL_MAP_PATH + 'mul8u_197B.bin',   # 0.206 mW - medium balance
        MUL_MAP_PATH + 'mul8u_1JJQ.bin',   # 0.391 mW - best performing
        MUL_MAP_PATH + 'mul8u_0AB.bin',    # 0.302 mW - medium-high
    ]
}

def run_nas(search_algo='random', num_trials=5, epochs=5, use_stl=False,
            quality_constraint=0.70, energy_constraint=50.0, architecture='cnn'):
    """Run NAS with specified search algorithm

    Args:
        search_algo: 'random', 'grid', or 'bayesian'
        num_trials: Number of architectures to evaluate
        epochs: Training epochs per architecture
        use_stl: Enable STL monitoring (approxAI constraints)
        quality_constraint: Qc - minimum accuracy threshold (approxAI)
        energy_constraint: Ec - maximum energy in mJ (approxAI)
        architecture: 'cnn' for simple CNN or 'resnet' for ResNet-20
    """

    # Select search space based on architecture
    use_resnet = (architecture.lower() == 'resnet')
    search_space = SEARCH_SPACE_RESNET if use_resnet else SEARCH_SPACE_CNN

    print(f"\n{'='*60}")
    print(f"Architecture: {'ResNet-20' if use_resnet else 'Simple CNN'}")
    print(f"Search algorithm: {search_algo}")
    print(f"Trials: {num_trials}, Epochs: {epochs}")
    print(f"{'='*60}\n")

    # Get architectures to evaluate
    bayes_nas = None
    if search_algo == 'bayesian':
        # Bayesian optimization - more efficient than random search
        objective = 'stl_robustness' if use_stl else 'accuracy'
        architectures, bayes_nas = bayesian_search(search_space, num_trials, objective)
        print(f"Using Bayesian optimization with objective: {objective}")
    elif use_resnet:
        # ResNet: sample multiplier combinations for 3 stages
        from nas_search import sample_resnet_multipliers
        architectures = [sample_resnet_multipliers(search_space) for _ in range(num_trials)]
    else:
        # CNN: use existing search
        if search_algo == 'random':
            architectures = random_search(search_space, num_trials)
        elif search_algo == 'grid':
            architectures = grid_search(search_space, max_trials=num_trials)
        else:
            raise ValueError(f"Unknown search algorithm: {search_algo}")

    results = []

    for i, arch in enumerate(architectures):
        print(f"\nTrial {i+1}/{len(architectures)}")
        print(f"Architecture: {arch}")

        result = train_and_evaluate(arch, x_train, y_train, x_test, y_test, epochs,
                                   use_stl, quality_constraint, energy_constraint, use_resnet)
        result['arch'] = arch
        results.append(result)

        # Update Bayesian optimization if using it
        if bayes_nas is not None:
            bayes_nas.update_observations(arch, result)

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
    # Run NAS with architecture selection
    # architecture='cnn' for simple CNN
    # architecture='resnet' for ResNet-20 (approxAI paper architecture - CURRENT DEFAULT)

    # ResNet-20 with STL monitoring (approxAI constraints)
    # Qc = 0.80 (80% minimum accuracy - ResNet should achieve this)
    # Ec = 100.0 mJ (maximum energy)
    #
    # Search algorithms:
    # 'random': Random sampling (baseline)
    # 'bayesian': Bayesian optimization (more efficient, recommended)
    # 'grid': Exhaustive grid search (slow, for small spaces)
    results = run_nas(
        search_algo='bayesian',  # Changed to Bayesian for better results
        num_trials=20,
        epochs=30,
        use_stl=True,
        quality_constraint=0.70,
        energy_constraint=100.0,
        architecture='resnet'  # Using ResNet-20 from approxAI paper
    )

    # Example: Switch to CNN (uncomment to use)
    # results = run_nas(
    #     search_algo='random',
    #     num_trials=60,
    #     epochs=50,
    #     use_stl=True,
    #     quality_constraint=0.70,
    #     energy_constraint=50.0,
    #     architecture='cnn'
    # )
