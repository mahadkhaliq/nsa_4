import tensorflow as tf
from nas_search import random_search, grid_search
from evaluator import train_and_evaluate
from data_loader import load_dataset
from stl_monitor import check_pareto_optimal
from bayesian_nas import bayesian_search
from logger import NASLogger
from plotter import setup_plot_dirs, plot_training_curves, plot_energy_breakdown, generate_all_plots

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
        #MUL_MAP_PATH + 'mul8u_197B.bin',   # 0.206 mW - medium balance
        MUL_MAP_PATH + 'mul8u_1JJQ.bin',   # 0.391 mW - best performing
        MUL_MAP_PATH + 'mul8u_2V0.bin',    # BEST - 0.0015% MAE, 64% energy saved
        #MUL_MAP_PATH + 'mul8u_0AB.bin',    # 0.302 mW - medium-high
    ],
    'use_batch_norm': [True, False]
}

# ResNet Search Space
# BASELINE: ResNet-18 (Paper's exact architecture)
SEARCH_SPACE_RESNET = {
    'num_stages': [4],  # ResNet-18: 4 stages
    'blocks_per_stage': [2],  # ResNet-18: 2 blocks per stage
    'base_filters': [64],  # ResNet-18: [64, 128, 256, 512]
    # NAS: Uncomment lines below for full search
    #'num_stages': [2, 3],
    #'blocks_per_stage': [2, 3, 4],
    #'base_filters': [16, 32],
    'mul_map_files': [
        MUL_MAP_PATH + 'mul8u_1JJQ.bin',   # EXACT - 0% error (baseline)
        #MUL_MAP_PATH + 'mul8u_2V0.bin',    # BEST - 0.0015% MAE, 64% energy saved
        MUL_MAP_PATH + 'mul8u_LK8.bin',    # EXCELLENT - 0.0046% MAE, 75% energy saved
        #MUL_MAP_PATH + 'mul8u_R92.bin',    # VERY GOOD - 0.017% MAE, 87.5% energy saved
        #MUL_MAP_PATH + 'mul8u_0AB.bin',    # GOOD - 0.057% MAE, 97.7% energy saved
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

    # Initialize logger
    experiment_name = f"{architecture}_{search_algo}_{epochs}ep"
    logger = NASLogger(log_dir='logs', experiment_name=experiment_name)

    # Setup plot directories
    plot_dirs = setup_plot_dirs(experiment_name, base_dir='plots')

    # Log experiment configuration
    config = {
        'architecture': 'ResNet-18' if use_resnet else 'Simple CNN',
        'search_algorithm': search_algo,
        'num_trials': num_trials,
        'epochs': epochs,
        'use_stl': use_stl,
        'quality_constraint': quality_constraint,
        'energy_constraint': energy_constraint,
        'search_space': str(search_space)
    }
    logger.log_config(config)

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
        logger.subheader(f"Trial {i+1}/{len(architectures)}")
        logger.info(f"Architecture: {arch}")

        print(f"\nTrial {i+1}/{len(architectures)}")
        print(f"Architecture: {arch}")

        result = train_and_evaluate(arch, x_train, y_train, x_test, y_test, epochs,
                                   use_stl, quality_constraint, energy_constraint, use_resnet)
        result['arch'] = arch
        results.append(result)

        # Update Bayesian optimization if using it
        if bayes_nas is not None:
            bayes_nas.update_observations(arch, result)

        # Log and print trial results
        logger.log_trial(i+1, len(architectures), arch, result)

        print(f"Exact accuracy: {result['exact_accuracy']:.4f}")
        if result['approx_accuracy']:
            print(f"Approx accuracy: {result['approx_accuracy']:.4f}")
            print(f"Accuracy drop: {result['exact_accuracy'] - result['approx_accuracy']:.4f}")
        if result['energy']:
            print(f"Total energy: {result['energy']:.4f} µJ")
        if result['energy_per_layer']:
            logger.log_energy_breakdown(result['energy_per_layer'])
            print("Energy breakdown per stage:")
            for layer_info in result['energy_per_layer']:
                stage = layer_info.get('stage', layer_info.get('layer', '?'))
                mult = layer_info['multiplier']
                energy_uj = layer_info.get('energy_uJ', layer_info.get('energy', 0))
                macs = layer_info.get('macs', 0)
                print(f"  Stage {stage}: {mult} - {energy_uj:.4f} µJ ({macs:,} MACs)")
        if result['stl_robustness'] is not None:
            status = "SATISFIED" if result['stl_robustness'] > 0 else "VIOLATED"
            print(f"STL robustness: {result['stl_robustness']:.4f} ({status})")

        # Plot training curves if history is available
        if 'history' in result and result['history']:
            plot_training_curves(result['history'], trial_num=i+1, save_dir=plot_dirs['training'])
            logger.info(f"Training curves saved for trial {i+1}")

        # Plot energy breakdown for this trial
        if result['energy_per_layer']:
            plot_energy_breakdown(result, trial_num=i+1, save_dir=plot_dirs['energy'])
            logger.info(f"Energy breakdown plot saved for trial {i+1}")

    # Find best by accuracy
    best = max(results, key=lambda x: x['approx_accuracy'] if x['approx_accuracy'] else x['exact_accuracy'])

    # Find Pareto-optimal solutions (approxAI methodology)
    pareto_indices = check_pareto_optimal(results)

    # Log summary
    logger.log_summary(results)

    print(f"\n{'='*60}")
    print("Best architecture by accuracy:")
    print(f"  Accuracy: {best['approx_accuracy']:.4f}, Energy: {best['energy']:.4f} µJ")
    print(f"  Architecture: {best['arch']}")

    if pareto_indices:
        logger.info(f"\nPareto-optimal architectures: {len(pareto_indices)} found")
        print(f"\n{'='*60}")
        print(f"Pareto-optimal architectures ({len(pareto_indices)} found):")
        print("(approxAI: No architecture dominates these in both accuracy AND energy)")
        for idx in pareto_indices:
            r = results[idx]
            print(f"\n  Trial {idx+1}:")
            print(f"    Accuracy: {r['approx_accuracy']:.4f}, Energy: {r['energy']:.4f} µJ")
            if r['stl_robustness'] is not None:
                status = "SATISFIED" if r['stl_robustness'] > 0 else "VIOLATED"
                print(f"    STL (Qc={quality_constraint}, Ec={energy_constraint}): {status}")

    # Generate comprehensive plots
    logger.header("Generating publication-quality plots")
    print(f"\n{'='*60}")
    print("Generating publication-quality plots...")
    print(f"{'='*60}")

    plots_generated = generate_all_plots(
        results=results,
        pareto_indices=pareto_indices,
        quality_constraint=quality_constraint,
        experiment_name=experiment_name
    )

    logger.info("All plots generated successfully")
    for plot_name, plot_path in plots_generated.items():
        if plot_path:
            logger.info(f"  {plot_name}: {plot_path}")

    # Save results to JSON
    logger.save_results()

    # Print log file locations
    log_files = logger.get_log_files()
    print(f"\n{'='*60}")
    print("Experiment completed!")
    print(f"{'='*60}")
    print(f"Logs saved to: {log_files['log_file']}")
    print(f"Results saved to: {log_files['results_file']}")
    print(f"Plots saved to: {plot_dirs['base']}")
    print(f"{'='*60}\n")

    return results

if __name__ == '__main__':
    # Run NAS with architecture selection
    # architecture='cnn' for simple CNN
    # architecture='resnet' for ResNet-18 (approxAI paper architecture - CURRENT DEFAULT)

    # ResNet-18 with STL monitoring (approxAI constraints)
    # Qc = 0.89 (89% minimum accuracy - 2% below baseline 91%, per paper Section V-B)
    # Ec = 100.0 mJ (maximum energy)
    #
    # Search algorithms:
    # 'random': Random sampling (baseline)
    # 'bayesian': Bayesian optimization (more efficient, recommended)
    # 'grid': Exhaustive grid search (slow, for small spaces)
    results = run_nas(
        search_algo='bayesian',  # Changed to Bayesian for better results
        num_trials=20,
        epochs=60,
        use_stl=True,
        quality_constraint=0.89,  # Paper: 2% below baseline (91% - 2% = 89%)
        energy_constraint=5000.0,
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
