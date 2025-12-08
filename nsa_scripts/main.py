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


MUL_MAP_PATH = './multipliers/'

SEARCH_SPACE_CNN = {
    'num_conv_layers': [2, 3, 4],
    'filters': [16, 32, 64, 128],
    'kernel_sizes': [3, 5],
    'dense_units': [64, 128, 256],
    'mul_map_files': [
        MUL_MAP_PATH + 'mul8u_1JJQ.bin',
        MUL_MAP_PATH + 'mul8u_2V0.bin',
    ],
    'use_batch_norm': [True, False]
}



MULTIPLIERS_ALL = [
    MUL_MAP_PATH + 'mul8u_1JJQ.bin',
    MUL_MAP_PATH + 'mul8u_2V0.bin',
    MUL_MAP_PATH + 'mul8u_LK8.bin',
    MUL_MAP_PATH + 'mul8u_17C8.bin',
    MUL_MAP_PATH + 'mul8u_R92.bin',
    MUL_MAP_PATH + 'mul8u_18UH.bin',
    MUL_MAP_PATH + 'mul8u_0AB.bin',
    MUL_MAP_PATH + 'mul8u_197B.bin',
]

MULTIPLIERS_CONSERVATIVE = MULTIPLIERS_ALL[:5]

MULTIPLIERS_AGGRESSIVE = MULTIPLIERS_ALL[4:]

SEARCH_SPACE_RESNET = {
    'num_stages': [3],
    'blocks_per_stage': [
        [3, 3, 3],
        [5, 5, 5],
        [7, 7, 7],
        [9, 9, 9],
        [3, 4, 5],
        [4, 5, 6],
        [5, 7, 9],
        [6, 4, 2],
        [2, 4, 6],
        [3, 6, 3],
        [4, 6, 4],
    ],
    'base_filters': [16],
    'mul_map_files': MULTIPLIERS_ALL
}

SEARCH_SPACE_RESNET_CONSERVATIVE = {
    'num_stages': [3],
    'blocks_per_stage': [
        [3, 3, 3],
        [5, 5, 5],
        [7, 7, 7],
    ],
    'base_filters': [16],
    'mul_map_files': MULTIPLIERS_CONSERVATIVE
}

SEARCH_SPACE_RESNET_FASHIONMNIST = {
    'num_stages': [3],
    'blocks_per_stage': [
        [2, 2, 2],
        [3, 3, 3],
        [4, 4, 4],
        [2, 3, 4],
        [4, 3, 2],
        [2, 4, 2],
    ],
    'base_filters': [16],
    'mul_map_files': MULTIPLIERS_ALL
}

def run_nas(search_algo='random', num_trials=5, epochs=5, use_stl=False,
            quality_constraint=0.70, energy_constraint=50.0, architecture='cnn',
            dataset='cifar10', batch_size=256):

    use_resnet = (architecture.lower() == 'resnet')

    if dataset.lower() == 'fashionmnist':
        search_space = SEARCH_SPACE_RESNET_FASHIONMNIST if use_resnet else SEARCH_SPACE_CNN
        input_shape = (28, 28, 1)
        num_classes = 10
    else:
        search_space = SEARCH_SPACE_RESNET if use_resnet else SEARCH_SPACE_CNN
        input_shape = (32, 32, 3)
        num_classes = 10

    (x_train, y_train), (x_test, y_test) = load_dataset(dataset)

    experiment_name = f"{dataset}_{architecture}_{search_algo}_{epochs}ep"
    logger = NASLogger(log_dir='logs', experiment_name=experiment_name)

    plot_dirs = setup_plot_dirs(experiment_name, base_dir='plots')

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

    bayes_nas = None
    if search_algo == 'bayesian':
        objective = 'stl_robustness' if use_stl else 'accuracy'
        architectures, bayes_nas = bayesian_search(search_space, num_trials, objective)
        print(f"Using Bayesian optimization with objective: {objective}")
    elif use_resnet:
        from nas_search import sample_resnet_multipliers
        architectures = [sample_resnet_multipliers(search_space) for _ in range(num_trials)]
    else:
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
                                   use_stl, quality_constraint, energy_constraint, use_resnet, input_shape, batch_size)
        result['arch'] = arch
        results.append(result)

        if bayes_nas is not None:
            bayes_nas.update_observations(arch, result)

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

        if use_resnet:
            arch_config = f"ResNet: {arch['num_stages']} stages, {arch['blocks_per_stage']} blocks, {arch['filters_per_stage']}"
        else:
            arch_config = f"CNN: {arch['num_conv_layers']} layers, {arch['filters']} filters"

        if 'history' in result and result['history']:
            plot_training_curves(result['history'], trial_num=i+1, save_dir=plot_dirs['training'], arch_config=arch_config)
            logger.info(f"Training curves saved for trial {i+1}")

        if result['energy_per_layer']:
            plot_energy_breakdown(result, trial_num=i+1, save_dir=plot_dirs['energy'], arch_config=arch_config)
            logger.info(f"Energy breakdown plot saved for trial {i+1}")

    best = max(results, key=lambda x: x['approx_accuracy'] if x['approx_accuracy'] else x['exact_accuracy'])

    pareto_indices = check_pareto_optimal(results)

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

    logger.header("Generating publication-quality plots")
    print(f"\n{'='*60}")
    print("Generating publication-quality plots...")
    print(f"{'='*60}")

    if use_resnet:
        first_arch = results[0]['arch']
        arch_config = f"ResNet: {first_arch['num_stages']} stages, {first_arch['blocks_per_stage']} blocks, {first_arch['filters_per_stage']}"
    else:
        first_arch = results[0]['arch']
        arch_config = f"CNN: {first_arch['num_conv_layers']} layers"

    plots_generated = generate_all_plots(
        results=results,
        pareto_indices=pareto_indices,
        quality_constraint=quality_constraint,
        experiment_name=experiment_name,
        arch_config=arch_config
    )

    logger.info("All plots generated successfully")
    for plot_name, plot_path in plots_generated.items():
        if plot_path:
            logger.info(f"  {plot_name}: {plot_path}")

    logger.save_results()

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
    print("\n" + "="*70)
    print("NEURAL ARCHITECTURE SEARCH - SEARCH SPACE")
    print("="*70)
    print(f"\n📊 MULTIPLIERS: {len(MULTIPLIERS_ALL)} total")
    print(f"   - Conservative (MAE < 0.02%): {len(MULTIPLIERS_CONSERVATIVE)}")
    print(f"   - Aggressive (MAE ≥ 0.02%): {len(MULTIPLIERS_AGGRESSIVE)}")

    print("\n🏗️  ARCHITECTURE VARIANTS:")
    for blocks in SEARCH_SPACE_RESNET['blocks_per_stage']:
        total_layers = sum(blocks) * 2 + 2
        print(f"   - ResNet-{total_layers}: {blocks} blocks → {total_layers} layers")

    num_archs = len(SEARCH_SPACE_RESNET['blocks_per_stage'])
    num_muls = len(SEARCH_SPACE_RESNET['mul_map_files'])
    num_stages = SEARCH_SPACE_RESNET['num_stages'][0]
    total_configs = num_archs * (num_muls ** num_stages)

    print(f"\n🔍 SEARCH SPACE SIZE:")
    print(f"   - {num_archs} architectures × {num_muls}^{num_stages} multiplier combos")
    print(f"   - Total configurations: {total_configs:,}")
    print(f"   - Recommended trials (Bayesian): 50-100")

    print("\n" + "="*70)
    print("STARTING NAS EXPERIMENT")
    print("="*70 + "\n")


    results = run_nas(
        search_algo='bayesian',
        num_trials=20,
        epochs=60,
        use_stl=True,
        quality_constraint=0.89,
        energy_constraint=5000.0,
        architecture='resnet'
    )


