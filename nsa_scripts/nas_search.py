import random

def random_search(search_space, num_trials):
    results = []
    for _ in range(num_trials):
        arch = sample_architecture(search_space)
        results.append(arch)
    return results

def grid_search(search_space, max_trials=None):
    from itertools import product

    num_layers_options = search_space['num_conv_layers']
    filter_options = search_space['filters']
    kernel_options = search_space['kernel_sizes']
    dense_options = search_space['dense_units']
    mul_map_options = search_space['mul_map_files']

    results = []
    for num_layers in num_layers_options:
        for filters_combo in product(filter_options, repeat=num_layers):
            for kernels_combo in product(kernel_options, repeat=num_layers):
                for dense in dense_options:
                    for mul_map in mul_map_options:
                        arch = {
                            'num_conv_layers': num_layers,
                            'filters': list(filters_combo),
                            'kernels': list(kernels_combo),
                            'dense_units': dense,
                            'mul_map_file': mul_map
                        }
                        results.append(arch)
                        if max_trials and len(results) >= max_trials:
                            return results
    return results

def evolutionary_search(search_space, population_size, num_generations):
    population = [sample_architecture(search_space) for _ in range(population_size)]

    for gen in range(num_generations):
        yield population

        population = [mutate_architecture(arch, search_space) for arch in population]

def sample_architecture(search_space):
    num_layers = random.choice(search_space['num_conv_layers'])
    filters = [random.choice(search_space['filters']) for _ in range(num_layers)]
    kernels = [random.choice(search_space['kernel_sizes']) for _ in range(num_layers)]
    dense = random.choice(search_space['dense_units'])

    mul_maps = [random.choice(search_space['mul_map_files']) for _ in range(num_layers)]

    use_batch_norm = random.choice(search_space['use_batch_norm']) if 'use_batch_norm' in search_space else False

    return {
        'num_conv_layers': num_layers,
        'filters': filters,
        'kernels': kernels,
        'dense_units': dense,
        'mul_map_files': mul_maps,
        'use_batch_norm': use_batch_norm
    }

def mutate_architecture(arch, search_space):
    new_arch = arch.copy()
    mutation_type = random.choice(['filters', 'kernels', 'dense', 'mul_map', 'layers'])

    if mutation_type == 'filters' and new_arch['filters']:
        idx = random.randint(0, len(new_arch['filters']) - 1)
        new_arch['filters'][idx] = random.choice(search_space['filters'])
    elif mutation_type == 'kernels' and new_arch['kernels']:
        idx = random.randint(0, len(new_arch['kernels']) - 1)
        new_arch['kernels'][idx] = random.choice(search_space['kernel_sizes'])
    elif mutation_type == 'dense':
        new_arch['dense_units'] = random.choice(search_space['dense_units'])
    elif mutation_type == 'mul_map':
        new_arch['mul_map_file'] = random.choice(search_space['mul_map_files'])
    elif mutation_type == 'layers':
        new_arch = sample_architecture(search_space)

    return new_arch


def sample_resnet_multipliers(search_space):
    num_stages = random.choice(search_space['num_stages'])

    blocks_options = search_space['blocks_per_stage']
    if isinstance(blocks_options[0], list):
        blocks_per_stage = random.choice(blocks_options)
    else:
        blocks_value = random.choice(blocks_options)
        blocks_per_stage = [blocks_value] * num_stages

    base_filters = random.choice(search_space['base_filters'])

    mul_options = search_space['mul_map_files']
    mul_maps = [random.choice(mul_options) for _ in range(num_stages)]

    filters_per_stage = [base_filters * (2 ** i) for i in range(num_stages)]

    return {
        'num_stages': num_stages,
        'blocks_per_stage': blocks_per_stage,
        'filters_per_stage': filters_per_stage,
        'mul_map_files': mul_maps
    }
