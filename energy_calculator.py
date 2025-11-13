def get_multiplier_power(mul_map_file):
    """Get power consumption for approximate multiplier from EvoApproxLib

    Power values in mW from EvoApproxLib (PDK45, 1V, 25C)
    Source: https://ehw.fit.vutbr.cz/evoapproxlib
    """
    power_table = {
        'mul8u_17C8.bin': 0.104,   # Low power
        'mul8u_197B.bin': 0.206,   # Medium power
        'mul8u_0AB.bin': 0.302,    # Medium-high power
        'mul8u_1JJQ.bin': 0.391,   # Higher accuracy
        'mul8u_1JFF.bin': 0.391,   # From EvoApproxLib v1.0
        'mul8u_125K.bin': 0.35,    # Estimated (TODO: verify)
        'mul8u_2AC.bin': 0.38,     # Estimated (TODO: verify)
        'mul8u_1AGV.bin': 0.30,    # Estimated (TODO: verify)
        '': 0.45  # Exact multiplier baseline
    }

    filename = mul_map_file.split('/')[-1] if mul_map_file else ''
    return power_table.get(filename, 0.45)

def estimate_network_energy(arch, num_operations=1e9):
    """Estimate total network energy consumption for CNN or ResNet

    Args:
        arch: Network architecture dict
              CNN: has 'num_conv_layers', 'filters', 'kernels'
              ResNet: has 'num_stages', 'blocks_per_stage', 'filters_per_stage'
        num_operations: Estimated number of multiply operations

    Returns:
        energy: Energy in mJ (millijoules)
        energy_per_layer: List of energy per stage/layer
    """
    total_energy = 0
    energy_per_layer = []

    if 'num_conv_layers' in arch:
        # CNN architecture - variable layers
        for i in range(arch['num_conv_layers']):
            mul_map = arch['mul_map_files'][i]
            power_per_mult = get_multiplier_power(mul_map)  # mW

            # Estimate multiplications for this layer
            filters = arch['filters'][i]
            kernel = arch['kernels'][i]
            layer_mults = filters * kernel * kernel * 1024  # Rough estimate

            # Energy = Power * operations * time
            layer_energy = power_per_mult * layer_mults * 1e-6  # mJ
            energy_per_layer.append({
                'layer': i,
                'multiplier': mul_map.split('/')[-1],
                'energy': layer_energy,
                'power': power_per_mult
            })
            total_energy += layer_energy

    else:
        # ResNet architecture - variable stages/blocks
        num_stages = arch['num_stages']
        blocks_per_stage = arch['blocks_per_stage']
        filters_per_stage = arch['filters_per_stage']

        # Feature map sizes (assuming CIFAR-10: 32x32 input, downsample by 2 each stage)
        feature_map_sizes = [32 * 32 // (2 ** i) for i in range(num_stages)]

        for stage_idx in range(num_stages):
            mul_map = arch['mul_map_files'][stage_idx]
            power_per_mult = get_multiplier_power(mul_map)  # mW

            # Each residual block has 2 conv layers (3x3)
            filters = filters_per_stage[stage_idx]
            feature_map = feature_map_sizes[stage_idx]
            num_convs = blocks_per_stage * 2  # 2 convs per block

            # Estimate multiplications for this stage (all 3x3 convs)
            stage_mults = filters * 3 * 3 * feature_map * num_convs

            # Energy = Power * operations * time
            stage_energy = power_per_mult * stage_mults * 1e-6  # mJ
            energy_per_layer.append({
                'layer': stage_idx,
                'multiplier': mul_map.split('/')[-1],
                'energy': stage_energy,
                'power': power_per_mult
            })
            total_energy += stage_energy

    return total_energy, energy_per_layer
