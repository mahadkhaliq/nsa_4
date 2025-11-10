def get_multiplier_power(mul_map_file):
    """Get power consumption for approximate multiplier from EvoApproxLib

    Power values in mW from EvoApproxLib for 8-bit unsigned multipliers
    TODO: Fill in actual values from EvoApproxLib CSV/documentation
    """
    power_table = {
        'mul8u_17C8.bin': 0.104,   # From EvoApproxLib website
        'mul8u_125K.bin': 0.25,    # TODO: Get actual value
        'mul8u_1JFF.bin': 0.30,    # TODO: Get actual value
        'mul8u_2AC.bin': 0.35,     # TODO: Get actual value
        'mul8u_1AGV.bin': 0.28,    # TODO: Get actual value
        '': 0.45  # Exact multiplier baseline (approximate)
    }

    filename = mul_map_file.split('/')[-1] if mul_map_file else ''
    return power_table.get(filename, 0.45)

def estimate_network_energy(arch, num_operations=1e9):
    """Estimate total network energy consumption

    Args:
        arch: Network architecture dict
        num_operations: Estimated number of multiply operations

    Returns:
        energy: Energy in mJ (millijoules)
        energy_per_layer: List of energy per layer
    """
    total_energy = 0
    energy_per_layer = []

    # Calculate energy for each conv layer with its specific multiplier
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

    return total_energy, energy_per_layer
