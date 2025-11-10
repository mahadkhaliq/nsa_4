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
    """
    power_per_mult = get_multiplier_power(arch['mul_map_file'])  # mW

    # Estimate total multiplications based on architecture
    total_mults = 0
    for i in range(arch['num_conv_layers']):
        # Conv layer: output_size * filters * kernel_size^2
        filters = arch['filters'][i]
        kernel = arch['kernels'][i]
        total_mults += filters * kernel * kernel * 1024  # Rough estimate

    # Energy = Power * Time (assuming 1ns per operation)
    # E (mJ) = P (mW) * operations * 1e-9 (s) * 1e3 (mJ/J)
    energy = power_per_mult * total_mults * 1e-6  # mJ

    return energy
