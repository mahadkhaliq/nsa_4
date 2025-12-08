def get_multiplier_specs(mul_map_file):
    """Get power and delay specs for approximate multiplier from EvoApproxLib

    Values extracted from PDK45 synthesis (1V, 25C, 45nm process)
    Source: EvoApproxLib Verilog files - pareto_pwr_wce directory
    Format: {'power_mW': float, 'delay_ns': float}
    """
    # PDK45 synthesis results for 8x8 unsigned multipliers (AVAILABLE FILES ONLY)
    specs_table = {
        # Exact multiplier
        'mul8u_1JJQ.bin': {'power_mW': 0.391, 'delay_ns': 1.43, 'MAE': 0.0000},  # EXACT

        # Very low error (< 0.005% MAE)
        'mul8u_2V0.bin':  {'power_mW': 0.386, 'delay_ns': 1.42, 'MAE': 0.0015},  # Pareto-optimal

        # Low error (0.005% - 0.01% MAE)
        'mul8u_LK8.bin':  {'power_mW': 0.370, 'delay_ns': 1.40, 'MAE': 0.0046},  # Pareto-optimal
        'mul8u_17C8.bin': {'power_mW': 0.355, 'delay_ns': 1.39, 'MAE': 0.0090},  # More savings

        # Medium error (0.01% - 0.02% MAE)
        'mul8u_R92.bin':  {'power_mW': 0.345, 'delay_ns': 1.41, 'MAE': 0.0170},  # Last week's best!

        # Medium-high error (0.02% - 0.04% MAE)
        'mul8u_18UH.bin': {'power_mW': 0.330, 'delay_ns': 1.42, 'MAE': 0.0250},  # Aggressive

        # High error (0.05% - 0.06% MAE)
        'mul8u_0AB.bin':  {'power_mW': 0.302, 'delay_ns': 1.44, 'MAE': 0.0570},  # Highest tested

        # Very high error (> 0.08% MAE)
        'mul8u_197B.bin': {'power_mW': 0.206, 'delay_ns': 1.50, 'MAE': 0.1200},  # Extreme savings

        # Default
        '': {'power_mW': 0.391, 'delay_ns': 1.43, 'MAE': 0.0000}  # Default to exact
    }

    filename = mul_map_file.split('/')[-1] if mul_map_file else ''
    return specs_table.get(filename, specs_table[''])

def calculate_energy_per_mac(multiplier_file):
    """Calculate energy for one MAC (Multiply-Accumulate) operation in picojoules

    MAC = 1 multiplication + 1 addition
    Energy = (Multiplier_Power × Multiplier_Delay) + (Adder_Power × Adder_Delay)

    Returns:
        float: Energy per MAC operation in picojoules (pJ)
    """
    # Get multiplier specs from PDK45 synthesis
    mult_specs = get_multiplier_specs(multiplier_file)

    # 8-bit adder specs (typical values from literature)
    # Adder is ~10-15% of multiplier power and much faster
    ADDER_POWER_MW = 0.050  # mW
    ADDER_DELAY_NS = 0.20   # ns

    # Calculate energy: E = P × t (converted to picojoules)
    mult_energy_pJ = (mult_specs['power_mW'] * 1e-3) * (mult_specs['delay_ns'] * 1e-9) * 1e12
    adder_energy_pJ = (ADDER_POWER_MW * 1e-3) * (ADDER_DELAY_NS * 1e-9) * 1e12

    return mult_energy_pJ + adder_energy_pJ

def count_conv_macs(in_channels, out_channels, kernel_size, feature_map_size):
    """Count MAC operations for a convolutional layer

    Args:
        in_channels: Number of input channels
        out_channels: Number of output filters
        kernel_size: Kernel size (assumes square kernel)
        feature_map_size: Spatial size of feature map (H×W)

    Returns:
        int: Total MAC operations
    """
    return in_channels * out_channels * (kernel_size ** 2) * feature_map_size

def estimate_network_energy(arch, num_operations=1e9, input_size=32):
    """Estimate total network energy consumption using MAC-based model

    Args:
        arch: Network architecture dict
              ResNet: has 'num_stages', 'blocks_per_stage', 'filters_per_stage'
              CNN: has 'num_conv_layers', 'filters', 'kernels'
        input_size: Spatial dimension of input (default 32 for CIFAR-10, use 28 for FashionMNIST)

    Returns:
        energy: Total energy in microjoules (µJ)
        energy_per_layer: List of dict with energy details per stage/layer
    """
    total_energy_pJ = 0
    energy_per_layer = []

    if 'num_conv_layers' in arch:
        # CNN architecture - variable layers
        # TODO: Implement proper MAC counting for CNNs
        raise NotImplementedError("CNN energy calculation needs proper MAC counting")

    else:
        # ResNet architecture - variable stages/blocks
        num_stages = arch['num_stages']
        blocks_per_stage_raw = arch['blocks_per_stage']
        # Handle both formats: integer or list
        if isinstance(blocks_per_stage_raw, int):
            blocks_per_stage = [blocks_per_stage_raw] * num_stages
        else:
            blocks_per_stage = blocks_per_stage_raw
        filters_per_stage = arch['filters_per_stage']

        # Feature map sizes based on input_size (stride-2 downsample per stage)
        # CIFAR-10 (input_size=32): Stage 0 (32×32=1024), Stage 1 (16×16=256), Stage 2 (8×8=64)
        # FashionMNIST (input_size=28): Stage 0 (28×28=784), Stage 1 (14×14=196), Stage 2 (7×7=49)
        feature_map_sizes = [input_size * input_size // (2 ** i) for i in range(num_stages)]

        # Initial conv layer (usually 3→16 channels for CIFAR, 3×3 kernel on 32×32)
        # Assuming first stage handles this
        in_channels = 3  # RGB input

        for stage_idx in range(num_stages):
            mul_map = arch['mul_map_files'][stage_idx]
            out_channels = filters_per_stage[stage_idx]
            feature_map_size = feature_map_sizes[stage_idx]
            num_blocks = blocks_per_stage[stage_idx]  # Variable blocks per stage

            # Get energy per MAC for this stage's multiplier
            energy_per_mac_pJ = calculate_energy_per_mac(mul_map)

            # Each ResNet block has 2 conv layers (both 3×3)
            # First conv in block: in_channels → out_channels
            # Second conv in block: out_channels → out_channels
            total_macs = 0

            for block in range(num_blocks):
                if block == 0 and stage_idx > 0:
                    # First block of new stage: downsample with stride=2
                    # Feature map size already halved in feature_map_sizes
                    conv1_macs = count_conv_macs(in_channels, out_channels, 3, feature_map_size)
                else:
                    # Regular block: no downsampling
                    conv1_macs = count_conv_macs(in_channels, out_channels, 3, feature_map_size)

                conv2_macs = count_conv_macs(out_channels, out_channels, 3, feature_map_size)
                total_macs += conv1_macs + conv2_macs

                # After first block, in_channels = out_channels
                in_channels = out_channels

            # Calculate total energy for this stage
            stage_energy_pJ = total_macs * energy_per_mac_pJ

            energy_per_layer.append({
                'stage': stage_idx,
                'multiplier': mul_map.split('/')[-1],
                'macs': total_macs,
                'energy_per_mac_pJ': energy_per_mac_pJ,
                'energy_pJ': stage_energy_pJ,
                'energy_uJ': stage_energy_pJ / 1e6
            })
            total_energy_pJ += stage_energy_pJ

    # Convert total energy from picojoules to microjoules
    total_energy_uJ = total_energy_pJ / 1e6

    return total_energy_uJ, energy_per_layer
