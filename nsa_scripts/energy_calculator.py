def get_multiplier_specs(mul_map_file):
    specs_table = {
        'mul8u_1JJQ.bin': {'power_mW': 0.391, 'delay_ns': 1.43, 'MAE': 0.0000},

        'mul8u_2V0.bin':  {'power_mW': 0.386, 'delay_ns': 1.42, 'MAE': 0.0015},

        'mul8u_LK8.bin':  {'power_mW': 0.370, 'delay_ns': 1.40, 'MAE': 0.0046},
        'mul8u_17C8.bin': {'power_mW': 0.355, 'delay_ns': 1.39, 'MAE': 0.0090},

        'mul8u_R92.bin':  {'power_mW': 0.345, 'delay_ns': 1.41, 'MAE': 0.0170},

        'mul8u_18UH.bin': {'power_mW': 0.330, 'delay_ns': 1.42, 'MAE': 0.0250},

        'mul8u_0AB.bin':  {'power_mW': 0.302, 'delay_ns': 1.44, 'MAE': 0.0570},

        'mul8u_197B.bin': {'power_mW': 0.206, 'delay_ns': 1.50, 'MAE': 0.1200},

        '': {'power_mW': 0.391, 'delay_ns': 1.43, 'MAE': 0.0000}
    }

    filename = mul_map_file.split('/')[-1] if mul_map_file else ''
    return specs_table.get(filename, specs_table[''])

def calculate_energy_per_mac(multiplier_file):
    mult_specs = get_multiplier_specs(multiplier_file)

    ADDER_POWER_MW = 0.050
    ADDER_DELAY_NS = 0.20

    mult_energy_pJ = (mult_specs['power_mW'] * 1e-3) * (mult_specs['delay_ns'] * 1e-9) * 1e12
    adder_energy_pJ = (ADDER_POWER_MW * 1e-3) * (ADDER_DELAY_NS * 1e-9) * 1e12

    return mult_energy_pJ + adder_energy_pJ

def count_conv_macs(in_channels, out_channels, kernel_size, feature_map_size):
    return in_channels * out_channels * (kernel_size ** 2) * feature_map_size

def estimate_network_energy(arch, num_operations=1e9, input_size=32):
    total_energy_pJ = 0
    energy_per_layer = []

    if 'num_conv_layers' in arch:
        raise NotImplementedError("CNN energy calculation needs proper MAC counting")

    else:
        num_stages = arch['num_stages']
        blocks_per_stage_raw = arch['blocks_per_stage']
        if isinstance(blocks_per_stage_raw, int):
            blocks_per_stage = [blocks_per_stage_raw] * num_stages
        else:
            blocks_per_stage = blocks_per_stage_raw
        filters_per_stage = arch['filters_per_stage']

        feature_map_sizes = [input_size * input_size // (2 ** i) for i in range(num_stages)]

        in_channels = 3 if input_size == 32 else 1

        for stage_idx in range(num_stages):
            mul_map = arch['mul_map_files'][stage_idx]
            out_channels = filters_per_stage[stage_idx]
            feature_map_size = feature_map_sizes[stage_idx]
            num_blocks = blocks_per_stage[stage_idx]

            energy_per_mac_pJ = calculate_energy_per_mac(mul_map)

            total_macs = 0

            for block in range(num_blocks):
                if block == 0 and stage_idx > 0:
                    conv1_macs = count_conv_macs(in_channels, out_channels, 3, feature_map_size)
                else:
                    conv1_macs = count_conv_macs(in_channels, out_channels, 3, feature_map_size)

                conv2_macs = count_conv_macs(out_channels, out_channels, 3, feature_map_size)
                total_macs += conv1_macs + conv2_macs

                in_channels = out_channels

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

    total_energy_uJ = total_energy_pJ / 1e6

    return total_energy_uJ, energy_per_layer
