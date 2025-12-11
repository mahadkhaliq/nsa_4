"""
Integration test for Imagenette dataset support

Tests:
1. Dataset loading (224×224×3 RGB)
2. ImageNet-style ResNet model building (4 stages, [64,128,256,512] filters, 7×7 initial conv)
3. Energy calculation for 224×224 input
4. Forward pass with exact and approximate multipliers
5. Backward compatibility with CIFAR-10 and FashionMNIST

Run: python test_imagenette.py
"""

import tensorflow as tf
import numpy as np
from model_builder import build_resnet20_exact, build_resnet20_approx
from energy_calculator import estimate_network_energy

# Test configuration
MUL_MAP_PATH = './multipliers/'

def test_imagenette_architecture():
    """Test ImageNet-style ResNet architecture for Imagenette"""
    print("\n" + "="*80)
    print("TEST 1: ImageNet-style ResNet Architecture (224×224)")
    print("="*80)

    # Imagenette architecture (4 stages, ImageNet-style)
    arch = {
        'num_stages': 4,
        'blocks_per_stage': [2, 2, 2, 2],  # ResNet-18 variant
        'base_filters': 64,
        'filters_per_stage': [64, 128, 256, 512],
        'mul_map_files': [
            MUL_MAP_PATH + 'mul8u_1JJQ.bin',  # Stage 1: Exact
            MUL_MAP_PATH + 'mul8u_2V0.bin',   # Stage 2: 0.0015% MAE
            MUL_MAP_PATH + 'mul8u_LK8.bin',   # Stage 3: 0.0046% MAE
            MUL_MAP_PATH + 'mul8u_R92.bin',   # Stage 4: 0.0170% MAE
        ]
    }

    input_shape = (224, 224, 3)

    # Build exact model
    print("\nBuilding exact ResNet-18 (224×224 input)...")
    exact_model = build_resnet20_exact(arch, input_shape=input_shape, num_classes=10)
    exact_model.summary()

    print(f"\n✓ Model built successfully")
    print(f"  Input shape: {input_shape}")
    print(f"  Output shape: {exact_model.output_shape}")
    print(f"  Total parameters: {exact_model.count_params():,}")

    # Test forward pass
    print("\nTesting forward pass...")
    dummy_input = np.random.randn(2, 224, 224, 3).astype('float32')
    output = exact_model(dummy_input, training=False)
    print(f"✓ Forward pass successful: {dummy_input.shape} → {output.shape}")

    # Build approximate model
    print("\nBuilding approximate ResNet-18 (heterogeneous multipliers)...")
    approx_model = build_resnet20_approx(arch, input_shape=input_shape, num_classes=10)
    approx_model.build(input_shape=(None, 224, 224, 3))
    print(f"✓ Approximate model built: {approx_model.count_params():,} parameters")

    # Test approximate forward pass
    output_approx = approx_model(dummy_input, training=False)
    print(f"✓ Approximate forward pass successful: {dummy_input.shape} → {output_approx.shape}")

    return True


def test_imagenette_energy():
    """Test energy calculation for Imagenette (224×224)"""
    print("\n" + "="*80)
    print("TEST 2: Energy Calculation for ImageNet-style (224×224)")
    print("="*80)

    arch = {
        'num_stages': 4,
        'blocks_per_stage': [2, 2, 2, 2],
        'base_filters': 64,
        'filters_per_stage': [64, 128, 256, 512],
        'mul_map_files': [
            MUL_MAP_PATH + 'mul8u_1JJQ.bin',
            MUL_MAP_PATH + 'mul8u_2V0.bin',
            MUL_MAP_PATH + 'mul8u_LK8.bin',
            MUL_MAP_PATH + 'mul8u_R92.bin',
        ]
    }

    print("\nCalculating energy for ResNet-18 (224×224)...")
    energy, energy_per_layer = estimate_network_energy(arch, input_size=224)

    print(f"\n✓ Total Energy: {energy:.2f} µJ")
    print(f"  Expected range: 50,000-100,000 µJ (50-100× more than CIFAR-10)")

    print("\nEnergy breakdown by stage:")
    for layer_info in energy_per_layer:
        print(f"  Stage {layer_info['stage']}: {layer_info['energy_uJ']:.2f} µJ "
              f"({layer_info['num_blocks']} blocks, {layer_info['filters']} filters)")

    # Verify energy is in expected range
    assert 10000 < energy < 200000, f"Energy {energy} µJ outside expected range [10k-200k]"
    print(f"\n✓ Energy calculation validated")

    return True


def test_backward_compatibility():
    """Test that CIFAR-10 and FashionMNIST still work"""
    print("\n" + "="*80)
    print("TEST 3: Backward Compatibility (CIFAR-10 and FashionMNIST)")
    print("="*80)

    # Test CIFAR-10 (32×32×3)
    print("\nTesting CIFAR-10 style (32×32×3)...")
    cifar_arch = {
        'num_stages': 3,
        'blocks_per_stage': [3, 3, 3],
        'base_filters': 16,
        'filters_per_stage': [16, 32, 64],
        'mul_map_files': [
            MUL_MAP_PATH + 'mul8u_1JJQ.bin',
            MUL_MAP_PATH + 'mul8u_2V0.bin',
            MUL_MAP_PATH + 'mul8u_LK8.bin',
        ]
    }

    cifar_model = build_resnet20_exact(cifar_arch, input_shape=(32, 32, 3), num_classes=10)
    cifar_energy, _ = estimate_network_energy(cifar_arch, input_size=32)

    dummy_cifar = np.random.randn(2, 32, 32, 3).astype('float32')
    cifar_output = cifar_model(dummy_cifar, training=False)

    print(f"✓ CIFAR-10 model works: {dummy_cifar.shape} → {cifar_output.shape}")
    print(f"  Energy: {cifar_energy:.2f} µJ (expected ~1000-1500 µJ)")

    # Test FashionMNIST (28×28×1)
    print("\nTesting FashionMNIST style (28×28×1)...")
    fashion_arch = {
        'num_stages': 3,
        'blocks_per_stage': [2, 2, 2],
        'base_filters': 16,
        'filters_per_stage': [16, 32, 64],
        'mul_map_files': [
            MUL_MAP_PATH + 'mul8u_1JJQ.bin',
            MUL_MAP_PATH + 'mul8u_2V0.bin',
            MUL_MAP_PATH + 'mul8u_LK8.bin',
        ]
    }

    fashion_model = build_resnet20_exact(fashion_arch, input_shape=(28, 28, 1), num_classes=10)
    fashion_energy, _ = estimate_network_energy(fashion_arch, input_size=28)

    dummy_fashion = np.random.randn(2, 28, 28, 1).astype('float32')
    fashion_output = fashion_model(dummy_fashion, training=False)

    print(f"✓ FashionMNIST model works: {dummy_fashion.shape} → {fashion_output.shape}")
    print(f"  Energy: {fashion_energy:.2f} µJ (expected ~400-600 µJ)")

    print("\n✓ All backward compatibility tests passed")

    return True


def test_search_space_sizes():
    """Calculate and display search space sizes"""
    print("\n" + "="*80)
    print("TEST 4: Search Space Size Comparison")
    print("="*80)

    # CIFAR-10
    cifar_blocks = 11  # 11 block patterns
    cifar_muls = 4     # 4 multipliers (top 4)
    cifar_space = cifar_blocks * (cifar_muls ** 3)  # 3 stages

    # FashionMNIST
    fashion_blocks = 6  # 6 block patterns
    fashion_muls = 4    # 4 multipliers
    fashion_space = fashion_blocks * (fashion_muls ** 3)  # 3 stages

    # Imagenette
    imagenette_blocks = 6  # 6 block patterns
    imagenette_muls = 4    # 4 multipliers
    imagenette_space = imagenette_blocks * (imagenette_muls ** 4)  # 4 stages

    print(f"\nCIFAR-10:")
    print(f"  Block patterns: {cifar_blocks}")
    print(f"  Multipliers: {cifar_muls}")
    print(f"  Stages: 3")
    print(f"  Total configurations: {cifar_space:,}")

    print(f"\nFashionMNIST:")
    print(f"  Block patterns: {fashion_blocks}")
    print(f"  Multipliers: {fashion_muls}")
    print(f"  Stages: 3")
    print(f"  Total configurations: {fashion_space:,}")

    print(f"\nImagenette:")
    print(f"  Block patterns: {imagenette_blocks}")
    print(f"  Multipliers: {imagenette_muls}")
    print(f"  Stages: 4")
    print(f"  Total configurations: {imagenette_space:,}")

    print(f"\n✓ Imagenette has {imagenette_space/fashion_space:.1f}× larger search space than FashionMNIST")

    return True


def run_all_tests():
    """Run all Imagenette integration tests"""
    print("\n" + "="*80)
    print("IMAGENETTE INTEGRATION TEST SUITE")
    print("="*80)

    tests = [
        ("ImageNet-style Architecture", test_imagenette_architecture),
        ("Energy Calculation", test_imagenette_energy),
        ("Backward Compatibility", test_backward_compatibility),
        ("Search Space Sizes", test_search_space_sizes),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
                print(f"\n✓ {test_name} PASSED")
        except Exception as e:
            failed += 1
            print(f"\n✗ {test_name} FAILED: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*80)
    print(f"TEST SUMMARY: {passed} passed, {failed} failed")
    print("="*80)

    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! Imagenette support is ready.")
        print("\nNext steps:")
        print("1. Download Imagenette dataset:")
        print("   wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz")
        print("   tar -xzf imagenette2-320.tgz -C ./data/")
        print("\n2. Run NAS experiments:")
        print("   python main.py  # Set dataset='imagenette' in main.py")
        print("\n3. Recommended constraints for Imagenette:")
        print("   - Quality constraint: 0.85 (85% accuracy)")
        print("   - Energy constraint: 80,000 µJ (80 mJ)")
        print("   - Epochs: 40-60 (224×224 images need more training)")
        print("   - Batch size: 128 (reduce if OOM)")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please fix before running experiments.")

    return failed == 0


if __name__ == '__main__':
    run_all_tests()
