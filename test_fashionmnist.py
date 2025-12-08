#!/usr/bin/env python3
"""
Quick test script to verify FashionMNIST integration works correctly
"""
import tensorflow as tf
from data_loader import load_dataset
from model_builder import build_resnet20_exact, build_resnet20_approx
from energy_calculator import estimate_network_energy

# Suppress TF warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("="*70)
print("TESTING FASHIONMNIST INTEGRATION")
print("="*70)

# Test 1: Data Loading
print("\n[Test 1] Loading FashionMNIST dataset...")
try:
    (x_train, y_train), (x_test, y_test) = load_dataset('fashionmnist')
    print(f"  ✓ Training set: {x_train.shape}, Labels: {y_train.shape}")
    print(f"  ✓ Test set: {x_test.shape}, Labels: {y_test.shape}")
    print(f"  ✓ Input shape: {x_train.shape[1:]}")
    assert x_train.shape == (50000, 28, 28, 1), "Training shape mismatch"
    assert x_test.shape == (10000, 28, 28, 1), "Test shape mismatch"
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    exit(1)

# Test 2: Model Building
print("\n[Test 2] Building ResNet-20 for FashionMNIST...")
try:
    test_arch = {
        'num_stages': 3,
        'blocks_per_stage': [3, 3, 3],  # ResNet-20
        'filters_per_stage': [16, 32, 64],
        'mul_map_files': [
            './multipliers/mul8u_1JJQ.bin',
            './multipliers/mul8u_R92.bin',
            './multipliers/mul8u_1JJQ.bin'
        ]
    }

    # Build exact model
    model_exact = build_resnet20_exact(test_arch, input_shape=(28, 28, 1))
    print(f"  ✓ Exact model built: {model_exact.name}")
    print(f"  ✓ Input shape: {model_exact.input_shape}")
    print(f"  ✓ Output shape: {model_exact.output_shape}")

    # Build approximate model
    model_approx = build_resnet20_approx(test_arch, input_shape=(28, 28, 1))
    print(f"  ✓ Approx model built: {model_approx.name}")

    assert model_exact.input_shape == (None, 28, 28, 1), "Input shape mismatch"
    assert model_exact.output_shape == (None, 10), "Output shape mismatch"
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 3: Energy Calculation
print("\n[Test 3] Testing energy calculation for FashionMNIST...")
try:
    energy_28, layers_28 = estimate_network_energy(test_arch, input_size=28)
    energy_32, layers_32 = estimate_network_energy(test_arch, input_size=32)

    print(f"  ✓ Energy (28×28): {energy_28:.2f} µJ")
    print(f"  ✓ Energy (32×32): {energy_32:.2f} µJ")
    print(f"  ✓ Energy reduction: {100*(energy_32-energy_28)/energy_32:.1f}%")
    print(f"  ✓ Stage-wise breakdown (28×28):")
    for layer in layers_28:
        print(f"      Stage {layer['stage']}: {layer['energy_uJ']:.2f} µJ ({layer['macs']:,} MACs)")

    assert energy_28 < energy_32, "FashionMNIST should use less energy than CIFAR-10"
    assert len(layers_28) == 3, "Should have 3 stages"
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 4: Forward Pass
print("\n[Test 4] Testing forward pass...")
try:
    model_exact.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Test with small batch
    test_batch = x_test[:32]
    test_labels = y_test[:32]

    predictions = model_exact.predict(test_batch, verbose=0)
    print(f"  ✓ Predictions shape: {predictions.shape}")
    print(f"  ✓ Predicted classes: {predictions.argmax(axis=1)[:10]}")

    assert predictions.shape == (32, 10), "Prediction shape mismatch"
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 5: Backward Compatibility (CIFAR-10 still works)
print("\n[Test 5] Testing backward compatibility with CIFAR-10...")
try:
    (x_cifar_train, y_cifar_train), (x_cifar_test, y_cifar_test) = load_dataset('cifar10')
    print(f"  ✓ CIFAR-10 loading still works: {x_cifar_train.shape}")

    model_cifar = build_resnet20_exact(test_arch, input_shape=(32, 32, 3))
    print(f"  ✓ CIFAR-10 model building still works: {model_cifar.input_shape}")

    energy_cifar, _ = estimate_network_energy(test_arch, input_size=32)
    print(f"  ✓ CIFAR-10 energy calculation still works: {energy_cifar:.2f} µJ")
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "="*70)
print("ALL TESTS PASSED! ✓")
print("="*70)
print("\nYou can now run FashionMNIST experiments by uncommenting the")
print("FashionMNIST example in main.py and running:")
print("  python main.py")
print("\nOr create a custom run:")
print("  results = run_nas(")
print("      search_algo='bayesian',")
print("      num_trials=20,")
print("      epochs=60,")
print("      dataset='fashionmnist',")
print("      architecture='resnet'")
print("  )")
print("="*70)
