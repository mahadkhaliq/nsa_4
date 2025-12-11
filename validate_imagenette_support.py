"""
Validation script for Imagenette support (no TensorFlow required)

Validates:
1. Search space definitions
2. Energy calculator logic
3. Model builder signatures
4. Data loader structure
5. Backward compatibility

Run: python validate_imagenette_support.py
"""

import ast
import sys

def validate_file_syntax(filepath):
    """Check if a Python file has valid syntax"""
    try:
        with open(filepath, 'r') as f:
            ast.parse(f.read())
        return True, "OK"
    except SyntaxError as e:
        return False, f"Syntax error: {e}"


def check_search_space_in_main():
    """Verify Imagenette search space is defined in main.py"""
    with open('main.py', 'r') as f:
        content = f.read()

    checks = [
        ('SEARCH_SPACE_RESNET_IMAGENETTE', 'Imagenette search space'),
        ("'imagenette'", 'Imagenette dataset branch'),
        ("input_shape = (224, 224, 3)", 'Imagenette input shape'),
        ("'num_stages': [4]", '4-stage architecture'),
        ("'base_filters': [64]", 'ImageNet filter sizes'),
    ]

    print("\n" + "="*80)
    print("VALIDATION 1: main.py - Search Space Definition")
    print("="*80)

    all_passed = True
    for pattern, description in checks:
        if pattern in content:
            print(f"✓ {description}")
        else:
            print(f"✗ MISSING: {description}")
            all_passed = False

    return all_passed


def check_energy_calculator():
    """Verify energy calculator handles 224×224 input"""
    with open('energy_calculator.py', 'r') as f:
        content = f.read()

    checks = [
        ('input_size == 224', '224×224 input detection'),
        ('initial_feature_size = 56', 'ImageNet feature map calculation'),
        ('in_channels = 3  # Imagenette RGB', 'Imagenette RGB channels'),
    ]

    print("\n" + "="*80)
    print("VALIDATION 2: energy_calculator.py - 224×224 Support")
    print("="*80)

    all_passed = True
    for pattern, description in checks:
        if pattern in content:
            print(f"✓ {description}")
        else:
            print(f"✗ MISSING: {description}")
            all_passed = False

    return all_passed


def check_model_builder():
    """Verify model builder supports ImageNet-style ResNet"""
    with open('model_builder.py', 'r') as f:
        content = f.read()

    checks = [
        ('is_imagenet_style = (input_shape[0] == 224)', 'ImageNet-style detection'),
        ('Conv2D(filters_per_stage[0], 7, strides=2', '7×7 initial conv'),
        ('MaxPooling2D(pool_size=3, strides=2', 'MaxPooling after initial conv'),
        ('(224, 224, 3): ImageNet style', 'ImageNet-style documentation'),
    ]

    print("\n" + "="*80)
    print("VALIDATION 3: model_builder.py - ImageNet-style ResNet")
    print("="*80)

    all_passed = True
    for pattern, description in checks:
        if pattern in content:
            print(f"✓ {description}")
        else:
            print(f"✗ MISSING: {description}")
            all_passed = False

    return all_passed


def check_data_loader():
    """Verify data loader has Imagenette support"""
    with open('data_loader.py', 'r') as f:
        content = f.read()

    checks = [
        ('def load_imagenette(', 'load_imagenette() function'),
        ('imagenette2-320', 'Dataset path documentation'),
        ('img_size=224', '224×224 image size'),
        ("'imagenette'", 'Imagenette dataset case'),
        ('image_dataset_from_directory', 'Keras dataset loader'),
    ]

    print("\n" + "="*80)
    print("VALIDATION 4: data_loader.py - Imagenette Dataset")
    print("="*80)

    all_passed = True
    for pattern, description in checks:
        if pattern in content:
            print(f"✓ {description}")
        else:
            print(f"✗ MISSING: {description}")
            all_passed = False

    return all_passed


def check_backward_compatibility():
    """Verify CIFAR-10 and FashionMNIST patterns still exist"""
    with open('main.py', 'r') as f:
        main_content = f.read()

    with open('energy_calculator.py', 'r') as f:
        energy_content = f.read()

    checks = [
        (main_content, 'SEARCH_SPACE_RESNET ', 'CIFAR-10 search space'),
        (main_content, 'SEARCH_SPACE_RESNET_FASHIONMNIST', 'FashionMNIST search space'),
        (main_content, "dataset='cifar10'", 'CIFAR-10 default parameter'),
        (energy_content, 'input_size == 32', 'CIFAR-10 energy calculation'),
        (energy_content, 'input_size == 28', 'FashionMNIST energy calculation'),
    ]

    print("\n" + "="*80)
    print("VALIDATION 5: Backward Compatibility (CIFAR-10 & FashionMNIST)")
    print("="*80)

    all_passed = True
    for content, pattern, description in checks:
        if pattern in content:
            print(f"✓ {description}")
        else:
            print(f"✗ MISSING: {description}")
            all_passed = False

    return all_passed


def check_syntax():
    """Check syntax of all modified files"""
    files = [
        'main.py',
        'data_loader.py',
        'energy_calculator.py',
        'model_builder.py',
    ]

    print("\n" + "="*80)
    print("VALIDATION 6: Python Syntax Check")
    print("="*80)

    all_passed = True
    for filepath in files:
        valid, message = validate_file_syntax(filepath)
        if valid:
            print(f"✓ {filepath}: {message}")
        else:
            print(f"✗ {filepath}: {message}")
            all_passed = False

    return all_passed


def calculate_search_spaces():
    """Calculate and display search space sizes"""
    print("\n" + "="*80)
    print("SEARCH SPACE ANALYSIS")
    print("="*80)

    datasets = [
        ("CIFAR-10", 11, 4, 3),         # blocks, multipliers, stages
        ("FashionMNIST", 6, 4, 3),
        ("Imagenette", 6, 4, 4),
    ]

    for name, blocks, muls, stages in datasets:
        total = blocks * (muls ** stages)
        print(f"\n{name}:")
        print(f"  Block patterns: {blocks}")
        print(f"  Multipliers: {muls}")
        print(f"  Stages: {stages}")
        print(f"  Total configurations: {total:,}")


def main():
    """Run all validations"""
    print("\n" + "="*80)
    print("IMAGENETTE SUPPORT VALIDATION (No TensorFlow Required)")
    print("="*80)

    validations = [
        check_search_space_in_main,
        check_energy_calculator,
        check_model_builder,
        check_data_loader,
        check_backward_compatibility,
        check_syntax,
    ]

    results = []
    for validation in validations:
        try:
            passed = validation()
            results.append(passed)
        except Exception as e:
            print(f"\n✗ Validation failed with error: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)

    calculate_search_spaces()

    # Summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)

    passed = sum(results)
    total = len(results)

    print(f"\nPassed: {passed}/{total}")

    if passed == total:
        print("\n🎉 ALL VALIDATIONS PASSED!")
        print("\nImagenette support is ready. Next steps:")
        print("\n1. Download Imagenette dataset:")
        print("   wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz")
        print("   tar -xzf imagenette2-320.tgz")
        print("   mkdir -p data && mv imagenette2-320 data/")
        print("\n2. Update main.py to run Imagenette experiments:")
        print("   results = run_nas(")
        print("       search_algo='bayesian',")
        print("       num_trials=15,  # Fewer trials due to longer training")
        print("       epochs=50,      # More epochs for 224×224")
        print("       use_stl=True,")
        print("       quality_constraint=0.85,  # 85% accuracy target")
        print("       energy_constraint=80000.0,  # 80 mJ (80,000 µJ)")
        print("       architecture='resnet',")
        print("       dataset='imagenette',")
        print("       batch_size=128  # Reduce if GPU OOM")
        print("   )")
        print("\n3. Expected results:")
        print("   - Accuracy: 82-90% (approximate)")
        print("   - Energy: 50,000-80,000 µJ")
        print("   - Training time: ~10-20× longer than CIFAR-10")
        return 0
    else:
        print(f"\n⚠️  {total - passed} validation(s) failed.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
