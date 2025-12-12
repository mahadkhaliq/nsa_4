#!/bin/bash
# Download and setup Imagenette dataset for NAS experiments
# Run this script from the nsa_4 directory: bash download_imagenette.sh

set -e  # Exit on error

echo "=================================================="
echo "Imagenette Dataset Download and Setup"
echo "=================================================="

# Create data directory if it doesn't exist
mkdir -p data
cd data

# Check if already downloaded
if [ -d "imagenette2-320" ]; then
    echo "✓ imagenette2-320 directory already exists"
    echo "Checking dataset structure..."

    if [ -d "imagenette2-320/train" ] && [ -d "imagenette2-320/val" ]; then
        echo "✓ Dataset structure looks good (train/ and val/ directories found)"

        # Count images (case-insensitive)
        train_count=$(find imagenette2-320/train -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.JPEG" \) 2>/dev/null | wc -l)
        val_count=$(find imagenette2-320/val -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.JPEG" \) 2>/dev/null | wc -l)

        echo "  Training images: $train_count (expected: ~9,469)"
        echo "  Validation images: $val_count (expected: ~3,925)"

        if [ "$train_count" -gt 9000 ] && [ "$val_count" -gt 3000 ]; then
            echo ""
            echo "✓ Dataset already downloaded and validated!"
            echo "You can now run: python main.py"
            exit 0
        else
            echo ""
            echo "⚠ Warning: Image counts seem low. Re-downloading..."
            rm -rf imagenette2-320
        fi
    else
        echo "⚠ Warning: Dataset structure incomplete. Re-downloading..."
        rm -rf imagenette2-320
    fi
fi

# Download dataset
echo ""
echo "Downloading imagenette2-320.tgz (~1.5 GB)..."
echo "This may take 5-15 minutes depending on your connection..."
echo ""

wget --progress=bar:force:noscroll https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz

# Extract dataset
echo ""
echo "Extracting imagenette2-320.tgz..."
tar -xzf imagenette2-320.tgz

# Verify extraction
if [ -d "imagenette2-320/train" ] && [ -d "imagenette2-320/val" ]; then
    echo "✓ Extraction successful!"

    # Count images (case-insensitive)
    train_count=$(find imagenette2-320/train -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.JPEG" \) 2>/dev/null | wc -l)
    val_count=$(find imagenette2-320/val -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.JPEG" \) 2>/dev/null | wc -l)

    echo ""
    echo "Dataset statistics:"
    echo "  Training images: $train_count"
    echo "  Validation images: $val_count"
    echo "  Total size: $(du -sh imagenette2-320 | cut -f1)"

    # Clean up tar file (auto-delete to save space)
    echo ""
    echo "Cleaning up tar file to save space..."
    rm -f imagenette2-320.tgz
    echo "✓ Cleaned up tar file"

    echo ""
    echo "=================================================="
    echo "✓ Imagenette setup complete!"
    echo "=================================================="
    echo ""
    echo "Dataset location: $(pwd)/imagenette2-320"
    echo ""
    echo "Next steps:"
    echo "  1. cd .."
    echo "  2. python main.py"
    echo ""
    echo "The NAS experiment will run with:"
    echo "  - 15 trials (Bayesian optimization)"
    echo "  - 50 epochs per trial"
    echo "  - 224×224 RGB images"
    echo "  - ImageNet-style ResNet architecture"
    echo "  - Expected runtime: ~30-60 hours on V100 GPU"
    echo ""
else
    echo "✗ Error: Extraction failed!"
    echo "Please check the tar file and try again."
    exit 1
fi
