#!/bin/bash
# Quick verification script for Imagenette dataset
# Run: bash verify_imagenette.sh

echo "=================================================="
echo "Imagenette Dataset Verification"
echo "=================================================="

DATA_DIR="./data/imagenette2-320"

# Check if dataset directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "✗ Error: Dataset directory not found at $DATA_DIR"
    echo ""
    echo "Please download the dataset first:"
    echo "  bash download_imagenette.sh"
    exit 1
fi

echo "✓ Dataset directory exists: $DATA_DIR"
echo ""

# Check train and val directories
if [ ! -d "$DATA_DIR/train" ]; then
    echo "✗ Error: Training directory not found: $DATA_DIR/train"
    exit 1
fi

if [ ! -d "$DATA_DIR/val" ]; then
    echo "✗ Error: Validation directory not found: $DATA_DIR/val"
    exit 1
fi

echo "✓ Directory structure looks good"
echo ""

# Count classes
train_classes=$(ls -d $DATA_DIR/train/*/ 2>/dev/null | wc -l)
val_classes=$(ls -d $DATA_DIR/val/*/ 2>/dev/null | wc -l)

echo "Classes found:"
echo "  Training classes: $train_classes (expected: 10)"
echo "  Validation classes: $val_classes (expected: 10)"
echo ""

# List class names
echo "Class directories in train/:"
ls -1 $DATA_DIR/train/ | head -10
echo ""

# Count images with multiple methods
echo "Counting images (this may take a moment)..."

# Method 1: Simple ls count (faster)
train_img_simple=$(find $DATA_DIR/train -type f 2>/dev/null | wc -l)
val_img_simple=$(find $DATA_DIR/val -type f 2>/dev/null | wc -l)

# Method 2: Count by extension
train_jpeg=$(find $DATA_DIR/train -type f -iname "*.JPEG" 2>/dev/null | wc -l)
val_jpeg=$(find $DATA_DIR/val -type f -iname "*.JPEG" 2>/dev/null | wc -l)

echo ""
echo "Image counts:"
echo "  Training images (all files): $train_img_simple"
echo "  Training images (.JPEG): $train_jpeg"
echo "  Validation images (all files): $val_img_simple"
echo "  Validation images (.JPEG): $val_jpeg"
echo ""
echo "  Expected: ~9,469 training + ~3,925 validation"
echo ""

# Check a sample image
echo "Sample images from first class:"
sample_class=$(ls -1 $DATA_DIR/train/ | head -1)
sample_images=$(ls $DATA_DIR/train/$sample_class/ | head -3)
echo "  Class: $sample_class"
for img in $sample_images; do
    echo "    - $img"
done
echo ""

# Check dataset size
total_size=$(du -sh $DATA_DIR 2>/dev/null | cut -f1)
echo "Dataset size: $total_size (expected: ~360M)"
echo ""

# Final verdict
if [ "$train_img_simple" -gt 9000 ] && [ "$val_img_simple" -gt 3000 ]; then
    echo "=================================================="
    echo "✓ Dataset verification PASSED!"
    echo "=================================================="
    echo ""
    echo "The dataset is ready to use."
    echo "You can now run: python main.py"
    echo ""
    exit 0
else
    echo "=================================================="
    echo "⚠ Dataset verification INCOMPLETE"
    echo "=================================================="
    echo ""
    echo "Image counts seem low. This might be normal if:"
    echo "  1. The extraction is still in progress"
    echo "  2. Some images are in a different format"
    echo ""
    echo "To test if the data loader works, run:"
    echo "  python -c 'from data_loader import load_imagenette; load_imagenette()'"
    echo ""
    exit 1
fi
