# Imagenette Dataset Download Instructions

## Quick Download (Automated)

Run the download script from the `nsa_4` directory:

```bash
bash download_imagenette.sh
```

This will:
- Create `data/` directory
- Download imagenette2-320.tgz (~1.5 GB)
- Extract to `data/imagenette2-320/`
- Verify dataset integrity
- Optionally clean up the tar file

---

## Manual Download (Alternative)

If the script doesn't work, download manually:

```bash
# 1. Create data directory
mkdir -p data
cd data

# 2. Download dataset (~1.5 GB, takes 5-15 minutes)
wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz

# 3. Extract
tar -xzf imagenette2-320.tgz

# 4. Verify structure
ls imagenette2-320/
# Should see: train/ val/

# 5. Count images
find imagenette2-320/train -type f -name "*.JPEG" | wc -l
# Expected: ~9,469 training images

find imagenette2-320/val -type f -name "*.JPEG" | wc -l
# Expected: ~3,925 validation images

# 6. Clean up (optional)
rm imagenette2-320.tgz

# 7. Go back to nsa_4 directory
cd ..
```

---

## Expected Directory Structure

```
nsa_4/
├── data/
│   └── imagenette2-320/
│       ├── train/
│       │   ├── n01440764/  (tench - 963 images)
│       │   ├── n02102040/  (English springer - 963 images)
│       │   ├── n02979186/  (cassette player - 963 images)
│       │   ├── n03000684/  (chain saw - 963 images)
│       │   ├── n03028079/  (church - 963 images)
│       │   ├── n03394916/  (French horn - 963 images)
│       │   ├── n03417042/  (garbage truck - 963 images)
│       │   ├── n03425413/  (gas pump - 963 images)
│       │   ├── n03445777/  (golf ball - 963 images)
│       │   └── n03888257/  (parachute - 963 images)
│       └── val/
│           ├── n01440764/  (tench - 394 images)
│           ├── n02102040/  (English springer - 394 images)
│           └── ... (same 10 classes)
└── main.py
```

---

## Verify Dataset is Ready

```bash
python -c "from data_loader import load_imagenette; load_imagenette()"
```

Expected output:
```
Loading from ./data/imagenette2-320/train...
Found 10 classes: ['n01440764', 'n02102040', ...]
  Class 0 (n01440764): 963 images
  Class 1 (n02102040): 963 images
  ...
Loading from ./data/imagenette2-320/val...
Found 10 classes: ['n01440764', 'n02102040', ...]
  Class 0 (n01440764): 394 images
  Class 1 (n02102040): 394 images
  ...

Imagenette loaded successfully!
  Training: (9469, 224, 224, 3) images, (9469,) labels
  Validation: (3925, 224, 224, 3) images, (3925,) labels
```

---

## If Download is Slow or Fails

### Alternative Download Mirrors

1. **Direct download (no wget):**
   ```bash
   curl -O https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz
   ```

2. **Using aria2c (faster, parallel downloads):**
   ```bash
   aria2c -x 16 https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz
   ```

3. **Pre-download on local machine, then transfer:**
   - Download on your local machine: https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz
   - Transfer to remote: `scp imagenette2-320.tgz user@remote:/path/to/nsa_4/data/`
   - Extract on remote: `tar -xzf imagenette2-320.tgz`

---

## Troubleshooting

### Error: "wget: command not found"

Use curl instead:
```bash
curl -O https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz
```

### Error: "tar: command not found"

Unlikely on Linux/Unix, but if it happens:
```bash
gunzip imagenette2-320.tgz
tar -xf imagenette2-320.tar
```

### Error: "No space left on device"

Check available space:
```bash
df -h .
```

Imagenette requires:
- ~1.5 GB for compressed tar file
- ~1.8 GB for extracted dataset
- **Total: ~3.3 GB** (or ~1.8 GB if you delete the tar file after extraction)

### Dataset downloaded but Python still can't find it

Check the path:
```bash
ls -la data/imagenette2-320/
```

If the directory is somewhere else, either:
1. Move it: `mv /path/to/imagenette2-320 data/`
2. Or create a symlink: `ln -s /path/to/imagenette2-320 data/imagenette2-320`

---

## After Dataset is Ready

Run the NAS experiments:

```bash
python main.py
```

Monitor progress:
```bash
tail -f logs/imagenette_resnet_bayesian_50ep_*.log
```

---

## Dataset Information

- **Name**: Imagenette
- **Source**: fast.ai (subset of ImageNet)
- **Classes**: 10 (tench, English springer, cassette player, chain saw, church, French horn, garbage truck, gas pump, golf ball, parachute)
- **Training images**: 9,469 (~950 per class)
- **Validation images**: 3,925 (~390 per class)
- **Resolution**: 320×320 (will be resized to 224×224)
- **Format**: JPEG
- **Size**: 1.5 GB compressed, 1.8 GB extracted
- **License**: Same as ImageNet (research/educational use)

---

## Quick Reference

| Command | Purpose |
|---------|---------|
| `bash download_imagenette.sh` | Automated download and setup |
| `ls data/imagenette2-320/` | Verify dataset exists |
| `find data/imagenette2-320/train -name "*.JPEG" \| wc -l` | Count training images |
| `python -c "from data_loader import load_imagenette; load_imagenette()"` | Test data loader |
| `python main.py` | Run NAS experiments |
| `tail -f logs/*.log` | Monitor experiment progress |
