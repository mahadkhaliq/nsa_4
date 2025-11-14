# NAS for Approximate Neural Networks - Session Summary

**Date:** 2025-11-14
**Project:** Neural Architecture Search with Approximate Multipliers for CIFAR-10

---

## Current Project Status

### What We've Built

A comprehensive NAS system that searches for optimal ResNet architectures with approximate multipliers for CIFAR-10 classification, incorporating STL (Signal Temporal Logic) monitoring for quality and energy constraints.

**Key Features:**
- ✅ Flexible ResNet architecture search (variable stages, blocks, filters)
- ✅ Heterogeneous approximate multipliers per stage
- ✅ STL constraint monitoring (quality ≥ 0.80, energy ≤ 100 mJ)
- ✅ Support for both CNN and ResNet architectures
- ✅ Training progress visibility (verbose=1)
- ✅ Energy estimation for different multipliers
- ✅ Pareto-optimal architecture selection

---

## Architecture Overview

### ResNet Basics (For Reference)

**Stages:** Groups of residual blocks at different resolutions
```
Stage 1: 32×32 feature maps, 16 filters (early features - edges, textures)
Stage 2: 16×16 feature maps, 32 filters (mid features - shapes, parts)
Stage 3: 8×8 feature maps, 64 filters (high-level - objects, concepts)
```

**Blocks per Stage:** Number of residual blocks in each stage
```
Residual Block:
  Input → [Conv→BN→ReLU→Conv→BN] + Skip Connection → Add → ReLU → Output
```

**Example Architectures:**
- ResNet-10: 2 stages, 2 blocks/stage = 1 + (2×2×2) + 1 = 10 layers
- ResNet-20: 3 stages, 3 blocks/stage = 1 + (3×3×2) + 1 = 20 layers
- ResNet-26: 3 stages, 4 blocks/stage = 1 + (3×4×2) + 1 = 26 layers

---

## Search Space Configuration

Located in [main.py:46-54](main.py)

```python
SEARCH_SPACE_RESNET = {
    'num_stages': [2, 3],           # 2 or 3 stages
    'blocks_per_stage': [2, 3, 4],  # 2-4 blocks per stage
    'base_filters': [16, 32],       # Starting filters (doubles each stage)
    'mul_map_files': [
        'mul8u_197B.bin',   # 0.206 mW - medium balance
        'mul8u_1JJQ.bin',   # 0.391 mW - best performing
        'mul8u_0AB.bin',    # 0.302 mW - medium-high
    ]
}
```

**Total search space:** 2 × 3 × 2 × 3³ = **324 possible configurations**

---

## How NAS Works (Our Implementation)

### 1. Random Sampling ([nas_search.py:94-127](nas_search.py))

For each of 20 trials:
```python
# Sample architecture
num_stages = random.choice([2, 3])
blocks_per_stage = random.choice([2, 3, 4])
base_filters = random.choice([16, 32])

# Sample one multiplier per stage (heterogeneous)
mul_maps = [random.choice(mul_options) for _ in range(num_stages)]

# Derive filters per stage (doubles each stage)
filters_per_stage = [base_filters * (2**i) for i in range(num_stages)]
```

### 2. Train & Evaluate ([evaluator.py](evaluator.py))

For each sampled architecture:
1. Build exact model with architecture
2. Train for 80 epochs (verbose=1 shows progress)
3. Build approximate model with same architecture + multipliers
4. Transfer weights from exact to approximate
5. Evaluate approximate model → accuracy
6. Calculate energy consumption
7. Evaluate STL constraints

### 3. STL Constraint Checking ([stl_monitor.py](stl_monitor.py))

```python
# φ = (accuracy ≥ Qc) ∧ (energy ≤ Ec)
quality_margin = accuracy - 0.80
energy_margin = 100.0 - energy
robustness = min(quality_margin, energy_margin)
```

- **Positive robustness:** Constraints satisfied
- **Negative robustness:** Constraints violated

### 4. Select Best ([main.py:117-145](main.py))

- Find architecture with highest STL robustness
- Identify Pareto-optimal solutions (accuracy vs energy tradeoff)

---

## Comparison with approxAI Paper

| Aspect | **approxAI Paper** | **Our Implementation** |
|--------|-------------------|------------------------|
| **Architecture** | Fixed (ResNet-18/34/50) | Variable (search stages, blocks, filters) |
| **Search Space** | Only multipliers | Architecture + multipliers |
| **Algorithm** | NSGA-II (evolutionary) + XAI | Random search + STL |
| **Guidance** | XAI (neuron conductance) | None (pure random) |
| **Speed** | 5× faster with XAI | Slower (no pruning) |
| **Flexibility** | Multiplier assignment only | Full architecture design |

**Key Difference:**
- **Paper:** Fixed architecture → XAI identifies layer importance → NSGA-II searches best multiplier per layer
- **Us:** Random sampling → Sample architecture + multipliers → Train → STL evaluation → Pareto-optimal

**Our approach is more comprehensive** - we do TRUE architecture search, not just multiplier assignment.

---

## File Structure

### Core Files

**[main.py](main.py)**
- Entry point
- Defines search spaces (CNN and ResNet)
- Default: ResNet, 20 trials, 80 epochs
- Constraints: Qc=0.80, Ec=100.0 mJ

**[nas_search.py](nas_search.py)**
- `sample_resnet_multipliers()`: Samples architecture + multipliers
- `run_nas()`: Orchestrates the search

**[model_builder.py](model_builder.py)**
- `build_resnet20_exact()`: Builds exact ResNet with variable architecture
- `build_resnet20_approx()`: Builds approximate ResNet with FakeApproxConv2D
- `build_model()`: CNN architecture (original)
- `build_approx_model()`: Approximate CNN

**[evaluator.py](evaluator.py)**
- `train_and_evaluate()`: Trains exact model, transfers to approximate, evaluates
- Changed `verbose=0` to `verbose=1` for training progress visibility

**[energy_calculator.py](energy_calculator.py)**
- `estimate_network_energy()`: Calculates energy for CNN or ResNet
- Handles variable architectures (checks for 'num_conv_layers' vs 'num_stages')
- Uses multiplier power specs from EvoApproxLib

**[stl_monitor.py](stl_monitor.py)**
- `check_stl_constraints()`: Evaluates φ = (Q ≥ Qc) ∧ (E ≤ Ec)
- Returns robustness score

**[test_resnet_approx.py](test_resnet_approx.py)**
- Standalone test for ResNet-20 with approximate multipliers
- Good for quick validation

---

## Current Configuration

**Default execution:**
```bash
python main.py
```

Uses:
- Architecture: ResNet (use_resnet=True)
- Trials: 20
- Epochs: 80
- Quality constraint: Qc = 0.80 (80% accuracy)
- Energy constraint: Ec = 100.0 mJ
- Multipliers: mul8u_197B, mul8u_1JJQ, mul8u_0AB

**To switch to CNN:**
```python
# In main.py, change:
use_resnet = False
```

---

## Approximate Multipliers (EvoApproxLib)

Located in: `mul_maps/EvoApprox8b/mul8u/`

| Multiplier | Power (mW) | Characteristics |
|------------|-----------|-----------------|
| mul8u_197B.bin | 0.206 | Low power, medium accuracy |
| mul8u_1JJQ.bin | 0.391 | High power, best accuracy |
| mul8u_0AB.bin | 0.302 | Medium power, medium accuracy |

**Heterogeneous approximation:** Different multipliers per stage based on sensitivity.

---

## Recent Changes & Fixes

### Issue 1: KeyError 'num_conv_layers'
**Problem:** Energy calculator only handled CNN architecture format
**Fix:** Updated `estimate_network_energy()` to detect architecture type (CNN vs ResNet)

### Issue 2: Architecture Search for ResNet
**Problem:** Initially only searched multipliers (like the paper)
**Enhancement:** Expanded to search architecture parameters (stages, blocks, filters) + multipliers

### Issue 3: Training Progress Visibility
**Problem:** `verbose=0` hid training progress
**Fix:** Changed to `verbose=1` in evaluator.py

---

## Git Status

**Current branch:** master
**Modified files:** test_resnet_approx.py

**Recent commits:**
```
f33cbd8 Enable flexible ResNet NAS with variable architecture
4b626ab Add ResNet-20 architecture with architecture selection flag
cbe4c22 Switch to better multiplier (mul8u_1JJQ.bin) for ResNet-20 test
62da119 increased the sample size for cifar for testing
55ffbf3 Update test to use proven ResNet-20 CIFAR-10 architecture
```

---

## Next Steps / Future Enhancements

### Potential Improvements:

1. **Bayesian Optimization** (user requested earlier)
   - Replace random search with Bayesian optimization
   - More efficient exploration of search space

2. **XAI Guidance** (like the paper)
   - Add neuron conductance analysis
   - Identify layer sensitivity
   - Guide multiplier assignment intelligently

3. **Evolutionary Search** (NSGA-II)
   - Multi-objective optimization
   - Population-based search
   - Pareto front discovery

4. **Performance Testing**
   - Test on Singularity machine
   - Benchmark best architectures
   - Compare with paper results

5. **Extended Search Space**
   - More multiplier options
   - Variable learning rates
   - Different optimizers

---

## How to Resume Work

### On MacBook:

1. **Clone/Pull the repository:**
   ```bash
   git pull origin master
   ```

2. **Verify environment:**
   ```bash
   python -c "import tensorflow as tf; print(tf.__version__)"
   # Should have tf-approximate installed
   ```

3. **Run current NAS:**
   ```bash
   python main.py
   ```

4. **Test single architecture:**
   ```bash
   python test_resnet_approx.py
   ```

5. **Read this summary:**
   ```bash
   cat SESSION_SUMMARY.md
   ```

---

## Key Questions Answered in This Session

### Q: What are stages?
**A:** Groups of residual blocks at different resolutions. Early stages detect simple patterns (edges), late stages detect high-level concepts (objects).

### Q: What are blocks per stage?
**A:** Number of residual blocks stacked in each stage. Each block has 2 convolutions + skip connection.

### Q: How did the paper do the search?
**A:** They used XAI-guided NSGA-II to search only multiplier assignments on fixed architectures (ResNet-18/34/50).

### Q: How are we different?
**A:** We search BOTH architecture (stages, blocks, filters) AND multipliers. More comprehensive but slower without XAI guidance.

### Q: How does our search actually work?
**A:** Random sampling → Sample arch + multipliers → Train exact model → Transfer to approximate → Evaluate accuracy + energy → STL robustness → Select best Pareto-optimal.

---

## Contact & Resources

- **approxAI Paper:** Uses XAI + NSGA-II for multiplier search on fixed architectures
- **EvoApproxLib:** https://ehw.fit.vutbr.cz/evoapproxlib/
- **tf-approximate:** TensorFlow library for approximate computing

---

## Important Notes

- Current implementation uses **random search** (20 trials)
- **STL constraints** ensure quality (≥80%) and energy (≤100 mJ)
- **Heterogeneous approximation** allows different multipliers per stage
- Training takes ~80 epochs per trial (visible with verbose=1)
- Search space: **324 possible configurations**
- More comprehensive than paper but less efficient (no XAI pruning yet)

---

**Last Updated:** 2025-11-14
**Status:** Fully functional, ready for testing and enhancement
