# Plotting Module Usage Guide

## Quick Start

The `plotter.py` module provides functions for generating publication-quality plots from your NAS experiments.

## Basic Usage

### Option 1: Individual Plots During Experiment

```python
from plotter import setup_plot_dirs, plot_training_curves, plot_energy_breakdown

# Setup directories once at the start
plot_dirs = setup_plot_dirs('resnet18_baseline')

# During training - plot each trial's training curves
for trial_num, (arch, history) in enumerate(trials):
    plot_training_curves(history, trial_num=trial_num+1, save_dir=plot_dirs['training'])

# After each trial - plot energy breakdown
for trial_num, result in enumerate(results):
    plot_energy_breakdown(result, trial_num=trial_num+1, save_dir=plot_dirs['energy'])
```

### Option 2: Generate All Plots After Experiment

```python
from plotter import generate_all_plots

# After all trials complete
plots = generate_all_plots(
    results=results,
    pareto_indices=pareto_indices,
    quality_constraint=0.89,
    experiment_name='resnet18_nas_80epochs'
)

# plots is a dict with paths to all generated plots
print(f"Plots saved: {plots}")
```

## Integration with main.py

Here's how to integrate plotting into your NAS experiment:

```python
# In main.py - add at the top
from plotter import setup_plot_dirs, plot_training_curves, plot_energy_breakdown, generate_all_plots

# In run_nas() function - after line 70 (after selecting search space)
# Setup plot directories
plot_dirs = setup_plot_dirs(f'{architecture}_{search_algo}_{epochs}ep')

# In the trial loop - after line 109 (after training completes)
# Plot training curves if history is available
if 'history' in result and result['history']:
    plot_training_curves(result['history'], trial_num=i+1, save_dir=plot_dirs['training'])

# Plot energy breakdown for this trial
if result['energy_per_layer']:
    plot_energy_breakdown(result, trial_num=i+1, save_dir=plot_dirs['energy'])

# At the end - after line 153 (after finding Pareto solutions)
# Generate comprehensive plots
print("\n" + "="*60)
print("Generating publication-quality plots...")
print("="*60)

generate_all_plots(
    results=results,
    pareto_indices=pareto_indices,
    quality_constraint=quality_constraint,
    experiment_name=f'{architecture}_{search_algo}_{epochs}ep'
)
```

## Available Plot Functions

### 1. `setup_plot_dirs(experiment_name, base_dir='plots')`
Creates organized directory structure for plots.

**Returns:** Dictionary with paths to subdirectories:
```python
{
    'base': 'plots/nas_20251119_143022',
    'training': 'plots/nas_20251119_143022/training',
    'energy': 'plots/nas_20251119_143022/energy',
    'stl': 'plots/nas_20251119_143022/stl',
    'summary': 'plots/nas_20251119_143022/summary'
}
```

### 2. `plot_training_curves(history, trial_num, save_dir=None)`
Plots training accuracy and loss curves side-by-side.

**Args:**
- `history`: Keras History object or dict with 'accuracy', 'val_accuracy', 'loss', 'val_loss'
- `trial_num`: Trial number for filename
- `save_dir`: Directory to save (if None, only displays)

**Saves:** `trial_001_training.png`

### 3. `plot_energy_vs_accuracy(results, pareto_indices=None, save_dir=None)`
Scatter plot of energy vs accuracy tradeoff (like paper's Fig. 8-9).

**Args:**
- `results`: List of result dicts with 'approx_accuracy', 'energy', 'energy_per_layer'
- `pareto_indices`: List of Pareto-optimal solution indices
- `save_dir`: Directory to save

**Saves:** `energy_vs_accuracy.png`

**Features:**
- Blue circles: Exact multiplier architectures
- Green squares: Approximate multiplier architectures
- Red stars: Pareto-optimal solutions

### 4. `plot_pareto_front(results, pareto_indices, quality_constraint=None, save_dir=None)`
Pareto front visualization with STL constraint indicators.

**Args:**
- `results`: List of result dicts
- `pareto_indices`: List of Pareto-optimal indices
- `quality_constraint`: Qc threshold line (optional)
- `save_dir`: Directory to save

**Saves:** `pareto_front.png`

**Features:**
- Green circles: STL satisfied
- Red X: STL violated
- Blue line: Pareto front
- Orange dashed: Quality constraint

### 5. `plot_energy_breakdown(result, trial_num=None, save_dir=None)`
Bar chart showing energy consumption per stage.

**Args:**
- `result`: Result dict with 'energy_per_layer'
- `trial_num`: Trial number (optional)
- `save_dir`: Directory to save

**Saves:** `energy_breakdown_trial_001.png`

**Features:**
- Blue bars: Exact multiplier (1JJQ)
- Green bars: Approximate multipliers
- Values labeled on each bar

### 6. `plot_stl_analysis(results, save_dir=None)`
Two-panel plot: pie chart of satisfaction rate + histogram of robustness scores.

**Args:**
- `results`: List of result dicts with 'stl_robustness'
- `save_dir`: Directory to save

**Saves:** `stl_analysis.png`

### 7. `generate_all_plots(results, pareto_indices=None, quality_constraint=None, experiment_name='nas')`
One-shot generation of all plots.

**Args:**
- `results`: List of all trial results
- `pareto_indices`: Pareto-optimal indices
- `quality_constraint`: Qc threshold
- `experiment_name`: Name for directories

**Returns:** Dict with paths to all generated plots

**Generates:**
- Energy vs accuracy plot
- Pareto front plot (if pareto_indices provided)
- STL analysis plot
- Energy breakdown for best architecture

## Output Directory Structure

```
plots/
└── resnet18_bayesian_80ep_20251119_143022/
    ├── training/
    │   ├── trial_001_training.png
    │   ├── trial_002_training.png
    │   └── ...
    ├── energy/
    │   ├── energy_vs_accuracy.png
    │   ├── pareto_front.png
    │   ├── energy_breakdown.png
    │   └── energy_breakdown_trial_001.png
    ├── stl/
    │   └── stl_analysis.png
    └── summary/
        └── (future summary plots)
```

## Plot Specifications

All plots are saved with:
- **Resolution:** 300 DPI (publication quality)
- **Format:** PNG with tight bounding box
- **Style:** Seaborn paper style
- **Font sizes:** 12pt body, 14pt axes, 16pt titles
- **Grid:** Light alpha gridlines
- **Colors:** Color-blind friendly palette

## Tips for Publications

1. **For papers:** Use 300 DPI PNG files directly or convert to PDF/EPS
2. **For presentations:** Plots are already sized appropriately (10x6 or 14x6 inches)
3. **Customization:** Edit `plotter.py` matplotlib rcParams at the top to match journal requirements
4. **Multiple experiments:** Use different `experiment_name` for each run to organize results

## Example: Complete Workflow

```python
from plotter import setup_plot_dirs, plot_training_curves, generate_all_plots

# 1. Setup at experiment start
dirs = setup_plot_dirs('resnet18_baseline_nov19')

# 2. During experiment - save training curves
for i, (arch, history) in enumerate(train_architectures()):
    plot_training_curves(history, trial_num=i+1, save_dir=dirs['training'])
    print(f"Trial {i+1} plots saved")

# 3. After experiment - generate all summary plots
plots = generate_all_plots(
    results=all_results,
    pareto_indices=pareto_solutions,
    quality_constraint=0.89,
    experiment_name='resnet18_baseline_nov19'
)

print(f"All plots generated in: {dirs['base']}")
```

## Common Issues

**Issue:** Plots not showing up
**Solution:** Make sure `save_dir` parameter is provided. Without it, plots only display (plt.show())

**Issue:** Directory already exists error
**Solution:** Each run creates a timestamped directory, so this shouldn't happen. Check permissions.

**Issue:** Missing data in plots
**Solution:** Ensure result dicts have required keys:
- Training: 'accuracy', 'val_accuracy', 'loss', 'val_loss'
- Energy: 'energy', 'energy_per_layer'
- STL: 'stl_robustness'

---

**Created:** 2025-11-19
**Version:** 1.0
**Module:** plotter.py
