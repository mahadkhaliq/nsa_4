
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

plt.style.use('seaborn-paper')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 11


def setup_plot_dirs(experiment_name='nas', base_dir='plots'):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = os.path.join(base_dir, f'{experiment_name}_{timestamp}')

    dirs = {
        'base': exp_dir,
        'training': os.path.join(exp_dir, 'training'),
        'energy': os.path.join(exp_dir, 'energy'),
        'stl': os.path.join(exp_dir, 'stl'),
        'summary': os.path.join(exp_dir, 'summary')
    }

    for directory in dirs.values():
        os.makedirs(directory, exist_ok=True)

    print(f"Plot directories created: {exp_dir}")
    return dirs


def plot_training_curves(history, trial_num, save_dir=None, arch_config=None):
    if hasattr(history, 'history'):
        history = history.history

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    if arch_config:
        fig.suptitle(f'{arch_config} - Trial {trial_num}', fontweight='bold', fontsize=14)

    if 'accuracy' in history:
        epochs = range(1, len(history['accuracy']) + 1)
        ax1.plot(epochs, history['accuracy'], 'b-', label='Training', linewidth=2)
        if 'val_accuracy' in history:
            ax1.plot(epochs, history['val_accuracy'], 'r--', label='Validation', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        title = 'Accuracy' if arch_config else f'Trial {trial_num}: Accuracy'
        ax1.set_title(title)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    if 'loss' in history:
        epochs = range(1, len(history['loss']) + 1)
        ax2.plot(epochs, history['loss'], 'b-', label='Training', linewidth=2)
        if 'val_loss' in history:
            ax2.plot(epochs, history['val_loss'], 'r--', label='Validation', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        title = 'Loss' if arch_config else f'Trial {trial_num}: Loss'
        ax2.set_title(title)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    saved_path = None
    if save_dir:
        filename = os.path.join(save_dir, f'trial_{trial_num:03d}_training.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
        saved_path = filename

    plt.close()
    return saved_path


def plot_energy_vs_accuracy(results, pareto_indices=None, save_dir=None, arch_config=None):
    fig, ax = plt.subplots(figsize=(10, 7))

    accuracies = [r.get('approx_accuracy', r.get('exact_accuracy', 0)) * 100 for r in results]
    energies = [r.get('energy', 0) for r in results]

    is_exact = []
    for r in results:
        if 'energy_per_layer' in r and r['energy_per_layer']:
            all_exact = all('1JJQ' in layer.get('multiplier', '') for layer in r['energy_per_layer'])
            is_exact.append(all_exact)
        else:
            is_exact.append(False)

    exact_acc = [acc for acc, exact in zip(accuracies, is_exact) if exact]
    exact_eng = [eng for eng, exact in zip(energies, is_exact) if exact]
    approx_acc = [acc for acc, exact in zip(accuracies, is_exact) if not exact]
    approx_eng = [eng for eng, exact in zip(energies, is_exact) if not exact]

    ax.scatter(exact_eng, exact_acc, c='blue', marker='o', s=100, alpha=0.6,
               label='Exact multiplier', edgecolors='black', linewidths=1)
    ax.scatter(approx_eng, approx_acc, c='green', marker='s', s=100, alpha=0.6,
               label='Approximate multiplier', edgecolors='black', linewidths=1)

    if pareto_indices:
        pareto_acc = [accuracies[i] for i in pareto_indices]
        pareto_eng = [energies[i] for i in pareto_indices]
        ax.scatter(pareto_eng, pareto_acc, c='red', marker='*', s=400,
                   label='Pareto-optimal', edgecolors='black', linewidths=2, zorder=5)

    ax.set_xlabel('Energy (µJ)', fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontweight='bold')

    title = 'Energy-Accuracy Tradeoff'
    if arch_config:
        title = f'{arch_config}\n{title}'
    ax.set_title(title, fontweight='bold', fontsize=16)

    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()

    saved_path = None
    if save_dir:
        filename = os.path.join(save_dir, 'energy_vs_accuracy.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
        saved_path = filename

    plt.close()
    return saved_path


def plot_pareto_front(results, pareto_indices, quality_constraint=None, save_dir=None, arch_config=None):
    fig, ax = plt.subplots(figsize=(10, 7))

    accuracies = [r.get('approx_accuracy', r.get('exact_accuracy', 0)) * 100 for r in results]
    energies = [r.get('energy', 0) for r in results]
    stl_satisfied = [r.get('stl_robustness', 0) > 0 for r in results]

    satisfied = [(acc, eng) for acc, eng, sat in zip(accuracies, energies, stl_satisfied) if sat]
    violated = [(acc, eng) for acc, eng, sat in zip(accuracies, energies, stl_satisfied) if not sat]

    if satisfied:
        sat_acc, sat_eng = zip(*satisfied)
        ax.scatter(sat_eng, sat_acc, c='green', marker='o', s=100, alpha=0.5,
                   label='STL Satisfied', edgecolors='black', linewidths=1)

    if violated:
        viol_acc, viol_eng = zip(*violated)
        ax.scatter(viol_eng, viol_acc, c='red', marker='x', s=100, alpha=0.5,
                   label='STL Violated', linewidths=2)

    pareto_acc = [accuracies[i] for i in pareto_indices]
    pareto_eng = [energies[i] for i in pareto_indices]

    pareto_sorted = sorted(zip(pareto_eng, pareto_acc))
    if pareto_sorted:
        p_eng, p_acc = zip(*pareto_sorted)
        ax.plot(p_eng, p_acc, 'b-', linewidth=3, alpha=0.7, label='Pareto Front')
        ax.scatter(p_eng, p_acc, c='blue', marker='*', s=400,
                   edgecolors='black', linewidths=2, zorder=5)

    if quality_constraint:
        ax.axhline(y=quality_constraint * 100, color='orange', linestyle='--',
                   linewidth=2, label=f'Quality Constraint (Qc={quality_constraint*100:.0f}%)')

    ax.set_xlabel('Energy (µJ)', fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontweight='bold')

    title = 'Pareto Front with STL Constraints'
    if arch_config:
        title = f'{arch_config}\n{title}'
    ax.set_title(title, fontweight='bold', fontsize=16)

    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()

    saved_path = None
    if save_dir:
        filename = os.path.join(save_dir, 'pareto_front.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
        saved_path = filename

    plt.close()
    return saved_path


def plot_energy_breakdown(result, trial_num=None, save_dir=None, arch_config=None):
    if 'energy_per_layer' not in result or not result['energy_per_layer']:
        print("No energy breakdown data available")
        return None

    fig, ax = plt.subplots(figsize=(10, 6))

    layers = result['energy_per_layer']
    stages = [f"Stage {layer.get('stage', i)}" for i, layer in enumerate(layers)]
    energies = [layer.get('energy_uJ', layer.get('energy', 0)) for layer in layers]
    multipliers = [layer.get('multiplier', 'unknown') for layer in layers]

    colors = ['blue' if '1JJQ' in mult else 'green' for mult in multipliers]

    bars = ax.bar(stages, energies, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

    for bar, energy, mult in zip(bars, energies, multipliers):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{energy:.1f}\n({mult})',
                ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Stage', fontweight='bold')
    ax.set_ylabel('Energy (µJ)', fontweight='bold')

    title = f'Energy Breakdown by Stage'
    if arch_config:
        title = f'{arch_config}\n{title}'
    if trial_num is not None:
        title += f' - Trial {trial_num}'
    ax.set_title(title, fontweight='bold', fontsize=14)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', alpha=0.7, edgecolor='black', label='Exact (1JJQ)'),
        Patch(facecolor='green', alpha=0.7, edgecolor='black', label='Approximate')
    ]
    ax.legend(handles=legend_elements, loc='best')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')

    plt.tight_layout()

    saved_path = None
    if save_dir:
        filename_suffix = f'_trial_{trial_num:03d}' if trial_num else ''
        filename = os.path.join(save_dir, f'energy_breakdown{filename_suffix}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
        saved_path = filename

    plt.close()
    return saved_path


def plot_stl_analysis(results, save_dir=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    stl_scores = [r.get('stl_robustness', 0) for r in results if 'stl_robustness' in r]
    if not stl_scores:
        print("No STL data available")
        plt.close()
        return None

    satisfied = sum(1 for score in stl_scores if score > 0)
    violated = len(stl_scores) - satisfied

    ax1.pie([satisfied, violated], labels=['Satisfied', 'Violated'],
            autopct='%1.1f%%', colors=['green', 'red'], startangle=90,
            textprops={'fontsize': 12, 'fontweight': 'bold'})
    ax1.set_title('STL Constraint Satisfaction', fontweight='bold', fontsize=14)

    ax2.hist(stl_scores, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Satisfaction Threshold')
    ax2.set_xlabel('STL Robustness Score', fontweight='bold')
    ax2.set_ylabel('Frequency', fontweight='bold')
    ax2.set_title('Robustness Score Distribution', fontweight='bold', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    saved_path = None
    if save_dir:
        filename = os.path.join(save_dir, 'stl_analysis.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
        saved_path = filename

    plt.close()
    return saved_path


def generate_all_plots(results, pareto_indices=None, quality_constraint=None, experiment_name='nas', arch_config=None):
    print(f"\n{'='*60}")
    print(f"Generating plots for experiment: {experiment_name}")
    if arch_config:
        print(f"Configuration: {arch_config}")
    print(f"{'='*60}")

    dirs = setup_plot_dirs(experiment_name)

    plots_generated = {}

    plots_generated['energy_vs_accuracy'] = plot_energy_vs_accuracy(
        results, pareto_indices, save_dir=dirs['energy'], arch_config=arch_config
    )

    if pareto_indices:
        plots_generated['pareto_front'] = plot_pareto_front(
            results, pareto_indices, quality_constraint, save_dir=dirs['energy'], arch_config=arch_config
        )

    plots_generated['stl_analysis'] = plot_stl_analysis(results, save_dir=dirs['stl'])

    best = max(results, key=lambda x: x.get('approx_accuracy', x.get('exact_accuracy', 0)))
    plots_generated['energy_breakdown'] = plot_energy_breakdown(
        best, trial_num=None, save_dir=dirs['energy'], arch_config=arch_config
    )

    print(f"{'='*60}")
    print(f"All plots saved to: {dirs['base']}")
    print(f"{'='*60}\n")

    return plots_generated
