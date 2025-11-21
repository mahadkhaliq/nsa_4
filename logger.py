"""
Logging utility for NAS experiments with approximate multipliers

Provides timestamped logging to both console and files for tracking:
- Experiment configurations
- Trial results (accuracy, energy, STL metrics)
- Architecture details
- Training progress (epoch-by-epoch)
- Network architecture details
"""

import logging
import os
import sys
from datetime import datetime
import json
from io import StringIO
import numpy as np
import tensorflow as tf
from tensorflow import keras


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class NASLogger:
    """Custom logger for Neural Architecture Search experiments"""

    def __init__(self, log_dir='logs', experiment_name='nas'):
        """Initialize logger with timestamped log files

        Args:
            log_dir: Directory to save log files
            experiment_name: Name prefix for log files
        """
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Create logs directory
        os.makedirs(log_dir, exist_ok=True)

        # Setup different log files
        self.main_log_file = os.path.join(log_dir, f'{experiment_name}_{self.timestamp}.log')
        self.results_json_file = os.path.join(log_dir, f'{experiment_name}_{self.timestamp}_results.json')

        # Configure main logger
        self.logger = logging.getLogger('NAS')
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = []  # Clear existing handlers

        # File handler
        file_handler = logging.FileHandler(self.main_log_file)
        file_handler.setLevel(logging.INFO)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Formatter with timestamp
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        # Storage for experiment results
        self.experiment_results = []
        self.experiment_config = {}

    def info(self, message):
        """Log info message"""
        self.logger.info(message)

    def warning(self, message):
        """Log warning message"""
        self.logger.warning(message)

    def error(self, message):
        """Log error message"""
        self.logger.error(message)

    def header(self, message):
        """Log a header with separators"""
        separator = '=' * 60
        self.logger.info(separator)
        self.logger.info(message)
        self.logger.info(separator)

    def subheader(self, message):
        """Log a subheader with dashes"""
        self.logger.info('-' * 40)
        self.logger.info(message)
        self.logger.info('-' * 40)

    def log_config(self, config):
        """Log experiment configuration

        Args:
            config: Dict with experiment settings
        """
        self.experiment_config = config
        self.experiment_config['timestamp'] = self.timestamp

        self.header("EXPERIMENT CONFIGURATION")
        for key, value in config.items():
            self.logger.info(f"  {key}: {value}")

    def log_trial(self, trial_num, total_trials, arch, result):
        """Log results from a single trial

        Args:
            trial_num: Current trial number (1-indexed)
            total_trials: Total number of trials
            arch: Architecture dict
            result: Result dict with accuracy, energy, STL metrics
        """
        self.subheader(f"Trial {trial_num}/{total_trials}")

        # Log architecture
        self.logger.info(f"Architecture: {arch}")

        # Log accuracy metrics
        if 'exact_accuracy' in result:
            self.logger.info(f"Exact accuracy: {result['exact_accuracy']:.4f}")
        if 'approx_accuracy' in result:
            self.logger.info(f"Approx accuracy: {result['approx_accuracy']:.4f}")
            if 'exact_accuracy' in result:
                acc_drop = result['exact_accuracy'] - result['approx_accuracy']
                self.logger.info(f"Accuracy drop: {acc_drop:.4f}")

        # Log energy metrics
        if 'energy' in result:
            self.logger.info(f"Total energy: {result['energy']:.4f} µJ")

        # Log STL metrics
        if 'stl_robustness' in result:
            status = "SATISFIED" if result['stl_robustness'] > 0 else "VIOLATED"
            self.logger.info(f"STL robustness: {result['stl_robustness']:.4f} ({status})")

        # Store result
        trial_data = {
            'trial': trial_num,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'architecture': str(arch),
            'results': result
        }
        self.experiment_results.append(trial_data)

    def log_energy_breakdown(self, energy_per_layer):
        """Log detailed energy breakdown per layer

        Args:
            energy_per_layer: List of dicts with layer energy info
        """
        if not energy_per_layer:
            return

        self.logger.info("Energy breakdown per stage:")
        for layer_info in energy_per_layer:
            stage = layer_info.get('stage', layer_info.get('layer', '?'))
            mult = layer_info.get('multiplier', 'unknown')
            energy_uj = layer_info.get('energy_uJ', layer_info.get('energy', 0))
            macs = layer_info.get('macs', 0)

            self.logger.info(
                f"  Stage {stage}: {mult} - "
                f"{energy_uj:.4f} µJ ({macs:,} MACs)"
            )

    def log_network_architecture(self, model):
        """Log network architecture details

        Args:
            model: Keras model to log
        """
        self.subheader("Network Architecture")

        # Capture model summary
        summary_buffer = StringIO()
        model.summary(print_fn=lambda x: summary_buffer.write(x + '\n'))
        summary_str = summary_buffer.getvalue()

        # Log to both file and console
        for line in summary_str.split('\n'):
            if line.strip():
                self.logger.info(line)

        # Log total parameters
        total_params = model.count_params()
        trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")

    def log_epoch(self, epoch, total_epochs, metrics):
        """Log training progress for an epoch

        Args:
            epoch: Current epoch number (0-indexed)
            total_epochs: Total number of epochs
            metrics: Dict with loss, accuracy, val_loss, val_accuracy, etc.
        """
        msg = f"Epoch {epoch+1}/{total_epochs}"

        # Add training metrics
        if 'loss' in metrics:
            msg += f" - loss: {metrics['loss']:.4f}"
        if 'accuracy' in metrics:
            msg += f" - accuracy: {metrics['accuracy']:.4f}"

        # Add validation metrics
        if 'val_loss' in metrics:
            msg += f" - val_loss: {metrics['val_loss']:.4f}"
        if 'val_accuracy' in metrics:
            msg += f" - val_accuracy: {metrics['val_accuracy']:.4f}"

        # Add learning rate if available
        if 'lr' in metrics:
            msg += f" - lr: {metrics['lr']:.6f}"

        self.logger.info(msg)

    def log_training_start(self, trial_num, arch):
        """Log the start of training for a trial

        Args:
            trial_num: Current trial number
            arch: Architecture dict
        """
        self.logger.info(f"\n[Training Started] Trial {trial_num}")
        self.logger.info(f"Architecture: {arch}")

    def log_summary(self, results):
        """Log experiment summary with best results and Pareto front

        Args:
            results: List of all trial results
        """
        self.header("EXPERIMENT SUMMARY")

        # Find best by accuracy
        best_acc = max(results, key=lambda x: x.get('approx_accuracy', x.get('exact_accuracy', 0)))
        self.logger.info("Best architecture by accuracy:")
        acc = best_acc.get('approx_accuracy', best_acc.get('exact_accuracy', 0))
        energy = best_acc.get('energy', 0)
        self.logger.info(f"  Accuracy: {acc:.4f}")
        self.logger.info(f"  Energy: {energy:.4f} µJ")
        self.logger.info(f"  Architecture: {best_acc.get('arch', 'N/A')}")

        # Find best by energy
        energy_results = [r for r in results if 'energy' in r]
        if energy_results:
            best_energy = min(energy_results, key=lambda x: x['energy'])
            self.logger.info("\nBest architecture by energy:")
            acc = best_energy.get('approx_accuracy', best_energy.get('exact_accuracy', 0))
            energy = best_energy.get('energy', 0)
            self.logger.info(f"  Accuracy: {acc:.4f}")
            self.logger.info(f"  Energy: {energy:.4f} µJ")
            self.logger.info(f"  Architecture: {best_energy.get('arch', 'N/A')}")

        # STL statistics
        stl_results = [r for r in results if 'stl_robustness' in r]
        if stl_results:
            satisfied = sum(1 for r in stl_results if r['stl_robustness'] > 0)
            total = len(stl_results)
            violation_rate = (total - satisfied) / total * 100
            self.logger.info(f"\nSTL Constraint Statistics:")
            self.logger.info(f"  Satisfied: {satisfied}/{total} ({100-violation_rate:.1f}%)")
            self.logger.info(f"  Violated: {total-satisfied}/{total} ({violation_rate:.1f}%)")

    def save_results(self):
        """Save experiment results to JSON file"""
        results_data = {
            'config': self.experiment_config,
            'timestamp': self.timestamp,
            'results': self.experiment_results
        }

        with open(self.results_json_file, 'w') as f:
            json.dump(results_data, f, indent=2, cls=NumpyEncoder)

        self.logger.info(f"\nResults saved to: {self.results_json_file}")
        self.logger.info(f"Log file: {self.main_log_file}")

    def get_log_files(self):
        """Return paths to log files

        Returns:
            dict: Paths to log and results files
        """
        return {
            'log_file': self.main_log_file,
            'results_file': self.results_json_file
        }

    def create_training_callback(self):
        """Create a Keras callback for logging training progress

        Returns:
            TrainingLoggerCallback: Callback for epoch-wise logging
        """
        return TrainingLoggerCallback(self)


class TrainingLoggerCallback(keras.callbacks.Callback):
    """Keras callback to log training progress with timestamps"""

    def __init__(self, nas_logger):
        """Initialize callback with NAS logger

        Args:
            nas_logger: NASLogger instance
        """
        super().__init__()
        self.nas_logger = nas_logger
        self.total_epochs = None

    def on_train_begin(self, logs=None):
        """Called at the beginning of training"""
        if hasattr(self.params, 'epochs'):
            self.total_epochs = self.params['epochs']
        self.nas_logger.info("Training started...")

    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of each epoch

        Args:
            epoch: Current epoch number (0-indexed)
            logs: Dict of metrics from the epoch
        """
        logs = logs or {}

        # Get learning rate if available
        if hasattr(self.model.optimizer, 'learning_rate'):
            lr = float(keras.backend.get_value(self.model.optimizer.learning_rate))
            logs['lr'] = lr

        # Log epoch metrics
        self.nas_logger.log_epoch(epoch, self.total_epochs or epoch + 1, logs)

    def on_train_end(self, logs=None):
        """Called at the end of training"""
        self.nas_logger.info("Training completed!")
