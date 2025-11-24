"""
Performance metrics tracking and visualization.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def save_metrics(metrics, filepath):
    """Save metrics dictionary to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Metrics saved to {filepath}")


def load_metrics(filepath):
    """Load metrics from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def plot_training_curves(train_losses, val_losses, save_path=None):
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses (list): Training losses per epoch
        val_losses (list): Validation losses per epoch
        save_path (str): Optional path to save figure
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Training curves saved to {save_path}")
    
    plt.show()


def generate_metrics_summary(results_dir):
    """
    Generate comprehensive metrics summary report.
    
    Args:
        results_dir (Path): Directory containing results
        
    Returns:
        dict: Summary statistics
    """
    results_dir = Path(results_dir)
    
    # Load training history
    history_file = results_dir / 'metrics' / 'training_history.json'
    if history_file.exists():
        history = load_metrics(history_file)
        
        summary = {
            'best_val_loss': min(history['val_losses']),
            'final_train_loss': history['train_losses'][-1],
            'total_epochs': len(history['train_losses']),
            'convergence_epoch': np.argmin(history['val_losses']) + 1
        }
        
        return summary
    else:
        print(f"Warning: {history_file} not found")
        return {}


if __name__ == "__main__":
    print("Metrics utilities loaded.")
