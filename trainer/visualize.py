"""
Visualization utilities for training metrics and results
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

from trainer.config import Config

logger = logging.getLogger("trainer")


def plot_training_history(metrics_path: str, save_dir: str = None):
    """
    Plot training history from metrics CSV
    
    Args:
        metrics_path: Path to metrics CSV file
        save_dir: Directory to save plots
    """
    # Load metrics
    df = pd.read_csv(metrics_path)
    
    # Set style
    sns.set_style("whitegrid")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training History', fontsize=16, fontweight='bold')
    
    # Plot 1: Loss
    axes[0, 0].plot(df['epoch'], df['train_loss'], label='Train Loss', marker='o')
    axes[0, 0].plot(df['epoch'], df['val_loss'], label='Val Loss', marker='s')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Accuracy
    axes[0, 1].plot(df['epoch'], df['train_acc'], label='Train Acc', marker='o')
    axes[0, 1].plot(df['epoch'], df['val_acc'], label='Val Acc', marker='s')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Learning Rate
    axes[1, 0].plot(df['epoch'], df['learning_rate'], marker='o', color='green')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Overfitting Analysis
    train_val_diff = df['train_acc'] - df['val_acc']
    axes[1, 1].plot(df['epoch'], train_val_diff, marker='o', color='red')
    axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy Gap (%)')
    axes[1, 1].set_title('Train-Val Accuracy Gap (Overfitting Indicator)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        save_path = Path(save_dir) / 'training_history.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training history plot saved to {save_path}")
    
    plt.show()


def plot_loss_comparison(metrics_path: str, save_dir: str = None):
    """
    Plot detailed loss comparison
    
    Args:
        metrics_path: Path to metrics CSV file
        save_dir: Directory to save plots
    """
    df = pd.read_csv(metrics_path)
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss', marker='o', linewidth=2)
    plt.plot(df['epoch'], df['val_loss'], label='Val Loss', marker='s', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Loss Curves', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss', marker='o', linewidth=2)
    plt.plot(df['epoch'], df['val_loss'], label='Val Loss', marker='s', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (log scale)', fontsize=12)
    plt.title('Loss Curves (Log Scale)', fontsize=14, fontweight='bold')
    plt.yscale('log')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        save_path = Path(save_dir) / 'loss_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Loss comparison plot saved to {save_path}")
    
    plt.show()


def print_training_summary(metrics_path: str):
    """
    Print training summary statistics
    
    Args:
        metrics_path: Path to metrics CSV file
    """
    df = pd.read_csv(metrics_path)
    
    logger.info("\n" + "="*60)
    logger.info("Training Summary")
    logger.info("="*60)
    
    logger.info(f"\nTotal Epochs: {len(df)}")
    logger.info(f"\nBest Validation Accuracy: {df['val_acc'].max():.2f}% (Epoch {df['val_acc'].idxmax() + 1})")
    logger.info(f"Best Training Accuracy: {df['train_acc'].max():.2f}% (Epoch {df['train_acc'].idxmax() + 1})")
    
    logger.info(f"\nFinal Validation Accuracy: {df['val_acc'].iloc[-1]:.2f}%")
    logger.info(f"Final Training Accuracy: {df['train_acc'].iloc[-1]:.2f}%")
    
    logger.info(f"\nLowest Validation Loss: {df['val_loss'].min():.4f} (Epoch {df['val_loss'].idxmin() + 1})")
    logger.info(f"Lowest Training Loss: {df['train_loss'].min():.4f} (Epoch {df['train_loss'].idxmin() + 1})")
    
    logger.info(f"\nFinal Validation Loss: {df['val_loss'].iloc[-1]:.4f}")
    logger.info(f"Final Training Loss: {df['train_loss'].iloc[-1]:.4f}")
    
    # Overfitting analysis
    final_gap = df['train_acc'].iloc[-1] - df['val_acc'].iloc[-1]
    logger.info(f"\nFinal Train-Val Gap: {final_gap:.2f}%")
    
    if final_gap > 10:
        logger.warning("⚠️  High overfitting detected")
    elif final_gap > 5:
        logger.warning("⚠️  Moderate overfitting detected")
    else:
        logger.info("✓ Low overfitting - good generalization")
    
    logger.info("="*60 + "\n")


def main():
    """Main visualization function"""
    config = Config()
    metrics_path = config.LOG_DIR / 'training_metrics.csv'
    
    if not metrics_path.exists():
        logger.error(f"Metrics file not found at {metrics_path}")
        logger.error("Please train the model first using train.py")
        return
    
    # Print summary
    print_training_summary(str(metrics_path))
    
    # Plot training history
    logger.info("Generating training history plots...")
    plot_training_history(str(metrics_path), save_dir=str(config.LOG_DIR))
    
    # Plot loss comparison
    logger.info("Generating loss comparison plots...")
    plot_loss_comparison(str(metrics_path), save_dir=str(config.LOG_DIR))
    
    logger.info("\n" + "="*60)
    logger.info("Visualization complete!")
    logger.info("="*60)


if __name__ == "__main__":
    main()

