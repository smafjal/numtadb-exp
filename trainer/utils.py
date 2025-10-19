"""
Utility functions for training
"""

import torch
import numpy as np
import random
import pandas as pd
from typing import List
import logging

logger = logging.getLogger("trainer")


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed}")


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss: float):
        """Check if training should stop"""
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            logger.debug(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


class MetricTracker:
    """Track training metrics"""
    
    def __init__(self):
        self.epochs = []
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []
        self.learning_rates = []
    
    def update(self, epoch: int, train_loss: float, train_acc: float,
               val_loss: float, val_acc: float, lr: float):
        """Update metrics"""
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.train_accs.append(train_acc)
        self.val_losses.append(val_loss)
        self.val_accs.append(val_acc)
        self.learning_rates.append(lr)
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'epoch': self.epochs,
            'train_loss': self.train_losses,
            'train_acc': self.train_accs,
            'val_loss': self.val_losses,
            'val_acc': self.val_accs,
            'learning_rate': self.learning_rates
        }
    
    def to_dataframe(self):
        """Convert to pandas DataFrame"""
        return pd.DataFrame(self.to_dict())


def save_metrics(metrics: MetricTracker, filepath: str):
    """Save metrics to CSV file"""
    df = metrics.to_dataframe()
    df.to_csv(filepath, index=False)
    logger.info(f"Metrics saved to {filepath}")


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count trainable parameters in a model
    
    Args:
        model: PyTorch model
    
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter:
    """Computes and stores the average and current value"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    """
    Computes the accuracy over the k top predictions
    
    Args:
        output: Model output
        target: Ground truth labels
        topk: Tuple of top-k values to compute
    
    Returns:
        List of top-k accuracies
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

