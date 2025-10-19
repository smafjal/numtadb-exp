"""
Configuration file for training models on NumtaDB dataset
"""

import torch
import logging
from pathlib import Path

class Config:
    """Training configuration"""
    
    # Paths
    DATA_DIR = Path("data/raw")
    PROCESSED_DIR = Path("data/processed")
    MODEL_DIR = Path("models")
    LOG_DIR = Path("logs")
    CHECKPOINT_DIR = Path("checkpoints")
    
    # Logging
    LOG_LEVEL = logging.INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    CONSOLE_LOGGING = True
    FILE_LOGGING = True
    
    # Data parameters
    IMAGE_SIZE = 224  # MobileNetV2 default input size
    NUM_CLASSES = 10  # Bengali digits 0-9
    TRAIN_SPLIT = 0.8
    VAL_SPLIT = 0.1
    TEST_SPLIT = 0.1
    
    # Training parameters
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    
    # Optimizer parameters
    OPTIMIZER = "adam"  # Options: 'adam', 'sgd', 'adamw'
    MOMENTUM = 0.9  # For SGD
    
    # Learning rate scheduler
    USE_SCHEDULER = True
    SCHEDULER_TYPE = "cosine"  # Options: 'step', 'cosine', 'plateau'
    STEP_SIZE = 10  # For StepLR
    GAMMA = 0.1  # For StepLR
    T_MAX = 50  # For CosineAnnealingLR
    
    # Model parameters
    MODEL_NAME = "mobilenetv2"  # Options: 'alexnet', 'mobilenetv2'
    PRETRAINED = True
    DROPOUT = 0.2
    FREEZE_BACKBONE = False  # Set to True for transfer learning
    
    # Data augmentation
    USE_AUGMENTATION = True
    ROTATION_DEGREES = 15
    HORIZONTAL_FLIP = False  # Not recommended for digits
    VERTICAL_FLIP = False
    
    # Training settings
    NUM_WORKERS = 4
    PIN_MEMORY = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Checkpointing
    SAVE_EVERY = 5  # Save checkpoint every N epochs
    SAVE_BEST_ONLY = True
    
    # Early stopping
    USE_EARLY_STOPPING = True
    PATIENCE = 10
    MIN_DELTA = 0.001
    
    # Logging
    LOG_INTERVAL = 10  # Print loss every N batches
    
    # Random seed for reproducibility
    SEED = 42
    
    @classmethod
    def create_dirs(cls):
        """Create necessary directories"""
        for dir_path in [cls.PROCESSED_DIR, cls.MODEL_DIR, 
                         cls.LOG_DIR, cls.CHECKPOINT_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def print_config(cls, logger=None):
        """Print configuration"""
        log_func = logger.info if logger else print
        
        log_func("\n" + "="*60)
        log_func("Training Configuration")
        log_func("="*60)
        log_func(f"Model: {cls.MODEL_NAME}")
        log_func(f"Device: {cls.DEVICE}")
        log_func(f"Batch Size: {cls.BATCH_SIZE}")
        log_func(f"Learning Rate: {cls.LEARNING_RATE}")
        log_func(f"Epochs: {cls.NUM_EPOCHS}")
        log_func(f"Image Size: {cls.IMAGE_SIZE}")
        log_func(f"Pretrained: {cls.PRETRAINED}")
        log_func(f"Use Augmentation: {cls.USE_AUGMENTATION}")
        log_func(f"Use Scheduler: {cls.USE_SCHEDULER}")
        log_func(f"Log Directory: {cls.LOG_DIR}")
        log_func("="*60 + "\n")

