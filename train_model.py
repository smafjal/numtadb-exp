#!/usr/bin/env python3
"""
Main script to train models on NumtaDB dataset

Usage:
    python train_model.py --model alexnet
    python train_model.py --model mobilenetv2
    python train_model.py --model alexnet --epochs 30 --batch-size 64
"""

import sys
import argparse
from pathlib import Path

# Add trainer to path
sys.path.insert(0, str(Path(__file__).parent))

from trainer.train import main
from trainer.config import Config
from trainer.model import get_available_models


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Train models on NumtaDB dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Model selection
    parser.add_argument(
        '--model', '-m',
        type=str,
        default=Config.MODEL_NAME,
        choices=get_available_models(),
        help=f'Model architecture to train (default: {Config.MODEL_NAME})'
    )
    
    # Training parameters
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--dropout', type=float, help='Dropout rate')
    
    # Model parameters
    parser.add_argument(
        '--no-pretrained',
        action='store_true',
        help='Do not use pretrained weights'
    )
    parser.add_argument(
        '--freeze-backbone',
        action='store_true',
        help='Freeze backbone for transfer learning'
    )
    
    # Resume training
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from'
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Override config with command line arguments
    if args.model:
        Config.MODEL_NAME = args.model
    if args.epochs:
        Config.NUM_EPOCHS = args.epochs
    if args.batch_size:
        Config.BATCH_SIZE = args.batch_size
    if args.lr:
        Config.LEARNING_RATE = args.lr
    if args.dropout:
        Config.DROPOUT = args.dropout
    if args.no_pretrained:
        Config.PRETRAINED = False
    if args.freeze_backbone:
        Config.FREEZE_BACKBONE = True
    if args.resume:
        Config.RESUME_CHECKPOINT = args.resume
    
    # Run training
    main()

