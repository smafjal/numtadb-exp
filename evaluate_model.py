#!/usr/bin/env python3
"""
Main script to evaluate trained models

Usage:
    python evaluate_model.py --model alexnet
    python evaluate_model.py --model mobilenetv2
    python evaluate_model.py --model alexnet --checkpoint checkpoints/best_model.pth
"""

import sys
import argparse
from pathlib import Path

# Add trainer to path
sys.path.insert(0, str(Path(__file__).parent))

from trainer.evaluate import main
from trainer.config import Config
from trainer.model import get_available_models


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Evaluate trained models on NumtaDB dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Model selection
    parser.add_argument(
        '--model', '-m',
        type=str,
        default=Config.MODEL_NAME,
        choices=get_available_models(),
        help=f'Model architecture to evaluate (default: {Config.MODEL_NAME})'
    )
    
    # Checkpoint path
    parser.add_argument(
        '--checkpoint', '-c',
        type=str,
        help='Path to model checkpoint (default: checkpoints/best_model.pth)'
    )
    
    # Model architecture  
    parser.add_argument(
        '--model-name',
        type=str,
        default=None,
        choices=get_available_models(),
        help='Override model architecture (useful if checkpoint filename doesn\'t match)'
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Override config with command line arguments
    if args.model:
        Config.MODEL_NAME = args.model
    if args.model_name:
        Config.MODEL_NAME = args.model_name
    if args.checkpoint:
        Config.CHECKPOINT_PATH = args.checkpoint
    
    # Run evaluation
    main()

