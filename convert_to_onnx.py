#!/usr/bin/env python3
"""
Convert trained PyTorch model to ONNX format for web deployment
"""

import torch
import sys
from pathlib import Path
from trainer.model import create_model
from trainer.config import Config

def convert_to_onnx(
    checkpoint_path: str,
    output_path: str,
    model_name: str = 'mobilenetv2',
    num_classes: int = 10
):
    """
    Convert PyTorch model to ONNX format
    
    Args:
        checkpoint_path: Path to the trained model checkpoint
        output_path: Path to save the ONNX model
        model_name: Name of the model architecture
        num_classes: Number of output classes
    """
    print(f"Loading model: {model_name}")
    
    # Create model
    model = create_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=False,
        dropout=0.2
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded from {checkpoint_path}")
    print(f"Epoch: {checkpoint.get('epoch', 'N/A')}, Accuracy: {checkpoint.get('best_acc', 'N/A'):.4f}")
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"\n✓ Model successfully converted to ONNX format")
    print(f"  Saved to: {output_path}")
    print(f"  File size: {Path(output_path).stat().st_size / (1024*1024):.2f} MB")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert PyTorch model to ONNX')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints/best_model.pth',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='models/best_model.onnx',
        help='Output ONNX file path'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='mobilenetv2',
        choices=['alexnet', 'mobilenetv2'],
        help='Model architecture'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # Convert
    convert_to_onnx(args.checkpoint, args.output, args.model)

