"""
Inference script for single image prediction
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision import transforms
import argparse
import logging

from trainer.config import Config
from trainer.model import create_model

logger = logging.getLogger("trainer")


class BengaliDigitPredictor:
    """Predictor class for Bengali digit recognition"""
    
    def __init__(self, model_path: str, config: Config):
        """
        Args:
            model_path: Path to trained model
            config: Configuration object
        """
        self.config = config
        self.device = config.DEVICE
        
        # Load model
        self.model = create_model(
            model_name=config.MODEL_NAME,
            num_classes=config.NUM_CLASSES,
            pretrained=False,
            dropout=config.DROPOUT
        ).to(self.device)
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        logger.info(f"Model loaded from {model_path}")
        
        # Define transform
        self.transform = transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def predict(self, image_path: str, return_probs: bool = False):
        """
        Predict digit from image
        
        Args:
            image_path: Path to image file
            return_probs: Whether to return probability distribution
        
        Returns:
            Predicted class (and probabilities if return_probs=True)
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            output = self.model(image_tensor)
            probs = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probs, 1)
        
        predicted_class = predicted.item()
        confidence_score = confidence.item()
        
        if return_probs:
            return predicted_class, confidence_score, probs.cpu().numpy()[0]
        else:
            return predicted_class, confidence_score
    
    def predict_batch(self, image_paths: list):
        """
        Predict multiple images
        
        Args:
            image_paths: List of image paths
        
        Returns:
            List of (predicted_class, confidence) tuples
        """
        results = []
        for image_path in image_paths:
            pred, conf = self.predict(image_path)
            results.append((pred, conf))
        return results
    
    def predict_with_details(self, image_path: str):
        """
        Predict with detailed output
        
        Args:
            image_path: Path to image file
        """
        predicted_class, confidence, probs = self.predict(image_path, return_probs=True)
        
        logger.info(f"\nImage: {image_path}")
        logger.info(f"Predicted Digit: {predicted_class}")
        logger.info(f"Confidence: {confidence:.2%}")
        logger.info("\nProbability distribution:")
        logger.info("-" * 30)
        for i, prob in enumerate(probs):
            bar = "█" * int(prob * 50)
            logger.info(f"Class {i}: {prob:.4f} {bar}")


def main():
    """Main inference function"""
    parser = argparse.ArgumentParser(description='Predict Bengali digit from image')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to image file')
    parser.add_argument('--checkpoint', '--model', type=str, default=None,
                       dest='checkpoint',
                       help='Path to model checkpoint')
    parser.add_argument('--model-name', type=str, default=None,
                       choices=['alexnet', 'mobilenetv2'],
                       help='Model architecture (default: from config)')
    parser.add_argument('--detailed', action='store_true',
                       help='Show detailed prediction results')
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config()
    
    # Override config if model name specified
    if args.model_name:
        config.MODEL_NAME = args.model_name
    
    # Setup logger
    from trainer.logger import setup_logger
    infer_logger = setup_logger(
        name="inference",
        log_dir=config.LOG_DIR,
        log_level=config.LOG_LEVEL,
        console_output=config.CONSOLE_LOGGING,
        file_output=config.FILE_LOGGING
    )
    
    # Get model path
    if args.checkpoint is None:
        model_path = config.CHECKPOINT_DIR / 'best_model.pth'
    else:
        model_path = Path(args.checkpoint)
    
    if not model_path.exists():
        infer_logger.error(f"Model not found at {model_path}")
        infer_logger.error("Please train the model first using train.py")
        return
    
    # Create predictor
    predictor = BengaliDigitPredictor(str(model_path), config)
    
    # Predict
    if args.detailed:
        predictor.predict_with_details(args.image)
    else:
        predicted_class, confidence = predictor.predict(args.image)
        infer_logger.info(f"\nPredicted Digit: {predicted_class}")
        infer_logger.info(f"Confidence: {confidence:.2%}")


if __name__ == "__main__":
    main()

