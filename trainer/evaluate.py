"""
Evaluation script for trained models
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path
from tqdm import tqdm
import logging
from trainer.config import Config
from trainer.model import create_model
from trainer.dataset import create_dataloaders

logger = logging.getLogger("trainer")   

class Evaluator:
    """Evaluator class for model evaluation"""
    
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
        
        self.criterion = nn.CrossEntropyLoss()
    
    def evaluate(self, test_loader):
        """
        Evaluate model on test set
        
        Args:
            test_loader: Test data loader
        
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        
        test_loss = 0
        correct = 0
        total = 0
        
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for inputs, targets in tqdm(test_loader, desc='Evaluating'):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                test_loss += loss.item()
                
                # Get predictions
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Store for metrics
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Calculate metrics
        test_loss = test_loss / len(test_loader)
        test_acc = 100. * correct / total
        
        results = {
            'loss': test_loss,
            'accuracy': test_acc,
            'predictions': np.array(all_preds),
            'targets': np.array(all_targets),
            'probabilities': np.array(all_probs)
        }
        
        logger.info(f'Test Loss: {test_loss:.4f}')
        logger.info(f'Test Accuracy: {test_acc:.2f}%')
        
        return results
    
    def plot_confusion_matrix(self, targets, predictions, save_path=None):
        """
        Plot confusion matrix
        
        Args:
            targets: Ground truth labels
            predictions: Predicted labels
            save_path: Path to save plot
        """
        cm = confusion_matrix(targets, predictions)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=range(self.config.NUM_CLASSES),
                   yticklabels=range(self.config.NUM_CLASSES))
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def print_classification_report(self, targets, predictions):
        """
        Print classification report
        
        Args:
            targets: Ground truth labels
            predictions: Predicted labels
        """
        class_names = [str(i) for i in range(self.config.NUM_CLASSES)]
        report = classification_report(targets, predictions, 
                                      target_names=class_names,
                                      digits=4)
        logger.info("\nClassification Report:")
        logger.info("=" * 60)
        logger.info(f"\n{report}")
    
    def plot_per_class_accuracy(self, targets, predictions, save_path=None):
        """
        Plot per-class accuracy
        
        Args:
            targets: Ground truth labels
            predictions: Predicted labels
            save_path: Path to save plot
        """
        cm = confusion_matrix(targets, predictions)
        per_class_acc = cm.diagonal() / cm.sum(axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(self.config.NUM_CLASSES), per_class_acc * 100)
        plt.xlabel('Class')
        plt.ylabel('Accuracy (%)')
        plt.title('Per-Class Accuracy')
        plt.xticks(range(self.config.NUM_CLASSES))
        plt.ylim([0, 105])
        
        for i, acc in enumerate(per_class_acc):
            plt.text(i, acc * 100 + 2, f'{acc*100:.1f}%', 
                    ha='center', va='bottom')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Per-class accuracy plot saved to {save_path}")
        
        plt.show()
    
    def analyze_errors(self, results, top_k=10):
        """
        Analyze misclassified samples
        
        Args:
            results: Dictionary with evaluation results
            top_k: Number of top errors to show
        """
        predictions = results['predictions']
        targets = results['targets']
        probs = results['probabilities']
        
        # Find misclassified samples
        errors = predictions != targets
        error_indices = np.where(errors)[0]
        
        logger.info(f"Total misclassified samples: {len(error_indices)}")
        logger.info(f"Error rate: {len(error_indices)/len(targets)*100:.2f}%")
        
        # Get confidence scores for errors
        error_confidences = []
        for idx in error_indices:
            pred_class = predictions[idx]
            confidence = probs[idx][pred_class]
            error_confidences.append((idx, targets[idx], predictions[idx], confidence))
        
        # Sort by confidence (high confidence errors are more interesting)
        error_confidences.sort(key=lambda x: x[3], reverse=True)
        
        logger.info(f"\nTop {top_k} high-confidence errors:")
        logger.info("Index | True | Pred | Confidence")
        logger.info("-" * 40)
        for i, (idx, true_label, pred_label, conf) in enumerate(error_confidences[:top_k]):
            logger.info(f"{idx:5d} | {true_label:4d} | {pred_label:4d} | {conf:6.2%}")


def main():
    """Main evaluation function"""
    # Load configuration
    config = Config()
    
    # Setup logger
    from trainer.logger import setup_logger
    eval_logger = setup_logger(
        name="evaluate",
        log_dir=config.LOG_DIR,
        log_level=config.LOG_LEVEL,
        console_output=config.CONSOLE_LOGGING,
        file_output=config.FILE_LOGGING
    )
    
    # Path to model checkpoint
    if hasattr(config, 'CHECKPOINT_PATH') and config.CHECKPOINT_PATH:
        model_path = Path(config.CHECKPOINT_PATH)
    else:
        model_path = config.CHECKPOINT_DIR / 'best_model.pth'
    
    if not model_path.exists():
        eval_logger.error(f"Model not found at {model_path}")
        eval_logger.error("Please train the model first using train.py")
        return
    
    # Create dataloaders
    eval_logger.info("Loading dataset...")
    _, _, test_loader = create_dataloaders(str(config.DATA_DIR), config)
    
    # Create evaluator
    evaluator = Evaluator(str(model_path), config)
    
    # Evaluate model
    eval_logger.info("Evaluating model...")
    results = evaluator.evaluate(test_loader)
    
    # Print classification report
    evaluator.print_classification_report(results['targets'], results['predictions'])
    
    # Plot confusion matrix
    cm_path = config.LOG_DIR / 'confusion_matrix.png'
    evaluator.plot_confusion_matrix(results['targets'], results['predictions'], 
                                   save_path=str(cm_path))
    
    # Plot per-class accuracy
    acc_path = config.LOG_DIR / 'per_class_accuracy.png'
    evaluator.plot_per_class_accuracy(results['targets'], results['predictions'],
                                     save_path=str(acc_path))
    
    # Analyze errors
    evaluator.analyze_errors(results, top_k=20)
    
    eval_logger.info("\n" + "="*60)
    eval_logger.info("Evaluation complete!")
    eval_logger.info("="*60)


if __name__ == "__main__":
    main()

