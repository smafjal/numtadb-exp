"""
Multi-model architecture support for NumtaDB classification
Supports: AlexNet, MobileNetV2, and more
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional, Dict, Type
import logging

logger = logging.getLogger("trainer")


# ============================================================================
# AlexNet Architecture
# ============================================================================
class AlexNet(nn.Module):
    """AlexNet model for Bengali digit classification - Simple and fast to train"""
    
    def __init__(
        self, 
        num_classes: int = 10, 
        pretrained: bool = False,
        dropout: float = 0.5,
        freeze_backbone: bool = False
    ):
        """
        Args:
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            dropout: Dropout probability
            freeze_backbone: Whether to freeze backbone weights
        """
        super(AlexNet, self).__init__()
        
        # AlexNet-style architecture adapted for smaller images
        self.features = nn.Sequential(
            # Conv1: 224x224x3 -> 55x55x64
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Conv2: 55x55x64 -> 27x27x192
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Conv3: 27x27x192 -> 13x13x384
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv4: 13x13x384 -> 13x13x256
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv5: 13x13x256 -> 6x6x256
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        # Adaptive pooling to handle different input sizes
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        
        # Initialize weights
        self._initialize_weights()
        
        self.num_classes = num_classes
        
        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False
            logger.info("AlexNet backbone frozen for transfer learning")
    
    def _initialize_weights(self):
        """Initialize weights using He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass"""
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def unfreeze_backbone(self):
        """Unfreeze the backbone for fine-tuning"""
        for param in self.features.parameters():
            param.requires_grad = True
        logger.info("AlexNet backbone unfrozen")
    
    def get_num_params(self):
        """Get number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# MobileNetV2 Architecture
# ============================================================================
class MobileNetV2Classifier(nn.Module):
    """MobileNetV2 model for Bengali digit classification"""
    
    def __init__(
        self, 
        num_classes: int = 10, 
        pretrained: bool = True,
        dropout: float = 0.2,
        freeze_backbone: bool = False
    ):
        """
        Args:
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            dropout: Dropout probability
            freeze_backbone: Whether to freeze backbone weights
        """
        super(MobileNetV2Classifier, self).__init__()
        
        # Load MobileNetV2
        if pretrained:
            weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
            self.mobilenet = models.mobilenet_v2(weights=weights)
        else:
            self.mobilenet = models.mobilenet_v2(weights=None)
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.mobilenet.features.parameters():
                param.requires_grad = False
            logger.info("MobileNetV2 backbone frozen for transfer learning")
        
        # Get the number of input features for the classifier
        in_features = self.mobilenet.classifier[1].in_features
        
        # Replace the classifier
        self.mobilenet.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes)
        )
        
        self.num_classes = num_classes
        
    def forward(self, x):
        """Forward pass"""
        return self.mobilenet(x)
    
    def unfreeze_backbone(self):
        """Unfreeze the backbone for fine-tuning"""
        for param in self.mobilenet.features.parameters():
            param.requires_grad = True
        logger.info("MobileNetV2 backbone unfrozen")
    
    def get_num_params(self):
        """Get number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# Model Registry
# ============================================================================
MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {
    'alexnet': AlexNet,
    'mobilenetv2': MobileNetV2Classifier,
}


def get_available_models():
    """Get list of available model names"""
    return list(MODEL_REGISTRY.keys())


def create_model(
    model_name: str = 'mobilenetv2',
    num_classes: int = 10, 
    pretrained: bool = True, 
    dropout: float = 0.2, 
    freeze_backbone: bool = False
) -> nn.Module:
    """
    Create and return a model by name
    
    Args:
        model_name: Name of the model ('alexnet', 'mobilenetv2')
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        dropout: Dropout probability
        freeze_backbone: Whether to freeze backbone
    
    Returns:
        Model instance
    """
    model_name = model_name.lower()
    
    if model_name not in MODEL_REGISTRY:
        available = ', '.join(get_available_models())
        raise ValueError(
            f"Unknown model: {model_name}. Available models: {available}"
        )
    
    model_class = MODEL_REGISTRY[model_name]
    model = model_class(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        freeze_backbone=freeze_backbone
    )
    
    num_params = model.get_num_params()
    logger.info(f"{model_name.upper()} created with {num_params:,} trainable parameters")
    
    return model


def load_checkpoint(model: nn.Module, checkpoint_path: str, device: torch.device):
    """
    Load model from checkpoint
    
    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
    
    Returns:
        model, epoch, best_acc
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint.get('epoch', 0)
    best_acc = checkpoint.get('best_acc', 0.0)
    
    logger.info(f"Loaded checkpoint from epoch {epoch} with accuracy {best_acc:.4f}")
    
    return model, epoch, best_acc


def save_checkpoint(model: nn.Module, optimizer, epoch: int, 
                   best_acc: float, filepath: str):
    """
    Save model checkpoint
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        best_acc: Best validation accuracy
        filepath: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc
    }
    torch.save(checkpoint, filepath)
    logger.info(f"Checkpoint saved to {filepath}")

