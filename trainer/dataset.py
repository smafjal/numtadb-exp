"""
Dataset and DataLoader utilities for NumtaDB
"""

import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from typing import Tuple, Optional
from trainer.config import Config
import logging

logger = logging.getLogger("trainer")


class NumtaDBDataset(Dataset):
    """NumtaDB Dataset class"""
    
    def __init__(self, data_dir: str, transform=None, csv_file: Optional[str] = None):
        """
        Args:
            data_dir: Directory with all the images
            transform: Optional transform to be applied on a sample
            csv_file: Optional CSV file with image paths and labels
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        # Load data
        if csv_file and os.path.exists(csv_file):
            self.df = pd.read_csv(csv_file)
            self.image_paths = self.df['image_path'].values
            self.labels = self.df['label'].values
        else:
            # If no CSV, try to load from directory structure
            self.image_paths, self.labels = self._load_from_directory()
        
        logger.info(f"Loaded {len(self.image_paths)} images with {len(set(self.labels))} classes")
    
    def _load_from_directory(self):
        """Load images from directory structure with CSV files"""
        image_paths = []
        labels = []
        
        # Check for NumtaDB structure with CSV files
        csv_files = list(self.data_dir.glob("training-*.csv"))
        
        if csv_files:
            # Load from CSV files (NumtaDB format)
            logger.info(f"Found {len(csv_files)} CSV files, loading NumtaDB dataset...")
            for csv_file in csv_files:
                df = pd.read_csv(csv_file)
                # Get the corresponding folder name (e.g., training-a.csv -> training-a/)
                folder_name = csv_file.stem  # e.g., "training-a"
                folder_path = self.data_dir / folder_name
                
                if folder_path.exists():
                    for _, row in df.iterrows():
                        # CSV format: filename, original filename, scanid, digit, ...
                        img_name = row['filename']
                        digit = int(row['digit'])
                        img_path = folder_path / img_name
                        
                        if img_path.exists():
                            image_paths.append(str(img_path))
                            labels.append(digit)
            
            logger.info(f"Loaded {len(image_paths)} images from CSV files")
        
        # Check if data is organized in class folders (0-9)
        elif (self.data_dir / "0").exists():
            for class_idx in range(10):
                class_dir = self.data_dir / str(class_idx)
                if class_dir.exists():
                    for img_file in class_dir.glob("*.png"):
                        image_paths.append(str(img_file))
                        labels.append(class_idx)
                    for img_file in class_dir.glob("*.jpg"):
                        image_paths.append(str(img_file))
                        labels.append(class_idx)
                    for img_file in class_dir.glob("*.JPG"):
                        image_paths.append(str(img_file))
                        labels.append(class_idx)
        else:
            # Try to load all images from flat directory
            for img_file in self.data_dir.glob("*.png"):
                image_paths.append(str(img_file))
                # Try to extract label from filename
                try:
                    label = int(img_file.stem.split('_')[0])
                    labels.append(label)
                except:
                    labels.append(0)  # Default label
        
        if len(image_paths) == 0:
            logger.warning(f"No images found in {self.data_dir}")
            logger.warning(f"Please ensure the dataset is downloaded to this directory.")
            logger.warning(f"Looking for training-*.csv files or class folders (0-9)")
        
        return np.array(image_paths), np.array(labels)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(augment: bool = True, image_size: int = 224):
    """
    Get training and validation transforms
    
    Args:
        augment: Whether to apply data augmentation
        image_size: Size of the output image
    
    Returns:
        train_transform, val_transform
    """
    
    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(Config.ROTATION_DEGREES),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


def create_dataloaders(data_dir: str, config: Config) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders
    
    Args:
        data_dir: Directory containing the dataset
        config: Configuration object
    
    Returns:
        train_loader, val_loader, test_loader
    """
    
    # Get transforms
    train_transform, val_transform = get_transforms(
        augment=config.USE_AUGMENTATION,
        image_size=config.IMAGE_SIZE
    )
    
    # Create full dataset
    full_dataset = NumtaDBDataset(data_dir, transform=None)
    
    # Calculate split sizes
    total_size = len(full_dataset)
    train_size = int(config.TRAIN_SPLIT * total_size)
    val_size = int(config.VAL_SPLIT * total_size)
    test_size = total_size - train_size - val_size
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(config.SEED)
    )
    
    # Apply transforms
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    test_dataset.dataset.transform = val_transform
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader

