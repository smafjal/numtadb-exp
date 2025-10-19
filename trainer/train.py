"""
Training script for models on NumtaDB dataset
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm
import time
from pathlib import Path

from trainer.config import Config
from trainer.model import create_model, save_checkpoint
from trainer.dataset import create_dataloaders
from trainer.utils import set_seed, EarlyStopping, MetricTracker, save_metrics
from trainer.logger import setup_logger, get_logger


class Trainer:
    """Trainer class for MobileNetV2"""
    
    def __init__(self, config: Config):
        """
        Args:
            config: Configuration object
        """
        self.config = config
        self.device = config.DEVICE
        
        # Set random seed
        set_seed(config.SEED)
        
        # Create directories
        config.create_dirs()
        
        # Setup logger
        self.logger = setup_logger(
            name="trainer",
            log_dir=config.LOG_DIR,
            log_level=config.LOG_LEVEL,
            console_output=config.CONSOLE_LOGGING,
            file_output=config.FILE_LOGGING
        )
        
        self.logger.info("="*60)
        self.logger.info("Initializing Trainer")
        self.logger.info("="*60)
        
        # Create model
        self.logger.info(f"Creating {config.MODEL_NAME} model...")
        self.model = create_model(
            model_name=config.MODEL_NAME,
            num_classes=config.NUM_CLASSES,
            pretrained=config.PRETRAINED,
            dropout=config.DROPOUT,
            freeze_backbone=config.FREEZE_BACKBONE
        ).to(self.device)
        self.logger.info(f"Model created and moved to {self.device}")
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        
        # Learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Early stopping
        self.early_stopping = None
        if config.USE_EARLY_STOPPING:
            self.early_stopping = EarlyStopping(
                patience=config.PATIENCE,
                min_delta=config.MIN_DELTA
            )
        
        # Metrics tracker
        self.metrics = MetricTracker()
        
        # Best accuracy
        self.best_acc = 0.0
    
    def _create_optimizer(self):
        """Create optimizer"""
        if self.config.OPTIMIZER.lower() == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.LEARNING_RATE,
                weight_decay=self.config.WEIGHT_DECAY
            )
        elif self.config.OPTIMIZER.lower() == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config.LEARNING_RATE,
                weight_decay=self.config.WEIGHT_DECAY
            )
        elif self.config.OPTIMIZER.lower() == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=self.config.LEARNING_RATE,
                momentum=self.config.MOMENTUM,
                weight_decay=self.config.WEIGHT_DECAY
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.OPTIMIZER}")
    
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        if not self.config.USE_SCHEDULER:
            return None
        
        if self.config.SCHEDULER_TYPE == 'step':
            return StepLR(
                self.optimizer,
                step_size=self.config.STEP_SIZE,
                gamma=self.config.GAMMA
            )
        elif self.config.SCHEDULER_TYPE == 'cosine':
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.T_MAX
            )
        elif self.config.SCHEDULER_TYPE == 'plateau':
            return ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.1,
                patience=5
            )
        else:
            return None
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{self.config.NUM_EPOCHS}')
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Zero the gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            if (batch_idx + 1) % self.config.LOG_INTERVAL == 0:
                pbar.set_postfix({
                    'loss': f'{running_loss/(batch_idx+1):.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        val_loss = running_loss / len(val_loader)
        val_acc = 100. * correct / total
        
        return val_loss, val_acc
    
    def train(self, train_loader, val_loader):
        """Full training loop"""
        self.logger.info("\n" + "="*60)
        self.logger.info("Starting Training")
        self.logger.info("="*60)
        
        start_time = time.time()
        
        for epoch in range(1, self.config.NUM_EPOCHS + 1):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            
            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_acc)
                else:
                    self.scheduler.step()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log metrics
            self.metrics.update(epoch, train_loss, train_acc, val_loss, val_acc, current_lr)
            
            # Log epoch results
            self.logger.info(f'\nEpoch {epoch}/{self.config.NUM_EPOCHS}')
            self.logger.info(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
            self.logger.info(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
            self.logger.info(f'Learning Rate: {current_lr:.6f}')
            
            # Save checkpoint
            is_best = val_acc > self.best_acc
            if is_best:
                self.best_acc = val_acc
                self.logger.info(f'✓ New best accuracy: {self.best_acc:.2f}%')
            
            if is_best or (not self.config.SAVE_BEST_ONLY and epoch % self.config.SAVE_EVERY == 0):
                checkpoint_name = 'best_model.pth' if is_best else f'checkpoint_epoch_{epoch}.pth'
                checkpoint_path = self.config.CHECKPOINT_DIR / checkpoint_name
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    self.best_acc,
                    str(checkpoint_path)
                )
            
            # Early stopping
            if self.early_stopping is not None:
                self.early_stopping(val_loss)
                if self.early_stopping.early_stop:
                    self.logger.warning(f'\nEarly stopping triggered at epoch {epoch}')
                    break
        
        # Training complete
        training_time = time.time() - start_time
        self.logger.info("\n" + "="*60)
        self.logger.info("Training Complete!")
        self.logger.info("="*60)
        self.logger.info(f"Total training time: {training_time/60:.2f} minutes")
        self.logger.info(f"Best validation accuracy: {self.best_acc:.2f}%")
        
        # Save final model
        final_model_path = self.config.MODEL_DIR / f'{self.config.MODEL_NAME}_final.pth'
        torch.save(self.model.state_dict(), final_model_path)
        self.logger.info(f"Final model saved to {final_model_path}")
        
        # Save metrics
        metrics_path = self.config.LOG_DIR / 'training_metrics.csv'
        save_metrics(self.metrics, str(metrics_path))
        self.logger.info(f"Training metrics saved to {metrics_path}")
        
        return self.metrics


def main():
    """Main training function"""
    # Load configuration
    config = Config()
    
    # Setup logger for main
    logger = setup_logger(
        name="main",
        log_dir=config.LOG_DIR,
        log_level=config.LOG_LEVEL,
        console_output=config.CONSOLE_LOGGING,
        file_output=config.FILE_LOGGING
    )
    
    config.print_config(logger)
    
    # Create dataloaders
    logger.info("Loading dataset...")
    train_loader, val_loader, test_loader = create_dataloaders(
        str(config.DATA_DIR),
        config
    )
    
    # Create trainer
    trainer = Trainer(config)
    
    # Train model
    metrics = trainer.train(train_loader, val_loader)
    
    # Test final model
    logger.info("\nEvaluating on test set...")
    test_loss, test_acc = trainer.validate(test_loader)
    logger.info(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")


if __name__ == "__main__":
    main()

