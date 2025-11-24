"""
Training utilities for urban growth prediction model.

Includes:
- Training loop with validation
- Learning rate scheduling
- Checkpoint management
- Metrics tracking
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
import json


class Trainer:
    """
    Training manager for ConvLSTM urban growth model.
    
    Args:
        model (nn.Module): ConvLSTM model instance
        device (str): Device for training ('cuda' or 'cpu')
        learning_rate (float): Initial learning rate
        checkpoint_dir (Path): Directory for saving checkpoints
    """
    def __init__(self, model, device='cpu', learning_rate=1e-3, checkpoint_dir='checkpoints'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = Adam(model.parameters(), lr=learning_rate)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=5, factor=0.5)
        self.criterion = nn.MSELoss()
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
    
    def train_epoch(self, train_loader):
        """Run one training epoch."""
        self.model.train()
        epoch_loss = 0.0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            predictions, _ = self.model(inputs)
            
            loss = self.criterion(predictions, targets)
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
        
        return epoch_loss / len(train_loader)
    
    def validate(self, val_loader):
        """Run validation."""
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                predictions, _ = self.model(inputs)
                loss = self.criterion(predictions, targets)
                val_loss += loss.item()
        
        return val_loss / len(val_loader)
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        
        # Save latest checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"✓ Saved best model (val_loss: {val_loss:.6f})")
    
    def train(self, train_loader, val_loader, num_epochs):
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs (int): Number of epochs to train
        """
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            self.scheduler.step(val_loss)
            
            # Save checkpoint if best validation loss
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            self.save_checkpoint(epoch, val_loss, is_best)
            
            print(f"Epoch {epoch}/{num_epochs} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | "
                  f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
        
        print(f"\nTraining complete! Best val loss: {self.best_val_loss:.6f}")
        
        # Save training history
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }
        with open(self.checkpoint_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)


def calculate_metrics(predictions, targets):
    """
    Calculate evaluation metrics.
    
    Args:
        predictions (torch.Tensor): Model predictions
        targets (torch.Tensor): Ground truth values
        
    Returns:
        dict: Dictionary of metrics (MSE, MAE, RMSE)
    """
    mse = torch.mean((predictions - targets) ** 2).item()
    mae = torch.mean(torch.abs(predictions - targets)).item()
    rmse = torch.sqrt(torch.tensor(mse)).item()
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse
    }


if __name__ == "__main__":
    print("Training utilities loaded.")
