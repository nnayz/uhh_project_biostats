"""
Shared Training Engine - Milestone 3

Provides reusable training and evaluation functions used by both
centralized and federated training pipelines.

Contract Reference: docs/milestone3/contracts/training_contract.md
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import json
import os
from pathlib import Path

import sys

# Use new torch.amp API (PyTorch 2.0+) or fallback to old API
try:
    if hasattr(torch.amp, 'GradScaler'):
        # New API (PyTorch 2.0+)
        GradScaler = lambda: torch.amp.GradScaler('cuda')
    else:
        # Old API (PyTorch < 2.0)
        from torch.cuda.amp import GradScaler
except (ImportError, AttributeError):
    # Fallback for older PyTorch versions
    GradScaler = None


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    verbose: bool = True,
    use_amp: bool = False,
    scaler: Optional[GradScaler] = None
) -> Dict[str, float]:
    """
    Train the model for one epoch.
    
    This function is reused by:
    - Centralized training
    - Federated client training
    
    Args:
        model: Model implementing the Model Contract interface
        dataloader: Training data loader
        optimizer: Optimizer for parameter updates
        device: Device to run training on (cpu/cuda)
        verbose: Whether to print progress
        use_amp: Whether to use Automatic Mixed Precision (AMP)
        scaler: GradScaler for AMP (required if use_amp=True)
        
    Returns:
        Dictionary with metrics:
        - 'loss': Average training loss
        - 'accuracy': Training accuracy
        - 'f1_macro': Macro-averaged F1 score
    """
    model.train()
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    num_batches = 0
    
    is_cuda = device.type == 'cuda'
    non_blocking = is_cuda  # Use non-blocking transfers for CUDA
    
    for batch_idx, batch in enumerate(dataloader):
        # Move batch to device (non-blocking for CUDA)
        features = batch['features'].to(device, non_blocking=non_blocking)
        labels = batch['label'].to(device, non_blocking=non_blocking)
        
        batch_dict = {'features': features, 'label': labels}
        
        # Forward pass with AMP if enabled
        optimizer.zero_grad()
        
        if use_amp and scaler is not None:
            # Use new torch.amp API if available, else fallback to old API
            if hasattr(torch.amp, 'autocast'):
                with torch.amp.autocast('cuda'):
                    logits = model(batch_dict)
                    loss = model.compute_loss(logits, labels)
            else:
                with torch.cuda.amp.autocast():
                    logits = model(batch_dict)
                    loss = model.compute_loss(logits, labels)
            
            # Backward pass with AMP
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(batch_dict)
            loss = model.compute_loss(logits, labels)
            loss.backward()
            optimizer.step()
        
        # Accumulate metrics
        total_loss += loss.item()
        predictions = torch.argmax(logits, dim=1).cpu().numpy()
        all_predictions.extend(predictions)
        all_labels.extend(labels.cpu().numpy())
        num_batches += 1
        
        if verbose and (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    # Compute epoch metrics
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    accuracy = accuracy_score(all_labels, all_predictions)
    f1_macro = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'f1_macro': f1_macro
    }


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    verbose: bool = True,
    use_amp: bool = False
) -> Dict[str, float]:
    """
    Evaluate the model on a dataset.
    
    This function is reused by:
    - Centralized validation/test
    - Federated client validation
    - Final evaluation
    
    Args:
        model: Model implementing the Model Contract interface
        dataloader: Evaluation data loader
        device: Device to run evaluation on (cpu/cuda)
        verbose: Whether to print results
        use_amp: Whether to use Automatic Mixed Precision (AMP)
        
    Returns:
        Dictionary with metrics:
        - 'loss': Average loss
        - 'accuracy': Accuracy
        - 'f1_macro': Macro-averaged F1 score
    """
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    num_batches = 0
    
    is_cuda = device.type == 'cuda'
    non_blocking = is_cuda  # Use non-blocking transfers for CUDA
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device (non-blocking for CUDA)
            features = batch['features'].to(device, non_blocking=non_blocking)
            labels = batch['label'].to(device, non_blocking=non_blocking)
            
            batch_dict = {'features': features, 'label': labels}
            
            # Forward pass with AMP if enabled
            if use_amp and is_cuda:
                # Use new torch.amp API if available, else fallback to old API
                if hasattr(torch.amp, 'autocast'):
                    with torch.amp.autocast('cuda'):
                        logits = model(batch_dict)
                        loss = model.compute_loss(logits, labels)
                else:
                    with torch.cuda.amp.autocast():
                        logits = model(batch_dict)
                        loss = model.compute_loss(logits, labels)
            else:
                logits = model(batch_dict)
                loss = model.compute_loss(logits, labels)
            
            # Accumulate metrics
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1).cpu().numpy()
            all_predictions.extend(predictions)
            all_labels.extend(labels.cpu().numpy())
            num_batches += 1
    
    # Compute metrics
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    accuracy = accuracy_score(all_labels, all_predictions)
    f1_macro = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
    
    if verbose:
        print(f"  Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, F1-Macro: {f1_macro:.4f}")
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'predictions': all_predictions,
        'labels': all_labels
    }


class TrainingHistory:
    """
    Tracks training history across epochs/rounds.
    """
    
    def __init__(self):
        self.train_loss = []
        self.train_accuracy = []
        self.train_f1_macro = []
        self.val_loss = []
        self.val_accuracy = []
        self.val_f1_macro = []
    
    def add_train_metrics(self, metrics: Dict[str, float]):
        """Add training metrics for an epoch."""
        self.train_loss.append(metrics['loss'])
        self.train_accuracy.append(metrics['accuracy'])
        self.train_f1_macro.append(metrics['f1_macro'])
    
    def add_val_metrics(self, metrics: Dict[str, float]):
        """Add validation metrics for an epoch."""
        self.val_loss.append(metrics['loss'])
        self.val_accuracy.append(metrics['accuracy'])
        self.val_f1_macro.append(metrics['f1_macro'])
    
    def to_dict(self) -> Dict:
        """Convert history to dictionary for JSON serialization."""
        return {
            'train': {
                'loss': self.train_loss,
                'accuracy': self.train_accuracy,
                'f1_macro': self.train_f1_macro
            },
            'val': {
                'loss': self.val_loss,
                'accuracy': self.val_accuracy,
                'f1_macro': self.val_f1_macro
            }
        }
    
    def save(self, filepath: str):
        """Save history to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


def save_model(model: nn.Module, filepath: str):
    """
    Save model state dictionary.
    
    Args:
        model: Model to save
        filepath: Path to save model
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(model.state_dict(), filepath)


def load_model(model: nn.Module, filepath: str):
    """
    Load model state dictionary.
    
    Args:
        model: Model to load weights into
        filepath: Path to saved model
    """
    model.load_state_dict(torch.load(filepath, map_location='cpu'))


def save_training_artifacts(
    output_dir: str,
    model: nn.Module,
    history: TrainingHistory,
    config: Dict,
    metrics: Optional[Dict] = None
):
    """
    Save all training artifacts as required by Training Contract.
    
    Outputs:
    - history.json: Training history
    - metrics.csv: Final metrics
    - model_final.pt: Final model weights
    - config.yaml: Training configuration
    
    Args:
        output_dir: Directory to save artifacts
        model: Trained model
        history: Training history
        config: Training configuration
        metrics: Additional metrics to save
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save history
    history_path = os.path.join(output_dir, "history.json")
    history.save(history_path)
    
    # Save model
    model_path = os.path.join(output_dir, "model_final.pt")
    save_model(model, model_path)
    
    # Save config as JSON (simpler than YAML for now)
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Save metrics
    if metrics:
        metrics_path = os.path.join(output_dir, "metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Also save as CSV for easy viewing
        import pandas as pd
        metrics_df = pd.DataFrame([metrics])
        metrics_csv_path = os.path.join(output_dir, "metrics.csv")
        metrics_df.to_csv(metrics_csv_path, index=False)


def create_optimizer(
    model: nn.Module,
    learning_rate: float = 1e-4,
    optimizer_type: str = "adam",
    weight_decay: float = 1e-5
) -> torch.optim.Optimizer:
    """
    Create optimizer for model training.
    
    Args:
        model: Model to optimize
        learning_rate: Learning rate
        optimizer_type: Type of optimizer ("adam", "sgd", "adamw")
        weight_decay: Weight decay (L2 regularization)
        
    Returns:
        Optimizer instance
    """
    # Get trainable parameters
    if hasattr(model, 'get_trainable_parameters'):
        params = model.get_trainable_parameters()
    else:
        params = [p for p in model.parameters() if p.requires_grad]
    
    if optimizer_type.lower() == "adam":
        return torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type.lower() == "adamw":
        return torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type.lower() == "sgd":
        return torch.optim.SGD(params, lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = "cosine",
    num_epochs: int = 10,
    **kwargs
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Create learning rate scheduler.
    
    Args:
        optimizer: Optimizer to schedule
        scheduler_type: Type of scheduler ("cosine", "step", "plateau", None)
        num_epochs: Number of training epochs
        **kwargs: Additional scheduler arguments
        
    Returns:
        Scheduler instance or None
    """
    if scheduler_type is None or scheduler_type.lower() == "none":
        return None
    
    if scheduler_type.lower() == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, **kwargs
        )
    elif scheduler_type.lower() == "step":
        step_size = kwargs.get('step_size', num_epochs // 3)
        gamma = kwargs.get('gamma', 0.1)
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )
    elif scheduler_type.lower() == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, **kwargs
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
