"""
Configuration Utilities - Milestone 3

Provides configuration management for fine-tuning modes and training parameters.
"""

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
import json


@dataclass
class FineTuningConfig:
    """Configuration for fine-tuning modes."""
    mode: str  # "head_only", "partial", "full"
    pretrained_path: Optional[str] = None
    freeze_backbone: bool = True  # For head_only mode
    freeze_layers: Optional[int] = None  # For partial mode: number of layers to freeze
    
    def __post_init__(self):
        """Validate configuration."""
        valid_modes = ["head_only", "partial", "full"]
        if self.mode not in valid_modes:
            raise ValueError(f"Invalid fine_tune_mode: {self.mode}. Must be one of {valid_modes}")


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    # Data
    data_dir: str = "data/processed"
    batch_size: int = 32
    num_workers: int = 0
    include_spatial: bool = True
    
    # Model
    num_genes: int = 248
    num_labels: int = 22
    fine_tune_mode: str = "head_only"
    pretrained_path: Optional[str] = None
    hidden_dim: int = 256
    num_layers: int = 4
    dropout: float = 0.1
    
    # Training
    num_epochs: int = 10
    learning_rate: float = 1e-4
    optimizer_type: str = "adam"
    weight_decay: float = 1e-5
    scheduler_type: Optional[str] = "cosine"
    
    # Device
    device: str = "cpu"  # "cpu" or "cuda"
    
    # Output
    output_dir: str = "outputs"
    experiment_name: str = "experiment"
    save_frequency: int = 1  # Save every N epochs
    
    # Validation
    validate_every: int = 1  # Validate every N epochs
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    def save(self, filepath: str):
        """Save configuration to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'TrainingConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**data)


def get_fine_tuning_config(mode: str, **kwargs) -> FineTuningConfig:
    """
    Get fine-tuning configuration for a specific mode.
    
    Args:
        mode: Fine-tuning mode ("head_only", "partial", "full")
        **kwargs: Additional configuration parameters
        
    Returns:
        FineTuningConfig instance
    """
    return FineTuningConfig(mode=mode, **kwargs)


def create_default_config(
    num_genes: int = 248,
    num_labels: int = 22,
    fine_tune_mode: str = "head_only",
    **overrides
) -> TrainingConfig:
    """
    Create default training configuration with optional overrides.
    
    Args:
        num_genes: Number of input genes
        num_labels: Number of output classes
        fine_tune_mode: Fine-tuning mode
        **overrides: Any configuration parameters to override
        
    Returns:
        TrainingConfig instance
    """
    config = TrainingConfig(
        num_genes=num_genes,
        num_labels=num_labels,
        fine_tune_mode=fine_tune_mode
    )
    
    # Apply overrides
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            print(f"Warning: Unknown config parameter: {key}")
    
    return config
