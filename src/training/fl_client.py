"""
Federated Learning Client - Flower NumPyClient Implementation

Implements the Federation Contract for client-side federated learning.
Each client:
  - loads its local data
  - calls the shared training engine
  - returns updated weights

See training.md (project root) for federated training summary.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

import flwr as fl
from flwr.common import NDArrays, Scalar

from src.data import (
    create_dataloader,
    load_client_data,
    load_gene_list,
    load_label_map,
)
from src.model import create_model
from src.training import create_optimizer, evaluate, train_one_epoch


# ---------------------------------------------------------------------------
# State dict <-> NumPy conversion helpers
# ---------------------------------------------------------------------------

def state_dict_to_ndarrays(state_dict: Dict[str, torch.Tensor]) -> List[np.ndarray]:
    """
    Convert a PyTorch state_dict to a list of NumPy arrays.
    
    The order is determined by state_dict.keys() which is stable (OrderedDict).
    Each tensor is moved to CPU and converted to numpy.
    
    Args:
        state_dict: PyTorch model state dictionary
        
    Returns:
        List of NumPy arrays in deterministic order
    """
    return [val.cpu().numpy() for val in state_dict.values()]


def ndarrays_to_state_dict(
    model: torch.nn.Module,
    ndarrays: List[np.ndarray]
) -> Dict[str, torch.Tensor]:
    """
    Convert a list of NumPy arrays back to a PyTorch state_dict.
    
    Uses the model's current state_dict keys to define the stable order.
    
    Args:
        model: PyTorch model (to get key names and dtypes)
        ndarrays: List of NumPy arrays in the same order as state_dict_to_ndarrays
        
    Returns:
        PyTorch state dictionary
    """
    keys = list(model.state_dict().keys())
    if len(keys) != len(ndarrays):
        raise ValueError(
            f"Mismatch: model has {len(keys)} parameters, "
            f"but received {len(ndarrays)} arrays"
        )
    
    state_dict = OrderedDict()
    for key, arr in zip(keys, ndarrays):
        state_dict[key] = torch.from_numpy(arr.copy())
    
    return state_dict


# ---------------------------------------------------------------------------
# Flower Client Implementation
# ---------------------------------------------------------------------------

class FlowerClient(fl.client.NumPyClient):
    """
    Flower NumPyClient for federated learning.
    
    This client:
    1. Loads local train/val data for a specific client
    2. Performs local training using the shared train_engine
    3. Returns updated weights and metrics to the server
    """
    
    def __init__(
        self,
        client_id: str,
        data_dir: str = "data/processed",
        local_epochs: int = 1,
        batch_size: int = 256,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        device: str = "cpu",
        fine_tune_mode: str = "head_only",
        include_spatial: bool = True,
        pretrained_path: Optional[str] = None,
        num_workers: int = 0,
        verbose: bool = True,
        use_amp: bool = False,
    ):
        """
        Initialize the Flower client.
        
        Args:
            client_id: Client identifier (e.g., "client_01")
            data_dir: Path to processed data directory
            local_epochs: Number of local training epochs per round
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
            device: Device to run on ("cpu" or "cuda")
            fine_tune_mode: Fine-tuning mode ("head_only", "partial", "full")
            include_spatial: Whether to include spatial coordinates
            pretrained_path: Optional path to pretrained weights
            num_workers: Number of data loading workers
            verbose: Whether to print progress
        """
        self.client_id = client_id
        self.data_dir = data_dir
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.device = torch.device(device)
        self.fine_tune_mode = fine_tune_mode
        self.include_spatial = include_spatial
        self.pretrained_path = pretrained_path
        self.num_workers = num_workers
        self.verbose = verbose
        self.use_amp = use_amp and self.device.type == 'cuda'
        
        # Setup AMP scaler if needed
        if self.use_amp:
            # Use new torch.amp API if available, else fallback to old API
            if hasattr(torch.amp, 'GradScaler'):
                self.scaler = torch.amp.GradScaler('cuda')
            else:
                from torch.cuda.amp import GradScaler
                self.scaler = GradScaler()
        else:
            self.scaler = None
        
        # Load gene list and label map
        self.genes = load_gene_list(data_dir)
        self.label_map = load_label_map(data_dir)
        self.num_genes = len(self.genes)
        self.num_labels = len(self.label_map)
        
        # Load local data
        self._load_data()
        
        # Create model
        self.model = create_model(
            num_genes=self.num_genes,
            num_labels=self.num_labels,
            pretrained_path=pretrained_path,
            fine_tune_mode=fine_tune_mode,
            include_spatial=include_spatial,
        )
        self.model.to(self.device)
    
    def _load_data(self) -> None:
        """Load local train and validation data for this client."""
        # Load train data
        train_df = load_client_data(
            self.client_id, "train",
            data_dir=self.data_dir,
            validate=True
        )
        self.train_loader = create_dataloader(
            train_df, self.genes,
            batch_size=self.batch_size,
            shuffle=True,
            include_spatial=self.include_spatial,
            num_workers=self.num_workers,
        )
        self.num_train_samples = len(train_df)
        
        # Load validation data
        val_df = load_client_data(
            self.client_id, "val",
            data_dir=self.data_dir,
            validate=True
        )
        self.val_loader = create_dataloader(
            val_df, self.genes,
            batch_size=self.batch_size,
            shuffle=False,
            include_spatial=self.include_spatial,
            num_workers=self.num_workers,
        )
        self.num_val_samples = len(val_df)
        
        if self.verbose:
            print(f"[{self.client_id}] Loaded {self.num_train_samples} train, "
                  f"{self.num_val_samples} val samples")
    
    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """
        Return current model parameters as a list of NumPy arrays.
        
        Args:
            config: Configuration dictionary (unused)
            
        Returns:
            List of NumPy arrays representing model weights
        """
        return state_dict_to_ndarrays(self.model.get_weights())
    
    def set_parameters(self, parameters: NDArrays) -> None:
        """
        Set model parameters from a list of NumPy arrays.
        
        Args:
            parameters: List of NumPy arrays representing model weights
        """
        state_dict = ndarrays_to_state_dict(self.model, parameters)
        self.model.set_weights(state_dict)
    
    def fit(
        self,
        parameters: NDArrays,
        config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """
        Train the model on local data and return updated parameters.
        
        Args:
            parameters: Global model parameters from server
            config: Training configuration from server
            
        Returns:
            Tuple of:
            - Updated model parameters
            - Number of training samples
            - Dictionary of training metrics
        """
        # Set global parameters
        self.set_parameters(parameters)
        
        # Get local epochs from config if provided
        local_epochs = int(config.get("local_epochs", self.local_epochs))
        
        # Create optimizer (fresh each round for clean state)
        optimizer = create_optimizer(
            self.model,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        
        # Train for local epochs
        total_loss = 0.0
        total_acc = 0.0
        total_f1 = 0.0
        
        for epoch in range(local_epochs):
            metrics = train_one_epoch(
                self.model,
                self.train_loader,
                optimizer,
                self.device,
                verbose=self.verbose,
                use_amp=self.use_amp,
                scaler=self.scaler,
            )
            total_loss += metrics["loss"]
            total_acc += metrics["accuracy"]
            total_f1 += metrics["f1_macro"]
            
            if self.verbose:
                print(f"[{self.client_id}] Local epoch {epoch+1}/{local_epochs} - "
                      f"Loss: {metrics['loss']:.4f}, Acc: {metrics['accuracy']:.4f}")
        
        # Average metrics over local epochs
        avg_metrics = {
            "loss": float(total_loss / local_epochs),
            "accuracy": float(total_acc / local_epochs),
            "f1_macro": float(total_f1 / local_epochs),
        }
        
        # Return updated parameters, number of samples, and metrics
        return (
            state_dict_to_ndarrays(self.model.get_weights()),
            self.num_train_samples,
            avg_metrics,
        )
    
    def evaluate(
        self,
        parameters: NDArrays,
        config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """
        Evaluate the model on local validation data.
        
        Args:
            parameters: Model parameters from server
            config: Evaluation configuration from server
            
        Returns:
            Tuple of:
            - Evaluation loss
            - Number of validation samples
            - Dictionary of evaluation metrics
        """
        # Set parameters
        self.set_parameters(parameters)
        
        # Evaluate on validation set
        metrics = evaluate(
            self.model,
            self.val_loader,
            self.device,
            verbose=self.verbose,
            use_amp=self.use_amp,
        )
        
        # Return loss, num_examples, and metrics
        return (
            float(metrics["loss"]),
            self.num_val_samples,
            {
                "accuracy": float(metrics["accuracy"]),
                "f1_macro": float(metrics["f1_macro"]),
            },
        )


def create_client_fn(
    client_ids: List[str],
    data_dir: str = "data/processed",
    local_epochs: int = 1,
    batch_size: int = 256,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    device: str = "cpu",
    fine_tune_mode: str = "head_only",
    include_spatial: bool = True,
    pretrained_path: Optional[str] = None,
    num_workers: int = 0,
    verbose: bool = True,
    use_amp: bool = False,
):
    """
    Create a client factory function for Flower simulation.
    
    This factory maps Flower's numeric client IDs ("0", "1", ...) to
    actual client folder names (client_01, client_02, ...).
    
    Args:
        client_ids: List of actual client folder names
        data_dir: Path to processed data directory
        local_epochs: Number of local training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for regularization
        device: Device to run on
        fine_tune_mode: Fine-tuning mode
        include_spatial: Whether to include spatial coordinates
        pretrained_path: Optional path to pretrained weights
        num_workers: Number of data loading workers
        verbose: Whether to print progress
        
    Returns:
        Client factory function for Flower simulation
    """
    def client_fn(cid: str) -> FlowerClient:
        """Create a Flower client for the given client ID."""
        # Map Flower's numeric cid to actual client folder name
        idx = int(cid)
        if idx >= len(client_ids):
            raise ValueError(f"Client ID {cid} out of range. Have {len(client_ids)} clients.")
        
        actual_client_id = client_ids[idx]
        
        return FlowerClient(
            client_id=actual_client_id,
            data_dir=data_dir,
            local_epochs=local_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            device=device,
            fine_tune_mode=fine_tune_mode,
            include_spatial=include_spatial,
            pretrained_path=pretrained_path,
            num_workers=num_workers,
            verbose=verbose,
            use_amp=use_amp,
        )
    
    return client_fn
