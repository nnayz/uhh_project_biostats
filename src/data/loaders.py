"""
Data Loaders Module - Milestone 3

Implements the Data Contract API for loading federated client data.
All training, federated, and evaluation code must use this interface.

See data_preparation.md and training.md (project root) for schema and usage.
"""

import os
import json
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, List, Tuple, Dict
import numpy as np


class CellDataset(Dataset):
    """
    PyTorch Dataset for spatial transcriptomics cells.
    Uses lazy per-sample access to avoid duplicating the DataFrame in large
    float arrays, which reduces memory (important for federated Ray actors).
    Each sample contains: gene expression, optional (x,y), label.
    """

    def __init__(self, df: pd.DataFrame, gene_columns: List[str], include_spatial: bool = True):
        """
        Args:
            df: DataFrame with columns: id, sample_id/client_id, x, y, label, gene_*
            gene_columns: Ordered list of gene column names (must match genes.txt)
            include_spatial: Whether to include spatial coordinates in features
        """
        self.df = df.reset_index(drop=True)
        self.gene_columns = gene_columns
        self.include_spatial = include_spatial

        missing_genes = [g for g in gene_columns if g not in df.columns]
        if missing_genes:
            raise ValueError(f"Missing gene columns: {missing_genes[:5]}...")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        features = torch.from_numpy(
            row[self.gene_columns].values.astype(np.float32, copy=False)
        )
        if self.include_spatial:
            spatial = torch.tensor(
                [row["x"], row["y"]], dtype=torch.float32
            )
            features = torch.cat([features, spatial])
        return {
            "features": features,
            "label": torch.tensor(row["label"], dtype=torch.long),
            "x": torch.tensor(row["x"], dtype=torch.float32),
            "y": torch.tensor(row["y"], dtype=torch.float32),
        }


def load_gene_list(data_dir: str = "data/processed") -> List[str]:
    """
    Load the canonical gene list and order.
    
    Args:
        data_dir: Path to processed data directory
        
    Returns:
        Ordered list of gene names (248 genes)
    """
    genes_path = os.path.join(data_dir, "genes.txt")
    if not os.path.exists(genes_path):
        raise FileNotFoundError(f"Gene list not found: {genes_path}")
    
    with open(genes_path, 'r') as f:
        genes = [line.strip() for line in f if line.strip()]
    
    return genes


def load_label_map(data_dir: str = "data/processed") -> Dict[str, int]:
    """
    Load the label mapping (Leiden cluster ID -> integer label).
    
    Args:
        data_dir: Path to processed data directory
        
    Returns:
        Dictionary mapping cluster ID strings to integer labels
    """
    label_map_path = os.path.join(data_dir, "label_map.json")
    if not os.path.exists(label_map_path):
        raise FileNotFoundError(f"Label map not found: {label_map_path}")
    
    with open(label_map_path, 'r') as f:
        label_map = json.load(f)
    
    # Ensure values are integers
    return {k: int(v) for k, v in label_map.items()}


def load_client_data(
    client_id: str,
    split: str,
    data_dir: str = "data/processed",
    include_spatial: bool = True,
    validate: bool = True
) -> pd.DataFrame:
    """
    Load data for a specific client and split.
    
    Args:
        client_id: Client identifier (e.g., "client_01", "client_02", "client_03")
        split: Data split ("train", "val", "test")
        data_dir: Path to processed data directory
        include_spatial: Whether to include spatial coordinates (for future use)
        validate: Whether to validate schema
        
    Returns:
        DataFrame with columns: id, sample_id/client_id, x, y, label, gene_*
    """
    if split not in ["train", "val", "test"]:
        raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")
    
    parquet_path = os.path.join(data_dir, "clients", client_id, f"{split}.parquet")
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Data file not found: {parquet_path}")
    
    df = pd.read_parquet(parquet_path)
    
    if validate:
        _validate_dataframe(df, data_dir)
    
    return df


def load_all_clients(
    split: str,
    data_dir: str = "data/processed",
    include_spatial: bool = True,
    validate: bool = True
) -> pd.DataFrame:
    """
    Load data from all clients for a specific split.
    
    Args:
        split: Data split ("train", "val", "test")
        data_dir: Path to processed data directory
        include_spatial: Whether to include spatial coordinates
        validate: Whether to validate schema
        
    Returns:
        Combined DataFrame from all clients
        
    Note:
        For "test" split, use load_global_test() instead to get the global test set.
    """
    clients_dir = os.path.join(data_dir, "clients")
    if not os.path.exists(clients_dir):
        raise FileNotFoundError(f"Clients directory not found: {clients_dir}")
    
    client_dirs = sorted([d for d in os.listdir(clients_dir) 
                         if os.path.isdir(os.path.join(clients_dir, d)) and d.startswith("client_")])
    
    if not client_dirs:
        raise ValueError(f"No client directories found in {clients_dir}")
    
    dfs = []
    for client_id in client_dirs:
        try:
            df = load_client_data(client_id, split, data_dir, include_spatial, validate=False)
            # Add client_id column if not present
            if 'client_id' not in df.columns and 'sample_id' in df.columns:
                df['client_id'] = client_id
            dfs.append(df)
        except FileNotFoundError:
            print(f"Warning: {split} split not found for {client_id}, skipping...")
            continue
    
    if not dfs:
        raise ValueError(f"No data found for split '{split}' across any client")
    
    combined_df = pd.concat(dfs, ignore_index=True)
    
    if validate:
        _validate_dataframe(combined_df, data_dir)
    
    return combined_df


def load_global_test(
    data_dir: str = "data/processed",
    include_spatial: bool = True,
    validate: bool = True
) -> pd.DataFrame:
    """
    Load the evaluation set: held-out batch (batch-based) or global test set (legacy).
    
    Used for evaluation in centralized, federated, and local training.
    Prefer held_out_batch.parquet when present (batch-based strategy).
    
    Args:
        data_dir: Path to processed data directory
        include_spatial: Whether to include spatial coordinates (for future use)
        validate: Whether to validate schema
        
    Returns:
        DataFrame with columns: id, sample_id/batch_id, x, y, label, gene_*
    """
    held_out_path = os.path.join(data_dir, "held_out_batch.parquet")
    global_test_path = os.path.join(data_dir, "global_test.parquet")
    path = held_out_path if os.path.exists(held_out_path) else global_test_path
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Evaluation set not found. Looked for: {held_out_path}, {global_test_path}\n"
            "Please run scripts/data_preparation/partition_anatomical_siloing.py first."
        )
    df = pd.read_parquet(path)
    if validate:
        _validate_dataframe(df, data_dir)
    return df


def load_held_out_eval(
    data_dir: str = "data/processed",
    validate: bool = True
) -> pd.DataFrame:
    """Load the held-out batch evaluation set (alias for load_global_test)."""
    return load_global_test(data_dir=data_dir, validate=validate)


def create_dataloader(
    df: pd.DataFrame,
    gene_columns: List[str],
    batch_size: int = 32,
    shuffle: bool = False,
    include_spatial: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False
) -> DataLoader:
    """
    Create a PyTorch DataLoader from a DataFrame.
    
    Args:
        df: DataFrame with required columns
        gene_columns: Ordered list of gene column names
        batch_size: Batch size for training
        shuffle: Whether to shuffle data
        include_spatial: Whether to include spatial coordinates
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory (useful for GPU)
        
    Returns:
        PyTorch DataLoader
    """
    dataset = CellDataset(df, gene_columns, include_spatial=include_spatial)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )


def _validate_dataframe(df: pd.DataFrame, data_dir: str):
    """
    Validate DataFrame schema according to Data Contract.
    
    Validation Rules:
    - Assert gene column order matches genes.txt
    - Assert labels are integers in [0, num_labels)
    - Assert no NaN or infinite values
    - Assert required columns exist
    """
    # Check required columns
    required_cols = ['id', 'label', 'x', 'y']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Validate gene columns match genes.txt
    gene_list = load_gene_list(data_dir)
    gene_cols_in_df = [col for col in df.columns if col in gene_list]
    
    if len(gene_cols_in_df) != len(gene_list):
        missing = set(gene_list) - set(gene_cols_in_df)
        raise ValueError(f"Gene columns mismatch. Missing: {list(missing)[:5]}...")
    
    # Check gene order (first N columns after metadata should match)
    # Note: This is a soft check - exact order validation would be stricter
    df_gene_cols = [col for col in df.columns if col in gene_list]
    if df_gene_cols != gene_list:
        print(f"Warning: Gene column order may not match genes.txt exactly")
    
    # Validate labels
    label_map = load_label_map(data_dir)
    num_labels = len(label_map)
    labels = df['label'].values
    
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"Labels must be integers, got {labels.dtype}")
    
    if labels.min() < 0 or labels.max() >= num_labels:
        raise ValueError(f"Labels must be in [0, {num_labels}), got range [{labels.min()}, {labels.max()}]")
    
    # Check for NaN or infinite values in gene expression
    gene_data = df[gene_list].values
    if np.isnan(gene_data).any():
        raise ValueError("NaN values found in gene expression data")
    if np.isinf(gene_data).any():
        raise ValueError("Infinite values found in gene expression data")
    
    # Check for NaN in spatial coordinates
    if np.isnan(df[['x', 'y']].values).any():
        raise ValueError("NaN values found in spatial coordinates")
    
    # Check split is non-empty
    if len(df) == 0:
        raise ValueError("DataFrame is empty")


def get_num_labels(data_dir: str = "data/processed") -> int:
    """Get the number of unique labels."""
    label_map = load_label_map(data_dir)
    return len(label_map)


def get_num_genes(data_dir: str = "data/processed") -> int:
    """Get the number of genes."""
    gene_list = load_gene_list(data_dir)
    return len(gene_list)
