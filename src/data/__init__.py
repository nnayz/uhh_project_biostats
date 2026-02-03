"""
Data Loading Module

Implements the Data Contract API for loading federated client data.
"""

from .loaders import (
    load_client_data,
    load_all_clients,
    load_gene_list,
    load_label_map,
    load_global_test,
    create_dataloader,
    get_num_labels,
    get_num_genes,
    CellDataset
)

__all__ = [
    'load_client_data',
    'load_all_clients',
    'load_gene_list',
    'load_label_map',
    'load_global_test',
    'create_dataloader',
    'get_num_labels',
    'get_num_genes',
    'CellDataset'
]
