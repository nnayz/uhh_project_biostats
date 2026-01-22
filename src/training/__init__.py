"""
Training Engine Module

Shared training and evaluation functions for centralized and federated training.
"""

from .train_engine import (
    train_one_epoch,
    evaluate,
    TrainingHistory,
    save_model,
    load_model,
    save_training_artifacts,
    create_optimizer,
    create_scheduler
)

# Federated learning components (optional import - requires flwr)
try:
    from .fl_client import (
        FlowerClient,
        create_client_fn,
        state_dict_to_ndarrays,
        ndarrays_to_state_dict,
    )
    from .fl_server import (
        create_fedavg_strategy,
        get_on_fit_config_fn,
        get_on_evaluate_config_fn,
        RoundLogger,
        weighted_average,
        aggregate_fit_metrics,
        aggregate_evaluate_metrics,
    )
    _FL_AVAILABLE = True
except ImportError:
    _FL_AVAILABLE = False

__all__ = [
    # Core training
    'train_one_epoch',
    'evaluate',
    'TrainingHistory',
    'save_model',
    'load_model',
    'save_training_artifacts',
    'create_optimizer',
    'create_scheduler',
    # Federated learning (if available)
    'FlowerClient',
    'create_client_fn',
    'state_dict_to_ndarrays',
    'ndarrays_to_state_dict',
    'create_fedavg_strategy',
    'get_on_fit_config_fn',
    'get_on_evaluate_config_fn',
    'RoundLogger',
    'weighted_average',
    'aggregate_fit_metrics',
    'aggregate_evaluate_metrics',
]
