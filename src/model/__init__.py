"""
Model Module

Nicheformer model wrapper implementing the Model Contract interface.
"""

from .nicheformer_wrapper import (
    NicheformerWrapper,
    create_model
)

__all__ = [
    'NicheformerWrapper',
    'create_model'
]
