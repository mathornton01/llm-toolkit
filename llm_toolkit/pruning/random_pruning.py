"""
Random pruning baseline.

Randomly selects neurons to keep. Lower bound for any meaningful method.
"""
import torch
from typing import Dict, Optional

from ..core.registry import Registry
from ..core.base import PruningMethod


@Registry.register("pruning", "random")
class RandomPruning(PruningMethod):
    """Random neuron selection (baseline)."""
    
    name = "random"
    needs_activations = False
    needs_gradients = False
    
    def __init__(self, keep_ratio: float = 0.7, seed: int = 42, **kwargs):
        super().__init__(keep_ratio=keep_ratio, **kwargs)
        self.seed = seed
    
    def compute_importance(
        self,
        model,
        activations=None,
        **kwargs,
    ) -> Dict[int, torch.Tensor]:
        n_layers = model.config.num_hidden_layers
        inter_size = model.config.intermediate_size
        importance = {}
        
        gen = torch.Generator().manual_seed(self.seed)
        for li in range(n_layers):
            importance[li] = torch.rand(inter_size, generator=gen)
        
        return importance
