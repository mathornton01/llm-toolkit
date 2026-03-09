"""
Magnitude-based pruning.

Simplest baseline: importance = L2 norm of weight vectors.
Removes neurons whose combined gate+up+down projection weights are smallest.
"""
import torch
from typing import Dict, Optional

from ..core.registry import Registry
from ..core.base import PruningMethod


@Registry.register("pruning", "magnitude")
class MagnitudePruning(PruningMethod):
    """Weight magnitude pruning (L2 norm of gate+up+down projections)."""
    
    name = "magnitude"
    needs_activations = False
    needs_gradients = False
    
    def compute_importance(
        self,
        model,
        activations=None,
        **kwargs,
    ) -> Dict[int, torch.Tensor]:
        n_layers = model.config.num_hidden_layers
        importance = {}
        
        for li in range(n_layers):
            layer = model.model.layers[li]
            w_gate = layer.mlp.gate_proj.weight.data.cpu()
            w_up = layer.mlp.up_proj.weight.data.cpu()
            w_down = layer.mlp.down_proj.weight.data.cpu()
            
            # L2 norm per neuron across all three projections
            importance[li] = (
                w_gate.norm(dim=1) +
                w_up.norm(dim=1) +
                w_down.norm(dim=0)
            )
        
        return importance
