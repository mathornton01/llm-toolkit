"""
Wanda pruning (Pruning by Weights AND Activations).

Reference: Sun et al., "A Simple and Effective Pruning Approach for Large Language Models" (2023)

importance(neuron_j) = ||W_j||_2 * ||X_j||_1

Combines weight magnitude with activation magnitude — captures both structural
and functional importance. No training required, just one forward pass for activations.
"""
import torch
from typing import Dict, Optional

from ..core.registry import Registry
from ..core.base import PruningMethod


@Registry.register("pruning", "wanda")
class WandaPruning(PruningMethod):
    """Wanda: Weight AND Activation pruning."""
    
    name = "wanda"
    needs_activations = True
    needs_gradients = False
    
    def __init__(self, keep_ratio: float = 0.7, weight_combine: str = "all", **kwargs):
        """
        Args:
            keep_ratio: Fraction of neurons to keep
            weight_combine: How to combine weight norms:
                "all" - gate + up + down (default)
                "gate" - gate projection only (original Wanda)
                "gate_up" - gate + up projections
        """
        super().__init__(keep_ratio=keep_ratio, **kwargs)
        self.weight_combine = weight_combine
    
    def compute_importance(
        self,
        model,
        activations=None,
        **kwargs,
    ) -> Dict[int, torch.Tensor]:
        if activations is None:
            raise ValueError(
                "Wanda requires activations. Set needs_activations=True "
                "or pass activations dict."
            )
        
        n_layers = model.config.num_hidden_layers
        importance = {}
        
        for li in range(n_layers):
            layer = model.model.layers[li]
            
            # Weight magnitude
            w_gate = layer.mlp.gate_proj.weight.data.cpu()
            w_up = layer.mlp.up_proj.weight.data.cpu()
            w_down = layer.mlp.down_proj.weight.data.cpu()
            
            if self.weight_combine == "gate":
                w_norm = w_gate.norm(dim=1)
            elif self.weight_combine == "gate_up":
                w_norm = w_gate.norm(dim=1) + w_up.norm(dim=1)
            else:  # "all"
                w_norm = w_gate.norm(dim=1) + w_up.norm(dim=1) + w_down.norm(dim=0)
            
            # Activation magnitude (mean across prompts)
            act_mag = activations[li].mean(dim=0) if li in activations else torch.ones_like(w_norm)
            
            # Wanda score = weight * activation
            importance[li] = w_norm * act_mag
        
        return importance
