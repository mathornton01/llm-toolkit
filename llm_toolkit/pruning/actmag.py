"""
Activation magnitude pruning.

importance(neuron_j) = mean(|activation_j|) across calibration prompts

Pure activation-based importance — no weight information.
Acts as an ablation of Wanda (removing the weight term).
Surprisingly competitive, especially at aggressive pruning ratios.
"""
import torch
from typing import Dict, Optional

from ..core.registry import Registry
from ..core.base import PruningMethod


@Registry.register("pruning", "actmag")
class ActMagPruning(PruningMethod):
    """Activation magnitude pruning."""
    
    name = "actmag"
    needs_activations = True
    needs_gradients = False
    
    def compute_importance(
        self,
        model,
        activations=None,
        **kwargs,
    ) -> Dict[int, torch.Tensor]:
        if activations is None:
            raise ValueError("ActMag requires activations.")
        
        n_layers = model.config.num_hidden_layers
        importance = {}
        
        for li in range(n_layers):
            if li in activations:
                importance[li] = activations[li].mean(dim=0)
            else:
                # Fallback: uniform importance
                importance[li] = torch.ones(model.config.intermediate_size)
        
        return importance
