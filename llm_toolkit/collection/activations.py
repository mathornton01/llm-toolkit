"""
Activation collection module.

Collects per-prompt MLP activation magnitudes for importance scoring.
"""
import torch
import torch.nn.functional as F
from typing import Dict, List, Any

from ..core.registry import Registry
from ..core.base import Collector


@Registry.register("collection", "activations")
class ActivationCollector(Collector):
    """Collect mean absolute MLP activations per prompt per layer."""
    
    name = "activations"
    
    def __init__(self, max_seq_len: int = 128, layers: str = "all", **kwargs):
        """
        Args:
            max_seq_len: Maximum sequence length for tokenization
            layers: "all" or comma-separated layer indices (e.g., "0,4,8,12")
        """
        self.max_seq_len = max_seq_len
        self.layers = layers
    
    def collect(
        self,
        model,
        tokenizer,
        prompts: List[str],
        **kwargs,
    ) -> Dict[str, Any]:
        """Collect activations from model.
        
        Returns:
            Dict with:
                "activations": {layer_idx: tensor(n_prompts, intermediate_size)}
                "n_prompts": int
                "n_layers": int
        """
        n_layers = model.config.num_hidden_layers
        device = next(model.parameters()).device
        
        # Determine which layers to collect
        if self.layers == "all":
            layer_indices = list(range(n_layers))
        else:
            layer_indices = [int(x) for x in self.layers.split(",")]
        
        per_prompt_acts = {li: [] for li in layer_indices}
        
        for prompt in prompts:
            enc = tokenizer(
                prompt, return_tensors="pt",
                truncation=True, max_length=self.max_seq_len
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            
            with torch.no_grad():
                out = model(**enc, output_hidden_states=True)
                
                for li in layer_indices:
                    layer = model.model.layers[li]
                    h = out.hidden_states[li]
                    h_norm = layer.post_attention_layernorm(h)
                    gate = F.silu(layer.mlp.gate_proj(h_norm))
                    up = layer.mlp.up_proj(h_norm)
                    act = gate * up
                    # Mean absolute activation over sequence
                    per_prompt_acts[li].append(act.abs().mean(dim=(0, 1)).cpu())
        
        activations = {}
        for li in layer_indices:
            activations[li] = torch.stack(per_prompt_acts[li])
        
        return {
            "activations": activations,
            "n_prompts": len(prompts),
            "n_layers": len(layer_indices),
            "layer_indices": layer_indices,
        }
