"""
Base classes for all toolkit modules.

Each base class defines the interface that plug-and-play modules must implement.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizer


@dataclass
class ModuleResult:
    """Standard result container for any module."""
    success: bool
    metrics: Dict[str, float] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)
    
    def log(self, msg: str):
        self.logs.append(msg)
    
    def __repr__(self):
        status = "OK" if self.success else "FAIL"
        metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in self.metrics.items())
        return f"ModuleResult({status}, {metrics_str})"


class PruningMethod(ABC):
    """Base class for all pruning methods.
    
    Pruning methods compute importance scores for neurons/heads/layers,
    then remove the least important ones.
    
    Subclasses must implement:
        - compute_importance(): Returns per-layer importance scores
        - prune(): Applies pruning to the model
    
    Optionally override:
        - needs_activations: Whether this method needs activation data
        - needs_gradients: Whether this method needs gradient data
        - name: Human-readable name
    """
    
    needs_activations: bool = False
    needs_gradients: bool = False
    name: str = "base"
    
    def __init__(self, keep_ratio: float = 0.7, **kwargs):
        self.keep_ratio = keep_ratio
        self.config = kwargs
    
    @abstractmethod
    def compute_importance(
        self,
        model: PreTrainedModel,
        activations: Optional[Dict[int, torch.Tensor]] = None,
        **kwargs,
    ) -> Dict[int, torch.Tensor]:
        """Compute per-neuron importance scores for each layer.
        
        Args:
            model: The model to analyze
            activations: Optional dict of {layer_idx: tensor(n_samples, intermediate_size)}
        
        Returns:
            Dict mapping layer_idx -> importance tensor of shape (intermediate_size,)
        """
        pass
    
    def prune(
        self,
        model: PreTrainedModel,
        importance: Dict[int, torch.Tensor],
    ) -> ModuleResult:
        """Apply structural pruning based on importance scores.
        
        Default implementation does MLP neuron pruning (gate/up/down projections).
        Override for different pruning granularities (heads, layers, etc.).
        """
        n_layers = model.config.num_hidden_layers
        inter_size = model.config.intermediate_size
        n_keep = int(inter_size * self.keep_ratio)
        
        before = sum(p.numel() for p in model.parameters())
        device = next(model.parameters()).device
        
        # Move to CPU for surgery
        was_on_device = str(device)
        model = model.cpu()
        
        for li in range(n_layers):
            if li not in importance:
                continue
            layer = model.model.layers[li]
            scores = importance[li]
            idx = torch.topk(scores, n_keep).indices.sort().values
            
            wg = layer.mlp.gate_proj.weight.data[idx, :].clone()
            wu = layer.mlp.up_proj.weight.data[idx, :].clone()
            wd = layer.mlp.down_proj.weight.data[:, idx].clone()
            
            in_features = wg.shape[1]
            out_features = wg.shape[0]
            
            layer.mlp.gate_proj = nn.Linear(in_features, out_features, bias=False)
            layer.mlp.gate_proj.weight.data = wg
            layer.mlp.up_proj = nn.Linear(in_features, out_features, bias=False)
            layer.mlp.up_proj.weight.data = wu
            layer.mlp.down_proj = nn.Linear(out_features, wd.shape[0], bias=False)
            layer.mlp.down_proj.weight.data = wd
        
        model.config.intermediate_size = n_keep
        after = sum(p.numel() for p in model.parameters())
        
        result = ModuleResult(
            success=True,
            metrics={
                "params_before": before,
                "params_after": after,
                "reduction_pct": (1 - after / before) * 100,
                "keep_ratio": self.keep_ratio,
                "n_keep": n_keep,
            },
        )
        result.log(f"Pruned {before:,} -> {after:,} params ({result.metrics['reduction_pct']:.1f}% reduction)")
        return result
    
    def __repr__(self):
        return f"{self.__class__.__name__}(keep_ratio={self.keep_ratio})"


class Collector(ABC):
    """Base class for data collection modules (activations, gradients, etc.)."""
    
    name: str = "base"
    
    @abstractmethod
    def collect(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        prompts: List[str],
        **kwargs,
    ) -> Dict[str, Any]:
        """Collect data from model.
        
        Returns:
            Dict with collected data (activations, gradients, etc.)
        """
        pass


class FineTuner(ABC):
    """Base class for fine-tuning methods."""
    
    name: str = "base"
    
    @abstractmethod
    def finetune(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        train_data: Any,
        **kwargs,
    ) -> ModuleResult:
        """Fine-tune the model.
        
        Returns:
            ModuleResult with training metrics
        """
        pass


class Evaluator(ABC):
    """Base class for evaluation methods."""
    
    name: str = "base"
    
    @abstractmethod
    def evaluate(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        **kwargs,
    ) -> ModuleResult:
        """Evaluate the model.
        
        Returns:
            ModuleResult with evaluation metrics
        """
        pass


class Quantizer(ABC):
    """Base class for quantization methods."""
    
    name: str = "base"
    
    @abstractmethod
    def quantize(
        self,
        model_path: str,
        output_path: str,
        **kwargs,
    ) -> ModuleResult:
        """Quantize a model.
        
        Returns:
            ModuleResult with quantization metrics
        """
        pass
