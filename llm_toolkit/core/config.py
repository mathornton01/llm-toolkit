"""
Configuration system for pipelines.

Configs can be loaded from YAML/JSON or constructed programmatically.
"""
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List
from pathlib import Path
import json


@dataclass
class ModuleConfig:
    """Configuration for a single module."""
    name: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    """Configuration for a complete pipeline."""
    model_name: str
    model_cache: str = "models"
    device: str = "auto"  # auto, cpu, cuda, cuda:0, directml
    dtype: str = "float32"  # float32, float16, bfloat16
    
    # Module selections
    collector: Optional[ModuleConfig] = None
    pruning: Optional[ModuleConfig] = None
    finetuning: Optional[ModuleConfig] = None
    evaluation: Optional[ModuleConfig] = None
    quantization: Optional[ModuleConfig] = None
    
    # Data
    calibration_prompts: List[str] = field(default_factory=list)
    eval_texts: List[str] = field(default_factory=list)
    
    # Output
    output_dir: str = "output"
    save_model: bool = True
    
    @classmethod
    def from_dict(cls, d: Dict) -> "PipelineConfig":
        """Create from dict (e.g., parsed YAML/JSON)."""
        modules = {}
        for key in ["collector", "pruning", "finetuning", "evaluation", "quantization"]:
            if key in d and d[key]:
                mod = d.pop(key)
                if isinstance(mod, str):
                    modules[key] = ModuleConfig(name=mod)
                elif isinstance(mod, dict):
                    modules[key] = ModuleConfig(
                        name=mod.pop("name"),
                        params=mod,
                    )
        return cls(**d, **modules)
    
    @classmethod
    def from_json(cls, path: str) -> "PipelineConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))
    
    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        try:
            import yaml
            with open(path) as f:
                return cls.from_dict(yaml.safe_load(f))
        except ImportError:
            raise ImportError("PyYAML required for YAML configs: pip install pyyaml")
    
    def resolve_device(self):
        """Resolve 'auto' device to best available."""
        if self.device != "auto":
            return self.device
        
        import torch
        if torch.cuda.is_available():
            return "cuda"
        
        try:
            import torch_directml
            return "directml"
        except ImportError:
            pass
        
        return "cpu"
