"""
Pipeline orchestrator — chains modules together.
"""
import time
import torch
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

from .registry import Registry
from .config import PipelineConfig
from .base import ModuleResult


class Pipeline:
    """Orchestrates a sequence of toolkit modules.
    
    Example:
        config = PipelineConfig(
            model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            pruning=ModuleConfig(name="wanda", params={"keep_ratio": 0.7}),
            evaluation=ModuleConfig(name="perplexity"),
        )
        pipeline = Pipeline(config)
        results = pipeline.run()
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.results = {}
    
    def _log(self, msg: str):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
    
    def _resolve_device(self):
        """Get the torch device."""
        device_str = self.config.resolve_device()
        if device_str == "directml":
            import torch_directml
            return torch_directml.device()
        return torch.device(device_str)
    
    def _get_dtype(self):
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        return dtype_map.get(self.config.dtype, torch.float32)
    
    def load_model(self):
        """Load model and tokenizer."""
        self._log(f"Loading {self.config.model_name}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name, cache_dir=self.config.model_cache
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        device = self._resolve_device()
        dtype = self._get_dtype()
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            cache_dir=self.config.model_cache,
            torch_dtype=dtype,
        ).to(device).eval()
        
        n_params = sum(p.numel() for p in self.model.parameters())
        self._log(f"  Loaded: {n_params:,} params on {device}")
        return self
    
    def run_collector(self) -> Optional[dict]:
        """Run the collection stage."""
        if not self.config.collector:
            return None
        
        cfg = self.config.collector
        self._log(f"Collecting with '{cfg.name}'...")
        collector = Registry.create("collection", cfg.name, **cfg.params)
        data = collector.collect(
            self.model, self.tokenizer,
            self.config.calibration_prompts,
        )
        self.results["collection"] = data
        self._log(f"  Collection done")
        return data
    
    def run_pruning(self, activations=None) -> Optional[ModuleResult]:
        """Run the pruning stage."""
        if not self.config.pruning:
            return None
        
        cfg = self.config.pruning
        self._log(f"Pruning with '{cfg.name}' (keep={cfg.params.get('keep_ratio', 0.7):.0%})...")
        
        pruner = Registry.create("pruning", cfg.name, **cfg.params)
        
        # Collect activations if needed and not provided
        if pruner.needs_activations and activations is None:
            self._log("  Collecting activations for pruner...")
            act_collector = Registry.create("collection", "activations")
            act_data = act_collector.collect(
                self.model, self.tokenizer,
                self.config.calibration_prompts,
            )
            activations = act_data.get("activations")
        
        importance = pruner.compute_importance(
            self.model, activations=activations
        )
        result = pruner.prune(self.model, importance)
        
        # Move model back to device after pruning (prune moves to CPU)
        device = self._resolve_device()
        self.model = self.model.to(device)
        
        self.results["pruning"] = result
        self._log(f"  {result.logs[0] if result.logs else 'Done'}")
        return result
    
    def run_finetuning(self) -> Optional[ModuleResult]:
        """Run the fine-tuning stage."""
        if not self.config.finetuning:
            return None
        
        cfg = self.config.finetuning
        self._log(f"Fine-tuning with '{cfg.name}'...")
        finetuner = Registry.create("finetuning", cfg.name, **cfg.params)
        result = finetuner.finetune(
            self.model, self.tokenizer,
            train_data=self.config.calibration_prompts,
        )
        self.results["finetuning"] = result
        self._log(f"  {result}")
        return result
    
    def run_evaluation(self) -> Optional[ModuleResult]:
        """Run the evaluation stage."""
        if not self.config.evaluation:
            return None
        
        cfg = self.config.evaluation
        self._log(f"Evaluating with '{cfg.name}'...")
        evaluator = Registry.create("evaluation", cfg.name, **cfg.params)
        result = evaluator.evaluate(
            self.model, self.tokenizer,
            eval_texts=self.config.eval_texts,
        )
        self.results["evaluation"] = result
        self._log(f"  {result}")
        return result
    
    def run(self) -> dict:
        """Run the full pipeline."""
        self._log("=" * 60)
        self._log("LLM Toolkit Pipeline")
        self._log("=" * 60)
        
        if self.model is None:
            self.load_model()
        
        # Evaluate baseline
        if self.config.evaluation:
            self._log("\n--- Baseline Evaluation ---")
            baseline_eval = self.config.evaluation
            evaluator = Registry.create("evaluation", baseline_eval.name, **baseline_eval.params)
            baseline = evaluator.evaluate(
                self.model, self.tokenizer,
                eval_texts=self.config.eval_texts,
            )
            self.results["baseline"] = baseline
            self._log(f"  Baseline: {baseline}")
        
        # Collection
        collected = self.run_collector()
        activations = collected.get("activations") if collected else None
        
        # Pruning
        self.run_pruning(activations=activations)
        
        # Fine-tuning
        self.run_finetuning()
        
        # Post-pruning evaluation
        if self.config.evaluation:
            self._log("\n--- Post-Processing Evaluation ---")
            result = self.run_evaluation()
        
        self._log("\n" + "=" * 60)
        self._log("Pipeline Complete")
        for stage, result in self.results.items():
            if isinstance(result, ModuleResult):
                self._log(f"  {stage}: {result}")
        self._log("=" * 60)
        
        return self.results
