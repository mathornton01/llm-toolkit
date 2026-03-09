# LLM Toolkit

A modular, plug-and-play framework for building LLM training pipelines. Mix and match components at every stage of the LLM lifecycle.

## The Idea

Every stage of training an LLM — data loading, pre-training, fine-tuning, alignment, pruning, quantization, evaluation — is implemented as a **swappable module** registered in a central registry. You build a pipeline by selecting one module per stage.

```python
from llm_toolkit.core import Pipeline, PipelineConfig
from llm_toolkit.core.config import ModuleConfig

config = PipelineConfig(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    pruning=ModuleConfig(name="wanda", params={"keep_ratio": 0.8}),
    evaluation=ModuleConfig(name="perplexity"),
    calibration_prompts=["My training text here..."],
)

results = Pipeline(config).run()
```

Swap `wanda` for `magnitude`, `actmag`, or `random`. Add a `finetuning` stage. Done.

## Architecture

```
llm_toolkit/
├── core/
│   ├── registry.py         Plugin registry (@Registry.register decorator)
│   ├── base.py             Abstract base classes for all module types
│   ├── pipeline.py         Pipeline orchestrator
│   └── config.py           Configuration system (dict/YAML/JSON)
│
├── data/
│   ├── causal_lm_dataset.py   Next-token prediction dataset
│   └── text_loader.py         Load text from files, JSONL, HuggingFace, etc.
│
├── training/               Pre-training methods
│   └── causal_lm.py        ✅ Causal LM (GPT-style next-token prediction)
│
├── finetuning/             Fine-tuning methods (coming soon)
│   ├── full.py             Full parameter fine-tuning
│   └── lora.py             LoRA (Low-Rank Adaptation)
│
├── alignment/              Alignment methods (coming soon)
│   ├── sft.py              Supervised Fine-Tuning
│   └── dpo.py              Direct Preference Optimization
│
├── pruning/                Pruning methods
│   ├── magnitude.py        ✅ Weight magnitude (L2 norm)
│   ├── wanda.py            ✅ Weight × Activation (Wanda)
│   ├── actmag.py           ✅ Activation magnitude
│   └── random_pruning.py   ✅ Random baseline
│
├── collection/             Data collection modules
│   └── activations.py      ✅ MLP activation collector
│
└── evaluation/             Evaluation methods
    └── perplexity.py       ✅ Perplexity
```

## Implemented Modules

| Stage | Module | Registry Key | Description |
|-------|--------|--------------|-------------|
| Training | CausalLMTrainer | `training/causal_lm` | GPT-style next-token prediction |
| Pruning | MagnitudePruning | `pruning/magnitude` | Weight L2 norm |
| Pruning | WandaPruning | `pruning/wanda` | Weight × Activation |
| Pruning | ActMagPruning | `pruning/actmag` | Activation magnitude |
| Pruning | RandomPruning | `pruning/random` | Random baseline |
| Collection | ActivationCollector | `collection/activations` | MLP activations |
| Evaluation | PerplexityEvaluator | `evaluation/perplexity` | Perplexity |

## Quick Start

```python
# Prune a model with Wanda, evaluate perplexity
from llm_toolkit.core import Pipeline, PipelineConfig
from llm_toolkit.core.config import ModuleConfig

import llm_toolkit.pruning      # register pruning modules
import llm_toolkit.evaluation   # register evaluation modules

config = PipelineConfig(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    pruning=ModuleConfig(name="wanda", params={"keep_ratio": 0.7}),
    evaluation=ModuleConfig(name="perplexity"),
    calibration_prompts=["The quick brown fox...", "In a hole in the ground..."],
)

results = Pipeline(config).run()
```

```python
# Train from scratch with causal LM
from llm_toolkit.training import CausalLMTrainer
from llm_toolkit.data import TextLoader

texts = TextLoader.from_huggingface("wikitext", "wikitext-2-raw-v1", split="train")

trainer = CausalLMTrainer(
    learning_rate=3e-4,
    num_epochs=1,
    batch_size=8,
    block_size=512,
    warmup_ratio=0.03,
    gradient_accumulation_steps=4,
)
result = trainer.train(model, tokenizer, texts)
```

## Writing Your Own Module

Any module is a class decorated with `@Registry.register`:

```python
from llm_toolkit.core.registry import Registry
from llm_toolkit.core.base import PruningMethod
import torch

@Registry.register("pruning", "my_method")
class MyPruningMethod(PruningMethod):
    needs_activations = False
    
    def compute_importance(self, model, activations=None, **kwargs):
        # Return {layer_idx: importance_tensor} for each layer
        importance = {}
        for li in range(model.config.num_hidden_layers):
            layer = model.model.layers[li]
            # Your importance scoring logic here
            importance[li] = layer.mlp.gate_proj.weight.data.norm(dim=1)
        return importance

# Now use it:
# pruner = Registry.create("pruning", "my_method", keep_ratio=0.7)
```

That's it. Drop your file in the `pruning/` directory, import it, and it works.

## Module Documentation

- [Causal LM Training](training/CAUSAL_LM.md)

## Running Tests

```bash
python tests/test_toolkit.py            # Core + pruning tests
python tests/test_causal_lm_training.py # Causal LM training tests (12 tests)
```
