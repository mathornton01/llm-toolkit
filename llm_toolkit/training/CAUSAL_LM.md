# Causal Language Model Pre-Training

## What is it?

Causal Language Modeling (CLM) is the foundational pre-training method used by every major autoregressive LLM: GPT-2, GPT-3, LLaMA, Mistral, Falcon, etc.

The model learns one simple task: **given all the previous tokens, predict the next one.**

```
Input:  [The] [quick] [brown] [fox]
Target: [quick] [brown] [fox] [jumps]
```

This is called *causal* because each token can only attend to tokens that came before it (causal masking). The model cannot "cheat" by looking ahead.

## Why does it work?

Predicting the next token requires understanding:
- Grammar and syntax
- Facts about the world
- Reasoning patterns
- Code structure
- And much more

By training on enough text, the model builds rich internal representations that can be applied to almost any downstream task — this is the "pre-training" that makes fine-tuning so effective.

## How the Training Works

### 1. Data Preparation

Raw text is tokenized and concatenated into one long sequence, then split into fixed-length chunks:

```
Text: "The quick brown fox. A hobbit lived in a hole."
      ↓ Tokenize
Tokens: [1, 450, 4996, 17354, 1701, 29916, 29889, ...]
      ↓ Split into blocks of size N
Block 0: [1, 450, 4996, 17354, ...]    → input_ids = block[:-1], labels = block[1:]
Block 1: [29916, 29889, 319, 298, ...] → input_ids = block[:-1], labels = block[1:]
```

Documents are separated by `<eos>` tokens so the model learns document boundaries.

### 2. Loss Function

Standard cross-entropy loss over the vocabulary at each position:

```
loss = -1/T * Σ log P(token_t | token_1, ..., token_{t-1})
```

Lower loss = the model assigns higher probability to the correct next token.

**Perplexity** = exp(loss) — more interpretable metric. A perplexity of 10 means the model is as uncertain as choosing uniformly among 10 options.

### 3. Optimizer: AdamW

```
θ_{t+1} = θ_t - lr * (m_t / (√v_t + ε) + λ * θ_t)
```

- Adaptive per-parameter learning rates
- Weight decay (`λ`) applied directly to weights (not to gradients) — this is the "W" in AdamW
- **Weight decay is NOT applied to biases or LayerNorm parameters** (standard practice)

### 4. Learning Rate Schedule: Cosine with Warmup

```
           ┌── Warmup (linear) ──┬────── Cosine Decay ──────┐
lr:   0 ─/─────────────────── peak ─────────────────────── ~0
step: 0        warmup_steps                         total_steps
```

- **Warmup:** prevents large gradient updates at the start when weights are random
- **Cosine decay:** smooth reduction avoids abrupt loss spikes

This schedule is used by GPT-3, LLaMA, and virtually all modern LLMs.

### 5. Gradient Accumulation

When GPU memory is limited, gradient accumulation simulates a larger batch:

```python
# Effective batch = batch_size × gradient_accumulation_steps
for i, batch in enumerate(dataloader):
    loss = model(batch) / grad_accum_steps
    loss.backward()
    if (i + 1) % grad_accum_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

## Usage

### Basic

```python
from llm_toolkit.training import CausalLMTrainer
from llm_toolkit.data import TextLoader, CausalLMDataset

# Load data
texts = TextLoader.from_file("my_corpus.txt")

# Create trainer
trainer = CausalLMTrainer(
    learning_rate=3e-4,
    num_epochs=3,
    batch_size=4,
    block_size=512,
)

# Train
result = trainer.train(model, tokenizer, texts)
print(result.metrics)
# {'final_loss': 2.14, 'final_perplexity': 8.5, 'total_steps': 1500, ...}
```

### Via Pipeline (Plug-and-Play)

```python
from llm_toolkit.core import Pipeline, PipelineConfig
from llm_toolkit.core.config import ModuleConfig

config = PipelineConfig(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    pruning=ModuleConfig(name="wanda", params={"keep_ratio": 0.8}),
    finetuning=ModuleConfig(name="causal_lm", params={
        "learning_rate": 1e-4,
        "num_epochs": 2,
        "block_size": 256,
    }),
    evaluation=ModuleConfig(name="perplexity"),
    calibration_prompts=["My training text..."],
)

results = Pipeline(config).run()
```

### From HuggingFace Dataset

```python
texts = TextLoader.from_huggingface(
    "wikitext", "wikitext-2-raw-v1",
    split="train", max_samples=10000,
)
```

## Key Hyperparameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `learning_rate` | 3e-4 | Higher for smaller models, lower for larger |
| `num_epochs` | 3 | Pre-training usually 1 epoch over massive data |
| `batch_size` | 4 | Larger is better; use `gradient_accumulation_steps` if GPU-limited |
| `block_size` | 128 | Context window size; larger = more context = more memory |
| `warmup_ratio` | 0.0 | Fraction of total steps for warmup (e.g., 0.03 = 3%) |
| `weight_decay` | 0.01 | L2 regularization; not applied to bias/norm params |
| `max_grad_norm` | 1.0 | Gradient clipping threshold |
| `lr_scheduler` | "cosine" | Options: "cosine", "linear", "constant" |
| `gradient_accumulation_steps` | 1 | Multiply to get effective batch size |

## Typical Values for Different Model Sizes

| Model Size | LR | Batch | Block Size | Warmup |
|------------|-----|-------|-----------|--------|
| ~100M params | 5e-4 | 64 | 512 | 2000 steps |
| ~1B params | 3e-4 | 256 | 1024 | 2000 steps |
| ~7B params | 3e-4 | 1024 | 2048 | 2000 steps |
| ~70B params | 1e-4 | 2048 | 4096 | 2000 steps |

*(All batch sizes are in tokens, not sequences)*

## Data Formats Supported

| Source | Method |
|--------|--------|
| Plain text file | `TextLoader.from_file("corpus.txt")` |
| One doc per line | `TextLoader.from_lines("docs.txt")` |
| JSONL with text field | `TextLoader.from_jsonl("data.jsonl", field="text")` |
| JSON array | `TextLoader.from_json("data.json")` |
| Directory of files | `TextLoader.from_directory("data/", pattern="*.txt")` |
| HuggingFace dataset | `TextLoader.from_huggingface("wikitext", ...)` |
| Python list | `TextLoader.from_strings(["text1", "text2"])` |

## References

- Radford et al. (2019) — [Language Models are Unsupervised Multitask Learners](https://openai.com/research/better-language-models) (GPT-2)
- Brown et al. (2020) — [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) (GPT-3)
- Touvron et al. (2023) — [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
- Loshchilov & Hutter (2019) — [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) (AdamW)
