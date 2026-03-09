# GPT Model Architecture

## What is it?

GPT (Generative Pre-trained Transformer) is a decoder-only transformer for autoregressive language modeling. This implementation is configurable from tiny (~1M params) to large (~1B+ params).

## Architecture

```
Input tokens
    ↓
Token Embedding (vocab_size → d_model) + Position Embedding (max_seq_len → d_model)
    ↓
Dropout
    ↓
┌─────────────── Repeat N times ──────────────┐
│ LayerNorm → Multi-Head Causal Self-Attention │
│     ↓ + Residual                             │
│ LayerNorm → MLP (expand → activate → project)│
│     ↓ + Residual                             │
└──────────────────────────────────────────────┘
    ↓
Final LayerNorm
    ↓
Linear projection → Vocabulary logits (d_model → vocab_size)
```

## Key Components

### Causal Self-Attention
Each token can only attend to itself and previous tokens (not future ones). This is enforced by a triangular mask that sets future positions to -infinity before softmax.

```
Q, K, V = Linear(x)
attention = softmax(Q @ K^T / sqrt(d_k) + causal_mask) @ V
```

### MLP (Feed-Forward Network)
Two variants:
- **GELU** (GPT-2 style): `output = Linear_down(GELU(Linear_up(x)))`
- **SwiGLU** (LLaMA style): `output = Linear_down(SiLU(Linear_gate(x)) * Linear_up(x))`

SwiGLU has ~50% more parameters per MLP but is more expressive.

### Weight Tying
The token embedding matrix and the output projection matrix share weights. This reduces parameters and provides a useful inductive bias.

## Size Presets

| Name | Layers | d_model | Heads | d_ff | Params | Training Time (A100) |
|------|--------|---------|-------|------|--------|---------------------|
| Tiny | 2 | 64 | 2 | 256 | ~100K | seconds |
| Small | 4 | 256 | 4 | 1024 | ~5M | minutes |
| Medium | 6 | 512 | 8 | 2048 | ~35M | ~1 hr |
| Base | 12 | 768 | 12 | 3072 | ~100M | ~3-4 hrs |
| Large | 24 | 1024 | 16 | 4096 | ~350M | ~12 hrs |
| XL | 24 | 2048 | 16 | 8192 | ~1B | ~2 days |

## Usage

```python
from llm_toolkit.models import GPT, GPTConfig

# Create a ~50M param model
config = GPTConfig(
    n_layers=6,
    n_heads=8,
    d_model=512,
    d_ff=2048,
    vocab_size=32000,
    max_seq_len=512,
    activation="gelu",      # or "swiglu" for LLaMA-style
    tie_weights=True,        # share embedding/output weights
)
model = GPT(config)

# Forward pass
logits, loss = model(input_ids, targets=labels)

# Generate text
tokens = model.generate(prompt_ids, max_new_tokens=100, temperature=0.8, top_k=40)

# Save/load
model.save_pretrained("my_model/")
model = GPT.from_pretrained("my_model/")
```

## Training with the Toolkit

```python
from llm_toolkit.models import GPT, GPTConfig
from llm_toolkit.training import CausalLMTrainer
from llm_toolkit.data import TextLoader

# Build model
config = GPTConfig(n_layers=6, d_model=512, n_heads=8, d_ff=2048, vocab_size=32000)
model = GPT(config)

# Load data
texts = TextLoader.from_huggingface("wikitext", "wikitext-2-raw-v1", split="train")

# Train
trainer = CausalLMTrainer(
    learning_rate=3e-4,
    num_epochs=5,
    batch_size=32,
    block_size=512,
    warmup_ratio=0.03,
)
result = trainer.train(model, tokenizer, texts)
```

## Implementation Details

- **Pre-norm**: LayerNorm is applied *before* attention/MLP (not after). Standard since GPT-2.
- **Initialization**: Weights are scaled by `0.02 / sqrt(2 * n_layers)` for residual projections. This prevents gradient explosion in deep networks.
- **No bias**: Following modern practice (LLaMA, Mistral), linear layers have no bias by default.
- **Causal mask**: Registered as a buffer (not a parameter), moves to GPU automatically.
