#!/usr/bin/env python3
"""
Train a GPT model from scratch.

Usage (A100):
    python scripts/train_gpt.py \
        --size medium \
        --dataset wikitext \
        --epochs 5 \
        --batch-size 32 \
        --compile

Usage (CPU, tiny test):
    python scripts/train_gpt.py --size tiny --epochs 1 --batch-size 2 --no-amp

Sizes:
    tiny:   2 layers, 64 hidden,  ~100K params  (seconds)
    small:  4 layers, 256 hidden, ~5M params    (minutes)
    medium: 6 layers, 512 hidden, ~35M params   (~1 hr on A100)
    base:   12 layers, 768 hidden, ~100M params (~3-4 hrs on A100)
    large:  24 layers, 1024 hidden, ~350M params (~12 hrs on A100)
"""
import argparse
import json
import os
import sys
import time
import math

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_toolkit.models import GPT, GPTConfig
from llm_toolkit.training.causal_lm import CausalLMTrainer
from llm_toolkit.data.text_loader import TextLoader


# ── Model size presets ────────────────────────────────────────────────
SIZE_PRESETS = {
    "tiny": GPTConfig(
        n_layers=2, n_heads=2, d_model=64, d_ff=256,
        vocab_size=32000, max_seq_len=128, dropout=0.1,
    ),
    "small": GPTConfig(
        n_layers=4, n_heads=4, d_model=256, d_ff=1024,
        vocab_size=32000, max_seq_len=256, dropout=0.1,
    ),
    "medium": GPTConfig(
        n_layers=6, n_heads=8, d_model=512, d_ff=2048,
        vocab_size=32000, max_seq_len=512, dropout=0.1,
    ),
    "base": GPTConfig(
        n_layers=12, n_heads=12, d_model=768, d_ff=3072,
        vocab_size=32000, max_seq_len=1024, dropout=0.1,
    ),
    "large": GPTConfig(
        n_layers=24, n_heads=16, d_model=1024, d_ff=4096,
        vocab_size=32000, max_seq_len=1024, dropout=0.1,
    ),
}


def get_device():
    """Detect best available device."""
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"GPU: {name} ({mem:.1f} GB)")
        return dev
    else:
        print("No GPU detected, using CPU")
        return torch.device("cpu")


def load_dataset(name: str, max_texts: int = 0):
    """Load a training dataset."""
    if name == "wikitext":
        print("Loading WikiText-103...")
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-103-raw-v1")
        train_texts = [t for t in ds["train"]["text"] if len(t.strip()) > 50]
        val_texts = [t for t in ds["validation"]["text"] if len(t.strip()) > 50]
        if max_texts > 0:
            train_texts = train_texts[:max_texts]
            val_texts = val_texts[:max_texts // 10]
        print(f"  Train: {len(train_texts):,} texts")
        print(f"  Val: {len(val_texts):,} texts")
        return train_texts, val_texts
    
    elif name == "fineweb":
        print("Loading FineWeb-Edu (sample)...")
        from datasets import load_dataset
        ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", 
                         split="train", streaming=True)
        train_texts = []
        for i, example in enumerate(ds):
            if max_texts > 0 and i >= max_texts:
                break
            if i >= 100000 and max_texts == 0:  # Default: 100K texts
                break
            train_texts.append(example["text"])
        # Use last 10% as validation
        split = int(len(train_texts) * 0.9)
        val_texts = train_texts[split:]
        train_texts = train_texts[:split]
        print(f"  Train: {len(train_texts):,} texts")
        print(f"  Val: {len(val_texts):,} texts")
        return train_texts, val_texts
    
    elif name == "tiny_shakespeare":
        print("Loading Tiny Shakespeare...")
        from datasets import load_dataset
        ds = load_dataset("karpathy/tiny_shakespeare")
        train_text = ds["train"]["text"][0]
        val_text = ds["validation"]["text"][0]
        # Split into paragraphs
        train_texts = [p for p in train_text.split("\n\n") if len(p.strip()) > 20]
        val_texts = [p for p in val_text.split("\n\n") if len(p.strip()) > 20]
        print(f"  Train: {len(train_texts):,} paragraphs")
        print(f"  Val: {len(val_texts):,} paragraphs")
        return train_texts, val_texts
    
    elif os.path.exists(name):
        print(f"Loading from file/directory: {name}")
        texts = TextLoader.from_file(name) if os.path.isfile(name) else TextLoader.from_directory(name)
        split = int(len(texts) * 0.9)
        return texts[:split], texts[split:]
    
    else:
        raise ValueError(f"Unknown dataset: {name}. Use: wikitext, fineweb, tiny_shakespeare, or a file path")


def main():
    parser = argparse.ArgumentParser(description="Train a GPT model from scratch")
    parser.add_argument("--size", default="medium", choices=SIZE_PRESETS.keys(),
                       help="Model size preset")
    parser.add_argument("--dataset", default="tiny_shakespeare",
                       help="Dataset: wikitext, fineweb, tiny_shakespeare, or file path")
    parser.add_argument("--max-texts", type=int, default=0,
                       help="Max training texts (0=all)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--grad-accum", type=int, default=1,
                       help="Gradient accumulation steps")
    parser.add_argument("--activation", default="gelu", choices=["gelu", "swiglu"])
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    parser.add_argument("--save-dir", default="checkpoints",
                       help="Directory to save checkpoints")
    parser.add_argument("--save-every", type=int, default=500,
                       help="Save checkpoint every N steps")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=0,
                       help="DataLoader workers")
    parser.add_argument("--tokenizer", default="gpt2",
                       help="HuggingFace tokenizer name")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"Training GPT ({args.size}) on {args.dataset}")
    print("=" * 60)
    
    # ── Device ──
    device = get_device()
    
    # ── Tokenizer ──
    print(f"\nLoading tokenizer: {args.tokenizer}")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # ── Model ──
    config = SIZE_PRESETS[args.size]
    config.activation = args.activation
    config.vocab_size = tokenizer.vocab_size
    # Ensure max_seq_len matches block_size we'll use
    block_size = min(config.max_seq_len, 512)  # Cap at 512 for memory
    config.max_seq_len = block_size
    
    print(f"\nModel config:")
    print(f"  Layers: {config.n_layers}, Heads: {config.n_heads}")
    print(f"  d_model: {config.d_model}, d_ff: {config.d_ff}")
    print(f"  Vocab: {config.vocab_size}, Max seq: {config.max_seq_len}")
    print(f"  Activation: {config.activation}")
    print(f"  Estimated params: {config.estimate_params():,}")
    
    model = GPT(config).to(device)
    
    # ── Dataset ──
    print()
    train_texts, val_texts = load_dataset(args.dataset, args.max_texts)
    
    # ── Train ──
    trainer = CausalLMTrainer(
        learning_rate=args.lr,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        block_size=block_size,
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=args.grad_accum,
        use_amp=not args.no_amp,
        amp_dtype="bfloat16",
        compile_model=args.compile,
        save_every_steps=args.save_every,
        save_dir=args.save_dir,
        log_every_steps=args.log_every,
        num_workers=args.num_workers,
    )
    
    print(f"\nTraining config:")
    print(f"  Epochs: {args.epochs}, Batch: {args.batch_size}")
    print(f"  LR: {args.lr}, Warmup: {args.warmup_ratio}")
    print(f"  Grad accum: {args.grad_accum}")
    print(f"  AMP: {not args.no_amp}, Compile: {args.compile}")
    print()
    
    start_time = time.time()
    result = trainer.train(model, tokenizer, train_texts, val_texts=val_texts)
    elapsed = time.time() - start_time
    
    # ── Results ──
    print(f"\n{'=' * 60}")
    print(f"Training complete in {elapsed/60:.1f} minutes")
    print(f"  Final loss: {result.metrics['final_loss']:.4f}")
    print(f"  Final perplexity: {result.metrics['final_perplexity']:.2f}")
    if result.metrics.get("best_val_loss", -1) > 0:
        print(f"  Best val loss: {result.metrics['best_val_loss']:.4f}")
    print(f"  Total steps: {result.metrics['total_steps']}")
    print(f"{'=' * 60}")
    
    # ── Test generation ──
    print("\n--- Sample Generation ---")
    model.eval()
    prompts = ["The ", "In the beginning ", "Once upon a time "]
    for prompt in prompts:
        tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            generated = model.generate(tokens, max_new_tokens=50, temperature=0.8, top_k=40)
        text = tokenizer.decode(generated[0], skip_special_tokens=True)
        print(f"  \"{text[:200]}\"")
    
    # Save training results
    results_path = os.path.join(args.save_dir, "training_results.json")
    os.makedirs(args.save_dir, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump({
            "model_size": args.size,
            "dataset": args.dataset,
            "elapsed_seconds": elapsed,
            "metrics": result.metrics,
            "config": result.artifacts.get("config", {}),
            "model_config": {
                "n_layers": config.n_layers,
                "n_heads": config.n_heads,
                "d_model": config.d_model,
                "d_ff": config.d_ff,
                "vocab_size": config.vocab_size,
                "max_seq_len": config.max_seq_len,
                "activation": config.activation,
            },
        }, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
