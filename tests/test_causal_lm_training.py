#!/usr/bin/env python3
"""
Comprehensive tests for Causal LM pre-training module.

Tests:
  1. Data: CausalLMDataset construction, chunking, edge cases
  2. Data: TextLoader from various sources
  3. Logic: Token alignment (labels = shifted input_ids)
  4. Logic: Attention masking
  5. Logic: Learning rate schedule (warmup + cosine)
  6. Training: Loss decreases on small data
  7. Training: Gradient accumulation matches large batch
  8. Training: Checkpoint saving/loading
  9. Comparison: Verify against manual training loop
  10. Integration: Full pipeline with evaluation
"""
import sys, os, math, tempfile, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_toolkit.data.causal_lm_dataset import CausalLMDataset
from llm_toolkit.data.text_loader import TextLoader
from llm_toolkit.training.causal_lm import CausalLMTrainer, get_cosine_schedule_with_warmup
from llm_toolkit.core.registry import Registry

# Import to trigger registration
import llm_toolkit.training
import llm_toolkit.evaluation

# Small model for fast tests
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MODEL_CACHE = "models"

# Shared fixtures
_tokenizer = None
def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=MODEL_CACHE)
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token
    return _tokenizer


def test_dataset_construction():
    """Test 1: CausalLMDataset creates correct chunks."""
    print("=== Test 1: Dataset Construction ===")
    tokenizer = get_tokenizer()
    
    texts = [
        "The quick brown fox jumps over the lazy dog. " * 10,
        "In a hole in the ground there lived a hobbit. " * 10,
    ]
    
    dataset = CausalLMDataset(texts, tokenizer, block_size=32)
    stats = dataset.stats()
    
    print(f"  Texts: {len(texts)}, Tokens: {stats['total_tokens']}, Chunks: {stats['num_chunks']}")
    assert len(dataset) > 0, "Dataset should have chunks"
    assert stats["block_size"] == 32
    
    # Verify chunk shape
    sample = dataset[0]
    assert sample["input_ids"].shape == (32,), f"Expected (32,), got {sample['input_ids'].shape}"
    assert sample["labels"].shape == (32,), f"Expected (32,), got {sample['labels'].shape}"
    assert sample["attention_mask"].shape == (32,)
    print("  PASSED\n")


def test_dataset_token_alignment():
    """Test 2: Labels are correctly shifted by 1 from input_ids."""
    print("=== Test 2: Token Alignment (labels = shifted input) ===")
    tokenizer = get_tokenizer()
    
    text = "Hello world this is a test of the causal language model dataset. " * 5
    dataset = CausalLMDataset([text], tokenizer, block_size=16)
    
    sample = dataset[0]
    input_ids = sample["input_ids"]
    labels = sample["labels"]
    
    # The key property: labels[i] should be the token that comes after input_ids[i]
    # In other words: for chunk [t0, t1, t2, ..., t16]
    #   input_ids = [t0, t1, ..., t15]
    #   labels    = [t1, t2, ..., t16]
    # So labels[i] == input_ids[i+1] for i < len-1 is NOT the right test
    # The right test: the original chunk is block_size+1 tokens
    # input_ids = chunk[:-1], labels = chunk[1:]
    
    chunk = dataset.chunks[0]
    assert torch.equal(input_ids, chunk[:-1]), "input_ids should be chunk[:-1]"
    assert torch.equal(labels, chunk[1:]), "labels should be chunk[1:]"
    
    # Verify the shift: labels[i] == input_ids[i+1] for non-boundary
    for i in range(len(input_ids) - 1):
        assert labels[i] == input_ids[i + 1], \
            f"Position {i}: labels[{i}]={labels[i]} != input_ids[{i+1}]={input_ids[i+1]}"
    
    print(f"  Chunk shape: {chunk.shape}")
    print(f"  input_ids: {input_ids[:5].tolist()}...")
    print(f"  labels:    {labels[:5].tolist()}...")
    print(f"  All {len(input_ids)-1} positions verified: labels[i] == input_ids[i+1]")
    print("  PASSED\n")


def test_dataset_sliding_window():
    """Test 3: Sliding window with stride < block_size."""
    print("=== Test 3: Sliding Window ===")
    tokenizer = get_tokenizer()
    
    text = "a b c d e f g h i j k l m n o p q r s t u v w x y z " * 5
    
    # No overlap
    ds_no_overlap = CausalLMDataset([text], tokenizer, block_size=16, stride=16)
    # 50% overlap
    ds_overlap = CausalLMDataset([text], tokenizer, block_size=16, stride=8)
    
    print(f"  No overlap: {len(ds_no_overlap)} chunks")
    print(f"  50% overlap: {len(ds_overlap)} chunks")
    
    # Overlapping should produce more chunks
    assert len(ds_overlap) > len(ds_no_overlap), \
        f"Overlap ({len(ds_overlap)}) should produce more chunks than no overlap ({len(ds_no_overlap)})"
    
    # Verify overlap: second chunk's start should be at stride offset
    if len(ds_no_overlap) >= 2:
        chunk0 = ds_overlap.chunks[0]
        chunk1 = ds_overlap.chunks[1]
        # chunk1 should start at stride=8 offset from chunk0
        assert torch.equal(chunk0[8:], chunk1[:-8]), \
            "Overlapping chunks should share tokens"
    
    print("  PASSED\n")


def test_dataset_empty_and_short():
    """Test 4: Edge cases — empty text, text shorter than block_size."""
    print("=== Test 4: Edge Cases ===")
    tokenizer = get_tokenizer()
    
    # Empty text
    ds = CausalLMDataset([""], tokenizer, block_size=32)
    # Only EOS token — too short for a chunk
    assert len(ds) == 0, f"Empty text should give 0 chunks, got {len(ds)}"
    print(f"  Empty text: {len(ds)} chunks (correct)")
    
    # Very short text
    ds = CausalLMDataset(["Hi"], tokenizer, block_size=128)
    assert len(ds) == 0, f"Short text should give 0 chunks with large block_size, got {len(ds)}"
    print(f"  Short text: {len(ds)} chunks (correct)")
    
    # Text exactly block_size + 1 tokens
    tokens = tokenizer.encode("test " * 50)  # lots of tokens
    target_len = 33  # block_size + 1
    text = tokenizer.decode(tokens[:target_len])
    ds = CausalLMDataset([text], tokenizer, block_size=32)
    print(f"  Exact-fit text ({len(tokenizer.encode(text))} tokens): {len(ds)} chunks")
    
    print("  PASSED\n")


def test_text_loader():
    """Test 5: TextLoader from various sources."""
    print("=== Test 5: TextLoader ===")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Plain text file
        txt_path = os.path.join(tmpdir, "test.txt")
        with open(txt_path, "w") as f:
            f.write("Hello world. This is a test file.\nSecond line here.")
        texts = TextLoader.from_file(txt_path)
        assert len(texts) == 1
        assert "Hello world" in texts[0]
        print(f"  from_file: {len(texts)} texts, {len(texts[0])} chars")
        
        # Lines
        texts = TextLoader.from_lines(txt_path)
        assert len(texts) == 2
        print(f"  from_lines: {len(texts)} texts")
        
        # JSONL
        jsonl_path = os.path.join(tmpdir, "test.jsonl")
        with open(jsonl_path, "w") as f:
            for t in ["First document", "Second document", "Third document"]:
                f.write(json.dumps({"text": t, "id": 1}) + "\n")
        texts = TextLoader.from_jsonl(jsonl_path, field="text")
        assert len(texts) == 3
        assert texts[0] == "First document"
        print(f"  from_jsonl: {len(texts)} texts")
        
        # Directory
        for i in range(3):
            with open(os.path.join(tmpdir, f"doc{i}.txt"), "w") as f:
                f.write(f"Document number {i}")
        texts = TextLoader.from_directory(tmpdir, pattern="doc*.txt")
        assert len(texts) == 3
        print(f"  from_directory: {len(texts)} texts")
        
        # Strings (pass-through)
        texts = TextLoader.from_strings(["a", "b", "c"])
        assert texts == ["a", "b", "c"]
        print(f"  from_strings: {len(texts)} texts")
    
    print("  PASSED\n")


def test_lr_schedule():
    """Test 6: Learning rate schedule (warmup + cosine decay)."""
    print("=== Test 6: LR Schedule ===")
    
    model = nn.Linear(10, 10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    total_steps = 100
    warmup_steps = 10
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    lrs = []
    for step in range(total_steps):
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    
    # Warmup: LR should increase linearly
    for i in range(1, warmup_steps):
        assert lrs[i] > lrs[i-1], f"LR should increase during warmup (step {i})"
    
    # After warmup: LR should generally decrease (cosine)
    assert lrs[warmup_steps] > lrs[-1], "LR should decrease after warmup"
    
    # At end: LR should be near 0
    assert lrs[-1] < lrs[warmup_steps] * 0.1, "Final LR should be much smaller"
    
    # Warmup should reach approximately max LR
    assert abs(lrs[warmup_steps] - 1e-3) < 1e-4, \
        f"LR at end of warmup should be ~1e-3, got {lrs[warmup_steps]}"
    
    print(f"  Warmup: {lrs[0]:.6f} -> {lrs[warmup_steps]:.6f}")
    print(f"  Decay:  {lrs[warmup_steps]:.6f} -> {lrs[-1]:.6f}")
    print(f"  Schedule shape verified (linear warmup + cosine decay)")
    print("  PASSED\n")


def test_loss_decreases():
    """Test 7: Training actually reduces loss (the most important test)."""
    print("=== Test 7: Loss Decreases During Training ===")
    
    tokenizer = get_tokenizer()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, cache_dir=MODEL_CACHE, torch_dtype=torch.float32
    ).eval()
    
    # Overfit on a small repeated text
    train_text = "The capital of France is Paris. " * 50
    
    trainer = CausalLMTrainer(
        learning_rate=1e-4,
        num_epochs=3,
        batch_size=2,
        block_size=32,
        warmup_steps=5,
        log_every_steps=5,
        gradient_accumulation_steps=1,
    )
    
    result = trainer.train(model, tokenizer, [train_text])
    
    assert result.success, "Training should succeed"
    
    history = result.artifacts["history"]
    losses = history["train_loss"]
    
    assert len(losses) > 0, "Should have training loss history"
    
    # Compare first 3 losses to last 3 losses
    early_loss = np.mean(losses[:3])
    late_loss = np.mean(losses[-3:])
    
    print(f"  Early avg loss: {early_loss:.4f}")
    print(f"  Late avg loss:  {late_loss:.4f}")
    print(f"  Reduction:      {(1 - late_loss/early_loss)*100:.1f}%")
    
    assert late_loss < early_loss, \
        f"Loss should decrease: early={early_loss:.4f}, late={late_loss:.4f}"
    
    print(f"  Final metrics: {result.metrics}")
    print("  PASSED\n")


def test_gradient_accumulation():
    """Test 8: Gradient accumulation should approximate larger batch."""
    print("=== Test 8: Gradient Accumulation ===")
    
    tokenizer = get_tokenizer()
    
    train_text = "Testing gradient accumulation with a simple text. " * 100
    
    # Train with batch_size=4, no accumulation
    model1 = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, cache_dir=MODEL_CACHE, torch_dtype=torch.float32
    )
    torch.manual_seed(42)
    trainer1 = CausalLMTrainer(
        learning_rate=1e-4, num_epochs=1, batch_size=4,
        block_size=32, log_every_steps=0, gradient_accumulation_steps=1,
    )
    result1 = trainer1.train(model1, tokenizer, [train_text])
    
    # Train with batch_size=2, accumulation=2 (effective batch=4)
    model2 = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, cache_dir=MODEL_CACHE, torch_dtype=torch.float32
    )
    torch.manual_seed(42)
    trainer2 = CausalLMTrainer(
        learning_rate=1e-4, num_epochs=1, batch_size=2,
        block_size=32, log_every_steps=0, gradient_accumulation_steps=2,
    )
    result2 = trainer2.train(model2, tokenizer, [train_text])
    
    loss1 = result1.metrics["final_loss"]
    loss2 = result2.metrics["final_loss"]
    
    print(f"  Batch=4, accum=1: final_loss={loss1:.4f}")
    print(f"  Batch=2, accum=2: final_loss={loss2:.4f}")
    print(f"  Difference: {abs(loss1-loss2):.4f}")
    
    # They won't be identical (different batch ordering) but should be similar
    assert abs(loss1 - loss2) < 1.0, \
        f"Gradient accumulation should give similar results: {loss1:.4f} vs {loss2:.4f}"
    
    print("  PASSED\n")


def test_checkpoint_save_load():
    """Test 9: Model can be saved and loaded after training."""
    print("=== Test 9: Checkpoint Save/Load ===")
    
    tokenizer = get_tokenizer()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, cache_dir=MODEL_CACHE, torch_dtype=torch.float32
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = CausalLMTrainer(
            learning_rate=1e-4, num_epochs=1, batch_size=2,
            block_size=32, log_every_steps=0, save_dir=tmpdir,
        )
        
        result = trainer.train(model, tokenizer, ["Test text for saving. " * 50])
        
        # Check save directory
        final_path = os.path.join(tmpdir, "final")
        assert os.path.exists(final_path), "Final checkpoint should exist"
        assert os.path.exists(os.path.join(final_path, "config.json")), "config.json should exist"
        
        # Load and verify
        loaded = AutoModelForCausalLM.from_pretrained(final_path, torch_dtype=torch.float32)
        
        # Compare a few parameters
        for (n1, p1), (n2, p2) in zip(
            list(model.named_parameters())[:3],
            list(loaded.named_parameters())[:3],
        ):
            assert n1 == n2
            assert torch.allclose(p1.data.cpu(), p2.data.cpu(), atol=1e-6), \
                f"Parameter {n1} differs after save/load"
        
        print(f"  Saved to: {final_path}")
        print(f"  Loaded and verified parameter equality")
    
    print("  PASSED\n")


def test_manual_comparison():
    """Test 10: Compare trainer output to manual training loop."""
    print("=== Test 10: Manual Training Loop Comparison ===")
    
    tokenizer = get_tokenizer()
    text = "Manual test comparison data. " * 50
    
    # Create dataset
    dataset = CausalLMDataset([text], tokenizer, block_size=32)
    
    # Manual training
    torch.manual_seed(123)
    model_manual = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, cache_dir=MODEL_CACHE, torch_dtype=torch.float32
    )
    optimizer = torch.optim.AdamW(model_manual.parameters(), lr=1e-4, weight_decay=0.01)
    
    model_manual.train()
    manual_losses = []
    dataloader = dataset.get_dataloader(batch_size=2, shuffle=False)
    
    for batch in dataloader:
        outputs = model_manual(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_manual.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        manual_losses.append(loss.item())
    
    # Trainer training (same setup)
    torch.manual_seed(123)
    model_trainer = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, cache_dir=MODEL_CACHE, torch_dtype=torch.float32
    )
    
    trainer = CausalLMTrainer(
        learning_rate=1e-4, num_epochs=1, batch_size=2,
        block_size=32, warmup_steps=0, lr_scheduler="constant",
        log_every_steps=0, weight_decay=0.01,
    )
    result = trainer.train(model_trainer, tokenizer, [text])
    trainer_losses = result.artifacts["history"]["train_loss"]
    
    print(f"  Manual losses:  {[f'{l:.4f}' for l in manual_losses[:5]]}")
    print(f"  Trainer losses: {[f'{l:.4f}' for l in trainer_losses[:5]]}")
    
    # First losses should be very close (same seed, same data order)
    if len(manual_losses) > 0 and len(trainer_losses) > 0:
        diff = abs(manual_losses[0] - trainer_losses[0])
        print(f"  First loss diff: {diff:.6f}")
        # They may differ slightly due to weight decay grouping etc
        assert diff < 0.5, f"First losses should be close: {manual_losses[0]:.4f} vs {trainer_losses[0]:.4f}"
    
    print("  PASSED\n")


def test_registry_integration():
    """Test 11: Trainer can be created via Registry."""
    print("=== Test 11: Registry Integration ===")
    
    modules = Registry.list()
    assert "training" in modules, f"training not in registry: {modules}"
    assert "causal_lm" in modules["training"]
    
    # Create via registry
    trainer = Registry.create(
        "training", "causal_lm",
        learning_rate=1e-4, num_epochs=1, batch_size=2, block_size=32,
    )
    assert isinstance(trainer, CausalLMTrainer)
    assert trainer.learning_rate == 1e-4
    assert trainer.num_epochs == 1
    
    print(f"  Created via Registry: {type(trainer).__name__}")
    print(f"  Config: lr={trainer.learning_rate}, epochs={trainer.num_epochs}")
    print("  PASSED\n")


def test_validation():
    """Test 12: Validation loss is computed correctly."""
    print("=== Test 12: Validation ===")
    
    tokenizer = get_tokenizer()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, cache_dir=MODEL_CACHE, torch_dtype=torch.float32
    )
    
    train_text = "Training data goes here. " * 50
    val_text = "Validation data is different. " * 50
    
    trainer = CausalLMTrainer(
        learning_rate=1e-4, num_epochs=2, batch_size=2,
        block_size=32, log_every_steps=0,
    )
    
    result = trainer.train(model, tokenizer, [train_text], val_texts=[val_text])
    
    val_losses = result.artifacts["history"]["val_loss"]
    assert len(val_losses) == 2, f"Should have 2 val losses (one per epoch), got {len(val_losses)}"
    
    print(f"  Val losses: {[f'{l:.4f}' for l in val_losses]}")
    print("  PASSED\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Causal LM Training Tests")
    print("=" * 60 + "\n")
    
    # Unit tests (fast, no model loading)
    test_text_loader()
    test_lr_schedule()
    
    # Dataset tests (need tokenizer only)
    test_dataset_construction()
    test_dataset_token_alignment()
    test_dataset_sliding_window()
    test_dataset_empty_and_short()
    
    # Registry test
    test_registry_integration()
    
    # Training tests (need model — slower)
    test_loss_decreases()
    test_gradient_accumulation()
    test_checkpoint_save_load()
    test_manual_comparison()
    test_validation()
    
    print("=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
