#!/usr/bin/env python3
"""
Comprehensive tests for the GPT model implementation.

Tests:
  1. Model construction at various sizes
  2. Parameter count matches expectations
  3. Forward pass shapes
  4. Causal masking (future tokens can't influence past)
  5. Loss computation
  6. Weight tying (embedding == lm_head)
  7. Generation (autoregressive decoding)
  8. Save/load roundtrip
  9. Gradient flow (all parameters receive gradients)
  10. SwiGLU activation variant
  11. Training: loss decreases on repeated data
  12. Comparison with known transformer math
"""
import sys, os, math, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from llm_toolkit.models.gpt import GPT, GPTConfig


def test_construction():
    """Test 1: Model construction at various sizes."""
    print("=== Test 1: Model Construction ===")
    
    configs = {
        "tiny (1M)": GPTConfig(n_layers=2, n_heads=2, d_model=64, d_ff=256, 
                                vocab_size=1000, max_seq_len=64),
        "small (10M)": GPTConfig(n_layers=4, n_heads=4, d_model=256, d_ff=1024,
                                  vocab_size=8000, max_seq_len=256),
        "medium (50M)": GPTConfig(n_layers=6, n_heads=8, d_model=512, d_ff=2048,
                                   vocab_size=32000, max_seq_len=512),
    }
    
    for name, config in configs.items():
        model = GPT(config)
        n_params = sum(p.numel() for p in model.parameters())
        estimated = config.estimate_params()
        print(f"  {name}: {n_params:,} actual, {estimated:,} estimated")
        # Estimate should be within 5% (doesn't count all small buffers)
        assert abs(n_params - estimated) / n_params < 0.05, \
            f"Estimate off by {abs(n_params-estimated)/n_params:.1%}"
    
    print("  PASSED\n")


def test_param_count_math():
    """Test 2: Verify parameter count matches hand-calculated values."""
    print("=== Test 2: Parameter Count Math ===")
    
    config = GPTConfig(
        n_layers=2, n_heads=2, d_model=64, d_ff=256,
        vocab_size=100, max_seq_len=32, bias=False, tie_weights=True,
    )
    model = GPT(config)
    counts = model.count_parameters()
    
    # Token embedding: 100 * 64 = 6,400
    assert counts["token_embedding"] == 100 * 64, f"tok_emb: {counts['token_embedding']}"
    # Position embedding: 32 * 64 = 2,048
    assert counts["position_embedding"] == 32 * 64, f"pos_emb: {counts['position_embedding']}"
    # lm_head should be 0 (tied)
    assert counts["lm_head"] == 0, f"lm_head should be tied: {counts['lm_head']}"
    
    # Attention per block: QKV proj (64*192) + out proj (64*64) = 12288 + 4096 = 16384
    # But also dropout layers (no params)
    # MLP per block: up (64*256) + down (256*64) = 16384 + 16384 = 32768
    # Total attention for 2 blocks should be 2 * (QKV + out) = 2 * (64*192 + 64*64)
    
    print(f"  Counts: {counts}")
    print(f"  Total: {counts['total']:,}")
    print("  PASSED\n")


def test_forward_shapes():
    """Test 3: Forward pass produces correct output shapes."""
    print("=== Test 3: Forward Pass Shapes ===")
    
    config = GPTConfig(
        n_layers=2, n_heads=4, d_model=128, d_ff=512,
        vocab_size=1000, max_seq_len=64,
    )
    model = GPT(config)
    
    batch_size, seq_len = 3, 20
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    # Without targets
    logits, loss = model(input_ids)
    assert logits.shape == (batch_size, seq_len, config.vocab_size), \
        f"Expected ({batch_size}, {seq_len}, {config.vocab_size}), got {logits.shape}"
    assert loss is None, "Loss should be None without targets"
    
    # With targets
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    logits, loss = model(input_ids, targets=targets)
    assert logits.shape == (batch_size, seq_len, config.vocab_size)
    assert loss is not None, "Loss should be computed with targets"
    assert loss.dim() == 0, "Loss should be scalar"
    assert loss.item() > 0, "Loss should be positive"
    
    print(f"  Logits shape: {logits.shape}")
    print(f"  Loss: {loss.item():.4f}")
    
    # Expected loss for random predictions: log(vocab_size)
    expected_random = math.log(config.vocab_size)
    assert abs(loss.item() - expected_random) < 1.5, \
        f"Random init loss ({loss.item():.2f}) should be near log(V)={expected_random:.2f}"
    print(f"  Expected ~{expected_random:.2f} for random init, got {loss.item():.2f}")
    print("  PASSED\n")


def test_causal_masking():
    """Test 4: Verify causal masking — future tokens don't influence past positions."""
    print("=== Test 4: Causal Masking ===")
    
    config = GPTConfig(
        n_layers=2, n_heads=2, d_model=64, d_ff=256,
        vocab_size=100, max_seq_len=32, dropout=0.0, attn_dropout=0.0,
    )
    model = GPT(config)
    model.eval()
    
    # Create two sequences that differ only at position 5
    seq1 = torch.tensor([[1, 2, 3, 4, 5, 10, 20, 30]])
    seq2 = torch.tensor([[1, 2, 3, 4, 5, 99, 88, 77]])  # Different from pos 5 onward
    
    with torch.no_grad():
        logits1, _ = model(seq1)
        logits2, _ = model(seq2)
    
    # Positions 0-4 should produce IDENTICAL logits (causal = can't see future)
    for pos in range(5):
        diff = (logits1[0, pos] - logits2[0, pos]).abs().max().item()
        assert diff < 1e-5, \
            f"Position {pos} differs by {diff:.6f} — causal masking broken!"
    
    # Position 5+ should differ (they see different context)
    for pos in range(5, 8):
        diff = (logits1[0, pos] - logits2[0, pos]).abs().max().item()
        assert diff > 1e-3, \
            f"Position {pos} should differ (diff={diff:.6f})"
    
    print("  Positions 0-4: identical (future tokens don't leak)")
    print("  Positions 5-7: different (correctly sees different past)")
    print("  PASSED\n")


def test_weight_tying():
    """Test 5: Token embedding and lm_head share the same weight tensor."""
    print("=== Test 5: Weight Tying ===")
    
    # With tying
    config_tied = GPTConfig(
        n_layers=2, n_heads=2, d_model=64, d_ff=256,
        vocab_size=100, max_seq_len=32, tie_weights=True,
    )
    model_tied = GPT(config_tied)
    assert model_tied.tok_emb.weight.data_ptr() == model_tied.lm_head.weight.data_ptr(), \
        "Weights should share same memory"
    
    n_tied = sum(p.numel() for p in model_tied.parameters())
    
    # Without tying
    config_untied = GPTConfig(
        n_layers=2, n_heads=2, d_model=64, d_ff=256,
        vocab_size=100, max_seq_len=32, tie_weights=False,
    )
    model_untied = GPT(config_untied)
    assert model_untied.tok_emb.weight.data_ptr() != model_untied.lm_head.weight.data_ptr(), \
        "Weights should NOT share memory"
    
    n_untied = sum(p.numel() for p in model_untied.parameters())
    
    # Untied should have exactly vocab_size * d_model more params
    expected_diff = 100 * 64
    actual_diff = n_untied - n_tied
    assert actual_diff == expected_diff, \
        f"Untied should have {expected_diff} more params, got {actual_diff}"
    
    print(f"  Tied: {n_tied:,} params")
    print(f"  Untied: {n_untied:,} params (diff={actual_diff}={100}*{64})")
    print("  PASSED\n")


def test_generation():
    """Test 6: Autoregressive generation produces valid tokens."""
    print("=== Test 6: Generation ===")
    
    config = GPTConfig(
        n_layers=2, n_heads=2, d_model=64, d_ff=256,
        vocab_size=100, max_seq_len=32,
    )
    model = GPT(config)
    model.eval()
    
    prompt = torch.tensor([[1, 2, 3]])
    
    # Greedy (temperature → 0 by using very low temp)
    output = model.generate(prompt, max_new_tokens=10, temperature=0.1)
    assert output.shape == (1, 13), f"Expected (1, 13), got {output.shape}"
    assert (output[:, :3] == prompt).all(), "Prompt should be preserved"
    assert (output >= 0).all() and (output < config.vocab_size).all(), \
        "All tokens should be valid vocab indices"
    
    # Top-k
    output_topk = model.generate(prompt, max_new_tokens=5, top_k=10)
    assert output_topk.shape == (1, 8)
    
    # Top-p
    output_topp = model.generate(prompt, max_new_tokens=5, top_p=0.9)
    assert output_topp.shape == (1, 8)
    
    print(f"  Generated: {output[0].tolist()}")
    print(f"  All tokens in valid range [0, {config.vocab_size})")
    print("  PASSED\n")


def test_save_load():
    """Test 7: Save and load preserves weights exactly."""
    print("=== Test 7: Save/Load ===")
    
    config = GPTConfig(
        n_layers=2, n_heads=2, d_model=64, d_ff=256,
        vocab_size=100, max_seq_len=32,
    )
    model = GPT(config)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        model.save_pretrained(tmpdir)
        loaded = GPT.from_pretrained(tmpdir)
        
        # Compare all parameters
        for (n1, p1), (n2, p2) in zip(
            model.named_parameters(), loaded.named_parameters()
        ):
            assert n1 == n2, f"Parameter names differ: {n1} vs {n2}"
            assert torch.equal(p1.data, p2.data), f"Parameter {n1} differs"
        
        # Compare forward pass
        x = torch.randint(0, 100, (1, 10))
        model.eval()
        loaded.eval()
        with torch.no_grad():
            l1, _ = model(x)
            l2, _ = loaded(x)
        assert torch.allclose(l1, l2, atol=1e-6), "Forward pass differs after load"
    
    print("  All parameters match after save/load")
    print("  Forward pass matches")
    print("  PASSED\n")


def test_gradient_flow():
    """Test 8: All parameters receive gradients during backprop."""
    print("=== Test 8: Gradient Flow ===")
    
    config = GPTConfig(
        n_layers=3, n_heads=2, d_model=64, d_ff=256,
        vocab_size=100, max_seq_len=32,
    )
    model = GPT(config)
    model.train()
    
    x = torch.randint(0, 100, (2, 16))
    _, loss = model(x, targets=x)
    loss.backward()
    
    no_grad = []
    zero_grad = []
    for name, param in model.named_parameters():
        if param.grad is None:
            no_grad.append(name)
        elif param.grad.abs().sum() == 0:
            zero_grad.append(name)
    
    if no_grad:
        print(f"  WARNING: No grad: {no_grad}")
    if zero_grad:
        print(f"  WARNING: Zero grad: {zero_grad}")
    
    assert len(no_grad) == 0, f"Parameters without gradients: {no_grad}"
    
    print(f"  All {sum(1 for _ in model.parameters())} parameters received gradients")
    print("  PASSED\n")


def test_swiglu():
    """Test 9: SwiGLU activation variant works correctly."""
    print("=== Test 9: SwiGLU Activation ===")
    
    # Standard GELU
    config_gelu = GPTConfig(
        n_layers=2, n_heads=2, d_model=64, d_ff=256,
        vocab_size=100, max_seq_len=32, activation="gelu",
    )
    model_gelu = GPT(config_gelu)
    n_gelu = sum(p.numel() for p in model_gelu.parameters())
    
    # SwiGLU (should have more params due to gate projection)
    config_swiglu = GPTConfig(
        n_layers=2, n_heads=2, d_model=64, d_ff=256,
        vocab_size=100, max_seq_len=32, activation="swiglu",
    )
    model_swiglu = GPT(config_swiglu)
    n_swiglu = sum(p.numel() for p in model_swiglu.parameters())
    
    assert n_swiglu > n_gelu, \
        f"SwiGLU ({n_swiglu}) should have more params than GELU ({n_gelu})"
    
    # Both should produce valid outputs
    x = torch.randint(0, 100, (2, 10))
    logits_gelu, _ = model_gelu(x)
    logits_swiglu, _ = model_swiglu(x)
    assert logits_gelu.shape == logits_swiglu.shape
    
    # SwiGLU extra params should be exactly n_layers * d_model * d_ff (gate_proj)
    expected_extra = 2 * 64 * 256  # n_layers * d_model * d_ff
    actual_extra = n_swiglu - n_gelu
    assert actual_extra == expected_extra, \
        f"SwiGLU extra params: expected {expected_extra}, got {actual_extra}"
    
    print(f"  GELU:   {n_gelu:,} params")
    print(f"  SwiGLU: {n_swiglu:,} params (+{actual_extra} for gate projection)")
    print("  PASSED\n")


def test_training_loss_decreases():
    """Test 10: Loss actually decreases when training on repeated data."""
    print("=== Test 10: Training Loss Decrease ===")
    
    config = GPTConfig(
        n_layers=2, n_heads=2, d_model=64, d_ff=256,
        vocab_size=100, max_seq_len=32,
    )
    model = GPT(config)
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Repeated data (should overfit quickly)
    data = torch.randint(0, 100, (4, 20))
    
    losses = []
    for step in range(50):
        _, loss = model(data, targets=data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    
    early = sum(losses[:5]) / 5
    late = sum(losses[-5:]) / 5
    reduction = (1 - late / early) * 100
    
    print(f"  Early loss: {early:.4f}")
    print(f"  Late loss:  {late:.4f}")
    print(f"  Reduction:  {reduction:.1f}%")
    
    assert late < early, f"Loss should decrease: {early:.4f} -> {late:.4f}"
    assert reduction > 30, f"Should reduce by >30%, got {reduction:.1f}%"
    print("  PASSED\n")


def test_seq_len_exceeded():
    """Test 11: Error on sequence exceeding max_seq_len."""
    print("=== Test 11: Sequence Length Check ===")
    
    config = GPTConfig(
        n_layers=2, n_heads=2, d_model=64, d_ff=256,
        vocab_size=100, max_seq_len=16,
    )
    model = GPT(config)
    
    # Should work at max length
    x = torch.randint(0, 100, (1, 16))
    logits, _ = model(x)
    assert logits.shape == (1, 16, 100)
    
    # Should fail beyond max length
    x_long = torch.randint(0, 100, (1, 17))
    try:
        model(x_long)
        assert False, "Should have raised AssertionError"
    except AssertionError as e:
        assert "exceeds max" in str(e)
        print(f"  Correctly rejected seq_len=17 > max=16")
    
    print("  PASSED\n")


def test_transformer_math():
    """Test 12: Verify attention computation matches textbook formula."""
    print("=== Test 12: Attention Math Verification ===")
    
    from llm_toolkit.models.gpt import CausalSelfAttention
    
    config = GPTConfig(
        n_layers=1, n_heads=2, d_model=8, d_ff=32,
        vocab_size=100, max_seq_len=8, dropout=0.0, attn_dropout=0.0,
    )
    
    attn = CausalSelfAttention(config)
    attn.eval()
    
    # Manual computation
    x = torch.randn(1, 4, 8)  # (batch=1, seq=4, d_model=8)
    
    with torch.no_grad():
        # Get Q, K, V from the module
        qkv = attn.qkv_proj(x)
        q, k, v = qkv.split(8, dim=2)
        
        # Reshape to heads
        q = q.view(1, 4, 2, 4).transpose(1, 2)  # (1, 2, 4, 4)
        k = k.view(1, 4, 2, 4).transpose(1, 2)
        v = v.view(1, 4, 2, 4).transpose(1, 2)
        
        # Manual attention
        scale = 1.0 / math.sqrt(4)  # head_dim = 4
        scores = (q @ k.transpose(-2, -1)) * scale
        
        # Causal mask
        mask = torch.tril(torch.ones(4, 4))
        scores = scores.masked_fill(mask == 0, float("-inf"))
        
        attn_weights = F.softmax(scores, dim=-1)
        manual_out = attn_weights @ v
        manual_out = manual_out.transpose(1, 2).contiguous().view(1, 4, 8)
        manual_out = attn.out_proj(manual_out)
        
        # Module output
        module_out = attn(x)
    
    diff = (manual_out - module_out).abs().max().item()
    assert diff < 1e-5, f"Manual vs module attention differs by {diff}"
    
    # Verify attention weights sum to 1
    assert torch.allclose(attn_weights.sum(dim=-1), torch.ones(1, 2, 4), atol=1e-5), \
        "Attention weights should sum to 1"
    
    # Verify causal: no attention to future positions
    for t in range(4):
        for t2 in range(t + 1, 4):
            assert attn_weights[0, :, t, t2].abs().max() < 1e-6, \
                f"Position {t} should not attend to future position {t2}"
    
    print(f"  Manual vs module diff: {diff:.8f}")
    print(f"  Attention weights sum to 1: verified")
    print(f"  No attention to future positions: verified")
    print("  PASSED\n")


if __name__ == "__main__":
    print("=" * 60)
    print("GPT Model Tests")
    print("=" * 60 + "\n")
    
    test_construction()
    test_param_count_math()
    test_forward_shapes()
    test_causal_masking()
    test_weight_tying()
    test_generation()
    test_save_load()
    test_gradient_flow()
    test_swiglu()
    test_training_loss_decreases()
    test_seq_len_exceeded()
    test_transformer_math()
    
    print("=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
