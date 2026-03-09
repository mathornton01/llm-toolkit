#!/usr/bin/env python3
"""
Integration test for LLM Toolkit.

Tests:
1. Registry discovery
2. Each pruning method (magnitude, wanda, actmag, random)
3. Activation collection
4. Perplexity evaluation
5. Full pipeline (collect -> prune -> evaluate)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from llm_toolkit.core import Registry, PipelineConfig, Pipeline
from llm_toolkit.core.config import ModuleConfig

# Import modules to trigger registration
import llm_toolkit.pruning
import llm_toolkit.collection
import llm_toolkit.evaluation


def test_registry():
    print("=== Test 1: Registry ===")
    modules = Registry.list()
    print(f"  Registered categories: {list(modules.keys())}")
    for cat, names in modules.items():
        print(f"    {cat}: {names}")
    
    assert "pruning" in modules
    assert "magnitude" in modules["pruning"]
    assert "wanda" in modules["pruning"]
    assert "actmag" in modules["pruning"]
    assert "random" in modules["pruning"]
    assert "evaluation" in modules
    assert "perplexity" in modules["evaluation"]
    assert "collection" in modules
    assert "activations" in modules["collection"]
    print("  PASSED\n")


def test_pruning_methods():
    """Test all pruning methods on TinyLlama."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import copy
    
    print("=== Test 2: Loading TinyLlama ===")
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="models")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name, cache_dir="models", torch_dtype=torch.float32
    ).eval()
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded: {n_params:,} params")
    print(f"  Layers: {model.config.num_hidden_layers}, MLP: {model.config.intermediate_size}")
    
    # Collect activations (needed for wanda/actmag)
    print("\n=== Test 3: Activation Collection ===")
    prompts = [
        "Write a Python function to sort a list.",
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Translate hello to Japanese.",
    ]
    collector = Registry.create("collection", "activations")
    act_data = collector.collect(model, tokenizer, prompts)
    print(f"  Collected {act_data['n_prompts']} prompts, {act_data['n_layers']} layers")
    activations = act_data["activations"]
    print(f"  Activation shape (layer 0): {activations[0].shape}")
    print("  PASSED\n")
    
    # Baseline evaluation
    print("=== Test 4: Baseline Perplexity ===")
    evaluator = Registry.create("evaluation", "perplexity")
    baseline = evaluator.evaluate(model, tokenizer)
    print(f"  Baseline: {baseline}")
    print("  PASSED\n")
    
    # Test each pruning method
    methods = ["magnitude", "wanda", "actmag", "random"]
    keep_ratio = 0.7
    results = {}
    
    for method_name in methods:
        print(f"=== Test 5.{methods.index(method_name)+1}: {method_name} pruning ===")
        pruner = Registry.create("pruning", method_name, keep_ratio=keep_ratio)
        
        # Deep copy model for each test
        m = copy.deepcopy(model)
        
        # Compute importance
        importance = pruner.compute_importance(m, activations=activations)
        print(f"  Importance layers: {len(importance)}")
        print(f"  Layer 0 shape: {importance[0].shape}")
        print(f"  Layer 0 range: [{importance[0].min():.4f}, {importance[0].max():.4f}]")
        
        # Prune
        result = pruner.prune(m, importance)
        print(f"  {result.logs[0]}")
        
        # Evaluate
        ppl_result = evaluator.evaluate(m, tokenizer)
        ppl = ppl_result.metrics["perplexity"]
        results[method_name] = ppl
        print(f"  Perplexity: {ppl:.2f} (baseline: {baseline.metrics['perplexity']:.2f})")
        print("  PASSED\n")
        
        del m
    
    # Summary
    print("=== Summary: Pruning @ {:.0%} keep ===".format(keep_ratio))
    print(f"  {'Method':<15} {'PPL':>10} {'vs Baseline':>12}")
    print(f"  {'-'*40}")
    base_ppl = baseline.metrics["perplexity"]
    print(f"  {'baseline':<15} {base_ppl:>10.2f} {'':>12}")
    for method, ppl in sorted(results.items(), key=lambda x: x[1]):
        ratio = ppl / base_ppl
        print(f"  {method:<15} {ppl:>10.2f} {ratio:>11.1f}x")
    print()


def test_pipeline():
    """Test the full pipeline API."""
    print("=== Test 6: Full Pipeline ===")
    
    config = PipelineConfig(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        model_cache="models",
        device="cpu",
        pruning=ModuleConfig(name="wanda", params={"keep_ratio": 0.8}),
        evaluation=ModuleConfig(name="perplexity"),
        calibration_prompts=[
            "Write a sorting algorithm in Python.",
            "What causes rainbows?",
            "Translate 'hello world' to Spanish.",
            "Explain the Pythagorean theorem.",
        ],
    )
    
    pipeline = Pipeline(config)
    results = pipeline.run()
    
    assert "baseline" in results
    assert "pruning" in results
    assert "evaluation" in results
    
    base_ppl = results["baseline"].metrics["perplexity"]
    final_ppl = results["evaluation"].metrics["perplexity"]
    print(f"\n  Baseline PPL: {base_ppl:.2f}")
    print(f"  Final PPL:    {final_ppl:.2f}")
    print(f"  Ratio:        {final_ppl/base_ppl:.2f}x")
    print("  PASSED\n")


if __name__ == "__main__":
    test_registry()
    test_pruning_methods()
    test_pipeline()
    print("ALL TESTS PASSED!")
