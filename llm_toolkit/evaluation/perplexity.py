"""
Perplexity evaluator.

Computes perplexity on a set of evaluation texts.
"""
import torch
import numpy as np
from typing import List, Optional

from ..core.registry import Registry
from ..core.base import Evaluator, ModuleResult


DEFAULT_EVAL_TEXTS = [
    "The quick brown fox jumps over the lazy dog. This sentence contains every letter.",
    "In machine learning, neural networks are computational models inspired by biological neurons.",
    "Python is a high-level programming language known for its simplicity and readability.",
    "The capital of France is Paris, which is known for the Eiffel Tower and the Louvre Museum.",
]


@Registry.register("evaluation", "perplexity")
class PerplexityEvaluator(Evaluator):
    """Compute perplexity on evaluation texts."""
    
    name = "perplexity"
    
    def __init__(self, max_seq_len: int = 256, **kwargs):
        self.max_seq_len = max_seq_len
    
    def evaluate(
        self,
        model,
        tokenizer,
        eval_texts: Optional[List[str]] = None,
        **kwargs,
    ) -> ModuleResult:
        texts = eval_texts or DEFAULT_EVAL_TEXTS
        device = next(model.parameters()).device
        
        total_loss = 0
        total_tokens = 0
        per_text_ppl = []
        
        for text in texts:
            enc = tokenizer(
                text, return_tensors="pt",
                truncation=True, max_length=self.max_seq_len
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            
            with torch.no_grad():
                out = model(**enc, labels=enc["input_ids"])
                n_tokens = enc["input_ids"].shape[1]
                total_loss += out.loss.cpu().item() * n_tokens
                total_tokens += n_tokens
                per_text_ppl.append(float(np.exp(out.loss.cpu().item())))
        
        ppl = float(np.exp(total_loss / total_tokens))
        
        return ModuleResult(
            success=True,
            metrics={
                "perplexity": ppl,
                "avg_loss": total_loss / total_tokens,
                "n_texts": len(texts),
                "n_tokens": total_tokens,
            },
            artifacts={
                "per_text_perplexity": per_text_ppl,
            },
        )
