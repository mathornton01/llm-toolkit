"""
Perplexity evaluator.

Computes perplexity on a set of evaluation texts.
Works with both HuggingFace models and our custom GPT model.
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
    """Compute perplexity on evaluation texts.
    
    Works with both:
    - HuggingFace models (output.loss attribute)
    - Custom GPT models (ModelOutput with .loss)
    
    Supports AMP (automatic mixed precision) for GPU evaluation.
    """
    
    name = "perplexity"
    
    def __init__(self, max_seq_len: int = 256, use_amp: bool = True, **kwargs):
        self.max_seq_len = max_seq_len
        self.use_amp = use_amp
    
    def evaluate(
        self,
        model,
        tokenizer,
        eval_texts: Optional[List[str]] = None,
        **kwargs,
    ) -> ModuleResult:
        texts = eval_texts or DEFAULT_EVAL_TEXTS
        device = next(model.parameters()).device
        is_cuda = device.type == "cuda"
        use_amp = self.use_amp and is_cuda
        amp_dtype = torch.bfloat16 if is_cuda else torch.float32
        
        model.eval()
        total_loss = 0
        total_tokens = 0
        per_text_ppl = []
        
        for text in texts:
            enc = tokenizer(
                text, return_tensors="pt",
                truncation=True, max_length=self.max_seq_len
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(device)
            
            with torch.no_grad():
                if use_amp:
                    with torch.amp.autocast("cuda", dtype=amp_dtype):
                        out = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=input_ids,
                        )
                else:
                    out = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=input_ids,
                    )
                
                # out.loss works for both HuggingFace and our ModelOutput
                loss_val = out.loss.cpu().item()
                n_tokens = input_ids.shape[1]
                total_loss += loss_val * n_tokens
                total_tokens += n_tokens
                per_text_ppl.append(float(np.exp(min(loss_val, 100))))
        
        ppl = float(np.exp(min(total_loss / total_tokens, 100)))
        
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
