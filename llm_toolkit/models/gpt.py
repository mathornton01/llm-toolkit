"""
Configurable GPT Model.

A clean, from-scratch implementation of the GPT (decoder-only transformer)
architecture. Configurable from tiny (~1M params) to large (~1B+ params).

This is the same architecture used by GPT-2, GPT-3, LLaMA (with minor
variations), and most modern autoregressive LLMs.

Architecture overview:
    Input tokens → Token Embedding + Position Embedding
        → N × TransformerBlock:
            → LayerNorm → Multi-Head Self-Attention (causal mask) → Residual
            → LayerNorm → MLP (expand → activation → project) → Residual
        → LayerNorm → Output projection (vocab logits)

Key design choices:
    - Pre-norm (LayerNorm before attention/MLP) — standard since GPT-2
    - Learned positional embeddings (simple, works well up to ~2K context)
    - Optional: RoPE, GQA, SwiGLU (modern variants)
    - Weight tying: output projection shares weights with token embedding
    - Proper weight initialization (scaled by 1/sqrt(2*n_layers))

Reference:
    Radford et al., "Language Models are Unsupervised Multitask Learners" (2019)
    Vaswani et al., "Attention Is All You Need" (2017)

Example:
    config = GPTConfig(n_layers=6, n_heads=8, d_model=512, vocab_size=32000)
    model = GPT(config)  # ~50M params
    logits = model(input_ids)  # (batch, seq_len, vocab_size)
"""
import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    """Configuration for a GPT model.
    
    Presets:
        ~10M:  n_layers=4,  d_model=256,  n_heads=4,  d_ff=1024
        ~50M:  n_layers=6,  d_model=512,  n_heads=8,  d_ff=2048
        ~100M: n_layers=12, d_model=768,  n_heads=12, d_ff=3072
        ~350M: n_layers=24, d_model=1024, n_heads=16, d_ff=4096
        ~1B:   n_layers=24, d_model=2048, n_heads=16, d_ff=8192
    """
    # Architecture
    n_layers: int = 6
    n_heads: int = 8
    d_model: int = 512
    d_ff: int = 2048          # MLP intermediate size (typically 4 * d_model)
    vocab_size: int = 32000
    max_seq_len: int = 512
    
    # Regularization
    dropout: float = 0.1
    attn_dropout: float = 0.1
    
    # Activation
    activation: str = "gelu"  # "gelu", "relu", "swiglu"
    
    # Options
    tie_weights: bool = True      # Share embedding and output projection weights
    bias: bool = False            # Use bias in linear layers (False = modern practice)
    norm_eps: float = 1e-5        # LayerNorm epsilon
    
    # Initialization
    init_std: float = 0.02        # Weight initialization std
    
    @property
    def head_dim(self) -> int:
        assert self.d_model % self.n_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        return self.d_model // self.n_heads
    
    def estimate_params(self) -> int:
        """Estimate total parameter count."""
        # Embeddings
        emb = self.vocab_size * self.d_model  # token embedding
        pos = self.max_seq_len * self.d_model  # position embedding
        
        # Per transformer block
        # Attention: Q, K, V projections + output projection
        attn = 4 * self.d_model * self.d_model  # (no bias)
        # MLP: up projection + down projection
        if self.activation == "swiglu":
            mlp = 3 * self.d_model * self.d_ff  # gate + up + down
        else:
            mlp = 2 * self.d_model * self.d_ff  # up + down
        # LayerNorms (2 per block)
        ln = 4 * self.d_model
        
        block = attn + mlp + ln
        
        # Output: final layernorm + output projection (tied or not)
        output = 2 * self.d_model  # final layernorm
        if not self.tie_weights:
            output += self.vocab_size * self.d_model
        
        return emb + pos + self.n_layers * block + output


class CausalSelfAttention(nn.Module):
    """Multi-head causal (masked) self-attention.
    
    Each position can only attend to itself and previous positions.
    This is what makes the model autoregressive — it can't "cheat"
    by looking at future tokens.
    
    Computation:
        Q, K, V = linear(x)           # Project to query, key, value
        attn = softmax(Q @ K^T / sqrt(d_k))  # Scaled dot-product attention
        attn = mask(attn)              # Zero out future positions
        output = attn @ V              # Weighted sum of values
        output = linear(output)        # Output projection
    """
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.head_dim = config.head_dim
        
        # Q, K, V projections (combined for efficiency)
        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        # Output projection
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        
        self.attn_dropout = nn.Dropout(config.attn_dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # Causal mask: lower triangular matrix
        # Registered as buffer so it moves to GPU with the model
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
                 .view(1, 1, config.max_seq_len, config.max_seq_len),
            persistent=False,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # batch, seq_len, d_model
        
        # Compute Q, K, V
        qkv = self.qkv_proj(x)  # (B, T, 3 * d_model)
        q, k, v = qkv.split(self.d_model, dim=2)
        
        # Reshape for multi-head attention: (B, n_heads, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        # attn = (Q @ K^T) / sqrt(d_k)
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale  # (B, n_heads, T, T)
        
        # Apply causal mask (set future positions to -inf → softmax gives 0)
        attn = attn.masked_fill(
            self.causal_mask[:, :, :T, :T] == 0,
            float("-inf"),
        )
        
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        
        # Weighted sum of values
        out = attn @ v  # (B, n_heads, T, head_dim)
        
        # Reshape back: (B, T, d_model)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection
        out = self.out_proj(out)
        out = self.resid_dropout(out)
        
        return out


class MLP(nn.Module):
    """Feed-forward network (MLP) within each transformer block.
    
    Standard: x → Linear(d_model, d_ff) → Activation → Linear(d_ff, d_model)
    SwiGLU:   x → (Linear_gate(x) * SiLU(Linear_up(x))) → Linear_down
    
    The MLP expands the representation to a higher dimension (d_ff),
    applies a non-linearity, then projects back down. This is where
    most of the model's "knowledge" is stored.
    """
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        
        if config.activation == "swiglu":
            # SwiGLU: used by LLaMA, Mistral (more expressive, slightly more params)
            self.gate_proj = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
            self.up_proj = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
            self.down_proj = nn.Linear(config.d_ff, config.d_model, bias=config.bias)
            self.act = nn.SiLU()
            self.use_swiglu = True
        else:
            # Standard: used by GPT-2, GPT-3
            self.up_proj = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
            self.down_proj = nn.Linear(config.d_ff, config.d_model, bias=config.bias)
            self.act = nn.GELU() if config.activation == "gelu" else nn.ReLU()
            self.use_swiglu = False
        
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_swiglu:
            out = self.act(self.gate_proj(x)) * self.up_proj(x)
        else:
            out = self.act(self.up_proj(x))
        
        out = self.down_proj(out)
        out = self.dropout(out)
        return out


class TransformerBlock(nn.Module):
    """Single transformer block (attention + MLP with residual connections).
    
    Architecture (pre-norm):
        x → LayerNorm → Attention → + (residual)
          → LayerNorm → MLP → + (residual)
    
    Pre-norm (LayerNorm before attention/MLP) is the standard since GPT-2.
    The residual connections allow gradients to flow directly through
    the network, enabling training of very deep models.
    """
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model, eps=config.norm_eps)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model, eps=config.norm_eps)
        self.mlp = MLP(config)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm attention with residual
        x = x + self.attn(self.ln1(x))
        # Pre-norm MLP with residual
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """GPT (Generative Pre-trained Transformer) model.
    
    A decoder-only transformer for autoregressive language modeling.
    
    Usage:
        config = GPTConfig(n_layers=6, n_heads=8, d_model=512, vocab_size=32000)
        model = GPT(config)
        
        # Forward pass
        logits = model(input_ids)  # (batch, seq_len, vocab_size)
        
        # With loss computation
        logits, loss = model(input_ids, targets=labels)
        
        # Generation
        tokens = model.generate(prompt_ids, max_new_tokens=100)
    """
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        
        # Token and position embeddings
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(config.d_model, eps=config.norm_eps)
        
        # Output projection (logits over vocabulary)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Weight tying: share token embedding weights with output projection
        if config.tie_weights:
            self.lm_head.weight = self.tok_emb.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        # Scale residual projections by 1/sqrt(2*n_layers) for stable training
        for pn, p in self.named_parameters():
            if pn.endswith("out_proj.weight") or pn.endswith("down_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=config.init_std / math.sqrt(2 * config.n_layers))
        
        n_params = sum(p.numel() for p in self.parameters())
        n_params_no_emb = sum(p.numel() for n, p in self.named_parameters() 
                              if "tok_emb" not in n and "pos_emb" not in n)
        print(f"GPT: {n_params:,} params ({n_params_no_emb:,} non-embedding)")
    
    def _init_weights(self, module):
        """Initialize weights with scaled normal distribution."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.
        
        Args:
            input_ids: Token indices (batch, seq_len)
            targets: Target token indices for loss (batch, seq_len)
            attention_mask: Not used for causal LM (kept for API compatibility)
        
        Returns:
            logits: (batch, seq_len, vocab_size)
            loss: Cross-entropy loss if targets provided, else None
        """
        B, T = input_ids.shape
        assert T <= self.config.max_seq_len, \
            f"Sequence length {T} exceeds max {self.config.max_seq_len}"
        
        # Token + position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=input_ids.device)
        tok_emb = self.tok_emb(input_ids)  # (B, T, d_model)
        pos_emb = self.pos_emb(pos)         # (T, d_model)
        x = self.drop(tok_emb + pos_emb)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_f(x)
        
        # Output logits
        logits = self.lm_head(x)  # (B, T, vocab_size)
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,  # Standard ignore index for padding
            )
        
        return logits, loss
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """Generate tokens autoregressively.
        
        Args:
            input_ids: Prompt tokens (batch, seq_len)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (1.0 = normal, <1 = sharper, >1 = flatter)
            top_k: Only sample from top-k most likely tokens
            top_p: Nucleus sampling threshold
        
        Returns:
            Generated token indices (batch, seq_len + max_new_tokens)
        """
        self.eval()
        
        for _ in range(max_new_tokens):
            # Crop to max_seq_len if needed
            idx_cond = input_ids if input_ids.size(1) <= self.config.max_seq_len \
                else input_ids[:, -self.config.max_seq_len:]
            
            # Forward pass
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # Last position only
            
            # Temperature scaling
            if temperature != 1.0:
                logits = logits / temperature
            
            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            
            # Top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                # Remove tokens with cumulative prob above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float("-inf")
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids
    
    def count_parameters(self) -> dict:
        """Count parameters by component."""
        counts = {}
        counts["token_embedding"] = self.tok_emb.weight.numel()
        counts["position_embedding"] = self.pos_emb.weight.numel()
        
        attn_params = 0
        mlp_params = 0
        norm_params = 0
        for block in self.blocks:
            attn_params += sum(p.numel() for p in block.attn.parameters())
            mlp_params += sum(p.numel() for p in block.mlp.parameters())
            norm_params += sum(p.numel() for p in block.ln1.parameters())
            norm_params += sum(p.numel() for p in block.ln2.parameters())
        
        counts["attention"] = attn_params
        counts["mlp"] = mlp_params
        counts["layer_norms"] = norm_params
        counts["final_norm"] = sum(p.numel() for p in self.ln_f.parameters())
        
        if not self.config.tie_weights:
            counts["lm_head"] = self.lm_head.weight.numel()
        else:
            counts["lm_head"] = 0  # Tied with token embedding
        
        counts["total"] = sum(p.numel() for p in self.parameters())
        return counts

    # ── HuggingFace-compatible interface ──────────────────────────────
    # These methods allow our GPT to work with the existing toolkit
    # pipeline (which expects HuggingFace-style model interface).
    
    @property
    def device(self):
        return next(self.parameters()).device
    
    def save_pretrained(self, path: str):
        """Save model weights and config."""
        import json, os
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, "model.pt"))
        config_dict = {
            "n_layers": self.config.n_layers,
            "n_heads": self.config.n_heads,
            "d_model": self.config.d_model,
            "d_ff": self.config.d_ff,
            "vocab_size": self.config.vocab_size,
            "max_seq_len": self.config.max_seq_len,
            "dropout": self.config.dropout,
            "attn_dropout": self.config.attn_dropout,
            "activation": self.config.activation,
            "tie_weights": self.config.tie_weights,
            "bias": self.config.bias,
        }
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def from_pretrained(cls, path: str, **kwargs):
        """Load model from saved weights and config."""
        import json, os
        with open(os.path.join(path, "config.json")) as f:
            config_dict = json.load(f)
        config = GPTConfig(**config_dict)
        model = cls(config)
        state_dict = torch.load(
            os.path.join(path, "model.pt"),
            map_location="cpu", weights_only=True,
        )
        model.load_state_dict(state_dict)
        return model
