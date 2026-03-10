"""
Causal Language Model Dataset.

Prepares text data for next-token prediction training (GPT-style).
Takes raw text, tokenizes it, and chunks it into fixed-length sequences
where each token predicts the next one.

How it works:
    1. Tokenize all text into one long token sequence
    2. Split into chunks of `block_size` tokens
    3. For each chunk: input_ids = tokens[:-1], labels = tokens[1:]
    4. The model learns to predict each next token

This is the standard approach used by GPT-2, GPT-3, LLaMA, etc.

Reference:
    Radford et al., "Language Models are Unsupervised Multitask Learners" (2019)
    - Section 2.1: Training data is simply concatenated text split into sequences
"""
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional, Dict, Any
from transformers import PreTrainedTokenizer


class CausalLMDataset(Dataset):
    """Dataset for causal (autoregressive) language model training.
    
    Each sample is a fixed-length sequence where the model predicts
    the next token at each position.
    
    Args:
        texts: List of text strings to train on
        tokenizer: HuggingFace tokenizer
        block_size: Sequence length for training (default: 128)
        stride: Overlap between consecutive chunks (default: None = no overlap)
            - None or block_size: no overlap (standard)
            - < block_size: sliding window with overlap
        drop_last: Drop the final chunk if shorter than block_size
    
    Example:
        >>> dataset = CausalLMDataset(
        ...     texts=["Hello world. This is a test."],
        ...     tokenizer=tokenizer,
        ...     block_size=128,
        ... )
        >>> sample = dataset[0]
        >>> sample["input_ids"].shape  # (block_size,)
        >>> sample["labels"].shape     # (block_size,)
    """
    
    def __init__(
        self,
        texts: List[str],
        tokenizer: PreTrainedTokenizer,
        block_size: int = 128,
        stride: Optional[int] = None,
        drop_last: bool = True,
    ):
        self.block_size = block_size
        self.stride = stride or block_size
        self.tokenizer = tokenizer
        
        # Tokenize all texts and concatenate into one long sequence
        # This is the standard approach (GPT-2, LLaMA, etc.)
        all_tokens = []
        for text in texts:
            tokens = tokenizer.encode(text, add_special_tokens=False)
            all_tokens.extend(tokens)
            # Add EOS between documents
            if tokenizer.eos_token_id is not None:
                all_tokens.append(tokenizer.eos_token_id)
        
        self.all_tokens = torch.tensor(all_tokens, dtype=torch.long)
        
        # Create chunks with optional sliding window
        self.chunks = []
        # Need block_size + 1 tokens to create input/label pairs
        seq_len = block_size + 1
        
        pos = 0
        while pos + seq_len <= len(self.all_tokens):
            self.chunks.append(self.all_tokens[pos:pos + seq_len])
            pos += self.stride
        
        # Optionally include the last partial chunk (padded)
        if not drop_last and pos < len(self.all_tokens):
            remaining = self.all_tokens[pos:]
            if len(remaining) > 1:  # Need at least 2 tokens for input/label
                pad_len = seq_len - len(remaining)
                pad_token = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
                padded = torch.cat([
                    remaining,
                    torch.full((pad_len,), pad_token, dtype=torch.long),
                ])
                self.chunks.append(padded)
    
    def __len__(self) -> int:
        return len(self.chunks)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        chunk = self.chunks[idx]
        
        # input_ids: all tokens except last
        # labels: all tokens except first (shifted by 1)
        input_ids = chunk[:-1]
        labels = chunk[1:]
        
        # Attention mask (1 for real tokens, 0 for padding)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }
    
    def get_dataloader(
        self,
        batch_size: int = 4,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> DataLoader:
        """Create a DataLoader for this dataset."""
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    
    def stats(self) -> Dict[str, Any]:
        """Return dataset statistics."""
        return {
            "total_tokens": len(self.all_tokens),
            "num_chunks": len(self.chunks),
            "block_size": self.block_size,
            "stride": self.stride,
            "vocab_size": self.tokenizer.vocab_size,
        }
