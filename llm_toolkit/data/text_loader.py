"""
Text loading utilities.

Loads raw text from various sources for training:
- Plain text files (.txt)
- JSON/JSONL files (extract specified field)
- Directories of text files
- HuggingFace datasets

This is the data *ingestion* step — feeds into CausalLMDataset for tokenization.
"""
from pathlib import Path
from typing import List, Optional, Union
import json


class TextLoader:
    """Load text data from various sources.
    
    Example:
        # From files
        texts = TextLoader.from_file("corpus.txt")
        texts = TextLoader.from_jsonl("data.jsonl", field="text")
        texts = TextLoader.from_directory("data/", pattern="*.txt")
        
        # From HuggingFace
        texts = TextLoader.from_huggingface("wikitext", "wikitext-2-raw-v1", split="train")
        
        # Then create dataset
        dataset = CausalLMDataset(texts, tokenizer)
    """
    
    @staticmethod
    def from_file(path: str, encoding: str = "utf-8") -> List[str]:
        """Load text from a single file. Returns list with one string."""
        with open(path, "r", encoding=encoding) as f:
            return [f.read()]
    
    @staticmethod
    def from_lines(path: str, encoding: str = "utf-8") -> List[str]:
        """Load text file, one document per line."""
        with open(path, "r", encoding=encoding) as f:
            return [line.strip() for line in f if line.strip()]
    
    @staticmethod
    def from_jsonl(
        path: str,
        field: str = "text",
        encoding: str = "utf-8",
    ) -> List[str]:
        """Load text from JSONL file, extracting specified field."""
        texts = []
        with open(path, "r", encoding=encoding) as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    if field in obj:
                        texts.append(str(obj[field]))
        return texts
    
    @staticmethod
    def from_json(
        path: str,
        field: str = "text",
    ) -> List[str]:
        """Load text from JSON file (expects list of objects)."""
        with open(path, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [str(item[field]) if isinstance(item, dict) else str(item) for item in data]
        elif isinstance(data, dict) and field in data:
            items = data[field]
            return [str(item) for item in items] if isinstance(items, list) else [str(items)]
        return []
    
    @staticmethod
    def from_directory(
        path: str,
        pattern: str = "*.txt",
        encoding: str = "utf-8",
        recursive: bool = False,
    ) -> List[str]:
        """Load text from all matching files in a directory."""
        p = Path(path)
        glob_fn = p.rglob if recursive else p.glob
        texts = []
        for f in sorted(glob_fn(pattern)):
            with open(f, "r", encoding=encoding) as fh:
                texts.append(fh.read())
        return texts
    
    @staticmethod
    def from_huggingface(
        dataset_name: str,
        config: Optional[str] = None,
        split: str = "train",
        field: str = "text",
        max_samples: Optional[int] = None,
    ) -> List[str]:
        """Load text from a HuggingFace dataset.
        
        Requires: pip install datasets
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("HuggingFace datasets required: pip install datasets")
        
        ds = load_dataset(dataset_name, config, split=split)
        
        if max_samples:
            ds = ds.select(range(min(max_samples, len(ds))))
        
        return [str(row[field]) for row in ds if row.get(field)]
    
    @staticmethod
    def from_strings(texts: List[str]) -> List[str]:
        """Pass-through for programmatic usage."""
        return texts
