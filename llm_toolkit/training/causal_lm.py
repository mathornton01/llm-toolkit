"""
Causal Language Model Pre-Training.

Standard next-token prediction training (GPT-style).
This is the most fundamental LLM training method — the model learns to predict
the next token given all previous tokens.

How it works:
    For each sequence [t1, t2, t3, t4, t5]:
        Input:  [t1, t2, t3, t4]
        Target: [t2, t3, t4, t5]
        Loss = CrossEntropy(model(input), target)
    
    The model uses causal (left-to-right) attention masking so each position
    can only attend to previous positions. This trains the model to generate
    coherent text autoregressively.

Training loop:
    1. Tokenize text corpus into fixed-length chunks
    2. For each batch:
        a. Forward pass: model predicts next token at each position
        b. Compute cross-entropy loss
        c. Backward pass: compute gradients
        d. Optimizer step (with optional gradient accumulation)
    3. Repeat for N epochs
    4. Save checkpoint

Key hyperparameters:
    - learning_rate: Typically 1e-4 to 3e-4 for pre-training
    - batch_size: Larger is better (gradient noise reduction)
    - block_size: Sequence length (context window)
    - warmup_steps: Linear warmup for learning rate
    - weight_decay: L2 regularization (typically 0.01-0.1)
    - gradient_accumulation: Simulate larger batches on small GPUs

Reference:
    Radford et al., "Language Models are Unsupervised Multitask Learners" (2019)
    Brown et al., "Language Models are Few-Shot Learners" (2020)
"""
import time
import math
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from typing import Optional, Dict, Any, List, Callable
from pathlib import Path

from ..core.registry import Registry
from ..core.base import ModuleResult
from ..data.causal_lm_dataset import CausalLMDataset


def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps: int, num_training_steps: int
):
    """Cosine annealing with linear warmup (standard for LLM training).
    
    Learning rate schedule:
        - Warmup: linear increase from 0 to lr over warmup_steps
        - Cosine decay: from lr to 0 over remaining steps
    
    This is the schedule used by GPT-3, LLaMA, and most modern LLMs.
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return LambdaLR(optimizer, lr_lambda)


@Registry.register("training", "causal_lm")
class CausalLMTrainer:
    """Causal Language Model pre-training.
    
    Standard next-token prediction with configurable:
    - Optimizer (AdamW default)
    - Learning rate schedule (cosine with warmup default)
    - Gradient accumulation
    - Checkpointing
    - Logging
    
    Example:
        trainer = CausalLMTrainer(
            learning_rate=3e-4,
            num_epochs=3,
            batch_size=4,
            block_size=128,
        )
        result = trainer.train(model, tokenizer, texts)
    """
    
    name = "causal_lm"
    
    def __init__(
        self,
        # Core training params
        learning_rate: float = 3e-4,
        num_epochs: int = 3,
        batch_size: int = 4,
        block_size: int = 128,
        
        # Optimizer
        weight_decay: float = 0.01,
        betas: tuple = (0.9, 0.95),
        eps: float = 1e-8,
        max_grad_norm: float = 1.0,
        
        # Schedule
        warmup_steps: int = 0,
        warmup_ratio: float = 0.0,  # Alternative: fraction of total steps
        lr_scheduler: str = "cosine",  # "cosine", "linear", "constant"
        min_lr_ratio: float = 0.1,  # Minimum LR as fraction of max LR
        
        # Gradient accumulation
        gradient_accumulation_steps: int = 1,
        
        # GPU performance
        use_amp: bool = True,       # Automatic Mixed Precision (fp16/bf16)
        amp_dtype: str = "bfloat16", # "float16" or "bfloat16" (bf16 preferred on A100)
        compile_model: bool = False, # torch.compile for extra speed (requires PyTorch 2.0+)
        
        # Checkpointing
        save_every_steps: int = 0,  # 0 = only save at end
        save_dir: Optional[str] = None,
        
        # Logging
        log_every_steps: int = 10,
        
        # DataLoader
        num_workers: int = 0,       # DataLoader workers (set >0 for large datasets)
        
        # Callbacks
        on_step_end: Optional[Callable] = None,
        on_epoch_end: Optional[Callable] = None,
        
        **kwargs,
    ):
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.block_size = block_size
        
        self.weight_decay = weight_decay
        self.betas = betas
        self.eps = eps
        self.max_grad_norm = max_grad_norm
        
        self.warmup_steps = warmup_steps
        self.warmup_ratio = warmup_ratio
        self.lr_scheduler_type = lr_scheduler
        self.min_lr_ratio = min_lr_ratio
        
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        self.use_amp = use_amp
        self.amp_dtype = amp_dtype
        self.compile_model = compile_model
        
        self.save_every_steps = save_every_steps
        self.save_dir = Path(save_dir) if save_dir else None
        
        self.log_every_steps = log_every_steps
        self.num_workers = num_workers
        
        self.on_step_end = on_step_end
        self.on_epoch_end = on_epoch_end
    
    def _log(self, msg: str):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
    
    def _create_optimizer(self, model) -> AdamW:
        """Create AdamW optimizer with weight decay only on non-bias/norm params.
        
        Standard practice: don't apply weight decay to biases or LayerNorm params.
        """
        decay_params = []
        no_decay_params = []
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "norm" in name or "layernorm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        param_groups = [
            {"params": decay_params, "weight_decay": self.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        
        return AdamW(
            param_groups,
            lr=self.learning_rate,
            betas=self.betas,
            eps=self.eps,
        )
    
    def _create_scheduler(self, optimizer, num_training_steps):
        """Create learning rate scheduler."""
        warmup = self.warmup_steps
        if warmup == 0 and self.warmup_ratio > 0:
            warmup = int(num_training_steps * self.warmup_ratio)
        
        if self.lr_scheduler_type == "cosine":
            return get_cosine_schedule_with_warmup(
                optimizer, warmup, num_training_steps
            )
        elif self.lr_scheduler_type == "linear":
            def lr_lambda(step):
                if step < warmup:
                    return float(step) / float(max(1, warmup))
                return max(0.0, 1.0 - float(step - warmup) / float(
                    max(1, num_training_steps - warmup)
                ))
            return LambdaLR(optimizer, lr_lambda)
        elif self.lr_scheduler_type == "constant":
            return LambdaLR(optimizer, lambda _: 1.0)
        else:
            raise ValueError(f"Unknown scheduler: {self.lr_scheduler_type}")
    
    def train(
        self,
        model: nn.Module,
        tokenizer,
        texts: List[str],
        val_texts: Optional[List[str]] = None,
        **kwargs,
    ) -> ModuleResult:
        """Train the model on text data.
        
        Args:
            model: The language model to train
            tokenizer: Tokenizer for the model
            texts: Training text data
            val_texts: Optional validation texts
        
        Returns:
            ModuleResult with training metrics and loss history
        """
        device = next(model.parameters()).device
        is_cuda = device.type == "cuda"
        
        # Setup AMP (Automatic Mixed Precision)
        # AMP gives ~2x throughput on A100 with negligible quality loss
        use_amp = self.use_amp and is_cuda
        if use_amp:
            amp_dtype = torch.bfloat16 if self.amp_dtype == "bfloat16" else torch.float16
            scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))
            # bf16 doesn't need scaler (no inf/nan risk), but fp16 does
            self._log(f"AMP enabled: {self.amp_dtype}")
        else:
            amp_dtype = torch.float32
            # On CPU, use a no-op scaler (scale/unscale/step/update all pass through)
            if is_cuda:
                scaler = torch.amp.GradScaler("cuda", enabled=False)
                self._log("AMP disabled (training in fp32)")
            else:
                scaler = None  # No scaler on CPU
        
        # Optional torch.compile (PyTorch 2.0+)
        if self.compile_model and hasattr(torch, "compile"):
            self._log("Compiling model with torch.compile...")
            model = torch.compile(model)
            self._log("Compilation complete")
        
        # Create dataset and dataloader
        dataset = CausalLMDataset(
            texts, tokenizer,
            block_size=self.block_size,
        )
        dataloader = dataset.get_dataloader(
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=is_cuda,  # Faster CPU→GPU transfer
        )
        
        stats = dataset.stats()
        self._log(f"Device: {device}")
        self._log(f"Dataset: {stats['total_tokens']:,} tokens, {stats['num_chunks']} chunks")
        self._log(f"Block size: {self.block_size}, Batch size: {self.batch_size}")
        
        if len(dataset) == 0:
            return ModuleResult(
                success=False,
                logs=["No training data (text too short for block_size)"],
            )
        
        # Create validation dataset if provided
        val_dataloader = None
        if val_texts:
            val_dataset = CausalLMDataset(val_texts, tokenizer, block_size=self.block_size)
            if len(val_dataset) > 0:
                val_dataloader = val_dataset.get_dataloader(
                    batch_size=self.batch_size, shuffle=False
                )
        
        # Setup optimizer and scheduler
        optimizer = self._create_optimizer(model)
        steps_per_epoch = math.ceil(len(dataloader) / self.gradient_accumulation_steps)
        total_steps = steps_per_epoch * self.num_epochs
        scheduler = self._create_scheduler(optimizer, total_steps)
        
        self._log(f"Training: {self.num_epochs} epochs, {steps_per_epoch} steps/epoch, {total_steps} total steps")
        self._log(f"LR: {self.learning_rate}, Warmup: {self.warmup_steps or int(total_steps * self.warmup_ratio)}")
        self._log(f"Gradient accumulation: {self.gradient_accumulation_steps}")
        
        # Training loop
        model.train()
        history = {
            "train_loss": [],
            "val_loss": [],
            "learning_rates": [],
            "step_times": [],
        }
        
        global_step = 0
        best_val_loss = float("inf")
        
        for epoch in range(self.num_epochs):
            epoch_loss = 0
            epoch_tokens = 0
            optimizer.zero_grad()
            
            for batch_idx, batch in enumerate(dataloader):
                # Move batch to device (non_blocking for async GPU transfer with pin_memory)
                input_ids = batch["input_ids"].to(device, non_blocking=is_cuda)
                labels = batch["labels"].to(device, non_blocking=is_cuda)
                attention_mask = batch["attention_mask"].to(device, non_blocking=is_cuda)
                
                step_start = time.time()
                
                # Forward pass with AMP autocast (only on CUDA)
                if use_amp:
                    with torch.amp.autocast("cuda", dtype=amp_dtype):
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                        )
                        loss = outputs.loss / self.gradient_accumulation_steps
                else:
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs.loss / self.gradient_accumulation_steps
                
                # Backward pass
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                epoch_loss += outputs.loss.item() * input_ids.shape[0]
                epoch_tokens += attention_mask.sum().item()
                
                # Optimizer step (with gradient accumulation)
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    if scaler is not None:
                        # Gradient clipping (unscale first for fp16)
                        if self.max_grad_norm > 0:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), self.max_grad_norm
                            )
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        # CPU path: direct gradient clipping and step
                        if self.max_grad_norm > 0:
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), self.max_grad_norm
                            )
                        optimizer.step()
                    
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)  # More memory efficient
                    global_step += 1
                    
                    step_time = time.time() - step_start
                    current_lr = scheduler.get_last_lr()[0]
                    current_loss = outputs.loss.item()
                    
                    history["train_loss"].append(current_loss)
                    history["learning_rates"].append(current_lr)
                    history["step_times"].append(step_time)
                    
                    # Logging
                    if self.log_every_steps > 0 and global_step % self.log_every_steps == 0:
                        avg_time = sum(history["step_times"][-10:]) / min(10, len(history["step_times"]))
                        self._log(
                            f"  epoch {epoch+1}/{self.num_epochs} "
                            f"step {global_step}/{total_steps} "
                            f"loss={current_loss:.4f} "
                            f"lr={current_lr:.2e} "
                            f"({avg_time:.2f}s/step)"
                        )
                    
                    # Checkpointing
                    if self.save_every_steps > 0 and global_step % self.save_every_steps == 0:
                        if self.save_dir:
                            ckpt_path = self.save_dir / f"checkpoint-{global_step}"
                            ckpt_path.mkdir(parents=True, exist_ok=True)
                            model.save_pretrained(ckpt_path)
                            self._log(f"  Saved checkpoint: {ckpt_path}")
                    
                    # Step callback
                    if self.on_step_end:
                        self.on_step_end(global_step, current_loss, current_lr)
            
            # End of epoch
            avg_train_loss = epoch_loss / max(1, len(dataloader))
            self._log(f"  Epoch {epoch+1} avg loss: {avg_train_loss:.4f}")
            
            # Validation
            if val_dataloader is not None:
                val_loss = self._validate(model, val_dataloader, device, amp_dtype, use_amp)
                history["val_loss"].append(val_loss)
                self._log(f"  Epoch {epoch+1} val loss: {val_loss:.4f}")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    if self.save_dir:
                        best_path = self.save_dir / "best"
                        best_path.mkdir(parents=True, exist_ok=True)
                        model.save_pretrained(best_path)
                        self._log(f"  New best model saved: {best_path}")
            
            # Epoch callback
            if self.on_epoch_end:
                self.on_epoch_end(epoch, avg_train_loss)
        
        # Final save
        if self.save_dir:
            final_path = self.save_dir / "final"
            final_path.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(final_path)
            tokenizer.save_pretrained(final_path)
            self._log(f"Final model saved: {final_path}")
        
        model.eval()
        
        # Compute final metrics
        final_loss = history["train_loss"][-1] if history["train_loss"] else 0
        final_ppl = math.exp(min(final_loss, 100))  # Cap to avoid overflow
        
        return ModuleResult(
            success=True,
            metrics={
                "final_loss": final_loss,
                "final_perplexity": final_ppl,
                "total_steps": global_step,
                "total_tokens_seen": sum(stats["total_tokens"] for _ in range(self.num_epochs)),
                "best_val_loss": best_val_loss if val_dataloader else -1,
            },
            artifacts={
                "history": history,
                "config": {
                    "learning_rate": self.learning_rate,
                    "num_epochs": self.num_epochs,
                    "batch_size": self.batch_size,
                    "block_size": self.block_size,
                    "weight_decay": self.weight_decay,
                    "warmup_steps": self.warmup_steps,
                    "lr_scheduler": self.lr_scheduler_type,
                    "gradient_accumulation_steps": self.gradient_accumulation_steps,
                },
            },
        )
    
    def _validate(self, model, dataloader, device, amp_dtype=torch.float32, use_amp=False) -> float:
        """Run validation and return average loss."""
        model.eval()
        total_loss = 0
        total_batches = 0
        is_cuda = device.type == "cuda"
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(device, non_blocking=is_cuda)
                labels = batch["labels"].to(device, non_blocking=is_cuda)
                attention_mask = batch["attention_mask"].to(device, non_blocking=is_cuda)
                
                if use_amp:
                    with torch.amp.autocast("cuda", dtype=amp_dtype):
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                        )
                else:
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                total_loss += outputs.loss.item()
                total_batches += 1
        
        model.train()
        return total_loss / max(1, total_batches)
