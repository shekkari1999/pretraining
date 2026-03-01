"""
Training script for GPT-2 on wikitext-103.
Phase 1: Single GPU, basic training loop, W&B logging.
"""

import os
import time
import math
import numpy as np
import torch
import wandb

from model import GPT, GPTConfig
from config import TRAIN_BIN, VAL_BIN, DEVICE

torch.backends.cudnn.benchmark = True

# -----------------------------------------------------------------------------
# Training hyperparameters
# -----------------------------------------------------------------------------
# model
block_size = 1024       # sequence length
vocab_size = 50304      # GPT-2 vocab padded to nearest multiple of 64
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False

# optimizer
learning_rate = 6e-4
weight_decay = 0.1
betas = (0.9, 0.95)
max_steps = 5000
warmup_steps = 100

# training
batch_size = 8          # micro batch size
eval_interval = 250     # evaluate every N steps
eval_steps = 20         # how many batches to average for val loss
log_interval = 10       # print every N steps

# system
device = DEVICE
dtype = 'float16'       # float16 for T4, bfloat16 for A100+
compile_model = True    # torch.compile — fuses ops, reduces kernel launches

# wandb
wandb_log = True
wandb_project = 'pretraining'
wandb_run_name = f'gpt2-124m-wikitext103-amp-fp16-compile-cudnn'

# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
def get_batch(split):
    """Sample a random batch of (x, y) pairs from train or val data."""
    data = np.memmap(str(TRAIN_BIN) if split == 'train' else str(VAL_BIN),
                     dtype=np.uint16, mode='r')
    # pick random starting indices
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # grab block_size tokens for input, shifted by 1 for target
    x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# -----------------------------------------------------------------------------
# Validation loss
# -----------------------------------------------------------------------------
@torch.no_grad()
def estimate_loss():
    """Average loss over eval_steps batches for both splits."""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_steps)
        for k in range(eval_steps):
            x, y = get_batch(split)
            with torch.amp.autocast(device_type='cuda', dtype=autocast_dtype, enabled=use_amp):
                logits, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

# -----------------------------------------------------------------------------
# Learning rate schedule: cosine decay with warmup
# -----------------------------------------------------------------------------
def get_lr(step):
    """Cosine decay with linear warmup."""
    # linear warmup
    if step < warmup_steps:
        return learning_rate * (step + 1) / warmup_steps
    # cosine decay down to 10% of max lr
    min_lr = learning_rate * 0.1
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    decay_ratio = min(decay_ratio, 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# -----------------------------------------------------------------------------
# Init
# -----------------------------------------------------------------------------
print(f"Device: {device}")

# model
config = GPTConfig(
    block_size=block_size,
    vocab_size=vocab_size,
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    dropout=dropout,
    bias=bias,
)

model = GPT(config)
model.to(device)

if compile_model:
    print("Compiling model with torch.compile()...")
    model = torch.compile(model)

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, betas, device)

# AMP scaler for mixed precision
use_amp = (dtype == 'float16' and device == 'cuda')
scaler = torch.amp.GradScaler(enabled=use_amp)
autocast_dtype = torch.float16 if dtype == 'float16' else torch.float32

# wandb init
if wandb_log:
    wandb.init(
        project=wandb_project,
        name=wandb_run_name,
        config={
            'block_size': block_size,
            'vocab_size': vocab_size,
            'n_layer': n_layer,
            'n_head': n_head,
            'n_embd': n_embd,
            'dropout': dropout,
            'bias': bias,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'batch_size': batch_size,
            'max_steps': max_steps,
            'warmup_steps': warmup_steps,
            'device': device,
            'dtype': dtype,
            'dataset': 'wikitext-103',
            'num_params': model.get_num_params(),
        },
        tags=['phase1', 'baseline', 'wikitext-103'],
    )

# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------
print(f"Starting training for {max_steps} steps...")
print(f"Batch size: {batch_size}, Block size: {block_size}")
print(f"Tokens per step: {batch_size * block_size:,}")

# reset memory stats for clean tracking
if device == 'cuda':
    torch.cuda.reset_peak_memory_stats()

best_val_loss = float('inf')

for step in range(max_steps):
    t0 = time.time()

    # evaluate periodically
    if step % eval_interval == 0 or step == max_steps - 1:
        losses = estimate_loss()
        print(f"step {step:5d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                'val/loss': losses['val'],
                'train/loss_eval': losses['train'],
                'step': step,
            })
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']

    # set learning rate for this step
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # forward pass
    x, y = get_batch('train')
    with torch.amp.autocast(device_type='cuda', dtype=autocast_dtype, enabled=use_amp):
        logits, loss = model(x, y)

    # backward pass
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    # gradient clipping (must unscale first)
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()

    # timing
    t1 = time.time()
    dt = t1 - t0
    tokens_per_sec = (batch_size * block_size) / dt

    # memory tracking
    if device == 'cuda':
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        reserved_mem_mb = torch.cuda.max_memory_reserved() / (1024 * 1024)

    # log
    if step % log_interval == 0:
        mem_str = f" | mem {peak_mem_mb:.0f}MB" if device == 'cuda' else ""
        print(f"step {step:5d} | loss {loss.item():.4f} | lr {lr:.6f} | "
              f"{dt*1000:.0f}ms | {tokens_per_sec:,.0f} tok/s{mem_str}")
        log_dict = {
            'train/loss': loss.item(),
            'train/lr': lr,
            'train/step_time_ms': dt * 1000,
            'train/tokens_per_sec': tokens_per_sec,
            'step': step,
        }
        if device == 'cuda':
            log_dict['memory/peak_allocated_mb'] = peak_mem_mb
            log_dict['memory/peak_reserved_mb'] = reserved_mem_mb
        if wandb_log:
            wandb.log(log_dict)

print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")

if wandb_log:
    wandb.finish()
