"""
Profile a few training steps with PyTorch Profiler.
Outputs a Chrome trace (viewable in chrome://tracing)
and prints a summary table of top CUDA kernels.
"""

import numpy as np
import torch
from torch.profiler import profile, record_function, ProfilerActivity, schedule

from model import GPT, GPTConfig
from config import TRAIN_BIN, DEVICE

torch.backends.cudnn.benchmark = True

# -----------------------------------------------------------------------------
# Config (same as train.py)
# -----------------------------------------------------------------------------
block_size = 1024
vocab_size = 50304
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False
batch_size = 8
learning_rate = 6e-4
weight_decay = 0.1
betas = (0.9, 0.95)
device = DEVICE
dtype = 'float16'
compile_model = True

# AMP setup
use_amp = (dtype == 'float16' and device == 'cuda')
scaler = torch.amp.GradScaler(enabled=use_amp)
autocast_dtype = torch.float16 if dtype == 'float16' else torch.float32

# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
def get_batch():
    data = np.memmap(str(TRAIN_BIN), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
    x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x, y

# -----------------------------------------------------------------------------
# Model setup
# -----------------------------------------------------------------------------
print(f"Device: {device}")
config = GPTConfig(
    block_size=block_size, vocab_size=vocab_size,
    n_layer=n_layer, n_head=n_head, n_embd=n_embd,
    dropout=dropout, bias=bias,
)
model = GPT(config)
model.to(device)

if compile_model:
    print("Compiling model with torch.compile()...")
    model = torch.compile(model)

optimizer = model.configure_optimizers(weight_decay, learning_rate, betas, device)

# -----------------------------------------------------------------------------
# Warmup: run a few steps so torch.compile finishes tracing
# -----------------------------------------------------------------------------
print("Warming up (torch.compile tracing)...")
for _ in range(5):
    x, y = get_batch()
    with torch.amp.autocast(device_type='cuda', dtype=autocast_dtype, enabled=use_amp):
        logits, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
torch.cuda.synchronize()
print("Warmup done.\n")

# -----------------------------------------------------------------------------
# Profile: 3 wait + 3 warmup + 5 active + 1 repeat
# -----------------------------------------------------------------------------
# schedule: skip first 3 steps, warmup profiler for 3, actively trace 5
prof_schedule = schedule(wait=3, warmup=3, active=5, repeat=1)

PROFILE_STEPS = 12  # wait(3) + warmup(3) + active(5) + buffer(1)

print(f"Profiling {PROFILE_STEPS} steps...")

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=prof_schedule,
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./profiler_logs'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for step in range(PROFILE_STEPS):
        # annotate each phase so it shows up in the trace
        with record_function("data_loading"):
            x, y = get_batch()

        with record_function("forward"):
            with torch.amp.autocast(device_type='cuda', dtype=autocast_dtype, enabled=use_amp):
                logits, loss = model(x, y)

        with record_function("backward"):
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()

        with record_function("optimizer"):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

        prof.step()

# -----------------------------------------------------------------------------
# Print summary tables
# -----------------------------------------------------------------------------
print("\n" + "="*80)
print("TOP 20 CUDA KERNELS (by total CUDA time)")
print("="*80)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

print("\n" + "="*80)
print("TOP 10 BY CPU TIME")
print("="*80)
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

print("\n" + "="*80)
print("GROUPED BY INPUT SHAPE (top 15)")
print("="*80)
print(prof.key_averages(group_by_input_shape=True).table(sort_by="cuda_time_total", row_limit=15))

print("\n" + "="*80)
print("MEMORY SUMMARY")
print("="*80)
print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))

# also export chrome trace for manual inspection
prof.export_chrome_trace("trace.json")
print("\nChrome trace saved to trace.json")
print("View it at: chrome://tracing (drag and drop the file)")
print("\nTensorBoard logs saved to ./profiler_logs/")
print("View with: tensorboard --logdir=./profiler_logs")
