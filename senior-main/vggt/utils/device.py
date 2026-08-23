# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Device abstraction layer for CUDA / MPS / CPU.

Design goals
────────────
 • CUDA path is **completely unchanged** — same dtype, same autocast, same code.
 • MPS path keeps model weights in float32 and uses float16 autocast for the
   aggregator (the heavy backbone) but runs heads in float32 for full quality.
 • Memory savings on MPS come from *structural* changes (pruning unused
   intermediate outputs, sequential head execution, CPU offload) — **not**
   from reducing precision of the final predictions.
"""

import gc
import contextlib
import torch


# ───────────────────────── device selection ──────────────────────────

def get_device() -> str:
    """Return the best available device string: cuda ▸ mps ▸ cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def is_mps(device) -> bool:
    return str(device).split(":")[0] == "mps"


def is_cuda(device) -> bool:
    return str(device).split(":")[0] == "cuda"


# ──────────────────────── dtype selection ────────────────────────────

def get_autocast_dtype(device=None) -> torch.dtype:
    """Mixed-precision dtype for the *autocast* context (aggregator backbone).

    CUDA: bfloat16 if capability ≥ 8, else float16.
    MPS:  float16 (only supported option).
    CPU:  float32 (autocast not really useful).
    """
    if device is None:
        device = get_device()
    d = str(device).split(":")[0]
    if d == "cuda":
        return torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    if d == "mps":
        return torch.float16
    return torch.float32


# ──────────────────────── autocast helpers ───────────────────────────

def autocast_on(device=None, dtype=None):
    """Autocast context — used around the aggregator backbone."""
    if device is None:
        device = get_device()
    d = str(device).split(":")[0]
    if dtype is None:
        dtype = get_autocast_dtype(d)
    if d == "cpu":
        return contextlib.nullcontext()          # no-op on CPU
    return torch.amp.autocast(device_type=d, dtype=dtype)


def autocast_off(device=None):
    """Disable autocast — used around the heads for full float32 precision.

    On MPS this truly disables autocast (model weights are float32, so there
    is no dtype mismatch).
    """
    if device is None:
        device = get_device()
    d = str(device).split(":")[0]
    if d == "cpu":
        return contextlib.nullcontext()
    return torch.amp.autocast(device_type=d, enabled=False)


# ──────────────────── high-precision dtype ───────────────────────────

def high_precision_dtype(device=None) -> torch.dtype:
    """Highest precision dtype supported on this device.

    CUDA / CPU → float64.   MPS → float32 (MPS cannot do float64).
    """
    if device is None:
        device = get_device()
    if is_mps(device):
        return torch.float32
    return torch.float64


# ──────────────────────── memory helpers ─────────────────────────────

def empty_cache(device=None):
    if device is None:
        device = get_device()
    d = str(device).split(":")[0]
    if d == "cuda":
        torch.cuda.empty_cache()
    elif d == "mps":
        torch.mps.empty_cache()


def aggressive_cleanup(device=None):
    """gc.collect + empty_cache — use between heavy stages on MPS."""
    gc.collect()
    empty_cache(device)
