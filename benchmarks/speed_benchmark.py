"""
benchmarks/speed_benchmark.py — Inference speed benchmarking.

Measures:
  • Prefill latency    (time to process the prompt)
  • Decode latency     (time per generated token)
  • Throughput         (tokens/second)
  • Time-to-first-token (TTFT)

Methodology
───────────
  1. Warm-up N runs to fill caches and let PyTorch JIT settle.
  2. Time M runs, record mean ± std.
  3. Sweep over (batch_size, seq_length) grid.
  4. Optionally profile GPU memory during inference.
"""

from __future__ import annotations

import time
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SpeedResult:
    """Single benchmark configuration result."""
    label: str
    batch_size: int
    seq_len: int
    gen_tokens: int

    # Prefill (forward pass through prompt)
    prefill_mean_ms: float = 0.0
    prefill_std_ms:  float = 0.0

    # Per-token generation latency
    decode_mean_ms:  float = 0.0
    decode_std_ms:   float = 0.0

    # Throughput: tokens generated per second
    throughput_tok_s: float = 0.0

    # Time to first token
    ttft_ms: float = 0.0

    # Peak GPU memory during inference (MB)
    peak_mem_mb: float = 0.0

    def to_dict(self) -> Dict:
        return vars(self)

    def __str__(self) -> str:
        return (
            f"[{self.label}] bs={self.batch_size} seq={self.seq_len}  "
            f"prefill={self.prefill_mean_ms:.1f}±{self.prefill_std_ms:.1f}ms  "
            f"decode={self.decode_mean_ms:.2f}±{self.decode_std_ms:.2f}ms/tok  "
            f"throughput={self.throughput_tok_s:.1f} tok/s  "
            f"mem={self.peak_mem_mb:.0f}MB"
        )


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

class CUDATimer:
    """High-precision GPU timer using CUDA events."""

    def __init__(self, device: torch.device):
        self.device = device
        self.use_cuda = device.type == "cuda"

    def __enter__(self):
        if self.use_cuda:
            self._start = torch.cuda.Event(enable_timing=True)
            self._end   = torch.cuda.Event(enable_timing=True)
            self._start.record()
        else:
            self._t0 = time.perf_counter()
        return self

    def __exit__(self, *_):
        if self.use_cuda:
            self._end.record()
            torch.cuda.synchronize()
            self.elapsed_ms = self._start.elapsed_time(self._end)
        else:
            self.elapsed_ms = (time.perf_counter() - self._t0) * 1000.0


def _peak_memory_mb(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    return 0.0


def _reset_peak_memory(device: torch.device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


# ---------------------------------------------------------------------------
# Prefill benchmark (single forward pass, no generation)
# ---------------------------------------------------------------------------

@torch.no_grad()
def benchmark_prefill(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    warmup: int = 3,
    timed: int = 10,
) -> Tuple[float, float]:
    """
    Measure forward-pass latency (ms) over a fixed prompt.

    Returns:
        (mean_ms, std_ms)
    """
    device = next(model.parameters()).device
    timer  = CUDATimer(device)
    times  = []

    for i in range(warmup + timed):
        with timer:
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
        if i >= warmup:
            times.append(timer.elapsed_ms)

    return statistics.mean(times), statistics.stdev(times) if len(times) > 1 else 0.0


# ---------------------------------------------------------------------------
# Token generation benchmark
# ---------------------------------------------------------------------------

@torch.no_grad()
def benchmark_generation(
    model: nn.Module,
    tokenizer,
    prompt_ids: torch.Tensor,
    prompt_mask: torch.Tensor,
    gen_tokens: int = 128,
    warmup: int = 2,
    timed: int = 5,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Measure end-to-end generation speed.

    Captures:
        - Time to first token (TTFT = prefill + first decode step)
        - Per-token decode latency (mean over generated tokens)
        - Total throughput (tokens/second)
        - Peak GPU memory

    Returns a dict with timing stats.
    """
    if device is None:
        device = next(model.parameters()).device

    timer = CUDATimer(device)
    results = []

    for run in range(warmup + timed):
        _reset_peak_memory(device)
        token_times = []

        # Track token-by-token times using a hook approach:
        # We run manual generation loop for precise per-token timing.
        cur_ids  = prompt_ids.clone()
        cur_mask = prompt_mask.clone()

        # TTFT: time for prefill + first token
        t_start = time.perf_counter()
        with timer:
            out = model(input_ids=cur_ids, attention_mask=cur_mask)
        ttft_ms = timer.elapsed_ms

        next_token = out.logits[:, -1, :].argmax(-1, keepdim=True)
        cur_ids  = torch.cat([cur_ids, next_token], dim=1)
        cur_mask = torch.cat([cur_mask, torch.ones(cur_mask.shape[0], 1,
                                                    device=device, dtype=cur_mask.dtype)], dim=1)

        # Decode remaining tokens
        for _ in range(gen_tokens - 1):
            with timer:
                out = model(input_ids=cur_ids, attention_mask=cur_mask)
            token_times.append(timer.elapsed_ms)
            next_token = out.logits[:, -1, :].argmax(-1, keepdim=True)
            cur_ids  = torch.cat([cur_ids, next_token], dim=1)
            cur_mask = torch.cat([cur_mask, torch.ones(
                cur_mask.shape[0], 1, device=device, dtype=cur_mask.dtype)], dim=1)

        total_ms = (time.perf_counter() - t_start) * 1000.0
        mem_mb   = _peak_memory_mb(device)

        if run >= warmup:
            bs = prompt_ids.shape[0]
            total_tokens = gen_tokens * bs
            results.append({
                "ttft_ms":        ttft_ms,
                "decode_mean_ms": statistics.mean(token_times) if token_times else 0.0,
                "decode_std_ms":  statistics.stdev(token_times) if len(token_times) > 1 else 0.0,
                "throughput_tok_s": total_tokens / (total_ms / 1000.0),
                "peak_mem_mb":    mem_mb,
            })

    # Aggregate over timed runs
    def avg(key): return statistics.mean(r[key] for r in results)
    def std(key): return statistics.stdev(r[key] for r in results) if len(results) > 1 else 0.0

    return {
        "ttft_ms":          avg("ttft_ms"),
        "decode_mean_ms":   avg("decode_mean_ms"),
        "decode_std_ms":    avg("decode_std_ms"),
        "throughput_tok_s": avg("throughput_tok_s"),
        "peak_mem_mb":      avg("peak_mem_mb"),
    }


# ---------------------------------------------------------------------------
# Full speed sweep
# ---------------------------------------------------------------------------

def run_speed_benchmark(
    model: nn.Module,
    tokenizer,
    label: str,
    batch_sizes: List[int] = (1, 4),
    seq_lengths: List[int] = (128, 512),
    gen_tokens: int = 64,
    warmup: int = 2,
    timed: int = 5,
    device: Optional[torch.device] = None,
) -> List[SpeedResult]:
    """
    Sweep over batch_sizes × seq_lengths and measure generation speed.

    Returns list of SpeedResult, one per (batch_size, seq_length) pair.
    """
    if device is None:
        device = next(model.parameters()).device

    all_results: List[SpeedResult] = []

    for bs in batch_sizes:
        for seq in seq_lengths:
            print(f"  Speed bench: {label}  bs={bs}  seq={seq}  gen={gen_tokens}")

            # Build a synthetic prompt of the right length
            prompt_ids = torch.randint(
                3, 1000, (bs, seq), device=device, dtype=torch.long
            )
            prompt_mask = torch.ones_like(prompt_ids)

            # Prefill-only timing
            pf_mean, pf_std = benchmark_prefill(
                model, prompt_ids, prompt_mask,
                warmup=warmup, timed=timed,
            )

            # Full generation timing
            gen_stats = benchmark_generation(
                model, tokenizer,
                prompt_ids, prompt_mask,
                gen_tokens=gen_tokens,
                warmup=warmup,
                timed=timed,
                device=device,
            )

            result = SpeedResult(
                label=label,
                batch_size=bs,
                seq_len=seq,
                gen_tokens=gen_tokens,
                prefill_mean_ms=pf_mean,
                prefill_std_ms=pf_std,
                **gen_stats,
            )
            print(f"    {result}")
            all_results.append(result)

    return all_results
