"""
benchmarks/memory_benchmark.py — Memory footprint analysis.

Measures:
  • Model weight size on disk / in memory (GB)
  • Per-layer compression ratios
  • Peak GPU VRAM during inference
  • Theoretical compression from bit-width reduction

Compression reference table (7B LLaMA 2 weights only)
──────────────────────────────────────────────────────
  fp32 :  28.0 GB
  fp16 :  14.0 GB   (1.0×  baseline)
  W8   :   7.0 GB   (2.0×)
  W4   :   3.5 GB   (4.0×)
  W2   :   1.75 GB  (8.0×)
  W1   :   0.88 GB  (16.0×)  ← + ~0.5 GB metadata ≈ 1.4 GB total
"""

from __future__ import annotations

import os
import gc
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from quantization.hqq_linear import HQQLinear
from quantization.hqq_core import estimate_model_size_gb, quantisation_ratio


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class MemoryResult:
    label: str

    # Weight memory (GB)
    weight_gb: float
    # Full model incl. activations peak (GB) — GPU VRAM
    peak_vram_gb: float

    # Compression relative to fp16 baseline
    compression_vs_fp16: float

    # Theoretical ideal compression (ignoring metadata)
    theoretical_compression: float

    # Metadata overhead (scale + zero tensors)
    meta_gb: float

    # Per-layer breakdown
    layer_compressions: List[float]

    nbits: int

    def to_dict(self) -> Dict:
        d = vars(self).copy()
        d.pop("layer_compressions")
        return d

    def __str__(self) -> str:
        return (
            f"[{self.label}]  "
            f"weights={self.weight_gb:.3f}GB  "
            f"peak_vram={self.peak_vram_gb:.3f}GB  "
            f"compression={self.compression_vs_fp16:.2f}×  "
            f"(theoretical={self.theoretical_compression:.1f}×)"
        )


# ---------------------------------------------------------------------------
# Model memory analysis
# ---------------------------------------------------------------------------

def analyse_model_memory(
    model: nn.Module,
    label: str,
    nbits: int,
    device: Optional[torch.device] = None,
) -> MemoryResult:
    """
    Measure the memory footprint of a (possibly quantised) model.

    Works for both fp16 nn.Linear models and HQQ-quantised models.
    """
    # ── Weight bytes ─────────────────────────────────────────────────────
    weight_bytes     = 0
    meta_bytes       = 0
    layer_compressions = []

    for name, module in model.named_modules():
        if isinstance(module, HQQLinear):
            # Packed weights
            w_bytes = module.W_packed.numel() * module.W_packed.element_size()
            m_bytes = (
                module.scale.numel() * module.scale.element_size()
                + module.zero.numel()  * module.zero.element_size()
            )
            weight_bytes += w_bytes
            meta_bytes   += m_bytes
            layer_compressions.append(module.compression_ratio())

        elif isinstance(module, nn.Linear):
            w_bytes = module.weight.numel() * module.weight.element_size()
            weight_bytes += w_bytes
            # No compression for plain fp16
            layer_compressions.append(1.0)

    # Other parameters (embeddings, norms, etc.)
    other_bytes = sum(
        p.numel() * p.element_size()
        for n, p in model.named_parameters()
        if not any(
            isinstance(m, (nn.Linear, HQQLinear))
            for m in [model]  # simplified
        )
    )

    total_weight_gb = (weight_bytes + meta_bytes + other_bytes) / (1024 ** 3)

    # ── Peak VRAM (GPU measurement) ──────────────────────────────────────
    peak_vram_gb = 0.0
    if device is not None and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        # Run a tiny forward pass to trigger allocation
        dummy = torch.randint(1, 100, (1, 32), device=device)
        with torch.no_grad():
            try:
                _ = model(input_ids=dummy, attention_mask=torch.ones_like(dummy))
            except Exception:
                pass
        peak_vram_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

    # ── Compression vs fp16 ──────────────────────────────────────────────
    # Count all weight parameters to estimate fp16 baseline
    total_weight_params = sum(
        (m.in_features * m.out_features)
        if isinstance(m, (nn.Linear, HQQLinear)) else 0
        for m in model.modules()
    )
    fp16_weight_gb = total_weight_params * 2 / (1024 ** 3)  # 2 bytes/param
    compression    = fp16_weight_gb / max(total_weight_gb, 1e-9)
    theoretical    = quantisation_ratio(nbits)

    meta_gb = meta_bytes / (1024 ** 3)

    return MemoryResult(
        label=label,
        weight_gb=total_weight_gb,
        peak_vram_gb=peak_vram_gb,
        compression_vs_fp16=compression,
        theoretical_compression=theoretical,
        meta_gb=meta_gb,
        layer_compressions=layer_compressions,
        nbits=nbits,
    )


# ---------------------------------------------------------------------------
# Theoretical memory table (for quick reference without loading models)
# ---------------------------------------------------------------------------

def theoretical_memory_table(
    n_params_billions: float = 7.0,
    bit_widths: Tuple[int, ...] = (1, 2, 4, 8, 16),
    meta_overhead: float = 0.03,
) -> str:
    """
    Print a theoretical memory table for a given model size.

    Args:
        n_params_billions: number of model parameters (billions)
        bit_widths        : bit-widths to compare
        meta_overhead     : metadata overhead fraction

    Returns:
        formatted table string
    """
    n_params = n_params_billions * 1e9
    lines = [
        f"\n  Theoretical Memory — LLaMA-2 {n_params_billions}B",
        f"  {'Bits':<8} {'Weights (GB)':<15} {'+ Meta (GB)':<15} "
        f"{'Compression':<14} {'vs fp16'}",
        "  " + "─" * 62,
    ]
    fp16_gb = n_params * 2 / 1e9

    for bits in bit_widths:
        w_gb    = n_params * bits / 8 / 1e9
        meta_gb = w_gb * meta_overhead
        total   = w_gb + meta_gb
        comp    = fp16_gb / total
        label   = f"fp{bits}" if bits == 16 else f"W{bits}"
        lines.append(
            f"  {label:<8} {w_gb:<15.3f} {total:<15.3f} "
            f"{comp:<14.2f}× {'(baseline)' if bits == 16 else ''}"
        )

    lines.append("  " + "─" * 62)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Per-layer compression histogram
# ---------------------------------------------------------------------------

def layer_compression_stats(result: MemoryResult) -> Dict[str, float]:
    """Summary statistics for per-layer compression ratios."""
    if not result.layer_compressions:
        return {}
    comps = result.layer_compressions
    return {
        "mean":   sum(comps) / len(comps),
        "min":    min(comps),
        "max":    max(comps),
        "median": sorted(comps)[len(comps) // 2],
    }
