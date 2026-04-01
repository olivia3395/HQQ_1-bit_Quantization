"""
quantization/hqq_model.py — Model-level HQQ quantisation.

Replaces nn.Linear layers in a model with HQQLinear equivalents,
honouring a skip-list (lm_head, embeddings, norms, etc.).

Quantisation strategy
─────────────────────
LLaMA 2 attention & MLP linear layers targeted:

  Attention:
    q_proj, k_proj, v_proj  — query/key/value projections
    o_proj                  — output projection

  MLP (SwiGLU gate):
    gate_proj, up_proj      — gating + expansion
    down_proj               — contraction

  Skipped (kept fp16):
    embed_tokens            — embedding table (lookup, not matmul)
    lm_head                 — final vocab projection (accuracy-critical)
    layernorm / rmsnorm     — tiny, no matmul
"""

from __future__ import annotations

import re
import time
import logging
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn

from .hqq_linear import HQQLinear
from config import QuantConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility: walk module tree and collect named linear layers
# ---------------------------------------------------------------------------

def _find_linear_layers(
    model: nn.Module,
    skip_patterns: List[str],
    prefix: str = "",
) -> List[Tuple[str, nn.Linear, nn.Module, str]]:
    """
    DFS walk returning (full_name, module, parent, attr_name) for every
    nn.Linear whose name does NOT match any skip pattern.
    """
    results = []
    for name, child in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        # Check skip patterns
        skip = any(re.search(pat, full_name) for pat in skip_patterns)
        if isinstance(child, nn.Linear) and not skip:
            results.append((full_name, child, model, name))
        else:
            # Recurse
            results.extend(
                _find_linear_layers(child, skip_patterns, prefix=full_name)
            )
    return results


# ---------------------------------------------------------------------------
# Main quantisation entry point
# ---------------------------------------------------------------------------

def quantise_model(
    model: nn.Module,
    quant_cfg: QuantConfig,
    verbose: bool = True,
) -> Dict[str, any]:
    """
    Quantise all eligible nn.Linear layers in the model with HQQ.

    This modifies the model in-place and returns a summary dict.

    Args:
        model      : the loaded fp16 model
        quant_cfg  : QuantConfig with nbits, group_size, etc.
        verbose    : print per-layer progress

    Returns:
        summary dict with:
            n_quantised    : number of layers quantised
            n_skipped      : number of layers skipped
            total_error    : mean quantisation MSE across all layers
            size_fp16_gb   : model size if kept in fp16 (GB)
            size_quant_gb  : model size after quantisation (GB)
            compression    : compression ratio
            elapsed_sec    : wall-clock time for quantisation
            per_layer      : list of per-layer stats dicts
    """
    model.eval()

    # Collect layers to quantise
    targets = _find_linear_layers(model, quant_cfg.skip_layers)

    if verbose:
        print(f"\n{'─'*62}")
        print(f"  HQQ Quantisation  →  W{quant_cfg.nbits} / G{quant_cfg.group_size}")
        print(f"  Eligible layers   :  {len(targets)}")
        print(f"  Skip patterns     :  {quant_cfg.skip_layers}")
        print(f"{'─'*62}")

    per_layer_stats: List[Dict] = []
    total_error    = 0.0
    fp16_bytes     = 0
    quant_bytes    = 0
    skipped_bytes  = 0

    t_start = time.time()

    for i, (full_name, linear, parent, attr) in enumerate(targets):
        W_rows, W_cols = linear.weight.shape
        fp16_layer = W_rows * W_cols * 2   # 2 bytes per fp16 element

        t0 = time.time()

        # ── Quantise this layer ──────────────────────────────────────────
        hqq_layer = HQQLinear.from_linear(
            linear,
            nbits=quant_cfg.nbits,
            group_size=quant_cfg.group_size,
            axis=quant_cfg.axis,
            optimize=quant_cfg.optimize,
            opt_iters=quant_cfg.opt_iters,
            opt_lr=quant_cfg.opt_lr,
        )

        # ── Replace in parent module ─────────────────────────────────────
        setattr(parent, attr, hqq_layer)

        elapsed_layer = time.time() - t0
        err = getattr(hqq_layer, "quant_error", 0.0)
        total_error  += err
        fp16_bytes   += fp16_layer
        quant_bytes  += hqq_layer.full_size_bytes()

        stat = {
            "name":         full_name,
            "shape":        (W_rows, W_cols),
            "quant_error":  err,
            "compression":  hqq_layer.compression_ratio(),
            "elapsed_sec":  elapsed_layer,
        }
        per_layer_stats.append(stat)

        if verbose:
            pct = (i + 1) / len(targets) * 100
            print(
                f"  [{pct:5.1f}%] {full_name:<45} "
                f"({W_rows}×{W_cols})  "
                f"err={err:.2e}  "
                f"{hqq_layer.compression_ratio():.1f}×  "
                f"{elapsed_layer:.2f}s"
            )

    # Account for skipped layers (kept fp16)
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and any(
            re.search(pat, name) for pat in quant_cfg.skip_layers
        ):
            skipped_bytes += mod.weight.numel() * 2

    elapsed_total = time.time() - t_start
    mean_error    = total_error / max(len(targets), 1)
    total_fp16_gb = (fp16_bytes + skipped_bytes) / (1024 ** 3)
    total_q_gb    = (quant_bytes + skipped_bytes) / (1024 ** 3)
    compression   = total_fp16_gb / max(total_q_gb, 1e-9)

    summary = {
        "n_quantised":   len(targets),
        "n_skipped":     0,          # counted separately above
        "mean_error":    mean_error,
        "size_fp16_gb":  total_fp16_gb,
        "size_quant_gb": total_q_gb,
        "compression":   compression,
        "elapsed_sec":   elapsed_total,
        "per_layer":     per_layer_stats,
    }

    if verbose:
        print(f"\n  {'─'*58}")
        print(f"  Quantisation complete in {elapsed_total:.1f}s")
        print(f"  Mean quant error  : {mean_error:.4e}")
        print(f"  Size (fp16)       : {total_fp16_gb:.2f} GB")
        print(f"  Size (quantised)  : {total_q_gb:.2f} GB")
        print(f"  Compression ratio : {compression:.2f}×")
        print(f"  {'─'*58}\n")

    return summary


# ---------------------------------------------------------------------------
# Restore model to fp16 (undo quantisation — useful for ablations)
# ---------------------------------------------------------------------------

def dequantise_model(model: nn.Module) -> int:
    """
    Replace all HQQLinear layers back with nn.Linear (fp16).
    Returns the number of layers restored.
    """
    count = 0
    for name_parts in _find_hqq_layers(model):
        full_name, hqq_layer, parent, attr = name_parts
        # Rebuild fp16 linear
        lin = nn.Linear(
            hqq_layer.in_features,
            hqq_layer.out_features,
            bias=hqq_layer.bias is not None,
        )
        lin.weight.data = hqq_layer.dequantise().contiguous()
        if hqq_layer.bias is not None:
            lin.bias.data = hqq_layer.bias.data.clone()
        setattr(parent, attr, lin)
        count += 1
    return count


def _find_hqq_layers(model, prefix=""):
    results = []
    for name, child in model.named_children():
        full = f"{prefix}.{name}" if prefix else name
        if isinstance(child, HQQLinear):
            results.append((full, child, model, name))
        else:
            results.extend(_find_hqq_layers(child, prefix=full))
    return results


# ---------------------------------------------------------------------------
# Model statistics
# ---------------------------------------------------------------------------

def model_stats(model: nn.Module) -> Dict[str, any]:
    """
    Return a statistics dict for a (possibly quantised) model.

    Counts parameters, HQQLinear layers, and estimates total memory.
    """
    total_params  = sum(p.numel() for p in model.parameters())
    hqq_layers    = list(_find_hqq_layers(model))
    hqq_count     = len(hqq_layers)
    hqq_bytes     = sum(h.full_size_bytes() for _, h, _, _ in hqq_layers)

    # fp16 params (non-quantised)
    fp16_params   = sum(
        p.numel() for n, p in model.named_parameters()
        if not any(isinstance(m, HQQLinear)
                   for m in [model])   # simplified
    )

    total_bytes_est = sum(
        p.numel() * p.element_size() for p in model.parameters()
    ) + hqq_bytes

    return {
        "total_params":    total_params,
        "hqq_layers":      hqq_count,
        "total_bytes_est": total_bytes_est,
        "size_gb_est":     total_bytes_est / (1024 ** 3),
    }
