"""
quantization/hqq_core.py — Half-Quadratic Quantization (HQQ) core algorithm.

Theory
──────
Standard post-training quantization (PTQ) minimises:

    ||W - dequant(quant(W))||²_F

The challenge: round() is non-differentiable, so naive gradient methods fail.

HQQ reformulates this as a Half-Quadratic (HQ) optimisation problem.
Introduce an auxiliary variable U ≈ W/scale + zero (the "pre-round" tensor):

    min_{scale, zero, U}
        ||W - (round(U) - zero) * scale||²_F
      + ρ/2 * ||U - (W/scale + zero)||²_F

For fixed U, the (scale, zero) sub-problem is a simple linear least squares.
For fixed (scale, zero), we have U* = W/scale + zero, then round.

The HQ update alternates between:
  1. Update U ← proximal_round(W/scale + zero)   [close to rounded values]
  2. Solve linear least-squares for (scale, zero) given U

This converges quickly (typically 10–20 iterations) and requires NO
calibration data — quantisation happens purely from the weight matrix W.

Per-group quantisation
──────────────────────
Weights are reshaped into groups of `group_size` elements.
Each group has its own (scale, zero) pair:
  W_group : (group_size,)  →  scale : scalar,  zero : scalar

Bit widths
──────────
  nbits=1 : binary weights {0,1},  qmax=1
  nbits=2 : 4-level,               qmax=3
  nbits=4 : 16-level,              qmax=15
  nbits=8 : 256-level,             qmax=255

References
──────────
  Badri & Shaji (2023) "HQQ: Half-Quadratic Quantization of Large Machine
  Learning Models"  https://mobiusml.github.io/hqq_blog/
"""

from __future__ import annotations

import math
from typing import Dict, Tuple, Optional

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Low-level quantisation / de-quantisation
# ---------------------------------------------------------------------------

def quantise(
    W: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    qmax: int,
) -> torch.Tensor:
    """
    Affine quantisation: W_q = clamp(round(W / scale + zero), 0, qmax)

    All tensors are broadcastable (W is grouped, scale/zero are per-group).
    """
    W_scaled = W / scale + zero
    W_q = torch.clamp(torch.round(W_scaled), 0, qmax)
    return W_q


def dequantise(
    W_q: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
) -> torch.Tensor:
    """
    Affine de-quantisation: W_hat = (W_q - zero) * scale
    """
    return (W_q - zero) * scale


def quantise_dequantise(
    W: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    qmax: int,
) -> torch.Tensor:
    """Round-trip: quantise then immediately de-quantise (for error measurement)."""
    return dequantise(quantise(W, scale, zero, qmax), scale, zero)


# ---------------------------------------------------------------------------
# Scale / zero initialisation
# ---------------------------------------------------------------------------

def init_scale_zero(
    W_group: torch.Tensor,
    qmax: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Initialise scale and zero using min-max of the weight group.

    For each group g with weights W_g ∈ R^{group_size}:

        scale = (max(W_g) - min(W_g)) / qmax
        zero  = -min(W_g) / scale        [so min maps to 0]

    Returns scale and zero both shaped (..., 1) for broadcasting.
    """
    W_min = W_group.min(dim=-1, keepdim=True).values
    W_max = W_group.max(dim=-1, keepdim=True).values

    scale = (W_max - W_min).clamp(min=1e-9) / qmax
    zero  = -W_min / scale

    return scale, zero


# ---------------------------------------------------------------------------
# Half-Quadratic proximal optimiser
# ---------------------------------------------------------------------------

def hqq_optimise(
    W_group: torch.Tensor,        # (num_groups, group_size)
    scale: torch.Tensor,          # (num_groups, 1)  — initial estimate
    zero: torch.Tensor,           # (num_groups, 1)  — initial estimate
    qmax: int,
    n_iters: int = 20,
    lr: float = 1e-3,
    lam: float = 1.0,
    verbose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Half-Quadratic alternating optimisation for scale and zero.

    Minimises:
        L(scale, zero) = ||W - dequant(quant(W, scale, zero))||²_F

    via the HQ auxiliary variable U:
        U* = argmin_U  ρ/2 * ||U - (W/s + z)||²  +  ||W - (round(U)-z)*s||²

    For each outer iteration:
      1. U  ← W/scale + zero           (float pre-rounded values)
      2. U_r ← round(U).clamp(0, qmax) (integer assignments)
      3. Solve for scale, zero by linear LS:
             [scale, zero] = argmin ||W - (U_r - zero)*scale||²
         This is a 2-parameter linear regression with columns:
             A = [U_r - zero_old,  -scale_old]  ← Jacobian approximation
         We use a gradient step (Adam) for speed.

    Args:
        W_group   : weight matrix reshaped to (num_groups, group_size)
        scale     : initial scale  (num_groups, 1)
        zero      : initial zero   (num_groups, 1)
        qmax      : max integer value for this bit-width
        n_iters   : number of alternating update steps
        lr        : learning rate for scale/zero Adam update
        lam       : proximal coefficient ρ

    Returns:
        Optimised (scale, zero)
    """
    scale = scale.clone().float().requires_grad_(True)
    zero  = zero.clone().float().requires_grad_(True)
    W_f   = W_group.float()

    optimiser = torch.optim.Adam([scale, zero], lr=lr, betas=(0.9, 0.999))

    for i in range(n_iters):
        optimiser.zero_grad()

        # ── Forward: quantise→dequantise ────────────────────────────────
        # Use straight-through estimator for the round() gradient
        W_scaled = W_f / scale + zero                            # U
        W_q_ste  = (torch.clamp(torch.round(W_scaled), 0, qmax)
                    - W_scaled.detach() + W_scaled)              # STE
        W_hat    = (W_q_ste - zero) * scale                      # W_hat

        # ── Reconstruction loss ─────────────────────────────────────────
        loss = F.mse_loss(W_hat, W_f)

        # ── Proximal term: keep scale well-conditioned ──────────────────
        loss = loss + lam * 1e-6 * (scale ** 2).mean()

        loss.backward()
        optimiser.step()

        # Clamp scale away from zero to prevent division instability
        with torch.no_grad():
            scale.clamp_(min=1e-9)
            zero.clamp_(0, qmax)

        if verbose and (i == 0 or (i + 1) % 5 == 0):
            print(f"    HQQ iter {i+1:>3}/{n_iters}  loss={loss.item():.6f}")

    return scale.detach(), zero.detach()


# ---------------------------------------------------------------------------
# Main quantisation function (operates on a full weight matrix)
# ---------------------------------------------------------------------------

def hqq_quantise_weight(
    W: torch.Tensor,
    nbits: int = 1,
    group_size: int = 64,
    axis: int = 1,
    optimize: bool = True,
    opt_iters: int = 20,
    opt_lr: float = 1e-3,
) -> Dict[str, torch.Tensor]:
    """
    Quantise a 2-D weight matrix W using HQQ.

    Steps:
      1. Reshape W into groups along the chosen axis.
      2. Initialise scale/zero with min-max.
      3. Optionally run the HQ proximal optimiser.
      4. Compute final W_q (integer) and store scale/zero.

    Args:
        W          : (out_features, in_features)  weight matrix
        nbits      : bits per weight (1, 2, 4, 8)
        group_size : number of weights per group
        axis       : 0 → group along out_features, 1 → along in_features
        optimize   : run HQ optimiser
        opt_iters  : optimiser iterations

    Returns dict with keys:
        W_q        : (out, in) int8/uint8 quantised weights (packed below)
        scale      : (num_groups, 1)
        zero       : (num_groups, 1)
        shape      : original W shape
        nbits      : bits per weight
        group_size : group size used
        axis       : axis used
        quant_error: mean squared reconstruction error
    """
    assert W.dim() == 2, f"Expected 2-D weight, got shape {W.shape}"

    orig_shape = W.shape
    orig_dtype = W.dtype
    device     = W.device
    qmax       = (1 << nbits) - 1

    # ── Reshape into groups ──────────────────────────────────────────────
    W_work = W.float()
    if axis == 1:
        # Group along in_features (columns)
        # W : (out, in) → (out * in // group_size, group_size)
        if W_work.shape[1] % group_size != 0:
            pad = group_size - W_work.shape[1] % group_size
            W_work = F.pad(W_work, (0, pad))
        W_grouped = W_work.reshape(-1, group_size)
    else:
        # Group along out_features (rows)
        if W_work.shape[0] % group_size != 0:
            pad = group_size - W_work.shape[0] % group_size
            W_work = F.pad(W_work, (0, 0, 0, pad))
        W_grouped = W_work.T.reshape(-1, group_size)

    num_groups = W_grouped.shape[0]

    # ── Init scale / zero ────────────────────────────────────────────────
    scale, zero = init_scale_zero(W_grouped, qmax)      # (G, 1) each

    # ── HQ optimisation ──────────────────────────────────────────────────
    if optimize:
        scale, zero = hqq_optimise(
            W_grouped, scale, zero, qmax,
            n_iters=opt_iters, lr=opt_lr,
        )

    # ── Final quantisation ───────────────────────────────────────────────
    W_q = quantise(W_grouped, scale, zero, qmax)        # (G, group_size)

    # ── Measure reconstruction error ─────────────────────────────────────
    W_hat = dequantise(W_q, scale, zero)
    quant_error = F.mse_loss(W_hat, W_grouped).item()

    # ── Pack W_q back to original shape (as uint8) ───────────────────────
    if axis == 1:
        W_q_2d = W_q.reshape(W_work.shape).to(torch.uint8)[:orig_shape[0], :orig_shape[1]]
    else:
        W_q_2d = W_q.reshape(W_work.T.shape).T.to(torch.uint8)[:orig_shape[0], :orig_shape[1]]

    return {
        "W_q":        W_q_2d,
        "scale":      scale.to(orig_dtype),
        "zero":       zero.to(orig_dtype),
        "shape":      orig_shape,
        "nbits":      nbits,
        "group_size": group_size,
        "axis":       axis,
        "quant_error": quant_error,
    }


# ---------------------------------------------------------------------------
# Bit-packing utilities (for memory-efficient storage)
# ---------------------------------------------------------------------------

def pack_weights(W_q: torch.Tensor, nbits: int) -> torch.Tensor:
    """
    Pack quantised uint8 weights into a compact integer tensor.

    For nbits=1: 8 weights packed into each uint8  → 8× compression
    For nbits=2: 4 weights packed into each uint8  → 4× compression
    For nbits=4: 2 weights packed into each uint8  → 2× compression
    For nbits=8: no packing needed

    Args:
        W_q  : (out, in)  uint8 values in [0, qmax]
        nbits: bits per weight

    Returns:
        packed : uint8 tensor (out, in // (8 // nbits))
    """
    if nbits == 8:
        return W_q

    values_per_byte = 8 // nbits
    orig_shape = W_q.shape

    # Flatten to 1-D, pad to multiple of values_per_byte
    flat = W_q.flatten().to(torch.uint8)
    pad  = (-len(flat)) % values_per_byte
    if pad:
        flat = F.pad(flat, (0, pad))

    # Pack: shift each value by its position within a byte
    flat = flat.reshape(-1, values_per_byte)
    packed = torch.zeros(flat.shape[0], dtype=torch.uint8, device=W_q.device)
    for i in range(values_per_byte):
        packed |= (flat[:, i] << (i * nbits)).to(torch.uint8)

    return packed


def unpack_weights(
    packed: torch.Tensor,
    nbits: int,
    orig_shape: Tuple[int, int],
) -> torch.Tensor:
    """
    Reverse of pack_weights — unpack a compact uint8 tensor to full uint8.
    """
    if nbits == 8:
        return packed.reshape(orig_shape)

    values_per_byte = 8 // nbits
    mask = (1 << nbits) - 1  # bitmask for one value

    unpacked = torch.zeros(
        packed.numel() * values_per_byte,
        dtype=torch.uint8,
        device=packed.device,
    )
    for i in range(values_per_byte):
        unpacked[i::values_per_byte] = (packed >> (i * nbits)).to(torch.uint8) & mask

    total = orig_shape[0] * orig_shape[1]
    return unpacked[:total].reshape(orig_shape)


# ---------------------------------------------------------------------------
# Model size estimation
# ---------------------------------------------------------------------------

def estimate_model_size_gb(
    num_params: int,
    nbits: int,
    meta_overhead_ratio: float = 0.02,
) -> float:
    """
    Estimate quantised model size in GB.

    Weights occupy (num_params * nbits / 8) bytes.
    Scale/zero metadata adds a small overhead (default ~2%).
    """
    weight_bytes = num_params * nbits / 8
    meta_bytes   = weight_bytes * meta_overhead_ratio
    total_gb     = (weight_bytes + meta_bytes) / (1024 ** 3)
    return total_gb


def quantisation_ratio(nbits: int) -> float:
    """Compression ratio relative to fp16 (16-bit) baseline."""
    return 16.0 / nbits
