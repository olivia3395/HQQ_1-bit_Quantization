"""
quantization/hqq_linear.py — HQQLinear: a quantised drop-in for nn.Linear.

Design
──────
HQQLinear stores weights in a low-bit packed format and performs
matrix-vector products by:

  1. Unpacking the integer weights W_q at forward time
  2. Dequantising:  W_hat = (W_q - zero) * scale
  3. Computing:     y = x @ W_hat.T + bias

This is "weight-only" quantisation — activations remain in fp16/bf16.
No custom CUDA kernel is required; the dequantisation is a lightweight
elementwise op before the standard matmul.

For production inference a fused kernel (e.g. torch._inductor,
marlin, or gemlite) would be used instead, giving 2-4× speedup.
This implementation is fully correct and hardware-agnostic.

Memory layout
─────────────
For nbits=1, group_size=64, W shape (4096, 4096):
  Original fp16:  4096 × 4096 × 2 B = 32 MB
  Packed uint8:   4096 × 4096 / 8 B =  2 MB   (+ ~0.5 MB for scale/zero)
  Total:         ≈ 2.5 MB  →  ~12.8× compression vs fp16
"""

from __future__ import annotations

import math
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hqq_core import (
    hqq_quantise_weight,
    dequantise,
    pack_weights,
    unpack_weights,
)


class HQQLinear(nn.Module):
    """
    Quantised linear layer using HQQ.

    Replaces nn.Linear with a weight-only quantised equivalent.
    The forward pass is numerically identical to:
        x @ dequantise(W_q, scale, zero).T + bias

    Parameters stored:
        W_packed : packed uint8 weight tensor
        scale    : per-group scale (float16)
        zero     : per-group zero-point (float16)
        bias     : original bias (float16), if present
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        nbits: int = 1,
        group_size: int = 64,
        axis: int = 1,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.nbits        = nbits
        self.group_size   = group_size
        self.axis         = axis

        # Quantisation metadata stored as buffers (not parameters)
        self.register_buffer("W_packed", torch.zeros(1, dtype=torch.uint8))
        self.register_buffer("scale", torch.ones(1, dtype=torch.float16))
        self.register_buffer("zero",  torch.zeros(1, dtype=torch.float16))
        self.orig_shape = (out_features, in_features)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

    # ------------------------------------------------------------------
    # Class method: construct from an existing nn.Linear
    # ------------------------------------------------------------------

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        nbits: int = 1,
        group_size: int = 64,
        axis: int = 1,
        optimize: bool = True,
        opt_iters: int = 20,
        opt_lr: float = 1e-3,
    ) -> "HQQLinear":
        """
        Quantise an existing nn.Linear layer and return an HQQLinear.

        Args:
            linear    : the source nn.Linear module
            nbits     : target bit-width
            group_size: quantisation group size
            axis      : quantisation axis
            optimize  : run HQ proximal optimiser
            opt_iters : optimiser iterations

        Returns:
            HQQLinear with quantised weights
        """
        in_f  = linear.in_features
        out_f = linear.out_features
        has_b = linear.bias is not None
        device = linear.weight.device

        hqq_layer = cls(
            in_features=in_f,
            out_features=out_f,
            bias=has_b,
            nbits=nbits,
            group_size=group_size,
            axis=axis,
            device=device,
        )

        # Run HQQ quantisation
        W = linear.weight.data.float()
        result = hqq_quantise_weight(
            W,
            nbits=nbits,
            group_size=group_size,
            axis=axis,
            optimize=optimize,
            opt_iters=opt_iters,
            opt_lr=opt_lr,
        )

        # Pack weights and store
        hqq_layer.W_packed  = pack_weights(result["W_q"], nbits).to(device)
        hqq_layer.scale     = result["scale"].to(linear.weight.dtype).to(device)
        hqq_layer.zero      = result["zero"].to(linear.weight.dtype).to(device)
        hqq_layer.orig_shape = result["shape"]
        hqq_layer.quant_error = result["quant_error"]

        if has_b:
            hqq_layer.bias = nn.Parameter(linear.bias.data.clone())

        return hqq_layer

    # ------------------------------------------------------------------
    # Dequantisation
    # ------------------------------------------------------------------

    def dequantise(self) -> torch.Tensor:
        """
        Unpack and dequantise weights on-the-fly.

        Returns W_hat : (out_features, in_features) in fp16/bf16
        """
        W_q = unpack_weights(self.W_packed, self.nbits, self.orig_shape)

        # Reshape W_q into groups matching scale/zero layout
        qmax = (1 << self.nbits) - 1  # noqa (used implicitly)
        if self.axis == 1:
            W_grouped = W_q.float().reshape(-1, self.group_size)
        else:
            W_grouped = W_q.float().T.reshape(-1, self.group_size)

        W_hat = dequantise(W_grouped, self.scale.float(), self.zero.float())

        if self.axis == 1:
            W_hat = W_hat.reshape(self.orig_shape)
        else:
            W_hat = W_hat.reshape(self.orig_shape[1], self.orig_shape[0]).T

        return W_hat.to(self.scale.dtype)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., in_features) activation tensor
        Returns:
            y: (..., out_features)
        """
        W_hat = self.dequantise()                            # (out, in) fp16
        return F.linear(x, W_hat, self.bias)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def weight_size_bytes(self) -> int:
        """Actual memory used by packed weights (bytes)."""
        return self.W_packed.numel() * self.W_packed.element_size()

    def full_size_bytes(self) -> int:
        """Total memory: packed weights + scale + zero + bias."""
        n = self.weight_size_bytes()
        n += self.scale.numel() * self.scale.element_size()
        n += self.zero.numel()  * self.zero.element_size()
        if self.bias is not None:
            n += self.bias.numel() * self.bias.element_size()
        return n

    def fp16_size_bytes(self) -> int:
        """What this layer would cost in fp16."""
        return self.out_features * self.in_features * 2  # 2 bytes per fp16

    def compression_ratio(self) -> float:
        return self.fp16_size_bytes() / max(self.full_size_bytes(), 1)

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"nbits={self.nbits}, group_size={self.group_size}, "
            f"axis={self.axis}, "
            f"compression={self.compression_ratio():.1f}×"
        )
