"""
models/llama_utils.py — LLaMA 2 loading utilities and synthetic test model.

Real LLaMA 2 requires a HuggingFace access token and model weights.
This module also provides a SyntheticLLaMA class that mirrors the
LLaMA architecture at tiny scale for unit testing without the real model.

Supported models:
    meta-llama/Llama-2-7b-hf    (recommended, ~13 GB fp16)
    meta-llama/Llama-2-13b-hf   (~26 GB fp16)
    meta-llama/Llama-2-70b-hf   (~140 GB fp16, requires sharding)
    huggyllama/llama-7b         (community weights, no token needed)
    openlm-research/open_llama_3b (small, good for local testing)
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Real LLaMA 2 loading
# ---------------------------------------------------------------------------

def load_llama(
    model_name: str,
    torch_dtype: str = "float16",
    device_map: Optional[str] = "auto",
    max_seq_len: int = 2048,
    trust_remote_code: bool = False,
    hf_token: Optional[str] = None,
) -> Tuple[nn.Module, any]:
    """
    Load a LLaMA 2 model and tokenizer from HuggingFace Hub.

    Args:
        model_name      : HF model identifier
        torch_dtype     : "float16" | "bfloat16" | "float32"
        device_map      : "auto" distributes across available GPUs/CPU
        max_seq_len     : model max sequence length
        trust_remote_code: needed for some community models
        hf_token        : HuggingFace API token (required for meta-llama)

    Returns:
        (model, tokenizer)
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        raise ImportError("Please install transformers: pip install transformers")

    dtype_map = {
        "float16":  torch.float16,
        "bfloat16": torch.bfloat16,
        "float32":  torch.float32,
    }
    dtype = dtype_map.get(torch_dtype, torch.float16)

    token = hf_token or os.environ.get("HF_TOKEN")

    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=token,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading model: {model_name}  dtype={torch_dtype}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device_map,
        token=token,
        trust_remote_code=trust_remote_code,
    )
    model.eval()

    logger.info(
        f"Model loaded: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params"
    )
    return model, tokenizer


# ---------------------------------------------------------------------------
# Synthetic LLaMA-style model (for testing without real weights)
# ---------------------------------------------------------------------------

class SyntheticRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps    = eps

    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x / norm * self.weight


class SyntheticMLP(nn.Module):
    """SwiGLU MLP as used in LLaMA 2."""
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj   = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.down_proj(
            nn.functional.silu(self.gate_proj(x)) * self.up_proj(x)
        )


class SyntheticAttention(nn.Module):
    """Simplified multi-head attention (no RoPE, no KV cache)."""
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.n_heads  = n_heads
        self.head_dim = dim // n_heads
        self.q_proj   = nn.Linear(dim, dim, bias=False)
        self.k_proj   = nn.Linear(dim, dim, bias=False)
        self.v_proj   = nn.Linear(dim, dim, bias=False)
        self.o_proj   = nn.Linear(dim, dim, bias=False)

    def forward(self, x, mask=None):
        B, T, D = x.shape
        H, Dh = self.n_heads, self.head_dim
        q = self.q_proj(x).reshape(B, T, H, Dh).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, H, Dh).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, H, Dh).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) * (Dh ** -0.5)
        if mask is not None:
            scores = scores + mask
        attn   = scores.softmax(-1)
        out    = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.o_proj(out)


class SyntheticDecoderLayer(nn.Module):
    def __init__(self, dim: int, n_heads: int, hidden_dim: int):
        super().__init__()
        self.self_attn        = SyntheticAttention(dim, n_heads)
        self.mlp              = SyntheticMLP(dim, hidden_dim)
        self.input_layernorm  = SyntheticRMSNorm(dim)
        self.post_attn_norm   = SyntheticRMSNorm(dim)

    def forward(self, x, mask=None):
        x = x + self.self_attn(self.input_layernorm(x), mask)
        x = x + self.mlp(self.post_attn_norm(x))
        return x


class SyntheticLLaMA(nn.Module):
    """
    Tiny LLaMA-2 style model for unit testing.

    Default dims: hidden=256, 4 layers, 4 heads — loads in seconds on CPU.
    Exposes the same attribute names as the real LLaMA 2 (model.layers,
    model.embed_tokens, etc.) so quantisation code works unchanged.
    """
    def __init__(
        self,
        vocab_size: int  = 32000,
        hidden_dim: int  = 256,
        n_layers: int    = 4,
        n_heads: int     = 4,
        mlp_ratio: float = 2.667,  # LLaMA uses ~2.667 × hidden_dim
        max_seq_len: int = 512,
    ):
        super().__init__()
        mlp_dim = int(hidden_dim * mlp_ratio)

        # Use a nested 'model' attribute to mirror HuggingFace LLaMA layout
        self.model = _SyntheticLLaMAInner(
            vocab_size, hidden_dim, n_layers, n_heads, mlp_dim
        )
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.config  = _FakeConfig(hidden_dim, n_heads, n_layers, vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        hidden = self.model(input_ids, attention_mask)
        logits = self.lm_head(hidden)
        # Return a simple namespace object (like HF ModelOutput)
        return _FakeOutput(logits=logits)

    def named_linear_layers(self):
        for name, mod in self.named_modules():
            if isinstance(mod, nn.Linear):
                yield name, mod


class _SyntheticLLaMAInner(nn.Module):
    def __init__(self, vocab_size, dim, n_layers, n_heads, mlp_dim):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            SyntheticDecoderLayer(dim, n_heads, mlp_dim)
            for _ in range(n_layers)
        ])
        self.norm = SyntheticRMSNorm(dim)

    def forward(self, input_ids, mask=None):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class _FakeConfig:
    def __init__(self, hidden, heads, layers, vocab):
        self.hidden_size     = hidden
        self.num_attention_heads = heads
        self.num_hidden_layers   = layers
        self.vocab_size      = vocab
        self.model_type      = "llama"


class _FakeOutput:
    def __init__(self, logits):
        self.logits = logits


class _FakeTok:
    """Minimal tokenizer shim for synthetic model tests."""
    pad_token_id = 0
    eos_token_id = 2
    bos_token_id = 1

    def __call__(self, text, return_tensors="pt", **kw):
        # Simple character-level tokenizer
        ids = [ord(c) % 100 + 3 for c in text[:50]]
        t   = torch.tensor([ids])
        return {"input_ids": t, "attention_mask": torch.ones_like(t)}

    def decode(self, ids, **kw):
        return "".join(chr(max(32, (i - 3) % 100 + 32)) for i in ids)

    def batch_decode(self, ids_list, **kw):
        return [self.decode(ids) for ids in ids_list]

    def encode(self, text, **kw):
        return [ord(c) % 100 + 3 for c in text[:50]]


def load_synthetic_model(device: str = "cpu") -> Tuple[SyntheticLLaMA, _FakeTok]:
    """Return a tiny synthetic LLaMA-style model for offline testing."""
    model = SyntheticLLaMA().to(device).eval()
    tok   = _FakeTok()
    return model, tok
