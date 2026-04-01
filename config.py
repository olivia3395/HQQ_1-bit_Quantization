"""
config.py — Central configuration for HQQ 1-bit quantization project.

Covers quantization settings, model paths, benchmark parameters,
and evaluation options.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple


# ---------------------------------------------------------------------------
# Quantization configuration
# ---------------------------------------------------------------------------

@dataclass
class QuantConfig:
    """
    Controls the HQQ quantization process.

    nbits        : bits per weight (1, 2, 3, 4, 8)
    group_size   : number of weights sharing a scale/zero (64, 128, 256)
    axis         : quantization axis (0 = per-output, 1 = per-input)
    optimize     : run Half-Quadratic proximal optimiser after rounding
    opt_iters    : number of HQQ proximal optimisation iterations
    opt_lr       : learning rate for scale/zero optimisation
    scale_dtype  : dtype for scale/zero parameters (float16 / bfloat16)
    offload_meta : keep scale/zero on CPU to save GPU VRAM
    skip_layers  : regex patterns for layers to skip (kept in fp16)
    """
    nbits: int = 1
    group_size: int = 64
    axis: int = 1
    optimize: bool = True
    opt_iters: int = 20
    opt_lr: float = 1e-3
    scale_dtype: str = "float16"
    offload_meta: bool = False
    skip_layers: List[str] = field(default_factory=lambda: [
        "lm_head",
        "embed_tokens",
        "norm",
    ])

    def __post_init__(self):
        assert self.nbits in (1, 2, 3, 4, 8), f"Unsupported nbits: {self.nbits}"
        assert self.group_size in (32, 64, 128, 256, 512, -1), \
            f"Unsupported group_size: {self.group_size}"
        assert self.axis in (0, 1), f"axis must be 0 or 1"

    @property
    def qmax(self) -> int:
        """Maximum quantised integer value."""
        return (1 << self.nbits) - 1  # 2^b - 1

    def label(self) -> str:
        return f"W{self.nbits}G{self.group_size}"


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """LLaMA 2 model settings."""

    # HuggingFace model identifier or local path
    model_name: str = "meta-llama/Llama-2-7b-hf"

    # Dtype for the base model before quantisation
    torch_dtype: str = "float16"

    # Device map for loading large models
    device_map: Optional[str] = None   # None → single GPU / CPU

    # Maximum sequence length
    max_seq_len: int = 2048

    # Trust remote code (needed for some community models)
    trust_remote_code: bool = False


# ---------------------------------------------------------------------------
# Benchmark configuration
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkConfig:
    """Controls the speed / accuracy / memory benchmark suite."""

    # ── Speed benchmark ──────────────────────────────────────────────────
    # Number of warm-up runs before timing
    warmup_runs: int = 3
    # Number of timed forward passes
    timed_runs: int = 20
    # Batch sizes to sweep
    batch_sizes: List[int] = field(default_factory=lambda: [1, 4, 8])
    # Sequence lengths to sweep
    seq_lengths: List[int] = field(default_factory=lambda: [128, 512, 1024])
    # Generation: number of new tokens for throughput measurement
    gen_tokens: int = 128

    # ── Accuracy benchmark ───────────────────────────────────────────────
    # Dataset for perplexity ("wikitext2" | "c4" | "ptb")
    ppl_dataset: str = "wikitext2"
    # Number of samples for perplexity evaluation
    ppl_samples: int = 128
    # Stride for sliding-window perplexity
    ppl_stride: int = 512

    # ── Memory benchmark ─────────────────────────────────────────────────
    measure_peak_memory: bool = True

    # ── Quant configs to sweep ───────────────────────────────────────────
    # List of (nbits, group_size) pairs
    quant_grid: List[Tuple[int, int]] = field(default_factory=lambda: [
        (1, 64),
        (2, 64),
        (4, 64),
        (8, 64),
    ])


# ---------------------------------------------------------------------------
# Master config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    model: ModelConfig    = field(default_factory=ModelConfig)
    quant: QuantConfig    = field(default_factory=QuantConfig)
    bench: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    output_dir: str       = "outputs/hqq"
    seed: int             = 42
    device: str           = "cuda"   # "cuda" | "cpu"

    def summary(self) -> str:
        lines = ["=" * 62, "  HQQ Quantisation — Configuration", "=" * 62]
        for sec in ("model", "quant", "bench"):
            obj = getattr(self, sec)
            lines.append(f"\n[{sec.upper()}]")
            for k, v in vars(obj).items():
                lines.append(f"  {k:<28} {v}")
        lines.append("=" * 62)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Preset factories
# ---------------------------------------------------------------------------

def w1_config(model_name: str = "meta-llama/Llama-2-7b-hf") -> Config:
    """1-bit HQQ quantisation with group size 64."""
    cfg = Config()
    cfg.model.model_name = model_name
    cfg.quant.nbits = 1
    cfg.quant.group_size = 64
    cfg.quant.optimize = True
    return cfg

def w2_config(model_name: str = "meta-llama/Llama-2-7b-hf") -> Config:
    """2-bit HQQ quantisation."""
    cfg = Config()
    cfg.model.model_name = model_name
    cfg.quant.nbits = 2
    cfg.quant.group_size = 64
    return cfg

def w4_config(model_name: str = "meta-llama/Llama-2-7b-hf") -> Config:
    """4-bit HQQ quantisation (GPTQ baseline comparable)."""
    cfg = Config()
    cfg.model.model_name = model_name
    cfg.quant.nbits = 4
    cfg.quant.group_size = 128
    return cfg

# Tiny model alias for unit testing (no LLaMA access needed)
def tiny_test_config() -> Config:
    cfg = Config()
    cfg.model.model_name = "__synthetic__"
    cfg.quant.nbits = 1
    cfg.quant.group_size = 64
    cfg.quant.optimize = True
    cfg.quant.opt_iters = 5
    cfg.device = "cpu"
    return cfg
