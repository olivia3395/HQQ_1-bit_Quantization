"""
quantize_model.py — CLI to quantise a LLaMA 2 model and save it.

Usage
─────
  # Quantise LLaMA-2-7B to 1-bit with group size 64
  python quantize_model.py --model meta-llama/Llama-2-7b-hf --bits 1 --group-size 64

  # Quantise to 4-bit (GPTQ-comparable quality)
  python quantize_model.py --model meta-llama/Llama-2-7b-hf --bits 4 --group-size 128

  # Use a small community model (no HF token needed)
  python quantize_model.py --model openlm-research/open_llama_3b --bits 2

  # Offline test with the synthetic tiny model
  python quantize_model.py --synthetic --bits 1

Output
──────
  outputs/hqq/<model>_W<b>G<g>/
    ├── backbone/        HF-format model weights (HQQLinear serialised)
    ├── quant_config.json
    └── summary.json     quantisation statistics
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(
        description="HQQ quantisation for LLaMA 2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default="meta-llama/Llama-2-7b-hf",
                   help="HuggingFace model name or local path")
    p.add_argument("--bits", type=int, default=1, choices=[1, 2, 3, 4, 8],
                   help="Quantisation bit-width")
    p.add_argument("--group-size", type=int, default=64,
                   help="Weight group size")
    p.add_argument("--axis", type=int, default=1, choices=[0, 1],
                   help="Quantisation axis")
    p.add_argument("--no-optimize", action="store_true",
                   help="Skip HQ proximal optimisation (faster, lower quality)")
    p.add_argument("--opt-iters", type=int, default=20,
                   help="HQQ optimiser iterations")
    p.add_argument("--output-dir", default="outputs/hqq",
                   help="Output directory")
    p.add_argument("--dtype", default="float16",
                   choices=["float16", "bfloat16"],
                   help="Model dtype")
    p.add_argument("--synthetic", action="store_true",
                   help="Use synthetic tiny model (offline test)")
    p.add_argument("--hf-token", default=None,
                   help="HuggingFace API token")
    p.add_argument("--verbose", action="store_true", default=True)
    return p.parse_args()


def main():
    args = parse_args()

    # ── Config ───────────────────────────────────────────────────────────
    from config import QuantConfig
    quant_cfg = QuantConfig(
        nbits=args.bits,
        group_size=args.group_size,
        axis=args.axis,
        optimize=not args.no_optimize,
        opt_iters=args.opt_iters,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── Load model ────────────────────────────────────────────────────────
    if args.synthetic:
        from models.llama_utils import load_synthetic_model
        model, tokenizer = load_synthetic_model(str(device))
        model_tag = "synthetic"
        logger.info("Using synthetic tiny LLaMA model (offline mode)")
    else:
        from models.llama_utils import load_llama
        model, tokenizer = load_llama(
            args.model,
            torch_dtype=args.dtype,
            device_map="auto" if device.type == "cuda" else None,
            hf_token=args.hf_token,
        )
        model_tag = args.model.replace("/", "_")

    # Count parameters before quantisation
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params / 1e9:.3f}B")

    # ── Quantise ─────────────────────────────────────────────────────────
    from quantization.hqq_model import quantise_model
    logger.info(f"\nStarting HQQ quantisation  (W{args.bits}G{args.group_size}) …")

    t0      = time.time()
    summary = quantise_model(model, quant_cfg, verbose=args.verbose)
    elapsed = time.time() - t0

    logger.info(f"\nQuantisation finished in {elapsed:.1f}s")
    logger.info(f"  Mean quant error : {summary['mean_error']:.4e}")
    logger.info(f"  Size (fp16)      : {summary['size_fp16_gb']:.2f} GB")
    logger.info(f"  Size (quantised) : {summary['size_quant_gb']:.2f} GB")
    logger.info(f"  Compression      : {summary['compression']:.2f}×")

    # ── Print theoretical table ───────────────────────────────────────────
    from benchmarks.memory_benchmark import theoretical_memory_table
    print(theoretical_memory_table(n_params / 1e9))

    # ── Save ─────────────────────────────────────────────────────────────
    out_tag = f"{model_tag}_W{args.bits}G{args.group_size}"
    out_dir = Path(args.output_dir) / out_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save quantisation config
    cfg_path = out_dir / "quant_config.json"
    with open(cfg_path, "w") as f:
        json.dump(vars(quant_cfg), f, indent=2, default=str)
    logger.info(f"Quant config saved → {cfg_path}")

    # Save summary (without per_layer list, which may be large)
    summary_path = out_dir / "summary.json"
    summary_save = {k: v for k, v in summary.items() if k != "per_layer"}
    summary_save["elapsed_sec"] = elapsed
    summary_save["n_params_B"]  = n_params / 1e9
    with open(summary_path, "w") as f:
        json.dump(summary_save, f, indent=2)
    logger.info(f"Summary saved      → {summary_path}")

    # Save the quantised model via PyTorch serialisation
    model_path = out_dir / "model.pt"
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model state dict   → {model_path}")
    logger.info(f"\n  ✅  Done!  Output: {out_dir}")


if __name__ == "__main__":
    main()
